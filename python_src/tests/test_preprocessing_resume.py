"""Focused integration coverage for scene-boundary extraction checkpoints."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from data.nuscenes_dataset import NuScenesPredictionDataset
from data.preprocessing import (
    DataPreparationPipeline,
    NuScenesRealDataExtractor,
    _tee_extraction_output,
    is_genuinely_moving,
)


class _FakeNuScenesMap:
    def __init__(self, dataroot, map_name):
        self.map_name = map_name

    def get_records_in_radius(self, x, y, radius, layers):
        return {"lane": ["lane-1"], "lane_connector": [], "walkway": [], "ped_crossing": [], "road_divider": []}

    def discretize_lanes(self, tokens, resolution_meters=1.5):
        return {token: [[-10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] for token in tokens}


class _FakeMapValidator:
    def is_map_expansion_available(self, dataroot):
        return True, dataroot, "fixture map layers available"


class _FakeNuScenes:
    fail_on_validation_once = True

    def __init__(self, version, dataroot, verbose=False):
        self.scene = []
        self.sample = []
        self.sample_annotation = []
        self.instance = []
        self.tables = {"sample": {}, "sample_annotation": {}, "log": {}}
        self._build_scene("scene-train", "train-token", "train-log", "train")
        self._build_scene("scene-val", "val-token", "val-log", "val")

    def _build_scene(self, name, scene_token, log_token, prefix):
        frame_count = 27  # 13 seconds at nuScenes' 2 Hz keyframe rate.
        records = {"moving": [], "stopped": []}
        for track_name in records:
            for i in range(frame_count):
                ann_token = f"{prefix}-{track_name}-{i}"
                sample_token = f"{prefix}-sample-{i}"
                x = i * 0.5 if track_name == "moving" else 10.0
                ann = {
                    "token": ann_token,
                    "sample_token": sample_token,
                    "instance_token": f"{prefix}-{track_name}-instance",
                    "category_name": "vehicle.car" if track_name == "moving" else "vehicle.truck",
                    "translation": [x, 0.0, 0.0],
                    "size": [4.0, 2.0, 1.5],
                    "prev": f"{prefix}-{track_name}-{i-1}" if i else "",
                    "next": f"{prefix}-{track_name}-{i+1}" if i + 1 < frame_count else "",
                }
                records[track_name].append(ann)
                self.tables["sample_annotation"][ann_token] = ann

        for i in range(frame_count):
            sample_token = f"{prefix}-sample-{i}"
            barrier_token = f"{prefix}-barrier-{i}"
            barrier = {
                "token": barrier_token,
                "sample_token": sample_token,
                "instance_token": f"{prefix}-barrier-instance-{i}",
                "category_name": "movable_object.barrier",
                "translation": [i * 0.5 + 3.0, 1.0, 0.0],
                "size": [2.0, 0.5, 1.0],
                "prev": "",
                "next": "",
            }
            self.tables["sample_annotation"][barrier_token] = barrier
            sample = {
                "token": sample_token,
                "timestamp": i * 500_000,
                "anns": [records["moving"][i]["token"], records["stopped"][i]["token"], barrier_token],
                "next": f"{prefix}-sample-{i+1}" if i + 1 < frame_count else "",
            }
            self.tables["sample"][sample_token] = sample
            self.sample.append(sample)

        self.tables["log"][log_token] = {"location": "singapore-onenorth"}
        self.scene.append({
            "name": name,
            "token": scene_token,
            "log_token": log_token,
            "first_sample_token": f"{prefix}-sample-0",
        })
        self.sample_annotation.extend(self.tables["sample_annotation"].values())
        self.instance.extend([f"{prefix}-moving-instance", f"{prefix}-stopped-instance"])

    def get(self, table_name, token):
        if table_name == "log" and token == "val-log" and type(self).fail_on_validation_once:
            type(self).fail_on_validation_once = False
            raise RuntimeError("simulated interruption before validation scene")
        return self.tables[table_name][token]


def _fake_nuscenes_modules():
    package = types.ModuleType("nuscenes")
    package.__path__ = []
    nusc_module = types.ModuleType("nuscenes.nuscenes")
    nusc_module.NuScenes = _FakeNuScenes
    map_package = types.ModuleType("nuscenes.map_expansion")
    map_package.__path__ = []
    map_module = types.ModuleType("nuscenes.map_expansion.map_api")
    map_module.NuScenesMap = _FakeNuScenesMap
    utils_package = types.ModuleType("nuscenes.utils")
    utils_package.__path__ = []
    splits_module = types.ModuleType("nuscenes.utils.splits")
    splits_module.create_splits_scenes = lambda: {"train": ["scene-train"], "val": ["scene-val"]}
    return {
        "nuscenes": package,
        "nuscenes.nuscenes": nusc_module,
        "nuscenes.map_expansion": map_package,
        "nuscenes.map_expansion.map_api": map_module,
        "nuscenes.utils": utils_package,
        "nuscenes.utils.splits": splits_module,
    }


class PreprocessingResumeIntegrationTests(unittest.TestCase):
    def test_stops_after_first_scene_resumes_second_and_matches_model_contract(self):
        _FakeNuScenes.fail_on_validation_once = True
        model_config = ModelConfig()
        with tempfile.TemporaryDirectory(prefix="trajectory_preprocessing_resume_") as temp_dir:
            config = DataConfig(
                dataroot=temp_dir,
                output_dir=temp_dir,
                mode="real",
                resume_extraction=True,
                max_train_scenes=None,
                max_val_scenes=None,
            )
            extractor = NuScenesRealDataExtractor(config=config, model_config=model_config)
            pipeline = DataPreparationPipeline(
                config=config,
                map_validator=_FakeMapValidator(),
                real_extractor=extractor,
                model_config=model_config,
            )
            Path(temp_dir, "v1.0-trainval").mkdir()
            report_log = str(Path(temp_dir) / "reports" / "data_extractor.log")
            # Fresh-start behavior must replace an older partial extraction
            # when no checkpoint exists, which is the user's current case.
            for filename in ("train_sequences.jsonl", "train_sequences.json", "val_sequences.jsonl", "val_sequences.json"):
                Path(temp_dir, filename).write_text('{"stale_partial_run": true}\n', encoding="utf-8")

            with patch.dict(sys.modules, _fake_nuscenes_modules()), patch(
                "data.preprocessing._tee_extraction_output",
                side_effect=lambda _project_log: _tee_extraction_output(report_log),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    pipeline.prepare(
                        dataroot=temp_dir, output_dir=temp_dir, target_count=None, mode="real"
                    )

                checkpoint_path = Path(temp_dir) / "preprocessing_checkpoint.json"
                train_path = Path(temp_dir) / "train_sequences.jsonl"
                val_path = Path(temp_dir) / "val_sequences.jsonl"
                checkpoint_after_stop = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                self.assertEqual(checkpoint_after_stop["splits"]["train"]["next_scene_index"], 1)
                self.assertEqual(checkpoint_after_stop["splits"]["train"]["last_processed_scene"]["name"], "scene-train")
                self.assertEqual(checkpoint_after_stop["splits"]["val"]["next_scene_index"], 0)
                train_before_resume = train_path.read_bytes()
                self.assertGreater(len(train_before_resume.splitlines()), 0)
                self.assertEqual(val_path.read_bytes(), b"")
                # Emulate a process killed mid-write after the last checkpoint.
                with train_path.open("ab") as output_file:
                    output_file.write(b'{"partial":')
                with Path(temp_dir, "train_sequences.json").open("ab") as output_file:
                    output_file.write(b'{"partial":')

                pipeline.prepare(
                    dataroot=temp_dir, output_dir=temp_dir, target_count=None, mode="real"
                )

            train_items = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            val_items = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(train_path.read_bytes(), Path(temp_dir, "train_sequences.json").read_bytes())
            self.assertEqual(val_path.read_bytes(), Path(temp_dir, "val_sequences.json").read_bytes())
            self.assertEqual(train_path.read_bytes()[:len(train_before_resume)], train_before_resume)

            self.assertTrue(train_items)
            self.assertTrue(val_items)
            self.assertEqual({item["scene_name"] for item in train_items}, {"scene-train"})
            self.assertEqual({item["scene_name"] for item in val_items}, {"scene-val"})
            self.assertEqual({item["category"] for item in train_items}, {"vehicle.car"})
            self.assertEqual({item["category"] for item in val_items}, {"vehicle.car"})
            self.assertTrue(any("vehicle.truck" in item.get("neighbor_categories", []) for item in train_items))
            self.assertTrue(any("vehicle.truck" in item.get("neighbor_categories", []) for item in val_items))
            self.assertEqual(
                len({(item["sample_token"], item["instance_token"]) for item in train_items}),
                len(train_items),
            )
            self.assertEqual(
                len({(item["sample_token"], item["instance_token"]) for item in val_items}),
                len(val_items),
            )
            self.assertNotIn("stale_partial_run", train_path.read_text(encoding="utf-8"))
            self.assertNotIn("stale_partial_run", val_path.read_text(encoding="utf-8"))
            self.assertIn("simulated interruption", Path(report_log).read_text(encoding="utf-8"))

            for item in train_items + val_items:
                self.assertEqual(len(item["past_traj"]), model_config.past_len)
                self.assertEqual(len(item["future_traj"]), model_config.future_len)
                self.assertEqual(len(item["timestamps_past"]), model_config.past_len)
                self.assertEqual(len(item["timestamps_future"]), model_config.future_len)
                self.assertTrue(np.allclose(np.diff(item["timestamps_past"]), 1 / model_config.hz))
                self.assertTrue(np.allclose(np.diff(item["timestamps_future"]), 1 / model_config.hz))
                self.assertAlmostEqual(item["timestamps_future"][-1] - item["timestamps_past"][-1], model_config.future_sec)
                self.assertTrue(item["neighbors_past"])
                self.assertTrue(item["static_objects"])
                self.assertEqual(len(item["map_lanes"]), 1)
                self.assertEqual(len(item["map_lanes"][0]), model_config.lane_pts)

            checkpoint_after_resume = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint_after_resume["splits"]["train"]["complete"])
            self.assertTrue(checkpoint_after_resume["splits"]["val"]["complete"])
            self.assertEqual(checkpoint_after_resume["splits"]["train"]["saved_count"], len(train_items))
            self.assertEqual(checkpoint_after_resume["splits"]["val"]["saved_count"], len(val_items))
            self.assertEqual(checkpoint_after_resume["splits"]["train"]["jsonl_bytes"], train_path.stat().st_size)
            self.assertEqual(checkpoint_after_resume["splits"]["val"]["jsonl_bytes"], val_path.stat().st_size)

            train_dataset = NuScenesPredictionDataset(data_dir=temp_dir, split="train", model_config=model_config)
            val_dataset = NuScenesPredictionDataset(data_dir=temp_dir, split="val", model_config=model_config)
            train_sample = train_dataset[0]
            val_sample = val_dataset[0]
            self.assertEqual(tuple(train_sample["past"].shape), (model_config.past_len, 2))
            self.assertEqual(tuple(train_sample["future"].shape), (model_config.future_len, 2))
            self.assertEqual(tuple(train_sample["agents_past"].shape), (model_config.past_len, model_config.max_agents, 2))
            self.assertEqual(tuple(train_sample["map_lanes"].shape), (model_config.max_lanes, model_config.lane_pts, 2))
            self.assertEqual(tuple(train_sample["static_obstacles"].shape), (model_config.max_obstacles, 4))
            self.assertEqual(tuple(val_sample["future"].shape), (model_config.future_len, 2))
            for tensor_key in ("past", "future", "agents_past", "map_lanes", "static_obstacles"):
                self.assertTrue(torch_all_finite(train_sample[tensor_key]))

    def test_movement_filter_and_console_log_tee(self):
        self.assertTrue(is_genuinely_moving([[0.0, 0.0], [1.0, 0.0]], [0.0, 1.0]))
        self.assertFalse(is_genuinely_moving([[2.0, 3.0], [2.0, 3.0]], [0.0, 1.0]))
        with tempfile.TemporaryDirectory(prefix="trajectory_extractor_log_") as temp_dir:
            log_path = str(Path(temp_dir) / "data_extractor.log")
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                with _tee_extraction_output(log_path):
                    print("checkpoint log probe")
            self.assertIn("checkpoint log probe", terminal.getvalue())
            self.assertIn("checkpoint log probe", Path(log_path).read_text(encoding="utf-8"))


def torch_all_finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all().item())


if __name__ == "__main__":
    unittest.main()

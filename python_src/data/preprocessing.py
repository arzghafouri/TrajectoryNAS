"""
TrajectoryLNAS Production Data Preparation Pipeline (SOLID Architecture)
======================================================================
Processes raw nuScenes v1.0-trainval benchmark files into trajectory sequences sized from ModelConfig
(configured history and prediction steps, surrounding agents, static objects, and HD map polylines).

Features & Quality Assurance:
  - Exact Target Agent Filtering: Moving Vehicles & Pedestrians.
  - Microsecond-Accurate Timestamp Interpolation: Uses actual sample timestamps (t_raw) from 2Hz keyframes.
  - Physical Static Obstacles: Extracts barriers, traffic cones, and static objects within 50m.
  - Spatial Distance-Sorted Interaction: Nearest dynamic neighbors sorted by Euclidean distance with categories & masks.
  - Multi-Layer HD Map Polylines: Preserves layer_type (lane, connector, walkway, crossing, divider) and lengths.
  - Strict Scene-Level Split: Official nuScenes scene partitioning guaranteeing ZERO cross-split leakage.
  - Explicit Diagnostics: No silent mock fallback in Real mode.

Author: Alireza Ghafouri (Student ID: 4031631209)
Advisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class _TeeStream:
    """Mirror console output to a log while preserving normal terminal output."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


@contextlib.contextmanager
def _tee_extraction_output(log_path: str):
    """Append extraction stdout/stderr to a project report log and the console."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            yield
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

# Single Source of Truth configuration import
try:
    from configs.data_config import DataConfig
    from configs.model_config import ModelConfig
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from configs.data_config import DataConfig
        from configs.model_config import ModelConfig
    except ImportError:
        from configs.search_config import DataConfig
        from configs.model_config import ModelConfig


# ==============================================================================
# Category Constants & Exact Matching Utility
# ==============================================================================

TARGET_VEHICLE_CATEGORIES: set[str] = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.rigid",
    "vehicle.bus.bendy",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.trailer",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.construction"
}

TARGET_PEDESTRIAN_CATEGORIES: set[str] = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair"
}

ALL_TARGET_CATEGORIES: set[str] = TARGET_VEHICLE_CATEGORIES | TARGET_PEDESTRIAN_CATEGORIES

STATIC_OBJECT_CATEGORIES: set[str] = {
    "movable_object.barrier",
    "movable_object.trafficcone",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "static_object.bicycle_rack"
}

# A target must move at least this fast on average across its observed history
# to be retained in the dataset (both Train and Val splits). The default matches
# common nuScenes motion-state thresholds and suppresses annotation jitter from stopped actors.
MIN_TARGET_SPEED_MPS = 0.5
MIN_VAL_TARGET_SPEED_MPS = MIN_TARGET_SPEED_MPS


def is_genuinely_moving(
    trajectory: list[list[float]],
    timestamps: list[float],
    min_speed_mps: float = MIN_TARGET_SPEED_MPS,
) -> bool:
    """Return whether an observed target's net displacement implies real motion."""
    if len(trajectory) < 2 or len(timestamps) < 2:
        return False

    elapsed = float(timestamps[-1]) - float(timestamps[0])
    if elapsed <= 0:
        return False

    start = trajectory[0]
    end = trajectory[-1]
    displacement = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    return displacement / elapsed >= min_speed_mps


def matches_category(cat_name: str, requested_filter: str | list[str] | None) -> bool:
    """
    Checks if an annotation category matches the requested moving target filter cleanly and safely.
    Excludes static obstacles from being primary Target Agents.
    """
    cat_clean = (cat_name or "").strip().lower()
    if cat_clean in STATIC_OBJECT_CATEGORIES or "animal" in cat_clean:
        return False

    if requested_filter is None:
        return cat_clean in ALL_TARGET_CATEGORIES

    if isinstance(requested_filter, str):
        filters = [f.strip().lower() for f in requested_filter.split(",") if f.strip()]
    else:
        filters = [f.strip().lower() for f in requested_filter if f and f.strip()]

    if not filters or "all" in filters:
        return cat_clean in ALL_TARGET_CATEGORIES

    for f in filters:
        if f in ("vehicle", "vehicles") and cat_clean in TARGET_VEHICLE_CATEGORIES or f in ("pedestrian", "pedestrians", "human") and cat_clean in TARGET_PEDESTRIAN_CATEGORIES or cat_clean == f or cat_clean.startswith(f + "."):
            return True

    return False


# ==============================================================================
# Timestamp & Geometry Utilities
# ==============================================================================

def interpolate_traj_by_timestamps(
    times: list[float],
    points: list[list[float]],
    target_times: list[float]
) -> list[list[float]]:
    """
    Interpolates 2D trajectory points using true physical timestamps (in seconds).
    Strictly avoids array-index spacing assumptions.
    """
    if len(points) == 0:
        return [[0.0, 0.0] for _ in range(len(target_times))]
    if len(points) == 1:
        return np.repeat(np.array(points, dtype=np.float32), len(target_times), axis=0).tolist()

    t_arr = np.array(times, dtype=np.float64)
    p_arr = np.array(points, dtype=np.float32)

    # Ensure strictly increasing timestamps
    sort_idx = np.argsort(t_arr)
    t_arr = t_arr[sort_idx]
    p_arr = p_arr[sort_idx]

    # Remove identical timestamps if any
    unique_mask = np.insert(np.diff(t_arr) > 1e-5, 0, True)
    t_arr = t_arr[unique_mask]
    p_arr = p_arr[unique_mask]

    if len(t_arr) < 2:
        return np.repeat(p_arr[:1], len(target_times), axis=0).tolist()

    tgt_arr = np.array(target_times, dtype=np.float64)
    interp_x = np.interp(tgt_arr, t_arr, p_arr[:, 0])
    interp_y = np.interp(tgt_arr, t_arr, p_arr[:, 1])
    return np.column_stack([interp_x, interp_y]).tolist()


def resample_polyline(points: list[list[float]], target_pts: int = 10) -> list[list[float]]:
    """Resamples a multi-point polyline to exactly target_pts points preserving path geometry."""
    if len(points) == 0:
        return [[0.0, 0.0] for _ in range(target_pts)]
    arr = np.array(points, dtype=np.float32)
    if len(arr) == 1:
        return np.repeat(arr, target_pts, axis=0).tolist()

    dists = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dist[-1]

    if total_dist <= 1e-6:
        return np.repeat(arr[:1], target_pts, axis=0).tolist()

    target_dists = np.linspace(0.0, total_dist, num=target_pts)
    resampled = np.zeros((target_pts, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_dists, cum_dist, arr[:, 0])
    resampled[:, 1] = np.interp(target_dists, cum_dist, arr[:, 1])
    return resampled.tolist()


def extract_map_polylines_with_metadata(
    nusc_map: Any,
    x: float,
    y: float,
    radius: float = 50.0,
    max_lanes: int | None = None,
    pts_per_lane: int | None = None
) -> tuple[list[list[list[float]]], list[str], list[float], list[float]]:
    """
    Extracts vectorized HD Map polylines across lane, lane_connector, walkway, ped_crossing,
    and road_divider layers. Preserves layer types, lengths, and distances.
    """
    from configs.model_config import ModelConfig
    _cfg = ModelConfig()
    max_lanes = max_lanes if max_lanes is not None else _cfg.max_lanes
    pts_per_lane = pts_per_lane if pts_per_lane is not None else _cfg.lane_pts
    candidates = []
    layers = ["lane", "lane_connector", "walkway", "ped_crossing", "road_divider"]
    try:
        records = nusc_map.get_records_in_radius(x, y, radius, layers)

        # 1. Lanes and Connectors
        lane_tokens = records.get("lane", []) + records.get("lane_connector", [])
        if lane_tokens:
            try:
                discretized = nusc_map.discretize_lanes(lane_tokens, resolution_meters=1.5)
                for l_tok, pts_3d in discretized.items():
                    if len(pts_3d) >= 2:
                        pts_2d = [[p[0], p[1]] for p in pts_3d]
                        l_type = "lane_connector" if l_tok in records.get("lane_connector", []) else "lane"
                        candidates.append((l_type, pts_2d))
            except Exception:
                for l_tok in lane_tokens:
                    try:
                        is_conn = l_tok in records.get("lane_connector", [])
                        l_type = "lane_connector" if is_conn else "lane"
                        record = nusc_map.get(l_type, l_tok)
                        polygon_tok = record.get("polygon_token")
                        if polygon_tok:
                            polygon = nusc_map.get("polygon", polygon_tok)
                            nodes = [nusc_map.get("node", n_tok) for n_tok in polygon["node_tokens"]]
                            pts_2d = [[n["x"], n["y"]] for n in nodes if "x" in n and "y" in n]
                            if len(pts_2d) >= 2:
                                candidates.append((l_type, pts_2d))
                    except Exception:
                        pass

        # 2. Walkways and Pedestrian Crossings
        ped_tokens = records.get("walkway", []) + records.get("ped_crossing", [])
        for p_tok in ped_tokens:
            try:
                layer_name = "walkway" if p_tok in records.get("walkway", []) else "ped_crossing"
                rec = nusc_map.get(layer_name, p_tok)
                polygon_tok = rec.get("polygon_token")
                if polygon_tok:
                    polygon = nusc_map.get("polygon", polygon_tok)
                    nodes = [nusc_map.get("node", n_tok) for n_tok in polygon["node_tokens"]]
                    pts_2d = [[n["x"], n["y"]] for n in nodes if "x" in n and "y" in n]
                    if len(pts_2d) >= 2:
                        candidates.append((layer_name, pts_2d))
            except Exception:
                pass

        # 3. Road Dividers
        divider_tokens = records.get("road_divider", [])
        for d_tok in divider_tokens:
            try:
                rec = nusc_map.get("road_divider", d_tok)
                n1 = nusc_map.get("node", rec["node1_token"])
                n2 = nusc_map.get("node", rec["node2_token"])
                candidates.append(("road_divider", [[n1["x"], n1["y"]], [n2["x"], n2["y"]]]))
            except Exception:
                pass

    except Exception:
        pass

    # Sort candidates by minimum distance to agent position (x, y)
    scored = []
    agent_pos = np.array([x, y], dtype=np.float32)
    for l_type, pts in candidates:
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.float32)
        min_dist = float(np.min(np.sqrt(np.sum((arr - agent_pos) ** 2, axis=1))))
        length = float(np.sum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))))
        scored.append((min_dist, length, l_type, pts))

    scored.sort(key=lambda item: item[0])

    result_polylines = []
    result_types = []
    result_lengths = []
    result_dists = []

    for min_dist, length, l_type, pts in scored[:max_lanes]:
        resampled = resample_polyline(pts, target_pts=pts_per_lane)
        result_polylines.append(resampled)
        result_types.append(l_type)
        result_lengths.append(round(length, 2))
        result_dists.append(round(min_dist, 2))

    return result_polylines, result_types, result_lengths, result_dists


# ==============================================================================
# Interfaces & Abstractions (ISP & OCP)
# ==============================================================================

class IMapExpansionValidator(ABC):
    @abstractmethod
    def is_map_expansion_available(self, dataroot: str) -> tuple[bool, str | None, str]:
        pass


class BaseTrajectoryExtractor(ABC):
    @abstractmethod
    def extract_sequences(
        self,
        dataroot: str,
        target_count: int | None = 12800,
        seed: int = 42,
        category: str | list[str] | None = None,
        max_train_scenes: int | None = None,
        max_val_scenes: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pass


class IDatasetWriter(ABC):
    @abstractmethod
    def write(self, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], output_dir: str, seed: int = 42) -> tuple[str, str]:
        pass


# ==============================================================================
# Concrete Implementations
# ==============================================================================

class NuScenesMapExpansionValidator(IMapExpansionValidator):
    EXPECTED_CITIES = [
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage"
    ]

    def is_map_expansion_available(self, dataroot: str) -> tuple[bool, str | None, str]:
        if not dataroot or not os.path.exists(dataroot):
            return False, None, f"Dataroot not found: {dataroot}"

        norm_root = os.path.normpath(dataroot)
        parent_dir = os.path.dirname(norm_root)


        candidates = [
            os.path.join(norm_root, "maps", "expansion"),
            os.path.join(norm_root, "maps"),
            os.path.join(norm_root, "map_expansion"),
            os.path.join(parent_dir, "maps", "expansion"),
            os.path.join(parent_dir, "maps"),
            os.path.join(parent_dir, "map_expansion"),
        ]


        for cand in candidates:
            if os.path.exists(cand) and os.path.isdir(cand):
                found_cities = [
                    city for city in self.EXPECTED_CITIES
                    if os.path.exists(os.path.join(cand, f"{city}.json"))
                ]
                if len(found_cities) >= 2:
                    return True, cand, f"Found {len(found_cities)} HD map expansion files in: {cand}"

        return False, None, "No valid maps/expansion JSON layers found in dataroot."


class NuScenesRealDataExtractor(BaseTrajectoryExtractor):
    """
    Extracts 100% real agent trajectories using official nuscenes-devkit.
    Includes microsecond-accurate timestamp interpolation, physical static obstacles,
    distance-sorted dynamic neighbors with categories, and multi-layer HD map polylines.
    """
    def __init__(
        self,
        config: DataConfig | None = None,
        model_config: ModelConfig | None = None,
        max_train_scenes: int | None = None,
        max_val_scenes: int | None = None,
        category: str | list[str] | None = None
    ):
        self.config = config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.max_train_scenes = max_train_scenes if max_train_scenes is not None else self.config.max_train_scenes
        self.max_val_scenes = max_val_scenes if max_val_scenes is not None else self.config.max_val_scenes
        self.category = category if category is not None else self.config.category

    def extract_sequences(
        self,
        dataroot: str,
        target_count: int | None = 12800,
        seed: int = 42,
        category: str | list[str] | None = None,
        max_train_scenes: int | None = None,
        max_val_scenes: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as e:
            raise ImportError(f"nuscenes-devkit is required for real extraction: {e}")

        active_category = category if category is not None else self.category
        active_train_scenes_limit = max_train_scenes if max_train_scenes is not None else self.max_train_scenes
        active_val_scenes_limit = max_val_scenes if max_val_scenes is not None else self.max_val_scenes
        model_cfg = self.model_config
        hz = float(model_cfg.hz)
        past_steps = int(model_cfg.past_len)
        future_steps = int(model_cfg.future_len)
        if hz <= 0 or past_steps <= 0 or future_steps <= 0:
            raise ValueError(
                "ModelConfig must have positive hz, past_len, and future_len "
                f"(got hz={hz}, past_len={past_steps}, future_len={future_steps})."
            )

        count_desc = f"{target_count:,} sequences" if target_count is not None else "ALL available sequences"
        cat_desc = str(active_category) if active_category is not None else "All Target Categories (Vehicles & Pedestrians)"

        print("=" * 80)
        print(" [Real Data Extractor] Extracting sequences via official nuScenes SDK...")
        print(f" Dataroot:        {dataroot}")
        print(f" Target Count:    {count_desc}")
        print(f" Category Filter: {cat_desc}")
        print(
            f" Model Sampling:  {hz:g} Hz | past={past_steps} | future={future_steps} | "
            f"agents={model_cfg.max_agents} | lanes={model_cfg.max_lanes} x {model_cfg.lane_pts} points"
        )
        print("=" * 80, flush=True)

        nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)

        print(f"[NuScenes SDK] Metadata Inspection for '{dataroot}':")
        print(f"  -> Total Scenes: {len(nusc.scene)}")
        print(f"  -> Total Keyframe Samples: {len(nusc.sample)}")
        print(f"  -> Total Annotations: {len(nusc.sample_annotation)}")
        print(f"  -> Total Instances:   {len(nusc.instance)}")
        print("=" * 80, flush=True)

        # Load HD Map Layers
        map_cities = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
        nusc_maps = {}
        for city in map_cities:
            try:
                nusc_maps[city] = NuScenesMap(dataroot=dataroot, map_name=city)
            except Exception:
                pass
        print(f"[NuScenes Maps] Loaded {len(nusc_maps)} vector HD map layers: {list(nusc_maps.keys())}\n", flush=True)

        # Strict scene split partition
        split_scenes = create_splits_scenes()
        train_scene_names = set(split_scenes.get("train", []))
        val_scene_names = set(split_scenes.get("val", []))

        all_train_scenes = [s for s in nusc.scene if s["name"] in train_scene_names]
        all_val_scenes = [s for s in nusc.scene if s["name"] in val_scene_names]

        if active_train_scenes_limit:
            all_train_scenes = all_train_scenes[:active_train_scenes_limit]
        if active_val_scenes_limit:
            all_val_scenes = all_val_scenes[:active_val_scenes_limit]

        train_names = set(s["name"] for s in all_train_scenes)
        val_names = set(s["name"] for s in all_val_scenes)
        intersect = train_names.intersection(val_names)
        if intersect:
            raise ValueError(f"CRITICAL: Scene leakage detected! Shared scenes: {intersect}")

        print(f"[Scene Split] Strict Scene Separation Verified: {len(all_train_scenes)} Train Scenes | {len(all_val_scenes)} Val Scenes (Shared: 0)", flush=True)

        # Spatial cache for HD map queries (bounded to 2,000 items to prevent RAM accumulation)
        map_query_cache: dict[tuple[str, float, float], tuple[list[list[list[float]]], list[str], list[float], list[float]]] = {}

        def get_cached_map_polylines(map_obj: Any, loc: str, qx: float, qy: float):
            if not map_obj:
                return [], [], [], []
            grid_key = (loc, round(qx, 1), round(qy, 1))
            if grid_key in map_query_cache:
                return map_query_cache[grid_key]
            res = extract_map_polylines_with_metadata(
                map_obj, qx, qy, radius=50.0,
                max_lanes=model_cfg.max_lanes,
                pts_per_lane=model_cfg.lane_pts,
            )
            if len(map_query_cache) > 2000:
                map_query_cache.clear()
            map_query_cache[grid_key] = res
            return res

        def get_agent_trajectory_with_timestamps(start_ann: dict[str, Any], t_curr: float) -> tuple[list[list[float]] | None, list[list[float]] | None, list[float], list[float]]:
            """Resamples the observed history and prediction horizon at configured Hz."""
            past_pts = []
            past_times = []
            curr_ann_tok = start_ann.get("prev")
            while curr_ann_tok:
                p_ann = nusc.get("sample_annotation", curr_ann_tok)
                p_sample = nusc.get("sample", p_ann["sample_token"])
                p_time = p_sample["timestamp"] / 1e6
                if (t_curr - p_time) > (model_cfg.past_sec + 0.05):
                    break
                past_pts.append(p_ann["translation"][:2])
                past_times.append(p_time)
                curr_ann_tok = p_ann.get("prev")

            fut_pts = []
            fut_times = []
            curr_ann_tok = start_ann.get("next")
            while curr_ann_tok:
                f_ann = nusc.get("sample_annotation", curr_ann_tok)
                f_sample = nusc.get("sample", f_ann["sample_token"])
                f_time = f_sample["timestamp"] / 1e6
                if (f_time - t_curr) > (model_cfg.future_sec + 0.05):
                    break
                fut_pts.append(f_ann["translation"][:2])
                fut_times.append(f_time)
                curr_ann_tok = f_ann.get("next")

            if len(past_pts) < 2 or len(fut_pts) < 3:
                return None, None, [], []

            all_past_times = list(reversed(past_times)) + [t_curr]
            all_past_pts = list(reversed(past_pts)) + [start_ann["translation"][:2]]

            all_fut_times = [t_curr] + fut_times
            all_fut_pts = [start_ann["translation"][:2]] + fut_pts

            # The model consumes a fixed number of uniformly spaced samples.
            # Include the current observation, then predict one full time step
            # ahead so the configured Hz is preserved in both sequences.
            tgt_past_times = (t_curr - np.arange(past_steps - 1, -1, -1) / hz).tolist()
            tgt_fut_times = (t_curr + np.arange(1, future_steps + 1) / hz).tolist()

            # np.interp clamps out-of-range queries to an endpoint. Reject a
            # sample when the raw annotations do not cover the configured
            # history or full prediction horizon, instead of repeating an
            # endpoint and presenting it as observed ground truth.
            if all_past_times[0] > tgt_past_times[0] + 1e-6:
                return None, None, [], []
            if all_fut_times[-1] < tgt_fut_times[-1] - 1e-6:
                return None, None, [], []

            sampled_past = interpolate_traj_by_timestamps(all_past_times, all_past_pts, tgt_past_times)
            sampled_future = interpolate_traj_by_timestamps(all_fut_times, all_fut_pts, tgt_fut_times)

            return sampled_past, sampled_future, tgt_past_times, tgt_fut_times


        output_dir = os.path.abspath(getattr(self.config, "output_dir", "data/nuscenes/v1.0-trainval"))
        os.makedirs(output_dir, exist_ok=True)
        train_jsonl = os.path.join(output_dir, "train_sequences.jsonl")
        train_json = os.path.join(output_dir, "train_sequences.json")
        val_jsonl = os.path.join(output_dir, "val_sequences.jsonl")
        val_json = os.path.join(output_dir, "val_sequences.json")
        checkpoint_path = os.path.join(output_dir, "preprocessing_checkpoint.json")
        resume_enabled = bool(getattr(self.config, "resume_extraction", True))
        val_target = (max(800, target_count // 4) if target_count is not None else None)

        fingerprint_data = {
            "schema_version": 1,
            "dataroot": os.path.normcase(os.path.abspath(dataroot)),
            "output_dir": os.path.normcase(output_dir),
            "train_scenes": [(s["name"], s["token"]) for s in all_train_scenes],
            "val_scenes": [(s["name"], s["token"]) for s in all_val_scenes],
            "train_target": target_count,
            "val_target": val_target,
            "category": sorted(active_category) if isinstance(active_category, (list, tuple, set)) else active_category,
            "model_config": {
                "hz": hz, "past_sec": model_cfg.past_sec, "future_sec": model_cfg.future_sec,
                "past_len": past_steps, "future_len": future_steps, "max_agents": model_cfg.max_agents,
                "max_obstacles": model_cfg.max_obstacles, "max_lanes": model_cfg.max_lanes,
                "lane_pts": model_cfg.lane_pts,
            },
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")).hexdigest()
        output_pairs = {
            "train": (train_jsonl, train_json),
            "val": (val_jsonl, val_json),
        }
        if resume_enabled and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
                checkpoint = json.load(checkpoint_file)
            if checkpoint.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Checkpoint configuration does not match this extraction: '{checkpoint_path}'. "
                    "Set DataConfig.resume_extraction=False to intentionally start a fresh extraction."
                )
            print(f"[Checkpoint] Resuming from '{checkpoint_path}'.", flush=True)
        else:
            if resume_enabled:
                print("[Checkpoint] No checkpoint found; starting a fresh extraction and replacing prior sequence outputs.", flush=True)
            checkpoint = {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "fingerprint_data": fingerprint_data,
                "splits": {
                    split: {
                        "next_scene_index": 0,
                        "last_processed_scene": None,
                        "saved_count": 0,
                        "jsonl_bytes": 0,
                        "json_bytes": 0,
                        "complete": False,
                    }
                    for split in output_pairs
                },
            }

        def save_checkpoint():
            fd, temporary_path = tempfile.mkstemp(prefix="preprocessing_checkpoint_", suffix=".tmp", dir=output_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as checkpoint_file:
                    json.dump(checkpoint, checkpoint_file, indent=2, ensure_ascii=False)
                    checkpoint_file.flush()
                    os.fsync(checkpoint_file.fileno())
                os.replace(temporary_path, checkpoint_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

        # Roll back any records written after the last fully committed scene.
        # This also discards a partially written scene after an interrupted run.
        for split, (jsonl_path, json_path) in output_pairs.items():
            split_state = checkpoint["splits"][split]
            for path, committed_size in ((jsonl_path, split_state["jsonl_bytes"]), (json_path, split_state["json_bytes"])):
                if not os.path.exists(path):
                    if committed_size:
                        raise RuntimeError(f"Checkpoint expects {committed_size} committed bytes, but output is missing: '{path}'.")
                    open(path, "wb").close()
                actual_size = os.path.getsize(path)
                if actual_size < committed_size:
                    raise RuntimeError(f"Output '{path}' is shorter than its checkpoint; refusing unsafe resume.")
                with open(path, "r+b") as output_file:
                    output_file.truncate(committed_size)

        save_checkpoint()

        def extract_split_stream(split_name: str, scenes: list, desired_count: int | None, split_key: str) -> int:
            seen_keys: set[tuple[str, str]] = set()
            jsonl_path, json_path = output_pairs[split_key]
            split_state = checkpoint["splits"][split_key]
            target_desc = f"{desired_count:,}" if desired_count is not None else "ALL available"
            print(f"[{split_name} Split] Streaming extraction across {len(scenes)} scenes to '{jsonl_path}' (Target: {target_desc})...", flush=True)

            saved_count = int(split_state["saved_count"])
            start_scene_idx = int(split_state["next_scene_index"])

            with open(jsonl_path, "a", encoding="utf-8") as f_l, open(json_path, "a", encoding="utf-8") as f_j:
                for scene_idx in range(start_scene_idx, len(scenes)):
                    scene = scenes[scene_idx]
                    if desired_count is not None and saved_count >= desired_count:
                        break

                    sample_token = scene["first_sample_token"]
                    log = nusc.get("log", scene["log_token"])
                    location = log["location"]
                    nusc_map = nusc_maps.get(location)
                    scene_seq_count = 0
                    scene_name = scene.get("name", f"scene_{scene_idx}")
                    scene_token = scene["token"]

                    while sample_token:
                        if desired_count is not None and saved_count >= desired_count:
                            break

                        sample = nusc.get("sample", sample_token)
                        t_curr = sample["timestamp"] / 1e6

                        for ann_token in sample["anns"]:
                            if desired_count is not None and saved_count >= desired_count:
                                break

                            ann = nusc.get("sample_annotation", ann_token)
                            cat_name = ann.get("category_name", "")

                            if not matches_category(cat_name, active_category):
                                continue

                            instance_token = ann["instance_token"]
                            unique_key = (sample_token, instance_token)
                            if unique_key in seen_keys:
                                continue

                            sampled_past, sampled_future, t_past, t_future = get_agent_trajectory_with_timestamps(ann, t_curr)
                            if sampled_past is None or sampled_future is None:
                                continue

                            # Uniform motion filtering for target selection across BOTH Train and Val splits.
                            # Stationary/low-motion targets (< 0.5 m/s) are excluded from independent target
                            # prediction supervision to avoid distribution mismatch and trivial zero-velocity shortcuts.
                            # Parked and low-motion agents remain fully available as neighbor/obstacle context below.
                            if not is_genuinely_moving(sampled_past, t_past):
                                continue

                            target_x, target_y = ann["translation"][0], ann["translation"][1]

                            neighbor_candidates = []
                            static_candidates = []

                            for other_ann_tok in sample["anns"]:
                                if other_ann_tok == ann_token:
                                    continue
                                other_ann = nusc.get("sample_annotation", other_ann_tok)
                                o_cat = other_ann.get("category_name", "")
                                o_x, o_y = other_ann["translation"][0], other_ann["translation"][1]
                                dist = math.hypot(o_x - target_x, o_y - target_y)

                                if dist > 50.0:
                                    continue

                                if o_cat in STATIC_OBJECT_CATEGORIES or o_cat.startswith("movable_object.") or o_cat.startswith("static_object."):
                                    static_candidates.append({
                                        "category": o_cat,
                                        "translation": [round(o_x, 3), round(o_y, 3)],
                                        "size": [round(v, 3) for v in other_ann.get("size", [])],
                                        "distance": round(dist, 2)
                                    })
                                else:
                                    n_past_pts = []
                                    n_past_times = []
                                    n_prev = other_ann.get("prev")
                                    while n_prev:
                                        np_ann = nusc.get("sample_annotation", n_prev)
                                        np_time = nusc.get("sample", np_ann["sample_token"])["timestamp"] / 1e6
                                        if (t_curr - np_time) > (model_cfg.past_sec + 0.05):
                                            break
                                        n_past_pts.append(np_ann["translation"][:2])
                                        n_past_times.append(np_time)
                                        n_prev = np_ann.get("prev")

                                    if len(n_past_pts) >= 1:
                                        all_n_times = list(reversed(n_past_times)) + [t_curr]
                                        all_n_pts = list(reversed(n_past_pts)) + [other_ann["translation"][:2]]
                                        sampled_neighbor_past = interpolate_traj_by_timestamps(all_n_times, all_n_pts, t_past)
                                        neighbor_candidates.append({
                                            "distance": dist,
                                            "trajectory": sampled_neighbor_past,
                                            "category": o_cat,
                                            "instance_token": other_ann["instance_token"]
                                        })
                                    elif other_ann.get("translation"):
                                        # Preserve parked or stationary agents with limited history:
                                        # replicate static coordinates across observation window for spatial occupancy.
                                        static_pt = other_ann["translation"][:2]
                                        sampled_neighbor_past = [static_pt[:] for _ in range(len(t_past))]
                                        neighbor_candidates.append({
                                            "distance": dist,
                                            "trajectory": sampled_neighbor_past,
                                            "category": o_cat,
                                            "instance_token": other_ann["instance_token"]
                                        })

                            neighbor_candidates.sort(key=lambda x: x["distance"])
                            selected_neighbors = neighbor_candidates[:max(0, model_cfg.max_agents - 1)]

                            neighbors_past = [n["trajectory"] for n in selected_neighbors]
                            neighbor_categories = [n["category"] for n in selected_neighbors]
                            neighbor_instances = [n["instance_token"] for n in selected_neighbors]
                            neighbor_distances = [round(n["distance"], 2) for n in selected_neighbors]

                            static_candidates.sort(key=lambda x: x["distance"])
                            selected_static = static_candidates[:model_cfg.max_obstacles]

                            map_lanes, map_types, map_lengths, map_dists = get_cached_map_polylines(nusc_map, location, target_x, target_y)

                            seen_keys.add(unique_key)
                            item = {
                                "dataset_type": "real",
                                "source": "nuScenes",
                                "dataset_version": "v1.0-trainval",
                                "sample_token": sample_token,
                                "instance_token": instance_token,
                                "scene_token": scene_token,
                                "scene_name": scene_name,
                                "category": cat_name,
                                "city": location,
                                "past_traj": sampled_past,
                                "future_traj": sampled_future,
                                "timestamps_past": [round(t, 4) for t in t_past],
                                "timestamps_future": [round(t, 4) for t in t_future],
                                "neighbors_past": neighbors_past,
                                "neighbor_categories": neighbor_categories,
                                "neighbor_instances": neighbor_instances,
                                "neighbor_distances": neighbor_distances,
                                "static_objects": selected_static,
                                "map_lanes": map_lanes,
                                "map_lane_types": map_types,
                                "map_lane_lengths": map_lengths,
                                "map_lane_distances": map_dists
                            }
                            line = json.dumps(item) + "\n"
                            f_l.write(line)
                            f_j.write(line)
                            saved_count += 1
                            scene_seq_count += 1

                        sample_token = sample["next"]

                    print(f"  [{split_name} Scene {scene_idx + 1:03d}/{len(scenes):03d} | {scene_name:<16} | {location:<24}]: "
                          f"Extracted {scene_seq_count:3d} sequences | Total Saved: {saved_count:,}", flush=True)

                    # A scene is committed only after both mirrored output streams reach disk.
                    f_l.flush()
                    f_j.flush()
                    os.fsync(f_l.fileno())
                    os.fsync(f_j.fileno())
                    split_state.update({
                        "next_scene_index": scene_idx + 1,
                        "last_processed_scene": {"name": scene_name, "token": scene_token},
                        "saved_count": saved_count,
                        "jsonl_bytes": os.path.getsize(jsonl_path),
                        "json_bytes": os.path.getsize(json_path),
                    })
                    save_checkpoint()

            split_state["complete"] = (split_state["next_scene_index"] >= len(scenes) or
                                        (desired_count is not None and saved_count >= desired_count))
            save_checkpoint()

            return saved_count

        tr_cnt = extract_split_stream("Train", all_train_scenes, desired_count=target_count, split_key="train")
        va_cnt = extract_split_stream("Val", all_val_scenes, desired_count=val_target, split_key="val")

        print(f"[Real Data Extractor Finished] Streaming Extraction Complete: {tr_cnt:,} train sequences, {va_cnt:,} val sequences.")
        return [], []


class SyntheticTrajectoryGenerator(BaseTrajectoryExtractor):
    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig()

    def extract_sequences(
        self,
        dataroot: str,
        target_count: int | None = 12800,
        seed: int = 42,
        category: str | list[str] | None = None,
        max_train_scenes: int | None = None,
        max_val_scenes: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        count = target_count if target_count is not None else 12800
        print("=" * 80)
        print(" [Synthetic Generator] Synthesizing kinematics trajectories (Explicit Synthetic Mode)...")
        print(f" Target Count: {count:,} sequences")
        print("=" * 80, flush=True)

        rng = random.Random(seed)
        cities = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
        hz = float(self.model_config.hz)
        past_steps = int(self.model_config.past_len)
        future_steps = int(self.model_config.future_len)
        if hz <= 0 or past_steps <= 0 or future_steps <= 0:
            raise ValueError(
                "ModelConfig must have positive hz, past_len, and future_len "
                f"(got hz={hz}, past_len={past_steps}, future_len={future_steps})."
            )
        dt = 1.0 / hz

        def generate_collection(count_num: int, split: str) -> list[dict[str, Any]]:
            items = []
            for i in range(count_num):
                cur_x = 400.0 + (i % 200) * 15.0 + rng.uniform(-5.0, 5.0)
                cur_y = 600.0 + (i // 200) * 15.0 + rng.uniform(-5.0, 5.0)

                is_ped = (i % 5 == 0)
                if is_ped:
                    speed = rng.uniform(0.8, 1.8)
                    turn_rate = rng.choice([-0.04, 0.0, 0.04]) * (dt / 0.5)
                    cat_name = "human.pedestrian.adult"
                else:
                    speed = rng.uniform(4.0, 15.0)
                    turn_rate = rng.choice([-0.012, 0.0, 0.012]) * (dt / 0.5)
                    cat_name = "vehicle.car"

                heading = rng.uniform(0.0, 2.0 * math.pi)

                past_traj = []
                for _ in range(past_steps):
                    past_traj.append([round(cur_x, 3), round(cur_y, 3)])
                    cur_x += speed * dt * math.cos(heading)
                    cur_y += speed * dt * math.sin(heading)
                    heading += turn_rate * 0.5

                future_traj = []
                for _ in range(future_steps):
                    cur_x += speed * dt * math.cos(heading)
                    cur_y += speed * dt * math.sin(heading)
                    future_traj.append([round(cur_x, 3), round(cur_y, 3)])
                    heading += turn_rate

                # Neighbors
                neighbors = []
                n_cats = []
                n_instances = []
                n_dists = []
                max_neighbors = max(0, min(6, self.model_config.max_agents - 1))
                num_neighbors = rng.randint(0, max_neighbors)
                for n_idx in range(num_neighbors):
                    nx = past_traj[0][0] + rng.uniform(-12.0, 12.0)
                    ny = past_traj[0][1] + rng.uniform(-12.0, 12.0)
                    n_speed = speed + rng.uniform(-1.5, 1.5)
                    n_traj = []
                    for _ in range(past_steps):
                        n_traj.append([round(nx, 3), round(ny, 3)])
                        nx += n_speed * dt * math.cos(heading)
                        ny += n_speed * dt * math.sin(heading)
                    neighbors.append(n_traj)

                    n_cats.append("vehicle.car")
                    n_instances.append(f"syn_neighbor_{split}_{i:06d}_{n_idx}")
                    n_dists.append(round(math.hypot(nx - past_traj[-1][0], ny - past_traj[-1][1]), 2))

                # Static objects
                static_objects = [
                    {
                        "category": "movable_object.trafficcone",
                        "translation": [round(past_traj[-1][0] + 5.0, 3), round(past_traj[-1][1] + 2.0, 3)],
                        "size": [0.4, 0.4, 0.8],
                        "distance": 5.39
                    }
                ]

                # HD map lanes
                map_lanes = []
                map_types = []
                map_lengths = []
                map_dists = []
                for lane_idx in range(min(6, self.model_config.max_lanes)):
                    lane_offset = (lane_idx - 2.5) * 3.5
                    p1 = [round(past_traj[-1][0] + lane_offset, 2), round(past_traj[-1][1] - 30.0, 2)]
                    p2 = [round(p1[0] + 60.0 * math.cos(heading), 2), round(p1[1] + 60.0 * math.sin(heading), 2)]
                    pts = resample_polyline([p1, p2], target_pts=self.model_config.lane_pts)
                    map_lanes.append(pts)
                    map_types.append("lane" if lane_idx < 4 else "walkway")
                    map_lengths.append(60.0)
                    map_dists.append(abs(lane_offset))

                items.append({
                    "dataset_type": "synthetic",
                    "source": "TrajectoryLNAS_Simulator",
                    "dataset_version": "v1.0-synthetic",
                    "sample_token": f"syn_sample_{split}_{i:06d}",
                    "instance_token": f"syn_inst_{split}_{i:06d}",
                    "scene_token": f"syn_scene_{split}_{i // 100:04d}",
                    "scene_name": f"syn_scene_{split}_{i // 100:04d}",
                    "category": cat_name,
                    "city": cities[i % len(cities)],
                    "past_traj": past_traj,
                    "future_traj": future_traj,
                    "timestamps_past": [round(-((past_steps - 1 - t) / hz), 4) for t in range(past_steps)],
                    "timestamps_future": [round((t + 1) / hz, 4) for t in range(future_steps)],
                    "neighbors_past": neighbors,
                    "neighbor_categories": n_cats,
                    "neighbor_instances": n_instances,
                    "neighbor_distances": n_dists,
                    "static_objects": static_objects,
                    "map_lanes": map_lanes,
                    "map_lane_types": map_types,
                    "map_lane_lengths": map_lengths,
                    "map_lane_distances": map_dists
                })

            rng.shuffle(items)
            return items

        val_count = max(800, count // 4)
        train_items = generate_collection(count, "train")
        val_items = generate_collection(val_count, "val")
        return train_items, val_items


class SequenceDatasetWriter(IDatasetWriter):
    def write(self, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], output_dir: str, seed: int = 42) -> tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        train_jsonl = os.path.join(output_dir, "train_sequences.jsonl")
        val_jsonl = os.path.join(output_dir, "val_sequences.jsonl")
        train_json = os.path.join(output_dir, "train_sequences.json")
        val_json = os.path.join(output_dir, "val_sequences.json")

        rng = random.Random(seed)
        if isinstance(train_items, list):
            rng.shuffle(train_items)
        if isinstance(val_items, list):
            rng.shuffle(val_items)

        print("=" * 80)
        print(f"[Dataset Writer] Streaming {len(train_items):,} train sequences (Lazy-Loading JSONL format) to: {train_jsonl}")
        with open(train_jsonl, "w", encoding="utf-8") as f_l, open(train_json, "w", encoding="utf-8") as f_j:
            for item in train_items:
                line = json.dumps(item) + "\n"
                f_l.write(line)
                f_j.write(line)
        size_tr = os.path.getsize(train_jsonl) / (1024 * 1024)
        print(f" Saved Train Sequences: {train_jsonl} ({size_tr:.2f} MB)")
        print("-" * 80)
        print(f"[Dataset Writer] Streaming {len(val_items):,} val sequences (Lazy-Loading JSONL format) to: {val_jsonl}")
        with open(val_jsonl, "w", encoding="utf-8") as f_l, open(val_json, "w", encoding="utf-8") as f_j:
            for item in val_items:
                line = json.dumps(item) + "\n"
                f_l.write(line)
                f_j.write(line)
        size_va = os.path.getsize(val_jsonl) / (1024 * 1024)
        print(f" Saved Val Sequences:   {val_jsonl} ({size_va:.2f} MB)")
        print("=" * 80, flush=True)

        return train_jsonl, val_jsonl


# ==============================================================================
# Pipeline Coordinator (Dependency Inversion Principle - DIP)
# ==============================================================================

class DataPreparationPipeline:
    def __init__(
        self,
        config: DataConfig | None = None,
        map_validator: IMapExpansionValidator | None = None,
        real_extractor: BaseTrajectoryExtractor | None = None,
        synthetic_generator: BaseTrajectoryExtractor | None = None,
        dataset_writer: IDatasetWriter | None = None,
        model_config: ModelConfig | None = None,
    ):
        self.config = config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.map_validator = map_validator or NuScenesMapExpansionValidator()
        self.real_extractor = real_extractor or NuScenesRealDataExtractor(
            config=self.config,
            model_config=self.model_config,
        )
        self.synthetic_generator = synthetic_generator or SyntheticTrajectoryGenerator(
            model_config=self.model_config,
        )
        self.dataset_writer = dataset_writer or SequenceDatasetWriter()

    def prepare(
        self,
        dataroot: str | None = None,
        output_dir: str | None = None,
        target_count: int | None = -999,
        mode: str | None = None,
        seed: int | None = None,
        category: str | list[str] | None = None,
        max_train_scenes: int | None = None,
        max_val_scenes: int | None = None,
        config: DataConfig | None = None
    ) -> bool:
        active_cfg = config or self.config
        active_dataroot = dataroot if dataroot is not None else active_cfg.dataroot
        active_output_dir = output_dir if output_dir is not None else active_cfg.output_dir
        active_count = target_count if target_count != -999 else active_cfg.target_count
        active_mode = mode if mode is not None else active_cfg.mode
        active_seed = seed if seed is not None else active_cfg.seed
        active_category = category if category is not None else active_cfg.category
        active_train_scenes = max_train_scenes if max_train_scenes is not None else active_cfg.max_train_scenes
        active_val_scenes = max_val_scenes if max_val_scenes is not None else active_cfg.max_val_scenes

        count_label = f"{active_count:,} sequences" if active_count is not None else "ALL available sequences"
        cat_label = str(active_category) if active_category is not None else "All Target Categories"

        if active_mode == "synthetic":
            print("[Pipeline Policy] Forced SYNTHETIC mode requested.", flush=True)
            train_items, val_items = self.synthetic_generator.extract_sequences(
                dataroot=active_dataroot,
                target_count=active_count,
                seed=active_seed,
                category=active_category
            )
            tr_p, va_p = self.dataset_writer.write(train_items, val_items, active_output_dir, seed=active_seed)
            return True

        if active_mode in ("real", "auto"):
            maps_available, map_path, map_msg = self.map_validator.is_map_expansion_available(active_dataroot)
            raw_meta_exists = os.path.exists(os.path.join(active_dataroot, "v1.0-trainval")) or os.path.exists(os.path.join(active_dataroot, "sample.json"))
            try:
                import nuscenes
                has_devkit = True
            except ImportError:
                has_devkit = False

            if not os.path.exists(active_dataroot):
                raise FileNotFoundError(
                    f"[DataPreparationPipeline Error] Dataroot '{active_dataroot}' does not exist.\n"
                    f"Expected path: '{active_dataroot}'.\n"
                    f"Relevant configuration key: 'DataConfig.dataroot'.\n"
                    f"Suggested corrective action: Please provide the path to your raw nuScenes dataset via --dataroot <path> or NUSCENES_DATAROOT env var.\n"
                    f"If you are running unit tests or development and explicitly intend to synthesize test data, pass --mode synthetic."
                )
            if not maps_available:
                raise FileNotFoundError(
                    f"[DataPreparationPipeline Error] HD Map expansion files not found in '{active_dataroot}' ({map_msg}).\n"
                    f"Expected path: '{os.path.join(active_dataroot, 'maps', 'expansion')}'.\n"
                    f"Relevant configuration key: 'DataConfig.dataroot'.\n"
                    f"Suggested corrective action: Ensure boston-seaport.json, singapore-onenorth.json, singapore-queenstown.json, singapore-hollandvillage.json exist in maps/expansion/.\n"
                    f"If synthetic testing data is explicitly intended, pass --mode synthetic."
                )
            if not raw_meta_exists:
                raise FileNotFoundError(
                    f"[DataPreparationPipeline Error] nuScenes metadata directory 'v1.0-trainval' not found inside '{active_dataroot}'.\n"
                    f"Expected path: '{os.path.join(active_dataroot, 'v1.0-trainval')}'.\n"
                    f"Relevant configuration key: 'DataConfig.dataroot'.\n"
                    f"Suggested corrective action: Place sample.json, sample_annotation.json, etc. inside '{active_dataroot}/v1.0-trainval/'.\n"
                    f"If synthetic testing data is explicitly intended, pass --mode synthetic."
                )
            if not has_devkit:
                raise ImportError(
                    "[DataPreparationPipeline Error] 'nuscenes-devkit' is not installed in the Python environment.\n"
                    "Relevant configuration key: Python environment dependencies.\n"
                    "Suggested corrective action: Install via 'pip install nuscenes-devkit'.\n"
                    "If synthetic testing data is explicitly intended, pass --mode synthetic."
                )

            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "data_extractor.log")
            with _tee_extraction_output(log_path):
                print(f"[Pipeline Policy] maps/expansion verified ({map_msg}). \nExecuting REAL nuScenes extraction ({count_label} | Category: {cat_label})...", flush=True)
                if isinstance(self.real_extractor, NuScenesRealDataExtractor):
                    self.real_extractor.config.output_dir = active_output_dir
                    self.real_extractor.config.resume_extraction = active_cfg.resume_extraction
                train_items, val_items = self.real_extractor.extract_sequences(
                    dataroot=active_dataroot,
                    target_count=active_count,
                    seed=active_seed,
                    category=active_category,
                    max_train_scenes=active_train_scenes,
                    max_val_scenes=active_val_scenes
                )
            if train_items or val_items:
                tr_p, va_p = self.dataset_writer.write(train_items, val_items, active_output_dir, seed=active_seed)
            return True

        raise ValueError(f"[DataPreparationPipeline Error] Unknown preparation mode: '{active_mode}'. Expected 'real', 'auto', or 'synthetic'.")


# ==============================================================================
# Backward-Compatible Function Facades
# ==============================================================================

def run_official_nuscenes_extraction(
    dataroot: str,
    output_dir: str | None = None,
    target_count: int | None = 12800,
    max_train_scenes: int | None = None,
    max_val_scenes: int | None = None,
    category: str | list[str] | None = None
) -> bool:
    """Backward-compatible facade for official real extraction."""
    pipeline = DataPreparationPipeline(
        real_extractor=NuScenesRealDataExtractor(
            max_train_scenes=max_train_scenes,
            max_val_scenes=max_val_scenes,
            category=category
        )
    )
    return pipeline.prepare(
        dataroot=dataroot,
        output_dir=output_dir,
        target_count=target_count,
        mode="real",
        category=category,
        max_train_scenes=max_train_scenes,
        max_val_scenes=max_val_scenes
    )


def run_synthetic_preparation(output_dir: str, count: int | None = 12800, category: str | list[str] | None = None) -> bool:
    """Backward-compatible facade for synthetic preparation."""
    pipeline = DataPreparationPipeline()
    return pipeline.prepare(dataroot="", output_dir=output_dir, target_count=count, mode="synthetic", category=category)


__all__ = [
    "STATIC_OBSTACLE_CATEGORIES",
    "TARGET_PEDESTRIAN_CATEGORIES",
    "TARGET_VEHICLE_CATEGORIES",
    "DataConfig",
    "DataPreparationPipeline",
    "NuScenesMapExpansionValidator",
    "NuScenesRealDataExtractor",
    "SequenceDatasetWriter",
    "SyntheticTrajectoryGenerator",
    "resample_polyline",
    "run_official_nuscenes_extraction",
    "run_synthetic_preparation"
]


def main():
    default_cfg = DataConfig()

    def parse_optional_count(val):
        if val is None or str(val).strip().lower() in ("none", "null", "all", "-1"):
            return None
        return int(val)

    parser = argparse.ArgumentParser(description="TrajectoryLNAS Data Preparation Pipeline (SOLID)")
    parser.add_argument("--dataroot", type=str, default=default_cfg.dataroot, help="Path to raw nuScenes root")
    parser.add_argument("--output_dir", type=str, default=default_cfg.output_dir, help="Destination directory for sequences JSON")
    parser.add_argument("--mode", type=str, default=default_cfg.mode, choices=["auto", "real", "synthetic"])
    parser.add_argument(
        "--count",
        type=parse_optional_count,
        default=default_cfg.target_count,
        help="Sequence count (e.g. 12800). Pass 'none' or 'all' to extract ALL available sequences without limitation."
    )
    parser.add_argument("--category", type=str, default=default_cfg.category, help="Filter agent category (e.g. 'vehicle', 'pedestrian')")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="Random seed for deterministic shuffling")
    parser.add_argument("--max_train_scenes", type=int, default=default_cfg.max_train_scenes, help="Limit number of train scenes")
    parser.add_argument("--max_val_scenes", type=int, default=default_cfg.max_val_scenes, help="Limit number of val scenes")
    args = parser.parse_args()

    pipeline = DataPreparationPipeline(config=default_cfg)
    pipeline.prepare(
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        target_count=args.count,
        mode=args.mode,
        seed=args.seed,
        category=args.category,
        max_train_scenes=args.max_train_scenes,
        max_val_scenes=args.max_val_scenes
    )


if __name__ == "__main__":
    main()

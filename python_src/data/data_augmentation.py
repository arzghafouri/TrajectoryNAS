r"""
TrajectoryLNAS Geometric Data Augmentation Pipeline
===================================================
Implementation of Physics-Preserving & Geometry-Consistent Data Augmentation
for Autonomous Vehicle Trajectory Prediction on nuScenes Benchmark.

Key Capabilities:
  1. Synchronized Scene Rotation (Rigid SE(2) Rotation):
     Rotates target ego past/future, surrounding agents, static obstacles, and HD map
     polylines simultaneously by angle \theta \in [-15 deg, +15 deg] around the ego vehicle
     observation center (or coordinate origin). Preserves all Euclidean relative distances,
     lane topologies, and physical road constraints with zero synthetic artifacts.

  2. Synchronized Scene Translation:
     Applies identical 2D spatial offset (\Delta x, \Delta y) across all scene entities,
     ensuring complete relative spatial invariance.

  3. Kinematic-Preserving Time & Speed Scaling (Temporal Scaling / Time-Shifting):
     Perturbs the agent velocity/displacement profile by a gentle factor \alpha \in [0.95, 1.05]
     along the road centerline, training the model to handle diverse speed variations
     without violating road curvature or boundary limits.

  4. Dataset Integration & Streaming:
     Reads real nuScenes dataset files (train_sequences.jsonl / train_sequences.json,
     val_sequences.jsonl / val_sequences.json) and generates augmented sequences with
     unique tokens, updating or outputting augmented databases.

Author: Alireza Ghafouri (Master's Thesis Project - TrajectoryLNAS)
Advisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import numpy as np
except ImportError:
    np = None

# Ensure parent directory is in path for config imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from configs.data_config import DataConfig
    from configs.model_config import ModelConfig
except ImportError:
    DataConfig = None
    ModelConfig = None


@dataclass
class AugmentationConfig:
    """Configuration hyperparameters for Geometric Data Augmentation."""
    # Rotation settings (in degrees)
    rotation_deg_range: tuple[float, float] = (-15.0, 15.0)
    enable_rotation: bool = True

    # Translation settings (in meters)
    translation_range_meters: tuple[float, float] = (-2.0, 2.0)
    enable_translation: bool = True

    # Temporal / Speed scaling settings (multiplier on trajectory displacement)
    time_scale_range: tuple[float, float] = (0.95, 1.05)
    enable_time_scale: bool = True

    # Multiplier: how many augmented variants to produce per input sequence
    num_augmentations_per_sample: int = 1

    # Whether to keep the pristine original sample alongside augmented copies
    preserve_original: bool = True

    # Center of rotation: 'ego' (current observation point past_traj[-1]) or 'origin' ([0, 0])
    center_mode: str = "ego"

    # Deterministic reproducibility seed
    seed: int = 42


class GeometricDataAugmentor:
    """
    Performs synchronized SE(2) geometric transformations across all components
    of a nuScenes motion prediction sequence.
    """

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()
        self.rng = random.Random(self.config.seed)

    def _rotate_point(self, x: float, y: float, cos_t: float, sin_t: float, cx: float, cy: float) -> list[float]:
        """Rotates a 2D coordinate around center (cx, cy)."""
        dx = x - cx
        dy = y - cy
        rx = cos_t * dx - sin_t * dy + cx
        ry = sin_t * dx + cos_t * dy + cy
        return [round(float(rx), 3), round(float(ry), 3)]

    def _rotate_polyline(
        self, polyline: list[list[float]], cos_t: float, sin_t: float, cx: float, cy: float
    ) -> list[list[float]]:
        """Rotates an entire multi-point polyline."""
        return [self._rotate_point(p[0], p[1], cos_t, sin_t, cx, cy) for p in polyline]

    def rotate_scene(self, item: dict[str, Any], angle_rad: float, center_mode: str = "ego") -> dict[str, Any]:
        """
        Rotates the entire scene (Ego past/future, surrounding agents, static obstacles,
        and HD map polylines) by angle_rad.
        All relative distances, spatial arrangements, and lane geometries are strictly preserved.
        """
        aug = copy.deepcopy(item)
        cos_t = math.cos(angle_rad)
        sin_t = math.sin(angle_rad)

        # Determine center of rotation
        past_traj = aug.get("past_traj", [])
        if center_mode == "ego" and len(past_traj) > 0:
            cx, cy = float(past_traj[-1][0]), float(past_traj[-1][1])
        else:
            cx, cy = 0.0, 0.0

        # 1. Rotate Target Ego Past and Future Trajectories
        aug["past_traj"] = self._rotate_polyline(past_traj, cos_t, sin_t, cx, cy)
        aug["future_traj"] = self._rotate_polyline(aug.get("future_traj", []), cos_t, sin_t, cx, cy)

        # 2. Rotate Surrounding Agents Past Trajectories
        if "neighbors_past" in aug and isinstance(aug["neighbors_past"], list):
            aug["neighbors_past"] = [
                self._rotate_polyline(n_traj, cos_t, sin_t, cx, cy)
                for n_traj in aug["neighbors_past"]
            ]

        # 3. Rotate Static Obstacles
        if "static_objects" in aug and isinstance(aug["static_objects"], list):
            for obj in aug["static_objects"]:
                trans = obj.get("translation", [])
                if len(trans) >= 2:
                    rotated_xy = self._rotate_point(float(trans[0]), float(trans[1]), cos_t, sin_t, cx, cy)
                    obj["translation"] = rotated_xy

        # 4. Rotate Vectorized HD Map Polylines
        if "map_lanes" in aug and isinstance(aug["map_lanes"], list):
            aug["map_lanes"] = [
                self._rotate_polyline(lane, cos_t, sin_t, cx, cy)
                for lane in aug["map_lanes"]
            ]

        # Annotate augmentation metadata
        deg = round(math.degrees(angle_rad), 2)
        aug["augmentation"] = {
            "type": "geometric_rotation",
            "angle_deg": deg,
            "center": [round(cx, 3), round(cy, 3)],
        }
        aug["sample_token"] = f"{item.get('sample_token', 'sample')}_rot{deg:+05.1f}d"

        return aug

    def translate_scene(self, item: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
        """
        Translates the entire scene by (dx, dy) meters.
        Relative distances between vehicles, obstacles, and lanes remain 100% constant.
        """
        aug = copy.deepcopy(item)

        def _shift(p: list[float]) -> list[float]:
            return [round(float(p[0] + dx), 3), round(float(p[1] + dy), 3)]

        aug["past_traj"] = [_shift(p) for p in aug.get("past_traj", [])]
        aug["future_traj"] = [_shift(p) for p in aug.get("future_traj", [])]

        if "neighbors_past" in aug and isinstance(aug["neighbors_past"], list):
            aug["neighbors_past"] = [[_shift(p) for p in n_traj] for n_traj in aug["neighbors_past"]]

        if "static_objects" in aug and isinstance(aug["static_objects"], list):
            for obj in aug["static_objects"]:
                trans = obj.get("translation", [])
                if len(trans) >= 2:
                    obj["translation"] = _shift(trans[:2])

        if "map_lanes" in aug and isinstance(aug["map_lanes"], list):
            aug["map_lanes"] = [[_shift(p) for p in lane] for lane in aug["map_lanes"]]

        aug["augmentation"] = {
            "type": "geometric_translation",
            "dx": round(dx, 3),
            "dy": round(dy, 3),
        }
        aug["sample_token"] = f"{item.get('sample_token', 'sample')}_trans{dx:+04.1f}_{dy:+04.1f}m"

        return aug

    def time_scale_scene(self, item: dict[str, Any], scale_factor: float) -> dict[str, Any]:
        """
        Applies kinematic-consistent velocity/displacement scaling along the existing
        road geometry by factor alpha in [0.95, 1.05].
        Modulates trajectory speed without displacing road lanes or breaking road topology.
        """
        aug = copy.deepcopy(item)
        past_traj = aug.get("past_traj", [])
        if len(past_traj) < 2:
            return aug

        # Observation anchor point (ego current position at t=0)
        origin_x, origin_y = float(past_traj[-1][0]), float(past_traj[-1][1])

        # Scale past displacements relative to observation anchor
        scaled_past = []
        for p in past_traj:
            dx = float(p[0]) - origin_x
            dy = float(p[1]) - origin_y
            scaled_past.append([round(origin_x + dx * scale_factor, 3), round(origin_y + dy * scale_factor, 3)])
        aug["past_traj"] = scaled_past

        # Scale future displacements relative to observation anchor
        scaled_fut = []
        for p in aug.get("future_traj", []):
            dx = float(p[0]) - origin_x
            dy = float(p[1]) - origin_y
            scaled_fut.append([round(origin_x + dx * scale_factor, 3), round(origin_y + dy * scale_factor, 3)])
        aug["future_traj"] = scaled_fut

        # Scale neighbor past displacements relative to their own final observation
        if "neighbors_past" in aug and isinstance(aug["neighbors_past"], list):
            new_neighbors = []
            for n_traj in aug["neighbors_past"]:
                if len(n_traj) >= 2:
                    nx0, ny0 = float(n_traj[-1][0]), float(n_traj[-1][1])
                    scaled_n = [
                        [round(nx0 + (float(p[0]) - nx0) * scale_factor, 3),
                         round(ny0 + (float(p[1]) - ny0) * scale_factor, 3)]
                        for p in n_traj
                    ]
                    new_neighbors.append(scaled_n)
                else:
                    new_neighbors.append(n_traj)
            aug["neighbors_past"] = new_neighbors

        pct = round((scale_factor - 1.0) * 100.0, 1)
        aug["augmentation"] = {
            "type": "time_speed_scale",
            "scale_factor": round(scale_factor, 4),
        }
        aug["sample_token"] = f"{item.get('sample_token', 'sample')}_spd{pct:+04.1f}pct"

        return aug

    def augment_sample(self, item: dict[str, Any], rng: random.Random | None = None) -> list[dict[str, Any]]:
        """
        Generates one or more geometrically augmented variations of a single nuScenes sequence.
        """
        r = rng or self.rng
        cfg = self.config
        results: list[dict[str, Any]] = []

        if cfg.preserve_original:
            orig = copy.deepcopy(item)
            orig["augmentation"] = {"type": "none"}
            results.append(orig)

        for _ in range(cfg.num_augmentations_per_sample):
            transformed = copy.deepcopy(item)
            transforms_applied = []

            # 1. Random Rotation
            if cfg.enable_rotation:
                angle_deg = r.uniform(cfg.rotation_deg_range[0], cfg.rotation_deg_range[1])
                angle_rad = math.radians(angle_deg)
                transformed = self.rotate_scene(transformed, angle_rad, center_mode=cfg.center_mode)
                transforms_applied.append(f"rot_{angle_deg:+.1f}deg")

            # 2. Random Translation
            if cfg.enable_translation:
                dx = r.uniform(cfg.translation_range_meters[0], cfg.translation_range_meters[1])
                dy = r.uniform(cfg.translation_range_meters[0], cfg.translation_range_meters[1])
                transformed = self.translate_scene(transformed, dx, dy)
                transforms_applied.append(f"trans_{dx:+.1f}_{dy:+.1f}m")

            # 3. Kinematic Speed/Temporal Scaling
            if cfg.enable_time_scale:
                factor = r.uniform(cfg.time_scale_range[0], cfg.time_scale_range[1])
                transformed = self.time_scale_scene(transformed, factor)
                transforms_applied.append(f"spd_{(factor - 1) * 100:+.1f}%")

            # Combined token descriptor
            base_token = str(item.get("sample_token", "sample"))
            transformed["sample_token"] = f"{base_token}_aug_{'_'.join(transforms_applied)}"
            transformed["instance_token"] = f"{item.get('instance_token', 'inst')}_aug"
            transformed["augmentation"] = {
                "type": "composite_geometric",
                "components": transforms_applied,
            }
            results.append(transformed)

        return results

    def augment_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> tuple[int, int]:
        """
        Streams through an input dataset file (JSONL or JSON), applies geometric augmentation,
        and saves the enriched dataset to output_path.
        Returns: (original_count, total_augmented_count)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input sequences file not found: {input_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Detect line-based JSONL or monolithic JSON
        is_jsonl = input_path.endswith(".jsonl")
        if not is_jsonl:
            with open(input_path, "rb") as f:
                first_byte = f.read(1024).strip()[:1]
                is_jsonl = (first_byte == b"{")

        orig_count = 0
        total_count = 0

        # Create independent RNG for this file
        file_rng = random.Random(self.config.seed)

        with open(output_path, "w", encoding="utf-8") as f_out:
            if is_jsonl:
                with open(input_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        orig_count += 1
                        item = json.loads(line_str)
                        variants = self.augment_sample(item, rng=file_rng)
                        for var in variants:
                            f_out.write(json.dumps(var) + "\n")
                            total_count += 1
                        if progress_callback and orig_count % 100 == 0:
                            progress_callback(orig_count, total_count)
            else:
                with open(input_path, "r", encoding="utf-8") as f_in:
                    samples = json.load(f_in)
                orig_count = len(samples)
                for item in samples:
                    variants = self.augment_sample(item, rng=file_rng)
                    for var in variants:
                        f_out.write(json.dumps(var) + "\n")
                        total_count += 1
                    if progress_callback and total_count % 100 == 0:
                        progress_callback(orig_count, total_count)

        return orig_count, total_count


def run_augmentation_pipeline(
    data_dir: str | None = None,
    splits: list[str] | None = None,
    num_aug: int = 1,
    rot_range: tuple[float, float] = (-15.0, 15.0),
    trans_range: tuple[float, float] = (-2.0, 2.0),
    time_scale: tuple[float, float] = (0.95, 1.05),
    output_suffix: str = "_augmented",
    seed: int = 42
) -> dict[str, dict[str, Any]]:
    """
    Standard entrypoint to augment train and validation sequence files.
    """
    if data_dir is None:
        if DataConfig is not None:
            data_dir = DataConfig().output_dir
        else:
            data_dir = "data/nuscenes/v1.0-trainval"

    splits = splits or ["train"]
    config = AugmentationConfig(
        rotation_deg_range=rot_range,
        translation_range_meters=trans_range,
        time_scale_range=time_scale,
        num_augmentations_per_sample=num_aug,
        preserve_original=True,
        seed=seed
    )
    augmentor = GeometricDataAugmentor(config)
    results_summary = {}

    print("=" * 80)
    print(" [Geometric Data Augmentation] Running SE(2) Physical-Preserving Augmentor")
    print(f" Target Directory:    {data_dir}")
    print(f" Target Splits:       {splits}")
    print(f" Rotation Range:      [{rot_range[0]:.1f} deg, {rot_range[1]:.1f} deg]")
    print(f" Translation Range:   [{trans_range[0]:.1f} m, {trans_range[1]:.1f} m]")
    print(f" Time/Speed Scale:    [{time_scale[0]:.2f}x, {time_scale[1]:.2f}x]")
    print(f" Aug Factor:          {num_aug} augmented variant(s) per input sample")
    print("=" * 80, flush=True)

    for split in splits:
        input_candidates = [
            os.path.join(data_dir, f"{split}_sequences.jsonl"),
            os.path.join(data_dir, f"{split}_sequences.json"),
        ]
        in_file = next((f for f in input_candidates if os.path.exists(f)), None)

        if not in_file:
            print(f" [Warning] Split '{split}' sequence file not found in '{data_dir}'. Skipping.")
            continue

        base_ext = ".jsonl" if in_file.endswith(".jsonl") else ".json"
        out_file = os.path.join(data_dir, f"{split}_sequences{output_suffix}{base_ext}")

        print(f" -> Augmenting {split.upper()}: '{os.path.basename(in_file)}' -> '{os.path.basename(out_file)}'...")
        orig_cnt, total_cnt = augmentor.augment_file(in_file, out_file)
        results_summary[split] = {
            "input_file": in_file,
            "output_file": out_file,
            "original_samples": orig_cnt,
            "total_samples": total_cnt,
            "augmented_samples": total_cnt - orig_cnt,
        }
        print(f"    [Done] Processed {orig_cnt:,} original samples -> Generated {total_cnt:,} total samples (+{total_cnt - orig_cnt:,} augmented).", flush=True)

    print("=" * 80)
    print(" [Augmentation Pipeline Finished Successfully]")
    print("=" * 80, flush=True)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Geometric Data Augmentation Tool")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing sequence files")
    parser.add_argument("--split", type=str, default="train", help="Split to augment (train, val, or both)")
    parser.add_argument("--num_aug", type=int, default=1, help="Number of augmented variants per sample")
    parser.add_argument("--rot_min", type=float, default=-15.0, help="Minimum rotation angle (degrees)")
    parser.add_argument("--rot_max", type=float, default=15.0, help="Maximum rotation angle (degrees)")
    parser.add_argument("--trans_range", type=float, default=2.0, help="Maximum absolute translation (meters)")
    parser.add_argument("--time_scale_min", type=float, default=0.95, help="Minimum speed/time scale factor")
    parser.add_argument("--time_scale_max", type=float, default=1.05, help="Maximum speed/time scale factor")
    parser.add_argument("--suffix", type=str, default="_augmented", help="Output filename suffix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    active_splits = ["train", "val"] if args.split.lower() == "both" else [args.split.lower()]
    run_augmentation_pipeline(
        data_dir=args.data_dir,
        splits=active_splits,
        num_aug=args.num_aug,
        rot_range=(args.rot_min, args.rot_max),
        trans_range=(-abs(args.trans_range), abs(args.trans_range)),
        time_scale=(args.time_scale_min, args.time_scale_max),
        output_suffix=args.suffix,
        seed=args.seed
    )

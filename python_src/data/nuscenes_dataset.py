"""
nuScenes Trajectory Prediction Dataset, DataLoader & Unified Factory
======================================================================
Author: Alireza Ghafouri (Master's Thesis Project - TrajectoryLNAS)
Paper Reference: "TrajectoryLNAS: Trajectory Prediction Architecture Optimization"

Contains NuScenesPredictionDataset, DataLoader builders, and unified factory dispatching.
"""

import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.data_config import DataConfig
from data.synthetic_dataset import SyntheticTrajectoryDataset


class _DynamicDatasetPaths(dict):
    """Dynamic dictionary ensuring DEFAULT_DATASET_PATHS always reflects DataConfig."""
    def __getitem__(self, item):
        if item == "nuscenes":
            d_cfg = DataConfig()
            return {
                "train": d_cfg.output_dir,
                "val": d_cfg.output_dir,
                "dataroot": d_cfg.dataroot,
                "download_url": "https://www.nuscenes.org/download",
                "description": "nuScenes Prediction Benchmark (v1.0-trainval metadata + map expansion)"
            }
        return super().__getitem__(item)


DEFAULT_DATASET_PATHS = _DynamicDatasetPaths({
    "nuscenes": {
        "train": DataConfig().output_dir,
        "val": DataConfig().output_dir,
        "dataroot": DataConfig().dataroot,
        "download_url": "https://www.nuscenes.org/download",
        "description": "nuScenes Prediction Benchmark (v1.0-trainval metadata + map expansion)"
    }
})


class NuScenesPredictionDataset(Dataset):
    """
    PyTorch Dataset for nuScenes Motion Prediction.
    Loads annotated agent trajectory sequences and extracts:
      - Configured history and prediction steps from ModelConfig
      - Surrounding agents within 50m radius
      - Vectorized HD Map lanes (drivable areas and lane dividers)
      - Ego-centric local frame normalization
    """
    def __init__(
        self,
        data_dir=None,
        split="train",
        past_len=None,
        future_len=None,
        max_agents=None,
        max_lanes=None,
        lane_pts=None,
        sample_percent=None,
        generate_mock_if_missing=False,
        max_obstacles=None,
        model_config=None,
    ):
        super().__init__()
        from configs.model_config import ModelConfig
        m_cfg = model_config or ModelConfig()
        self.model_config = m_cfg
        self.data_dir = data_dir or DataConfig().output_dir
        self.split = split
        self.past_len = past_len if past_len is not None else m_cfg.past_len
        self.future_len = future_len if future_len is not None else m_cfg.future_len
        self.max_agents = max_agents if max_agents is not None else m_cfg.max_agents
        self.max_lanes = max_lanes if max_lanes is not None else m_cfg.max_lanes
        self.lane_pts = lane_pts if lane_pts is not None else m_cfg.lane_pts
        self.max_obstacles = max_obstacles if max_obstacles is not None else m_cfg.max_obstacles


        self.sample_percent = sample_percent

        if os.path.isfile(self.data_dir):
            self.samples_file = self.data_dir
        else:
            jsonl_file = os.path.join(data_dir, f"{split}_sequences.jsonl")
            json_file = os.path.join(data_dir, f"{split}_sequences.json")
            if os.path.exists(jsonl_file):
                self.samples_file = jsonl_file
            elif os.path.exists(json_file):
                self.samples_file = json_file
            else:
                self.samples_file = jsonl_file

        if not os.path.exists(self.samples_file):
            if generate_mock_if_missing:
                print(f"[NuScenesDataset] '{self.samples_file}' not found. Generating high-capacity mock nuScenes sequence database (6,400 sequences)...")
                self._generate_mock_nuscenes_data(self.samples_file, count=6400)
            else:
                raise FileNotFoundError(
                    f"[NuScenesDataset Error] Real nuScenes dataset sequences file not found.\n"
                    f"  1. Missing file: '{os.path.basename(self.samples_file)}'\n"
                    f"  2. Expected path: '{self.samples_file}'\n"
                    f"  3. Relevant configuration key: 'DataConfig.dataroot' or 'SearchConfig.train_path'/'val_path'\n"
                    f"  4. Suggested corrective action: Extract real dataset using 'python data/preprocessing.py --dataroot <path_to_nuscenes>' "
                    f"or place 'train_sequences.jsonl' or 'train_sequences.json' in '{data_dir}'. "
                    f"For explicit synthetic test execution, pass '--dataset synthetic'."
                )

        self.use_line_indexing = False
        self.offsets = []
        self.samples = []

        # Detect line-delimited JSON (JSONL) format for Lazy-Loading byte offset streaming
        with open(self.samples_file, "rb") as f:
            first_byte = b""
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                stripped = chunk.strip()
                if stripped:
                    first_byte = stripped[:1]
                    break

        if first_byte == b"{" or self.samples_file.endswith(".jsonl"):
            self.use_line_indexing = True
            with open(self.samples_file, "rb") as f:
                offset = 0
                for line in f:
                    if line.strip():
                        self.offsets.append(offset)
                    offset += len(line)
            total_available = len(self.offsets)
        else:
            with open(self.samples_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
            total_available = len(self.samples)

        if self.sample_percent is not None and 0 < self.sample_percent < 100.0:
            target_count = max(1, int(total_available * (self.sample_percent / 100.0)))
            import random
            rng = random.Random(42)
            indices = list(range(total_available))
            rng.shuffle(indices)
            self.indices = indices[:target_count]
            print(f"[NuScenesDataset] Uniformly sampled & shuffled {len(self.indices):,} {split} sequences ({self.sample_percent}% of total {total_available:,} available samples across all scenes, seed=42).")
        else:
            self.indices = list(range(total_available))
            print(f"[NuScenesDataset] Successfully indexed {total_available:,} {split} sequences from: {self.samples_file} (Lazy-Loading: {self.use_line_indexing})")

    def __len__(self):
        return len(self.indices)

    def _rotate_points(self, points, origin, angle_rad):
        shifted = points - origin
        rot_mat = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        return np.matmul(shifted, rot_mat.T)

    def _validate_temporal_grid(self, item: dict) -> None:
        """Fail early when timestamped sequences were extracted for another model grid."""
        for trajectory_key, timestamp_key, expected_len in (
            ("past_traj", "timestamps_past", self.past_len),
            ("future_traj", "timestamps_future", self.future_len),
        ):
            timestamps = item.get(timestamp_key)
            if timestamps is None:
                continue  # Legacy files without timestamps remain loadable.
            trajectory = item.get(trajectory_key, [])
            if len(timestamps) != len(trajectory) or len(trajectory) != expected_len:
                raise ValueError(
                    f"Sequence {item.get('sample_token', '<unknown>')} has {len(trajectory)} "
                    f"{trajectory_key} points and {len(timestamps)} timestamps; "
                    f"ModelConfig expects {expected_len}. Re-extract sequences after changing ModelConfig."
                )
            if len(timestamps) > 1:
                intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
                expected_dt = 1.0 / float(self.model_config.hz)
                if not np.allclose(intervals, expected_dt, rtol=0.0, atol=1e-3):
                    observed = sorted({round(float(dt), 4) for dt in intervals})
                    raise ValueError(
                        f"Sequence {item.get('sample_token', '<unknown>')} uses time intervals "
                        f"{observed}, but ModelConfig.hz={self.model_config.hz} expects "
                        f"{expected_dt:g}s. Re-extract sequences before training/evaluation."
                    )

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        if self.use_line_indexing:
            offset = self.offsets[file_idx]
            with open(self.samples_file, "r", encoding="utf-8") as f:
                f.seek(offset)
                line = f.readline()
            item = json.loads(line)
        else:
            item = self.samples[file_idx]

        self._validate_temporal_grid(item)

        # Extract raw coordinates
        past_raw = np.array(item["past_traj"], dtype=np.float32)
        future_raw = np.array(item["future_traj"], dtype=np.float32)

        # Slice or pad past_raw to exact self.past_len
        if len(past_raw) >= self.past_len:
            past_raw = past_raw[-self.past_len:]
        else:
            pad_p = self.past_len - len(past_raw)
            past_raw = np.pad(past_raw, ((pad_p, 0), (0, 0)), mode="edge")

        # Slice or extrapolate future_raw to exact self.future_len (supports 6s horizon)
        if len(future_raw) >= self.future_len:
            future_raw = future_raw[:self.future_len]
        else:
            pad_f = self.future_len - len(future_raw)
            if len(future_raw) >= 2:
                v = future_raw[-1] - future_raw[-2]
                extra = [future_raw[-1] + (s + 1) * v for s in range(pad_f)]
                future_raw = np.vstack([future_raw, np.array(extra, dtype=np.float32)])
            elif len(future_raw) == 1:
                future_raw = np.pad(future_raw, ((0, pad_f), (0, 0)), mode="edge")
            else:
                future_raw = np.zeros((self.future_len, 2), dtype=np.float32)

        # Ego-centric transformation at observation timestamp
        origin = past_raw[-1].copy()
        dx = past_raw[-1, 0] - past_raw[-2, 0] if len(past_raw) >= 2 else 1.0
        dy = past_raw[-1, 1] - past_raw[-2, 1] if len(past_raw) >= 2 else 0.0
        heading = math.atan2(dy, dx)

        norm_past = self._rotate_points(past_raw, origin, heading)
        norm_future = self._rotate_points(future_raw, origin, heading)


        # Surrounding agent interactions
        all_agents = [norm_past]
        agent_mask = [True]  # Target ego vehicle is always valid at index 0
        raw_neighbors = item.get("neighbors_past", [])
        for n_traj in raw_neighbors[:self.max_agents - 1]:
            n_arr = np.array(n_traj, dtype=np.float32)
            if len(n_arr) < self.past_len:
                pad = np.repeat(n_arr[-1:], self.past_len - len(n_arr), axis=0)
                n_arr = np.vstack([n_arr, pad])
            norm_n = self._rotate_points(n_arr[:self.past_len], origin, heading)
            all_agents.append(norm_n)
            agent_mask.append(True)

        # Pad remaining agents with zeros and mark as invalid in agent_mask
        while len(all_agents) < self.max_agents:
            all_agents.append(np.zeros((self.past_len, 2), dtype=np.float32))
            agent_mask.append(False)

        if len(all_agents) > 0:
            agents_tensor = np.stack(all_agents, axis=1) # [past_len, max_agents, 2]
        else:
            agents_tensor = np.zeros((self.past_len, self.max_agents, 2), dtype=np.float32)

        # Vectorized HD Map lanes (Strictly normalized to [max_lanes, lane_pts, 2])
        lanes_list = []
        lane_mask = []
        raw_lanes = item.get("map_lanes", [])
        for lane in raw_lanes[:self.max_lanes]:
            l_arr = np.array(lane, dtype=np.float32)
            if len(l_arr) < 2:
                continue
            norm_l = self._rotate_points(l_arr, origin, heading)
            if len(norm_l) != self.lane_pts:
                orig_x = np.linspace(0, 1, len(norm_l))
                target_x = np.linspace(0, 1, self.lane_pts)
                resampled_lane = np.zeros((self.lane_pts, 2), dtype=np.float32)
                resampled_lane[:, 0] = np.interp(target_x, orig_x, norm_l[:, 0])
                resampled_lane[:, 1] = np.interp(target_x, orig_x, norm_l[:, 1])
                lanes_list.append(resampled_lane)
            else:
                lanes_list.append(norm_l)
            lane_mask.append(True)

        # Pad remaining lanes with strict zero-padding and mark as False in lane_mask (NO synthetic lanes)
        while len(lanes_list) < self.max_lanes:
            lanes_list.append(np.zeros((self.lane_pts, 2), dtype=np.float32))
            lane_mask.append(False)

        if len(lanes_list) > 0:
            map_lanes_tensor = np.stack(lanes_list, axis=0) # [max_lanes, lane_pts, 2]
        else:
            map_lanes_tensor = np.zeros((self.max_lanes, self.lane_pts, 2), dtype=np.float32)

        # Nearby static obstacles in the same ego-centric local frame.
        obstacle_features = []
        obstacle_types = []
        obstacle_type_map = {
            "movable_object.barrier": 0,
            "movable_object.trafficcone": 1,
            "movable_object.debris": 2,
            "movable_object.pushable_pullable": 3,
            "static_object.bicycle_rack": 4,
        }
        for obstacle in item.get("static_objects", [])[:self.max_obstacles]:
            translation = obstacle.get("translation", [])
            if len(translation) < 2:
                continue
            local_xy = self._rotate_points(
                np.asarray([translation[:2]], dtype=np.float32), origin, heading
            )[0]
            size = obstacle.get("size", [])
            size_x = float(size[0]) if len(size) > 0 else 0.0
            size_y = float(size[1]) if len(size) > 1 else size_x
            obstacle_features.append([float(local_xy[0]), float(local_xy[1]), size_x, size_y])
            obstacle_types.append(obstacle_type_map.get(str(obstacle.get("category", "")), 5))

        obstacle_mask = [True] * len(obstacle_features)
        while len(obstacle_features) < self.max_obstacles:
            obstacle_features.append([0.0, 0.0, 0.0, 0.0])
            obstacle_types.append(0)
            obstacle_mask.append(False)

        if len(obstacle_features) == 0:
            obstacle_features = np.zeros((self.max_obstacles, 4), dtype=np.float32)
            obstacle_types = np.zeros(self.max_obstacles, dtype=np.int64)
            obstacle_mask = np.zeros(self.max_obstacles, dtype=bool)

        # Canonical category mapping
        cat_map = {
            "vehicle.car": 0, "vehicle.truck": 1, "vehicle.bus.rigid": 2, "vehicle.bus.bendy": 2,
            "vehicle.motorcycle": 3, "vehicle.bicycle": 4, "vehicle.trailer": 5, "vehicle.construction": 6,
            "vehicle.emergency.police": 1, "vehicle.emergency.ambulance": 1,
            "human.pedestrian.adult": 7, "human.pedestrian.child": 7, "human.pedestrian.construction_worker": 7,
            "human.pedestrian.police_officer": 7, "human.pedestrian.personal_mobility": 8,
            "human.pedestrian.stroller": 8, "human.pedestrian.wheelchair": 8
        }
        lane_type_map = {
            "lane": 0, "lane_connector": 1, "connector": 1,
            "walkway": 2, "ped_crossing": 3,
            "divider": 4, "road_divider": 4,
        }

        raw_cat = item.get("category", "vehicle.car")
        cat_id = cat_map.get(raw_cat, 7 if ("human" in raw_cat or "pedestrian" in raw_cat) else 0)

        raw_neighbor_cats = item.get("neighbor_categories", [])
        neighbor_cat_ids = [cat_id]
        for nc in raw_neighbor_cats[:self.max_agents - 1]:
            nc_str = str(nc)
            nc_id = cat_map.get(nc_str, 7 if ("human" in nc_str or "pedestrian" in nc_str) else 0)
            neighbor_cat_ids.append(nc_id)
        while len(neighbor_cat_ids) < self.max_agents:
            neighbor_cat_ids.append(0)

        raw_lane_types = item.get("map_lane_types", [])
        lane_type_ids = []
        for lt in raw_lane_types[:self.max_lanes]:
            lane_type_ids.append(lane_type_map.get(str(lt), 0))
        while len(lane_type_ids) < self.max_lanes:
            lane_type_ids.append(0)

        return {
            "past": torch.tensor(norm_past, dtype=torch.float32),                 # [20, 2]
            "future": torch.tensor(norm_future, dtype=torch.float32),             # [30, 2]
            "agents_past": torch.tensor(agents_tensor, dtype=torch.float32),      # [20, N, 2]
            "agent_mask": torch.tensor(agent_mask, dtype=torch.bool),             # [N]
            "map_lanes": torch.tensor(map_lanes_tensor, dtype=torch.float32),     # [L, 10, 2]
            "lane_mask": torch.tensor(lane_mask, dtype=torch.bool),               # [L]
            "category_id": torch.tensor(cat_id, dtype=torch.long),                # scalar
            "neighbor_categories": torch.tensor(neighbor_cat_ids, dtype=torch.long), # [N]
            "static_obstacles": torch.tensor(obstacle_features, dtype=torch.float32), # [O, 4]
            "static_obstacle_types": torch.tensor(obstacle_types, dtype=torch.long), # [O]
            "static_obstacle_mask": torch.tensor(obstacle_mask, dtype=torch.bool), # [O]
            "map_lane_types": torch.tensor(lane_type_ids, dtype=torch.long),      # [L]
            "origin": torch.tensor(origin, dtype=torch.float32),                  # [2]
            "heading": torch.tensor(heading, dtype=torch.float32),                # [1]
            "instance_token": item.get("instance_token", f"token_{idx}"),
            "city": item.get("city", "singapore-onenorth"),
            "sample_idx": torch.tensor(idx, dtype=torch.long)
        }

    def _generate_mock_nuscenes_data(self, output_path, count=6400):
        """Generates realistic mock nuScenes trajectory sequences with simulated HD map lanes."""
        target_dir = os.path.dirname(os.path.abspath(output_path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        items = []

        cities = ["singapore-onenorth", "boston-seaport", "singapore-queenstown", "singapore-hollandvillage"]
        import random
        rng = random.Random(42)
        model_cfg = self.model_config
        dt = 1.0 / float(model_cfg.hz)
        turn_scale = dt / 0.5

        for i in range(count):
            cur_x = 400.0 + (i % 200) * 15.0 + rng.uniform(-5.0, 5.0)
            cur_y = 600.0 + (i // 200) * 15.0 + rng.uniform(-5.0, 5.0)
            speed = rng.uniform(4.0, 16.0)
            heading = rng.uniform(0.0, 2.0 * math.pi)
            turn_rate = rng.choice([-0.015, -0.008, 0.0, 0.008, 0.015]) * turn_scale

            past_traj = []
            for t in range(self.past_len):
                past_traj.append([round(cur_x, 3), round(cur_y, 3)])
                cur_x += speed * dt * math.cos(heading)
                cur_y += speed * dt * math.sin(heading)
                heading += turn_rate

            future_traj = []
            for t in range(self.future_len):
                cur_x += speed * dt * math.cos(heading)
                cur_y += speed * dt * math.sin(heading)
                future_traj.append([round(cur_x, 3), round(cur_y, 3)])
                heading += turn_rate

            neighbors = []
            neighbor_cats = []
            num_neighbors = rng.randint(0, min(4, max(0, self.max_agents - 1)))
            for n in range(num_neighbors):
                nx = past_traj[0][0] + rng.uniform(-15.0, 15.0)
                ny = past_traj[0][1] + rng.uniform(-15.0, 15.0)
                n_speed = speed + rng.uniform(-2.0, 2.0)
                n_pts = []
                for t in range(self.past_len):
                    n_pts.append([round(nx, 3), round(ny, 3)])
                    nx += n_speed * dt * math.cos(heading)
                    ny += n_speed * dt * math.sin(heading)
                neighbors.append(n_pts)
                neighbor_cats.append(rng.choice(["vehicle.car", "vehicle.truck", "human.pedestrian.adult"]))

            map_lanes = []
            lane_types = []
            for lane_idx in range(min(6, self.max_lanes)):
                lane_offset = (lane_idx - 2.5) * 3.5
                p1 = [round(past_traj[-1][0] + lane_offset, 2), round(past_traj[-1][1] - 30.0, 2)]
                p2 = [round(p1[0] + 60.0 * math.cos(heading), 2), round(p1[1] + 60.0 * math.sin(heading), 2)]
                map_lanes.append([p1, p2])
                lane_types.append(rng.choice(["lane", "lane_connector"]))

            items.append({
                "instance_token": f"nu_token_{i:06d}",
                "city": cities[i % len(cities)],
                "category": rng.choice(["vehicle.car", "vehicle.truck", "human.pedestrian.adult", "vehicle.motorcycle"]),
                "past_traj": past_traj,
                "future_traj": future_traj,
                "timestamps_past": [round(-((self.past_len - 1 - t) / model_cfg.hz), 4) for t in range(self.past_len)],
                "timestamps_future": [round((t + 1) / model_cfg.hz, 4) for t in range(self.future_len)],
                "neighbors_past": neighbors,
                "neighbor_categories": neighbor_cats,
                "map_lanes": map_lanes,
                "map_lane_types": lane_types
            })

        rng.shuffle(items)
        with open(output_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")


def build_nuscenes_dataloader(
    data_dir=None,
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=0,
    past_len=None,
    future_len=None,
    max_agents=None,
    max_lanes=None,
    lane_pts=None,
    sample_percent=None,
    max_obstacles=None,
    model_config=None,
    hard_sample_mining: float = 0.0,
    hard_sample_ema: float = 0.7,
):
    """Creates a production-ready DataLoader for nuScenes Prediction Benchmark with optional Hard Sample Mining."""
    if data_dir is None:
        data_dir = DataConfig().output_dir

    if os.name == "nt" and num_workers > 0:
        num_workers = 0

    dataset = NuScenesPredictionDataset(
        data_dir=data_dir,
        split=split,
        past_len=past_len,
        future_len=future_len,
        max_agents=max_agents,
        max_lanes=max_lanes,
        lane_pts=lane_pts,
        sample_percent=sample_percent,
        max_obstacles=max_obstacles,
        model_config=model_config,
    )
    effective_bs = min(batch_size, len(dataset)) if len(dataset) > 0 else batch_size
    should_drop = (split == "train" and len(dataset) > effective_bs and len(dataset) % effective_bs != 0)

    sampler = None
    hsm_val = float(hard_sample_mining or 0.0)
    if split == "train" and hsm_val > 0.0 and len(dataset) > 0:
        from data.hard_sample_mining import HardSampleMiningManager
        manager = HardSampleMiningManager(
            num_samples=len(dataset),
            sensitivity=hsm_val,
            ema_momentum=hard_sample_ema,
        )
        sampler = manager.get_sampler(replacement=True)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=should_drop
    )
    return loader


def get_trajectory_dataloaders(
    dataset_name: str = "nuscenes",
    train_dir: str | None = None,
    val_dir: str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    sample_percent: float | None = None,
    past_len: int | None = None,
    future_len: int | None = None,
    hard_sample_mining: float | None = None,
    **kwargs
) -> tuple[DataLoader, DataLoader]:
    """
    Unified DataLoader Factory function.
    Provides DataLoaders for nuScenes and synthetic dataset benchmarks with Hard Sample Mining.
    """

    from configs.model_config import ModelConfig
    model_cfg = kwargs.get("model_config") or ModelConfig()
    past_len = past_len if past_len is not None else model_cfg.past_len
    future_len = future_len if future_len is not None else model_cfg.future_len
    max_agents = kwargs.get("max_agents", model_cfg.max_agents)
    max_lanes = kwargs.get("max_lanes", model_cfg.max_lanes)
    lane_pts = kwargs.get("lane_pts", model_cfg.lane_pts)
    max_obstacles = kwargs.get("max_obstacles", model_cfg.max_obstacles)

    raw_ds = kwargs.get("dataset_type") or kwargs.get("dataset_name") or dataset_name or "nuscenes"
    ds_name = str(raw_ds).lower() if raw_ds else "nuscenes"

    t_path = kwargs.get("train_path", train_dir) or DEFAULT_DATASET_PATHS["nuscenes"]["train"]
    v_path = kwargs.get("val_path", val_dir) or DEFAULT_DATASET_PATHS["nuscenes"]["val"]

    if sample_percent is None:
        sample_percent = kwargs.get("search_samples_percent", kwargs.get("train_samples_percent", kwargs.get("sample_percent", None)))

    # Resolve hard sample mining sensitivity
    hsm = hard_sample_mining
    if hsm is None:
        hsm = kwargs.get("hard_sample_sensitivity", kwargs.get("hard_mining_sensitivity"))
    if hsm is None and "train_config" in kwargs:
        hsm = getattr(kwargs["train_config"], "hard_sample_mining", 0.0)
    hsm = float(hsm or 0.0)
    hsm_ema = float(kwargs.get("hard_sample_ema", 0.7))

    if ds_name == "synthetic":
        train_samples = 1200
        val_samples = 300
        if sample_percent is not None and sample_percent > 0:
            train_samples = max(100, int(train_samples * (sample_percent / 100.0)))
            val_samples = max(50, int(val_samples * (sample_percent / 100.0)))

        train_ds = SyntheticTrajectoryDataset(num_samples=train_samples, num_agents=max_agents, model_config=model_cfg)
        val_ds = SyntheticTrajectoryDataset(num_samples=val_samples, num_agents=max_agents, model_config=model_cfg)

        train_sampler = None
        if hsm > 0.0:
            from data.hard_sample_mining import HardSampleMiningManager
            manager = HardSampleMiningManager(
                num_samples=len(train_ds),
                sensitivity=hsm,
                ema_momentum=hsm_ema,
            )
            train_sampler = manager.get_sampler(replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = build_nuscenes_dataloader(
            t_path, split="train", batch_size=batch_size, shuffle=True, num_workers=num_workers,
            sample_percent=sample_percent, past_len=past_len, future_len=future_len,
            max_agents=max_agents, max_lanes=max_lanes, lane_pts=lane_pts,
            max_obstacles=max_obstacles, model_config=model_cfg,
            hard_sample_mining=hsm, hard_sample_ema=hsm_ema
        )
        val_loader = build_nuscenes_dataloader(
            v_path, split="val", batch_size=batch_size, shuffle=False, num_workers=num_workers,
            sample_percent=sample_percent, past_len=past_len, future_len=future_len,
            max_agents=max_agents, max_lanes=max_lanes, lane_pts=lane_pts,
            max_obstacles=max_obstacles, model_config=model_cfg
        )
        return train_loader, val_loader

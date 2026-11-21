"""
Synthetic Trajectory Dataset
============================
Single Responsibility Principle (SRP):
Dedicated exclusively to generating synthetic, kinematic-compliant multi-agent vehicle
trajectories for testing, validation, and zero-hardware fallback scenarios.
"""

import math

import torch
from torch.utils.data import Dataset


class SyntheticTrajectoryDataset(Dataset):
    """
    Generates synthetic trajectories and context tensors using ModelConfig dimensions.
      - realistic vehicle kinematics: curved turns, straight cruising, speed variations, multi-agent interactions.
    """
    def __init__(
        self,
        num_samples: int = 1200,
        num_agents: int | None = None,
        seed: int = 42,
        past_len: int | None = None,
        future_len: int | None = None,
        model_config=None,
    ):
        self.num_samples = num_samples
        from configs.model_config import ModelConfig
        self.model_cfg = model_config or ModelConfig()
        self.num_agents = num_agents if num_agents is not None else self.model_cfg.max_agents
        num_agents = self.num_agents
        self.past_len = past_len if past_len is not None else self.model_cfg.past_len
        self.future_len = future_len if future_len is not None else self.model_cfg.future_len
        self.max_obstacles = self.model_cfg.max_obstacles
        self.lane_pts = self.model_cfg.lane_pts
        self.max_lanes = self.model_cfg.max_lanes

        self.past_data: list[torch.Tensor] = []
        self.future_data: list[torch.Tensor] = []

        g = torch.Generator()
        g.manual_seed(seed)

        if self.model_cfg.hz <= 0:
            raise ValueError(f"ModelConfig.hz must be positive, got {self.model_cfg.hz}.")
        dt = 1.0 / float(self.model_cfg.hz)

        for _ in range(num_samples):
            # Base velocity (8 - 14 m/s approx 30 - 50 km/h) and road curvature
            speed = 8.0 + torch.rand(1, generator=g).item() * 6.0
            curvature = (torch.rand(1, generator=g).item() - 0.5) * 0.05 * (dt / 0.5)

            past_traj = []
            cur_x, cur_y, cur_heading = 0.0, 0.0, 0.0

            # Past timesteps
            for _ in range(self.past_len):
                past_traj.append([cur_x, cur_y])
                cur_heading += curvature
                cur_x += speed * dt * math.cos(cur_heading) + (torch.randn(1, generator=g).item() * 0.05)
                cur_y += speed * dt * math.sin(cur_heading) + (torch.randn(1, generator=g).item() * 0.05)

            # Future timesteps
            future_traj = []
            for _ in range(self.future_len):
                cur_heading += curvature
                cur_x += speed * dt * math.cos(cur_heading) + (torch.randn(1, generator=g).item() * 0.08)
                cur_y += speed * dt * math.sin(cur_heading) + (torch.randn(1, generator=g).item() * 0.08)
                future_traj.append([cur_x, cur_y])

            # Multi-agent tensor shape: [T, N, 2]
            past_tensor = torch.tensor(past_traj, dtype=torch.float32).unsqueeze(1).repeat(1, num_agents, 1)
            future_tensor = torch.tensor(future_traj, dtype=torch.float32).unsqueeze(1).repeat(1, num_agents, 1)


            # Add neighbor vehicle lane offsets
            for a in range(1, num_agents):
                offset_x = (a * 4.0) + torch.randn(1, generator=g).item()
                offset_y = (a * 8.0) + torch.randn(1, generator=g).item()
                past_tensor[:, a, 0] += offset_x
                past_tensor[:, a, 1] += offset_y
                future_tensor[:, a, 0] += offset_x
                future_tensor[:, a, 1] += offset_y

            self.past_data.append(past_tensor)
            self.future_data.append(future_tensor)

        # Pre-generate map lanes with the same capacity as the model input.
        lanes = []
        if self.max_lanes > 0:
            for l in range(self.max_lanes):
                offset_y = (l - (self.max_lanes // 2)) * 3.5
                xs = torch.linspace(-25.0, 65.0, self.lane_pts)
                ys = torch.ones(self.lane_pts) * offset_y + torch.sin(xs * 0.04) * 1.5
                lane = torch.stack([xs, ys], dim=-1)
                lanes.append(lane)
            self.default_map_lanes = torch.stack(lanes, dim=0)
        else:
            self.default_map_lanes = torch.zeros((0, self.lane_pts, 2), dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_past = self.past_data[idx]
        full_future = self.future_data[idx]
        obstacle = torch.zeros(self.max_obstacles, 4, dtype=torch.float32)
        obstacle_types = torch.zeros(self.max_obstacles, dtype=torch.long)
        obstacle_mask = torch.zeros(self.max_obstacles, dtype=torch.bool)
        if self.max_obstacles > 0:
            obstacle[0] = torch.tensor(
                [full_past[-1, 0, 0] + 8.0, full_past[-1, 0, 1] + 2.0, 0.4, 0.4],
                dtype=torch.float32,
            )
            obstacle_types[0] = 1
            obstacle_mask[0] = True
        return {
            "past": full_past[:, 0, :],
            "future": full_future[:, 0, :],
            "agents_past": full_past,
            "agent_mask": torch.ones(self.num_agents, dtype=torch.bool),
            "map_lanes": self.default_map_lanes,
            "lane_mask": torch.ones(self.max_lanes, dtype=torch.bool),
            "static_obstacles": obstacle,
            "static_obstacle_types": obstacle_types,
            "static_obstacle_mask": obstacle_mask,
            "sample_idx": torch.tensor(idx, dtype=torch.long)
        }


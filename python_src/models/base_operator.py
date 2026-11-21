r"""
Base Operator & FLOP Helper Utilities
======================================
Shared base classes and numerical safety helpers for PyTorch operators.
"""

import torch
from torch import nn


def trajectory_kinematics(trajectory: torch.Tensor, hz: float = 2.0) -> torch.Tensor:
    """Append finite-difference velocity, acceleration, and heading to 2D tracks.

    Accepts target tracks shaped [B, T, 2] and multi-agent tracks shaped
    [B, T, N, 2], returning eight features per point.
    """
    if hz <= 0:
        raise ValueError(f"Sampling frequency hz must be positive, got {hz}.")
    if trajectory.shape[-1] == 2:
        positions = trajectory
        velocity = torch.zeros_like(positions)
        velocity[:, 1:, ...] = (positions[:, 1:, ...] - positions[:, :-1, ...]) * hz
        if positions.shape[1] > 1:
            velocity[:, 0, ...] = velocity[:, 1, ...]
    elif trajectory.shape[-1] == 4:
        positions = trajectory[..., :2]
        velocity = trajectory[..., 2:4]
    else:
        raise ValueError(f"Expected 2D positions or 4D position/velocity features, got {trajectory.shape[-1]} channels")

    acceleration = torch.zeros_like(velocity)
    acceleration[:, 1:, ...] = (velocity[:, 1:, ...] - velocity[:, :-1, ...]) * hz
    if acceleration.shape[1] > 1:
        acceleration[:, 0, ...] = acceleration[:, 1, ...]

    direction_velocity = velocity.float() if velocity.dtype in (torch.float16, torch.bfloat16) else velocity
    speed = torch.linalg.vector_norm(direction_velocity, dim=-1, keepdim=True)
    direction = direction_velocity / speed.clamp_min(1e-8)
    moving = speed > 1e-8
    heading_sin = torch.where(moving, direction[..., 1:2], torch.zeros_like(speed))
    heading_cos = torch.where(moving, direction[..., 0:1], torch.ones_like(speed))
    return torch.cat((positions, velocity, acceleration, heading_sin, heading_cos), dim=-1)


def batch_scale(batch_size: int = 1) -> int:
    """Return a positive batch multiplier for per-sample FLOPs estimates."""
    b = int(batch_size)
    if b < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    return b


def normalize_future_tensor(future: torch.Tensor) -> torch.Tensor:
    """Normalize target future trajectories to canonical [B, T, 2] shape.

    Accepts [B, T, 2], or [B, T, 1, 2] (legacy single-target wrapper) and raises
    a descriptive ValueError on any unsupported dimensionality or coordinate size.
    """
    if not isinstance(future, torch.Tensor):
        raise TypeError(f"future must be a torch.Tensor, got {type(future).__name__}.")
    if future.ndim == 3:
        if future.shape[0] < 1 or future.shape[1] < 1 or future.shape[2] != 2:
            raise ValueError(
                f"Expected future tensor shape [B, T, 2] with B>=1, T>=1, got {tuple(future.shape)}."
            )
        return future
    if future.ndim == 4:
        if future.shape[2] != 1 or future.shape[0] < 1 or future.shape[1] < 1 or future.shape[3] != 2:
            raise ValueError(
                f"Unsupported 4D future tensor shape {tuple(future.shape)}; "
                "only [B, T, 1, 2] can be normalized to [B, T, 2]."
            )
        return future[:, :, 0, :]
    raise ValueError(
        f"Unsupported future tensor rank {future.ndim} with shape {tuple(future.shape)}; "
        "expected [B, T, 2] (or [B, T, 1, 2])."
    )


def compute_base_cost_breakdown(
    past_len: int,
    future_len: int,
    d_model: int,
    num_modes: int,
    max_agents: int,
    max_obstacles: int,
    max_lanes: int,
    lane_pts: int,
    map_encoder_flops: float,
    batch_size: int = 1,
    map_encoder_params: int = 0,
) -> dict[str, object]:
    """Centralized analytical cost breakdown for fixed SuperNet & FinalTrajectoryModel layers.

    Convention: 1 MAC = 2 FLOPs for linear, convolution, and recurrent matrix multiplies.
    Non-MAC operations (LayerNorm, ReLU, GRU elementwise gates, cumsum) are documented separately.
    """
    scale = batch_scale(batch_size)
    input_proj_flops = scale * (2 * past_len * 8 * d_model)
    neighbor_proj_flops = scale * (2 * past_len * max_agents * 8 * d_model)
    obstacle_proj_flops = scale * (2 * max_obstacles * 4 * d_model)
    scaled_map_enc_flops = scale * float(map_encoder_flops)
    fusion_flops = scale * (2 * past_len * 3 * d_model * d_model)
    decoder_flops = scale * (future_len * (12 * d_model * d_model + 18 * d_model))
    out_proj_flops = scale * (2 * future_len * d_model * (num_modes * 2))
    prob_head_flops = scale * (2 * d_model * d_model + 2 * d_model * num_modes)

    total_base_flops = (
        input_proj_flops
        + neighbor_proj_flops
        + obstacle_proj_flops
        + scaled_map_enc_flops
        + fusion_flops
        + decoder_flops
        + out_proj_flops
        + prob_head_flops
    )
    return {
        "batch_size": scale,
        "counting_convention": "1 MAC = 2 FLOPs",
        "theoretical_flops": total_base_flops,
        "theoretical_macs": total_base_flops / 2.0,
        "input_dimensions": {
            "past_len": past_len,
            "future_len": future_len,
            "d_model": d_model,
            "num_modes": num_modes,
            "max_agents": max_agents,
            "max_obstacles": max_obstacles,
            "max_lanes": max_lanes,
            "lane_pts": lane_pts,
        },
        "components": {
            "input_proj": {"flops": input_proj_flops, "macs": input_proj_flops / 2.0},
            "neighbor_proj": {"flops": neighbor_proj_flops, "macs": neighbor_proj_flops / 2.0},
            "obstacle_proj": {"flops": obstacle_proj_flops, "macs": obstacle_proj_flops / 2.0},
            "map_encoder": {
                "flops": scaled_map_enc_flops,
                "macs": scaled_map_enc_flops / 2.0,
                "trainable_params": map_encoder_params,
            },
            "fusion": {"flops": fusion_flops, "macs": fusion_flops / 2.0, "non_mac_ops": ["LayerNorm", "ReLU"]},
            "decoder": {
                "flops": decoder_flops,
                "macs": scale * (future_len * 6 * d_model * d_model),
                "non_mac_ops": ["GRU gate activations (sigmoid/tanh) & biases"],
            },
            "out_proj": {"flops": out_proj_flops, "macs": out_proj_flops / 2.0, "non_mac_ops": ["cumsum"]},
            "prob_head": {"flops": prob_head_flops, "macs": prob_head_flops / 2.0, "non_mac_ops": ["ReLU"]},
        },
    }


def compute_base_flops(
    past_len: int,
    future_len: int,
    d_model: int,
    num_modes: int,
    max_agents: int,
    max_obstacles: int,
    max_lanes: int,
    lane_pts: int,
    map_encoder_flops: float,
    batch_size: int = 1,
) -> float:
    """Return total analytical FLOPs for shared base architecture components."""
    return float(
        compute_base_cost_breakdown(
            past_len=past_len,
            future_len=future_len,
            d_model=d_model,
            num_modes=num_modes,
            max_agents=max_agents,
            max_obstacles=max_obstacles,
            max_lanes=max_lanes,
            lane_pts=lane_pts,
            map_encoder_flops=map_encoder_flops,
            batch_size=batch_size,
        )["theoretical_flops"]
    )


def resolve_architecture_indices(
    best_ops: tuple[int, int, int] | list[int] | None = None,
    checkpoint_path: str | None = None,
    search_config: object | None = None,
) -> tuple[tuple[int, int, int], str]:
    """Resolve discrete architecture indices from explicit values, search artifact, or deterministic fallback.

    Never returns out-of-bounds indices relative to SearchConfig operator lists.
    """
    import json
    import os
    from pathlib import Path
    from configs.search_config import SearchConfig

    s_cfg = search_config or SearchConfig()
    limits = (len(s_cfg.temporal_ops), len(s_cfg.interaction_ops), len(s_cfg.map_ops))

    def _is_valid(ops_tuple: tuple[int, int, int]) -> bool:
        return len(ops_tuple) == 3 and all(0 <= int(idx) < lim for idx, lim in zip(ops_tuple, limits))

    if best_ops is not None:
        candidate = (int(best_ops[0]), int(best_ops[1]), int(best_ops[2]))
        if not _is_valid(candidate):
            raise ValueError(
                f"Architecture indices {candidate} are out of range for SearchConfig limits {limits}."
            )
        return candidate, "explicit"

    pkg_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    candidates.extend([
        Path.cwd() / "checkpoints" / "best_architecture.json",
        pkg_root / "checkpoints" / "best_architecture.json",
        pkg_root.parent / "checkpoints" / "best_architecture.json",
    ])
    for cand in candidates:
        if cand.is_file():
            try:
                with cand.open("r", encoding="utf-8") as f:
                    doc = json.load(f)
                if isinstance(doc, dict):
                    raw = doc.get("best_ops")
                    if isinstance(raw, (list, tuple)) and len(raw) == 3:
                        parsed = (int(raw[0]), int(raw[1]), int(raw[2]))
                        if _is_valid(parsed):
                            return parsed, f"search_checkpoint:{cand}"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

    return (0, 0, 0), "deterministic_fallback"


def valid_non_ego_mask(agent_mask: torch.Tensor | None, num_agents: int, device: torch.device) -> torch.Tensor | None:
    """Build a [B, N] boolean validity mask for surrounding agents (excluding ego at index 0)."""
    if agent_mask is None:
        return None
    valid = agent_mask.to(device=device, dtype=torch.bool).clone()
    if valid.shape[-1] != num_agents:
        raise ValueError(f"agent_mask last dimension must be {num_agents}, got {valid.shape[-1]}")
    if num_agents > 0:
        valid[:, 0] = False
    return valid


def safe_cross_attention(
    mha_layer: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run cross-attention safely without NaNs when all keys in a batch row are masked."""
    batch_size, target_len, embed_dim = query.shape
    if key.shape[1] == 0 or value.shape[1] == 0:
        return query.new_zeros((batch_size, target_len, embed_dim))

    if key_padding_mask is None:
        out, _ = mha_layer(query, key, value, need_weights=False)
        return out

    if key_padding_mask.shape != key.shape[:2]:
        raise ValueError(
            f"key_padding_mask shape mismatch: expected {tuple(key.shape[:2])}, got {tuple(key_padding_mask.shape)}"
        )

    key_padding_mask = key_padding_mask.to(device=key.device, dtype=torch.bool)
    all_masked = key_padding_mask.all(dim=-1)
    if not all_masked.any():
        out, _ = mha_layer(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        return out

    sanitized_mask = key_padding_mask.clone()
    sanitized_mask[all_masked, 0] = False
    out, _ = mha_layer(query, key, value, key_padding_mask=sanitized_mask, need_weights=False)
    valid_rows = (~all_masked).to(dtype=out.dtype).view(batch_size, 1, 1)
    return out * valid_rows


class BaseOperator(nn.Module):
    """Base operator class with default FLOP calculation interface."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

    def get_cost_breakdown(self, **kwargs) -> dict[str, object]:
        """Return detailed analytical MACs/FLOPs/parameter/assumption breakdown."""
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": batch_scale(kwargs.get("batch_size", 1)),
            "theoretical_macs": 0.0,
            "theoretical_flops": 0.0,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {},
            "non_mac_ops": [],
        }

    def get_flops(self, **kwargs) -> float:
        """Estimate per-pass FLOPs for the operator."""
        return 0.0

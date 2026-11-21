"""
Trajectory Forecasting Comprehensive Metrics Suite
==================================================
Unified evaluation suite consolidating all dimensions of trajectory model evaluation:
  1. Geometric Displacement Accuracy Metrics (Regression): ADE, FDE, Step Errors
  2. Multi-Modal Forecasting Metrics: minADE@K, minFDE@K, Endpoint Miss Rate@K, Max-Displacement Miss Rate@K
  3. Physical Safety & Collision Risk Assessment: Collision Rate, Safety Margins
  4. Hardware Efficiency & Latency Metrics: HardwareProfiler, Expected FLOPs/MACs, CUDA Peak Memory, FPS
  5. Zero-Cost Architectural Saliency Metrics (ZCP): SNIP Score, SynFlow Score
"""

from __future__ import annotations

import inspect
import math
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# ==============================================================================
# Internal Validation, Precision & Distribution Helpers
# ==============================================================================

def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile matching NumPy's default behavior without NumPy dependency."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (q / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    frac = pos - lower
    return float(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)


def normalize_future_tensor(future: torch.Tensor) -> torch.Tensor:
    """Normalize target future tensor to [B, T, 2], allowing legacy [B, T, 1, 2] wrapper only."""
    if not isinstance(future, torch.Tensor):
        raise TypeError(f"target/future must be a torch.Tensor, got {type(future).__name__}.")
    if future.ndim == 3:
        if future.shape[0] < 1 or future.shape[1] < 1 or future.shape[2] != 2:
            raise ValueError(
                f"Expected target shape [B, T, 2] with B>=1, T>=1, got {tuple(future.shape)}."
            )
        return future
    if future.ndim == 4:
        if future.shape[2] != 1 or future.shape[0] < 1 or future.shape[1] < 1 or future.shape[3] != 2:
            raise ValueError(
                f"Unsupported 4D target shape {tuple(future.shape)}; "
                "only [B, T, 1, 2] can be normalized to [B, T, 2]."
            )
        return future[:, :, 0, :]
    raise ValueError(
        f"Unsupported target rank {future.ndim} with shape {tuple(future.shape)}; "
        "expected [B, T, 2] (or [B, T, 1, 2])."
    )


def _validate_shapes(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Validate prediction ([B, T, 2] or [B, K, T, 2]) against normalized target [B, T, 2]."""
    target_norm = normalize_future_tensor(target)
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred must be a torch.Tensor, got {type(pred).__name__}.")
    if pred.ndim not in (3, 4):
        raise ValueError(
            f"Expected 3D [B, T, 2] or 4D [B, K, T, 2] predictions, got shape {tuple(pred.shape)}."
        )
    if pred.shape[0] != target_norm.shape[0]:
        raise ValueError(
            f"Batch size mismatch: pred has {pred.shape[0]}, target has {target_norm.shape[0]}."
        )
    if pred.shape[-2] != target_norm.shape[1]:
        raise ValueError(
            f"Horizon length mismatch: pred has {pred.shape[-2]}, target has {target_norm.shape[1]}."
        )
    if pred.shape[-1] != 2:
        raise ValueError(f"Coordinate dimension must be 2, got {pred.shape[-1]}.")
    if pred.ndim == 4 and pred.shape[1] < 1:
        raise ValueError("Number of trajectory modes K must be positive.")
    return target_norm


def safe_norm(diff: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Compute Euclidean L2 norm with FP32 accumulation for FP16/BF16 stability."""
    values = diff.float() if diff.dtype in (torch.float16, torch.bfloat16) else diff
    return torch.linalg.vector_norm(values, dim=dim, keepdim=keepdim)


# ==============================================================================
# 1. Geometric Accuracy Metrics (Single-Trajectory & Multi-Modal WTA Loss)
# ==============================================================================

def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Average Displacement Error (ADE) strictly for single-mode 3D predictions [B, T, 2]."""
    target_norm = _validate_shapes(pred, target)
    if pred.ndim != 3:
        raise ValueError(
            f"ade() expects 3D single-trajectory tensor [B, T, 2], got {tuple(pred.shape)}. "
            "Use min_ade_k() for 4D multi-modal predictions [B, K, T, 2]."
        )
    dist = safe_norm(pred - target_norm, dim=-1)
    return torch.mean(dist)


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Final Displacement Error (FDE) strictly for single-mode 3D predictions [B, T, 2]."""
    target_norm = _validate_shapes(pred, target)
    if pred.ndim != 3:
        raise ValueError(
            f"fde() expects 3D single-trajectory tensor [B, T, 2], got {tuple(pred.shape)}. "
            "Use min_fde_k() for 4D multi-modal predictions [B, K, T, 2]."
        )
    dist = safe_norm(pred[:, -1, :] - target_norm[:, -1, :], dim=-1)
    return torch.mean(dist)


def best_of_k_loss(
    pred_trajs: torch.Tensor,
    target: torch.Tensor,
    mode_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Winner-Takes-All (WTA) multi-modal loss:
      1. Selects winning trajectory mode k* minimizing ADE + 0.5 * FDE to ground truth.
      2. Computes regression loss on the winning trajectory k*.
      3. Computes cross-entropy classification loss pulling mode_logits toward k*.
    """
    target_norm = _validate_shapes(pred_trajs, target)
    if pred_trajs.ndim == 3:
        loss_reg = ade(pred_trajs, target_norm) + 0.5 * fde(pred_trajs, target_norm)
        loss_cls = pred_trajs.new_zeros(())
        return loss_reg, loss_cls

    B, K, _, _ = pred_trajs.shape
    if mode_logits is not None:
        if mode_logits.shape != (B, K):
            raise ValueError(f"mode_logits must have shape ({B}, {K}), got {tuple(mode_logits.shape)}.")
        if not torch.isfinite(mode_logits).all():
            raise ValueError("mode_logits contains NaN or Inf values.")

    target_exp = target_norm.unsqueeze(1)  # [B, 1, T, 2]
    step_errors = safe_norm(pred_trajs - target_exp, dim=-1).mean(dim=-1)  # [B, K]
    end_errors = safe_norm(pred_trajs[:, :, -1, :] - target_exp[:, :, -1, :], dim=-1)  # [B, K]
    total_mode_errors = step_errors + 0.5 * end_errors  # [B, K]

    best_mode_idx = torch.argmin(total_mode_errors, dim=-1)  # [B]
    batch_idx = torch.arange(B, device=pred_trajs.device)
    winning_trajs = pred_trajs[batch_idx, best_mode_idx]  # [B, T, 2]

    loss_reg = ade(winning_trajs, target_norm) + 0.5 * fde(winning_trajs, target_norm)
    if mode_logits is not None:
        loss_cls = torch.nn.functional.cross_entropy(mode_logits, best_mode_idx)
    else:
        loss_cls = pred_trajs.new_zeros(())

    return loss_reg, loss_cls


# ==============================================================================
# 2. Multi-Modal Forecasting Metrics (minADE@K, minFDE@K, Miss Rate@K)
# ==============================================================================

def _select_top_k(
    multi_preds: torch.Tensor,
    k: int | None,
    mode_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select top-K modes by descending mode_logits; omitted logits imply already ranked inputs."""
    if multi_preds.ndim != 4 or multi_preds.shape[1] < 1:
        raise ValueError(f"Expected 4D [B, K>=1, T, 2] tensor, got {tuple(multi_preds.shape)}.")
    num_modes = multi_preds.shape[1]
    if k is None:
        k_eval = num_modes
    else:
        k_int = int(k)
        if k_int < 1:
            raise ValueError(f"k must be positive (>= 1), got {k}.")
        k_eval = min(k_int, num_modes)

    if mode_logits is None:
        return multi_preds[:, :k_eval]
    if mode_logits.shape != multi_preds.shape[:2]:
        raise ValueError(
            f"mode_logits must have shape {tuple(multi_preds.shape[:2])}, got {tuple(mode_logits.shape)}."
        )
    if not torch.isfinite(mode_logits).all():
        raise ValueError("mode_logits contains NaN or Inf values.")
    indices = mode_logits.topk(k_eval, dim=1).indices
    return multi_preds.gather(1, indices[:, :, None, None].expand(-1, -1, *multi_preds.shape[2:]))


_top_k_predictions = _select_top_k


def min_ade_k(
    multi_preds: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    mode_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute minADE@K across top-K predicted modes (uses all modes if k is None)."""
    target_norm = _validate_shapes(multi_preds, target)
    if multi_preds.ndim == 3:
        return ade(multi_preds, target_norm)

    preds_k = _select_top_k(multi_preds, k, mode_logits)
    step_dists = safe_norm(preds_k - target_norm.unsqueeze(1), dim=-1)  # [B, K_eff, T]
    mode_ades = torch.mean(step_dists, dim=-1)  # [B, K_eff]
    min_ades, _ = torch.min(mode_ades, dim=-1)  # [B]
    return torch.mean(min_ades)


def min_fde_k(
    multi_preds: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    mode_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute minFDE@K (final timestep L2 error) across top-K predicted modes."""
    target_norm = _validate_shapes(multi_preds, target)
    if multi_preds.ndim == 3:
        return fde(multi_preds, target_norm)

    end_preds = _select_top_k(multi_preds, k, mode_logits)[:, :, -1, :]  # [B, K_eff, 2]
    end_target = target_norm[:, -1, :].unsqueeze(1)  # [B, 1, 2]
    mode_fdes = safe_norm(end_preds - end_target, dim=-1)  # [B, K_eff]
    min_fdes, _ = torch.min(mode_fdes, dim=-1)  # [B]
    return torch.mean(min_fdes)


def endpoint_miss_rate_k(
    multi_preds: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    threshold: float = 2.0,
    mode_logits: torch.Tensor | None = None,
) -> float:
    """
    Standard Endpoint Miss Rate@K (nuScenes/Argoverse convention):
    Fraction of scenarios where every top-K mode has final-timestep error >= threshold (meters).
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    target_norm = _validate_shapes(multi_preds, target)
    if multi_preds.ndim == 3:
        end_err = safe_norm(multi_preds[:, -1, :] - target_norm[:, -1, :], dim=-1)
        return float((end_err >= threshold).float().mean().item())

    end_preds = _select_top_k(multi_preds, k, mode_logits)[:, :, -1, :]  # [B, K_eff, 2]
    end_target = target_norm[:, -1, :].unsqueeze(1)  # [B, 1, 2]
    min_end_err = safe_norm(end_preds - end_target, dim=-1).amin(dim=-1)  # [B]
    return float((min_end_err >= threshold).float().mean().item())


def max_displacement_miss_rate_k(
    multi_preds: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    threshold: float = 2.0,
    mode_logits: torch.Tensor | None = None,
) -> float:
    """
    Max-Over-Horizon Miss Rate@K:
    Fraction of scenarios where every top-K mode has maximum horizon error >= threshold (meters).
    Preserved separately from standard endpoint miss_rate_k for longitudinal comparison.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    target_norm = _validate_shapes(multi_preds, target)
    if multi_preds.ndim == 3:
        max_err = safe_norm(multi_preds - target_norm, dim=-1).amax(dim=-1)
        return float((max_err >= threshold).float().mean().item())

    preds_k = _select_top_k(multi_preds, k, mode_logits)
    mode_max_errors = safe_norm(preds_k - target_norm.unsqueeze(1), dim=-1).amax(dim=-1)
    return float((mode_max_errors.amin(dim=-1) >= threshold).float().mean().item())


def miss_rate_k(
    multi_preds: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    threshold: float = 2.0,
    mode_logits: torch.Tensor | None = None,
) -> float:
    """Alias for standard endpoint_miss_rate_k (final-timestep displacement >= threshold)."""
    return endpoint_miss_rate_k(
        multi_preds, target, k=k, threshold=threshold, mode_logits=mode_logits
    )


def miss_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    threshold: float = 2.0,
    mode_logits: torch.Tensor | None = None,
) -> float:
    """Compute endpoint Miss Rate (routes 3D and 4D consistently to endpoint_miss_rate_k)."""
    return endpoint_miss_rate_k(
        pred, target, k=k, threshold=threshold, mode_logits=mode_logits
    )


def ade_at_horizon(
    pred: torch.Tensor,
    target: torch.Tensor,
    steps: int = 10,
    k: int | None = None,
    mode_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ADE (or minADE@K for 4D predictions) truncated to the first `steps` timesteps."""
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    target_norm = _validate_shapes(pred, target)
    steps_eval = min(int(steps), target_norm.shape[1])
    if pred.ndim == 4:
        preds_k = _select_top_k(pred, k, mode_logits)[:, :, :steps_eval, :]
        step_dists = safe_norm(preds_k - target_norm[:, :steps_eval, :].unsqueeze(1), dim=-1)
        return torch.mean(torch.min(torch.mean(step_dists, dim=-1), dim=-1).values)
    dist = safe_norm(pred[:, :steps_eval, :] - target_norm[:, :steps_eval, :], dim=-1)
    return torch.mean(dist)


# ==============================================================================
# 3. Physical Safety & Collision Risk Metrics
# ==============================================================================

def evaluate_collision_risk(
    pred: torch.Tensor,
    neighbors_future: torch.Tensor | None = None,
    neighbor_future_mask: torch.Tensor | None = None,
    collision_distance_m: float = 1.8,
    mode_logits: torch.Tensor | None = None,
    mode_policy: str = "top1",
    neighbor_radii: torch.Tensor | None = None,
    vehicle_radius: float | None = None,
) -> dict[str, Any]:
    """
    Compute Collision Rate (CR) against surrounding dynamic agents' FUTURE trajectories.

    Requirements:
      - Requires genuine neighbors_future [B, Horizon, N, 2] matching pred's future horizon.
      - Historical neighbor trajectories (agents_past) must never be substituted here.
      - Returns available=False with an explicit dataset-limitation reason if missing.
    """
    if vehicle_radius is not None:
        collision_distance_m = float(vehicle_radius)
    if collision_distance_m <= 0:
        raise ValueError(f"collision_distance_m must be positive, got {collision_distance_m}.")

    if pred.ndim == 4:
        B, K, horizon, _ = pred.shape
        if mode_policy == "top1":
            if mode_logits is not None:
                if mode_logits.shape != (B, K):
                    raise ValueError(f"mode_logits must have shape ({B}, {K}), got {tuple(mode_logits.shape)}.")
                best_idx = mode_logits.argmax(dim=1)
                p_eval = pred[torch.arange(B, device=pred.device), best_idx]
            else:
                p_eval = pred[:, 0, :, :]
        elif mode_policy == "min_over_modes":
            p_eval = pred  # [B, K, T, 2]
        else:
            raise ValueError(f"Unsupported mode_policy '{mode_policy}'. Expected 'top1' or 'min_over_modes'.")
        batch_size = B
    elif pred.ndim == 3:
        batch_size, horizon, _ = pred.shape
        p_eval = pred
    else:
        raise ValueError(f"pred must be 3D or 4D tensor, got shape {tuple(pred.shape)}.")

    if (
        neighbors_future is None
        or not isinstance(neighbors_future, torch.Tensor)
        or neighbors_future.ndim != 4
        or neighbors_future.shape[1] != horizon
        or neighbors_future.shape[0] != batch_size
        or neighbors_future.shape[2] == 0
        or neighbors_future.shape[-1] < 2
    ):
        return {
            "available": False,
            "status": "Unavailable",
            "reason": (
                f"Future trajectories of neighbors (neighbors_future with horizon={horizon}) "
                "are not provided in the dataset sequences. Historical trajectories (agents_past) "
                "cannot be used for valid future collision assessment."
            ),
            "collision_rate_percent": None,
            "safety_compliance_percent": None,
            "scenarios_evaluated": 0,
            "collided_scenarios": 0,
        }

    num_neighbors = neighbors_future.shape[2]
    if neighbor_future_mask is None:
        valid_steps = torch.ones(batch_size, horizon, num_neighbors, dtype=torch.bool, device=pred.device)
    else:
        valid_steps = neighbor_future_mask.to(device=pred.device, dtype=torch.bool)
        if valid_steps.shape == (batch_size, num_neighbors):
            valid_steps = valid_steps.unsqueeze(1).expand(-1, horizon, -1)
        elif valid_steps.shape != (batch_size, horizon, num_neighbors):
            raise ValueError(
                f"neighbor_future_mask must have shape [{batch_size}, {num_neighbors}] or "
                f"[{batch_size}, {horizon}, {num_neighbors}], got {tuple(valid_steps.shape)}."
            )

    neighbors = neighbors_future[..., :2].to(device=pred.device)
    if neighbor_radii is not None:
        thresh = neighbor_radii.to(device=pred.device, dtype=neighbors.dtype)
        if thresh.ndim == 2:
            thresh = thresh.unsqueeze(1)  # [B, 1, N]
    else:
        thresh = collision_distance_m

    if p_eval.ndim == 3:
        distances = safe_norm(p_eval[:, :, None, :2] - neighbors, dim=-1)  # [B, T, N]
        collision_mask = ((distances < thresh) & valid_steps).flatten(1).any(dim=1)
    else:
        # min_over_modes: scenario collides if EVERY evaluated mode collides at some valid (t, n)
        distances = safe_norm(p_eval[:, :, :, None, :2] - neighbors.unsqueeze(1), dim=-1)  # [B, K, T, N]
        thresh_4d = thresh.unsqueeze(1) if isinstance(thresh, torch.Tensor) else thresh
        mode_collided = ((distances < thresh_4d) & valid_steps.unsqueeze(1)).flatten(2).any(dim=2)  # [B, K]
        collision_mask = mode_collided.all(dim=1)

    valid_scenarios_mask = valid_steps.flatten(1).any(dim=1)
    num_valid_scenarios = int(valid_scenarios_mask.sum().item())

    if num_valid_scenarios == 0:
        return {
            "available": False,
            "status": "No Valid Neighbors in Batch",
            "reason": "No active surrounding future neighbors found in this evaluation batch.",
            "collision_rate_percent": None,
            "safety_compliance_percent": None,
            "scenarios_evaluated": 0,
            "collided_scenarios": 0,
        }

    collided_count = int((collision_mask & valid_scenarios_mask).sum().item())
    collision_rate = (collided_count / num_valid_scenarios) * 100.0
    return {
        "available": True,
        "status": "Computed",
        "collision_rate_percent": round(collision_rate, 2),
        "safety_compliance_percent": round(100.0 - collision_rate, 2),
        "scenarios_evaluated": num_valid_scenarios,
        "collided_scenarios": collided_count,
        "collision_distance_m": collision_distance_m,
        "mode_policy": mode_policy if pred.ndim == 4 else "single_mode",
    }


# ==============================================================================
# 4. Hardware Efficiency & Latency Metrics
# ==============================================================================

def get_supernet_costs(
    alphas_dict: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    all_costs: dict[str, Any],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute expected FLOPs of the SuperNet under architectural distribution alpha.
    Differentiable hardware penalty used in DARTS:
      L_arch = L_reg + 0.1 * L_cls + lambda_flops * get_supernet_costs(alphas, all_costs)
    """
    a_temp, a_inter, a_map = alphas_dict

    flops_temp = sum(w * c for w, c in zip(a_temp, all_costs["temporal"]))
    flops_inter = sum(w * c for w, c in zip(a_inter, all_costs["interaction"]))
    flops_map = sum(w * c for w, c in zip(a_map, all_costs["map"]))
    total_flops = flops_temp + flops_inter + flops_map + all_costs.get("base", 0.0)

    if normalize:
        min_ops = min(all_costs["temporal"]) + min(all_costs["interaction"]) + min(all_costs["map"])
        max_ops = max(all_costs["temporal"]) + max(all_costs["interaction"]) + max(all_costs["map"])
        denom = max(1.0, float(max_ops - min_ops))
        return (flops_temp + flops_inter + flops_map - min_ops) / denom

    return total_flops / 1e9


def get_supernet_flops(
    alphas_dict: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    all_costs: dict[str, Any],
) -> torch.Tensor:
    """Alias for get_supernet_costs."""
    return get_supernet_costs(alphas_dict, all_costs)


def get_supernet_complexity(
    alphas_dict: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    all_costs: dict[str, Any],
) -> torch.Tensor:
    """Alias for get_supernet_costs."""
    return get_supernet_costs(alphas_dict, all_costs)


def simple_complexity_proxy(
    alphas_dict: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    all_costs: dict[str, Any],
) -> torch.Tensor:
    """Return expected GFLOPs using measured or analytical per-operator costs."""
    return get_supernet_costs(alphas_dict, all_costs)


class HardwareProfiler:
    """
    Profiles inference latency, throughput (FPS), CUDA peak memory, and model footprint.
    Preserves full model training/eval state and supports real-batch input factories.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        past_len: int | None = None,
        max_agents: int | None = None,
        max_lanes: int | None = None,
        lane_pts: int | None = None,
        max_obstacles: int | None = None,
        batch_size: int = 1,
        input_factory: Callable[[], dict[str, Any]] | None = None,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.past_len = past_len if past_len is not None else getattr(model, "past_len", 4)
        self.max_agents = max_agents if max_agents is not None else getattr(model, "max_agents", 16)
        self.max_lanes = max_lanes if max_lanes is not None else getattr(model, "max_lanes", 20)
        self.lane_pts = lane_pts if lane_pts is not None else getattr(model, "lane_pts", 10)
        self.max_obstacles = max_obstacles if max_obstacles is not None else getattr(model, "max_obstacles", 20)
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self.batch_size = int(batch_size)
        self.input_factory = input_factory

    def _build_synthetic_inputs(self, batch_size: int) -> dict[str, Any]:
        dummy_past = torch.randn(batch_size, self.past_len, 2, device=self.device)
        dummy_agents = torch.randn(batch_size, self.past_len, self.max_agents, 2, device=self.device)
        dummy_lanes = torch.randn(batch_size, self.max_lanes, self.lane_pts, 2, device=self.device)
        dummy_obstacles = torch.zeros(batch_size, self.max_obstacles, 4, device=self.device)
        dummy_obstacle_types = torch.zeros(batch_size, self.max_obstacles, dtype=torch.long, device=self.device)
        dummy_obstacle_mask = torch.zeros(batch_size, self.max_obstacles, dtype=torch.bool, device=self.device)
        if self.max_obstacles > 0:
            dummy_obstacles[:, 0, :] = torch.tensor([8.0, 2.0, 0.4, 0.4], device=self.device)
            dummy_obstacle_mask[:, 0] = True
        return {
            "past": dummy_past,
            "agents_past": dummy_agents,
            "map_lanes": dummy_lanes,
            "static_obstacles": dummy_obstacles,
            "static_obstacle_types": dummy_obstacle_types,
            "static_obstacle_mask": dummy_obstacle_mask,
        }

    def _forward_model(self, model_inputs: dict[str, Any]) -> Any:
        past = model_inputs["past"]
        kwargs = {
            k: v for k, v in model_inputs.items()
            if k in (
                "agents_past",
                "map_lanes",
                "category_id",
                "map_lane_types",
                "neighbor_categories",
                "agent_mask",
                "lane_mask",
                "static_obstacles",
                "static_obstacle_types",
                "static_obstacle_mask",
            ) and v is not None
        }
        return self.model(past, **kwargs)

    def profile(
        self,
        num_warmup: int = 25,
        num_runs: int = 100,
        input_factory: Callable[[], dict[str, Any]] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        if num_warmup < 0 or num_runs < 1:
            raise ValueError("num_warmup must be nonnegative and num_runs must be positive.")

        bs = int(batch_size) if batch_size is not None else self.batch_size
        if bs < 1:
            raise ValueError(f"batch_size must be >= 1, got {bs}.")

        active_factory = input_factory or self.input_factory
        if active_factory is not None:
            model_inputs = active_factory()
            input_source = "real_batch"
            if isinstance(model_inputs.get("past"), torch.Tensor):
                bs = int(model_inputs["past"].shape[0])
        else:
            model_inputs = self._build_synthetic_inputs(bs)
            input_source = "synthetic_dummy"

        was_training = self.model.training
        cuda_peak_alloc_mb: float | None = None
        cuda_peak_resv_mb: float | None = None
        try:
            self.model.eval()
            with torch.inference_mode():
                for _ in range(num_warmup):
                    _ = self._forward_model(model_inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)

                latencies: list[float] = []
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = self._forward_model(model_inputs)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    latencies.append((time.perf_counter() - t0) * 1000.0)

                if self.device.type == "cuda":
                    cuda_peak_alloc_mb = round(
                        torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0), 3
                    )
                    cuda_peak_resv_mb = round(
                        torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0), 3
                    )
        finally:
            self.model.train(was_training)

        lat_sorted = sorted(latencies)
        mean_lat = float(statistics.fmean(latencies))
        std_lat = float(statistics.pstdev(latencies)) if len(latencies) >= 2 else 0.0
        p95_lat = _percentile(lat_sorted, 95.0)
        throughput_fps = float((1000.0 * bs) / max(1e-4, mean_lat))

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary_fn = getattr(self.model, "get_summary", None)
        if callable(summary_fn):
            try:
                sig = inspect.signature(summary_fn)
                if "batch_size" in sig.parameters:
                    summary = summary_fn(batch_size=bs)
                else:
                    summary = summary_fn()
            except (TypeError, ValueError):
                summary = summary_fn()
        else:
            summary = {}

        flops_val_g = summary.get("theoretical_flops_g")
        flops_val_m = summary.get("theoretical_flops_m")
        if flops_val_g is None and flops_val_m is not None:
            flops_val_g = flops_val_m / 1000.0
        if flops_val_m is None and flops_val_g is not None:
            flops_val_m = round(flops_val_g * 1000.0, 3)
        macs_val_m = summary.get("theoretical_macs_m")
        if macs_val_m is None and flops_val_m is not None:
            macs_val_m = round(flops_val_m / 2.0, 3)

        return {
            "device": str(self.device),
            "batch_size": bs,
            "input_source": input_source,
            "measurement_scope": "forward_pass_only_excluding_dataloader",
            "latency_ms_mean": round(mean_lat, 2),
            "latency_ms_std": round(std_lat, 2),
            "latency_ms_p95": round(p95_lat, 2),
            "throughput_fps": round(throughput_fps, 1),
            "total_params": total_params,
            "total_params_m": round(total_params / 1e6, 3),
            "trainable_params": trainable_params,
            "theoretical_flops_g": flops_val_g,
            "theoretical_flops_m": flops_val_m,
            "theoretical_macs_m": macs_val_m,
            "base_flops_m": summary.get("base_flops_m"),
            "op_flops_m": summary.get("op_flops_m"),
            "cuda_peak_memory_allocated_mb": cuda_peak_alloc_mb,
            "cuda_peak_memory_reserved_mb": cuda_peak_resv_mb,
            "operators": {
                "temporal": summary.get("temporal_op", "N/A"),
                "interaction": summary.get("interaction_op", "N/A"),
                "map": summary.get("map_op", "N/A"),
            },
        }


# ==============================================================================
# 5. Zero-Cost Architecture Saliency Metrics (SNIP, SynFlow)
# ==============================================================================

def _is_supernet_model(net: nn.Module) -> bool:
    """Detect whether model expects architectural alpha weights in forward()."""
    if hasattr(net, "temporal_branch") and hasattr(net, "interaction_branch") and hasattr(net, "map_branch"):
        return True
    try:
        sig = inspect.signature(net.forward)
        return "alphas_tuple" in sig.parameters
    except (TypeError, ValueError):
        return False


def _resolve_uniform_alphas(
    net: nn.Module,
    device: str | torch.device,
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve uniform architectural weights dynamically from model branch op counts."""
    if alphas is not None:
        return tuple(a.to(device) for a in alphas)
    if hasattr(net, "temporal_branch") and hasattr(net, "interaction_branch") and hasattr(net, "map_branch"):
        n_t = len(net.temporal_branch.ops)
        n_i = len(net.interaction_branch.ops)
        n_m = len(net.map_branch.ops)
    else:
        from configs.search_config import SearchConfig
        s_cfg = SearchConfig()
        n_t, n_i, n_m = len(s_cfg.temporal_ops), len(s_cfg.interaction_ops), len(s_cfg.map_ops)
    return (
        torch.ones(n_t, device=device) / float(n_t),
        torch.ones(n_i, device=device) / float(n_i),
        torch.ones(n_m, device=device) / float(n_m),
    )


@contextmanager
def _saliency_context(net: nn.Module, linearize: bool = False):
    """Preserve buffers, gradients, module modes, and optional absolute-valued parameters."""
    module_modes = [(module, module.training) for module in net.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in net.buffers()]
    gradients = [
        (parameter, parameter.grad.detach().clone() if parameter.grad is not None else None)
        for parameter in net.parameters()
    ]
    parameters = [(parameter, parameter.detach().clone()) for parameter in net.parameters()] if linearize else []
    original_cudnn = torch.backends.cudnn.enabled
    try:
        net.train()
        net.zero_grad(set_to_none=True)
        torch.backends.cudnn.enabled = False
        if linearize:
            with torch.no_grad():
                for parameter, _ in parameters:
                    parameter.abs_()
        yield
    finally:
        with torch.no_grad():
            for parameter, original in parameters:
                parameter.copy_(original)
            for buffer, original in buffers:
                buffer.copy_(original)
        for parameter, gradient in gradients:
            parameter.grad = gradient
        for module, training in module_modes:
            module.training = training
        torch.backends.cudnn.enabled = original_cudnn


def snip_score(
    net: nn.Module,
    batch: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> float:
    """
    Single-shot Network Pruning (SNIP) Zero-Cost Proxy Metric:
      S = sum | (dL / d_theta) * theta |
    Supports both continuous SearchSuperNet (using dynamic branch operator counts)
    and discrete candidate models (e.g., FinalTrajectoryModel) without mutating state.
    """
    with _saliency_context(net):
        past = batch["past"].to(device)
        target = normalize_future_tensor(batch["future"].to(device))

        optional_keys = (
            "agents_past",
            "map_lanes",
            "category_id",
            "map_lane_types",
            "neighbor_categories",
            "agent_mask",
            "lane_mask",
            "static_obstacles",
            "static_obstacle_types",
            "static_obstacle_mask",
        )
        forward_kwargs = {
            k: batch[k].to(device)
            for k in optional_keys
            if batch.get(k) is not None and hasattr(batch.get(k), "to")
        }

        if _is_supernet_model(net):
            resolved_alphas = _resolve_uniform_alphas(net, device, alphas=alphas)
            output = net(past, resolved_alphas, **forward_kwargs)
        else:
            output = net(past, **forward_kwargs)

        if isinstance(output, tuple):
            pred_trajs, mode_logits = output[0], output[1]
            loss_reg, loss_cls = best_of_k_loss(pred_trajs, target, mode_logits)
            loss = loss_reg + 0.1 * loss_cls
        else:
            loss = ade(output, target)

        loss.backward()

        score = 0.0
        for p in net.parameters():
            if p.grad is not None:
                score += torch.sum(torch.abs(p.grad * p)).item()

        return float(score)


def synflow_score(
    net: nn.Module,
    device: str | torch.device = "cpu",
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> float:
    """
    Synaptic Saliency Flow (SynFlow) Zero-Cost Proxy Metric:
      Data-agnostic saliency using |W| linearization with guaranteed full parameter/state restoration.
    Supports both continuous SearchSuperNet and discrete candidate models.
    """
    with _saliency_context(net, linearize=True):
        past_len = getattr(net, "past_len", 4)
        max_agents = getattr(net, "max_agents", 16)
        max_lanes = getattr(net, "max_lanes", 20)
        lane_pts = getattr(net, "lane_pts", 10)
        max_obstacles = getattr(net, "max_obstacles", 0)

        dummy_input = torch.ones((1, past_len, 2), device=device, requires_grad=True)
        dummy_agents = torch.ones((1, past_len, max_agents, 2), device=device, requires_grad=True)
        dummy_lanes = torch.ones((1, max_lanes, lane_pts, 2), device=device, requires_grad=True)
        dummy_agent_mask = torch.ones(1, max_agents, dtype=torch.bool, device=device)
        dummy_lane_mask = torch.ones(1, max_lanes, dtype=torch.bool, device=device)

        forward_kwargs: dict[str, Any] = {
            "agents_past": dummy_agents,
            "map_lanes": dummy_lanes,
            "agent_mask": dummy_agent_mask,
            "lane_mask": dummy_lane_mask,
        }
        if max_obstacles and max_obstacles > 0:
            forward_kwargs["static_obstacles"] = torch.ones(
                (1, max_obstacles, 4), device=device, requires_grad=True
            )
            forward_kwargs["static_obstacle_mask"] = torch.ones(
                1, max_obstacles, dtype=torch.bool, device=device
            )

        if _is_supernet_model(net):
            resolved_alphas = _resolve_uniform_alphas(net, device, alphas=alphas)
            output = net(dummy_input, resolved_alphas, **forward_kwargs)
        else:
            output = net(dummy_input, **forward_kwargs)

        if isinstance(output, tuple):
            output = output[0]

        torch.sum(output).backward()

        score = 0.0
        for p in net.parameters():
            if p.grad is not None:
                score += torch.sum(torch.abs(p.grad * p)).item()

        return float(score)


__all__ = [
    # Geometric metrics
    "ade",
    "fde",
    "ade_at_horizon",
    "miss_rate",
    "best_of_k_loss",
    "min_ade_k",
    "min_fde_k",
    "miss_rate_k",
    "endpoint_miss_rate_k",
    "max_displacement_miss_rate_k",
    "normalize_future_tensor",
    "safe_norm",
    # Physical safety metrics
    "evaluate_collision_risk",
    # Hardware metrics
    "HardwareProfiler",
    "get_supernet_costs",
    "get_supernet_flops",
    "get_supernet_complexity",
    "simple_complexity_proxy",
    # Zero-cost proxy metrics
    "snip_score",
    "synflow_score",
]

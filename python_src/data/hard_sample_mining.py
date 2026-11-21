r"""
Hard Sample Mining (HSM) & Weighted Sampling Engine
===================================================
Author: Alireza Ghafouri (Master's Thesis Project - TrajectoryLNAS)

Forces model optimization to prioritize challenging trajectories (sharp turns,
high accelerations, dense agent interactions, cut-ins) over ubiquitous simple
straight cruising sequences.

Mathematical Formulation:
-------------------------
For N training samples, let e_i >= 0 be the running prediction error (e.g. minADE
or WTA regression error) for trajectory sample i.

The sensitivity parameter gamma in [0.0, 1.0] controls the mixture between uniform
sampling and error-proportional sampling:
    P(i) = (1 - gamma) * (1 / N) + gamma * (e_i / sum_j(e_j))
    Weight(i) = N * P(i) = (1 - gamma) + gamma * (N * e_i / (sum_j(e_j) + eps))

Properties:
    - gamma = 0.0: Strict uniform distribution (1/N everywhere, zero sensitivity, disabled).
    - gamma = 1.0: Purely error-weighted sampling.
    - 0 < gamma < 1: Smooth convex trade-off ensuring robust convergence and exploration.
"""

from __future__ import annotations
from typing import Any, Iterable
import math

try:
    import torch
    from torch.utils.data import Sampler
    HAS_TORCH = True
except ImportError:
    torch = None
    class Sampler:
        def __init__(self, data_source=None):
            pass
    HAS_TORCH = False


class HardSampleMiningManager:
    """
    Tracks and updates per-sample prediction errors across training epochs
    and computes dynamic sampling weights based on the sensitivity parameter.
    """

    def __init__(
        self,
        num_samples: int,
        sensitivity: float = 0.0,
        ema_momentum: float = 0.7,
        eps: float = 1e-7,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        self.num_samples = int(num_samples)
        self.sensitivity = max(0.0, min(1.0, float(sensitivity)))
        self.ema_momentum = max(0.0, min(1.0, float(ema_momentum)))
        self.eps = float(eps)

        # Track per-sample errors (in meters or WTA loss)
        if HAS_TORCH:
            self.errors = torch.ones(self.num_samples, dtype=torch.float32)
            self.weights = torch.ones(self.num_samples, dtype=torch.float64)
            self.counts = torch.zeros(self.num_samples, dtype=torch.int32)
        else:
            self.errors = [1.0] * self.num_samples
            self.weights = [1.0] * self.num_samples
            self.counts = [0] * self.num_samples

        self.visited_count = 0
        self.total_updates = 0
        self._recompute_weights()

    @property
    def is_active(self) -> bool:
        """Returns True if hard sample mining is actively weighting samples (gamma > 0)."""
        return self.sensitivity > 0.0

    def set_sensitivity(self, sensitivity: float):
        """Dynamically adjust sensitivity parameter gamma in [0.0, 1.0]."""
        self.sensitivity = max(0.0, min(1.0, float(sensitivity)))
        self._recompute_weights()

    def update_sample_errors(
        self,
        sample_indices: torch.Tensor | list[int] | Iterable[int],
        sample_errors: torch.Tensor | list[float] | Iterable[float]
    ):
        """
        Updates prediction errors for visited batch sample indices using Exponential Moving Average (EMA).
        """
        if HAS_TORCH and (isinstance(sample_indices, torch.Tensor) or isinstance(sample_errors, torch.Tensor)):
            if isinstance(sample_indices, torch.Tensor):
                indices = sample_indices.detach().cpu().long().view(-1).tolist()
            else:
                indices = [int(x) for x in sample_indices]

            if isinstance(sample_errors, torch.Tensor):
                errs = sample_errors.detach().cpu().float().view(-1).tolist()
            else:
                errs = [float(x) for x in sample_errors]
        else:
            indices = [int(x) for x in sample_indices]
            errs = [float(x) for x in sample_errors]

        if len(indices) != len(errs):
            raise ValueError(f"Indices ({len(indices)}) and errors ({len(errs)}) count mismatch.")

        # Filter valid indices within dataset range
        filtered_pairs = [(idx, max(0.0, err)) for idx, err in zip(indices, errs) if 0 <= idx < self.num_samples]
        if not filtered_pairs:
            return

        for idx, err in filtered_pairs:
            c = int(self.counts[idx]) if HAS_TORCH else self.counts[idx]
            if c == 0:
                self.errors[idx] = err
                self.visited_count += 1
            else:
                beta = self.ema_momentum
                old_val = float(self.errors[idx].item()) if HAS_TORCH else float(self.errors[idx])
                new_val = (1.0 - beta) * old_val + beta * err
                self.errors[idx] = new_val

            if HAS_TORCH:
                self.counts[idx] += 1
            else:
                self.counts[idx] += 1

        self.total_updates += len(filtered_pairs)
        self._recompute_weights()

    def _recompute_weights(self):
        """
        Recomputes sampling weights W_i based on current error distribution and sensitivity.
        When sensitivity == 0, weights are strictly uniform (all 1.0).
        """
        if self.sensitivity <= 0.0:
            if HAS_TORCH:
                self.weights.fill_(1.0)
            else:
                self.weights = [1.0] * self.num_samples
            return

        if HAS_TORCH:
            err_tensor = self.errors.clone()
            visited_mask = self.counts > 0

            if visited_mask.any():
                mean_visited = err_tensor[visited_mask].mean()
                err_tensor[~visited_mask] = mean_visited
            else:
                mean_visited = torch.tensor(1.0)

            err_sum = err_tensor.sum().item()
            if err_sum <= self.eps:
                self.weights.fill_(1.0)
                return

            gamma = self.sensitivity
            N = float(self.num_samples)
            normalized_errors = (N * err_tensor) / (err_sum + self.eps)
            computed_weights = (1.0 - gamma) + gamma * normalized_errors
            self.weights = computed_weights.to(dtype=torch.float64)
        else:
            err_list = list(self.errors)
            visited_errs = [err_list[i] for i in range(self.num_samples) if self.counts[i] > 0]
            mean_visited = (sum(visited_errs) / len(visited_errs)) if visited_errs else 1.0
            for i in range(self.num_samples):
                if self.counts[i] == 0:
                    err_list[i] = mean_visited

            err_sum = sum(err_list)
            if err_sum <= self.eps:
                self.weights = [1.0] * self.num_samples
                return

            gamma = self.sensitivity
            N = float(self.num_samples)
            self.weights = [(1.0 - gamma) + gamma * (N * e / (err_sum + self.eps)) for e in err_list]

    def get_weights(self):
        """Returns the current sampling weights."""
        return self.weights

    def get_sampler(
        self,
        num_samples: int | None = None,
        replacement: bool = True,
        generator: Any = None
    ) -> HardSampleWeightedSampler:
        """Factory method to instantiate a HardSampleWeightedSampler attached to this manager."""
        return HardSampleWeightedSampler(
            manager=self,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator
        )

    def get_summary(self) -> dict[str, Any]:
        """Provides analytical telemetry on hard sample mining distribution for logging and UI."""
        if HAS_TORCH:
            visited_mask = self.counts > 0
            num_visited = int(visited_mask.sum().item())

            if num_visited > 0:
                visited_errors = self.errors[visited_mask]
                min_e = float(visited_errors.min().item())
                max_e = float(visited_errors.max().item())
                mean_e = float(visited_errors.mean().item())

                k_10 = max(1, num_visited // 10)
                top_weights = torch.topk(self.weights[visited_mask], k=k_10).values.mean().item()
                bottom_weights = torch.topk(self.weights[visited_mask], k=k_10, largest=False).values.mean().item()
                ratio = top_weights / max(1e-6, bottom_weights)
            else:
                min_e, max_e, mean_e, ratio = 1.0, 1.0, 1.0, 1.0
        else:
            visited_indices = [i for i in range(self.num_samples) if self.counts[i] > 0]
            num_visited = len(visited_indices)
            if num_visited > 0:
                visited_errs = [self.errors[i] for i in visited_indices]
                min_e = min(visited_errs)
                max_e = max(visited_errs)
                mean_e = sum(visited_errs) / num_visited

                visited_w = sorted([self.weights[i] for i in visited_indices], reverse=True)
                k_10 = max(1, num_visited // 10)
                top_weights = sum(visited_w[:k_10]) / k_10
                bottom_weights = sum(visited_w[-k_10:]) / k_10
                ratio = top_weights / max(1e-6, bottom_weights)
            else:
                min_e, max_e, mean_e, ratio = 1.0, 1.0, 1.0, 1.0

        return {
            "active": self.is_active,
            "sensitivity": self.sensitivity,
            "ema_momentum": self.ema_momentum,
            "total_samples": self.num_samples,
            "samples_visited": num_visited,
            "coverage_percent": round((num_visited / max(1, self.num_samples)) * 100.0, 2),
            "min_error": round(min_e, 4),
            "max_error": round(max_e, 4),
            "mean_error": round(mean_e, 4),
            "hard_to_easy_weight_ratio": round(ratio, 2)
        }


class HardSampleWeightedSampler(Sampler):
    """
    PyTorch Sampler implementing error-weighted sampling across training epochs.
    When sensitivity == 0.0, behaves as standard random permutation.
    When sensitivity > 0.0, samples with higher past prediction errors have higher probability.
    """

    def __init__(
        self,
        manager: HardSampleMiningManager,
        num_samples: int | None = None,
        replacement: bool = True,
        generator: torch.Generator | None = None
    ):
        super().__init__(None)
        self.manager = manager
        self.num_samples = int(num_samples or manager.num_samples)
        self.replacement = replacement
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if not HAS_TORCH:
            return iter(range(self.num_samples))

        # If HSM is inactive (sensitivity = 0), yield a uniform random permutation without replacement
        if not self.manager.is_active:
            return iter(torch.randperm(self.num_samples, generator=self.generator).tolist())

        # If HSM is active, sample indices according to dynamic weights
        weights = self.manager.get_weights()
        sampled_indices = torch.multinomial(
            weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=self.generator
        )
        return iter(sampled_indices.tolist())

    def update_sample_errors(
        self,
        sample_indices: torch.Tensor | list[int] | Iterable[int],
        sample_errors: torch.Tensor | list[float] | Iterable[float]
    ):
        """Direct proxy to update errors in the underlying manager."""
        self.manager.update_sample_errors(sample_indices, sample_errors)

    def get_summary(self) -> dict[str, Any]:
        """Returns manager telemetry."""
        return self.manager.get_summary()

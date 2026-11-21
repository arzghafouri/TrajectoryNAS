r"""
Independent Zero-Cost Proxy (ZCP) Discrete Architecture Ranking & Screening
=========================================================================
Evaluates discrete candidate architectures (temporal_op, interaction_op, map_op)
from SearchConfig independently of continuous Bi-Level DARTS optimization.

Key Guarantees:
  1. Strict Separation: ZCP scores are never added to DARTS L_arch, L_reg, L_cls,
     weight gradients, or architectural alpha gradients.
  2. Sequential Evaluation (4 GB VRAM Safe): Each candidate is instantiated, scored,
     and freed sequentially with gc.collect() + torch.cuda.empty_cache().
  3. Controlled Initialization: Candidate scoring uses explicit deterministic seed
     control (and optionally supports weight inheritance comparison) and measures
     multi-seed initialization stability.
  4. Screening Disabled by Default: Candidate pruning remains inactive
     (screening_enabled=False) unless explicit empirical acceptance criteria pass.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


@dataclass
class ZCPAcceptanceCriteria:
    """Predefined statistical gates required before enabling ZCP screening."""
    min_spearman_vs_accuracy: float = 0.40
    min_kendall_vs_accuracy: float = 0.25
    min_top5_retention: float = 0.40
    min_top10_retention: float = 0.60
    min_multiseed_spearman: float = 0.70


@dataclass(frozen=True)
class CandidateSpec:
    """Discrete 3-branch operator specification derived from SearchConfig."""
    candidate_id: int
    temporal_idx: int
    interaction_idx: int
    map_idx: int
    temporal_op: str
    interaction_op: str
    map_op: str

    @property
    def indices(self) -> tuple[int, int, int]:
        return (self.temporal_idx, self.interaction_idx, self.map_idx)

    @property
    def label(self) -> str:
        return f"({self.temporal_idx},{self.interaction_idx},{self.map_idx}): {self.temporal_op} + {self.interaction_op} + {self.map_op}"


def enumerate_discrete_candidates(search_config: Any | None = None) -> list[CandidateSpec]:
    """Enumerate all discrete candidate combinations directly from SearchConfig operator lists."""
    from configs.search_config import SearchConfig
    s_cfg = search_config or SearchConfig()
    candidates: list[CandidateSpec] = []
    cid = 0
    for t_idx, t_name in enumerate(s_cfg.temporal_ops):
        for i_idx, i_name in enumerate(s_cfg.interaction_ops):
            for m_idx, m_name in enumerate(s_cfg.map_ops):
                candidates.append(
                    CandidateSpec(
                        candidate_id=cid,
                        temporal_idx=t_idx,
                        interaction_idx=i_idx,
                        map_idx=m_idx,
                        temporal_op=t_name,
                        interaction_op=i_name,
                        map_op=m_name,
                    )
                )
                cid += 1
    return candidates


# ==============================================================================
# Statistical Rank & Correlation Utilities (Pure Python, Tie-Aware)
# ==============================================================================

def average_ranks(values: list[float], descending: bool = True) -> list[float]:
    """Assign 1-based fractional average ranks (1.0 = best under ordering)."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        val = indexed[i][1]
        while j + 1 < n and indexed[j + 1][1] == val:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for pos in range(i, j + 1):
            ranks[indexed[pos][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return float(num / (dx * dy))


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman's rank correlation rho with average tie handling."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = average_ranks(xs, descending=False)
    ry = average_ranks(ys, descending=False)
    return _pearson(rx, ry)


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Compute Kendall's tau-b rank correlation with tie correction."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0.0 and dy == 0.0:
                ties_x += 1
                ties_y += 1
            elif dx == 0.0:
                ties_x += 1
            elif dy == 0.0:
                ties_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) / 2.0
    denom = math.sqrt(max(0.0, (total_pairs - ties_x) * (total_pairs - ties_y)))
    if denom == 0.0:
        return 0.0
    return float((concordant - discordant) / denom)


def minmax_normalize(values: list[float]) -> list[float]:
    """Normalize scores into [0.0, 1.0]; constant vectors map to 0.5."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def rank_normalize(values: list[float], descending: bool = True) -> list[float]:
    """Normalize ranks into [0.0, 1.0] where 1.0 corresponds to the top-ranked candidate."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    ranks = average_ranks(values, descending=descending)  # 1.0 = best
    return [(n - r) / (n - 1.0) for r in ranks]


def top_k_retention(
    zcp_ordered_ids: list[Any],
    true_ordered_ids: list[Any],
    top_k: int,
    zcp_budget: int | None = None,
) -> dict[str, Any]:
    """
    Measure Top-K retention: fraction of true Top-K architectures retained in the top-`zcp_budget`
    (defaults to `top_k`) ZCP-ranked candidates.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    budget = top_k if zcp_budget is None else int(zcp_budget)
    if budget < 1:
        raise ValueError(f"zcp_budget must be >= 1, got {budget}.")

    true_top = true_ordered_ids[: min(top_k, len(true_ordered_ids))]
    zcp_top = zcp_ordered_ids[: min(budget, len(zcp_ordered_ids))]
    retained = [cid for cid in true_top if cid in set(zcp_top)]
    rate = len(retained) / max(1, len(true_top))
    return {
        "top_k": int(top_k),
        "zcp_budget": int(budget),
        "retained_count": len(retained),
        "true_count": len(true_top),
        "retention_rate": round(rate, 4),
        "retained_ids": retained,
        "true_top_ids": true_top,
        "zcp_top_ids": zcp_top,
    }


# ==============================================================================
# Discrete Candidate ZCP Scoring & Ranking
# ==============================================================================

def _cleanup_device(model: Any = None, device: torch.device | str | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if HAS_TORCH and torch is not None and torch.cuda.is_available():
        if device is None or str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _copy_inherited_weights(discrete_model: Any, supernet: Any, spec: CandidateSpec) -> None:
    """Optionally initialize discrete candidate weights from a SearchSuperNet for comparison."""
    shared = (
        "input_proj",
        "neighbor_proj",
        "category_emb",
        "obstacle_proj",
        "obstacle_type_emb",
        "map_encoder",
        "fusion",
        "decoder",
        "out_proj",
        "prob_head",
    )
    for name in shared:
        getattr(discrete_model, name).load_state_dict(getattr(supernet, name).state_dict())
    discrete_model.temporal_op.load_state_dict(supernet.temporal_branch.ops[spec.temporal_idx].state_dict())
    discrete_model.interaction_op.load_state_dict(supernet.interaction_branch.ops[spec.interaction_idx].state_dict())
    discrete_model.map_op.load_state_dict(supernet.map_branch.ops[spec.map_idx].state_dict())


def score_discrete_candidates(
    train_batch: dict[str, Any],
    search_config: Any | None = None,
    model_config: Any | None = None,
    device: str | torch.device = "cpu",
    init_seed: int = 42,
    weight_mode: str = "fresh_seeded",
    supernet: Any | None = None,
    combined_method: str = "rank_mean",
    candidates: list[CandidateSpec] | None = None,
) -> dict[str, Any]:
    """
    Score all discrete candidate architectures independently via SNIP & SynFlow.

    Args:
        train_batch: Fixed training mini-batch (from train_loader, never val_loader).
        search_config: SearchConfig specifying temporal_ops, interaction_ops, map_ops.
        model_config: ModelConfig specifying d_model, past_len, future_len, num_modes, etc.
        device: Target compute device.
        init_seed: Deterministic seed reset before initializing each candidate so comparisons
            are not biased by initialization drift across the candidate loop.
        weight_mode: 'fresh_seeded' (independent seed-controlled init) or 'inherited_supernet'.
        supernet: Required only when weight_mode == 'inherited_supernet'.
        combined_method: 'rank_mean' (default) or 'minmax_mean'.
        candidates: Optional subset of CandidateSpec items (defaults to all discrete SearchConfig combos).
    """
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required to score discrete candidates with SNIP and SynFlow.")
    if weight_mode not in ("fresh_seeded", "inherited_supernet"):
        raise ValueError(f"Unsupported weight_mode '{weight_mode}'.")
    if weight_mode == "inherited_supernet" and supernet is None:
        raise ValueError("supernet must be provided when weight_mode='inherited_supernet'.")
    if combined_method not in ("rank_mean", "minmax_mean"):
        raise ValueError(f"Unsupported combined_method '{combined_method}'.")

    from configs.model_config import ModelConfig
    from configs.search_config import SearchConfig
    from evaluations.metrics import snip_score, synflow_score
    from models.final_model import FinalTrajectoryModel
    from utils.seed import set_seed

    s_cfg = search_config or SearchConfig()
    m_cfg = model_config or ModelConfig()
    target_device = torch.device(device)
    specs = candidates if candidates is not None else enumerate_discrete_candidates(s_cfg)

    raw_records: list[dict[str, Any]] = []
    t_all_start = time.perf_counter()

    for spec in specs:
        t0 = time.perf_counter()
        set_seed(int(init_seed))
        model = FinalTrajectoryModel(
            best_temporal_idx=spec.temporal_idx,
            best_interaction_idx=spec.interaction_idx,
            best_map_idx=spec.map_idx,
            d_model=m_cfg.d_model,
            past_len=m_cfg.past_len,
            future_len=m_cfg.future_len,
            num_modes=m_cfg.num_modes,
            max_agents=m_cfg.max_agents,
            max_obstacles=m_cfg.max_obstacles,
            max_lanes=m_cfg.max_lanes,
            lane_pts=m_cfg.lane_pts,
            hz=m_cfg.hz,
            search_config=s_cfg,
            model_config=m_cfg,
        ).to(target_device)

        if weight_mode == "inherited_supernet":
            _copy_inherited_weights(model, supernet, spec)

        summary = model.get_summary(batch_size=1)
        snip = snip_score(model, train_batch, device=target_device)
        synflow = synflow_score(model, device=target_device)
        elapsed = time.perf_counter() - t0

        raw_records.append({
            "candidate_id": spec.candidate_id,
            "indices": list(spec.indices),
            "temporal_idx": spec.temporal_idx,
            "interaction_idx": spec.interaction_idx,
            "map_idx": spec.map_idx,
            "temporal_op": spec.temporal_op,
            "interaction_op": spec.interaction_op,
            "map_op": spec.map_op,
            "snip": float(snip),
            "synflow": float(synflow),
            "theoretical_flops": summary["theoretical_flops"],
            "theoretical_flops_m": summary["theoretical_flops_m"],
            "theoretical_macs_m": summary["theoretical_macs_m"],
            "parameters": summary["parameters"],
            "eval_seconds": round(elapsed, 4),
        })

        _cleanup_device(model, target_device)

    total_zcp_seconds = time.perf_counter() - t_all_start

    snip_vals = [r["snip"] for r in raw_records]
    synflow_vals = [r["synflow"] for r in raw_records]

    snip_ranks = average_ranks(snip_vals, descending=True)
    synflow_ranks = average_ranks(synflow_vals, descending=True)
    snip_rank_norm = rank_normalize(snip_vals, descending=True)
    synflow_rank_norm = rank_normalize(synflow_vals, descending=True)
    snip_minmax = minmax_normalize(snip_vals)
    synflow_minmax = minmax_normalize(synflow_vals)

    for idx, rec in enumerate(raw_records):
        rec["snip_rank"] = snip_ranks[idx]
        rec["synflow_rank"] = synflow_ranks[idx]
        rec["snip_norm_rank"] = round(snip_rank_norm[idx], 6)
        rec["synflow_norm_rank"] = round(synflow_rank_norm[idx], 6)
        rec["snip_norm_minmax"] = round(snip_minmax[idx], 6)
        rec["synflow_norm_minmax"] = round(synflow_minmax[idx], 6)
        if combined_method == "rank_mean":
            rec["combined_score"] = round(0.5 * (snip_rank_norm[idx] + synflow_rank_norm[idx]), 6)
        else:
            rec["combined_score"] = round(0.5 * (snip_minmax[idx] + synflow_minmax[idx]), 6)
        rec["proxy_rank_disagreement"] = abs(snip_ranks[idx] - synflow_ranks[idx])

    combined_scores = [r["combined_score"] for r in raw_records]
    combined_ranks = average_ranks(combined_scores, descending=True)
    for idx, rec in enumerate(raw_records):
        rec["combined_rank"] = combined_ranks[idx]

    ranked = sorted(raw_records, key=lambda r: (-r["combined_score"], r["theoretical_flops"], r["candidate_id"]))

    proxy_spearman = spearman_correlation(snip_vals, synflow_vals)
    proxy_kendall = kendall_tau(snip_vals, synflow_vals)
    max_gap = max((r["proxy_rank_disagreement"] for r in raw_records), default=0.0)
    mean_gap = sum(r["proxy_rank_disagreement"] for r in raw_records) / max(1, len(raw_records))

    return {
        "search_space_size": len(specs),
        "init_seed": int(init_seed),
        "weight_mode": weight_mode,
        "combined_method": combined_method,
        "source_split": "train",
        "coupled_to_darts_loss": False,
        "zcp_compute_seconds": round(total_zcp_seconds, 4),
        "proxy_agreement": {
            "snip_vs_synflow_spearman": round(proxy_spearman, 4),
            "snip_vs_synflow_kendall": round(proxy_kendall, 4),
            "mean_rank_disagreement": round(mean_gap, 2),
            "max_rank_disagreement": round(max_gap, 2),
        },
        "candidates": ranked,
    }


# ==============================================================================
# Multi-Seed Stability & Empirical Short-Training Validation
# ==============================================================================

def evaluate_zcp_multi_seed_stability(
    train_batch: dict[str, Any],
    seeds: tuple[int, ...] = (42, 101, 2026),
    search_config: Any | None = None,
    model_config: Any | None = None,
    device: str | torch.device = "cpu",
    candidates: list[CandidateSpec] | None = None,
) -> dict[str, Any]:
    """Quantify ZCP ranking sensitivity to weight initialization across multiple random seeds."""
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for multi-seed stability analysis.")

    per_seed: dict[int, dict[int, dict[str, float]]] = {}
    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        res = score_discrete_candidates(
            train_batch=train_batch,
            search_config=search_config,
            model_config=model_config,
            device=device,
            init_seed=seed,
            candidates=candidates,
        )
        seed_runs.append(res)
        per_seed[seed] = {c["candidate_id"]: c for c in res["candidates"]}

    cand_ids = sorted(per_seed[seeds[0]].keys())
    pairwise_spearman: list[float] = []
    pairwise_kendall: list[float] = []
    pairwise_snip_spearman: list[float] = []
    pairwise_synflow_spearman: list[float] = []

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s_a, s_b = seeds[i], seeds[j]
            comb_a = [per_seed[s_a][cid]["combined_score"] for cid in cand_ids]
            comb_b = [per_seed[s_b][cid]["combined_score"] for cid in cand_ids]
            snip_a = [per_seed[s_a][cid]["snip"] for cid in cand_ids]
            snip_b = [per_seed[s_b][cid]["snip"] for cid in cand_ids]
            syn_a = [per_seed[s_a][cid]["synflow"] for cid in cand_ids]
            syn_b = [per_seed[s_b][cid]["synflow"] for cid in cand_ids]
            pairwise_spearman.append(spearman_correlation(comb_a, comb_b))
            pairwise_kendall.append(kendall_tau(comb_a, comb_b))
            pairwise_snip_spearman.append(spearman_correlation(snip_a, snip_b))
            pairwise_synflow_spearman.append(spearman_correlation(syn_a, syn_b))

    rank_stds: list[float] = []
    for cid in cand_ids:
        ranks = [per_seed[s][cid]["combined_rank"] for s in seeds]
        mean_r = sum(ranks) / len(ranks)
        var = sum((r - mean_r) ** 2 for r in ranks) / len(ranks)
        rank_stds.append(math.sqrt(var))

    return {
        "seeds": list(seeds),
        "num_candidates": len(cand_ids),
        "mean_pairwise_spearman": round(sum(pairwise_spearman) / len(pairwise_spearman), 4),
        "mean_pairwise_kendall": round(sum(pairwise_kendall) / len(pairwise_kendall), 4),
        "mean_snip_pairwise_spearman": round(sum(pairwise_snip_spearman) / len(pairwise_snip_spearman), 4),
        "mean_synflow_pairwise_spearman": round(sum(pairwise_synflow_spearman) / len(pairwise_synflow_spearman), 4),
        "mean_candidate_rank_std": round(sum(rank_stds) / max(1, len(rank_stds)), 3),
        "max_candidate_rank_std": round(max(rank_stds, default=0.0), 3),
    }


def evaluate_candidate_short_training(
    spec: CandidateSpec,
    train_loader: Any,
    val_loader: Any,
    model_config: Any,
    search_config: Any,
    epochs: int = 2,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | torch.device = "cpu",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Controlled short training + validation evaluation of a single discrete candidate."""
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required for candidate training validation.")

    from evaluations.metrics import (
        best_of_k_loss,
        endpoint_miss_rate_k,
        max_displacement_miss_rate_k,
        min_ade_k,
        min_fde_k,
    )
    from models.final_model import FinalTrajectoryModel
    from train.base_trainer import unpack_batch_to_device
    from utils.seed import set_seed

    target_device = torch.device(device)
    t_start = time.perf_counter()
    set_seed(int(seed))

    model = FinalTrajectoryModel(
        best_temporal_idx=spec.temporal_idx,
        best_interaction_idx=spec.interaction_idx,
        best_map_idx=spec.map_idx,
        d_model=model_config.d_model,
        past_len=model_config.past_len,
        future_len=model_config.future_len,
        num_modes=model_config.num_modes,
        max_agents=model_config.max_agents,
        max_obstacles=model_config.max_obstacles,
        max_lanes=model_config.max_lanes,
        lane_pts=model_config.lane_pts,
        hz=model_config.hz,
        search_config=search_config,
        model_config=model_config,
    ).to(target_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    last_train_loss = 0.0

    for _ in range(max(1, int(epochs))):
        model.train()
        total_loss = 0.0
        steps = 0
        for b_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and b_idx >= max_train_batches:
                break
            b = unpack_batch_to_device(batch, target_device)
            optimizer.zero_grad()
            pred_trajs, mode_logits = model(
                b["past"],
                agents_past=b.get("agents_past"),
                map_lanes=b.get("map_lanes"),
                category_id=b.get("category_id"),
                map_lane_types=b.get("map_lane_types"),
                neighbor_categories=b.get("neighbor_categories"),
                agent_mask=b.get("agent_mask"),
                lane_mask=b.get("lane_mask"),
                static_obstacles=b.get("static_obstacles"),
                static_obstacle_types=b.get("static_obstacle_types"),
                static_obstacle_mask=b.get("static_obstacle_mask"),
            )
            loss_reg, loss_cls = best_of_k_loss(pred_trajs, b["future"], mode_logits)
            loss = loss_reg + 0.1 * loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        last_train_loss = total_loss / max(1, steps)

    model.eval()
    k_eval = int(model.num_modes)
    sum_ade = 0.0
    sum_fde = 0.0
    sum_ep_mr = 0.0
    sum_md_mr = 0.0
    total_samples = 0

    with torch.inference_mode():
        for b_idx, batch in enumerate(val_loader):
            if max_val_batches is not None and b_idx >= max_val_batches:
                break
            b = unpack_batch_to_device(batch, target_device)
            pred_trajs, mode_logits = model(
                b["past"],
                agents_past=b.get("agents_past"),
                map_lanes=b.get("map_lanes"),
                category_id=b.get("category_id"),
                map_lane_types=b.get("map_lane_types"),
                neighbor_categories=b.get("neighbor_categories"),
                agent_mask=b.get("agent_mask"),
                lane_mask=b.get("lane_mask"),
                static_obstacles=b.get("static_obstacles"),
                static_obstacle_types=b.get("static_obstacle_types"),
                static_obstacle_mask=b.get("static_obstacle_mask"),
            )
            B = b["past"].shape[0]
            eff_k = min(k_eval, pred_trajs.shape[1])
            sum_ade += min_ade_k(pred_trajs, b["future"], k=eff_k, mode_logits=mode_logits).item() * B
            sum_fde += min_fde_k(pred_trajs, b["future"], k=eff_k, mode_logits=mode_logits).item() * B
            sum_ep_mr += endpoint_miss_rate_k(pred_trajs, b["future"], k=eff_k, threshold=2.0, mode_logits=mode_logits) * B
            sum_md_mr += max_displacement_miss_rate_k(pred_trajs, b["future"], k=eff_k, threshold=2.0, mode_logits=mode_logits) * B
            total_samples += B

    elapsed = time.perf_counter() - t_start
    _cleanup_device(model, target_device)

    n = max(1, total_samples)
    return {
        "candidate_id": spec.candidate_id,
        "indices": list(spec.indices),
        "k_modes": k_eval,
        "train_loss": round(last_train_loss, 4),
        "val_min_ade": round(sum_ade / n, 4),
        "val_min_fde": round(sum_fde / n, 4),
        "val_endpoint_miss_rate_pct": round((sum_ep_mr / n) * 100.0, 2),
        "val_max_disp_miss_rate_pct": round((sum_md_mr / n) * 100.0, 2),
        "train_eval_seconds": round(elapsed, 4),
    }


def compute_zcp_validation_correlations(
    zcp_results: dict[str, Any],
    validation_records: list[dict[str, Any]],
    criteria: ZCPAcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """
    Compare ZCP rankings against empirical short-training validation outcomes.

    Reports:
      - Spearman & Kendall correlation vs accuracy (-error, where +1.0 = perfect agreement)
        and vs raw error (where -1.0 = higher ZCP predicts lower error)
      - Top-5 and Top-10 retention rates
      - Direct candidate-by-candidate ZCP vs empirical ranking table
      - Compute overhead vs potential screening savings
      - Acceptance-gate verdict for enabling screening
    """
    crit = criteria or ZCPAcceptanceCriteria()
    val_by_id = {r["candidate_id"]: r for r in validation_records}
    matched = [c for c in zcp_results["candidates"] if c["candidate_id"] in val_by_id]
    if len(matched) < 2:
        raise ValueError("Need at least 2 matched candidates to compute ZCP validation correlations.")

    snip_scores = [c["snip"] for c in matched]
    synflow_scores = [c["synflow"] for c in matched]
    comb_scores = [c["combined_score"] for c in matched]

    val_ades = [val_by_id[c["candidate_id"]]["val_min_ade"] for c in matched]
    val_fdes = [val_by_id[c["candidate_id"]]["val_min_fde"] for c in matched]
    val_mrs = [val_by_id[c["candidate_id"]]["val_endpoint_miss_rate_pct"] for c in matched]

    neg_ades = [-v for v in val_ades]
    neg_fdes = [-v for v in val_fdes]
    neg_mrs = [-v for v in val_mrs]

    def _corr_bundle(proxy_vals: list[float]) -> dict[str, float]:
        return {
            "spearman_vs_ade_accuracy": round(spearman_correlation(proxy_vals, neg_ades), 4),
            "kendall_vs_ade_accuracy": round(kendall_tau(proxy_vals, neg_ades), 4),
            "spearman_vs_fde_accuracy": round(spearman_correlation(proxy_vals, neg_fdes), 4),
            "kendall_vs_fde_accuracy": round(kendall_tau(proxy_vals, neg_fdes), 4),
            "spearman_vs_mr_accuracy": round(spearman_correlation(proxy_vals, neg_mrs), 4),
            "spearman_vs_raw_ade": round(spearman_correlation(proxy_vals, val_ades), 4),
        }

    zcp_ordered = [c["candidate_id"] for c in sorted(matched, key=lambda c: (-c["combined_score"], c["candidate_id"]))]
    true_ordered = [
        cid for cid, _ in sorted(
            ((c["candidate_id"], val_by_id[c["candidate_id"]]["val_min_ade"]) for c in matched),
            key=lambda kv: (kv[1], kv[0]),
        )
    ]
    true_rank_map = {cid: idx + 1 for idx, cid in enumerate(true_ordered)}
    zcp_rank_map = {cid: idx + 1 for idx, cid in enumerate(zcp_ordered)}

    top5 = top_k_retention(zcp_ordered, true_ordered, top_k=min(5, len(matched)))
    top10 = top_k_retention(zcp_ordered, true_ordered, top_k=min(10, len(matched)))

    total_train_seconds = sum(val_by_id[c["candidate_id"]].get("train_eval_seconds", 0.0) for c in matched)
    zcp_seconds = float(zcp_results.get("zcp_compute_seconds", 0.0))
    half_budget = max(1, len(matched) // 2)
    screened_train_seconds = (total_train_seconds / len(matched)) * half_budget
    net_savings_seconds = total_train_seconds - (zcp_seconds + screened_train_seconds)

    comparison_table = []
    for cid in true_ordered:
        c = next(x for x in matched if x["candidate_id"] == cid)
        v = val_by_id[cid]
        comparison_table.append({
            "candidate_id": cid,
            "indices": c["indices"],
            "operators": f"{c['temporal_op']}+{c['interaction_op']}+{c['map_op']}",
            "zcp_rank": zcp_rank_map[cid],
            "val_ade_rank": true_rank_map[cid],
            "rank_delta": abs(zcp_rank_map[cid] - true_rank_map[cid]),
            "snip": round(c["snip"], 4),
            "synflow": round(c["synflow"], 4),
            "combined_score": c["combined_score"],
            "val_min_ade": v["val_min_ade"],
            "val_min_fde": v["val_min_fde"],
            "val_endpoint_miss_rate_pct": v["val_endpoint_miss_rate_pct"],
            "theoretical_flops_m": c["theoretical_flops_m"],
        })

    comb_corr = _corr_bundle(comb_scores)
    return {
        "num_candidates_compared": len(matched),
        "correlations": {
            "snip": _corr_bundle(snip_scores),
            "synflow": _corr_bundle(synflow_scores),
            "combined": comb_corr,
        },
        "retention": {
            "top_5": top5,
            "top_10": top10,
        },
        "compute_budget": {
            "zcp_overhead_seconds": round(zcp_seconds, 4),
            "full_short_train_seconds": round(total_train_seconds, 4),
            "projected_50pct_screened_seconds": round(zcp_seconds + screened_train_seconds, 4),
            "net_savings_at_50pct_keep_seconds": round(net_savings_seconds, 4),
            "zcp_fraction_of_full_train": round(zcp_seconds / max(1e-6, total_train_seconds), 4),
        },
        "comparison_table": comparison_table,
    }


def evaluate_screening_acceptance(
    correlation_report: dict[str, Any],
    stability_report: dict[str, Any] | None = None,
    criteria: ZCPAcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Decide whether empirical validation meets criteria to enable ZCP candidate screening."""
    crit = criteria or ZCPAcceptanceCriteria()
    comb = correlation_report["correlations"]["combined"]
    ret5 = correlation_report["retention"]["top_5"]["retention_rate"]
    ret10 = correlation_report["retention"]["top_10"]["retention_rate"]

    checks = {
        "spearman_vs_ade_accuracy": comb["spearman_vs_ade_accuracy"] >= crit.min_spearman_vs_accuracy,
        "kendall_vs_ade_accuracy": comb["kendall_vs_ade_accuracy"] >= crit.min_kendall_vs_accuracy,
        "top_5_retention": ret5 >= crit.min_top5_retention,
        "top_10_retention": ret10 >= crit.min_top10_retention,
    }
    if stability_report is not None:
        checks["multiseed_spearman"] = (
            stability_report["mean_pairwise_spearman"] >= crit.min_multiseed_spearman
        )

    passed = all(checks.values())
    return {
        "screening_recommended": passed,
        "criteria": asdict(crit),
        "checks": checks,
        "observed": {
            "spearman_vs_ade_accuracy": comb["spearman_vs_ade_accuracy"],
            "kendall_vs_ade_accuracy": comb["kendall_vs_ade_accuracy"],
            "top_5_retention": ret5,
            "top_10_retention": ret10,
            "multiseed_spearman": (
                stability_report["mean_pairwise_spearman"] if stability_report is not None else None
            ),
        },
    }


def screen_candidates(
    ranked_candidates: list[dict[str, Any]],
    top_k: int | None = None,
    keep_ratio: float = 0.5,
    screening_enabled: bool = False,
) -> dict[str, Any]:
    """
    Filter ZCP-ranked discrete candidates ONLY when screening_enabled=True.

    By default (screening_enabled=False), returns the full candidate list unpruned
    to prevent unvalidated zero-cost proxies from discarding optimal architectures.
    """
    total = len(ranked_candidates)
    if not screening_enabled:
        return {
            "screening_applied": False,
            "reason": "Screening disabled by default pending empirical correlation validation.",
            "total_candidates": total,
            "retained_count": total,
            "pruned_count": 0,
            "candidates": list(ranked_candidates),
        }

    if top_k is not None:
        keep_n = max(1, min(total, int(top_k)))
    else:
        if not (0.0 < float(keep_ratio) <= 1.0):
            raise ValueError(f"keep_ratio must be in (0.0, 1.0], got {keep_ratio}.")
        keep_n = max(1, min(total, int(math.ceil(total * float(keep_ratio)))))

    ordered = sorted(ranked_candidates, key=lambda c: (-c["combined_score"], c.get("theoretical_flops", 0.0)))
    kept = ordered[:keep_n]
    return {
        "screening_applied": True,
        "reason": f"Empirical screening active (retained top {keep_n}/{total} candidates).",
        "total_candidates": total,
        "retained_count": len(kept),
        "pruned_count": total - len(kept),
        "candidates": kept,
    }


__all__ = [
    "CandidateSpec",
    "ZCPAcceptanceCriteria",
    "average_ranks",
    "compute_zcp_validation_correlations",
    "enumerate_discrete_candidates",
    "evaluate_candidate_short_training",
    "evaluate_screening_acceptance",
    "evaluate_zcp_multi_seed_stability",
    "kendall_tau",
    "minmax_normalize",
    "rank_normalize",
    "score_discrete_candidates",
    "screen_candidates",
    "spearman_correlation",
    "top_k_retention",
]

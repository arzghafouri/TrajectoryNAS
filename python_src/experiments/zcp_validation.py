"""
Zero-Cost Proxy Empirical Validation & Ranking Experiment
=========================================================
Runs independent discrete-candidate SNIP/SynFlow ranking across all SearchConfig
candidates (4 x 4 x 3 = 48 by default), optionally runs short controlled validation
training, evaluates multi-seed initialization stability, and generates a structured
JSON/Markdown correlation & screening report without touching DARTS L_arch or saved
search checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.model_config import ModelConfig
from configs.search_config import SearchConfig
from core.zcp_ranking import (
    ZCPAcceptanceCriteria,
    compute_zcp_validation_correlations,
    enumerate_discrete_candidates,
    evaluate_candidate_short_training,
    evaluate_screening_acceptance,
    evaluate_zcp_multi_seed_stability,
    score_discrete_candidates,
    screen_candidates,
)


def run_zcp_empirical_experiment(
    dataset: str = "synthetic",
    train_path: str | None = None,
    val_path: str | None = None,
    batch_size: int = 16,
    short_epochs: int = 2,
    max_candidates: int | None = None,
    max_train_batches: int | None = 10,
    max_val_batches: int | None = 5,
    seeds: tuple[int, ...] = (42, 101, 2026),
    device: str = "cpu",
    output_path: str | None = None,
    model_config: ModelConfig | None = None,
    search_config: SearchConfig | None = None,
) -> dict[str, Any]:
    """Execute full ZCP ranking + controlled empirical short-training validation."""
    from utils.dataloaders import get_dataloaders

    m_cfg = model_config or ModelConfig()
    s_cfg = search_config or SearchConfig(dataset=dataset, train_path=train_path, val_path=val_path)

    train_loader, val_loader = get_dataloaders(
        train_path=train_path or s_cfg.train_path,
        val_path=val_path or s_cfg.val_path,
        batch_size=batch_size,
        dataset_type=dataset,
        model_config=m_cfg,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Train or Val DataLoader is empty; cannot run ZCP empirical experiment.")

    fixed_train_batch = next(iter(train_loader))
    all_specs = enumerate_discrete_candidates(s_cfg)
    specs = all_specs[:max_candidates] if max_candidates is not None else all_specs

    zcp_res = score_discrete_candidates(
        train_batch=fixed_train_batch,
        search_config=s_cfg,
        model_config=m_cfg,
        device=device,
        init_seed=seeds[0],
        candidates=specs,
    )
    stability_res = evaluate_zcp_multi_seed_stability(
        train_batch=fixed_train_batch,
        seeds=seeds,
        search_config=s_cfg,
        model_config=m_cfg,
        device=device,
        candidates=specs,
    )

    val_records = [
        evaluate_candidate_short_training(
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            model_config=m_cfg,
            search_config=s_cfg,
            epochs=short_epochs,
            seed=seeds[0],
            device=device,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
        for spec in specs
    ]

    corr_res = compute_zcp_validation_correlations(zcp_res, val_records)
    acceptance_res = evaluate_screening_acceptance(
        corr_res, stability_report=stability_res, criteria=ZCPAcceptanceCriteria()
    )
    screening_preview = screen_candidates(
        zcp_res["candidates"], screening_enabled=False
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "num_modes": m_cfg.num_modes,
        "search_space_total": len(all_specs),
        "candidates_evaluated": len(specs),
        "zcp_ranking": zcp_res,
        "multi_seed_stability": stability_res,
        "short_training_correlations": corr_res,
        "screening_acceptance_gate": acceptance_res,
        "default_screening_behavior": screening_preview,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[ZCP Validation] Saved empirical correlation report to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Independent ZCP Ranking & Validation")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "nuscenes"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--short_epochs", type=int, default=2)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="reports/zcp_validation_report.json")
    args = parser.parse_args()

    run_zcp_empirical_experiment(
        dataset=args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        short_epochs=args.short_epochs,
        max_candidates=args.max_candidates,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

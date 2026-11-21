"""
TrajectoryLNAS Main Orchestrator Pipeline
=========================================
Master Entry Point executing the complete 3-Stage Scientific Workflow:
  Stage 0: Automatic Dataset Verification & Preparation (data/prepare_data.py)
  Stage 1: Differentiable Neural Architecture Search (DARTS in train/search.py)
  Stage 2: Standalone Retraining of Discovered Model (train/train_final.py)
  Stage 3: Project Evaluation, Literature Protocol Review & Metric Plotting (evaluations/)

All execution logs, metrics summaries, and visualization plots are saved into reports/.

Author: Alireza Ghafouri (Student ID: 4031631209)
Advisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini
"""
import os
import sys
import json
import time
import argparse
# Ensure python_src directory is in sys.path for direct execution and IDEs
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from data.data_validation import run_acceptance_tests
except ImportError:
    def run_acceptance_tests(*args, **kwargs):
        print("[Notice] run_acceptance_tests requires PyTorch environment.")

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.search_config import SearchConfig
from configs.train_config import TrainConfig
try:
    from data.preprocessing import DataPreparationPipeline
except ImportError:
    DataPreparationPipeline = None

try:
    from core.progressive_search import run_architecture_search
except ImportError:
    run_architecture_search = None

try:
    from train.train_final import run_final_training
except ImportError:
    run_final_training = None

try:
    from evaluations.evaluator import TrajectoryEvaluator
except ImportError:
    TrajectoryEvaluator = None

try:
    from evaluations.visualization import plot_benchmark_comparison, plot_horizon_error_curve
except ImportError:
    plot_benchmark_comparison = lambda *a, **kw: None
    plot_horizon_error_curve = lambda *a, **kw: None

from typing import Any
from utils.logger import DualLogger

from configs.search_config import (
    DEFAULT_INTERACTION_OPS,
    DEFAULT_MAP_OPS,
    DEFAULT_TEMPORAL_OPS,
    INTERACTION_NAMES,
    MAP_NAMES,
    TEMPORAL_NAMES,
)


class CachedArchitectureSearch:
    """Mock/Proxy object mirroring DARTSAlgorithm results when loaded from JSON checkpoint."""
    def __init__(self, data: dict):
        self.cached = True
        self.elapsed_seconds = data.get("elapsed_seconds")
        proxies_data = data.get("zero_cost_proxies") or data.get("proxies") or {}
        self.proxies = {
            "snip": proxies_data.get("snip"),
            "synflow": proxies_data.get("synflow")
        }
        self.search_history = data.get("search_history", [])
        self.epochs = data.get("epochs")


def save_architecture_checkpoint(
    checkpoint_path: str,
    dataset: str,
    best_ops: tuple[int, int, int],
    searcher_obj: Any = None,
    search_cfg: SearchConfig | None = None
) -> str:
    """
    Saves discovered architecture operator combination and search metadata to JSON in checkpoints folder.
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    resolved_search_cfg = search_cfg if search_cfg is not None else SearchConfig()
    t_ops = resolved_search_cfg.temporal_ops
    i_ops = resolved_search_cfg.interaction_ops
    m_ops = resolved_search_cfg.map_ops

    t_idx, i_idx, m_idx = int(best_ops[0]), int(best_ops[1]), int(best_ops[2])
    t_name = t_ops[t_idx] if t_idx < len(t_ops) else f"Op_{t_idx}"
    i_name = i_ops[i_idx] if i_idx < len(i_ops) else f"Op_{i_idx}"
    m_name = m_ops[m_idx] if m_idx < len(m_ops) else f"Op_{m_idx}"

    search_dur = getattr(searcher_obj, "elapsed_seconds", None) if searcher_obj else None
    search_history = getattr(searcher_obj, "search_history", []) if searcher_obj else []
    proxies = getattr(searcher_obj, "proxies", {}) if searcher_obj else {}

    checkpoint_payload = {
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "best_ops": [t_idx, i_idx, m_idx],
        "best_indices": {
            "temporal": t_idx,
            "interaction": i_idx,
            "map": m_idx
        },
        "discovered_architecture": {
            "temporal": {"index": t_idx, "name": t_name},
            "interaction": {"index": i_idx, "name": i_name},
            "map": {"index": m_idx, "name": m_name},
            "indices": [t_idx, i_idx, m_idx]
        },
        "epochs": getattr(search_cfg, "epochs", None) if search_cfg else None,
        "elapsed_seconds": round(search_dur, 2) if search_dur is not None else None,
        "zero_cost_proxies": {
            "snip": round(proxies["snip"], 4) if proxies.get("snip") is not None else None,
            "synflow": round(proxies["synflow"], 4) if proxies.get("synflow") is not None else None
        },
        "search_history": search_history,
        "search_config": {
            "seed": resolved_search_cfg.seed,
            "warmup_epochs": resolved_search_cfg.warmup_epochs,
            "batch_size": resolved_search_cfg.batch_size,
            "grad_accum_steps": resolved_search_cfg.grad_accum_steps,
            "lambda_flops": resolved_search_cfg.lambda_flops,
            "search_samples_percent": resolved_search_cfg.search_samples_percent
        }
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, indent=2)

    return checkpoint_path


def load_architecture_checkpoint(checkpoint_path: str) -> tuple[tuple[int, int, int], CachedArchitectureSearch, dict]:
    """
    Loads saved architecture search results from a JSON checkpoint file.
    Returns (best_ops, searcher_proxy, raw_dict).
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "best_ops" in data and isinstance(data["best_ops"], (list, tuple)) and len(data["best_ops"]) == 3:
        best_ops = tuple(int(x) for x in data["best_ops"])
    elif "best_indices" in data and isinstance(data["best_indices"], dict):
        best_ops = (
            int(data["best_indices"]["temporal"]),
            int(data["best_indices"]["interaction"]),
            int(data["best_indices"]["map"])
        )
    elif "discovered_architecture" in data and isinstance(data["discovered_architecture"], dict):
        disc = data["discovered_architecture"]
        if "indices" in disc and isinstance(disc["indices"], (list, tuple)) and len(disc["indices"]) == 3:
            best_ops = tuple(int(x) for x in disc["indices"])
        else:
            best_ops = (
                int(disc["temporal"]["index"]),
                int(disc["interaction"]["index"]),
                int(disc["map"]["index"])
            )
    else:
        raise ValueError(f"No valid operator indices found in architecture checkpoint: {checkpoint_path}")

    searcher_obj = CachedArchitectureSearch(data)
    return best_ops, searcher_obj, data



def ensure_dataset_ready(
    dataset: str,
    train_path: str | None = None,
    count: int | None = None,
    data_cfg: DataConfig | None = None,
    dataroot: str | None = None,
    output_dir: str | None = None,
    mode: str | None = None
):
    """
    Checks if train and val dataset files exist.
    If not found, invokes DataPreparationPipeline which checks maps/expansion availability
    and extracts shuffled sequences (if count is None, extracts ALL available sequences).
    """
    if data_cfg is None:
        data_cfg = DataConfig()

    active_count = count if count is not None else data_cfg.target_count
    active_mode = mode if mode is not None else data_cfg.mode

    if dataset == "nuscenes":
        t_dir = os.path.normpath(train_path or output_dir or data_cfg.output_dir)
        has_train = os.path.exists(os.path.join(t_dir, "train_sequences.jsonl")) or os.path.exists(os.path.join(t_dir, "train_sequences.json"))
        has_val = os.path.exists(os.path.join(t_dir, "val_sequences.jsonl")) or os.path.exists(os.path.join(t_dir, "val_sequences.json"))

        if not (has_train and has_val):
            count_desc = f"{active_count:,} shuffled sequences" if active_count is not None else "ALL available sequences (No truncation)"
            print(f"[Stage 0: Data Verification] Dataset sequences missing in: {t_dir}")
            print(f"  -> Executing DataPreparationPipeline to prepare {count_desc}...")
            
            # Prioritize explicit dataroot, then DataConfig.dataroot
            raw_root = os.path.normpath(dataroot or data_cfg.dataroot)

            pipeline = DataPreparationPipeline(config=data_cfg)
            pipeline.prepare(
                dataroot=raw_root,
                output_dir=t_dir,
                target_count=active_count,
                mode=active_mode,
                seed=data_cfg.seed,
                category=data_cfg.category
            )
        else:
            print(f"[Stage 0: Data Acceptance Tests] Running for dataset in: {t_dir}")
            run_acceptance_tests()


def main_pipeline(
    search_cfg: SearchConfig = None,
    train_cfg: TrainConfig = None,
    data_cfg: DataConfig = None,
    dataset: str = None,
    dataroot: str = None,
    output_dir: str = None,
    mode: str = None,
    train_path: str = None,
    val_path: str = None,
    search_epochs: int = None,
    warmup_epochs: int = None,
    final_epochs: int = None,
    patience: int = None,
    min_delta: float = None,
    batch_size: int = None,
    search_batch_size: int = None,
    train_batch_size: int = None,
    grad_accum_steps: int = None,
    search_grad_accum: int = None,
    train_grad_accum: int = None,
    lambda_val: float = None,
    train_samples_percent: float = None,
    search_samples_percent: float = None,
    report_dir: str = "reports",
    output_checkpoint: str = None,
    arch_checkpoint: str = None,
    force_search: bool = False,
    skip_search: bool = False,
    temp_op: int | None = None,
    inter_op: int | None = None,
    map_op: int | None = None,
    hard_sample_mining: float = None
):
    """
    Executes the master pipeline, routing all logs and outputs to reports/
    and model weights to checkpoints/

    Follows clean Dependency Injection: SearchConfig, TrainConfig, and DataConfig
    serve as the single source of truth for all hyperparameters. Any parameter changed in
    configs/data_config.py or search_config.py is automatically respected across all pipeline stages.
    """
    # 1. Instantiate configuration objects (Single Source of Truth)
    if data_cfg is None:
        data_cfg = DataConfig()
    if dataroot is not None:
        data_cfg.dataroot = os.path.normpath(dataroot)
    if output_dir is not None:
        data_cfg.output_dir = os.path.normpath(output_dir)
    if mode is not None:
        data_cfg.mode = mode

    if search_cfg is None:
        search_cfg = SearchConfig(train_path=data_cfg.output_dir, val_path=data_cfg.output_dir)
    if train_cfg is None:
        train_cfg = TrainConfig(train_path=data_cfg.output_dir, val_path=data_cfg.output_dir)

    # 2. Apply explicit overrides only if provided (not None)
    if dataset is not None:
        search_cfg.dataset = dataset
        train_cfg.dataset = dataset
    if train_path is not None:
        search_cfg.train_path = os.path.normpath(train_path)
        train_cfg.train_path = os.path.normpath(train_path)
    elif search_cfg.train_path is None:
        search_cfg.train_path = data_cfg.output_dir
        train_cfg.train_path = data_cfg.output_dir

    if val_path is not None:
        search_cfg.val_path = os.path.normpath(val_path)
        train_cfg.val_path = os.path.normpath(val_path)
    elif search_cfg.val_path is None:
        search_cfg.val_path = data_cfg.output_dir
        train_cfg.val_path = data_cfg.output_dir

    if search_epochs is not None:
        search_cfg.epochs = search_epochs
    if warmup_epochs is not None:
        search_cfg.warmup_epochs = warmup_epochs
    if final_epochs is not None:
        train_cfg.epochs = final_epochs
    if patience is not None:
        train_cfg.early_stopping_patience = patience
    if min_delta is not None:
        train_cfg.early_stopping_min_delta = min_delta

    # Distinct Search vs Final Retrain Batch Sizes & Accumulation
    if search_batch_size is not None:
        search_cfg.batch_size = search_batch_size
    elif batch_size is not None:
        search_cfg.batch_size = batch_size

    if train_batch_size is not None:
        train_cfg.batch_size = train_batch_size
    elif batch_size is not None:
        train_cfg.batch_size = batch_size

    if search_grad_accum is not None:
        search_cfg.grad_accum_steps = search_grad_accum
    elif grad_accum_steps is not None:
        search_cfg.grad_accum_steps = grad_accum_steps

    if train_grad_accum is not None:
        train_cfg.grad_accum_steps = train_grad_accum
    elif grad_accum_steps is not None:
        train_cfg.grad_accum_steps = grad_accum_steps

    if lambda_val is not None:
        search_cfg.lambda_flops = lambda_val
    if train_samples_percent is not None:
        train_cfg.train_samples_percent = train_samples_percent
    if search_samples_percent is not None:
        search_cfg.search_samples_percent = search_samples_percent
    if hard_sample_mining is not None:
        train_cfg.hard_sample_mining = hard_sample_mining
    if output_checkpoint is not None:
        train_cfg.save_path = output_checkpoint

    active_dataset = search_cfg.dataset
    search_effective_batch = search_cfg.batch_size * search_cfg.grad_accum_steps
    train_effective_batch = train_cfg.batch_size * train_cfg.grad_accum_steps
    active_checkpoint = train_cfg.save_path
    active_batch_size = train_cfg.batch_size
    searcher_obj = None
    trainer_obj = None

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(active_checkpoint) if os.path.dirname(active_checkpoint) else "checkpoints", exist_ok=True)

    log_file = os.path.join(report_dir, "search_execution.log")
    export_json = os.path.join(report_dir, f"evaluation_results_{active_dataset}.json")

    from utils import set_seed
    set_seed(getattr(search_cfg, "seed", 42))

    logger = DualLogger(log_file)
    orig_stdout = sys.stdout
    sys.stdout = logger

    t_global_start = time.time()

    data_count_str = f"{data_cfg.target_count:,}" if data_cfg.target_count is not None else "ALL available sequences"

    train_pct_str = f"{train_cfg.train_samples_percent}%" if getattr(train_cfg, "train_samples_percent", None) is not None else "100% (Full Dataset)"
    search_pct_str = f"{search_cfg.search_samples_percent}%" if getattr(search_cfg, "search_samples_percent", None) is not None else "100% (Full Dataset)"
    print("=" * 80)
    print(" TRAJECTORY LNAS: MASTER ORCHESTRATION PIPELINE")
    print(f" Target Dataset:      {active_dataset.upper()}")
    print(f" DARTS Search Epochs: {search_cfg.epochs} (Warmup: {getattr(search_cfg, 'warmup_epochs', 5)} epochs) | Final Retrain Ceiling: {train_cfg.epochs}")
    print(f" Reproducibility:     Seed = {getattr(search_cfg, 'seed', 42)} (100% Deterministic Execution)")
    print(f" Subsampling:         Train = {train_pct_str} | DARTS = {search_pct_str}")
    print(f" Early Stopping:      Patience = {train_cfg.early_stopping_patience} epochs | Min Delta = {train_cfg.early_stopping_min_delta}m")
    print(f" Search Batch Size:   {search_cfg.batch_size} (Physical) | Accum Steps: {search_cfg.grad_accum_steps} (Effective: {search_effective_batch})")
    print(f" Train Batch Size:    {train_cfg.batch_size} (Physical) | Accum Steps: {train_cfg.grad_accum_steps} (Effective: {train_effective_batch})")
    print(f" Weight Decay (L2):   {search_cfg.weight_decay} (Network Weights Regularization)")
    print(f" Execution Log File:  {log_file}")
    print("=" * 80)

    try:
        # =========================================================================
        # STAGE 0: Automatic Dataset Verification & Preparation
        # =========================================================================
        ensure_dataset_ready(
            dataset=active_dataset,
            train_path=search_cfg.train_path,
            count=data_cfg.target_count,
            data_cfg=data_cfg,
            dataroot=dataroot,
            output_dir=output_dir,
            mode=mode
        )


        # =========================================================================
        # STAGE 1: Differentiable Neural Architecture Search (DARTS)
        # =========================================================================
        checkpoints_dir = os.path.dirname(active_checkpoint) if os.path.dirname(active_checkpoint) else "checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)

        target_arch_file = os.path.normpath(arch_checkpoint) if arch_checkpoint else os.path.join(checkpoints_dir, f"best_architecture.json")

        loaded_from_checkpoint = False
        selected_checkpoint_file = None

        if not force_search:
            # Check for existing architecture search checkpoint JSON in checkpoints directory
            if arch_checkpoint and os.path.isfile(arch_checkpoint):
                selected_checkpoint_file = arch_checkpoint
            elif os.path.isfile(target_arch_file):
                selected_checkpoint_file = target_arch_file

            # If user explicitly supplied custom operators via skip_search, respect them
            if skip_search and any(x is not None for x in (temp_op, inter_op, map_op)):
                selected_checkpoint_file = None

            if selected_checkpoint_file:
                try:
                    best_ops, searcher_obj, raw_arch_data = load_architecture_checkpoint(selected_checkpoint_file)
                    loaded_from_checkpoint = True

                    t_idx, i_idx, m_idx = best_ops
                    t_ops = search_cfg.temporal_ops if search_cfg else TEMPORAL_NAMES
                    i_ops = search_cfg.interaction_ops if search_cfg else INTERACTION_NAMES
                    m_ops = search_cfg.map_ops if search_cfg else MAP_NAMES
                    t_name = t_ops[t_idx] if t_idx < len(t_ops) else f"Op_{t_idx}"
                    i_name = i_ops[i_idx] if i_idx < len(i_ops) else f"Op_{i_idx}"
                    m_name = m_ops[m_idx] if m_idx < len(m_ops) else f"Op_{m_idx}"

                    print("=" * 80)
                    print(f" [STAGE 1/3] Architecture Search: Checkpoint Found in '{checkpoints_dir}/'")
                    print(f" Source File: {selected_checkpoint_file}")
                    if "timestamp" in raw_arch_data:
                        print(f" Discovered At: {raw_arch_data['timestamp']} | Dataset: {raw_arch_data.get('dataset', active_dataset).upper()}")
                    print(f" [*] Optimal Temporal Operator:    Index {t_idx} -> {t_name}")
                    print(f" [*] Optimal Interaction Operator: Index {i_idx} -> {i_name}")
                    print(f" [*] Optimal HD Map Operator:      Index {m_idx} -> {m_name}")
                    if searcher_obj.elapsed_seconds is not None:
                        print(f" [*] Original Search Duration:     {searcher_obj.elapsed_seconds:.1f}s")
                    print("  -> DARTS search bypassed; reusing best operator combination from previous search.")
                    print("  -> (To force a new search and overwrite checkpoint, pass --force_search)")
                    print("=" * 80)
                except Exception as e:
                    print(f"[STAGE 1/3 Warning] Could not parse architecture checkpoint '{selected_checkpoint_file}': {e}")
                    print("  -> Falling back to standard search execution...")
                    loaded_from_checkpoint = False

        if not loaded_from_checkpoint:
            if not skip_search:
                print("=" * 80)
                print("[STAGE 1/3] Launching Differentiable Neural Architecture Search (DARTS)...")
                print("=" * 80)
                best_ops, searcher_obj = run_architecture_search(
                    search_cfg=search_cfg,
                    log_file=None
                )

                # Save discovered architecture to checkpoints directory
                for sp in [target_arch_file]:
                    save_architecture_checkpoint(
                        checkpoint_path=sp,
                        dataset=active_dataset,
                        best_ops=best_ops,
                        searcher_obj=searcher_obj,
                        search_cfg=search_cfg
                    )
                    print(f"[STAGE 1/3 Checkpoint] Optimal architecture results saved to: {sp}")

                t_idx, i_idx, m_idx = best_ops
                t_ops = search_cfg.temporal_ops if search_cfg else TEMPORAL_NAMES
                i_ops = search_cfg.interaction_ops if search_cfg else INTERACTION_NAMES
                m_ops = search_cfg.map_ops if search_cfg else MAP_NAMES
                t_name = t_ops[t_idx] if t_idx < len(t_ops) else f"Op_{t_idx}"
                i_name = i_ops[i_idx] if i_idx < len(i_ops) else f"Op_{i_idx}"
                m_name = m_ops[m_idx] if m_idx < len(m_ops) else f"Op_{m_idx}"
                print("=" * 80)
                print(" [STAGE 1/3] Architecture Search Completed & Persisted Successfully!")
                print(f" Discovered Best Operators: {best_ops}")
                print(f"  -> Temporal:    Index {t_idx} -> {t_name}")
                print(f"  -> Interaction: Index {i_idx} -> {i_name}")
                print(f"  -> HD Map:      Index {m_idx} -> {m_name}")
                print(f" Saved in: '{checkpoints_dir}/' for immediate reuse in subsequent runs.")
                print("=" * 80)
            else:
                from models.base_operator import resolve_architecture_indices
                default_ops, ops_source = resolve_architecture_indices(
                    checkpoint_path=selected_checkpoint_file or target_arch_file,
                    search_config=search_cfg,
                )
                if any(x is not None for x in (temp_op, inter_op, map_op)):
                    explicit_ops = (
                        temp_op if temp_op is not None else default_ops[0],
                        inter_op if inter_op is not None else default_ops[1],
                        map_op if map_op is not None else default_ops[2],
                    )
                    best_ops, ops_source = resolve_architecture_indices(
                        best_ops=explicit_ops, search_config=search_cfg
                    )
                else:
                    best_ops = default_ops
                searcher_obj = None
                print(f"[STAGE 1/3 Skipped] Using resolved operator indices: {best_ops} (source: {ops_source})")

        # =========================================================================
        # STAGE 2: Standalone Retraining of Discovered Model (with Early Stopping)
        # =========================================================================
        print("[STAGE 2/3] Launching Standalone Training of Discovered Architecture (with Early Stopping)...")
        final_model, trainer_obj = run_final_training(
            best_op_indices=best_ops,
            train_cfg=train_cfg
        )

        # =========================================================================
        # STAGE 3: Project Evaluation, Literature Protocol Review & Plotting
        # =========================================================================
        print("[STAGE 3/3] Launching Academic Evaluation & Hardware Benchmarking...")

        evaluator = TrajectoryEvaluator(
            checkpoint_path=active_checkpoint,
            dataset_name=active_dataset,
            best_ops=best_ops,
            allow_untrained=True
        )
        eval_metrics = evaluator.evaluate_dataset(batch_size=active_batch_size)
        hw_metrics = evaluator.profile_hardware(num_runs=100)
        evaluator.print_evaluation_report(eval_metrics, hw_metrics)
        search_dur = getattr(searcher_obj, "elapsed_seconds", None) if searcher_obj else None

        # Export unified real_experiment_results.json FIRST so baselines & plots can ingest fresh records
        temporal_names = search_cfg.temporal_ops if search_cfg else TEMPORAL_NAMES
        inter_names = search_cfg.interaction_ops if search_cfg else INTERACTION_NAMES
        map_names = search_cfg.map_ops if search_cfg else MAP_NAMES

        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        model_cfg = getattr(trainer_obj, "model_cfg", ModelConfig())
        real_experiment_data = {
            "version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "thesis": {
                "project": "TrajectoryLNAS",
                "student": "Alireza Ghafouri (4031631209)",
                "supervisors": ["Dr. Zeinab Ghasemi Naraaghi", "Dr. Zeinab Abedini"]
            },
            "environment": {
                "device": device_name,
                "cuda_available": torch.cuda.is_available(),
                "dataset": active_dataset.upper()
            },
            "zero_cost_proxies": {
                "snip": round(searcher_obj.proxies["snip"], 4) if (searcher_obj and hasattr(searcher_obj, "proxies") and searcher_obj.proxies.get("snip") is not None) else None,
                "synflow": round(searcher_obj.proxies["synflow"], 4) if (searcher_obj and hasattr(searcher_obj, "proxies") and searcher_obj.proxies.get("synflow") is not None) else None
            },
            "darts_search": {
                "epochs": getattr(searcher_obj, "epochs", search_cfg.epochs) if (not skip_search or getattr(searcher_obj, "cached", False)) else 0,
                "elapsed_seconds": round(search_dur, 2) if search_dur is not None else None,
                "history": searcher_obj.search_history if searcher_obj and hasattr(searcher_obj, "search_history") else []
            },
            "discovered_architecture": {
                "temporal": {"index": best_ops[0], "name": temporal_names[best_ops[0]] if best_ops[0] < len(temporal_names) else f"Op_{best_ops[0]}"},
                "interaction": {"index": best_ops[1], "name": inter_names[best_ops[1]] if best_ops[1] < len(inter_names) else f"Op_{best_ops[1]}"},
                "map": {"index": best_ops[2], "name": map_names[best_ops[2]] if best_ops[2] < len(map_names) else f"Op_{best_ops[2]}"}
            },
            "final_training": {
                "history": trainer_obj.training_history if trainer_obj and hasattr(trainer_obj, "training_history") else [],
                "best_val_ade": round(trainer_obj.best_val_ade, 4) if trainer_obj else eval_metrics.get("min_ade_6s", None),
                "best_val_fde": round(trainer_obj.best_val_fde, 4) if trainer_obj else eval_metrics.get("min_fde_6s", None)
            },
            "edge_hardware_profiling": hw_metrics,
            "benchmark_assessment": {
                "direct_literature_ranking_supported": False,
                "note": "Published results use different datasets, populations, sampling rates, and/or metrics. See experiments/baselines.py.",
                "project_protocol": {
                    "dataset": active_dataset,
                    "observed_steps": getattr(final_model, "past_len", model_cfg.past_len),
                    "configured_sample_rate_hz": model_cfg.hz,
                    "observed_seconds_at_configured_rate": model_cfg.past_sec,
                    "future_steps": getattr(final_model, "future_len", model_cfg.future_len),
                    "prediction_horizon_seconds": model_cfg.future_sec,
                    "num_modes": getattr(final_model, "num_modes", model_cfg.num_modes),
                    "metrics": f"minADE@{model_cfg.future_sec:g}s / minFDE@{model_cfg.future_sec:g}s",
                    "samples_evaluated": eval_metrics.get("samples_evaluated"),
                },
                "validation_consistency": {
                    "training_best_ade": round(trainer_obj.best_val_ade, 4) if trainer_obj else None,
                    "training_best_fde": round(trainer_obj.best_val_fde, 4) if trainer_obj else None,
                    "full_evaluation_ade": eval_metrics.get("min_ade_6s"),
                    "full_evaluation_fde": eval_metrics.get("min_fde_6s"),
                    "consistent": (
                        abs(trainer_obj.best_val_ade - eval_metrics.get("min_ade_6s", trainer_obj.best_val_ade)) < 1e-3
                        and abs(trainer_obj.best_val_fde - eval_metrics.get("min_fde_6s", trainer_obj.best_val_fde)) < 1e-3
                    ) if trainer_obj else None,
                    "note": "Resolve any mismatch before using validation metrics in a paper comparison.",
                },
            },
            "evaluation_metrics": eval_metrics
        }

        # Save to report_dir and companion reports/ path for cross-directory stability
        target_report_dirs = [report_dir]
        companion_dir = os.path.join(current_dir, "reports")
        if companion_dir not in target_report_dirs:
            target_report_dirs.append(companion_dir)

        for r_dir in target_report_dirs:
            os.makedirs(r_dir, exist_ok=True)
            real_report_path = os.path.join(r_dir, "real_experiment_results.json")
            with open(real_report_path, "w", encoding="utf-8") as f:
                json.dump(real_experiment_data, f, indent=2)

        print(f"[Results Export] Real experiment data successfully saved to: {os.path.join(report_dir, 'real_experiment_results.json')}")

        # Also write backward-compatible evaluation_results_{dataset}.json
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(real_experiment_data, f, indent=2)

        # Print Literature Protocol Comparison & Audit Matrix
        evaluator.print_sota_comparison(eval_metrics=eval_metrics, hw_metrics=hw_metrics, search_runtime_seconds=search_dur)

        # Save publication plots into reports/
        plots_dir = report_dir
        step_errs = eval_metrics.get("step_errors", [])
        if step_errs:
            plot_horizon_error_curve(step_errors=step_errs, dataset=active_dataset, output_dir=plots_dir)
        our_ade = eval_metrics.get("min_ade_3s", eval_metrics.get("min_ade_6s"))
        our_fde = eval_metrics.get("min_fde_3s", eval_metrics.get("min_fde_6s"))
        if our_ade is not None and our_fde is not None:
            plot_benchmark_comparison(dataset=active_dataset, output_dir=plots_dir, our_ade=our_ade, our_fde=our_fde)

        t_total = time.time() - t_global_start
        h = int(t_total // 3600)
        m = int((t_total % 3600) // 60)
        s = int(t_total % 60)

        print("=" * 80)
        print(" Execution Finished Successfully!")
        print(f" Total Runtime: {h}h {m:02d}m {s:02d}s ({t_total:.1f} seconds)")
        print(f" Full execution trace permanently saved to: {log_file}")
        print(f" Reports & Plots stored in: {report_dir}/")
        print("=" * 80)


    finally:
        sys.stdout = orig_stdout
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Master 3-Stage Pipeline")
    parser.add_argument("--dataset", type=str, default=None, choices=["nuscenes", "synthetic"], help="Target dataset (defaults to configs/search_config.py)")
    parser.add_argument("--dataroot", type=str, default=None, help="Path to raw nuScenes dataset (defaults to DataConfig.dataroot or NUSCENES_DATAROOT env var)")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination directory for processed sequence JSON files (defaults to DataConfig.output_dir)")
    parser.add_argument("--mode", type=str, default=None, choices=["auto", "real", "synthetic"], help="Dataset preparation mode (defaults to DataConfig.mode)")
    parser.add_argument("--train_path", type=str, default=None, help="Train path override (defaults to DataConfig.output_dir)")
    parser.add_argument("--val_path", type=str, default=None, help="Val path override (defaults to DataConfig.output_dir)")
    parser.add_argument("--search_epochs", type=int, default=None, help="DARTS search epochs (defaults to SearchConfig.epochs)")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="DARTS warmup epochs where alpha is frozen (defaults to SearchConfig.warmup_epochs)")
    parser.add_argument("--final_epochs", type=int, default=None, help="Final training epochs (defaults to TrainConfig.epochs)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (defaults to TrainConfig.early_stopping_patience)")
    parser.add_argument("--min_delta", type=float, default=None, help="Minimum ADE improvement threshold (defaults to TrainConfig.early_stopping_min_delta)")
    parser.add_argument("--batch_size", type=int, default=None, help="Global batch size override (applies to both phases if specific not set)")
    parser.add_argument("--search_batch_size", type=int, default=None, help="Physical batch size for DARTS search phase (defaults to SearchConfig.batch_size)")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Physical batch size for final retraining phase (defaults to TrainConfig.batch_size)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Global grad accumulation override (applies to both phases if specific not set)")
    parser.add_argument("--search_grad_accum", type=int, default=None, help="Grad accumulation steps for DARTS search (defaults to SearchConfig.grad_accum_steps=4)")
    parser.add_argument("--train_grad_accum", type=int, default=None, help="Grad accumulation steps for final retraining (defaults to TrainConfig.grad_accum_steps=2)")
    parser.add_argument("--lambda_val", type=float, default=None, help="FLOPs penalty weight (defaults to configs/search_config.py)")
    parser.add_argument("--train_samples_percent", type=float, default=None, help="Percentage of dataset for final retraining (defaults to TrainConfig.train_samples_percent)")
    parser.add_argument("--search_samples_percent", type=float, default=None, help="Percentage of dataset for DARTS search (defaults to SearchConfig.search_samples_percent)")
    parser.add_argument("--report_dir", type=str, default="reports", help="Destination folder for logs, json, and plots")
    parser.add_argument("--checkpoint", type=str, default=None, help="Final model checkpoint save path (defaults to TrainConfig.save_path)")
    parser.add_argument("--arch_checkpoint", type=str, default=None, help="Path to architecture search checkpoint JSON (defaults to checkpoints/best_architecture.json)")
    parser.add_argument("--force_search", action="store_true", help="Force re-running DARTS search even if a search checkpoint exists in checkpoints/")
    parser.add_argument("--skip_search", action="store_true", help="Skip search and train default operators")
    parser.add_argument("--temp_op", type=int, default=None)
    parser.add_argument("--inter_op", type=int, default=None)
    parser.add_argument("--map_op", type=int, default=None)
    parser.add_argument("--hard_sample_mining", "--hsm", type=float, default=None, help="Sensitivity parameter [0.0, 1.0] for Weighted Sampling of hard cases")
    args = parser.parse_args()

    main_pipeline(
        dataset=args.dataset,
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        mode=args.mode,
        train_path=args.train_path,
        val_path=args.val_path,
        search_epochs=args.search_epochs,
        warmup_epochs=args.warmup_epochs,
        final_epochs=args.final_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        search_batch_size=args.search_batch_size,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum,
        search_grad_accum=args.search_grad_accum,
        train_grad_accum=args.train_grad_accum,
        lambda_val=args.lambda_val,
        train_samples_percent=args.train_samples_percent,
        search_samples_percent=args.search_samples_percent,
        report_dir=args.report_dir,
        output_checkpoint=args.checkpoint,
        arch_checkpoint=args.arch_checkpoint,
        force_search=args.force_search,
        skip_search=args.skip_search,
        temp_op=args.temp_op,
        inter_op=args.inter_op,
        map_op=args.map_op,
        hard_sample_mining=args.hard_sample_mining
    )


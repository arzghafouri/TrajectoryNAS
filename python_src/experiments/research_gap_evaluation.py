r"""
Independent Research Gaps Evaluation & Experiment Suite for TrajectoryLNAS
==========================================================================
Self-contained experimental runner investigating the three core research gaps
and optimization ablations for the TrajectoryLNAS research paper:

  1. Mandatory Data Integrity & Evaluation Reliability Audit (--experiment audit)
  2. Research Gap A: Factorized Three-Branch NAS Ablation A0-A7 (--experiment factorized)
  3. Research Gap B: Hardware-Aware NAS Profiling & FLOPs Penalty Trade-off (--experiment hardware)
  4. Research Gap C: Scene-Adaptive Architecture & Stratification (--experiment scene)
  5. Warmup & Temperature Annealing Optimization Ablation T0-T3 (--experiment warmup_temperature)

Design Principles:
  - Non-regression: Zero modifications to core NAS, model, dataset, or config modules.
  - Strict scientific honesty: Distinguishes implemented, executed, measured, pending,
    and unverified claims; never fabricates numbers or replaces missing metrics with 0.
  - Resource-aware: Supports 4GB VRAM GPUs (AMP, gradient accumulation, gradient
    checkpointing, explicit CUDA OOM handling, memory cleanup) and CPU-only hosts.
  - Cross-platform: Uses pathlib.Path, runs from repo root or python_src/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

# ------------------------------------------------------------------------------
# Path & Import Setup (Executable from project root or python_src/)
# ------------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PYTHON_SRC_DIR = _THIS_FILE.parent.parent
_REPO_ROOT_DIR = _PYTHON_SRC_DIR.parent

for _p in (str(_PYTHON_SRC_DIR), str(_REPO_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.search_config import SearchConfig
from configs.train_config import TrainConfig
from experiments.execution_manager import (
    ALL_STAGE_IDS,
    COMPARATIVE_STAGE_IDS,
    ConfigValidationError,
    ExecutionStatusStore,
    FACTORIZED_STAGE_IDS,
    STAGE_MAP,
    STAGE_SPECS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    WARMUP_TEMP_STAGE_IDS,
    atomic_write_json,
    extract_config_dict,
    format_execution_plan,
    format_final_summary,
    load_evaluate_config_module,
    validate_and_plan_pipeline,
)

# Optional scientific dependencies (guarded so static audit / --dry-run work anywhere)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

# (configs imported above)

if HAS_TORCH:
    from core.darts_algorithm import DARTSAlgorithm
    from evaluations.metrics import (
        best_of_k_loss,
        get_supernet_costs,
        min_ade_k,
        min_fde_k,
        miss_rate_k,
    )
    from models.final_model import FinalTrajectoryModel
    from train.base_trainer import unpack_batch_to_device
    from train.final_trainer import FinalModelTrainer
    from utils.dataloaders import get_dataloaders
    from utils.seed import set_seed
else:
    DARTSAlgorithm = None  # type: ignore[assignment,misc]
    FinalTrajectoryModel = None  # type: ignore[assignment,misc]
    FinalModelTrainer = None  # type: ignore[assignment,misc]
    get_dataloaders = None  # type: ignore[assignment]
    set_seed = None  # type: ignore[assignment]


# ------------------------------------------------------------------------------
# Constants & Specifications
# ------------------------------------------------------------------------------

FACTORIZED_ABLATION_MATRIX: list[dict[str, Any]] = [
    {"id": "A0", "temporal": "Fixed",  "interaction": "Fixed",  "map": "Fixed",  "description": "All three branches fixed to reference operators"},
    {"id": "A1", "temporal": "Search", "interaction": "Fixed",  "map": "Fixed",  "description": "Only Temporal branch searched; Interaction and Map fixed"},
    {"id": "A2", "temporal": "Fixed",  "interaction": "Search", "map": "Fixed",  "description": "Only Interaction branch searched; Temporal and Map fixed"},
    {"id": "A3", "temporal": "Fixed",  "interaction": "Fixed",  "map": "Search", "description": "Only HD Map branch searched; Temporal and Interaction fixed"},
    {"id": "A4", "temporal": "Search", "interaction": "Search", "map": "Fixed",  "description": "Temporal and Interaction searched; HD Map fixed"},
    {"id": "A5", "temporal": "Search", "interaction": "Fixed",  "map": "Search", "description": "Temporal and HD Map searched; Interaction fixed"},
    {"id": "A6", "temporal": "Fixed",  "interaction": "Search", "map": "Search", "description": "Interaction and HD Map searched; Temporal fixed"},
    {"id": "A7", "temporal": "Search", "interaction": "Search", "map": "Search", "description": "Full tri-branch factorized DARTS search"},
]

WARMUP_TEMP_MATRIX: list[dict[str, Any]] = [
    {"id": "T0", "warmup_enabled": False, "temp_annealing_enabled": False, "description": "No warmup, fixed temperature (T=T_start)"},
    {"id": "T1", "warmup_enabled": True,  "temp_annealing_enabled": False, "description": "Warmup enabled, fixed temperature (T=T_start)"},
    {"id": "T2", "warmup_enabled": False, "temp_annealing_enabled": True,  "description": "No warmup, temperature annealing enabled"},
    {"id": "T3", "warmup_enabled": True,  "temp_annealing_enabled": True,  "description": "Both warmup and temperature annealing enabled (default TrajectoryLNAS)"},
]

LITERATURE_MODELS_SPEC: list[dict[str, str]] = [
    {
        "model": "VectorNet (CVPR 2020)",
        "paradigm": "Manual hierarchical graph network (polyline subgraph + global interaction graph)",
        "temporal": "1D Polyline Node MLP + MaxPool",
        "interaction": "Global self-attention interaction graph over agent/lane nodes",
        "map": "Vectorized HD-map polyline subgraphs",
        "nas_strategy": "None (hand-crafted architecture)",
        "hardware_objective": "None",
        "scene_adaptive": "Static fixed architecture",
        "dataset_protocol": "Argoverse 1 (past 2s @10Hz, future 3s, K=1/K=6) & internal benchmarks; NOT directly comparable to nuScenes 6s K=5",
        "published_metrics": "Argoverse val: minADE@6=0.79m (paper Table 1/leaderboard variants); incompatible dataset/horizon/K vs nuScenes 6s",
    },
    {
        "model": "LaneGCN (ECCV 2020)",
        "paradigm": "Manual multi-scale 1D CNN + Lane Graph / Actor-Map fusion network",
        "temporal": "Multi-scale 1D ResNet + FPN (ActorNet)",
        "interaction": "Actor-to-Actor (A2A) spatial attention after map fusion",
        "map": "4-type directed lane graph convolution (LaneGCN) + A2L/L2A/L2L attention",
        "nas_strategy": "None (hand-crafted architecture)",
        "hardware_objective": "None",
        "scene_adaptive": "Static fixed architecture",
        "dataset_protocol": "Argoverse 1 (2s->3s @10Hz, K=6); incompatible dataset/horizon/K vs nuScenes 6s",
        "published_metrics": "Argoverse test: minADE@6=0.87m, minFDE@6=1.36m, MR@2m=16.3% (ECCV 2020)",
    },
    {
        "model": "HiVT (CVPR 2022)",
        "paradigm": "Manual translation/rotation-invariant hierarchical vectorized Transformer",
        "temporal": "Temporal Transformer encoder over local relative displacements",
        "interaction": "Local Transformer + Global inter-agent Transformer interaction",
        "map": "Local agent-lane cross-attention (vectorized lane segments)",
        "nas_strategy": "None (hand-crafted architecture)",
        "hardware_objective": "Explicit lightweight edge parameter budget (~0.66M-2.5M params) without NAS FLOPs penalty",
        "scene_adaptive": "Static fixed architecture (local-global hierarchy)",
        "dataset_protocol": "Argoverse 1 (2s->3s @10Hz, K=6); incompatible dataset/horizon/K vs nuScenes 6s",
        "published_metrics": "Argoverse test (HiVT-64/128): minADE@6=0.77-0.80m, minFDE@6=1.17-1.27m (CVPR 2022)",
    },
    {
        "model": "TrajectoryNAS (Sensors 2024)",
        "paradigm": "End-to-end 3D LiDAR detection, tracking & trajectory forecasting NAS",
        "temporal": "Searched spatiotemporal backbone blocks",
        "interaction": "Implicit BEV multi-object spatial features",
        "map": "Implicit BEV feature encoding (raw LiDAR + optional HD map raster/BEV)",
        "nas_strategy": "Hardware-constrained architecture search over perception+prediction blocks",
        "hardware_objective": "Latency/Compute constrained (22 ms reported on RTX A4000)",
        "scene_adaptive": "Static searched architecture",
        "dataset_protocol": "nuScenes Mini/Trainval end-to-end APf detection+forecasting (K=1/K=5); incompatible metric definition (mAPf vs minADE@K)",
        "published_metrics": "nuScenes K=5: vehicle mAPf=36.2, ped mAPf=32.5; RTX A4000 latency=22ms (Sharifi et al., Sensors 2024)",
    },
    {
        "model": "TrajGNAS (Literature GNN-NAS)",
        "paradigm": "Differentiable/evolutionary NAS over spatiotemporal GNN trajectory encoders",
        "temporal": "Searched temporal / recurrent GNN message-passing blocks",
        "interaction": "Searched graph topology / spatial aggregation edges",
        "map": "Scene/map context graph nodes when applicable (benchmark dependent)",
        "nas_strategy": "NAS over graph neural architecture space",
        "hardware_objective": "Parameter/FLOPs awareness depending on target deployment configuration",
        "scene_adaptive": "Static searched graph architecture (or topology weights per sample)",
        "dataset_protocol": "Traffic/urban trajectory benchmarks; exact split, horizon, K, and codebase not present in this workspace",
        "published_metrics": "Published exact protocol/source not present in repository; numerical values omitted to avoid unverified comparison",
    },
    {
        "model": "TrajectoryLNAS (This Repo)",
        "paradigm": "Factorized 3-branch differentiable DARTS SuperNet -> discrete standalone FinalTrajectoryModel",
        "temporal": "Searched from {GRUEncoder, LSTMEncoder, TemporalConvEncoder, TemporalSelfAttention}",
        "interaction": "Searched from {GCNLayer, GATLayer, SocialPooling, PairwiseAttention} (+ static obstacles)",
        "map": "Searched from {LaneAttention, LaneGraphEncoder, MapFeatureFusion} over PolylineEncoder",
        "nas_strategy": "Bi-level first-order DARTS with architecture warmup + exponential temperature annealing",
        "hardware_objective": "Expected SuperNet GFLOPs penalty (lambda_flops * E[GFLOPs]) in validation alpha loss",
        "scene_adaptive": "Static globally discovered architecture (Gap C investigates scene-conditioned prototype)",
        "dataset_protocol": "nuScenes vectorized sequences (past 2.0s [4 steps] @2Hz, future 6.0s [12 steps] @2Hz, K=5 modes from ModelConfig)",
        "published_metrics": "Populated below strictly from newly measured runs in this session (never extrapolated)",
    },
]


# ------------------------------------------------------------------------------
# Generic Utilities (JSON, CSV, Math, System Specs, Cleanup)
# ------------------------------------------------------------------------------

def _now_timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _csv_cell(value: Any) -> Any:
    """Format cell for CSV: keep NaN for missing floats/None so downstream parsers don't see fake 0s."""
    if value is None:
        return "NaN"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NaN"
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _stats_summary(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    s = sorted(clean)
    return {
        "count": len(s),
        "mean": round(sum(s) / len(s), 6),
        "median": round(statistics.median(s), 6),
        "std": round(statistics.stdev(s), 6) if len(s) > 1 else 0.0,
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
        "p50": round(_quantile(s, 0.50), 6),
        "p95": round(_quantile(s, 0.95), 6),
        "p99": round(_quantile(s, 0.99), 6),
    }


def _shannon_entropy(probs: list[float]) -> float:
    eps = 1e-12
    return float(-sum(max(eps, p) * math.log(max(eps, p)) for p in probs if p > 0))


def _resolve_device(requested_device: str) -> Any:
    if not HAS_TORCH:
        return "unavailable"
    req = (requested_device or "auto").strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available on this host; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(req)


def _cleanup_torch_resources(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if HAS_TORCH and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cuda error: out of memory" in msg)


def collect_hardware_environment(device: Any = None) -> dict[str, Any]:
    cuda_avail = bool(HAS_TORCH and torch.cuda.is_available())
    gpu_name = None
    gpu_total_vram_mb = None
    if cuda_avail:
        try:
            gpu_idx = device.index if (hasattr(device, "index") and device.index is not None) else 0
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            props = torch.cuda.get_device_properties(gpu_idx)
            gpu_total_vram_mb = round(props.total_memory / (1024 ** 2), 2)
        except Exception:
            pass
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_count": os.cpu_count(),
        "has_torch": HAS_TORCH,
        "has_numpy": HAS_NUMPY,
        "has_matplotlib": HAS_MATPLOTLIB,
        "torch_version": torch.__version__ if HAS_TORCH else None,
        "cuda_available": cuda_avail,
        "cuda_version": getattr(torch.version, "cuda", None) if HAS_TORCH else None,
        "gpu_model": gpu_name,
        "gpu_total_vram_mb": gpu_total_vram_mb,
        "selected_device": str(device) if device is not None else ("cuda" if cuda_avail else "cpu"),
    }


def load_reference_architecture() -> dict[str, Any]:
    """Loads reference discovered architecture from checkpoints/best_architecture.json cleanly."""
    s_cfg = SearchConfig()
    ckpt_candidates = [
        _PYTHON_SRC_DIR / "checkpoints" / "best_architecture.json",
        _REPO_ROOT_DIR / "checkpoints" / "best_architecture.json",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    default_indices = [3, 2, 0]
    source = "fallback_default_[3,2,0]"
    raw_meta: dict[str, Any] = {}

    if ckpt_path is not None:
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            if isinstance(raw_meta.get("best_ops"), list) and len(raw_meta["best_ops"]) == 3:
                default_indices = [int(x) for x in raw_meta["best_ops"]]
                source = str(ckpt_path)
        except Exception as e:
            source = f"fallback_default_[3,2,0] (failed reading {ckpt_path}: {e})"

    t_idx, i_idx, m_idx = default_indices
    t_name = s_cfg.temporal_ops[t_idx] if 0 <= t_idx < len(s_cfg.temporal_ops) else f"Op_{t_idx}"
    i_name = s_cfg.interaction_ops[i_idx] if 0 <= i_idx < len(s_cfg.interaction_ops) else f"Op_{i_idx}"
    m_name = s_cfg.map_ops[m_idx] if 0 <= m_idx < len(s_cfg.map_ops) else f"Op_{m_idx}"

    return {
        "indices": [t_idx, i_idx, m_idx],
        "names": [t_name, i_name, m_name],
        "temporal": {"index": t_idx, "name": t_name},
        "interaction": {"index": i_idx, "name": i_name},
        "map": {"index": m_idx, "name": m_name},
        "source": source,
        "checkpoint_metadata": raw_meta.get("search_config", {}),
    }


def resolve_checkpoint_path(ckpt_arg: str | None) -> Path | None:
    if not ckpt_arg:
        return None
    p = Path(ckpt_arg)
    candidates = [
        p,
        _PYTHON_SRC_DIR / ckpt_arg,
        _REPO_ROOT_DIR / ckpt_arg,
        _PYTHON_SRC_DIR / "checkpoints" / p.name,
        _REPO_ROOT_DIR / "checkpoints" / p.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def resolve_dataset_files(data_path: str | None, data_cfg: DataConfig) -> dict[str, Any]:
    """Locate processed train/val sequence files without modifying files or fabricating fallbacks."""
    search_roots: list[Path] = []
    if data_path:
        p = Path(data_path)
        search_roots.extend([p, p / "v1.0-trainval"])
    else:
        search_roots.extend([
            Path(data_cfg.output_dir),
            Path(data_cfg.dataroot) / "v1.0-trainval",
            Path(data_cfg.dataroot),
            _PYTHON_SRC_DIR / "data" / "nuscenes" / "v1.0-trainval",
            _PYTHON_SRC_DIR / "data" / "nuscenes",
            _REPO_ROOT_DIR / "data" / "nuscenes" / "v1.0-trainval",
            _REPO_ROOT_DIR / "data" / "nuscenes",
        ])

    train_file: Path | None = None
    val_file: Path | None = None
    for root in search_roots:
        for ext in ("jsonl", "json"):
            tf = root / f"train_sequences.{ext}"
            vf = root / f"val_sequences.{ext}"
            if train_file is None and tf.is_file():
                train_file = tf.resolve()
            if val_file is None and vf.is_file():
                val_file = vf.resolve()

    resolved_dir = str(train_file.parent) if train_file else str(Path(data_path or data_cfg.output_dir))
    return {
        "dataroot_configured": str(data_cfg.dataroot),
        "output_dir_configured": str(data_cfg.output_dir),
        "resolved_data_dir": resolved_dir,
        "train_file": str(train_file) if train_file else None,
        "val_file": str(val_file) if val_file else None,
        "train_exists": train_file is not None,
        "val_exists": val_file is not None,
        "dataset_ready": (train_file is not None and val_file is not None),
    }


def _iter_sequence_records(file_path: Path):
    """Stream JSONL line-by-line or load JSON array without excessive memory footprint."""
    with open(file_path, "rb") as fb:
        first_byte = b""
        while True:
            chunk = fb.read(1024)
            if not chunk:
                break
            stripped = chunk.strip()
            if stripped:
                first_byte = stripped[:1]
                break
    with open(file_path, "r", encoding="utf-8") as f:
        if first_byte == b"{" or file_path.suffix.lower() == ".jsonl":
            for line in f:
                s = line.strip()
                if s:
                    yield json.loads(s)
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item


# ------------------------------------------------------------------------------
# Section 5: Mandatory Data Integrity & Evaluation Reliability Audit
# ------------------------------------------------------------------------------

def run_data_audit(
    dataset_name: str,
    data_path: str | None,
    device: Any,
    model_cfg: ModelConfig,
    search_cfg: SearchConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """
    Executes mandatory Section 5 Data Integrity and Evaluation Reliability audit.
    Works in pure Python for sequence files and adds PyTorch tensor/mode checks when available.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []
    blocking_issues: list[str] = []

    ds_paths = resolve_dataset_files(data_path, data_cfg)
    expected_k = int(model_cfg.num_modes)
    expected_past = int(model_cfg.past_len)
    expected_future = int(model_cfg.future_len)
    expected_hz = float(model_cfg.hz)
    expected_dt = 1.0 / expected_hz if expected_hz > 0 else 0.5

    # Check K consistency and known DARTSAlgorithm.validate observation
    metric_k_audit = {
        "model_config_num_modes": expected_k,
        "model_config_past_len": expected_past,
        "model_config_future_len": expected_future,
        "model_config_hz": expected_hz,
        "model_config_past_sec": float(model_cfg.past_sec),
        "model_config_future_sec": float(model_cfg.future_sec),
        "darts_validate_k_hardcoded_note": (
            "Observed in python_src/core/darts_algorithm.py (lines 237-242): "
            "DARTSAlgorithm.validate() passes k=6 to min_ade_k/min_fde_k while ModelConfig.num_modes=5. "
            "Inside evaluations/metrics.py (_top_k_predictions), k_eval=min(k, shape[1]) clamps k to 5, "
            "so numerical results reflect 5 modes. This experiment runner derives K dynamically from model.num_modes "
            "and pred_trajs.shape[1] across minADE, minFDE, and MissRate."
        ),
        "model_output_k_verified": None,
        "all_metrics_same_k": None,
    }

    if HAS_TORCH and FinalTrajectoryModel is not None:
        try:
            ref = load_reference_architecture()
            probe_model = FinalTrajectoryModel(
                best_temporal_idx=ref["indices"][0],
                best_interaction_idx=ref["indices"][1],
                best_map_idx=ref["indices"][2],
                model_config=model_cfg,
                search_config=search_cfg,
            )
            probe_past = torch.zeros(2, expected_past, 2)
            with torch.inference_mode():
                probe_preds, probe_logits = probe_model(probe_past)
            actual_k = int(probe_preds.shape[1])
            metric_k_audit["model_output_k_verified"] = actual_k
            metric_k_audit["logits_shape"] = list(probe_logits.shape)
            metric_k_audit["pred_shape"] = list(probe_preds.shape)
            if actual_k != expected_k or probe_logits.shape[1] != expected_k:
                blocking_issues.append(
                    f"Mode count mismatch: ModelConfig.num_modes={expected_k}, "
                    f"but FinalTrajectoryModel produced shape {list(probe_preds.shape)}."
                )
            else:
                metric_k_audit["all_metrics_same_k"] = True
            del probe_model, probe_past, probe_preds, probe_logits
        except Exception as e:
            warnings.append(f"PyTorch model K-probe failed: {e}")

    # Inspect dataset split files
    split_stats: dict[str, Any] = {}
    tr_scenes: set[str] = set()
    va_scenes: set[str] = set()

    if dataset_name == "synthetic":
        split_stats["note"] = "Dataset explicitly set to 'synthetic' (no real nuScenes JSONL files audited)."
        status = "completed"
        leakage_info = {
            "scene_tokens_available": False,
            "train_unique_scenes": None,
            "val_unique_scenes": None,
            "shared_scenes_count": None,
            "leakage_detected": False,
            "note": "Synthetic dataset generates independent train and val objects.",
        }
    else:
        for split_name, file_str in (("train", ds_paths["train_file"]), ("val", ds_paths["val_file"])):
            if not file_str:
                split_stats[split_name] = {
                    "exists": False,
                    "path": None,
                    "sample_count": 0,
                    "status": "missing",
                }
                continue

            fpath = Path(file_str)
            count = 0
            nan_inf_samples = 0
            missing_past_or_future = 0
            temporal_grid_mismatches = 0
            zero_neighbor_samples = 0
            zero_map_samples = 0
            zero_obstacle_samples = 0
            has_lane_types_field = 0
            has_scene_field = 0
            has_neighbors_future = 0
            has_lane_change_label = 0
            neighbor_counts: list[int] = []
            lane_counts: list[int] = []
            obstacle_counts: list[int] = []
            categories: dict[str, int] = {}
            cities: dict[str, int] = {}
            scenes_set: set[str] = set()

            try:
                for item in _iter_sequence_records(fpath):
                    count += 1
                    past = item.get("past_traj")
                    future = item.get("future_traj")
                    if not isinstance(past, list) or not isinstance(future, list) or len(past) == 0 or len(future) == 0:
                        missing_past_or_future += 1
                        past = past if isinstance(past, list) else []
                        future = future if isinstance(future, list) else []

                    has_bad_num = False
                    for pt in past + future:
                        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                            has_bad_num = True
                            break
                        for v in pt[:2]:
                            try:
                                fv = float(v)
                                if math.isnan(fv) or math.isinf(fv):
                                    has_bad_num = True
                                    break
                            except (TypeError, ValueError):
                                has_bad_num = True
                                break
                        if has_bad_num:
                            break
                    if has_bad_num:
                        nan_inf_samples += 1

                    ts_p = item.get("timestamps_past")
                    ts_f = item.get("timestamps_future")
                    grid_bad = False
                    if len(past) != expected_past or len(future) != expected_future:
                        grid_bad = True
                    if isinstance(ts_p, list) and len(ts_p) > 1:
                        if any(abs((float(ts_p[i + 1]) - float(ts_p[i])) - expected_dt) > 1e-2 for i in range(len(ts_p) - 1)):
                            grid_bad = True
                    if isinstance(ts_f, list) and len(ts_f) > 1:
                        if any(abs((float(ts_f[i + 1]) - float(ts_f[i])) - expected_dt) > 1e-2 for i in range(len(ts_f) - 1)):
                            grid_bad = True
                    if grid_bad:
                        temporal_grid_mismatches += 1

                    n_neigh = len(item.get("neighbors_past") or [])
                    n_lanes = len(item.get("map_lanes") or [])
                    n_obs = len(item.get("static_objects") or [])
                    neighbor_counts.append(n_neigh)
                    lane_counts.append(n_lanes)
                    obstacle_counts.append(n_obs)

                    if n_neigh == 0:
                        zero_neighbor_samples += 1
                    if n_lanes == 0:
                        zero_map_samples += 1
                    if n_obs == 0:
                        zero_obstacle_samples += 1
                    if "map_lane_types" in item:
                        has_lane_types_field += 1
                    if "neighbors_future" in item:
                        has_neighbors_future += 1
                    if "lane_change" in item or "maneuver" in item:
                        has_lane_change_label += 1

                    scene_id = item.get("scene_token") or item.get("scene_name")
                    if scene_id:
                        has_scene_field += 1
                        scenes_set.add(str(scene_id))

                    cat = str(item.get("category", "unknown"))
                    categories[cat] = categories.get(cat, 0) + 1
                    city = str(item.get("city", "unknown"))
                    cities[city] = cities.get(city, 0) + 1

                if split_name == "train":
                    tr_scenes = scenes_set
                else:
                    va_scenes = scenes_set

                zero_map_pct = round((zero_map_samples / count) * 100.0, 2) if count > 0 else None
                zero_neigh_pct = round((zero_neighbor_samples / count) * 100.0, 2) if count > 0 else None
                if zero_map_samples > 0:
                    warnings.append(
                        f"Zero-coverage HD map detected in {split_name} split: "
                        f"{zero_map_samples}/{count} ({zero_map_pct}%) samples have 0 map lanes."
                    )
                if nan_inf_samples > 0:
                    blocking_issues.append(f"Detected {nan_inf_samples} NaN/Inf/corrupt coordinate samples in {split_name} split.")

                split_stats[split_name] = {
                    "exists": True,
                    "path": str(fpath),
                    "file_size_mb": round(fpath.stat().st_size / (1024 ** 2), 3),
                    "sample_count": count,
                    "unique_scenes": len(scenes_set),
                    "samples_with_scene_id": has_scene_field,
                    "nan_inf_samples": nan_inf_samples,
                    "missing_past_or_future_samples": missing_past_or_future,
                    "temporal_grid_mismatches": temporal_grid_mismatches,
                    "zero_neighbor_samples": zero_neighbor_samples,
                    "zero_neighbor_percent": zero_neigh_pct,
                    "zero_coverage_hd_map_samples": zero_map_samples,
                    "zero_coverage_hd_map_percent": zero_map_pct,
                    "zero_static_obstacle_samples": zero_obstacle_samples,
                    "samples_with_map_lane_types": has_lane_types_field,
                    "samples_with_neighbors_future": has_neighbors_future,
                    "samples_with_explicit_lane_change_label": has_lane_change_label,
                    "avg_neighbors": round(sum(neighbor_counts) / count, 3) if count else None,
                    "avg_map_lanes": round(sum(lane_counts) / count, 3) if count else None,
                    "avg_static_obstacles": round(sum(obstacle_counts) / count, 3) if count else None,
                    "categories_top10": dict(sorted(categories.items(), key=lambda kv: -kv[1])[:10]),
                    "cities": cities,
                }
            except Exception as e:
                blocking_issues.append(f"Failed auditing {split_name} file '{fpath}': {e}")
                split_stats[split_name] = {"exists": True, "path": str(fpath), "error": str(e)}

        shared_scenes = sorted(tr_scenes.intersection(va_scenes)) if (tr_scenes and va_scenes) else []
        if shared_scenes:
            blocking_issues.append(
                f"Scene-level split leakage detected: {len(shared_scenes)} scenes appear in both train and val splits."
            )
        if not ds_paths["dataset_ready"]:
            warnings.append(
                f"Real nuScenes sequence files not found under '{ds_paths['resolved_data_dir']}'. "
                f"Run 'python python_src/data/preprocessing.py --dataroot <path_to_nuscenes>' before running training/search experiments."
            )

        leakage_info = {
            "scene_tokens_available": bool(tr_scenes or va_scenes),
            "train_unique_scenes": len(tr_scenes) if ds_paths["train_exists"] else None,
            "val_unique_scenes": len(va_scenes) if ds_paths["val_exists"] else None,
            "shared_scenes_count": len(shared_scenes) if (ds_paths["train_exists"] and ds_paths["val_exists"]) else None,
            "shared_scenes_sample": shared_scenes[:10],
            "leakage_detected": len(shared_scenes) > 0,
            "status": (
                "PASSED (Zero Scene Leakage)"
                if (ds_paths["dataset_ready"] and len(shared_scenes) == 0 and (tr_scenes or va_scenes))
                else ("NOT_VERIFIED_MISSING_FILES" if not ds_paths["dataset_ready"] else "LEAKAGE_DETECTED")
            ),
        }
        status = "completed" if not blocking_issues else "failed"

    # Mask validity spot check using real NuScenesPredictionDataset if files and torch exist
    mask_spot_check: dict[str, Any] = {"executed": False, "reason": None}
    if dataset_name == "nuscenes" and ds_paths["dataset_ready"] and HAS_TORCH:
        try:
            from data.nuscenes_dataset import NuScenesPredictionDataset
            val_ds = NuScenesPredictionDataset(
                data_dir=ds_paths["resolved_data_dir"],
                split="val",
                model_config=model_cfg,
                generate_mock_if_missing=False,
            )
            n_probe = min(8, len(val_ds))
            ego_valid_all = True
            padded_lanes_zero_all = True
            padded_agents_zero_all = True
            for idx in range(n_probe):
                sample = val_ds[idx]
                am = sample["agent_mask"]
                lm = sample["lane_mask"]
                if not bool(am[0].item()):
                    ego_valid_all = False
                if (~am).any() and (sample["agents_past"][:, ~am, :].abs().sum().item() > 1e-6):
                    padded_agents_zero_all = False
                if (~lm).any() and (sample["map_lanes"][~lm].abs().sum().item() > 1e-6):
                    padded_lanes_zero_all = False
            mask_spot_check = {
                "executed": True,
                "samples_probed": n_probe,
                "ego_mask_index0_valid": ego_valid_all,
                "padded_agents_strictly_zero": padded_agents_zero_all,
                "padded_lanes_strictly_zero": padded_lanes_zero_all,
            }
            if not (ego_valid_all and padded_agents_zero_all and padded_lanes_zero_all):
                blocking_issues.append("Mask/padding contract check failed on NuScenesPredictionDataset val samples.")
        except Exception as e:
            mask_spot_check = {"executed": False, "reason": str(e)}
    else:
        mask_spot_check["reason"] = "Requires PyTorch and extracted dataset sequence files."

    ckpt_resolved = resolve_checkpoint_path(checkpoint_path)
    return {
        "experiment": "audit",
        "status": status,
        "timestamp_utc": _utc_iso(),
        "duration_seconds": round(time.perf_counter() - t0, 3),
        "dataset_name": dataset_name,
        "dataset_paths": ds_paths,
        "checkpoint_audit": {
            "requested_checkpoint": checkpoint_path,
            "checkpoint_exists": ckpt_resolved is not None,
            "resolved_checkpoint_path": str(ckpt_resolved) if ckpt_resolved else None,
            "best_architecture_reference": load_reference_architecture(),
        },
        "splits": split_stats,
        "scene_leakage": leakage_info,
        "metric_and_mode_audit": metric_k_audit,
        "mask_and_padding_check": mask_spot_check,
        "validation_vs_test_policy": (
            "Processed 'val_sequences' represent the validation split. "
            "Do NOT report validation-split metrics as held-out final test benchmark scores."
        ),
        "unavailable_dataset_fields": [
            "neighbors_future (cannot compute real Collision Rate without ground-truth neighbor futures)",
            "explicit lane_change / turn signal annotations (only lane_connector, ped_crossing, and kinematic curvature proxies are derivable)",
        ],
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "safe_to_compare_experiments": len(blocking_issues) == 0 and (dataset_name == "synthetic" or ds_paths["dataset_ready"]),
    }


# ------------------------------------------------------------------------------
# DARTS Controlled Runner Wrapper (No modification to core/darts_algorithm.py)
# ------------------------------------------------------------------------------

if HAS_TORCH:

    class ControlledDARTSRunner(DARTSAlgorithm):
        """
        Experiment-specific DARTS wrapper enabling:
          1. Independent freezing of Temporal, Interaction, and HD Map branches
             to fixed reference operators (Research Gap A: A0-A7).
          2. Independent control of Temperature Annealing (disabled => constant T=T_start)
             and Architecture Warmup (Section 6: T0-T3).
          3. Tracking per-epoch alpha entropy and post-warmup operator stability
             plus peak CUDA memory without altering original core/darts_algorithm.py.
        """

        def __init__(
            self,
            search_config: SearchConfig,
            model_config: ModelConfig,
            branch_modes: dict[str, str] | None = None,
            fixed_indices: tuple[int, int, int] = (3, 2, 0),
            enable_temp_annealing: bool = True,
        ):
            self.branch_modes = branch_modes or {
                "temporal": "Search",
                "interaction": "Search",
                "map": "Search",
            }
            self.fixed_indices = tuple(int(x) for x in fixed_indices)
            self.enable_temp_annealing = bool(enable_temp_annealing)
            super().__init__(search_config=search_config, model_config=model_config)

            self._setup_controlled_branches()
            self.alpha_entropy_history: list[dict[str, float]] = []
            self.argmax_history: list[tuple[int, int, int]] = []

        def _setup_controlled_branches(self) -> None:
            branch_map = [
                ("temporal", self.arch_params.alpha_temporal, self.fixed_indices[0]),
                ("interaction", self.arch_params.alpha_interaction, self.fixed_indices[1]),
                ("map", self.arch_params.alpha_map, self.fixed_indices[2]),
            ]
            for name, param, _idx in branch_map:
                if self.branch_modes.get(name, "Search") == "Fixed":
                    param.requires_grad_(False)

            orig_softmax_all = self.arch_params.softmax_all
            branch_modes = self.branch_modes
            fixed_indices = self.fixed_indices

            def controlled_softmax_all(temperature: float = 1.0) -> tuple:
                raw_t, raw_i, raw_m = orig_softmax_all(temperature=temperature)
                outs = []
                for name, raw_p, fix_idx in (
                    ("temporal", raw_t, fixed_indices[0]),
                    ("interaction", raw_i, fixed_indices[1]),
                    ("map", raw_m, fixed_indices[2]),
                ):
                    if branch_modes.get(name, "Search") == "Fixed":
                        one_hot = torch.zeros_like(raw_p)
                        clamped_idx = max(0, min(int(fix_idx), one_hot.numel() - 1))
                        one_hot[clamped_idx] = 1.0
                        outs.append(one_hot)
                    else:
                        outs.append(raw_p)
                return tuple(outs)

            self.arch_params.softmax_all = controlled_softmax_all  # type: ignore[method-assign]

        def _get_current_temperature(self, epoch: int) -> float:
            if not self.enable_temp_annealing:
                return float(self.cfg.temperature_start)
            return super()._get_current_temperature(epoch)

        def validate(self, epoch: int, is_warmup: bool = False) -> dict[str, Any]:
            res = super().validate(epoch, is_warmup=is_warmup)
            alphas = res["alphas"]
            self.alpha_entropy_history.append({
                "epoch": epoch,
                "temporal_entropy": round(_shannon_entropy(alphas["temporal"]), 5),
                "interaction_entropy": round(_shannon_entropy(alphas["interaction"]), 5),
                "map_entropy": round(_shannon_entropy(alphas["map"]), 5),
            })
            t_idx = int(max(range(len(alphas["temporal"])), key=lambda k: alphas["temporal"][k]))
            i_idx = int(max(range(len(alphas["interaction"])), key=lambda k: alphas["interaction"][k]))
            m_idx = int(max(range(len(alphas["map"])), key=lambda k: alphas["map"][k]))
            self.argmax_history.append((t_idx, i_idx, m_idx))
            return res

        def discretize_architecture(self) -> tuple[int, int, int]:
            with torch.no_grad():
                a_temp, a_inter, a_map = self.arch_params.softmax_all()
                best_t = self.fixed_indices[0] if self.branch_modes.get("temporal") == "Fixed" else int(torch.argmax(a_temp).item())
                best_i = self.fixed_indices[1] if self.branch_modes.get("interaction") == "Fixed" else int(torch.argmax(a_inter).item())
                best_m = self.fixed_indices[2] if self.branch_modes.get("map") == "Fixed" else int(torch.argmax(a_map).item())
            return best_t, best_i, best_m

        def compute_stability_metrics(self) -> dict[str, Any]:
            if len(self.argmax_history) < 2:
                return {"argmax_flips_total": 0, "argmax_flips_second_half": 0}
            flips_total = sum(
                1 for idx in range(1, len(self.argmax_history))
                if self.argmax_history[idx] != self.argmax_history[idx - 1]
            )
            half = len(self.argmax_history) // 2
            second_half = self.argmax_history[half:]
            flips_second = sum(
                1 for idx in range(1, len(second_half))
                if second_half[idx] != second_half[idx - 1]
            )
            return {
                "argmax_flips_total": flips_total,
                "argmax_flips_second_half": flips_second,
            }


def _evaluate_standalone_model(
    model: Any,
    val_loader: Any,
    device: Any,
    model_cfg: ModelConfig,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluates a FinalTrajectoryModel on val_loader using dynamic K across minADE, minFDE, and MissRate."""
    model.eval()
    total_samples = 0
    sum_ade = 0.0
    sum_fde = 0.0
    sum_mr = 0.0
    k_model = int(getattr(model, "num_modes", model_cfg.num_modes))
    k_verified = None

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            b = unpack_batch_to_device(batch, device)
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
            k_out = int(pred_trajs.shape[1])
            if k_verified is None:
                k_verified = k_out
                if k_out != k_model:
                    raise ValueError(f"Output K={k_out} != model.num_modes={k_model}")
            bsz = int(b["past"].shape[0])
            total_samples += bsz
            sum_ade += min_ade_k(pred_trajs, b["future"], k=k_out, mode_logits=mode_logits).item() * bsz
            sum_fde += min_fde_k(pred_trajs, b["future"], k=k_out, mode_logits=mode_logits).item() * bsz
            sum_mr += miss_rate_k(pred_trajs, b["future"], k=k_out, threshold=2.0, mode_logits=mode_logits) * bsz

    n = max(1, total_samples)
    return {
        "samples_evaluated": total_samples,
        "k_modes": k_verified if k_verified is not None else k_model,
        "min_ade_k": round(sum_ade / n, 4),
        "min_fde_k": round(sum_fde / n, 4),
        "miss_rate_2m_percent": round((sum_mr / n) * 100.0, 2),
    }


def _make_search_config(args: argparse.Namespace, seed: int, lambda_flops: float | None = None, warmup_epochs: int | None = None) -> SearchConfig:
    cfg = SearchConfig()
    cfg.dataset = args.dataset
    cfg.seed = int(seed)
    cfg.device = str(_resolve_device(args.device))
    if args.data_path:
        cfg.train_path = args.data_path
        cfg.val_path = args.data_path
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = int(args.grad_accum_steps)
    if args.search_epochs is not None:
        cfg.epochs = int(args.search_epochs)
    if warmup_epochs is not None:
        cfg.warmup_epochs = int(warmup_epochs)
    elif args.warmup_epochs is not None:
        cfg.warmup_epochs = int(args.warmup_epochs)
    if args.search_samples_percent is not None:
        cfg.search_samples_percent = float(args.search_samples_percent)
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    if args.no_amp:
        cfg.use_amp = False
    if lambda_flops is not None:
        cfg.lambda_flops = float(lambda_flops)
    return cfg


def _make_train_config(args: argparse.Namespace, seed: int, save_path: Path) -> TrainConfig:
    cfg = TrainConfig()
    cfg.dataset = args.dataset
    cfg.seed = int(seed)
    cfg.device = str(_resolve_device(args.device))
    cfg.save_path = str(save_path)
    if args.data_path:
        cfg.train_path = args.data_path
        cfg.val_path = args.data_path
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = int(args.grad_accum_steps)
    if args.final_epochs is not None:
        cfg.epochs = int(args.final_epochs)
    if args.train_samples_percent is not None:
        cfg.train_samples_percent = float(args.train_samples_percent)
    if args.no_amp:
        cfg.use_amp = False
    return cfg


def _execute_search_and_final_train(
    run_id: str,
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    seed: int,
    checkpoint_dir: Path,
    branch_modes: dict[str, str] | None = None,
    fixed_indices: tuple[int, int, int] = (3, 2, 0),
    enable_temp_annealing: bool = True,
    warmup_epochs: int | None = None,
    lambda_flops: float | None = None,
) -> dict[str, Any]:
    """
    Executes one controlled DARTS search (unless all 3 branches are Fixed) followed by
    standalone FinalModelTrainer retraining and validation evaluation.
    Catches CUDA OOM cleanly and releases all tensors/models before returning.
    """
    if not HAS_TORCH:
        return {
            "status": "not_run",
            "run_id": run_id,
            "seed": seed,
            "checkpoint_path": None,
            "error": "PyTorch is not installed in the active Python environment.",
        }
    device = _resolve_device(args.device)
    s_cfg = _make_search_config(args, seed=seed, lambda_flops=lambda_flops, warmup_epochs=warmup_epochs)
    t_cfg = _make_train_config(args, seed=seed, save_path=checkpoint_dir / f"{run_id}_seed{seed}.pt")
    branch_modes = branch_modes or {"temporal": "Search", "interaction": "Search", "map": "Search"}
    all_fixed = all(v == "Fixed" for v in branch_modes.values())

    searcher = None
    trainer = None
    peak_search_mem_mb = None
    peak_train_mem_mb = None

    try:
        search_val_ade = None
        search_val_fde = None
        search_duration_s = 0.0 if all_fixed else None
        final_alphas = None
        stability = None
        zero_cost = {}

        if not all_fixed:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            searcher = ControlledDARTSRunner(
                search_config=s_cfg,
                model_config=model_cfg,
                branch_modes=branch_modes,
                fixed_indices=fixed_indices,
                enable_temp_annealing=enable_temp_annealing,
            )
            t_s = time.perf_counter()
            best_ops = searcher.run_search()
            search_duration_s = round(time.perf_counter() - t_s, 3)
            if device.type == "cuda":
                peak_search_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if searcher.search_history:
                last_rec = searcher.search_history[-1]
                search_val_ade = last_rec.get("val_ade")
                search_val_fde = last_rec.get("val_fde")
                final_alphas = last_rec.get("alphas")
            stability = searcher.compute_stability_metrics()
            zero_cost = getattr(searcher, "proxies", {}) or {}
            _cleanup_torch_resources(searcher)
            searcher = None
        else:
            best_ops = fixed_indices

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_f = time.perf_counter()
        trainer = FinalModelTrainer(best_op_indices=best_ops, train_config=t_cfg, model_config=model_cfg)
        trained_model = trainer.train_and_evaluate()
        train_duration_s = round(time.perf_counter() - t_f, 3)
        if device.type == "cuda":
            peak_train_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        eval_res = _evaluate_standalone_model(trained_model, trainer.val_loader, device, model_cfg)
        model_summary = trained_model.get_summary()

        peak_mem_candidates = [x for x in (peak_search_mem_mb, peak_train_mem_mb) if x is not None]
        peak_gpu_memory_mb = max(peak_mem_candidates) if peak_mem_candidates else None
        ckpt_size_mb = (
            round(Path(t_cfg.save_path).stat().st_size / (1024 ** 2), 4)
            if Path(t_cfg.save_path).is_file()
            else None
        )

        result = {
            "status": "completed",
            "run_id": run_id,
            "seed": seed,
            "checkpoint_path": str(t_cfg.save_path),
            "checkpoint_size_mb": ckpt_size_mb,
            "selected_indices": list(best_ops),
            "selected_temporal": model_summary["temporal_op"],
            "selected_interaction": model_summary["interaction_op"],
            "selected_map": model_summary["map_op"],
            "parameters": model_summary["parameters"],
            "parameters_m": round(model_summary["parameters"] / 1e6, 4),
            "theoretical_flops_g": model_summary["theoretical_flops_g"],
            "search_duration_s": search_duration_s,
            "final_training_duration_s": train_duration_s,
            "peak_search_gpu_memory_mb": peak_search_mem_mb,
            "peak_train_gpu_memory_mb": peak_train_mem_mb,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "search_supernet_val_ade": search_val_ade,
            "search_supernet_val_fde": search_val_fde,
            "final_val_min_ade": eval_res["min_ade_k"],
            "final_val_min_fde": eval_res["min_fde_k"],
            "final_val_miss_rate_2m": eval_res["miss_rate_2m_percent"],
            "k_modes": eval_res["k_modes"],
            "samples_evaluated": eval_res["samples_evaluated"],
            "final_alphas": final_alphas,
            "stability": stability,
            "zero_cost_proxies": zero_cost,
            "temperature_schedule": {
                "annealing_enabled": enable_temp_annealing,
                "start": float(s_cfg.temperature_start),
                "min": float(s_cfg.temperature_min if enable_temp_annealing else s_cfg.temperature_start),
            },
            "warmup_epochs": int(s_cfg.warmup_epochs),
            "lambda_flops": float(s_cfg.lambda_flops),
            "error": None,
        }
        _cleanup_torch_resources(trained_model, trainer)
        return result

    except Exception as exc:
        err_kind = "CUDA_OOM" if _is_cuda_oom(exc) else type(exc).__name__
        _cleanup_torch_resources(searcher, trainer)
        return {
            "status": "failed",
            "run_id": run_id,
            "seed": seed,
            "checkpoint_path": None,
            "checkpoint_size_mb": None,
            "selected_indices": None,
            "selected_temporal": None,
            "selected_interaction": None,
            "selected_map": None,
            "parameters": None,
            "parameters_m": None,
            "theoretical_flops_g": None,
            "search_duration_s": None,
            "final_training_duration_s": None,
            "peak_search_gpu_memory_mb": None,
            "peak_train_gpu_memory_mb": None,
            "peak_gpu_memory_mb": None,
            "search_supernet_val_ade": None,
            "search_supernet_val_fde": None,
            "final_val_min_ade": None,
            "final_val_min_fde": None,
            "final_val_miss_rate_2m": None,
            "k_modes": int(model_cfg.num_modes),
            "samples_evaluated": None,
            "error": f"[{err_kind}] {exc}",
            "traceback": traceback.format_exc(),
        }


# ------------------------------------------------------------------------------
# Section 2: Research Gap A — Factorized Three-Branch NAS (A0-A7)
# ------------------------------------------------------------------------------

def _build_factorized_summary(
    rows: list[dict[str, Any]],
    seeds: list[int],
    ref: dict[str, Any],
) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") == "failed"]
    overall_status = "completed" if completed and not failed else ("partial" if completed else ("failed" if failed else "not_run"))
    return {
        "experiment": "factorized",
        "status": overall_status,
        "preliminary_single_seed": len(seeds) < 3,
        "seeds": seeds,
        "reference_architecture": ref,
        "rows": rows,
        "summary_counts": {
            "total_planned": len(rows),
            "completed": len(completed),
            "failed": len(failed),
            "not_run": sum(1 for r in rows if r.get("status") == "not_run"),
        },
    }


def run_factorized_ablation(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    output_dir: Path,
    prior_results: list[dict[str, Any]] | None = None,
    enabled_stage_ids: set[str] | None = None,
    force_rerun_ids: set[str] | None = None,
    skip_completed: bool = True,
    continue_on_error: bool = True,
    on_stage_start: Callable[[str], None] | None = None,
    on_stage_complete: Callable[[str, list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ref = load_reference_architecture()
    fixed_indices = tuple(int(x) for x in ref["indices"])
    seeds = list(args.seeds)
    ckpt_dir = output_dir / "checkpoints" / "factorized"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prior_map: dict[tuple[str, int], dict[str, Any]] = {}
    if prior_results:
        for row in prior_results:
            if row.get("status") == "completed" and row.get("config_id") and row.get("seed") is not None:
                prior_map[(str(row["config_id"]), int(row["seed"]))] = row

    rows: list[dict[str, Any]] = []
    executed_count = 0
    halted = False

    for cfg_spec in FACTORIZED_ABLATION_MATRIX:
        cid = cfg_spec["id"]
        is_enabled = True if enabled_stage_ids is None else (cid in enabled_stage_ids)
        is_forced = False if force_rerun_ids is None else (cid in force_rerun_ids)

        if not is_enabled or halted:
            for seed in seeds:
                if (cid, seed) in prior_map:
                    rows.append(prior_map[(cid, seed)])
            continue

        if on_stage_start is not None:
            on_stage_start(cid)

        stage_rows: list[dict[str, Any]] = []
        stage_failed = False

        for seed in seeds:
            if args.max_experiments is not None and executed_count >= args.max_experiments:
                r_skip = {
                    "config_id": cid,
                    "seed": seed,
                    "temporal_mode": cfg_spec["temporal"],
                    "interaction_mode": cfg_spec["interaction"],
                    "map_mode": cfg_spec["map"],
                    "status": "not_run",
                    "error": f"Skipped due to --max-experiments={args.max_experiments}",
                }
                rows.append(r_skip)
                stage_rows.append(r_skip)
                continue

            if not is_forced and skip_completed and (cid, seed) in prior_map:
                r_prev = prior_map[(cid, seed)]
                rows.append(r_prev)
                stage_rows.append(r_prev)
                continue

            executed_count += 1
            branch_modes = {
                "temporal": cfg_spec["temporal"],
                "interaction": cfg_spec["interaction"],
                "map": cfg_spec["map"],
            }
            try:
                res = _execute_search_and_final_train(
                    run_id=f"factorized_{cid}",
                    args=args,
                    model_cfg=model_cfg,
                    seed=seed,
                    checkpoint_dir=ckpt_dir,
                    branch_modes=branch_modes,
                    fixed_indices=fixed_indices,
                    enable_temp_annealing=True,
                )
            except Exception as exc:
                res = {
                    "status": "failed",
                    "run_id": f"factorized_{cid}",
                    "seed": seed,
                    "error": f"[{type(exc).__name__}] {exc}",
                    "traceback": traceback.format_exc(),
                }

            res.update({
                "config_id": cid,
                "temporal_mode": cfg_spec["temporal"],
                "interaction_mode": cfg_spec["interaction"],
                "map_mode": cfg_spec["map"],
                "reference_fixed_indices": list(fixed_indices),
                "reference_fixed_names": ref["names"],
                "preliminary_single_seed": len(seeds) < 3,
            })
            if res.get("status") != "completed":
                stage_failed = True
            rows.append(res)
            stage_rows.append(res)

        if on_stage_complete is not None:
            partial_summary = _build_factorized_summary(list(rows), seeds, ref)
            on_stage_complete(cid, stage_rows, partial_summary)

        if stage_failed and not continue_on_error:
            halted = True

    return _build_factorized_summary(rows, seeds, ref)


# ------------------------------------------------------------------------------
# Section 3: Research Gap B — Hardware-Aware NAS & Profiling
# ------------------------------------------------------------------------------

def _profile_standalone_model(
    model: Any,
    device: Any,
    num_warmup: int,
    num_runs: int,
    precision_mode: str,
) -> dict[str, Any]:
    """Rigorous hardware profiler for extracted FinalTrajectoryModel (mean/median/std/p50/p95/p99 + peak VRAM)."""
    model = model.to(device)
    model.eval()

    past_len = int(getattr(model, "past_len", 4))
    max_agents = int(getattr(model, "max_agents", 20))
    max_lanes = int(getattr(model, "max_lanes", 30))
    lane_pts = int(getattr(model, "lane_pts", 15))
    max_obstacles = int(getattr(model, "max_obstacles", 10))

    dummy_past = torch.randn(1, past_len, 2, device=device)
    dummy_agents = torch.randn(1, past_len, max_agents, 2, device=device)
    dummy_lanes = torch.randn(1, max_lanes, lane_pts, 2, device=device)
    dummy_obstacles = torch.zeros(1, max_obstacles, 4, device=device)
    dummy_obstacle_types = torch.zeros(1, max_obstacles, dtype=torch.long, device=device)
    dummy_obstacle_mask = torch.zeros(1, max_obstacles, dtype=torch.bool, device=device)
    if max_obstacles > 0:
        dummy_obstacles[:, 0, :] = torch.tensor([8.0, 2.0, 0.4, 0.4], device=device)
        dummy_obstacle_mask[:, 0] = True

    use_fp16_amp = (device.type == "cuda" and "fp16" in precision_mode.lower())
    peak_mem_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        try:
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_fp16_amp)
        except (AttributeError, TypeError):
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16_amp)

        for _ in range(num_warmup):
            with autocast_ctx:
                _ = model(
                    dummy_past,
                    agents_past=dummy_agents,
                    map_lanes=dummy_lanes,
                    static_obstacles=dummy_obstacles,
                    static_obstacle_types=dummy_obstacle_types,
                    static_obstacle_mask=dummy_obstacle_mask,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with autocast_ctx:
                _ = model(
                    dummy_past,
                    agents_past=dummy_agents,
                    map_lanes=dummy_lanes,
                    static_obstacles=dummy_obstacles,
                    static_obstacle_types=dummy_obstacle_types,
                    static_obstacle_mask=dummy_obstacle_mask,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    if device.type == "cuda":
        peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3)

    stats = _stats_summary(latencies_ms)
    mean_ms = stats["mean"]
    fps = round(1000.0 / mean_ms, 2) if (mean_ms is not None and mean_ms > 0) else None
    summary = model.get_summary()

    return {
        "latency_ms_mean": round(mean_ms, 4) if mean_ms is not None else None,
        "latency_ms_median": round(stats["median"], 4) if stats["median"] is not None else None,
        "latency_ms_std": round(stats["std"], 4) if stats["std"] is not None else None,
        "latency_ms_p50": round(stats["p50"], 4) if stats["p50"] is not None else None,
        "latency_ms_p95": round(stats["p95"], 4) if stats["p95"] is not None else None,
        "latency_ms_p99": round(stats["p99"], 4) if stats["p99"] is not None else None,
        "throughput_fps": fps,
        "peak_gpu_memory_mb": peak_mem_mb,
        "parameters": summary["parameters"],
        "parameters_m": round(summary["parameters"] / 1e6, 4),
        "theoretical_flops_g": summary["theoretical_flops_g"],
        "theoretical_flops_m": summary["theoretical_flops_m"],
        "precision_mode": "AMP_FP16" if use_fp16_amp else "FP32",
    }


def _build_hardware_summary(
    rows: list[dict[str, Any]],
    hw_env: dict[str, Any],
    args: argparse.Namespace,
    dedup_lambdas: list[float],
) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    pending = [r for r in rows if r.get("status") == "pending"]
    failed = [r for r in rows if r.get("status") == "failed"]
    overall_status = (
        "completed"
        if completed and not pending and not failed
        else ("partial" if completed else ("failed" if failed else "not_run"))
    )
    return {
        "experiment": "hardware",
        "status": overall_status,
        "hardware_environment": hw_env,
        "profiling_protocol": {
            "warmup_iterations": args.hw_warmup,
            "timed_iterations": args.hw_runs,
            "cuda_synchronized": True,
            "batch_size_profiled": 1,
            "model_inspected": "FinalTrajectoryModel (extracted standalone discrete model, NOT SearchSuperNet)",
        },
        "lambda_flops_values_configured": dedup_lambdas,
        "rows": rows,
    }


def run_hardware_profiling(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    search_cfg: SearchConfig,
    output_dir: Path,
    prior_results: list[dict[str, Any]] | None = None,
    run_baselines: bool = True,
    run_lambda_sweep: bool | None = None,
    force_rerun_baselines: bool = False,
    force_rerun_lambda: bool = False,
    continue_on_error: bool = True,
    on_stage_start: Callable[[str], None] | None = None,
    on_stage_complete: Callable[[str, list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Profiles extracted standalone FinalTrajectoryModel candidates:
      - TrajectoryLNAS discovered reference ([3, 2, 0] or from --checkpoint)
      - Manual lightweight & heavy fixed operator baselines
      - Configurable lambda_flops penalty sweep (marked pending if search/training not run)
    """
    ref = load_reference_architecture()
    ref_idx = tuple(int(x) for x in ref["indices"])
    s_cfg_default = SearchConfig()
    lambda_values = list(args.lambda_flops_values) if args.lambda_flops_values else [0.0, 0.0001, float(s_cfg_default.lambda_flops), 0.001, 0.01]
    # Deduplicate preserving order
    dedup_lambdas: list[float] = []
    for lv in lambda_values:
        if lv not in dedup_lambdas:
            dedup_lambdas.append(float(lv))

    ckpt_resolved = resolve_checkpoint_path(args.checkpoint)
    ckpt_dir = output_dir / "checkpoints" / "hardware"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    hw_env = collect_hardware_environment(device)

    active_lambda_sweep = bool(args.run_lambda_search) if run_lambda_sweep is None else bool(run_lambda_sweep)
    prior_by_cid: dict[str, dict[str, Any]] = {}
    if prior_results:
        for pr in prior_results:
            if pr.get("status") == "completed" and pr.get("candidate_id"):
                prior_by_cid[str(pr["candidate_id"])] = pr

    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "manual_lightweight",
            "category": "manual_baseline",
            "ops_indices": [0, 0, 0],  # GRUEncoder + GCNLayer + LaneAttention
            "lambda_flops": None,
            "checkpoint": None,
        },
        {
            "candidate_id": "manual_heavy_attention",
            "category": "manual_baseline",
            "ops_indices": [3, 3, 1],  # TemporalSelfAttention + PairwiseAttention + LaneGraphEncoder
            "lambda_flops": None,
            "checkpoint": None,
        },
        {
            "candidate_id": "trajectory_lnas_reference",
            "category": "discovered_reference",
            "ops_indices": list(ref_idx),
            "lambda_flops": float(ref["checkpoint_metadata"].get("lambda_flops", s_cfg_default.lambda_flops)) if isinstance(ref["checkpoint_metadata"], dict) else float(s_cfg_default.lambda_flops),
            "checkpoint": str(ckpt_resolved) if ckpt_resolved else None,
        },
    ]

    for lv in dedup_lambdas:
        candidates.append({
            "candidate_id": f"lambda_flops_{lv:g}",
            "category": "flops_penalty_sweep",
            "ops_indices": None,
            "lambda_flops": lv,
            "checkpoint": None,
        })

    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    executed = 0

    baseline_cands = [c for c in candidates if c["category"] != "flops_penalty_sweep"]
    sweep_cands = [c for c in candidates if c["category"] == "flops_penalty_sweep"]

    # Stage: HARDWARE (standalone baselines & reference)
    if not run_baselines:
        for cand in baseline_cands:
            cid = cand["candidate_id"]
            if cid in prior_by_cid:
                rows.append(prior_by_cid[cid])
    else:
        if on_stage_start is not None:
            on_stage_start("HARDWARE")
        for cand in baseline_cands:
            cid = cand["candidate_id"]
            if not force_rerun_baselines and cid in prior_by_cid and on_stage_start is None:
                rows.append(prior_by_cid[cid])
                baseline_rows.append(prior_by_cid[cid])
                continue
            if args.max_experiments is not None and executed >= args.max_experiments:
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "not_run",
                    "error": f"Skipped due to --max-experiments={args.max_experiments}",
                }
                rows.append(r)
                baseline_rows.append(r)
                continue
            if not HAS_TORCH:
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "not_run",
                    "accuracy_status": "pending",
                    "error": "PyTorch is not installed in the active Python environment.",
                }
                rows.append(r)
                baseline_rows.append(r)
                continue

            executed += 1
            try:
                ops = cand["ops_indices"]
                model = FinalTrajectoryModel(ops[0], ops[1], ops[2], model_config=model_cfg, search_config=search_cfg)
                ckpt_path_str = cand.get("checkpoint")
                ckpt_size_mb = None
                acc_status = "untrained_architecture_profile_only"
                val_ade = None
                val_fde = None
                val_mr = None

                if ckpt_path_str and Path(ckpt_path_str).is_file():
                    ckpt_p = Path(ckpt_path_str)
                    ckpt_size_mb = round(ckpt_p.stat().st_size / (1024 ** 2), 4)
                    try:
                        ckpt_obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                        model.load_state_dict(ckpt_obj.get("model_state_dict", ckpt_obj))
                        acc_status = "checkpoint_loaded"
                    except Exception as e:
                        acc_status = f"checkpoint_load_failed ({e})"

                if acc_status == "checkpoint_loaded" or args.allow_untrained:
                    try:
                        _, val_loader = get_dataloaders(
                            train_path=args.data_path or search_cfg.train_path,
                            val_path=args.data_path or search_cfg.val_path,
                            batch_size=args.batch_size or search_cfg.batch_size,
                            dataset_type=args.dataset,
                            model_config=model_cfg,
                        )
                        model = model.to(device)
                        eval_res = _evaluate_standalone_model(model, val_loader, device, model_cfg)
                        val_ade = eval_res["min_ade_k"]
                        val_fde = eval_res["min_fde_k"]
                        val_mr = eval_res["miss_rate_2m_percent"]
                        acc_status = "measured_trained" if acc_status == "checkpoint_loaded" else "measured_untrained_smoke"
                    except Exception as e:
                        if acc_status == "checkpoint_loaded":
                            acc_status = f"checkpoint_loaded_eval_pending ({e})"

                prec = "FP32" if args.no_amp else "AMP_FP16"
                gpu_prof = _profile_standalone_model(model, device, args.hw_warmup, args.hw_runs, prec) if device.type == "cuda" else None
                cpu_prof = _profile_standalone_model(model, torch.device("cpu"), max(5, args.hw_warmup // 2), max(20, args.hw_runs // 2), "FP32")
                m_sum = model.get_summary()
                _cleanup_torch_resources(model)

                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "completed",
                    "accuracy_status": acc_status,
                    "selected_indices": list(ops),
                    "temporal_op": m_sum["temporal_op"],
                    "interaction_op": m_sum["interaction_op"],
                    "map_op": m_sum["map_op"],
                    "parameters": cpu_prof["parameters"],
                    "parameters_m": cpu_prof["parameters_m"],
                    "theoretical_flops_g": cpu_prof["theoretical_flops_g"],
                    "gpu_latency_ms_mean": gpu_prof["latency_ms_mean"] if gpu_prof else None,
                    "gpu_latency_ms_median": gpu_prof["latency_ms_median"] if gpu_prof else None,
                    "gpu_latency_ms_std": gpu_prof["latency_ms_std"] if gpu_prof else None,
                    "gpu_latency_ms_p95": gpu_prof["latency_ms_p95"] if gpu_prof else None,
                    "gpu_latency_ms_p99": gpu_prof["latency_ms_p99"] if gpu_prof else None,
                    "gpu_fps": gpu_prof["throughput_fps"] if gpu_prof else None,
                    "cpu_latency_ms_mean": cpu_prof["latency_ms_mean"],
                    "cpu_latency_ms_median": cpu_prof["latency_ms_median"],
                    "cpu_latency_ms_std": cpu_prof["latency_ms_std"],
                    "cpu_latency_ms_p95": cpu_prof["latency_ms_p95"],
                    "cpu_fps": cpu_prof["throughput_fps"],
                    "peak_gpu_memory_mb": gpu_prof["peak_gpu_memory_mb"] if gpu_prof else None,
                    "checkpoint_size_mb": ckpt_size_mb,
                    "search_duration_s": None,
                    "final_training_duration_s": None,
                    "final_val_min_ade": val_ade,
                    "final_val_min_fde": val_fde,
                    "final_val_miss_rate_2m": val_mr,
                    "precision_mode": gpu_prof["precision_mode"] if gpu_prof else cpu_prof["precision_mode"],
                    "error": None,
                }
            except Exception as exc:
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "failed",
                    "accuracy_status": "failed",
                    "error": f"[{type(exc).__name__}] {exc}",
                }
            rows.append(r)
            baseline_rows.append(r)

        if on_stage_complete is not None:
            on_stage_complete("HARDWARE", baseline_rows, _build_hardware_summary(list(rows), hw_env, args, dedup_lambdas))
            if any(x.get("status") in ("failed", "not_run") for x in baseline_rows) and not continue_on_error:
                for cand in sweep_cands:
                    cid = cand["candidate_id"]
                    if cid in prior_by_cid:
                        rows.append(prior_by_cid[cid])
                return _build_hardware_summary(rows, hw_env, args, dedup_lambdas)

    # Stage: HARDWARE_LAMBDA_SEARCH
    if not active_lambda_sweep:
        for cand in sweep_cands:
            cid = cand["candidate_id"]
            if cid in prior_by_cid:
                rows.append(prior_by_cid[cid])
            elif run_baselines:
                rows.append({
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "pending",
                    "accuracy_status": "pending_lambda_search_not_executed",
                    "error": (
                        "Checkpoint/search result for this lambda_flops value does not exist yet. "
                        "Re-run with '--experiment hardware --run-lambda-search' (or RUN_HARDWARE_LAMBDA_SEARCH=True) once dataset sequences are ready."
                    ),
                })
    else:
        if on_stage_start is not None:
            on_stage_start("HARDWARE_LAMBDA_SEARCH")
        for cand in sweep_cands:
            cid = cand["candidate_id"]
            if not force_rerun_lambda and cid in prior_by_cid and on_stage_start is None:
                rows.append(prior_by_cid[cid])
                sweep_rows.append(prior_by_cid[cid])
                continue
            if args.max_experiments is not None and executed >= args.max_experiments:
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "not_run",
                    "error": f"Skipped due to --max-experiments={args.max_experiments}",
                }
                rows.append(r)
                sweep_rows.append(r)
                continue

            executed += 1
            try:
                res = _execute_search_and_final_train(
                    run_id=f"hw_{cid}",
                    args=args,
                    model_cfg=model_cfg,
                    seed=int(args.seed),
                    checkpoint_dir=ckpt_dir,
                    branch_modes={"temporal": "Search", "interaction": "Search", "map": "Search"},
                    fixed_indices=ref_idx,
                    lambda_flops=cand["lambda_flops"],
                )
            except Exception as exc:
                res = {
                    "status": "failed",
                    "error": f"[{type(exc).__name__}] {exc}",
                }

            if res.get("status") == "completed" and res.get("selected_indices") and HAS_TORCH and FinalTrajectoryModel is not None:
                sel = res["selected_indices"]
                model = FinalTrajectoryModel(sel[0], sel[1], sel[2], model_config=model_cfg, search_config=search_cfg)
                ckpt_p = res.get("checkpoint_path")
                if ckpt_p and Path(ckpt_p).is_file():
                    ckpt_obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt_obj.get("model_state_dict", ckpt_obj))
                prec = "FP32" if args.no_amp else "AMP_FP16"
                gpu_prof = _profile_standalone_model(model, device, args.hw_warmup, args.hw_runs, prec) if device.type == "cuda" else None
                cpu_prof = _profile_standalone_model(model, torch.device("cpu"), max(5, args.hw_warmup // 2), max(20, args.hw_runs // 2), "FP32")
                _cleanup_torch_resources(model)

                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "completed",
                    "accuracy_status": "measured_trained",
                    "selected_indices": sel,
                    "temporal_op": res.get("selected_temporal"),
                    "interaction_op": res.get("selected_interaction"),
                    "map_op": res.get("selected_map"),
                    "parameters": cpu_prof["parameters"],
                    "parameters_m": cpu_prof["parameters_m"],
                    "theoretical_flops_g": cpu_prof["theoretical_flops_g"],
                    "gpu_latency_ms_mean": gpu_prof["latency_ms_mean"] if gpu_prof else None,
                    "gpu_latency_ms_median": gpu_prof["latency_ms_median"] if gpu_prof else None,
                    "gpu_latency_ms_std": gpu_prof["latency_ms_std"] if gpu_prof else None,
                    "gpu_latency_ms_p95": gpu_prof["latency_ms_p95"] if gpu_prof else None,
                    "gpu_latency_ms_p99": gpu_prof["latency_ms_p99"] if gpu_prof else None,
                    "gpu_fps": gpu_prof["throughput_fps"] if gpu_prof else None,
                    "cpu_latency_ms_mean": cpu_prof["latency_ms_mean"],
                    "cpu_latency_ms_median": cpu_prof["latency_ms_median"],
                    "cpu_latency_ms_std": cpu_prof["latency_ms_std"],
                    "cpu_latency_ms_p95": cpu_prof["latency_ms_p95"],
                    "cpu_fps": cpu_prof["throughput_fps"],
                    "peak_gpu_memory_mb": gpu_prof["peak_gpu_memory_mb"] if gpu_prof else None,
                    "checkpoint_size_mb": res.get("checkpoint_size_mb"),
                    "search_duration_s": res.get("search_duration_s"),
                    "final_training_duration_s": res.get("final_training_duration_s"),
                    "final_val_min_ade": res.get("final_val_min_ade"),
                    "final_val_min_fde": res.get("final_val_min_fde"),
                    "final_val_miss_rate_2m": res.get("final_val_miss_rate_2m"),
                    "precision_mode": gpu_prof["precision_mode"] if gpu_prof else cpu_prof["precision_mode"],
                    "error": None,
                }
            elif res.get("status") == "completed":
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": "completed",
                    "accuracy_status": "measured_trained",
                    "selected_indices": res.get("selected_indices"),
                    "temporal_op": res.get("selected_temporal"),
                    "interaction_op": res.get("selected_interaction"),
                    "map_op": res.get("selected_map"),
                    "parameters": res.get("parameters"),
                    "parameters_m": res.get("parameters_m"),
                    "theoretical_flops_g": res.get("theoretical_flops_g"),
                    "checkpoint_size_mb": res.get("checkpoint_size_mb"),
                    "search_duration_s": res.get("search_duration_s"),
                    "final_training_duration_s": res.get("final_training_duration_s"),
                    "final_val_min_ade": res.get("final_val_min_ade"),
                    "final_val_min_fde": res.get("final_val_min_fde"),
                    "final_val_miss_rate_2m": res.get("final_val_miss_rate_2m"),
                    "error": None,
                }
            else:
                r = {
                    "candidate_id": cid,
                    "category": cand["category"],
                    "lambda_flops": cand["lambda_flops"],
                    "status": res.get("status", "failed"),
                    "accuracy_status": "failed" if res.get("status") == "failed" else "pending",
                    "error": res.get("error"),
                }
            rows.append(r)
            sweep_rows.append(r)

        if on_stage_complete is not None:
            on_stage_complete("HARDWARE_LAMBDA_SEARCH", sweep_rows, _build_hardware_summary(list(rows), hw_env, args, dedup_lambdas))

    return _build_hardware_summary(rows, hw_env, args, dedup_lambdas)


# ------------------------------------------------------------------------------
# Section 4: Research Gap C — Scene Complexity Stratification & Prototype
# ------------------------------------------------------------------------------

if HAS_TORCH:

    class ExperimentalSceneConditionedPrototype(nn.Module):
        """
        EXPERIMENTAL PROTOTYPE (Research Gap C):
        Non-invasive scene-conditioned multi-branch mixture wrapper over standalone-style
        operator banks, using a compact scene descriptor extracted purely from existing
        observable batch inputs (valid neighbors, close interactions <=20m, valid lanes,
        map presence, connector/crossing presence, static obstacles, ego turn angle).

        Clearly marked EXPERIMENTAL: requires explicit training (--run-scene-prototype)
        and is never claimed as part of the base TrajectoryLNAS architecture.
        """

        def __init__(self, model_cfg: ModelConfig, search_cfg: SearchConfig):
            super().__init__()
            from models.base_operator import trajectory_kinematics
            from models.supernet_operators import (
                PolylineEncoder,
                build_interaction_operator,
                build_map_operator,
                build_temporal_operator,
            )
            self._trajectory_kinematics = trajectory_kinematics
            self.model_cfg = model_cfg
            self.d_model = int(model_cfg.d_model)
            self.hz = float(model_cfg.hz)
            self.future_len = int(model_cfg.future_len)
            self.num_modes = int(model_cfg.num_modes)

            self.temporal_names = list(search_cfg.temporal_ops)
            self.interaction_names = list(search_cfg.interaction_ops)
            self.map_names = list(search_cfg.map_ops)

            self.input_proj = nn.Linear(8, self.d_model)
            self.neighbor_proj = nn.Linear(8, self.d_model)
            self.category_emb = nn.Embedding(10, self.d_model)
            self.obstacle_proj = nn.Linear(4, self.d_model)
            self.obstacle_type_emb = nn.Embedding(6, self.d_model)
            self.map_encoder = PolylineEncoder(in_channels=4, d_model=self.d_model)

            self.temporal_ops = nn.ModuleList([build_temporal_operator(n, self.d_model) for n in self.temporal_names])
            self.interaction_ops = nn.ModuleList([build_interaction_operator(n, self.d_model) for n in self.interaction_names])
            self.map_ops = nn.ModuleList([build_map_operator(n, self.d_model) for n in self.map_names])

            # 7 observed scene complexity features -> per-branch softmax gates
            self.router = nn.Sequential(
                nn.Linear(7, 32),
                nn.ReLU(),
                nn.Linear(32, len(self.temporal_names) + len(self.interaction_names) + len(self.map_names)),
            )

            self.fusion = nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
            )
            self.decoder = nn.GRU(self.d_model, self.d_model, batch_first=True)
            self.out_proj = nn.Linear(self.d_model, self.num_modes * 2)
            self.prob_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.num_modes),
            )

        @staticmethod
        def extract_scene_descriptor(batch: dict[str, Any], device: Any) -> torch.Tensor:
            """Extract 7 normalized scene indicators strictly from existing batch tensors."""
            past = batch["past"].to(device)
            bsz = past.shape[0]

            agent_mask = batch.get("agent_mask")
            if agent_mask is not None:
                am = agent_mask.to(device=device, dtype=torch.bool)
                num_neigh = am[:, 1:].sum(dim=1).float() if am.shape[1] > 1 else torch.zeros(bsz, device=device)
            else:
                num_neigh = torch.zeros(bsz, device=device)

            agents_past = batch.get("agents_past")
            if agents_past is not None and agent_mask is not None and agents_past.shape[2] > 1:
                ap = agents_past.to(device=device)[:, -1, 1:, :2]
                ego_p = past[:, -1:, :2]
                dists = torch.linalg.vector_norm(ap - ego_p, dim=-1)
                am_n = agent_mask.to(device=device, dtype=torch.bool)[:, 1:]
                close_20m = ((dists <= 20.0) & am_n).sum(dim=1).float()
            else:
                close_20m = torch.zeros(bsz, device=device)

            lane_mask = batch.get("lane_mask")
            if lane_mask is not None:
                lm = lane_mask.to(device=device, dtype=torch.bool)
                num_lanes = lm.sum(dim=1).float()
                has_map = (num_lanes > 0).float()
            else:
                num_lanes = torch.zeros(bsz, device=device)
                has_map = torch.zeros(bsz, device=device)

            lane_types = batch.get("map_lane_types")
            if lane_types is not None and lane_mask is not None:
                lt = lane_types.to(device=device)
                lm = lane_mask.to(device=device, dtype=torch.bool)
                has_conn = (((lt == 1) | (lt == 3)) & lm).any(dim=1).float()
            else:
                has_conn = torch.zeros(bsz, device=device)

            obs_mask = batch.get("static_obstacle_mask")
            num_obs = obs_mask.to(device=device, dtype=torch.bool).sum(dim=1).float() if obs_mask is not None else torch.zeros(bsz, device=device)

            if past.shape[1] >= 3:
                vel = past[:, 1:, :2] - past[:, :-1, :2]
                ang = torch.atan2(vel[:, :, 1], vel[:, :, 0] + 1e-6)
                dang = torch.abs(ang[:, 1:] - ang[:, :-1])
                dang = torch.minimum(dang, 2.0 * math.pi - dang).sum(dim=1)
            else:
                dang = torch.zeros(bsz, device=device)

            desc = torch.stack([
                num_neigh / 20.0,
                close_20m / 10.0,
                num_lanes / 30.0,
                has_map,
                has_conn,
                num_obs / 10.0,
                torch.clamp(dang / math.pi, 0.0, 1.0),
            ], dim=-1)
            return desc

        def forward(self, batch: dict[str, Any], device: Any) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            past = batch["past"].to(device)
            bsz = past.shape[0]
            scene_desc = self.extract_scene_descriptor(batch, device)
            gate_logits = self.router(scene_desc)
            nt, ni, nm = len(self.temporal_names), len(self.interaction_names), len(self.map_names)
            gt = torch.softmax(gate_logits[:, :nt], dim=-1)
            gi = torch.softmax(gate_logits[:, nt:nt + ni], dim=-1)
            gm = torch.softmax(gate_logits[:, nt + ni:], dim=-1)

            kin_ego = self._trajectory_kinematics(past, hz=self.hz)
            target_feat = self.input_proj(kin_ego)
            cat_id = batch.get("category_id")
            if cat_id is not None:
                target_feat = target_feat + self.category_emb(cat_id.to(device).view(bsz).clamp(0, 9)).unsqueeze(1)

            agents_past = batch.get("agents_past")
            agent_mask = batch.get("agent_mask")
            if agents_past is not None:
                agents_past = agents_past.to(device)
                am = agent_mask.to(device=device, dtype=torch.bool).clone() if agent_mask is not None else (agents_past.abs().sum(dim=(1, 3)) > 1e-4)
                am[:, 0] = True
                kin_ag = self._trajectory_kinematics(agents_past, hz=self.hz)
                emb_n = self.neighbor_proj(kin_ag)
                n_feat = 0.5 * (emb_n.mean(dim=1) + emb_n[:, -1, :, :]) * am.unsqueeze(-1).float()
                a_pos = agents_past[:, -1, :, :2]
            else:
                n_feat = torch.zeros(bsz, 1, self.d_model, device=device)
                am = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
                a_pos = torch.zeros(bsz, 1, 2, device=device)

            map_lanes = batch.get("map_lanes")
            lane_mask = batch.get("lane_mask")
            lane_types = batch.get("map_lane_types")
            if map_lanes is not None and map_lanes.shape[1] > 0:
                map_lanes = map_lanes.to(device)
                lm = lane_mask.to(device=device, dtype=torch.bool) if lane_mask is not None else (map_lanes.abs().sum(dim=(2, 3)) > 1e-4)
                lt = lane_types.to(device) if lane_types is not None else None
                l_feat = self.map_encoder(map_lanes, lane_types=lt, mask=lm) * lm.unsqueeze(-1).float()
            else:
                l_feat = torch.zeros(bsz, 1, self.d_model, device=device)
                lm = torch.zeros(bsz, 1, dtype=torch.bool, device=device)

            f_t = sum(gt[:, idx].view(bsz, 1, 1) * op(target_feat) for idx, op in enumerate(self.temporal_ops))
            f_i = sum(gi[:, idx].view(bsz, 1, 1) * op(target_feat, neighbor_feats=n_feat, agent_mask=am, agent_pos=a_pos) for idx, op in enumerate(self.interaction_ops))
            f_m = sum(gm[:, idx].view(bsz, 1, 1) * op(target_feat, lane_feats=l_feat, lane_mask=lm) for idx, op in enumerate(self.map_ops))

            x_fused = self.fusion(torch.cat([f_t, f_i, f_m], dim=-1))
            dec_in = x_fused[:, -1:, :].repeat(1, self.future_len, 1)
            dec_out, _ = self.decoder(dec_in)
            deltas = self.out_proj(dec_out).view(bsz, self.future_len, self.num_modes, 2).permute(0, 2, 1, 3)
            pred = torch.cumsum(deltas, dim=2)
            logits = self.prob_head(x_fused[:, -1, :])
            return pred, logits, (gt, gi, gm)


def _compute_batch_scene_indicators(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts per-sample scene complexity descriptors strictly from actual batch fields."""
    past = batch["past"]
    bsz = past.shape[0]
    agent_mask = batch.get("agent_mask")
    agents_past = batch.get("agents_past")
    lane_mask = batch.get("lane_mask")
    lane_types = batch.get("map_lane_types")
    obs_mask = batch.get("static_obstacle_mask")

    out: list[dict[str, Any]] = []
    for idx in range(bsz):
        if agent_mask is not None and agent_mask.shape[1] > 1:
            n_neigh = int(agent_mask[idx, 1:].bool().sum().item())
        else:
            n_neigh = 0

        if agents_past is not None and agent_mask is not None and agents_past.shape[2] > 1:
            ego_xy = past[idx, -1, :2]
            nbr_xy = agents_past[idx, -1, 1:, :2]
            dists = torch.linalg.vector_norm(nbr_xy - ego_xy.unsqueeze(0), dim=-1)
            valid_nbr = agent_mask[idx, 1:].bool()
            close_20m = int(((dists <= 20.0) & valid_nbr).sum().item())
        else:
            close_20m = 0

        n_lanes = int(lane_mask[idx].bool().sum().item()) if lane_mask is not None else 0
        has_map = n_lanes > 0
        n_obs = int(obs_mask[idx].bool().sum().item()) if obs_mask is not None else 0

        if lane_types is not None and lane_mask is not None:
            valid_lt = lane_types[idx][lane_mask[idx].bool()]
            has_intersection = bool(((valid_lt == 1) | (valid_lt == 3)).any().item())
        else:
            has_intersection = None

        p_seq = past[idx, :, :2]
        turn_rad = 0.0
        if p_seq.shape[0] >= 3:
            vel = p_seq[1:] - p_seq[:-1]
            for t in range(vel.shape[0] - 1):
                a1 = math.atan2(float(vel[t, 1]), float(vel[t, 0]))
                a2 = math.atan2(float(vel[t + 1, 1]), float(vel[t + 1, 0]))
                da = abs(a2 - a1)
                turn_rad += min(da, 2.0 * math.pi - da)

        # Transparent normalized composite complexity score
        comp_score = (
            min(1.0, n_neigh / 10.0) * 0.35
            + min(1.0, close_20m / 6.0) * 0.25
            + min(1.0, n_lanes / 20.0) * 0.20
            + min(1.0, n_obs / 6.0) * 0.10
            + min(1.0, turn_rad / (math.pi / 2.0)) * 0.10
        )
        out.append({
            "num_neighbors": n_neigh,
            "close_agents_20m": close_20m,
            "num_valid_lanes": n_lanes,
            "has_map": has_map,
            "num_static_obstacles": n_obs,
            "has_intersection_proxy": has_intersection,
            "ego_turn_rad": round(turn_rad, 4),
            "complexity_score": round(comp_score, 4),
        })
    return out


def run_scene_complexity_evaluation(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    search_cfg: SearchConfig,
    output_dir: Path,
    run_stratification: bool = True,
    run_prototype: bool | None = None,
    prior_result: dict[str, Any] | None = None,
    continue_on_error: bool = True,
    on_stage_start: Callable[[str], None] | None = None,
    on_stage_complete: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Research Gap C:
      1. Diagnostic stratification of validation scenes using transparent dataset quantiles.
      2. Evaluation of the fixed TrajectoryLNAS model across scene strata (if trained
         checkpoint exists or --allow-untrained is explicitly passed).
      3. Experimental prototype of SceneConditionedArchitecture (clearly marked experimental).
    """
    unavailable_features = [
        "explicit lane_change annotation (not present in dataset JSONL/Tensor contract)",
        "neighbors_future trajectories (not extracted in preprocessing.py)",
    ]
    active_prototype = bool(args.run_scene_prototype) if run_prototype is None else bool(run_prototype)

    if run_stratification and on_stage_start is not None:
        on_stage_start("SCENE")

    if not HAS_TORCH:
        res_not_torch = {
            "experiment": "scene",
            "status": "not_run",
            "error": "PyTorch is not installed in the active Python environment.",
            "unavailable_dataset_fields": unavailable_features,
            "rows": list((prior_result or {}).get("rows") or []),
            "experimental_scene_conditioned_prototype": (prior_result or {}).get("experimental_scene_conditioned_prototype", {
                "experimental": True,
                "status": "not_run",
            }),
        }
        if run_stratification and on_stage_complete is not None:
            on_stage_complete("SCENE", res_not_torch)
        elif active_prototype and on_stage_start is not None:
            on_stage_start("SCENE_PROTOTYPE")
            if on_stage_complete is not None:
                on_stage_complete("SCENE_PROTOTYPE", res_not_torch)
        return res_not_torch

    ref = load_reference_architecture()
    ref_idx = tuple(int(x) for x in ref["indices"])
    ckpt_resolved = resolve_checkpoint_path(args.checkpoint)
    device = _resolve_device(args.device)

    try:
        train_loader, val_loader = get_dataloaders(
            train_path=args.data_path or search_cfg.train_path,
            val_path=args.data_path or search_cfg.val_path,
            batch_size=args.batch_size or search_cfg.batch_size,
            dataset_type=args.dataset,
            model_config=model_cfg,
        )
    except Exception as e:
        err_res = {
            "experiment": "scene",
            "status": "pending",
            "error": f"Dataset loader unavailable ({e}). Extract dataset sequences first.",
            "unavailable_dataset_fields": unavailable_features,
            "rows": list((prior_result or {}).get("rows") or []),
        }
        if run_stratification and on_stage_complete is not None:
            on_stage_complete("SCENE", err_res)
        elif active_prototype and on_stage_start is not None:
            on_stage_start("SCENE_PROTOTYPE")
            if on_stage_complete is not None:
                on_stage_complete("SCENE_PROTOTYPE", err_res)
        return err_res

    k_eval = int(model_cfg.num_modes)
    if run_stratification:
        can_evaluate_model = (ckpt_resolved is not None) or bool(args.allow_untrained)
        model = None
        model_state_label = "pending_checkpoint"

        if can_evaluate_model:
            model = FinalTrajectoryModel(ref_idx[0], ref_idx[1], ref_idx[2], model_config=model_cfg, search_config=search_cfg).to(device)
            if ckpt_resolved is not None:
                ckpt_obj = torch.load(ckpt_resolved, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt_obj.get("model_state_dict", ckpt_obj))
                model_state_label = "trained_checkpoint"
            else:
                model_state_label = "untrained_smoke_test"
            model.eval()

        # Single streaming pass collecting per-sample indicators and (if model active) per-sample errors
        sample_records: list[dict[str, Any]] = []

        with torch.inference_mode():
            for batch in val_loader:
                indicators = _compute_batch_scene_indicators(batch)
                if model is not None:
                    b = unpack_batch_to_device(batch, device)
                    preds, logits = model(
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
                    k_out = int(preds.shape[1])
                    if logits is not None:
                        top_idx = logits.topk(k_out, dim=1).indices
                        preds_k = preds.gather(1, top_idx[:, :, None, None].expand(-1, -1, *preds.shape[2:]))
                    else:
                        preds_k = preds[:, :k_out]
                    step_dists = torch.linalg.vector_norm(preds_k.float() - b["future"].float().unsqueeze(1), dim=-1)
                    per_sample_ade = step_dists.mean(dim=-1).amin(dim=-1).cpu().tolist()
                    per_sample_fde = step_dists[:, :, -1].amin(dim=-1).cpu().tolist()
                    per_sample_mr = (step_dists.amax(dim=-1).amin(dim=-1) >= 2.0).float().cpu().tolist()
                    for ind, ade_v, fde_v, mr_v in zip(indicators, per_sample_ade, per_sample_fde, per_sample_mr):
                        ind["ade"] = float(ade_v)
                        ind["fde"] = float(fde_v)
                        ind["miss"] = float(mr_v)
                sample_records.extend(indicators)

        if model is not None:
            _cleanup_torch_resources(model)

        scores_sorted = sorted(r["complexity_score"] for r in sample_records)
        q33 = _quantile(scores_sorted, 1.0 / 3.0)
        q67 = _quantile(scores_sorted, 2.0 / 3.0)
        neigh_sorted = sorted(float(r["num_neighbors"]) for r in sample_records)
        neigh_q50 = _quantile(neigh_sorted, 0.50)
        lane_sorted = sorted(float(r["num_valid_lanes"]) for r in sample_records)
        lane_q67 = _quantile(lane_sorted, 2.0 / 3.0)

        def _agg_group(group_name: str, subset: list[dict[str, Any]], criterion: str) -> dict[str, Any]:
            n = len(subset)
            ades = [x["ade"] for x in subset if "ade" in x]
            fdes = [x["fde"] for x in subset if "fde" in x]
            mrs = [x["miss"] for x in subset if "miss" in x]
            return {
                "group_name": group_name,
                "criterion": criterion,
                "model_state": model_state_label,
                "num_samples": n,
                "sample_share_percent": round((n / max(1, len(sample_records))) * 100.0, 2),
                "k_modes": k_eval,
                "min_ade_k": round(sum(ades) / len(ades), 4) if ades else None,
                "min_fde_k": round(sum(fdes) / len(fdes), 4) if fdes else None,
                "miss_rate_2m_percent": round((sum(mrs) / len(mrs)) * 100.0, 2) if mrs else None,
                "avg_neighbors": round(sum(x["num_neighbors"] for x in subset) / max(1, n), 3),
                "avg_close_agents_20m": round(sum(x["close_agents_20m"] for x in subset) / max(1, n), 3),
                "avg_valid_lanes": round(sum(x["num_valid_lanes"] for x in subset) / max(1, n), 3),
                "avg_static_obstacles": round(sum(x["num_static_obstacles"] for x in subset) / max(1, n), 3),
                "avg_complexity_score": round(sum(x["complexity_score"] for x in subset) / max(1, n), 4),
            }

        strata: list[tuple[str, list[dict[str, Any]], str]] = [
            ("complexity_low", [r for r in sample_records if r["complexity_score"] <= q33], f"complexity_score <= Q33 ({q33:.4f})"),
            ("complexity_medium", [r for r in sample_records if q33 < r["complexity_score"] <= q67], f"Q33 ({q33:.4f}) < score <= Q67 ({q67:.4f})"),
            ("complexity_high", [r for r in sample_records if r["complexity_score"] > q67], f"complexity_score > Q67 ({q67:.4f})"),
            ("density_low", [r for r in sample_records if r["num_neighbors"] <= neigh_q50], f"num_neighbors <= median ({neigh_q50:.1f})"),
            ("density_high", [r for r in sample_records if r["num_neighbors"] > neigh_q50], f"num_neighbors > median ({neigh_q50:.1f})"),
            ("map_zero_coverage", [r for r in sample_records if not r["has_map"]], "num_valid_lanes == 0"),
            ("map_rich", [r for r in sample_records if r["num_valid_lanes"] >= max(1.0, lane_q67)], f"num_valid_lanes >= Q67 ({lane_q67:.1f})"),
            ("intersection_proxy", [r for r in sample_records if r["has_intersection_proxy"] is True], "map_lane_types contains lane_connector(1) or ped_crossing(3)"),
            ("non_intersection_proxy", [r for r in sample_records if r["has_intersection_proxy"] is False], "no lane_connector/ped_crossing in valid lanes"),
            ("turning_motion_proxy", [r for r in sample_records if r["ego_turn_rad"] > 0.15], "ego cumulative past turn > 0.15 rad"),
            ("straight_motion_proxy", [r for r in sample_records if r["ego_turn_rad"] <= 0.15], "ego cumulative past turn <= 0.15 rad"),
        ]

        rows = [_agg_group(name, sub, crit) for name, sub, crit in strata]
        prior_proto = (prior_result or {}).get("experimental_scene_conditioned_prototype")
        prototype_result: dict[str, Any] = copy.deepcopy(prior_proto) if isinstance(prior_proto, dict) and prior_proto.get("status") == "completed" else {
            "experimental": True,
            "status": "not_run",
            "note": (
                "Experimental SceneConditionedArchitecturePrototype implemented in research_gap_evaluation.py. "
                "Pass '--run-scene-prototype' with a ready dataset to train and compare against the fixed model."
            ),
            "val_min_ade_k": None,
            "val_min_fde_k": None,
            "val_miss_rate_2m_percent": None,
        }
        scene_out: dict[str, Any] = {
            "experiment": "scene",
            "status": "completed" if can_evaluate_model else "partial_stratification_only",
            "model_state": model_state_label,
            "checkpoint_used": str(ckpt_resolved) if ckpt_resolved else None,
            "total_val_samples": len(sample_records),
            "quantiles_used": {"q33": q33, "q67": q67, "neighbors_q50": neigh_q50, "lanes_q67": lane_q67},
            "unavailable_dataset_fields": unavailable_features,
            "rows": rows,
            "experimental_scene_conditioned_prototype": prototype_result,
        }
        if on_stage_complete is not None:
            on_stage_complete("SCENE", scene_out)
    else:
        scene_out = copy.deepcopy(prior_result) if prior_result else {
            "experiment": "scene",
            "status": "partial_stratification_only",
            "unavailable_dataset_fields": unavailable_features,
            "rows": [],
            "experimental_scene_conditioned_prototype": {},
        }
        prototype_result = copy.deepcopy(scene_out.get("experimental_scene_conditioned_prototype") or {
            "experimental": True,
            "status": "not_run",
            "val_min_ade_k": None,
            "val_min_fde_k": None,
            "val_miss_rate_2m_percent": None,
        })

    if active_prototype:
        if on_stage_start is not None:
            on_stage_start("SCENE_PROTOTYPE")
        proto = None
        try:
            set_seed(int(args.seed))
            proto = ExperimentalSceneConditionedPrototype(model_cfg=model_cfg, search_cfg=search_cfg).to(device)
            opt = torch.optim.Adam(proto.parameters(), lr=7.5e-4, weight_decay=1e-4)
            proto_epochs = max(1, min(int(args.final_epochs or 5), 10))
            proto.train()
            t_p = time.perf_counter()
            for _ep in range(proto_epochs):
                for batch in train_loader:
                    b = unpack_batch_to_device(batch, device)
                    opt.zero_grad()
                    p_trajs, p_logits, _ = proto(b, device)
                    l_reg, l_cls = best_of_k_loss(p_trajs, b["future"], p_logits)
                    (l_reg + 0.1 * l_cls).backward()
                    opt.step()
            proto_dur = round(time.perf_counter() - t_p, 3)

            proto.eval()
            tot_n, s_ade, s_fde, s_mr = 0, 0.0, 0.0, 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    b = unpack_batch_to_device(batch, device)
                    p_trajs, p_logits, _ = proto(b, device)
                    bsz = int(b["past"].shape[0])
                    tot_n += bsz
                    s_ade += min_ade_k(p_trajs, b["future"], k=k_eval, mode_logits=p_logits).item() * bsz
                    s_fde += min_fde_k(p_trajs, b["future"], k=k_eval, mode_logits=p_logits).item() * bsz
                    s_mr += miss_rate_k(p_trajs, b["future"], k=k_eval, threshold=2.0, mode_logits=p_logits) * bsz
            n = max(1, tot_n)
            prototype_result.update({
                "status": "completed",
                "epochs_trained": proto_epochs,
                "duration_s": proto_dur,
                "val_min_ade_k": round(s_ade / n, 4),
                "val_min_fde_k": round(s_fde / n, 4),
                "val_miss_rate_2m_percent": round((s_mr / n) * 100.0, 2),
                "note": "Experimental prototype trained & evaluated on the identical val split.",
            })
        except Exception as exc:
            prototype_result.update({"status": "failed", "error": str(exc)})
        finally:
            _cleanup_torch_resources(proto)
        scene_out["experimental_scene_conditioned_prototype"] = prototype_result
        if on_stage_complete is not None:
            on_stage_complete("SCENE_PROTOTYPE", scene_out)

    return scene_out


# ------------------------------------------------------------------------------
# Section 6: Warmup & Temperature Annealing Ablation (T0-T3)
# ------------------------------------------------------------------------------

def _build_warmup_temp_summary(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") == "failed"]
    overall_status = "completed" if completed and not failed else ("partial" if completed else ("failed" if failed else "not_run"))
    return {
        "experiment": "warmup_temperature",
        "status": overall_status,
        "preliminary_single_seed": len(seeds) < 3,
        "seeds": seeds,
        "rows": rows,
    }


def run_warmup_temperature_ablation(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    output_dir: Path,
    prior_results: list[dict[str, Any]] | None = None,
    enabled_stage_ids: set[str] | None = None,
    force_rerun_ids: set[str] | None = None,
    skip_completed: bool = True,
    continue_on_error: bool = True,
    on_stage_start: Callable[[str], None] | None = None,
    on_stage_complete: Callable[[str, list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ref = load_reference_architecture()
    ref_idx = tuple(int(x) for x in ref["indices"])
    s_cfg_default = SearchConfig()
    seeds = list(args.seeds)
    ckpt_dir = output_dir / "checkpoints" / "warmup_temperature"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prior_map: dict[tuple[str, int], dict[str, Any]] = {}
    if prior_results:
        for row in prior_results:
            if row.get("status") == "completed" and row.get("config_id") and row.get("seed") is not None:
                prior_map[(str(row["config_id"]), int(row["seed"]))] = row

    rows: list[dict[str, Any]] = []
    executed_count = 0
    halted = False

    for spec in WARMUP_TEMP_MATRIX:
        cid = spec["id"]
        is_enabled = True if enabled_stage_ids is None else (cid in enabled_stage_ids)
        is_forced = False if force_rerun_ids is None else (cid in force_rerun_ids)

        if not is_enabled or halted:
            for seed in seeds:
                if (cid, seed) in prior_map:
                    rows.append(prior_map[(cid, seed)])
            continue

        if on_stage_start is not None:
            on_stage_start(cid)

        active_warmup = (int(args.warmup_epochs) if args.warmup_epochs is not None else int(s_cfg_default.warmup_epochs)) if spec["warmup_enabled"] else 0
        stage_rows: list[dict[str, Any]] = []
        stage_failed = False

        for seed in seeds:
            if args.max_experiments is not None and executed_count >= args.max_experiments:
                r_skip = {
                    "config_id": cid,
                    "seed": seed,
                    "warmup_enabled": spec["warmup_enabled"],
                    "temp_annealing_enabled": spec["temp_annealing_enabled"],
                    "warmup_epochs": active_warmup,
                    "status": "not_run",
                    "error": f"Skipped due to --max-experiments={args.max_experiments}",
                }
                rows.append(r_skip)
                stage_rows.append(r_skip)
                continue

            if not is_forced and skip_completed and (cid, seed) in prior_map:
                r_prev = prior_map[(cid, seed)]
                rows.append(r_prev)
                stage_rows.append(r_prev)
                continue

            executed_count += 1
            try:
                res = _execute_search_and_final_train(
                    run_id=f"wt_{cid}",
                    args=args,
                    model_cfg=model_cfg,
                    seed=seed,
                    checkpoint_dir=ckpt_dir,
                    branch_modes={"temporal": "Search", "interaction": "Search", "map": "Search"},
                    fixed_indices=ref_idx,
                    enable_temp_annealing=spec["temp_annealing_enabled"],
                    warmup_epochs=active_warmup,
                )
            except Exception as exc:
                res = {
                    "status": "failed",
                    "run_id": f"wt_{cid}",
                    "seed": seed,
                    "error": f"[{type(exc).__name__}] {exc}",
                    "traceback": traceback.format_exc(),
                }

            res.update({
                "config_id": cid,
                "warmup_enabled": spec["warmup_enabled"],
                "temp_annealing_enabled": spec["temp_annealing_enabled"],
                "preliminary_single_seed": len(seeds) < 3,
            })
            if res.get("status") != "completed":
                stage_failed = True
            rows.append(res)
            stage_rows.append(res)

        if on_stage_complete is not None:
            on_stage_complete(cid, stage_rows, _build_warmup_temp_summary(list(rows), seeds))

        if stage_failed and not continue_on_error:
            halted = True

    return _build_warmup_temp_summary(rows, seeds)


# ------------------------------------------------------------------------------
# Section 8: Matplotlib Visualization (Strictly from Saved Results)
# ------------------------------------------------------------------------------

def generate_all_plots(
    output_dir: Path,
    results: dict[str, Any],
) -> list[str]:
    """Generates Matplotlib charts exclusively from actual non-NaN results in results dict."""
    if not HAS_MATPLOTLIB:
        return []

    fig_dir = output_dir / "figures"
    generated: list[str] = []

    def _save(fig: Any, filename: str) -> None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_p = fig_dir / filename
        fig.savefig(out_p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(out_p.relative_to(output_dir)))

    # 1. Factorized Ablation Performance & Peak Memory + Multi-Seed Operator Frequency
    fac = results.get("factorized")
    if fac and fac.get("rows"):
        comp_rows = [r for r in fac["rows"] if r.get("status") == "completed" and _safe_float(r.get("final_val_min_ade")) is not None]
        if comp_rows:
            ids = [spec["id"] for spec in FACTORIZED_ABLATION_MATRIX]
            by_id: dict[str, list[dict[str, Any]]] = {cid: [] for cid in ids}
            for r in comp_rows:
                by_id.setdefault(str(r.get("config_id")), []).append(r)

            present_ids = [cid for cid in ids if by_id.get(cid)]
            ades_mean = [sum(float(x["final_val_min_ade"]) for x in by_id[cid]) / len(by_id[cid]) for cid in present_ids]
            fdes_mean = [sum(float(x["final_val_min_fde"]) for x in by_id[cid]) / len(by_id[cid]) for cid in present_ids]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            xs = list(range(len(present_ids)))
            ax.bar([x - 0.18 for x in xs], ades_mean, width=0.35, label="Final minADE@K (m)", color="#2b6cb0")
            ax.bar([x + 0.18 for x in xs], fdes_mean, width=0.35, label="Final minFDE@K (m)", color="#dd6b20")
            ax.set_xticks(xs)
            ax.set_xticklabels(present_ids)
            ax.set_ylabel("Error (meters)")
            ax.set_title("Factorized Three-Branch NAS Ablation (A0-A7)")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            _save(fig, "factorized_ablation_performance.png")

            # GPU memory comparison if recorded
            mem_ids = [cid for cid in present_ids if any(_safe_float(x.get("peak_gpu_memory_mb")) is not None for x in by_id[cid])]
            if mem_ids:
                mems = [
                    sum(float(x["peak_gpu_memory_mb"]) for x in by_id[cid] if x.get("peak_gpu_memory_mb") is not None)
                    / max(1, sum(1 for x in by_id[cid] if x.get("peak_gpu_memory_mb") is not None))
                    for cid in mem_ids
                ]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(mem_ids, mems, color="#319795")
                ax.set_ylabel("Peak GPU Memory (MB)")
                ax.set_title("Peak CUDA Memory Across Branch Ablations")
                ax.grid(axis="y", alpha=0.3)
                _save(fig, "gpu_memory_comparison.png")

            # Operator selection frequency across seeds if >1 unique seed
            unique_seeds = {r.get("seed") for r in comp_rows}
            if len(unique_seeds) > 1:
                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                for ax, key, title in zip(
                    axes,
                    ("selected_temporal", "selected_interaction", "selected_map"),
                    ("Temporal Operators", "Interaction Operators", "HD Map Operators"),
                ):
                    counts: dict[str, int] = {}
                    for r in comp_rows:
                        op_n = str(r.get(key) or "N/A")
                        counts[op_n] = counts.get(op_n, 0) + 1
                    ax.bar(list(counts.keys()), list(counts.values()), color="#4a5568")
                    ax.set_title(title)
                    ax.tick_params(axis="x", rotation=25)
                fig.suptitle("Operator Selection Frequency Across Seeds & Ablations")
                fig.tight_layout()
                _save(fig, "operator_selection_frequency.png")

    # 2. Hardware Profiling: FLOPs vs minADE, Latency vs minADE, GPU Memory
    hw = results.get("hardware")
    if hw and hw.get("rows"):
        hw_valid = [
            r for r in hw["rows"]
            if r.get("status") == "completed" and _safe_float(r.get("final_val_min_ade")) is not None
        ]
        if hw_valid:
            flops_pts = [(float(r["theoretical_flops_g"]), float(r["final_val_min_ade"]), r["candidate_id"]) for r in hw_valid if _safe_float(r.get("theoretical_flops_g")) is not None]
            if flops_pts:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for fx, fy, lbl in flops_pts:
                    ax.scatter([fx], [fy], s=70)
                    ax.annotate(lbl, (fx, fy), xytext=(5, 4), textcoords="offset points", fontsize=8)
                ax.set_xlabel("Theoretical FLOPs (GFLOPs)")
                ax.set_ylabel("Validation minADE@K (m)")
                ax.set_title("Theoretical FLOPs vs. Prediction Error")
                ax.grid(True, alpha=0.3)
                _save(fig, "flops_vs_minade.png")

            lat_pts = [
                (float(r.get("gpu_latency_ms_mean") or r["cpu_latency_ms_mean"]), float(r["final_val_min_ade"]), r["candidate_id"])
                for r in hw_valid
                if _safe_float(r.get("gpu_latency_ms_mean") or r.get("cpu_latency_ms_mean")) is not None
            ]
            if lat_pts:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for lx, ly, lbl in lat_pts:
                    ax.scatter([lx], [ly], s=70, color="#c53030")
                    ax.annotate(lbl, (lx, ly), xytext=(5, 4), textcoords="offset points", fontsize=8)
                ax.set_xlabel("Measured Inference Latency (ms)")
                ax.set_ylabel("Validation minADE@K (m)")
                ax.set_title("Measured Inference Latency vs. Prediction Error")
                ax.grid(True, alpha=0.3)
                _save(fig, "latency_vs_minade.png")

    # 3. Scene Complexity vs Prediction Error
    sc = results.get("scene")
    if sc and sc.get("rows"):
        sc_valid = [r for r in sc["rows"] if _safe_float(r.get("min_ade_k")) is not None]
        if sc_valid:
            names = [r["group_name"] for r in sc_valid]
            ades = [float(r["min_ade_k"]) for r in sc_valid]
            fdes = [float(r["min_fde_k"]) for r in sc_valid]
            fig, ax = plt.subplots(figsize=(11, 4.8))
            xs = list(range(len(names)))
            ax.bar([x - 0.18 for x in xs], ades, width=0.35, label="minADE@K (m)", color="#2f855a")
            ax.bar([x + 0.18 for x in xs], fdes, width=0.35, label="minFDE@K (m)", color="#b7791f")
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel("Error (meters)")
            ax.set_title("Forecasting Error Across Scene Complexity & Density Strata")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            _save(fig, "scene_complexity_vs_error.png")

    # 4. Warmup & Temperature Annealing Ablation (T0-T3)
    wt = results.get("warmup_temperature")
    if wt and wt.get("rows"):
        wt_valid = [r for r in wt["rows"] if r.get("status") == "completed" and _safe_float(r.get("final_val_min_ade")) is not None]
        if wt_valid:
            ids = [spec["id"] for spec in WARMUP_TEMP_MATRIX]
            by_id = {cid: [r for r in wt_valid if r.get("config_id") == cid] for cid in ids}
            present = [cid for cid in ids if by_id[cid]]
            ades = [sum(float(x["final_val_min_ade"]) for x in by_id[cid]) / len(by_id[cid]) for cid in present]
            fdes = [sum(float(x["final_val_min_fde"]) for x in by_id[cid]) / len(by_id[cid]) for cid in present]
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            xs = list(range(len(present)))
            ax.bar([x - 0.18 for x in xs], ades, width=0.35, label="Final minADE@K (m)", color="#6b46c1")
            ax.bar([x + 0.18 for x in xs], fdes, width=0.35, label="Final minFDE@K (m)", color="#d53f8c")
            ax.set_xticks(xs)
            ax.set_xticklabels(present)
            ax.set_ylabel("Error (meters)")
            ax.set_title("Warmup & Temperature Annealing Ablation (T0-T3)")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            _save(fig, "warmup_temperature_ablation.png")

    return generated


# ------------------------------------------------------------------------------
# Section 8 & 9: Markdown Research Gap Report Generator (All 12 Sections)
# ------------------------------------------------------------------------------

def _fmt_val(val: Any, suffix: str = "") -> str:
    if val is None:
        return "NaN / Pending"
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return "NaN / Pending"
        return f"{val:.4g}{suffix}"
    return f"{val}{suffix}"


def build_research_gap_report(
    output_dir: Path,
    config_payload: dict[str, Any],
    results: dict[str, Any],
    generated_figures: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# TrajectoryLNAS Research Gap Evaluation Report\n")
    lines.append(f"- **Generated (UTC)**: `{_utc_iso()}`")
    lines.append(f"- **Output Directory**: `{output_dir}`")
    lines.append(f"- **Requested Experiment Suite**: `{config_payload['cli_args']['experiment']}`")
    lines.append(f"- **Dataset**: `{config_payload['cli_args']['dataset']}` | **Seeds**: `{config_payload['cli_args']['seeds']}`")
    hw_env = config_payload["hardware_environment"]
    lines.append(f"- **Execution Environment**: PyTorch `{hw_env.get('torch_version') or 'Not installed'}` | CUDA Available: `{hw_env.get('cuda_available')}` | Device: `{hw_env.get('gpu_model') or hw_env.get('cpu')}`\n")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary\n")
    lines.append("| Component | Status | Scientific Classification | Summary |")
    lines.append("| --- | --- | --- | --- |")
    for exp_key, label in (
        ("audit", "Data Integrity & Reliability Audit"),
        ("factorized", "Gap A: Factorized Three-Branch NAS (A0-A7)"),
        ("hardware", "Gap B: Hardware-Aware NAS Profiling"),
        ("scene", "Gap C: Scene-Adaptive Evaluation"),
        ("warmup_temperature", "Warmup & Temperature Annealing (T0-T3)"),
    ):
        sec = results.get(exp_key)
        if not sec:
            lines.append(f"| {label} | `not_run` | Implemented (Not Executed) | Not selected in this invocation |")
        else:
            st = sec.get("status", "unknown")
            cls_tag = "Executed & Measured" if st == "completed" else ("Partially Executed" if "partial" in str(st) else "Implemented / Pending Prerequisites")
            err = sec.get("error") or ("; ".join(sec.get("warnings", [])[:2]) if isinstance(sec.get("warnings"), list) else "") or "See section details"
            lines.append(f"| {label} | `{st}` | {cls_tag} | {err} |")
    lines.append("")

    # 2. Dataset and Data Integrity Audit
    lines.append("## 2. Dataset and Data Integrity Audit\n")
    audit = results.get("audit")
    if not audit:
        lines.append("_Data audit was not executed in this run (`not_run`)._\n")
    else:
        ds_paths = audit.get("dataset_paths", {})
        lines.append(f"- **Audit Status**: `{audit.get('status')}` | **Safe for Comparable Experiments**: `{audit.get('safe_to_compare_experiments')}`")
        lines.append(f"- **Configured Dataroot**: `{ds_paths.get('dataroot_configured')}`")
        lines.append(f"- **Resolved Sequence Dir**: `{ds_paths.get('resolved_data_dir')}` (Ready: `{ds_paths.get('dataset_ready')}`)")
        k_aud = audit.get("metric_and_mode_audit", {})
        lines.append(
            f"- **Horizon & Sampling Grid**: past=`{k_aud.get('model_config_past_len')}` steps (`{k_aud.get('model_config_past_sec')}s`), "
            f"future=`{k_aud.get('model_config_future_len')}` steps (`{k_aud.get('model_config_future_sec')}s`), "
            f"hz=`{k_aud.get('model_config_hz')}`, K=`{k_aud.get('model_config_num_modes')}` "
            f"(verified output K=`{k_aud.get('model_output_k_verified')}`)"
        )
        lines.append(f"- **DARTS K Observation**: {k_aud.get('darts_validate_k_hardcoded_note')}")
        leak = audit.get("scene_leakage", {})
        lines.append(f"- **Scene-Level Split Separation**: `{leak.get('status')}` (train scenes=`{leak.get('train_unique_scenes')}`, val scenes=`{leak.get('val_unique_scenes')}`, shared=`{leak.get('shared_scenes_count')}`)")
        for sp_name, sp_info in (audit.get("splits") or {}).items():
            if isinstance(sp_info, dict) and sp_info.get("exists"):
                lines.append(
                    f"- **{sp_name.upper()} Split**: {sp_info.get('sample_count', 0):,} samples | "
                    f"NaN/Inf=`{sp_info.get('nan_inf_samples')}` | Zero-Neighbor=`{sp_info.get('zero_neighbor_samples')}` ({sp_info.get('zero_neighbor_percent')}%) | "
                    f"**Zero-Coverage HD Map=`{sp_info.get('zero_coverage_hd_map_samples')}` ({sp_info.get('zero_coverage_hd_map_percent')}%)**"
                )
        for w in audit.get("warnings", []):
            lines.append(f"- **Warning**: {w}")
        for b in audit.get("blocking_issues", []):
            lines.append(f"- **Blocking Issue**: {b}")
        lines.append("")

    # 3. Factorized NAS Ablation Results (Gap A)
    lines.append("## 3. Factorized NAS Ablation Results (Research Gap A)\n")
    fac = results.get("factorized")
    if not fac:
        lines.append("_Factorized ablation was not executed in this run (`not_run`)._\n")
    else:
        if fac.get("preliminary_single_seed"):
            lines.append("> **Notice**: Executed with fewer than 3 seeds; mark any observed differences as **preliminary** until >=3 seeds complete.\n")
        lines.append("| Config | Temporal | Interaction | HD Map | Seed | Status | Search Val minADE | Final minADE@K | Final minFDE@K | MR@2m (%) | Selected Ops | Params | FLOPs (G) | Search (s) | Train (s) | Peak VRAM (MB) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in fac.get("rows", []):
            ops_str = f"[{r.get('selected_temporal')}, {r.get('selected_interaction')}, {r.get('selected_map')}]" if r.get("selected_temporal") else "NaN"
            lines.append(
                f"| {r.get('config_id')} | {r.get('temporal_mode')} | {r.get('interaction_mode')} | {r.get('map_mode')} | "
                f"{r.get('seed')} | `{r.get('status')}` | {_fmt_val(r.get('search_supernet_val_ade'))} | {_fmt_val(r.get('final_val_min_ade'))} | "
                f"{_fmt_val(r.get('final_val_min_fde'))} | {_fmt_val(r.get('final_val_miss_rate_2m'))} | {ops_str} | "
                f"{_fmt_val(r.get('parameters'))} | {_fmt_val(r.get('theoretical_flops_g'))} | "
                f"{_fmt_val(r.get('search_duration_s'))} | {_fmt_val(r.get('final_training_duration_s'))} | {_fmt_val(r.get('peak_gpu_memory_mb'))} |"
            )
        lines.append("")

    # 4. Hardware Efficiency Results (Gap B)
    lines.append("## 4. Hardware Efficiency Results (Research Gap B)\n")
    hw = results.get("hardware")
    if not hw:
        lines.append("_Hardware profiling was not executed in this run (`not_run`)._\n")
    else:
        lines.append("| Candidate | Category | Lambda FLOPs | Status | Accuracy Status | Params | FLOPs (G) | GPU Latency Mean/P95 (ms) | CPU Latency Mean (ms) | Peak VRAM (MB) | Ckpt Size (MB) | Final minADE@K |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in hw.get("rows", []):
            gpu_lat = f"{_fmt_val(r.get('gpu_latency_ms_mean'))} / {_fmt_val(r.get('gpu_latency_ms_p95'))}"
            lines.append(
                f"| {r.get('candidate_id')} | {r.get('category')} | {_fmt_val(r.get('lambda_flops'))} | `{r.get('status')}` | "
                f"`{r.get('accuracy_status', 'N/A')}` | {_fmt_val(r.get('parameters'))} | {_fmt_val(r.get('theoretical_flops_g'))} | "
                f"{gpu_lat} | {_fmt_val(r.get('cpu_latency_ms_mean'))} | {_fmt_val(r.get('peak_gpu_memory_mb'))} | "
                f"{_fmt_val(r.get('checkpoint_size_mb'))} | {_fmt_val(r.get('final_val_min_ade'))} |"
            )
        lines.append("")

    # 5. Scene Complexity Analysis (Gap C)
    lines.append("## 5. Scene Complexity Analysis (Research Gap C)\n")
    sc = results.get("scene")
    if not sc:
        lines.append("_Scene complexity evaluation was not executed in this run (`not_run`)._\n")
    else:
        lines.append(f"- **Model Evaluation State**: `{sc.get('model_state')}` | **Unavailable Fields**: `{'; '.join(sc.get('unavailable_dataset_fields', []))}`\n")
        if sc.get("rows"):
            lines.append("| Group | Criterion | Samples | Share (%) | minADE@K | minFDE@K | MR@2m (%) | Avg Neighbors | Avg Valid Lanes | Avg Complexity Score |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in sc["rows"]:
                lines.append(
                    f"| {r.get('group_name')} | {r.get('criterion')} | {r.get('num_samples')} | {r.get('sample_share_percent')}% | "
                    f"{_fmt_val(r.get('min_ade_k'))} | {_fmt_val(r.get('min_fde_k'))} | {_fmt_val(r.get('miss_rate_2m_percent'))} | "
                    f"{_fmt_val(r.get('avg_neighbors'))} | {_fmt_val(r.get('avg_valid_lanes'))} | {_fmt_val(r.get('avg_complexity_score'))} |"
                )
        proto = sc.get("experimental_scene_conditioned_prototype", {})
        lines.append(f"\n- **Experimental Scene-Conditioned Prototype**: status=`{proto.get('status')}`, experimental=`{proto.get('experimental', True)}`, minADE@K=`{_fmt_val(proto.get('val_min_ade_k'))}`, minFDE@K=`{_fmt_val(proto.get('val_min_fde_k'))}` — {proto.get('note', '')}\n")

    # 6. Warmup & Temperature Annealing Ablation
    lines.append("## 6. Warmup and Temperature Annealing Ablation\n")
    wt = results.get("warmup_temperature")
    if not wt:
        lines.append("_Warmup & temperature ablation was not executed in this run (`not_run`)._\n")
    else:
        lines.append("| Config | Warmup | Temp Annealing | Seed | Status | Search Val minADE | Final minADE@K | Final minFDE@K | Selected Ops | Search (s) | Argmax Flips |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in wt.get("rows", []):
            ops_str = f"[{r.get('selected_temporal')}, {r.get('selected_interaction')}, {r.get('selected_map')}]" if r.get("selected_temporal") else "NaN"
            flips = (r.get("stability") or {}).get("argmax_flips_total")
            lines.append(
                f"| {r.get('config_id')} | {r.get('warmup_enabled')} ({r.get('warmup_epochs')} ep) | {r.get('temp_annealing_enabled')} | "
                f"{r.get('seed')} | `{r.get('status')}` | {_fmt_val(r.get('search_supernet_val_ade'))} | {_fmt_val(r.get('final_val_min_ade'))} | "
                f"{_fmt_val(r.get('final_val_min_fde'))} | {ops_str} | {_fmt_val(r.get('search_duration_s'))} | {_fmt_val(flips)} |"
            )
        lines.append("")

    # 7. Comparison Tables (Including Literature Comparison Table)
    lines.append("## 7. Literature Comparison Table (Protocol-Aware Audit)\n")
    lines.append("> **Protocol Disclaimer**: Published literature values below come from heterogeneous datasets, horizons, prediction mode counts (K), and metric definitions. Do NOT treat cross-protocol numbers as direct numerical rankings. TrajectoryLNAS figures in this report reflect only measurements verified in this execution.\n")
    lines.append("| Model | Architecture / NAS Paradigm | Temporal Modeling | Interaction Modeling | HD-Map Modeling | NAS Strategy | Hardware Objective | Scene-Adaptive | Dataset & Evaluation Protocol | Published / Newly Measured Metrics |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for lm in LITERATURE_MODELS_SPEC:
        metrics_cell = lm["published_metrics"]
        if lm["model"].startswith("TrajectoryLNAS"):
            hw_rows = [r for r in (results.get("hardware", {}).get("rows") or []) if r.get("candidate_id") == "trajectory_lnas_reference" and r.get("final_val_min_ade") is not None]
            if hw_rows:
                r0 = hw_rows[0]
                metrics_cell = (
                    f"Newly measured (K={config_payload['model_config']['num_modes']}): "
                    f"minADE={r0.get('final_val_min_ade')}m, minFDE={r0.get('final_val_min_fde')}m, "
                    f"MR@2m={r0.get('final_val_miss_rate_2m')}%, params={r0.get('parameters')}, FLOPs={r0.get('theoretical_flops_g')}G"
                )
            else:
                metrics_cell = "Pending trained TrajectoryLNAS checkpoint evaluation on extracted nuScenes validation set"
        lines.append(
            f"| {lm['model']} | {lm['paradigm']} | {lm['temporal']} | {lm['interaction']} | "
            f"{lm['map']} | {lm['nas_strategy']} | {lm['hardware_objective']} | {lm['scene_adaptive']} | "
            f"{lm['dataset_protocol']} | {metrics_cell} |"
        )
    lines.append("")

    # 8. Figures and Charts
    lines.append("## 8. Figures and Charts\n")
    if generated_figures:
        for fig_rel in generated_figures:
            lines.append(f"- `{fig_rel}`")
    else:
        lines.append("_No charts generated (requires Matplotlib and at least one completed numerical measurement series)._")
    lines.append("")

    # 9. Failed or Incomplete Experiments
    lines.append("## 9. Failed or Incomplete Experiments\n")
    incomplete_items: list[str] = []
    for k, sec in results.items():
        if isinstance(sec, dict) and isinstance(sec.get("rows"), list):
            for r in sec["rows"]:
                if r.get("status") in ("failed", "not_run", "pending"):
                    incomplete_items.append(f"- `{k}` / `{r.get('config_id') or r.get('candidate_id') or 'item'}` (status=`{r.get('status')}`): {r.get('error')}")
        elif isinstance(sec, dict) and sec.get("status") != "completed":
            incomplete_items.append(f"- `{k}` (status=`{sec.get('status')}`): {sec.get('error') or 'See warnings/blocking issues'}")
    lines.extend(incomplete_items if incomplete_items else ["- None."])
    lines.append("")

    # 10. Limitations and Threats to Validity
    lines.append("## 10. Limitations and Threats to Validity\n")
    lines.append("- **No Future Neighbor Ground Truth**: `neighbors_future` is absent from preprocessed sequences; physical Collision Rate cannot be evaluated reliably.")
    lines.append("- **No Explicit Lane-Change Annotations**: Gap C intersection and turning slices rely on `map_lane_types` (`lane_connector`/`ped_crossing`) and ego trajectory curvature proxies.")
    lines.append("- **Search-to-Eval Gap**: DARTS SuperNet validation loss reflects continuous mixed-op weights, whereas paper claims must be grounded in discrete standalone `FinalTrajectoryModel` metrics.")
    lines.append("- **Protocol Differences Across Literature**: Argoverse 3s K=6 and LiDAR APf benchmarks are not numerically comparable to nuScenes 6s K=5.\n")

    # 11. Conclusions Supported by Actual Measurements
    lines.append("## 11. Conclusions Supported by Actual Measurements\n")
    lines.append("- Only claims backed by `completed` runs in Sections 2-6 above are empirically verified; `pending` or `not_run` hypotheses remain **unverified** until executed on real GPU hardware with extracted nuScenes sequences.\n")

    # 12. Recommended Next Experiments
    lines.append("## 12. Recommended Next Experiments\n")
    lines.append("1. Verify dataset sequences exist (`python python_src/data/preprocessing.py --dataroot <path_to_nuscenes>`) and run `--experiment audit`.")
    lines.append("2. Train reference TrajectoryLNAS checkpoint (`python python_src/main.py`) and run `--experiment hardware` & `--experiment scene`.")
    lines.append("3. Execute multi-seed (`--seeds 42 123 456`) runs for `--experiment factorized` and `--experiment warmup_temperature` on GPU.")
    lines.append("4. Run `--experiment hardware --run-lambda-search` and `--experiment scene --run-scene-prototype` for full empirical Pareto & adaptive curves.\n")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# CLI & Orchestration
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrajectoryLNAS Independent Research Gaps Evaluation & Experiment Runner"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["all", "factorized", "hardware", "scene", "warmup_temperature", "audit"],
        help="Optional experiment suite filter (default: controlled by configs/evaluate_config.py)",
    )
    parser.add_argument("--eval-config", type=str, default=None, help="Optional path to custom evaluate_config.py module")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "synthetic"], help="Target dataset")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default=None, help="Override path to processed sequence JSONL/JSON dir")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_trajectory_lnas.pt", help="Trained FinalTrajectoryModel checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed (default: 42)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional multi-seed list (e.g., --seeds 42 123 456)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cuda, or cpu")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None, help="Physical batch size override")
    parser.add_argument("--grad-accum-steps", "--grad_accum", dest="grad_accum_steps", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--search-epochs", type=int, default=None, help="DARTS search epochs override")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="DARTS warmup epochs override")
    parser.add_argument("--final-epochs", type=int, default=None, help="Final standalone training epochs override")
    parser.add_argument("--search-samples-percent", type=float, default=None, help="Search subsample percentage override")
    parser.add_argument("--train-samples-percent", type=float, default=None, help="Train subsample percentage override")
    parser.add_argument("--lambda-flops-values", type=float, nargs="+", default=None, help="Configurable lambda_flops values for Gap B")
    parser.add_argument("--run-lambda-search", action="store_true", help="Actually execute DARTS search + training for each lambda_flops value in Gap B")
    parser.add_argument("--run-scene-prototype", action="store_true", help="Train and evaluate experimental SceneConditionedArchitecturePrototype in Gap C")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerunning explicitly enabled evaluations even if previously completed")
    parser.add_argument("--hw-warmup", type=int, default=25, help="Hardware profiler warmup iterations")
    parser.add_argument("--hw-runs", type=int, default=100, help="Hardware profiler timed iterations")
    parser.add_argument("--max-experiments", type=int, default=None, help="Maximum number of individual sub-experiments to execute per section")
    parser.add_argument("--resume", type=str, default=None, help="Existing python_src/reports/research_gaps/<dir> path to resume")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory override (relative paths resolve under python_src/; defaults to evaluate_config.OUTPUT_DIR or timestamped dir)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to fit 4GB VRAM GPUs")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP FP16 mixed precision")
    parser.add_argument("--allow-untrained", action="store_true", help="Allow smoke-test forward evaluation without trained .pt checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs, paths, and planned runs without training or writing full outputs")
    return parser


def _rel_output_paths(output_dir: Path, paths: list[Path | str | None]) -> list[str]:
    rels: list[str] = []
    for p in paths:
        if not p:
            continue
        po = Path(p)
        if not po.is_file():
            continue
        try:
            rels.append(str(po.resolve().relative_to(output_dir.resolve())))
        except ValueError:
            rels.append(str(po))
    return list(dict.fromkeys(rels))


def main(
    argv: list[str] | None = None,
    evaluate_config_override: dict[str, Any] | Any | None = None,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.seeds = list(args.seeds) if args.seeds else [int(args.seed)]

    model_cfg = ModelConfig()
    search_cfg = SearchConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()

    try:
        cfg_source = (
            evaluate_config_override
            if evaluate_config_override is not None
            else load_evaluate_config_module(args.eval_config)
        )
        raw_eval_cfg = extract_config_dict(cfg_source)
    except ConfigValidationError as exc:
        print(f"[Configuration Error] {exc}", file=sys.stderr)
        return 2

    cfg_output_dir = raw_eval_cfg.get("OUTPUT_DIR")
    status_filename = str(raw_eval_cfg.get("STATUS_FILENAME") or "execution_status.json")

    def _resolve_project_output_path(raw_path: str, check_cwd_existing: bool = False) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        # Avoid double-prefixing if caller explicitly passed 'python_src/...'
        if p.parts and p.parts[0] == _PYTHON_SRC_DIR.name:
            return _REPO_ROOT_DIR / p
        candidate_proj = _PYTHON_SRC_DIR / p
        if check_cwd_existing and not candidate_proj.exists():
            cwd_candidate = Path.cwd() / p
            if cwd_candidate.exists():
                return cwd_candidate
        return candidate_proj

    if args.resume:
        output_dir = _resolve_project_output_path(args.resume, check_cwd_existing=True)
    elif args.output_dir:
        output_dir = _resolve_project_output_path(args.output_dir, check_cwd_existing=False)
    elif cfg_output_dir:
        output_dir = _resolve_project_output_path(str(cfg_output_dir), check_cwd_existing=False)
    else:
        output_dir = _PYTHON_SRC_DIR / "reports" / "research_gaps" / _now_timestamp_slug()

    device = _resolve_device(args.device)
    hw_env = collect_hardware_environment(device)
    ref_arch = load_reference_architecture()
    ds_paths = resolve_dataset_files(args.data_path, data_cfg)
    ckpt_resolved = resolve_checkpoint_path(args.checkpoint)

    # Load prior summary.json if it exists in output_dir (supports resume via --resume OR stable OUTPUT_DIR)
    prior_summary: dict[str, Any] = {}
    if (output_dir / "summary.json").is_file():
        try:
            with open(output_dir / "summary.json", "r", encoding="utf-8") as f:
                prior_summary = json.load(f)
        except Exception:
            prior_summary = {}

    status_store = ExecutionStatusStore(output_dir / status_filename, output_dir=output_dir)
    try:
        status_store.load()
        plan = validate_and_plan_pipeline(
            raw_eval_cfg,
            status_store,
            output_dir,
            cli_experiment=args.experiment,
            cli_run_lambda_search=bool(args.run_lambda_search),
            cli_run_scene_prototype=bool(args.run_scene_prototype),
            cli_force_rerun=bool(args.force_rerun),
            dataset_name=args.dataset,
            dataset_ready=bool(ds_paths["dataset_ready"]),
            resolved_data_dir=str(ds_paths["resolved_data_dir"]),
            data_path_override=args.data_path,
            checkpoint_arg=args.checkpoint,
            checkpoint_resolved=ckpt_resolved,
            seeds=args.seeds,
            max_experiments=args.max_experiments,
            hw_warmup=args.hw_warmup,
            hw_runs=args.hw_runs,
            dry_run=bool(args.dry_run),
            summary_data=prior_summary if prior_summary else None,
        )
    except ConfigValidationError as exc:
        print(f"[Configuration Error] {exc}", file=sys.stderr)
        return 2

    eval_cfg = plan.config_values
    skip_completed: bool = eval_cfg["SKIP_COMPLETED"]
    continue_on_error: bool = eval_cfg["CONTINUE_ON_ERROR"]
    save_intermediate: bool = eval_cfg["SAVE_INTERMEDIATE_RESULTS"]
    save_status: bool = eval_cfg["SAVE_EXECUTION_STATUS"]
    gen_plots: bool = eval_cfg["GENERATE_PLOTS"]
    gen_report: bool = eval_cfg["GENERATE_REPORT"]

    # Sync effective flags onto args so helper functions observe config choices transparently
    args.run_lambda_search = bool(eval_cfg["RUN_HARDWARE_LAMBDA_SEARCH"])
    args.run_scene_prototype = bool(eval_cfg["RUN_SCENE_PROTOTYPE"])

    for notice in plan.notices:
        print(f"[Config Notice] {notice}")
    if plan.recovered_stages:
        print(
            f"[Recovery Notice] Recovered interrupted RUNNING stage(s) without valid completion record: "
            f"{plan.recovered_stages}. They will not be treated as completed."
        )

    print(format_execution_plan(plan))

    planned_sections: list[str] = []
    for d in plan.decisions:
        if d.action == "RUN" and d.section not in planned_sections:
            planned_sections.append(d.section)

    cli_args_serialized = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items()}
    cli_args_serialized["experiment"] = args.experiment or "evaluate_config"

    config_payload = {
        "timestamp_utc": _utc_iso(),
        "repo_root": str(_REPO_ROOT_DIR),
        "python_src": str(_PYTHON_SRC_DIR),
        "output_dir": str(output_dir),
        "cli_args": cli_args_serialized,
        "evaluate_config": {
            k: eval_cfg[k]
            for k in sorted(eval_cfg.keys())
            if isinstance(eval_cfg[k], (bool, str, int, float)) or eval_cfg[k] is None
        },
        "planned_sections": planned_sections,
        "scheduled_stages": plan.scheduled_ids,
        "hardware_environment": hw_env,
        "reference_architecture": ref_arch,
        "dataset_resolution": ds_paths,
        "checkpoint_resolved": str(ckpt_resolved) if ckpt_resolved else None,
        "model_config": asdict(model_cfg),
        "default_search_config": asdict(search_cfg),
        "default_train_config": asdict(train_cfg),
        "default_data_config": asdict(data_cfg),
    }

    if args.dry_run:
        print("=" * 80)
        print(" TrajectoryLNAS Research Gap Evaluation — DRY RUN")
        print("=" * 80)
        print(f" Experiment Selection : {args.experiment or 'evaluate_config'} -> {planned_sections}")
        print(f" Scheduled Stages     : {plan.scheduled_ids}")
        print(f" Dataset Target       : {args.dataset} (Ready: {ds_paths['dataset_ready']})")
        print(f" Resolved Data Dir    : {ds_paths['resolved_data_dir']}")
        print(f" Checkpoint           : {ckpt_resolved or 'Not found (hardware/scene eval will require checkpoint or --allow-untrained)'}")
        print(f" Reference Arch       : {ref_arch['indices']} -> {ref_arch['names']} (Source: {ref_arch['source']})")
        print(f" Model Grid           : past={model_cfg.past_len}, future={model_cfg.future_len}, hz={model_cfg.hz}, K={model_cfg.num_modes}")
        print(f" Seeds                : {args.seeds} | Max Experiments/Section: {args.max_experiments}")
        print(f" Environment          : PyTorch={hw_env['torch_version']} | CUDA={hw_env['cuda_available']} | Device={hw_env['selected_device']}")
        print(f" Target Output Dir    : {output_dir}")
        print(" Dry run completed successfully. No training executed and no files written.")
        print("=" * 80)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment_config.json", config_payload)

    # Preserve existing completed results across resumed runs
    results: dict[str, Any] = copy.deepcopy(prior_summary.get("experiments") or {})

    # Preserve snapshot of prior summary before overwriting any force-rerun completed stage
    if prior_summary and any(d.action == "RUN" and d.force_rerun for d in plan.decisions):
        backup_path = output_dir / "history" / f"summary_before_rerun_{_now_timestamp_slug()}.json"
        _write_json(backup_path, prior_summary)

    # Record SKIP decisions in status store without clobbering prior COMPLETED metadata
    for d in plan.decisions:
        if d.action == "SKIP":
            status_store.mark_skipped(
                d.stage_id,
                reason=d.reason,
                enabled_flag=d.enabled,
                preserve_completed=True,
                save_enabled=save_status,
            )

    def _build_summary_payload(cur_results: dict[str, Any], cur_figs: list[str] | None = None) -> dict[str, Any]:
        return {
            "timestamp_utc": _utc_iso(),
            "output_dir": str(output_dir),
            "dataset": args.dataset,
            "seed": args.seed,
            "seeds": args.seeds,
            "hardware_environment": hw_env,
            "config": config_payload,
            "experiments": cur_results,
            "generated_figures": cur_figs if cur_figs is not None else list(prior_summary.get("generated_figures") or []),
        }

    def _persist_intermediate() -> None:
        if save_intermediate:
            _write_json(output_dir / "summary.json", _build_summary_payload(results))

    active_stage: list[tuple[str | None, float]] = [(None, 0.0)]
    any_stage_failed = False
    pipeline_stopped_on_error = False
    audit_halted_comparatives = False
    failed_stage_ids: set[str] = set()

    def _handle_stage_start(sid: str) -> None:
        t0 = status_store.mark_running(sid, save_enabled=save_status)
        active_stage[0] = (sid, t0)

    try:
        # 1. AUDIT
        audit_dec = plan.by_id("AUDIT")
        if audit_dec.action == "RUN":
            _handle_stage_start("AUDIT")
            t0 = active_stage[0][1]
            try:
                audit_res = run_data_audit(
                    dataset_name=args.dataset,
                    data_path=args.data_path,
                    device=device,
                    model_cfg=model_cfg,
                    search_cfg=search_cfg,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    checkpoint_path=args.checkpoint,
                )
            except Exception as exc:
                audit_res = {
                    "experiment": "audit",
                    "status": "failed",
                    "error": f"[{type(exc).__name__}] {exc}",
                    "blocking_issues": [str(exc)],
                    "warnings": [],
                }
            results["audit"] = audit_res
            _write_json(output_dir / "dataset_audit.json", audit_res)
            _persist_intermediate()

            blocking = audit_res.get("blocking_issues") or []
            if audit_res.get("status") == "failed" or blocking:
                any_stage_failed = True
                failed_stage_ids.add("AUDIT")
                err_msg = "; ".join(str(x) for x in blocking) if blocking else str(audit_res.get("error") or "Audit failed")
                status_store.mark_failed(
                    "AUDIT",
                    t0,
                    error=err_msg,
                    output_files=_rel_output_paths(output_dir, [output_dir / "dataset_audit.json", output_dir / "summary.json"]),
                    save_enabled=save_status,
                )
                print(f"[Audit Blocking Issues Detected] {blocking or err_msg}")
                print("[Audit Warning] Halting comparative search/training experiments due to blocking audit issues.")
                audit_halted_comparatives = True
                for cid in COMPARATIVE_STAGE_IDS:
                    c_dec = plan.by_id(cid)
                    if c_dec.action == "RUN":
                        status_store.mark_skipped(
                            cid,
                            reason="halted by mandatory data integrity audit",
                            enabled_flag=True,
                            preserve_completed=True,
                            save_enabled=save_status,
                        )
            else:
                status_store.mark_completed(
                    "AUDIT",
                    t0,
                    output_files=_rel_output_paths(output_dir, [output_dir / "dataset_audit.json", output_dir / "summary.json"]),
                    save_enabled=save_status,
                )
            active_stage[0] = (None, 0.0)

        def _filter_prior_ablation_rows(
            raw_rows: list[dict[str, Any]] | None,
            run_ids: set[str],
            csv_filename: str,
        ) -> list[dict[str, Any]] | None:
            if not raw_rows:
                return raw_rows
            csv_ok = (output_dir / csv_filename).is_file() and (output_dir / csv_filename).stat().st_size > 0
            kept: list[dict[str, Any]] = []
            for row in raw_rows:
                sid = str(row.get("config_id") or "")
                if sid not in run_ids:
                    kept.append(row)
                    continue
                if not skip_completed or plan.by_id(sid).force_rerun or sid in status_store.recovered_interrupted_stages:
                    continue
                if not csv_ok:
                    continue
                st_rec = status_store.get_stage(sid)
                if st_rec.get("status") != "COMPLETED" or not st_rec.get("output_files"):
                    continue
                all_outputs_ok = all(
                    (Path(str(p)) if Path(str(p)).is_absolute() else (output_dir / str(p))).is_file()
                    and (Path(str(p)) if Path(str(p)).is_absolute() else (output_dir / str(p))).stat().st_size > 0
                    for p in st_rec.get("output_files", [])
                )
                if not all_outputs_ok:
                    continue
                ckpt_p = row.get("checkpoint_path")
                if ckpt_p:
                    ckpt_path_obj = Path(str(ckpt_p)) if Path(str(ckpt_p)).is_absolute() else (output_dir / str(ckpt_p))
                    if not ckpt_path_obj.is_file() or ckpt_path_obj.stat().st_size == 0:
                        continue
                kept.append(row)
            return kept

        # 2. Research Gap A: Factorized Three-Branch NAS (A0-A7)
        fac_run_ids = {
            sid for sid in FACTORIZED_STAGE_IDS
            if plan.by_id(sid).action == "RUN" and not audit_halted_comparatives
        }
        fac_force_ids = {
            sid for sid in fac_run_ids if plan.by_id(sid).force_rerun
        }
        if fac_run_ids and not pipeline_stopped_on_error:
            prior_fac = _filter_prior_ablation_rows(
                (results.get("factorized") or {}).get("rows"),
                fac_run_ids,
                "factorized_nas_results.csv",
            )

            def _on_fac_complete(sid: str, stage_rows: list[dict[str, Any]], partial_fac: dict[str, Any]) -> None:
                nonlocal any_stage_failed, pipeline_stopped_on_error
                t0 = active_stage[0][1]
                results["factorized"] = partial_fac
                _write_csv(output_dir / "factorized_nas_results.csv", partial_fac["rows"])
                _persist_intermediate()

                bad_rows = [r for r in stage_rows if r.get("status") != "completed"]
                if bad_rows:
                    any_stage_failed = True
                    failed_stage_ids.add(sid)
                    err_msg = "; ".join(str(r.get("error") or r.get("status")) for r in bad_rows)
                    status_store.mark_failed(
                        sid,
                        t0,
                        error=err_msg,
                        output_files=_rel_output_paths(output_dir, [output_dir / "factorized_nas_results.csv", output_dir / "summary.json"]),
                        save_enabled=save_status,
                    )
                    if not continue_on_error:
                        pipeline_stopped_on_error = True
                else:
                    ckpt_paths = [r.get("checkpoint_path") for r in stage_rows]
                    out_paths = _rel_output_paths(
                        output_dir,
                        [*ckpt_paths, output_dir / "factorized_nas_results.csv", output_dir / "summary.json"],
                    )
                    status_store.mark_completed(sid, t0, output_files=out_paths, save_enabled=save_status)
                active_stage[0] = (None, 0.0)

            fac_res = run_factorized_ablation(
                args,
                model_cfg=model_cfg,
                output_dir=output_dir,
                prior_results=prior_fac,
                enabled_stage_ids=fac_run_ids,
                force_rerun_ids=fac_force_ids,
                skip_completed=skip_completed,
                continue_on_error=continue_on_error,
                on_stage_start=_handle_stage_start,
                on_stage_complete=_on_fac_complete,
            )
            results["factorized"] = fac_res
            _write_csv(output_dir / "factorized_nas_results.csv", fac_res["rows"])
            _persist_intermediate()

        # 3. Research Gap B: Hardware-Aware NAS Profiling
        hw_base_dec = plan.by_id("HARDWARE")
        hw_lam_dec = plan.by_id("HARDWARE_LAMBDA_SEARCH")
        run_hw_base = hw_base_dec.action == "RUN" and not audit_halted_comparatives and not pipeline_stopped_on_error
        run_hw_lam = hw_lam_dec.action == "RUN" and not audit_halted_comparatives and not pipeline_stopped_on_error

        if run_hw_base or run_hw_lam:
            prior_hw = ((results.get("hardware") or {}).get("rows"))

            def _on_hw_complete(sid: str, stage_rows: list[dict[str, Any]], partial_hw: dict[str, Any]) -> None:
                nonlocal any_stage_failed, pipeline_stopped_on_error
                t0 = active_stage[0][1]
                results["hardware"] = partial_hw
                _write_csv(output_dir / "hardware_profile_results.csv", partial_hw["rows"])
                _persist_intermediate()

                bad_rows = [r for r in stage_rows if r.get("status") != "completed"]
                if bad_rows:
                    any_stage_failed = True
                    failed_stage_ids.add(sid)
                    err_msg = "; ".join(str(r.get("error") or r.get("status")) for r in bad_rows)
                    status_store.mark_failed(
                        sid,
                        t0,
                        error=err_msg,
                        output_files=_rel_output_paths(output_dir, [output_dir / "hardware_profile_results.csv", output_dir / "summary.json"]),
                        save_enabled=save_status,
                    )
                    if not continue_on_error:
                        pipeline_stopped_on_error = True
                else:
                    status_store.mark_completed(
                        sid,
                        t0,
                        output_files=_rel_output_paths(output_dir, [output_dir / "hardware_profile_results.csv", output_dir / "summary.json"]),
                        save_enabled=save_status,
                    )
                active_stage[0] = (None, 0.0)

            hw_res = run_hardware_profiling(
                args,
                model_cfg=model_cfg,
                search_cfg=search_cfg,
                output_dir=output_dir,
                prior_results=prior_hw,
                run_baselines=run_hw_base,
                run_lambda_sweep=run_hw_lam,
                force_rerun_baselines=hw_base_dec.force_rerun,
                force_rerun_lambda=hw_lam_dec.force_rerun,
                continue_on_error=continue_on_error,
                on_stage_start=_handle_stage_start,
                on_stage_complete=_on_hw_complete,
            )
            results["hardware"] = hw_res
            _write_csv(output_dir / "hardware_profile_results.csv", hw_res["rows"])
            _persist_intermediate()

        # 4. Research Gap C: Scene Complexity & Prototype
        sc_dec = plan.by_id("SCENE")
        sc_proto_dec = plan.by_id("SCENE_PROTOTYPE")
        run_sc = sc_dec.action == "RUN" and not audit_halted_comparatives and not pipeline_stopped_on_error
        run_sc_proto = sc_proto_dec.action == "RUN" and not audit_halted_comparatives and not pipeline_stopped_on_error

        if run_sc or run_sc_proto:
            prior_sc = results.get("scene")

            def _on_sc_complete(sid: str, partial_sc: dict[str, Any]) -> None:
                nonlocal any_stage_failed, pipeline_stopped_on_error, run_sc_proto
                t0 = active_stage[0][1]
                results["scene"] = partial_sc
                if partial_sc.get("rows"):
                    _write_csv(output_dir / "scene_complexity_results.csv", partial_sc["rows"])
                _persist_intermediate()

                if sid == "SCENE":
                    st = partial_sc.get("status")
                    if st in ("failed", "not_run", "pending") or not partial_sc.get("rows"):
                        any_stage_failed = True
                        failed_stage_ids.add("SCENE")
                        status_store.mark_failed(
                            "SCENE",
                            t0,
                            error=str(partial_sc.get("error") or f"SCENE finished with status '{st}'"),
                            output_files=_rel_output_paths(output_dir, [output_dir / "scene_complexity_results.csv", output_dir / "summary.json"]),
                            save_enabled=save_status,
                        )
                        if run_sc_proto:
                            status_store.mark_skipped(
                                "SCENE_PROTOTYPE",
                                reason="prerequisite SCENE failed",
                                enabled_flag=True,
                                preserve_completed=True,
                                save_enabled=save_status,
                            )
                            run_sc_proto = False
                        if not continue_on_error:
                            pipeline_stopped_on_error = True
                    else:
                        status_store.mark_completed(
                            "SCENE",
                            t0,
                            output_files=_rel_output_paths(output_dir, [output_dir / "scene_complexity_results.csv", output_dir / "summary.json"]),
                            save_enabled=save_status,
                        )
                elif sid == "SCENE_PROTOTYPE":
                    proto = partial_sc.get("experimental_scene_conditioned_prototype") or {}
                    if proto.get("status") == "completed":
                        status_store.mark_completed(
                            "SCENE_PROTOTYPE",
                            t0,
                            output_files=_rel_output_paths(output_dir, [output_dir / "summary.json"]),
                            save_enabled=save_status,
                        )
                    else:
                        any_stage_failed = True
                        failed_stage_ids.add("SCENE_PROTOTYPE")
                        status_store.mark_failed(
                            "SCENE_PROTOTYPE",
                            t0,
                            error=str(proto.get("error") or partial_sc.get("error") or "SCENE_PROTOTYPE failed"),
                            output_files=_rel_output_paths(output_dir, [output_dir / "summary.json"]),
                            save_enabled=save_status,
                        )
                        if not continue_on_error:
                            pipeline_stopped_on_error = True
                active_stage[0] = (None, 0.0)

            sc_res = run_scene_complexity_evaluation(
                args,
                model_cfg=model_cfg,
                search_cfg=search_cfg,
                output_dir=output_dir,
                run_stratification=run_sc,
                run_prototype=run_sc_proto,
                prior_result=prior_sc,
                continue_on_error=continue_on_error,
                on_stage_start=_handle_stage_start,
                on_stage_complete=_on_sc_complete,
            )
            results["scene"] = sc_res
            if sc_res.get("rows"):
                _write_csv(output_dir / "scene_complexity_results.csv", sc_res["rows"])
            _persist_intermediate()

        # 5. Warmup & Temperature Annealing Ablation (T0-T3)
        wt_run_ids = {
            sid for sid in WARMUP_TEMP_STAGE_IDS
            if plan.by_id(sid).action == "RUN" and not audit_halted_comparatives
        }
        wt_force_ids = {
            sid for sid in wt_run_ids if plan.by_id(sid).force_rerun
        }
        if wt_run_ids and not pipeline_stopped_on_error:
            prior_wt = _filter_prior_ablation_rows(
                (results.get("warmup_temperature") or {}).get("rows"),
                wt_run_ids,
                "warmup_temperature_results.csv",
            )

            def _on_wt_complete(sid: str, stage_rows: list[dict[str, Any]], partial_wt: dict[str, Any]) -> None:
                nonlocal any_stage_failed, pipeline_stopped_on_error
                t0 = active_stage[0][1]
                results["warmup_temperature"] = partial_wt
                _write_csv(output_dir / "warmup_temperature_results.csv", partial_wt["rows"])
                _persist_intermediate()

                bad_rows = [r for r in stage_rows if r.get("status") != "completed"]
                if bad_rows:
                    any_stage_failed = True
                    failed_stage_ids.add(sid)
                    err_msg = "; ".join(str(r.get("error") or r.get("status")) for r in bad_rows)
                    status_store.mark_failed(
                        sid,
                        t0,
                        error=err_msg,
                        output_files=_rel_output_paths(output_dir, [output_dir / "warmup_temperature_results.csv", output_dir / "summary.json"]),
                        save_enabled=save_status,
                    )
                    if not continue_on_error:
                        pipeline_stopped_on_error = True
                else:
                    ckpt_paths = [r.get("checkpoint_path") for r in stage_rows]
                    out_paths = _rel_output_paths(
                        output_dir,
                        [*ckpt_paths, output_dir / "warmup_temperature_results.csv", output_dir / "summary.json"],
                    )
                    status_store.mark_completed(sid, t0, output_files=out_paths, save_enabled=save_status)
                active_stage[0] = (None, 0.0)

            wt_res = run_warmup_temperature_ablation(
                args,
                model_cfg=model_cfg,
                output_dir=output_dir,
                prior_results=prior_wt,
                enabled_stage_ids=wt_run_ids,
                force_rerun_ids=wt_force_ids,
                skip_completed=skip_completed,
                continue_on_error=continue_on_error,
                on_stage_start=_handle_stage_start,
                on_stage_complete=_on_wt_complete,
            )
            results["warmup_temperature"] = wt_res
            _write_csv(output_dir / "warmup_temperature_results.csv", wt_res["rows"])
            _persist_intermediate()

    except KeyboardInterrupt:
        sid_cur, t0_cur = active_stage[0]
        if sid_cur is not None:
            status_store.mark_failed(
                sid_cur,
                t0_cur,
                error="Execution interrupted by KeyboardInterrupt.",
                save_enabled=save_status,
            )
        status_store.data["pipeline_state"] = "INTERRUPTED"
        status_store.save(enabled=save_status)
        _write_json(output_dir / "summary.json", _build_summary_payload(results))
        print("\n[Interrupted] Execution interrupted; status and completed results saved safely.")
        print(format_final_summary(status_store))
        return 130

    generated_figures = generate_all_plots(output_dir, results) if gen_plots else list(prior_summary.get("generated_figures") or [])
    summary_payload = _build_summary_payload(results, cur_figs=generated_figures)
    _write_json(output_dir / "summary.json", summary_payload)

    if gen_report:
        report_md = build_research_gap_report(output_dir, config_payload, results, generated_figures)
        with open(output_dir / "research_gap_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

    status_store.data["pipeline_state"] = STATUS_FAILED if any_stage_failed else STATUS_COMPLETED
    status_store.save(enabled=save_status)

    print(format_final_summary(status_store))
    print("=" * 80)
    print(f" Research Gap Evaluation Finished. Output directory: {output_dir}")
    print("=" * 80)
    return 1 if any_stage_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

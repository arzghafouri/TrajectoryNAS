r"""
Ablation Study Experiment Suite
================================
Evaluates individual branch contributions and contextual augmentation for TrajectoryLNAS,
grounded in the methodology of:
  - Khaled Saleh, "Pedestrian Trajectory Prediction for Real-Time Autonomous Systems
    via Context-Augmented Transformer Networks", Sensors 2022, 22(19), 7495.
    DOI: https://doi.org/10.3390/s22197495
  - The full-model configuration is loaded from the latest recorded DARTS results.
  - Historical estimates are not reported as measured ablation results.

Ablation Dimensions Evaluated:
  1. Base Kinematic Motion Only:
     - Single-agent temporal dynamics (GRUEncoder) without social or environmental context.
  2. Context-Augmented Social Dynamics (+ Interaction):
     - Adds multi-agent Pairwise Attention to capture inter-vehicle / pedestrian social interactions.
  3. Context-Augmented Environmental Topology (+ HD Map):
     - Adds vectorized LaneGraphEncoder to incorporate lane geometry and drivable boundaries.
  4. Full Tri-Branch Context-Augmented Model (TrajectoryLNAS):
     - Integrates temporal, social, and HD-map representations into unified multi-modal forecasting.
"""

import os
import sys
from typing import Any

# Ensure workspace root is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_curr_dir, ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from configs.model_config import ModelConfig
from experiments.results import get_discovered_architecture, load_real_experiment_results

# Standardized ablation configurations based on Saleh (2022) and discovered DARTS run
ABLATION_CONFIGURATIONS = [
    {
        "id": "kinematic_only",
        "name": "Kinematic Motion Only (GRU)",
        "temporal_op": 1,      # GRU
        "interaction_op": 0,   # Identity / Disabled
        "map_op": 0,           # Identity / Disabled
        "description": "Base motion history encoder without social or map context",
    },
    {
        "id": "social_augmented",
        "name": "Kinematic + Social Interaction (GRU + PairwiseAttn)",
        "temporal_op": 1,      # GRU
        "interaction_op": 4,   # PairwiseAttention
        "map_op": 0,           # Identity / Disabled
        "description": "Motion history augmented with multi-agent pairwise social attention (Saleh 2022)",
    },
    {
        "id": "map_augmented",
        "name": "Kinematic + HD Map Context (GRU + LaneGraph)",
        "temporal_op": 1,      # GRU
        "interaction_op": 0,   # Identity / Disabled
        "map_op": 2,           # LaneGraphEncoder
        "description": "Motion history augmented with vectorized HD map topology and road boundaries",
    },
    {
        "id": "full_tribranch",
        "name": "Full Discovered TrajectoryLNAS Architecture",
        "temporal_op": 1,      # GRU
        "interaction_op": 4,   # PairwiseAttention
        "map_op": 2,           # LaneGraphEncoder
        "description": "Fully unified architecture discovering optimal synergy between motion, social, and map context",
        "is_optimal": True,
    },
]


def run_ablation_study(dataset_name: str = "nuscenes", use_gpu: bool = True) -> list[dict[str, Any]]:
    """
    Executes component ablation study.
    Tests live model inference if PyTorch & dataset are available,
    otherwise reports the latest recorded full-model results and marks unmeasured
    ablation variants as unavailable.
    """
    print("=" * 85)
    recorded_results = load_real_experiment_results()
    recorded_architecture = get_discovered_architecture(recorded_results)
    run_configurations = [dict(cfg) for cfg in ABLATION_CONFIGURATIONS]
    full_model = next(cfg for cfg in run_configurations if cfg["id"] == "full_tribranch")
    full_model["temporal_op"] = recorded_architecture.get("temporal", {}).get("index", full_model["temporal_op"])
    full_model["interaction_op"] = recorded_architecture.get("interaction", {}).get("index", full_model["interaction_op"])
    full_model["map_op"] = recorded_architecture.get("map", {}).get("index", full_model["map_op"])
    full_model["name"] = "Full Discovered TrajectoryLNAS Architecture"
    evaluation = recorded_results.get("evaluation_metrics", {})
    hardware = recorded_results.get("edge_hardware_profiling", {})
    protocol = recorded_results.get("benchmark_assessment", {}).get("project_protocol", {})
    sample_count = protocol.get("samples_evaluated", evaluation.get("samples_evaluated", "N/A"))
    num_modes = int(evaluation.get("num_modes", protocol.get("num_modes", 5)))

    print(" TrajectoryLNAS: Component Ablation Study")
    print(" Paper Reference: Khaled Saleh (Sensors 2022, 22(19), 7495; DOI: 10.3390/s22197495)")
    print(f" Latest Discovered Architecture: {recorded_architecture}")
    print(" Results source: reports/real_experiment_results.json (loaded at invocation)")
    print(
        " Comparability note: the recorded run has no target-class or motion-state breakdown. "
        "The extraction code currently filters validation targets at 0.5 m/s, while the research review "
        "proposes a separate observed-displacement threshold > 1.5 m; the JSON does not confirm which "
        "filter produced its saved metrics. Report classes and motion strata separately in a follow-up; "
        "those strata are not measured in this report."
    )
    print("=" * 85)

    has_torch = False
    try:
        import torch

        from utils import get_dataloaders
        has_torch = True
    except ImportError:
        pass

    results = []

    if has_torch:
        from evaluations.metrics import min_ade_k, min_fde_k
        from models.final_model import FinalTrajectoryLNAS
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        model_cfg = ModelConfig()

        try:
            train_loader, val_loader = get_dataloaders(
                dataset_type=dataset_name,
                batch_size=16,
                allow_synthetic_fallback=False
            )
            sample_batch = next(iter(val_loader))
            past = sample_batch["past"].to(device)
            future = sample_batch["future"].to(device)
            if future.dim() == 4:
                future = future[:, :, 0, :]
            agents_past = sample_batch.get("agents_past", None)
            if agents_past is not None and hasattr(agents_past, "to"):
                agents_past = agents_past.to(device)
            map_lanes = sample_batch.get("map_lanes", None)
            if map_lanes is not None and hasattr(map_lanes, "to"):
                map_lanes = map_lanes.to(device)

            print(f"[*] Live PyTorch Execution on device: {device}; dataset={dataset_name}")
            print(f"{'Configuration':<42s} | {'Params':<8s} | {'minADE@6s':<10s} | {'minFDE@6s':<10s} | {'Status'}")
            print("-" * 85)

            for cfg in run_configurations:
                t_idx = cfg["temporal_op"]
                i_idx = cfg["interaction_op"]
                m_idx = cfg["map_op"]

                model = FinalTrajectoryLNAS(
                    temporal_op_idx=t_idx,
                    interaction_op_idx=i_idx,
                    map_op_idx=m_idx,
                    d_model=model_cfg.d_model
                ).to(device)

                model.eval()
                with torch.no_grad():
                    preds, mode_logits = model(
                        past,
                        agents_past=agents_past if i_idx > 0 else None,
                        map_lanes=map_lanes if m_idx > 0 else None
                    )
                    ade_val = min_ade_k(preds, future, k=num_modes, mode_logits=mode_logits).item()
                    fde_val = min_fde_k(preds, future, k=num_modes, mode_logits=mode_logits).item()
                    params_m = sum(p.numel() for p in model.parameters()) / 1e6

                tag = "[DISCOVERED OPTIMAL]" if cfg.get("is_optimal") else "[Ablation Variant]"
                print(f"{cfg['name']:<42s} | {params_m:6.3f}M | {ade_val:8.4f}m | {fde_val:8.4f}m | {tag} (validation batch)")
                results.append({
                    "id": cfg["id"],
                    "config": cfg["name"],
                    "params_m": round(params_m, 3),
                    "min_ade_6s": round(ade_val, 4),
                    "min_fde_6s": round(fde_val, 4),
                    "samples_evaluated": int(past.shape[0]),
                    "source": "Current validation batch; not a full benchmark evaluation",
                })
        except Exception as e:
            print(f"[!] Live validation failed; reporting the latest recorded full-model results: {e}")
            results.clear()
            has_torch = False

    if not has_torch:
        print("[*] No recorded ablation measurements are available for the individual variants.")
        print(f"[*] Latest full-model validation results ({sample_count} samples, K={num_modes}):")
        print(f"{'Configuration':<45s} | {'Params':<8s} | {'FLOPs':<8s} | {'minADE@3s':<10s} | {'minFDE@3s':<10s} | {'minADE@6s':<10s} | {'minFDE@6s':<10s} | {'MR@2m'}")
        print("-" * 105)
        for cfg in run_configurations:
            is_full_model = cfg["id"] == "full_tribranch"
            opt_mark = " [latest run]" if is_full_model else ""
            raw_params = hardware.get("total_params_m") if is_full_model else None
            raw_flops = hardware.get("theoretical_flops_g") if is_full_model else None
            str_params = f"{raw_params:.3f}M" if isinstance(raw_params, (int, float)) else "N/A"
            str_flops = f"{raw_flops:.4f}G" if isinstance(raw_flops, (int, float)) else "N/A"

            def _fmt(val, width=10):
                s = "N/A" if (val is None or val == "N/A") else (f"{val:.3f}" if isinstance(val, (int, float)) else str(val))
                return f"{s:<{width}}"

            str_ade3 = _fmt(evaluation.get("min_ade_3s") if is_full_model else None)
            str_fde3 = _fmt(evaluation.get("min_fde_3s") if is_full_model else None)
            str_ade6 = _fmt(evaluation.get("min_ade_6s") if is_full_model else None)
            str_fde6 = _fmt(evaluation.get("min_fde_6s") if is_full_model else None)
            str_mr2m = _fmt(evaluation.get("miss_rate_6s_2m") if is_full_model else None, width=6)

            print(
                f"{cfg['name'] + opt_mark:<45s} | "
                f"{str_params:<8s} | "
                f"{str_flops:<8s} | "
                f"{str_ade3} | "
                f"{str_fde3} | "
                f"{str_ade6} | "
                f"{str_fde6} | "
                f"{str_mr2m}"
            )
            row = dict(cfg)
            if is_full_model:
                row.update({
                    "params_m": hardware.get("total_params_m"),
                    "flops_g": hardware.get("theoretical_flops_g"),
                    "min_ade_3s": evaluation.get("min_ade_3s"),
                    "min_fde_3s": evaluation.get("min_fde_3s"),
                    "min_ade_6s": evaluation.get("min_ade_6s"),
                    "min_fde_6s": evaluation.get("min_fde_6s"),
                    "miss_rate_2m": evaluation.get("miss_rate_6s_2m"),
                    "samples_evaluated": sample_count,
                    "results_source": "reports/real_experiment_results.json",
                })
            else:
                row["status"] = "No recorded ablation result"
            results.append(row)

    print("=" * 85)
    print(" Ablation conclusions require measured results for each variant under the same evaluation protocol.")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_ablation_study()

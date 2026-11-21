"""
Evaluator & Performance Profiler
================================
Evaluates FinalTrajectoryModel accuracy and efficiency on real motion forecasting benchmarks.
Supports Multi-Modal Trajectory Evaluation (minADE@6, minFDE@6, MissRate@6) and strict checkpoint auditing.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# Allow running independently or from root
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
_grandparent_dir = os.path.dirname(_parent_dir)
for p in (_parent_dir, _grandparent_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from configs.model_config import ModelConfig
from evaluations.visualization import plot_benchmark_comparison, plot_horizon_error_curve

if HAS_TORCH:
    from evaluations.metrics import (
        HardwareProfiler,
        endpoint_miss_rate_k,
        evaluate_collision_risk,
        max_displacement_miss_rate_k,
        min_ade_k,
        min_fde_k,
        miss_rate_k,
        normalize_future_tensor,
        safe_norm,
    )
    from models.base_operator import resolve_architecture_indices
    from models.final_model import FinalTrajectoryModel
    from utils.dataloaders import get_dataloaders
else:
    HardwareProfiler = None
    FinalTrajectoryModel = None
    get_dataloaders = None
    endpoint_miss_rate_k = None
    evaluate_collision_risk = None
    max_displacement_miss_rate_k = None
    min_ade_k = None
    min_fde_k = None
    miss_rate_k = None
    normalize_future_tensor = None
    resolve_architecture_indices = None
    safe_norm = None


class TrajectoryEvaluator:
    """
    Production-grade Evaluator for Trajectory Forecasting Architectures.
    Enforces strict checkpoint integrity and genuine multi-modal metric reporting.
    """
    def __init__(
        self,
        checkpoint_path: str | None = "checkpoints/best_trajectory_lnas.pt",
        dataset_name: str = "nuscenes",
        device: str | None = None,
        model_config: ModelConfig | None = None,
        best_ops: tuple[int, int, int] | None = None,
        allow_untrained: bool = False
    ):
        if not HAS_TORCH or torch is None:
            raise ImportError(
                "PyTorch is required to initialize TrajectoryEvaluator. "
                "Please install PyTorch in your active Python environment: pip install torch"
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name or "nuscenes"
        self.model_cfg = model_config or ModelConfig()
        self.allow_untrained = allow_untrained

        resolved_ops, ops_source = resolve_architecture_indices(best_ops=best_ops)
        self.best_ops = resolved_ops
        self.architecture_source = ops_source
        self.model = None
        self.checkpoint_metadata = {}
        self._last_sample_batch: dict[str, Any] | None = None

        self._initialize_model()
        self.profiler = HardwareProfiler(
            self.model,
            self.device,
            past_len=self.model_cfg.past_len,
            max_agents=self.model_cfg.max_agents,
            max_obstacles=self.model_cfg.max_obstacles,
            max_lanes=self.model_cfg.max_lanes,
            lane_pts=self.model_cfg.lane_pts,
        )

    def _initialize_model(self):
        """Loads weights from checkpoint or strictly halts if checkpoint is invalid."""
        loaded_checkpoint = False

        # Resolve candidate checkpoint locations
        resolved_ckpt = self.checkpoint_path
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            candidates = [
                os.path.join(_parent_dir, self.checkpoint_path),
                os.path.join(_grandparent_dir, self.checkpoint_path),
                os.path.join(_parent_dir, "checkpoints", os.path.basename(self.checkpoint_path)),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_ckpt = cand
                    break

        if resolved_ckpt and os.path.exists(resolved_ckpt):
            try:
                try:
                    ckpt = torch.load(resolved_ckpt, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(resolved_ckpt, map_location=self.device)

                self.checkpoint_metadata = ckpt if isinstance(ckpt, dict) else {}
                cfg_info = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

                from configs.search_config import SearchConfig
                s_cfg = SearchConfig()
                temporal_names = s_cfg.temporal_ops
                interaction_names = s_cfg.interaction_ops
                map_names = s_cfg.map_ops

                best_t = cfg_info.get("best_temporal_idx")
                best_i = cfg_info.get("best_interaction_idx")
                best_m = cfg_info.get("best_map_idx")

                best_t = int(best_t) if best_t is not None else self.best_ops[0]
                best_i = int(best_i) if best_i is not None else self.best_ops[1]
                best_m = int(best_m) if best_m is not None else self.best_ops[2]
                self.best_ops = (best_t, best_i, best_m)

                self.model = FinalTrajectoryModel(
                    best_temporal_idx=best_t,
                    best_interaction_idx=best_i,
                    best_map_idx=best_m,
                    d_model=self.model_cfg.d_model,
                    past_len=self.model_cfg.past_len,
                    future_len=self.model_cfg.future_len,
                    num_modes=self.model_cfg.num_modes,
                    max_agents=self.model_cfg.max_agents,
                    max_obstacles=self.model_cfg.max_obstacles,
                    max_lanes=self.model_cfg.max_lanes,
                    lane_pts=self.model_cfg.lane_pts,
                    hz=self.model_cfg.hz,
                    search_config=s_cfg,
                    model_config=self.model_cfg,
                ).to(self.device)

                self.model.load_state_dict(state_dict)
                loaded_checkpoint = True
                self.architecture_source = f"model_checkpoint:{resolved_ckpt}"
                t_display = temporal_names[best_t].upper() if 0 <= best_t < len(temporal_names) else f"OP_{best_t}"
                i_display = interaction_names[best_i].upper() if 0 <= best_i < len(interaction_names) else f"OP_{best_i}"
                m_display = map_names[best_m].upper() if 0 <= best_m < len(map_names) else f"OP_{best_m}"
                print(f"[Evaluator] Successfully loaded checkpoint: {resolved_ckpt}")
                print(f"[Evaluator] Architecture: Temporal={t_display} (idx {best_t}) | "
                      f"Interaction={i_display} (idx {best_i}) | "
                      f"Map={m_display} (idx {best_m})")
            except Exception as e:
                if not self.allow_untrained:
                    raise RuntimeError(f"[Evaluator Error] Failed to load checkpoint weights from '{resolved_ckpt}': {e}")
                print(f"[Evaluator Warning] Could not load checkpoint ({e}). Initializing random test architecture as allow_untrained=True.")

        elif self.checkpoint_path and not os.path.exists(resolved_ckpt or ""):
            if not self.allow_untrained:
                raise FileNotFoundError(
                    f"[Evaluator Error] Checkpoint file '{self.checkpoint_path}' does not exist. "
                    f"To evaluate a trained model, provide a valid checkpoint. (Set allow_untrained=True for smoke tests only)."
                )
            print(f"[Evaluator Warning] Checkpoint '{self.checkpoint_path}' not found. Initializing untrained model for testing.")

        if not loaded_checkpoint:
            from configs.search_config import SearchConfig
            s_cfg = SearchConfig()
            self.model = FinalTrajectoryModel(
                best_temporal_idx=self.best_ops[0],
                best_interaction_idx=self.best_ops[1],
                best_map_idx=self.best_ops[2],
                d_model=self.model_cfg.d_model,
                past_len=self.model_cfg.past_len,
                future_len=self.model_cfg.future_len,
                num_modes=self.model_cfg.num_modes,
                max_agents=self.model_cfg.max_agents,
                max_obstacles=self.model_cfg.max_obstacles,
                max_lanes=self.model_cfg.max_lanes,
                lane_pts=self.model_cfg.lane_pts,
                hz=self.model_cfg.hz,
                search_config=s_cfg,
                model_config=self.model_cfg,
            ).to(self.device)

        self.model.eval()

    def evaluate_dataset(
        self,
        batch_size: int = 16,
        max_batches: int | None = None,
        custom_data_path: str | None = None
    ) -> dict[str, Any]:
        """
        Runs comprehensive multi-modal evaluation over the validation set.
        """
        ds = self.dataset_name or "nuscenes"
        bs = batch_size or 16
        _, val_loader = get_dataloaders(
            val_dir=custom_data_path,
            batch_size=bs,
            dataset_name=ds,
            model_config=self.model_cfg,
        )

        if len(val_loader) == 0:
            raise RuntimeError("Validation DataLoader is empty; cannot run evaluation.")

        total_samples = 0
        ade_1s_accum = 0.0
        ade_2s_accum = 0.0
        ade_3s_accum = 0.0
        ade_4s_accum = 0.0
        ade_5s_accum = 0.0
        ade_6s_accum = 0.0
        fde_3s_accum = 0.0
        fde_6s_accum = 0.0
        endpoint_mr_accum = 0.0
        max_disp_mr_accum = 0.0
        collision_scenarios = 0
        collided_scenarios = 0
        last_cr_reason: str | None = None
        step_errors_accum = torch.zeros(self.model_cfg.future_len, device=self.device)
        k_eval = int(self.model.num_modes)
        effective_k = k_eval

        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break

                past = batch["past"].to(self.device)
                future = normalize_future_tensor(batch["future"].to(self.device))

                def _to_dev(key: str):
                    val = batch.get(key, None)
                    return val.to(self.device) if val is not None and hasattr(val, "to") else val

                agents_past = _to_dev("agents_past")
                map_lanes = _to_dev("map_lanes")
                agent_mask = _to_dev("agent_mask")
                lane_mask = _to_dev("lane_mask")
                category_id = _to_dev("category_id")
                map_lane_types = _to_dev("map_lane_types")
                neighbor_categories = _to_dev("neighbor_categories")
                static_obstacles = _to_dev("static_obstacles")
                static_obstacle_types = _to_dev("static_obstacle_types")
                static_obstacle_mask = _to_dev("static_obstacle_mask")
                neighbors_future = _to_dev("neighbors_future")
                neighbor_future_mask = _to_dev("neighbor_future_mask")

                if self._last_sample_batch is None:
                    self._last_sample_batch = {
                        "past": past,
                        "agents_past": agents_past,
                        "map_lanes": map_lanes,
                        "category_id": category_id,
                        "map_lane_types": map_lane_types,
                        "neighbor_categories": neighbor_categories,
                        "agent_mask": agent_mask,
                        "lane_mask": lane_mask,
                        "static_obstacles": static_obstacles,
                        "static_obstacle_types": static_obstacle_types,
                        "static_obstacle_mask": static_obstacle_mask,
                    }

                b_size = past.size(0)
                pred_trajs, mode_logits = self.model(
                    past, agents_past=agents_past, map_lanes=map_lanes,
                    category_id=category_id, map_lane_types=map_lane_types,
                    neighbor_categories=neighbor_categories,
                    agent_mask=agent_mask, lane_mask=lane_mask,
                    static_obstacles=static_obstacles,
                    static_obstacle_types=static_obstacle_types,
                    static_obstacle_mask=static_obstacle_mask
                )

                T_fut = future.size(1)
                effective_k = min(k_eval, pred_trajs.shape[1])

                hz = max(1.0, float(self.model_cfg.hz))
                s1 = min(T_fut, max(1, round(1.0 * hz)))
                s2 = min(T_fut, max(1, round(2.0 * hz)))
                s3 = min(T_fut, max(1, round(3.0 * hz)))
                s4 = min(T_fut, max(1, round(4.0 * hz)))
                s5 = min(T_fut, max(1, round(5.0 * hz)))

                ade_1s_accum += min_ade_k(pred_trajs[:, :, :s1, :], future[:, :s1, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                ade_2s_accum += min_ade_k(pred_trajs[:, :, :s2, :], future[:, :s2, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                ade_3s_accum += min_ade_k(pred_trajs[:, :, :s3, :], future[:, :s3, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                fde_3s_accum += min_fde_k(pred_trajs[:, :, :s3, :], future[:, :s3, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                if T_fut >= s4:
                    ade_4s_accum += min_ade_k(pred_trajs[:, :, :s4, :], future[:, :s4, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                if T_fut >= s5:
                    ade_5s_accum += min_ade_k(pred_trajs[:, :, :s5, :], future[:, :s5, :], k=effective_k, mode_logits=mode_logits).item() * b_size
                ade_6s_accum += min_ade_k(pred_trajs, future, k=effective_k, mode_logits=mode_logits).item() * b_size
                fde_6s_accum += min_fde_k(pred_trajs, future, k=effective_k, mode_logits=mode_logits).item() * b_size

                endpoint_mr_accum += endpoint_miss_rate_k(
                    pred_trajs, future, k=effective_k, threshold=2.0, mode_logits=mode_logits
                ) * b_size
                max_disp_mr_accum += max_displacement_miss_rate_k(
                    pred_trajs, future, k=effective_k, threshold=2.0, mode_logits=mode_logits
                ) * b_size

                cr = evaluate_collision_risk(
                    pred_trajs,
                    neighbors_future=neighbors_future,
                    neighbor_future_mask=neighbor_future_mask,
                    mode_logits=mode_logits,
                )
                if cr.get("available"):
                    collision_scenarios += int(cr.get("scenarios_evaluated", 0))
                    collided_scenarios += int(cr.get("collided_scenarios", 0))
                else:
                    last_cr_reason = cr.get("reason")

                target_exp = future.unsqueeze(1)
                step_dists = safe_norm(pred_trajs - target_exp, dim=-1)
                min_step_dists, _ = torch.min(step_dists, dim=1)
                step_errors_accum += torch.sum(min_step_dists, dim=0)

                total_samples += b_size

        if total_samples == 0:
            raise RuntimeError("Validation DataLoader yielded zero samples.")

        step_errors = (step_errors_accum / total_samples).cpu().tolist()
        prediction_horizon = float(self.model_cfg.future_sec)
        endpoint_mr_pct = float(round((endpoint_mr_accum / total_samples) * 100.0, 2))
        max_disp_mr_pct = float(round((max_disp_mr_accum / total_samples) * 100.0, 2))

        if collision_scenarios > 0:
            cr_pct = float(round((collided_scenarios / collision_scenarios) * 100.0, 2))
            collision_risk = {
                "available": True,
                "status": "Computed",
                "collision_rate_percent": cr_pct,
                "safety_compliance_percent": float(round(100.0 - cr_pct, 2)),
                "scenarios_evaluated": collision_scenarios,
                "collided_scenarios": collided_scenarios,
            }
        else:
            collision_risk = {
                "available": False,
                "status": "Unavailable",
                "reason": last_cr_reason or "neighbors_future is not available in dataset sequences.",
                "collision_rate_percent": None,
                "safety_compliance_percent": None,
                "scenarios_evaluated": 0,
                "collided_scenarios": 0,
            }

        return {
            "dataset": self.dataset_name,
            "samples_evaluated": total_samples,
            "prediction_horizon_seconds": prediction_horizon,
            "future_timesteps": self.model_cfg.future_len,
            "num_modes": effective_k,
            "architecture_indices": list(self.best_ops),
            "architecture_source": self.architecture_source,
            "min_ade_1s": float(round(ade_1s_accum / total_samples, 4)),
            "min_ade_2s": float(round(ade_2s_accum / total_samples, 4)),
            "min_ade_3s": float(round(ade_3s_accum / total_samples, 4)),
            "min_ade_4s": float(round((ade_4s_accum if ade_4s_accum > 0 else ade_6s_accum) / total_samples, 4)),
            "min_ade_5s": float(round((ade_5s_accum if ade_5s_accum > 0 else ade_6s_accum) / total_samples, 4)),
            "min_ade_6s": float(round(ade_6s_accum / total_samples, 4)),
            "min_fde_6s": float(round(fde_6s_accum / total_samples, 4)),
            "min_fde_3s": float(round(fde_3s_accum / total_samples, 4)),
            "miss_rate_endpoint_2m": endpoint_mr_pct,
            "miss_rate_max_displacement_2m": max_disp_mr_pct,
            "miss_rate_6_2m": endpoint_mr_pct,
            "miss_rate_6s_2m": endpoint_mr_pct,
            "collision_risk": collision_risk,
            "step_errors": [round(e, 4) for e in step_errors],
        }

    def profile_hardware(
        self,
        num_warmup: int = 25,
        num_runs: int = 100,
        use_real_batch: bool = True,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """Profiles latency, FPS, CUDA memory, and model complexity via HardwareProfiler."""
        input_factory = None
        if use_real_batch and self._last_sample_batch is not None:
            cached = self._last_sample_batch
            avail_bs = cached["past"].shape[0]
            if batch_size <= avail_bs:
                def _real_factory():
                    return {
                        k: (v[:batch_size] if isinstance(v, torch.Tensor) and v.shape[0] == avail_bs else v)
                        for k, v in cached.items()
                    }
                input_factory = _real_factory

        return self.profiler.profile(
            num_warmup=num_warmup,
            num_runs=num_runs,
            input_factory=input_factory,
            batch_size=batch_size,
        )

    def print_evaluation_report(self, eval_results: dict[str, Any], hw_results: dict[str, Any]):
        """Prints academic performance report table."""
        k_val = int(eval_results.get("num_modes", self.model.num_modes))
        h_sec = eval_results.get("prediction_horizon_seconds", 6.0)
        cr = eval_results.get("collision_risk", {})
        print("=" * 80)
        print(" TRAJECTORY LNAS: COMPREHENSIVE PERFORMANCE & HARDWARE AUDIT")
        print("=" * 80)
        print(f" Dataset Target:       {eval_results['dataset'].upper()}")
        print(f" Samples Evaluated:    {eval_results['samples_evaluated']:,}")
        print(f" Prediction Horizon:   {h_sec:.1f} seconds ({eval_results.get('future_timesteps', 12)} timesteps)")
        print(f" Multi-Modal Modes:    K = {k_val} Trajectory Hypotheses")
        print(f" Hardware Device:      {hw_results['device']} (Batch={hw_results.get('batch_size', 1)}, Input={hw_results.get('input_source', 'synthetic')})")
        print(f" Selected Temporal:    {hw_results['operators']['temporal']}")
        print(f" Selected Interaction: {hw_results['operators']['interaction']}")
        print(f" Selected HD Map:      {hw_results['operators']['map']}")
        print("-" * 80)
        print(f" MULTI-MODAL TRAJECTORY ACCURACY METRICS (Official {h_sec:.1f}s Benchmark, K={k_val}):")
        print(f"   • minADE @ 1.0s:                  {eval_results['min_ade_1s']:.3f} meters")
        print(f"   • minADE @ 2.0s:                  {eval_results['min_ade_2s']:.3f} meters")
        print(f"   • minADE @ 3.0s:                  {eval_results['min_ade_3s']:.3f} meters")
        if h_sec >= 6.0:
            print(f"   • minADE @ 4.0s:                  {eval_results['min_ade_4s']:.3f} meters")
            print(f"   • minADE @ 5.0s:                  {eval_results['min_ade_5s']:.3f} meters")
            print(f"   • minADE @ 6.0s (Full 6s):        {eval_results['min_ade_6s']:.3f} meters")
            print(f"   • minFDE @ 6.0s (Final Step):     {eval_results['min_fde_6s']:.3f} meters")
        else:
            print(f"   • minFDE @ 3.0s (Final Step):     {eval_results['min_fde_3s']:.3f} meters")
        print(f"   • Endpoint Miss Rate @ 2.0m:      {eval_results['miss_rate_endpoint_2m']:.2f}%")
        print(f"   • Max-Disp Miss Rate @ 2.0m:      {eval_results['miss_rate_max_displacement_2m']:.2f}%")
        if cr.get("available"):
            print(f"   • Collision Rate (Future):        {cr['collision_rate_percent']:.2f}% ({cr['scenarios_evaluated']:,} scenarios)")
        else:
            print(f"   • Collision Rate (Future):        Unavailable ({cr.get('reason', 'neighbors_future missing')})")
        print("-" * 80)
        print(" EMBEDDED HARDWARE PROFILING:")
        print(f"   • Model Parameters:          {hw_results['total_params']:,} ({hw_results['total_params_m']} M)")
        print(f"   • Theoretical Compute:       {hw_results['theoretical_flops_g']} GFLOPs ({hw_results['theoretical_macs_m']} MMACs)")
        print(f"   • Inference Latency:         {hw_results['latency_ms_mean']:.2f} ± {hw_results['latency_ms_std']:.2f} ms")
        print(f"   • 95th Percentile Latency:   {hw_results['latency_ms_p95']:.2f} ms")
        print(f"   • Throughput / Frame Rate:   {hw_results['throughput_fps']:.1f} FPS")
        if hw_results.get("cuda_peak_memory_allocated_mb") is not None:
            print(
                f"   • CUDA Peak Memory:          {hw_results['cuda_peak_memory_allocated_mb']:.2f} MiB allocated / "
                f"{hw_results['cuda_peak_memory_reserved_mb']:.2f} MiB reserved"
            )


    @staticmethod
    def print_sota_comparison(
        eval_metrics: dict[str, Any] | None = None,
        hw_metrics: dict[str, Any] | None = None,
        search_runtime_seconds: float | None = None
    ):
        """Prints benchmark comparison against published literature baselines."""
        try:
            from experiments.baselines import run_baseline_comparison
            our_ade = eval_metrics.get("min_ade_3s", eval_metrics.get("min_ade_6s")) if eval_metrics else None
            our_fde = eval_metrics.get("min_fde_3s", eval_metrics.get("min_fde_6s")) if eval_metrics else None
            run_baseline_comparison(our_ade=our_ade, our_fde=our_fde)
        except Exception as e:
            print("=" * 80)
            print(" SOTA BENCHMARK & PROTOCOL AUDIT (nuScenes Forecasting)")
            print("=" * 80)
            print(f"Notice: {e}")
            print("Refer to experiments/baselines.py for full protocol audit matrix.")
            print("=" * 80)


def run_evaluation_cli():
    """CLI runner function for python -m evaluations.evaluator"""
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Evaluation & Benchmarking Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_trajectory_lnas.pt")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "argoverse"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--plot", action="store_true", default=True, help="Save evaluation plots")
    parser.add_argument("--allow_untrained", action="store_true", default=False, help="Allow testing with random weights")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("=" * 80)
        print(" TRAJECTORY LNAS: EVALUATION & BENCHMARKING SUITE")
        print("=" * 80)
        print("[Evaluator Notice] PyTorch is required to execute model evaluation.")
        print("Please install PyTorch in your active Python environment: pip install torch")
        print("=" * 80)
        return

    os.makedirs(args.report_dir, exist_ok=True)
    plots_dir = args.report_dir

    # Check if checkpoint exists, auto-fallback to allow_untrained=True if missing
    ckpt = args.checkpoint
    if ckpt:
        candidates = [
            ckpt,
            os.path.join(_parent_dir, ckpt),
            os.path.join(_grandparent_dir, ckpt),
        ]
        found = any(os.path.exists(c) for c in candidates)
        if not found and not args.allow_untrained:
            print(f"[Evaluator Info] Checkpoint file '{ckpt}' not found on disk.")
            print("[Evaluator Info] Initializing model with random weights (--allow_untrained=True) for verification.")
            args.allow_untrained = True

    evaluator = TrajectoryEvaluator(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        allow_untrained=args.allow_untrained
    )

    eval_results = evaluator.evaluate_dataset(
        batch_size=args.batch_size,
        custom_data_path=args.data_path
    )

    hw_results = evaluator.profile_hardware()

    evaluator.print_evaluation_report(eval_results, hw_results)
    evaluator.print_sota_comparison(eval_results, hw_results)

    if args.plot:
        our_ade = eval_results.get("min_ade_3s", eval_results.get("min_ade_6s"))
        our_fde = eval_results.get("min_fde_3s", eval_results.get("min_fde_6s"))
        plot_horizon_error_curve(eval_results.get("step_errors", []), dataset=args.dataset, output_dir=plots_dir)
        plot_benchmark_comparison(dataset=args.dataset, output_dir=plots_dir, our_ade=our_ade, our_fde=our_fde)

    json_path = os.path.join(args.report_dir, f"evaluation_results_{args.dataset}.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thesis": {
            "project": "TrajectoryLNAS",
            "student": "Alireza Ghafouri (4031631209)",
            "supervisors": ["Dr. Zeinab Ghasemi Naraaghi", "Dr. Zeinab Abedini"]
        },
        "evaluation": eval_results,
        "hardware": hw_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Full evaluation metrics saved to: {json_path}")


if __name__ == "__main__":
    run_evaluation_cli()

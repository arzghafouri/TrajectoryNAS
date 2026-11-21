r"""Hardware profile for the recorded TrajectoryLNAS run.

Saved measurements describe the RTX 3050 Laptop GPU run. Live measurements
are machine-specific and are not used for cross-paper efficiency rankings.
"""

import os
import sys
from typing import Any

# Ensure workspace root is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_curr_dir, ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from experiments.results import get_discovered_architecture, load_real_experiment_results



def load_real_run_profiling_stats() -> dict[str, Any]:
    """Build the recorded hardware profile from the latest experiment JSON."""
    results = load_real_experiment_results()
    profile = results.get("edge_hardware_profiling", {})
    architecture = get_discovered_architecture(results)
    return {
        "device": profile.get("device", "unknown"),
        "gpu_name": results.get("environment", {}).get("device", "Not recorded"),
        "latency_ms_mean": profile.get("latency_ms_mean"),
        "latency_ms_std": profile.get("latency_ms_std"),
        "latency_ms_p95": profile.get("latency_ms_p95"),
        "throughput_fps": profile.get("throughput_fps"),
        "total_params": profile.get("total_params"),
        "total_params_m": profile.get("total_params_m"),
        "trainable_params": profile.get("trainable_params"),
        "theoretical_flops_g": profile.get("theoretical_flops_g"),
        "operators": {
            branch: f"{item.get('name', 'Unknown')} (Index {item.get('index', 'N/A')})"
            for branch, item in architecture.items()
            if isinstance(item, dict)
        },
        "results_source": str(os.path.relpath(
            os.path.join(_root_dir, "reports", "real_experiment_results.json"), _root_dir
        )),
    }


# Refresh this shared dictionary in place at each report invocation.
REAL_RUN_PROFILING_STATS = load_real_run_profiling_stats()


def run_hardware_benchmark(use_gpu: bool = True, num_runs: int = 50) -> dict[str, Any]:
    """Runs hardware latency and throughput profiling."""
    latest_stats = load_real_run_profiling_stats()
    REAL_RUN_PROFILING_STATS.clear()
    REAL_RUN_PROFILING_STATS.update(latest_stats)
    results = load_real_experiment_results()
    architecture = get_discovered_architecture(results)
    indices = {
        name: architecture.get(name, {}).get("index")
        for name in ("temporal", "interaction", "map")
    }
    print("=" * 80)
    print("TrajectoryLNAS: Hardware Profiler")
    print(f"Recorded Hardware: {REAL_RUN_PROFILING_STATS['gpu_name']}")
    print(f"Recorded Architecture: {REAL_RUN_PROFILING_STATS['operators']}")
    print(f"Results Source: {REAL_RUN_PROFILING_STATS['results_source']}")
    print("=" * 80)

    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass

    stats = REAL_RUN_PROFILING_STATS

    if has_torch:
        from evaluations.metrics import HardwareProfiler
        from models.final_model import FinalTrajectoryLNAS
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        print(f"[*] Live PyTorch Environment Detected: {device}")
        try:
            best_t = indices.get("temporal") if indices.get("temporal") is not None else 3
            best_i = indices.get("interaction") if indices.get("interaction") is not None else 0
            best_m = indices.get("map") if indices.get("map") is not None else 1
            model = FinalTrajectoryLNAS(
                best_temporal_idx=best_t,
                best_interaction_idx=best_i,
                best_map_idx=best_m,
                d_model=128
            ).to(device)

            profiler = HardwareProfiler(model, device=device)
            live_stats = profiler.profile(num_warmup=15, num_runs=num_runs)
            print(f"[*] Live Latency (mean): {live_stats['latency_ms_mean']} ms (Std: {live_stats['latency_ms_std']} ms)")
            print(f"[*] Live Latency (P95):  {live_stats['latency_ms_p95']} ms")
            print(f"[*] Live Throughput:     {live_stats['throughput_fps']} FPS")
            print(f"[*] Total Parameters:    {live_stats['total_params_m']} M ({live_stats['total_params']:,})")
            print(f"[*] Theoretical FLOPs:   {live_stats['theoretical_flops_g']} GigaFLOPs")
            stats = live_stats
        except Exception as e:
            print(f"[!] Live profiling failed; reporting the latest recorded results: {e}")

    print("-" * 80)
    print(f" HARDWARE PROFILE SUMMARY ({stats.get('device', 'N/A')}):")
    print(f"   Mean Inference Latency:   {stats['latency_ms_mean']} ms (± {stats['latency_ms_std']} ms)")
    print(f"   95th-Percentile Latency:  {stats['latency_ms_p95']} ms")
    print(f"   Throughput:               {stats['throughput_fps']} FPS")
    print(f"   Total Parameters:         {stats['total_params_m']} M ({stats['total_params']:,} params)")
    print(f"   Theoretical FLOPs:        {stats['theoretical_flops_g']} GFLOPs")
    print("   Literature efficiency ranking is unavailable from this single-device profile.")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    run_hardware_benchmark()

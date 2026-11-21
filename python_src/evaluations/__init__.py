"""
Evaluation Package for TrajectoryLNAS
====================================
Consolidated metrics suite (geometric, hardware, zero-cost proxies),
scientific visualization, and master evaluator.
"""

try:
    from .metrics import (
        HardwareProfiler,
        ade,
        best_of_k_loss,
        endpoint_miss_rate_k,
        evaluate_collision_risk,
        fde,
        get_supernet_complexity,
        get_supernet_costs,
        get_supernet_flops,
        max_displacement_miss_rate_k,
        min_ade_k,
        min_fde_k,
        miss_rate,
        miss_rate_k,
        simple_complexity_proxy,
        snip_score,
        synflow_score,
    )
except ImportError:
    HardwareProfiler = None
    ade = None
    best_of_k_loss = None
    endpoint_miss_rate_k = None
    evaluate_collision_risk = None
    fde = None
    get_supernet_complexity = None
    get_supernet_costs = None
    get_supernet_flops = None
    max_displacement_miss_rate_k = None
    min_ade_k = None
    min_fde_k = None
    miss_rate = None
    miss_rate_k = None
    simple_complexity_proxy = None
    snip_score = None
    synflow_score = None

from .evaluator import TrajectoryEvaluator, run_evaluation_cli
from .visualization import (
    plot_architecture_search_trends,
    plot_benchmark_comparison,
    plot_horizon_error_curve,
    plot_training_convergence_dynamics,
)

__all__ = [
    "HardwareProfiler",
    "TrajectoryEvaluator",
    "ade",
    "best_of_k_loss",
    "endpoint_miss_rate_k",
    "evaluate_collision_risk",
    "fde",
    "get_supernet_complexity",
    "get_supernet_costs",
    "get_supernet_flops",
    "max_displacement_miss_rate_k",
    "min_ade_k",
    "min_fde_k",
    "miss_rate",
    "miss_rate_k",
    "plot_architecture_search_trends",
    "plot_benchmark_comparison",
    "plot_horizon_error_curve",
    "plot_training_convergence_dynamics",
    "run_evaluation_cli",
    "simple_complexity_proxy",
    "snip_score",
    "synflow_score",
]

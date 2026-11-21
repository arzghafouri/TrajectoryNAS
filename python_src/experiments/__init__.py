"""
Experiments Package for TrajectoryLNAS
======================================
Contains component ablation studies, literature protocol comparisons,
and hardware profiling suites grounded in:
  1. The latest recorded search, evaluation, and hardware results in
     reports/real_experiment_results.json.
  2. Khaled Saleh, "Pedestrian Trajectory Prediction for Real-Time Autonomous Systems
     via Context-Augmented Transformer Networks", Sensors 2022, 22(19), 7495.
     DOI: https://doi.org/10.3390/s22197495
"""

from experiments.ablation import ABLATION_CONFIGURATIONS, run_ablation_study
from experiments.baselines import (
    LITERATURE_AUDIT_BENCHMARKS,
    SOTA_BASELINES,
    ConstantVelocityBaseline,
    run_baseline_comparison,
)
from experiments.benchmark import REAL_RUN_PROFILING_STATS, run_hardware_benchmark


def run_all_experiments():
    """Runs the full academic experimental suite."""
    print("\n" + "#" * 90)
    print(" TRAJECTORY LNAS: EXECUTING FULL EXPERIMENTAL SUITE")
    print("#" * 90 + "\n")

    abl_results = run_ablation_study()
    print("\n")
    base_results = run_baseline_comparison()
    print("\n")
    bench_results = run_hardware_benchmark()

    print("\n" + "#" * 90)
    print(" EXPERIMENT SUITE COMPLETED")
    print("#" * 90 + "\n")
    return {
        "ablation": abl_results,
        "baselines": base_results,
        "hardware_benchmark": bench_results,
    }


__all__ = [
    "ABLATION_CONFIGURATIONS",
    "REAL_RUN_PROFILING_STATS",
    "LITERATURE_AUDIT_BENCHMARKS",
    "SOTA_BASELINES",
    "ConstantVelocityBaseline",
    "run_ablation_study",
    "run_all_experiments",
    "run_baseline_comparison",
    "run_hardware_benchmark",
]

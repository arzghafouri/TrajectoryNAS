r"""
Research Evaluation Pipeline Execution Configuration
==================================================
Centralized Boolean execution controls, resumability settings, and persistence
options for `experiments/research_gap_evaluation.py`.

This file controls ONLY which evaluation stages run and how execution state is
persisted and resumed. Evaluation algorithms and metrics remain inside
`experiments/research_gap_evaluation.py`.
"""

# ------------------------------------------------------------------------------
# Stage 1: Mandatory Data Integrity & Evaluation Reliability Audit
# ------------------------------------------------------------------------------
RUN_AUDIT: bool = True

# ------------------------------------------------------------------------------
# Stage 2: Research Gap A — Factorized Three-Branch NAS Ablation (A0-A7)
# ------------------------------------------------------------------------------
RUN_A0: bool = True   # Temporal=Fixed,  Interaction=Fixed,  Map=Fixed
RUN_A1: bool = True   # Temporal=Search, Interaction=Fixed,  Map=Fixed
RUN_A2: bool = True   # Temporal=Fixed,  Interaction=Search, Map=Fixed
RUN_A3: bool = True   # Temporal=Fixed,  Interaction=Fixed,  Map=Search
RUN_A4: bool = True   # Temporal=Search, Interaction=Search, Map=Fixed
RUN_A5: bool = True   # Temporal=Search, Interaction=Fixed,  Map=Search
RUN_A6: bool = True   # Temporal=Fixed,  Interaction=Search, Map=Search
RUN_A7: bool = True   # Temporal=Search, Interaction=Search, Map=Search (Full DARTS)

# ------------------------------------------------------------------------------
# Stage 3: Research Gap B — Hardware-Aware NAS Profiling
# ------------------------------------------------------------------------------
RUN_HARDWARE: bool = True
RUN_HARDWARE_LAMBDA_SEARCH: bool = False  # Enable full search+train lambda_flops sweep

# ------------------------------------------------------------------------------
# Stage 4: Research Gap C — Scene Complexity Stratification & Prototype
# ------------------------------------------------------------------------------
RUN_SCENE: bool = True
RUN_SCENE_PROTOTYPE: bool = False         # ExperimentalSceneConditionedPrototype (depends on SCENE)

# ------------------------------------------------------------------------------
# Stage 5: Warmup & Temperature Annealing Optimization Ablation (T0-T3)
# ------------------------------------------------------------------------------
RUN_T0: bool = True   # No warmup, fixed temperature (T=T_start)
RUN_T1: bool = True   # Warmup enabled, fixed temperature
RUN_T2: bool = True   # No warmup, temperature annealing enabled
RUN_T3: bool = True   # Warmup + temperature annealing enabled (default TrajectoryLNAS)

# ------------------------------------------------------------------------------
# General Execution & Resumability Settings
# ------------------------------------------------------------------------------
SKIP_COMPLETED: bool = True
FORCE_RERUN: bool = False
CONTINUE_ON_ERROR: bool = False
REQUIRE_AUDIT: bool = False

# ------------------------------------------------------------------------------
# Output, Reporting, and Persistence
# ------------------------------------------------------------------------------
SAVE_INTERMEDIATE_RESULTS: bool = True
SAVE_EXECUTION_STATUS: bool = True
GENERATE_PLOTS: bool = True
GENERATE_REPORT: bool = True

# Default output directory used when neither --output-dir nor --resume is passed.
# Relative paths resolve inside the project root (`python_src/`), e.g.:
#   python_src/reports/research_gaps/pipeline
# Set to None to create a new timestamped directory under
# `python_src/reports/research_gaps/` for each fresh run.
OUTPUT_DIR: str | None = "reports/research_gaps/pipeline"
STATUS_FILENAME: str = "execution_status.json"

# Task: Implement an Independent Research Experiments and Evaluation Module for TrajectoryLNAS

You are an experienced Deep Learning Research Engineer specializing in Neural Architecture Search (NAS), autonomous driving, multi-agent trajectory forecasting, and PyTorch.

Your task is to inspect the existing TrajectoryLNAS repository and create a new, self-contained research experiment runner in the existing `experiments/` or `evaluations/` directory.

The purpose of this module is to experimentally investigate and evaluate the three main research gaps identified for the TrajectoryLNAS research paper.

**Critical requirement: Minimize modifications to the existing project.**

Do not redesign or rewrite the existing NAS algorithm. Do not make significant changes to existing model, training, dataset, or configuration files. Prefer implementing new experiment logic in one new Python file that imports and reuses existing classes, functions, and configurations.

Suggested filename:

`experiments/research_gap_evaluation.py`

If the repository structure makes `evaluations/` more appropriate, use that directory instead. Inspect the actual repository before deciding.

---

# 1. First: Inspect the Existing Implementation

Before writing code, inspect and understand the following modules and their dependencies:

- `darts_algorithm.py`
- `search_supernet.py`
- `supernet_operators.py`
- `final_model.py`
- `final_trainer.py`
- `SearchConfig`
- `ModelConfig`
- Dataset loading and preprocessing modules
- Existing evaluation metrics
- Checkpoint saving and loading mechanisms
- Existing experiment and report directories

Identify the actual interfaces, configuration fields, tensor shapes, operator names, checkpoint formats, and dataset fields.

Do not assume that configuration values or operator lists are identical to previously documented values. Read them from the actual project.

The new module must be compatible with the current implementation.

If an experiment cannot be performed using the existing interfaces, document the limitation rather than silently changing the original architecture.

# 2. Research Gap A: Factorized Three-Branch Neural Architecture Search

## Research question

Does independently searching the Temporal, Interaction, and HD Map branches provide an advantage over fixed architectures or partially searched architectures?

The current TrajectoryLNAS architecture has three independently parameterized branches:

- Temporal branch
- Interaction branch
- HD Map branch

Implement an ablation study that evaluates the contribution of each branch.

## Required experiments

At minimum, implement the following experimental configurations:

| Experiment | Temporal | Interaction | HD Map |
| ---------- | -------- | ----------- | ------ |
| A0         | Fixed    | Fixed       | Fixed  |
| A1         | Search   | Fixed       | Fixed  |
| A2         | Fixed    | Search      | Fixed  |
| A3         | Fixed    | Fixed       | Search |
| A4         | Search   | Search      | Fixed  |
| A5         | Search   | Fixed       | Search |
| A6         | Fixed    | Search      | Search |
| A7         | Search   | Search      | Search |

For fixed branches, use clearly documented fixed operator selections. Prefer the operators selected by the existing search result as the reference architecture, unless a different baseline is required by the existing implementation.

For searched branches, reuse the existing DARTS architecture search implementation wherever possible.

Do not duplicate the entire DARTS implementation unless absolutely necessary.

If the current DARTS implementation cannot directly freeze individual branches, implement a minimal experiment-specific wrapper or controlled architecture-parameter update mechanism in the new file. Do not modify the original DARTS source.

## Evaluation

For each configuration, record:

- minADE\@K
- minFDE\@K
- MissRate\@2m
- Parameter count
- Theoretical FLOPs
- Search duration
- Final training duration
- Peak GPU memory
- Selected operators in each branch

Use identical data splits, training budgets, random seeds, and evaluation protocols across comparable experiments.

Run at least three seeds when computational resources permit. If only one seed is used, explicitly mark the result as preliminary.

The report must distinguish search-supernetwork metrics from standalone final-model metrics.

# 3. Research Gap B: Hardware-Aware Neural Architecture Search

## Research question

Does optimizing theoretical computational cost result in an actual improvement in inference efficiency on the target hardware?

The current implementation includes an expected-FLOPs penalty in the architecture objective.

However, theoretical FLOPs alone are not sufficient to establish hardware efficiency.

Implement a hardware profiling experiment for the extracted standalone architecture.

## Required measurements

Measure and report:

1. Total model parameters.
2. Theoretical FLOPs.
3. GPU inference latency.
4. CPU inference latency, if available.
5. Peak GPU memory usage.
6. Model checkpoint size.
7. Search duration.
8. Final model training duration.

## Profiling requirements

- Use the actual extracted final model, not only the mixed-operation Supernet.
- Include configurable warmup iterations before timing.
- Synchronize CUDA before and after GPU timing.
- Run multiple inference iterations.
- Report mean, median, standard deviation, and suitable percentile latency statistics.
- Reset and record CUDA peak memory statistics.
- Report the hardware model, PyTorch version, CUDA version, and precision mode.
- Support CPU-only execution when CUDA is unavailable.
- Do not report theoretical FLOPs as measured latency or energy consumption.

If possible, compare:

- A manually selected fixed architecture.
- The architecture selected by TrajectoryLNAS.
- Architectures produced by different FLOPs penalty strengths.

For the FLOPs penalty experiment, support a configurable list of lambda values. Use the existing SearchConfig mechanism and do not hard-code values without checking the current configuration.

For example, allow the user to configure:

`lambda_flops_values = [0.0, 0.0001, 0.001, 0.01]`

These are experimental candidates, not assumed optimal values.

For each value, report the accuracy-efficiency trade-off.

Generate an accuracy-versus-latency plot and an accuracy-versus-FLOPs plot.

If the required architecture checkpoints do not exist, do not fabricate comparisons. Mark those experiments as pending and explain how to generate the missing checkpoints.

# 4. Research Gap C: Scene-Adaptive Architecture Evaluation

## Research question

Does the best architecture depend on the complexity and structure of the traffic scene?

The current implementation extracts a single architecture for the dataset. This experiment investigates whether different scene types exhibit different forecasting requirements.

## Scene complexity analysis

Use the actual available dataset fields to define scene-level or sample-level complexity indicators.

Potential indicators include:

- Number of surrounding agents.
- Agent density.
- Number of valid HD map lanes.
- Presence or absence of map information.
- Number of static obstacles.
- Interaction density.
- Traffic intersection or lane-change indicators, if available.

Do not invent features that are not present in the dataset.

If a feature is unavailable, explicitly report that it cannot be evaluated.

## Scene groups

Create configurable complexity groups:

- Low complexity
- Medium complexity
- High complexity

Use transparent thresholds based on the actual dataset distribution, such as quantiles.

Do not select arbitrary thresholds without justification.

Where sufficient information exists, additionally analyze:

- High-density scenes.
- Low-density scenes.
- Map-rich scenes.
- Map-poor or map-missing scenes.
- Intersection-related scenes.
- Lane-change-related scenes.

## Evaluation

Evaluate the same trained architecture separately on each group.

For every group, report:

- Number of samples.
- minADE\@K.
- minFDE\@K.
- MissRate\@2m.
- Average number of neighboring agents.
- Average number of valid lanes.
- Relevant complexity statistics.

Generate plots comparing forecasting performance across scene groups.

## Important distinction

Stratified evaluation of a fixed architecture is not, by itself, proof of Scene-Adaptive NAS.

If the existing implementation does not support dynamic architecture selection, do not claim that it does.

Instead:

1. Implement the scene-stratified diagnostic evaluation first.
2. If technically feasible without modifying the core architecture, implement an experimental prototype of scene-conditioned operator selection in this new module.
3. Compare the prototype against the globally selected fixed architecture.
4. Clearly label the prototype as experimental and report whether it was actually trained and evaluated.

Do not claim an adaptive architecture improvement unless it is supported by measured results.

# 5. Data Integrity and Evaluation Reliability

This section is mandatory.

Before running experiments, verify:

- Actual training and validation sample counts.
- Dataset version and paths.
- Scene-level train/validation separation.
- Missing or invalid samples.
- Availability of neighboring-agent data.
- Availability of HD map features.
- Validity of agent and lane masks.
- Number of prediction modes K.
- Prediction horizon and sampling frequency.

Important implementation checks:

1. Derive K from the actual model configuration or model output.
2. Do not hard-code K=6 in evaluation if the model uses another number of modes.
3. Check consistency between the model's number of output modes and the metric implementation.
4. Verify that minADE, minFDE, and MissRate use the same K.
5. Verify that invalid or missing map inputs are handled consistently.
6. Detect zero-coverage HD-map situations and report them prominently.
7. Ensure that validation data is not used for final test reporting.
8. Ensure that the same samples are evaluated across comparable experiments.

If any mismatch is detected, produce a warning and stop the affected comparison rather than silently changing the metric or dataset.

Do not modify existing dataset files.

# 6. Warmup and Temperature Annealing Ablation

Since TrajectoryLNAS already implements architecture warmup and temperature-scaled softmax, investigate their contribution as an additional optimization ablation.

Implement a configurable comparison:

| Experiment | Architecture Warmup | Temperature Annealing |
| ---------- | ------------------- | --------------------- |
| T0         | Disabled            | Disabled              |
| T1         | Enabled             | Disabled              |
| T2         | Disabled            | Enabled               |
| T3         | Enabled             | Enabled               |

Reuse the existing implementation and configuration fields wherever possible.

Record:

- Validation minADE\@K and minFDE\@K.
- Architecture parameter distributions.
- Selected operator stability.
- Search duration.
- Final model metrics.

Do not modify the original DARTS implementation. If controlling warmup or temperature independently requires an experiment-specific wrapper, implement it in the new file.

Report the actual temperature schedule and warmup duration used in each experiment.

# 7. Experimental Execution and Resource Management

The project may run on GPUs with limited VRAM, including 4 GB laptop GPUs.

Therefore:

- Support CPU and CUDA.
- Reuse existing AMP and gradient checkpointing options.
- Use configurable batch sizes.
- Support gradient accumulation where applicable.
- Avoid loading multiple large models simultaneously.
- Release model and tensor references after each experiment.
- Use `torch.cuda.empty_cache()` where appropriate.
- Catch CUDA out-of-memory errors and report the failed experiment.
- Do not automatically reduce the dataset or change the experiment protocol without recording it.
- Allow experiments to be executed individually or as a complete suite.
- Support resuming completed experiments.
- Save experiment configuration and random seeds.

Provide command-line options similar to:

`python experiments/research_gap_evaluation.py --experiment all`

`python experiments/research_gap_evaluation.py --experiment factorized`

`python experiments/research_gap_evaluation.py --experiment hardware`

`python experiments/research_gap_evaluation.py --experiment scene`

`python experiments/research_gap_evaluation.py --experiment warmup_temperature`

`python experiments/research_gap_evaluation.py --experiment audit`

Also support:

- `--seed`
- `--device`
- `--batch-size`
- `--max-experiments`
- `--resume`
- `--dry-run`

Adapt the CLI to the existing project conventions.

# 8. Reporting and Visualization

Create a separate output directory for each execution, for example:

`report/research_gaps/<timestamp>/`

The experiment runner must automatically generate:

## Machine-readable outputs

- `experiment_config.json`
- `dataset_audit.json`
- `factorized_nas_results.csv`
- `hardware_profile_results.csv`
- `scene_complexity_results.csv`
- `warmup_temperature_results.csv`
- `summary.json`

Only create files relevant to experiments that actually ran.

## Research report

Generate:

`research_gap_report.md`

The report must contain:

1. Executive summary.
2. Dataset and data integrity audit.
3. Factorized NAS ablation results.
4. Hardware efficiency results.
5. Scene complexity analysis.
6. Warmup and temperature annealing ablation.
7. Comparison tables.
8. Figures and charts.
9. Failed or incomplete experiments.
10. Limitations and threats to validity.
11. Conclusions supported by actual measurements.
12. Recommended next experiments.

Generate plots using Matplotlib.

Required figures:

- Performance comparison across branch ablations.
- FLOPs versus minADE.
- Inference latency versus minADE.
- GPU memory comparison.
- Scene complexity versus prediction error.
- Operator selection frequency across seeds, if multiple seeds exist.
- Warmup/temperature ablation comparison.

All charts must be generated from saved experiment results, not manually entered values.

Use `NaN` or an explicit missing-value field for unavailable measurements. Never insert fabricated metrics.

# 9. Comparison with Published Research

Include a literature comparison table in the Markdown report.

At minimum, include:

- VectorNet
- LaneGCN
- HiVT
- TrajectoryNAS
- TrajGNAS
- TrajectoryLNAS

Compare:

- Architecture/search paradigm.
- Temporal modeling.
- Agent interaction modeling.
- HD-map modeling.
- NAS strategy.
- Hardware cost objective.
- Scene-adaptive behavior.
- Dataset and evaluation protocol.

Use published results only when their exact source and evaluation protocol are available.

Do not compare numerical ADE/FDE values from different datasets, different K values, or different evaluation protocols as if they were directly comparable.

Clearly distinguish published literature values from newly measured TrajectoryLNAS results.

# 10. Code Quality and Non-Regression Requirements

The new module must:

- Be modular and maintainable.
- Use type hints where appropriate.
- Include docstrings.
- Include clear logging.
- Handle exceptions and incomplete experiments.
- Avoid duplicated implementations of existing models and metrics.
- Reuse existing project components.
- Avoid unnecessary dependencies.
- Be executable from the project root.
- Support Windows paths.
- Avoid assumptions about a specific GPU.

**Do not modify existing core files unless absolutely necessary.**

The preferred solution is a new standalone experiment file.

If a change to an existing file is unavoidable:

1. Explain why it is necessary.
2. Keep the change minimal.
3. Identify the exact file and lines affected.
4. Preserve backward compatibility.
5. Do not change default training or search behavior.

Do not change existing model architecture, dataset preprocessing, default SearchConfig values, or checkpoint formats.

# 11. Final Deliverables

After implementation:

1. Create the new experiment runner.
2. Run static syntax and import checks.
3. Run `--dry-run` to validate configuration and experiment discovery.
4. Run the dataset audit if the configured dataset is accessible.
5. Run a small smoke test only if it does not alter the actual research protocol.
6. Confirm that existing training and search entry points remain unchanged.
7. Provide the exact command for running each experiment.
8. Explain which research gaps can be evaluated immediately and which require additional training or checkpoints.
9. List any unresolved issues.
10. Provide a concise implementation summary.

**Most important scientific requirement:**

The module must distinguish between:

- Implemented functionality.
- Experiments successfully executed.
- Results actually measured.
- Experiments that are pending.
- Hypotheses that remain unverified.

Never report an experiment as successful merely because its code was created.

The final goal is to make TrajectoryLNAS experimentally testable against its three research gaps while preserving the existing project architecture and minimizing regression risk.

**Output Directory Requirement**

The project already contains a `reports/` directory in its root. Use this existing directory as the output location.

Save all research gap evaluation outputs under:

`reports/research_gaps/<timestamp>/`

Do not create a separate `report/` directory or save outputs inside `python_src/`.

The script must:

- Automatically create the required subdirectories if they do not exist.
- Save all experiment results as structured JSON files.
- Save aggregated results as CSV.
- Generate a Markdown summary report.
- Preserve previous experiment results by creating a unique timestamped directory for every run.
- Include experiment status (`completed`, `failed`, `not_run`) and error details where applicable.
- Record the configuration, dataset information, random seed, hardware specifications, and execution timestamp in the JSON outputs.
- Never replace missing or unmeasured values with fabricated numbers or zero.
- Print the final output directory when execution finishes.

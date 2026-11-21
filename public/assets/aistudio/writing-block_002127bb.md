# Task: Implement Configurable and Resumable Research Evaluation Pipeline in TrajectoryLNAS

Inspect the existing TrajectoryLNAS project, focusing specifically on:

- `research_gap_evaluation.py`
- The existing evaluation stages A0–A7 and any other research evaluations invoked by this script.
- The current configuration structure and output directories used by the research evaluation pipeline.

The goal is to implement a configurable, resumable execution system controlled by a new file:

`config/evaluate_config.py`

**Scope restriction:** Only implement the configuration and execution-management functionality for `research_gap_evaluation.py`. Do not modify `evaluator.py` or `visualization.py` in this task.

## 1. Create config/evaluate_config.py

Create a centralized configuration file containing Boolean parameters for all existing evaluation stages.

At minimum, inspect the actual implementation and define parameters for:

- A0
- A1
- A2
- A3
- A4
- A5
- A6
- A7

Also identify any additional evaluations already implemented or invoked by `research_gap_evaluation.py` and expose their execution controls as Boolean parameters.

Example:

```python
# config/evaluate_config.py

RUN_A0 = True
RUN_A1 = True
RUN_A2 = True
RUN_A3 = True
RUN_A4 = True
RUN_A5 = True
RUN_A6 = True
RUN_A7 = True

# General execution settings
SKIP_COMPLETED = True
FORCE_RERUN = False
CONTINUE_ON_ERROR = False

# Output and persistence
SAVE_INTERMEDIATE_RESULTS = True
SAVE_EXECUTION_STATUS = True
```

These are illustrative names. Use the actual evaluation identifiers found in the project.

Every existing evaluation stage must have an explicit Boolean execution parameter.

## 2. Integrate configuration into research_gap_evaluation.py

Import and respect the configuration file.

Execution rules:

- `True`: execute the corresponding evaluation if it is eligible to run.
- `False`: skip that evaluation completely.
- Disabled evaluations must not consume computational resources.
- Preserve the original evaluation order.
- Preserve the existing evaluation algorithms and research methodology.
- Print a clear execution plan before starting.

Do not move evaluation algorithms into the configuration file. It must only control execution.

## 3. Implement resumable execution

This is the most important requirement.

The user may run the system for several hours, complete A0–A3, and postpone the remaining evaluations.

For example, the initial configuration may enable all stages:

```python
RUN_A0 = True
RUN_A1 = True
RUN_A2 = True
RUN_A3 = True
RUN_A4 = True
RUN_A5 = True
RUN_A6 = True
RUN_A7 = True
```

After A0–A3 have completed, the user can change the configuration:

```python
RUN_A0 = False
RUN_A1 = False
RUN_A2 = False
RUN_A3 = False
RUN_A4 = True
RUN_A5 = True
RUN_A6 = True
RUN_A7 = True
```

The next execution must run only A4–A7, without repeating A0–A3.

## 4. Persistent execution status

Implement a persistent status-tracking mechanism, preferably using a JSON file in the existing research evaluation output directory.

For every evaluation stage, record:

- Evaluation identifier.
- Execution status.
- Start timestamp.
- Completion timestamp.
- Duration.
- Output file paths.
- Error information, if applicable.

Use the following statuses:

- PENDING
- RUNNING
- COMPLETED
- FAILED
- SKIPPED

The system must distinguish between disabled evaluations and previously completed evaluations.

### Execution behavior

`SKIP_COMPLETED = True`:
- Skip evaluations previously completed successfully.
- Verify their expected output files before treating them as completed.

`FORCE_RERUN = True`:
- Allow explicitly enabled evaluations to run again even if they were previously completed.
- Preserve previous results unless overwriting is explicitly intended.

`CONTINUE_ON_ERROR = False`:
- Stop the pipeline when an evaluation fails.

`CONTINUE_ON_ERROR = True`:
- Continue with other eligible evaluations when a stage fails.
- Never mark failed evaluations as completed.

If execution is interrupted, the next run must safely recover from the previous status.

A stage marked RUNNING without a valid completion record must not be treated as completed.

## 5. Respect evaluation dependencies

Inspect the existing code to identify actual dependencies between A0–A7 and other evaluations.

Do not invent dependencies.

Before running an evaluation, verify that its required prerequisites are completed and their expected outputs exist.

Do not silently execute a disabled prerequisite.

If a prerequisite is disabled and incomplete, clearly report the dependency and skip the dependent evaluation or raise an informative configuration error.

## 6. Execution summary

Before starting expensive computations, print a clear execution plan.

For example:

```text
Research Evaluation Pipeline
--------------------------------
A0: SKIP (disabled)
A1: SKIP (already completed)
A2: SKIP (disabled)
A3: SKIP (already completed)
A4: RUN
A5: RUN
A6: RUN
A7: RUN
--------------------------------
Scheduled evaluations: 4
Previously completed: 2
Disabled evaluations: 2
```

The summary must reflect actual execution decisions, including completion status and dependency constraints.

After execution, print a final summary of successful, failed, skipped, and pending stages.

## 7. Preserve existing evaluation behavior

Do not rewrite or modify the mathematical implementation of any evaluation.

Do not change:

- Research-gap definitions.
- Metric calculations.
- Dataset handling.
- Model inference.
- Checkpoint selection.
- Statistical methodology.
- Existing result formats unless absolutely necessary.

Keep the existing evaluation functions intact and introduce configuration checks around their execution.

Avoid unnecessary refactoring.

## 8. Configuration validation

Before expensive execution, validate:

- All Boolean parameters have valid Boolean values.
- All evaluation identifiers are recognized.
- Required input files and checkpoints exist.
- Output directories are accessible.
- Evaluation dependencies are satisfied.
- Configuration options are not contradictory.

Provide clear error messages.

## 9. Testing

Add lightweight tests using mocked evaluation functions.

Verify:

1. Each Boolean parameter correctly controls its evaluation.
2. Disabled evaluations are never executed.
3. Completed evaluations are skipped when `SKIP_COMPLETED=True`.
4. A0–A3 can be completed in one run and A4–A7 in a subsequent run.
5. Failed evaluations are not marked as completed.
6. Interrupted executions can be recovered safely.
7. `FORCE_RERUN` works correctly.
8. Evaluation dependencies are respected.
9. Previous results are preserved.
10. Existing evaluation functions and their mathematical behavior remain unchanged.

Do not require a complete nuScenes training or evaluation run to test the execution manager.

## 10. Strict scope restrictions

For this task:

- Modify `research_gap_evaluation.py` only as necessary.
- Create `config/evaluate_config.py`.
- Add a small status-tracking utility module only if necessary.
- Add focused tests.
- Do not modify `evaluator.py`.
- Do not modify `visualization.py`.
- Do not implement new evaluation metrics.
- Do not redesign the research evaluation methodology.

## 11. Final deliverables

Provide:

1. The new `config/evaluate_config.py`.
2. The modified `research_gap_evaluation.py`.
3. Any necessary status-tracking utility.
4. Tests and their results.
5. A list of all Boolean execution parameters.
6. The actual dependency structure between evaluation stages.
7. Example configurations for:
   - Running all evaluations.
   - Running A0–A3 only.
   - Continuing with A4–A7.
   - Rerunning selected evaluations.

**Critical acceptance criterion:** A user must be able to complete A0–A3, change their Boolean parameters to False, enable A4–A7, and resume the pipeline later without rerunning completed stages or losing previous results.
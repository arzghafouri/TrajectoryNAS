r"""
Execution Management, Configuration Validation, and Persistent Status Tracking
==============================================================================
Provides resumable stage-level orchestration for `research_gap_evaluation.py`:
  - Strict validation of Boolean configuration flags, identifiers, inputs,
    output directory accessibility, dependencies, and contradictions.
  - Atomic JSON status persistence (`execution_status.json`) recording per-stage
    status (PENDING, RUNNING, COMPLETED, FAILED, SKIPPED), timestamps, duration,
    output file paths, and error metadata.
  - Safe recovery from interrupted (`RUNNING`) stages and verification of
    expected output files before honoring `SKIP_COMPLETED`.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
VALID_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
)


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    config_flag: str
    section: str
    dependencies: tuple[str, ...]
    requires_dataset: bool
    description: str


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("AUDIT", "RUN_AUDIT", "audit", (), False, "Data Integrity & Evaluation Reliability Audit"),
    StageSpec("A0", "RUN_A0", "factorized", (), True, "Factorized NAS A0: Temporal=Fixed, Interaction=Fixed, Map=Fixed"),
    StageSpec("A1", "RUN_A1", "factorized", (), True, "Factorized NAS A1: Temporal=Search, Interaction=Fixed, Map=Fixed"),
    StageSpec("A2", "RUN_A2", "factorized", (), True, "Factorized NAS A2: Temporal=Fixed, Interaction=Search, Map=Fixed"),
    StageSpec("A3", "RUN_A3", "factorized", (), True, "Factorized NAS A3: Temporal=Fixed, Interaction=Fixed, Map=Search"),
    StageSpec("A4", "RUN_A4", "factorized", (), True, "Factorized NAS A4: Temporal=Search, Interaction=Search, Map=Fixed"),
    StageSpec("A5", "RUN_A5", "factorized", (), True, "Factorized NAS A5: Temporal=Search, Interaction=Fixed, Map=Search"),
    StageSpec("A6", "RUN_A6", "factorized", (), True, "Factorized NAS A6: Temporal=Fixed, Interaction=Search, Map=Search"),
    StageSpec("A7", "RUN_A7", "factorized", (), True, "Factorized NAS A7: Full Tri-Branch DARTS Search"),
    StageSpec("HARDWARE", "RUN_HARDWARE", "hardware", (), False, "Hardware-Aware NAS Baseline & Reference Profiling"),
    StageSpec("HARDWARE_LAMBDA_SEARCH", "RUN_HARDWARE_LAMBDA_SEARCH", "hardware", (), True, "Hardware-Aware FLOPs Penalty (lambda_flops) Sweep"),
    StageSpec("SCENE", "RUN_SCENE", "scene", (), True, "Scene Complexity Stratification & Fixed-Model Evaluation"),
    StageSpec("SCENE_PROTOTYPE", "RUN_SCENE_PROTOTYPE", "scene", ("SCENE",), True, "Experimental Scene-Conditioned Prototype"),
    StageSpec("T0", "RUN_T0", "warmup_temperature", (), True, "Warmup/Temp T0: No Warmup, Fixed Temperature"),
    StageSpec("T1", "RUN_T1", "warmup_temperature", (), True, "Warmup/Temp T1: Warmup Enabled, Fixed Temperature"),
    StageSpec("T2", "RUN_T2", "warmup_temperature", (), True, "Warmup/Temp T2: No Warmup, Temp Annealing Enabled"),
    StageSpec("T3", "RUN_T3", "warmup_temperature", (), True, "Warmup/Temp T3: Warmup + Temp Annealing Enabled"),
)

STAGE_MAP: dict[str, StageSpec] = {s.stage_id: s for s in STAGE_SPECS}
ALL_STAGE_IDS: tuple[str, ...] = tuple(s.stage_id for s in STAGE_SPECS)
FACTORIZED_STAGE_IDS: tuple[str, ...] = tuple(f"A{i}" for i in range(8))
WARMUP_TEMP_STAGE_IDS: tuple[str, ...] = tuple(f"T{i}" for i in range(4))
COMPARATIVE_STAGE_IDS: tuple[str, ...] = tuple(s.stage_id for s in STAGE_SPECS if s.stage_id != "AUDIT")

GENERAL_BOOL_FLAGS: tuple[str, ...] = (
    "SKIP_COMPLETED",
    "FORCE_RERUN",
    "CONTINUE_ON_ERROR",
    "REQUIRE_AUDIT",
    "SAVE_INTERMEDIATE_RESULTS",
    "SAVE_EXECUTION_STATUS",
    "GENERATE_PLOTS",
    "GENERATE_REPORT",
)


class ConfigValidationError(ValueError):
    """Raised when evaluate_config or CLI pipeline configuration is invalid."""


def utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_evaluate_config_module(config_path: str | Path | None = None) -> ModuleType:
    """Loads `configs.evaluate_config` or an explicit Python config file path."""
    if config_path is not None:
        p = Path(config_path).resolve()
        if not p.is_file():
            raise ConfigValidationError(f"Evaluation config file does not exist: {p}")
        spec = importlib.util.spec_from_file_location("custom_evaluate_config", p)
        if spec is None or spec.loader is None:
            raise ConfigValidationError(f"Unable to load evaluation config module from: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module("configs.evaluate_config")


def extract_config_dict(cfg_source: Any) -> dict[str, Any]:
    """Extracts configuration attributes from a module, object, or dict."""
    if isinstance(cfg_source, dict):
        data = dict(cfg_source)
    else:
        data = {k: getattr(cfg_source, k) for k in dir(cfg_source) if k.isupper()}
    return data


def _check_directory_accessible(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ConfigValidationError(
                f"Output path exists and is not a directory: '{output_dir}'."
            )
        if not os.access(output_dir, os.W_OK | os.X_OK):
            raise ConfigValidationError(
                f"Output directory is not writable/accessible: '{output_dir}'."
            )
        return

    # Walk up to nearest existing parent and ensure it is a writable directory
    probe = output_dir.parent
    while probe != probe.parent and not probe.exists():
        probe = probe.parent
    if not probe.exists() or not probe.is_dir():
        raise ConfigValidationError(
            f"Cannot access parent directory for output_dir '{output_dir}'."
        )
    if not os.access(probe, os.W_OK | os.X_OK):
        raise ConfigValidationError(
            f"Parent directory '{probe}' is not writable; cannot create '{output_dir}'."
        )


def _normalize_output_path(path_str: str, output_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return output_dir / p


class ExecutionStatusStore:
    """Persistent JSON status tracker for research evaluation stages."""

    def __init__(self, status_path: Path, output_dir: Path):
        self.status_path = Path(status_path)
        self.output_dir = Path(output_dir)
        self.data: dict[str, Any] = self._empty_payload()
        self.recovered_interrupted_stages: list[str] = []

    def _empty_payload(self) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for spec in STAGE_SPECS:
            stages[spec.stage_id] = {
                "stage_id": spec.stage_id,
                "section": spec.section,
                "status": STATUS_PENDING,
                "start_timestamp": None,
                "completion_timestamp": None,
                "duration_seconds": None,
                "output_files": [],
                "error": None,
                "skip_reason": None,
                "enabled": None,
                "previous_runs": [],
            }
        return {
            "version": "1.0.0",
            "output_dir": str(self.output_dir),
            "updated_at_utc": utc_iso_now(),
            "pipeline_state": STATUS_PENDING,
            "stages": stages,
        }

    def load(self) -> dict[str, Any]:
        """Loads status JSON if present and safely recovers any interrupted RUNNING stages."""
        self.recovered_interrupted_stages = []
        if not self.status_path.is_file():
            self.data = self._empty_payload()
            return self.data

        try:
            with open(self.status_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as exc:
            raise ConfigValidationError(
                f"Existing execution status file '{self.status_path}' is corrupted or unreadable: {exc}"
            ) from exc

        if not isinstance(loaded, dict) or not isinstance(loaded.get("stages"), dict):
            raise ConfigValidationError(
                f"Existing execution status file '{self.status_path}' has an invalid schema."
            )

        unknown_ids = [sid for sid in loaded["stages"].keys() if sid not in STAGE_MAP]
        if unknown_ids:
            raise ConfigValidationError(
                f"Unrecognized evaluation stage identifier(s) in '{self.status_path}': {unknown_ids}. "
                f"Valid identifiers: {list(ALL_STAGE_IDS)}"
            )

        base = self._empty_payload()
        base["pipeline_state"] = loaded.get("pipeline_state", STATUS_PENDING)
        for sid in ALL_STAGE_IDS:
            rec = loaded["stages"].get(sid)
            if not isinstance(rec, dict):
                continue
            merged = dict(base["stages"][sid])
            merged.update(rec)
            merged.setdefault("previous_runs", [])
            merged.setdefault("output_files", [])

            st = merged.get("status")
            if st not in VALID_STATUSES:
                raise ConfigValidationError(
                    f"Invalid execution status '{st}' for stage '{sid}' in '{self.status_path}'."
                )

            # Recovery rule: a stage marked RUNNING without a valid completion record
            # must NEVER be treated as completed.
            if st == STATUS_RUNNING:
                self.recovered_interrupted_stages.append(sid)
                interrupted_rec = {
                    "status": STATUS_RUNNING,
                    "start_timestamp": merged.get("start_timestamp"),
                    "completion_timestamp": merged.get("completion_timestamp"),
                    "duration_seconds": merged.get("duration_seconds"),
                    "output_files": list(merged.get("output_files") or []),
                    "error": "Execution interrupted while RUNNING without a valid completion record.",
                    "recovered_at_utc": utc_iso_now(),
                }
                merged["previous_runs"].append(interrupted_rec)
                merged["status"] = STATUS_PENDING
                merged["completion_timestamp"] = None
                merged["duration_seconds"] = None
                merged["error"] = interrupted_rec["error"]
                merged["skip_reason"] = None
            elif st == STATUS_COMPLETED and not merged.get("completion_timestamp"):
                self.recovered_interrupted_stages.append(sid)
                merged["status"] = STATUS_PENDING
                merged["error"] = "Previous COMPLETED record lacked a completion timestamp."

            base["stages"][sid] = merged

        if base.get("pipeline_state") == STATUS_RUNNING:
            base["pipeline_state"] = "INTERRUPTED"

        self.data = base
        return self.data

    def save(self, enabled: bool = True) -> None:
        if not enabled:
            return
        self.data["updated_at_utc"] = utc_iso_now()
        self.data["output_dir"] = str(self.output_dir)
        atomic_write_json(self.status_path, self.data)

    def get_stage(self, stage_id: str) -> dict[str, Any]:
        return self.data["stages"][stage_id]

    def verify_stage_completed(
        self,
        stage_id: str,
        seeds: list[int] | None = None,
        summary_data: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Verifies a stage is COMPLETED with non-missing expected output files and result records."""
        rec = self.get_stage(stage_id)
        if rec.get("status") != STATUS_COMPLETED:
            return False, f"status is {rec.get('status')}"
        if not rec.get("completion_timestamp"):
            return False, "missing completion_timestamp"

        output_files = rec.get("output_files") or []
        if not output_files:
            return False, "no recorded output_files"

        for rel_or_abs in output_files:
            p = _normalize_output_path(str(rel_or_abs), self.output_dir)
            if not p.is_file() or p.stat().st_size == 0:
                return False, f"expected output file missing or empty: {rel_or_abs}"

        if summary_data is not None:
            exps = summary_data.get("experiments") or {}
            if stage_id == "AUDIT":
                audit_sec = exps.get("audit")
                if not isinstance(audit_sec, dict) or audit_sec.get("status") != "completed":
                    return False, "audit result not marked completed in summary.json"
            elif stage_id in FACTORIZED_STAGE_IDS:
                fac_rows = (exps.get("factorized") or {}).get("rows") or []
                req_seeds = list(seeds) if seeds else []
                for sd in req_seeds:
                    matched = any(
                        r.get("config_id") == stage_id
                        and r.get("seed") == sd
                        and r.get("status") == "completed"
                        for r in fac_rows
                    )
                    if not matched:
                        return False, f"missing completed summary row for ({stage_id}, seed={sd})"
            elif stage_id in WARMUP_TEMP_STAGE_IDS:
                wt_rows = (exps.get("warmup_temperature") or {}).get("rows") or []
                req_seeds = list(seeds) if seeds else []
                for sd in req_seeds:
                    matched = any(
                        r.get("config_id") == stage_id
                        and r.get("seed") == sd
                        and r.get("status") == "completed"
                        for r in wt_rows
                    )
                    if not matched:
                        return False, f"missing completed summary row for ({stage_id}, seed={sd})"
            elif stage_id == "HARDWARE":
                hw_rows = (exps.get("hardware") or {}).get("rows") or []
                baselines_ok = any(
                    r.get("category") in ("manual_baseline", "discovered_reference")
                    and r.get("status") == "completed"
                    for r in hw_rows
                )
                if not baselines_ok:
                    return False, "missing completed hardware baseline rows in summary.json"
            elif stage_id == "HARDWARE_LAMBDA_SEARCH":
                hw_rows = (exps.get("hardware") or {}).get("rows") or []
                sweep_rows = [r for r in hw_rows if r.get("category") == "flops_penalty_sweep"]
                if not sweep_rows or not all(r.get("status") == "completed" for r in sweep_rows):
                    return False, "incomplete lambda_flops sweep rows in summary.json"
            elif stage_id == "SCENE":
                sc_sec = exps.get("scene") or {}
                if not sc_sec.get("rows"):
                    return False, "missing scene complexity rows in summary.json"
            elif stage_id == "SCENE_PROTOTYPE":
                proto = (exps.get("scene") or {}).get("experimental_scene_conditioned_prototype") or {}
                if proto.get("status") != "completed":
                    return False, "scene prototype not completed in summary.json"

        return True, "verified"

    def mark_running(self, stage_id: str, save_enabled: bool = True) -> float:
        rec = self.get_stage(stage_id)
        if rec.get("status") in (STATUS_COMPLETED, STATUS_FAILED) and rec.get("start_timestamp"):
            snapshot = {
                "status": rec.get("status"),
                "start_timestamp": rec.get("start_timestamp"),
                "completion_timestamp": rec.get("completion_timestamp"),
                "duration_seconds": rec.get("duration_seconds"),
                "output_files": list(rec.get("output_files") or []),
                "error": rec.get("error"),
            }
            rec.setdefault("previous_runs", []).append(snapshot)

        rec["status"] = STATUS_RUNNING
        rec["start_timestamp"] = utc_iso_now()
        rec["completion_timestamp"] = None
        rec["duration_seconds"] = None
        rec["error"] = None
        rec["skip_reason"] = None
        rec["enabled"] = True
        self.data["pipeline_state"] = STATUS_RUNNING
        self.save(enabled=save_enabled)
        return time.perf_counter()

    def mark_completed(
        self,
        stage_id: str,
        t_start_perf: float,
        output_files: list[str],
        save_enabled: bool = True,
    ) -> None:
        rec = self.get_stage(stage_id)
        rec["status"] = STATUS_COMPLETED
        rec["completion_timestamp"] = utc_iso_now()
        rec["duration_seconds"] = round(max(0.0, time.perf_counter() - t_start_perf), 4)
        rec["output_files"] = list(dict.fromkeys(output_files))
        rec["error"] = None
        rec["skip_reason"] = None
        rec["enabled"] = True
        self.save(enabled=save_enabled)

    def mark_failed(
        self,
        stage_id: str,
        t_start_perf: float,
        error: str,
        output_files: list[str] | None = None,
        save_enabled: bool = True,
    ) -> None:
        rec = self.get_stage(stage_id)
        rec["status"] = STATUS_FAILED
        rec["completion_timestamp"] = utc_iso_now()
        rec["duration_seconds"] = round(max(0.0, time.perf_counter() - t_start_perf), 4)
        if output_files is not None:
            rec["output_files"] = list(dict.fromkeys(output_files))
        rec["error"] = str(error)
        rec["skip_reason"] = None
        rec["enabled"] = True
        self.save(enabled=save_enabled)

    def mark_skipped(
        self,
        stage_id: str,
        reason: str,
        enabled_flag: bool,
        preserve_completed: bool = True,
        save_enabled: bool = True,
    ) -> None:
        rec = self.get_stage(stage_id)
        rec["enabled"] = bool(enabled_flag)
        rec["skip_reason"] = reason
        # Preserve prior COMPLETED metadata & outputs when a completed stage is disabled or skipped
        if preserve_completed and rec.get("status") == STATUS_COMPLETED:
            self.save(enabled=save_enabled)
            return
        # Never overwrite an error/interruption record with SKIPPED if we want to keep track;
        # record SKIPPED as active status and preserve prior failure in previous_runs if present.
        if rec.get("status") == STATUS_FAILED and rec.get("error"):
            rec.setdefault("previous_runs", []).append({
                "status": rec.get("status"),
                "start_timestamp": rec.get("start_timestamp"),
                "completion_timestamp": rec.get("completion_timestamp"),
                "duration_seconds": rec.get("duration_seconds"),
                "output_files": list(rec.get("output_files") or []),
                "error": rec.get("error"),
            })
        rec["status"] = STATUS_SKIPPED
        self.save(enabled=save_enabled)


@dataclass
class PlannedStageDecision:
    stage_id: str
    section: str
    enabled: bool
    action: str          # "RUN" or "SKIP"
    reason: str
    category: str        # "run", "disabled", "previously_completed", "dependency_blocked"
    previously_completed: bool
    force_rerun: bool


@dataclass
class ExecutionPlan:
    decisions: list[PlannedStageDecision]
    config_values: dict[str, Any]
    notices: list[str]
    recovered_stages: list[str]

    def by_id(self, stage_id: str) -> PlannedStageDecision:
        for d in self.decisions:
            if d.stage_id == stage_id:
                return d
        raise KeyError(f"Unknown stage_id: {stage_id}")

    @property
    def scheduled_ids(self) -> list[str]:
        return [d.stage_id for d in self.decisions if d.action == "RUN"]

    @property
    def scheduled_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == "RUN")

    @property
    def previously_completed_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == "SKIP" and d.previously_completed)

    @property
    def disabled_count(self) -> int:
        return sum(1 for d in self.decisions if not d.enabled)


def validate_and_plan_pipeline(
    raw_cfg: dict[str, Any],
    status_store: ExecutionStatusStore,
    output_dir: Path,
    *,
    cli_experiment: str | None = None,
    cli_run_lambda_search: bool = False,
    cli_run_scene_prototype: bool = False,
    cli_force_rerun: bool = False,
    dataset_name: str = "nuscenes",
    dataset_ready: bool = True,
    resolved_data_dir: str = "",
    data_path_override: str | None = None,
    checkpoint_arg: str | None = None,
    checkpoint_resolved: Path | None = None,
    default_checkpoint_arg: str = "checkpoints/best_trajectory_lnas.pt",
    seeds: list[int] | None = None,
    max_experiments: int | None = None,
    hw_warmup: int = 25,
    hw_runs: int = 100,
    dry_run: bool = False,
    summary_data: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """
    Validates configuration types, stage identifiers, output directory accessibility,
    dependencies, contradictions, and required input files, then builds the stage
    execution plan.
    """
    notices: list[str] = []
    cfg = dict(raw_cfg)

    # 1. Validate optional explicit stage override map / dependency map keys if present
    if "STAGE_FLAGS" in cfg:
        sf = cfg["STAGE_FLAGS"]
        if not isinstance(sf, dict):
            raise ConfigValidationError("evaluate_config.STAGE_FLAGS must be a dict if defined.")
        unknown = [k for k in sf.keys() if k not in STAGE_MAP]
        if unknown:
            raise ConfigValidationError(
                f"Unrecognized evaluation identifier(s) in STAGE_FLAGS: {unknown}. "
                f"Valid identifiers: {list(ALL_STAGE_IDS)}"
            )
    if "DEPENDENCIES" in cfg:
        deps_cfg = cfg["DEPENDENCIES"]
        if not isinstance(deps_cfg, dict):
            raise ConfigValidationError("evaluate_config.DEPENDENCIES must be a dict if defined.")
        for k, reqs in deps_cfg.items():
            if k not in STAGE_MAP:
                raise ConfigValidationError(
                    f"Unrecognized evaluation identifier '{k}' in DEPENDENCIES. "
                    f"Valid identifiers: {list(ALL_STAGE_IDS)}"
                )
            for req in reqs:
                if req not in STAGE_MAP:
                    raise ConfigValidationError(
                        f"Unrecognized dependency identifier '{req}' for '{k}' in DEPENDENCIES. "
                        f"Valid identifiers: {list(ALL_STAGE_IDS)}"
                    )

    # 2. Validate ALL Boolean parameters strictly (bool is a subclass of int, so check isinstance(x, bool))
    required_bool_names = [s.config_flag for s in STAGE_SPECS] + list(GENERAL_BOOL_FLAGS)
    for flag_name in required_bool_names:
        if flag_name not in cfg:
            raise ConfigValidationError(
                f"Missing required Boolean configuration parameter '{flag_name}' in evaluate_config."
            )
        val = cfg[flag_name]
        if not isinstance(val, bool):
            raise ConfigValidationError(
                f"Configuration parameter '{flag_name}' must be a Python bool (True/False), "
                f"got {type(val).__name__}: {val!r}."
            )

    # Apply explicit CLI overrides cleanly after validating base config
    if cli_run_lambda_search:
        cfg["RUN_HARDWARE_LAMBDA_SEARCH"] = True
    if cli_run_scene_prototype:
        cfg["RUN_SCENE_PROTOTYPE"] = True
    if cli_force_rerun:
        cfg["FORCE_RERUN"] = True

    if cli_experiment is not None:
        valid_exps = {"all", "audit", "factorized", "hardware", "scene", "warmup_temperature"}
        if cli_experiment not in valid_exps:
            raise ConfigValidationError(
                f"Unrecognized --experiment '{cli_experiment}'. Valid options: {sorted(valid_exps)}"
            )
        if cli_experiment != "all":
            for spec in STAGE_SPECS:
                if spec.section != cli_experiment:
                    cfg[spec.config_flag] = False

    skip_completed: bool = cfg["SKIP_COMPLETED"]
    force_rerun: bool = cfg["FORCE_RERUN"]
    require_audit: bool = cfg["REQUIRE_AUDIT"]
    save_status: bool = cfg["SAVE_EXECUTION_STATUS"]

    if skip_completed and force_rerun:
        notices.append(
            "FORCE_RERUN=True overrides SKIP_COMPLETED=True for explicitly enabled stages."
        )
    if not save_status and skip_completed:
        notices.append(
            "SAVE_EXECUTION_STATUS=False: completion status from this run will not be persisted for future resume."
        )

    # 3. Validate numeric / path parameters & output directory accessibility
    if seeds is not None:
        if not seeds:
            raise ConfigValidationError("At least one seed must be provided in --seeds.")
        if len(set(seeds)) != len(seeds):
            raise ConfigValidationError(f"Duplicate seeds provided in --seeds: {seeds}")
    if max_experiments is not None and max_experiments < 0:
        raise ConfigValidationError(f"--max-experiments must be >= 0, got {max_experiments}.")
    if hw_warmup < 0 or hw_runs < 1:
        raise ConfigValidationError(
            f"Invalid hardware profiler settings: hw_warmup={hw_warmup} (must be >=0), hw_runs={hw_runs} (must be >=1)."
        )

    _check_directory_accessible(output_dir)

    if data_path_override is not None and not Path(data_path_override).exists():
        raise ConfigValidationError(
            f"Explicit --data-path does not exist: '{data_path_override}'."
        )
    if (
        checkpoint_arg
        and checkpoint_arg != default_checkpoint_arg
        and checkpoint_resolved is None
    ):
        raise ConfigValidationError(
            f"Explicitly requested checkpoint file not found: '{checkpoint_arg}'."
        )

    # 4. Verify prior completion status & outputs for every stage
    verified_map: dict[str, tuple[bool, str]] = {}
    for spec in STAGE_SPECS:
        ok, reason = status_store.verify_stage_completed(
            spec.stage_id,
            seeds=seeds,
            summary_data=summary_data,
        )
        verified_map[spec.stage_id] = (ok, reason)

    # Check if AUDIT previously ran and recorded blocking issues
    audit_prior_failed = (
        status_store.get_stage("AUDIT").get("status") == STATUS_FAILED
        or bool(((summary_data or {}).get("experiments") or {}).get("audit", {}).get("blocking_issues"))
    )

    # 5. Validate dependencies and non-contradictory stage toggles
    enabled_map = {spec.stage_id: bool(cfg[spec.config_flag]) for spec in STAGE_SPECS}

    if force_rerun and not any(enabled_map.values()):
        notices.append("FORCE_RERUN=True is set, but all evaluation stages are disabled.")

    for spec in STAGE_SPECS:
        sid = spec.stage_id
        if not enabled_map[sid]:
            continue

        # Effective dependencies: static code dependencies + optional REQUIRE_AUDIT
        deps = list(spec.dependencies)
        if require_audit and sid != "AUDIT" and "AUDIT" not in deps:
            deps.append("AUDIT")

        for dep_id in deps:
            dep_enabled = enabled_map[dep_id]
            dep_completed, dep_reason = verified_map[dep_id]
            if not dep_enabled and not dep_completed:
                raise ConfigValidationError(
                    f"Dependency error: Stage '{sid}' ({spec.config_flag}=True) requires prerequisite "
                    f"'{dep_id}' ({STAGE_MAP[dep_id].config_flag}), which is disabled and incomplete "
                    f"({dep_reason}). Enable {STAGE_MAP[dep_id].config_flag}=True or disable {spec.config_flag}."
                )

    # 6. Build per-stage execution decisions
    decisions: list[PlannedStageDecision] = []
    for spec in STAGE_SPECS:
        sid = spec.stage_id
        is_enabled = enabled_map[sid]
        already_completed, comp_reason = verified_map[sid]

        if not is_enabled:
            reason = "disabled, previously completed" if already_completed else "disabled"
            decisions.append(
                PlannedStageDecision(
                    stage_id=sid,
                    section=spec.section,
                    enabled=False,
                    action="SKIP",
                    reason=reason,
                    category="disabled",
                    previously_completed=already_completed,
                    force_rerun=False,
                )
            )
            continue

        # If AUDIT previously failed with blocking issues and is NOT being re-run, block comparative stages
        if (
            sid != "AUDIT"
            and audit_prior_failed
            and not (enabled_map["AUDIT"] and (force_rerun or not already_completed))
            and not enabled_map["AUDIT"]
        ):
            decisions.append(
                PlannedStageDecision(
                    stage_id=sid,
                    section=spec.section,
                    enabled=True,
                    action="SKIP",
                    reason="blocked by failed AUDIT (blocking issues detected)",
                    category="dependency_blocked",
                    previously_completed=already_completed,
                    force_rerun=False,
                )
            )
            continue

        if force_rerun:
            reason = "force rerun" if already_completed else "scheduled"
            decisions.append(
                PlannedStageDecision(
                    stage_id=sid,
                    section=spec.section,
                    enabled=True,
                    action="RUN",
                    reason=reason,
                    category="run",
                    previously_completed=already_completed,
                    force_rerun=already_completed,
                )
            )
        elif skip_completed and already_completed:
            decisions.append(
                PlannedStageDecision(
                    stage_id=sid,
                    section=spec.section,
                    enabled=True,
                    action="SKIP",
                    reason="already completed",
                    category="previously_completed",
                    previously_completed=True,
                    force_rerun=False,
                )
            )
        else:
            prev_status = status_store.get_stage(sid).get("status")
            reason = "scheduled"
            if sid in status_store.recovered_interrupted_stages:
                reason = "recover interrupted run"
            elif prev_status == STATUS_FAILED:
                reason = "retry failed stage"
            elif already_completed and not skip_completed:
                reason = "rerun (SKIP_COMPLETED=False)"
            decisions.append(
                PlannedStageDecision(
                    stage_id=sid,
                    section=spec.section,
                    enabled=True,
                    action="RUN",
                    reason=reason,
                    category="run",
                    previously_completed=already_completed,
                    force_rerun=False,
                )
            )

    # 7. Validate dataset files for stages scheduled to RUN (unless --dry-run)
    if dataset_name == "nuscenes" and not dry_run:
        needs_data_runs = [
            d.stage_id for d in decisions if d.action == "RUN" and STAGE_MAP[d.stage_id].requires_dataset
        ]
        if needs_data_runs and not dataset_ready:
            raise ConfigValidationError(
                f"Required nuScenes sequence files (train_sequences.jsonl / val_sequences.jsonl) "
                f"were not found in '{resolved_data_dir}', but dataset-dependent stages are scheduled: "
                f"{needs_data_runs}. Run preprocessing.py first, pass --data-path, or use --dataset synthetic."
            )

    return ExecutionPlan(
        decisions=decisions,
        config_values=cfg,
        notices=notices,
        recovered_stages=list(status_store.recovered_interrupted_stages),
    )


def format_execution_plan(plan: ExecutionPlan) -> str:
    lines = [
        "Research Evaluation Pipeline",
        "--------------------------------",
    ]
    for d in plan.decisions:
        if d.action == "RUN":
            tag = "RUN (force rerun)" if d.force_rerun else (
                f"RUN ({d.reason})" if d.reason not in ("scheduled",) else "RUN"
            )
        else:
            tag = f"SKIP ({d.reason})"
        lines.append(f"{d.stage_id}: {tag}")
    lines.append("--------------------------------")
    lines.append(f"Scheduled evaluations: {plan.scheduled_count}")
    lines.append(f"Previously completed: {plan.previously_completed_count}")
    lines.append(f"Disabled evaluations: {plan.disabled_count}")
    blocked = sum(1 for d in plan.decisions if d.category == "dependency_blocked")
    if blocked:
        lines.append(f"Dependency-blocked evaluations: {blocked}")
    return "\n".join(lines)


def format_final_summary(status_store: ExecutionStatusStore) -> str:
    counts = {st: 0 for st in VALID_STATUSES}
    lines = [
        "Research Evaluation Final Summary",
        "--------------------------------",
    ]
    for spec in STAGE_SPECS:
        rec = status_store.get_stage(spec.stage_id)
        st = rec.get("status", STATUS_PENDING)
        counts[st] = counts.get(st, 0) + 1
        detail_parts = []
        if st == STATUS_COMPLETED and rec.get("duration_seconds") is not None:
            detail_parts.append(f"{rec['duration_seconds']:.2f}s")
        elif st == STATUS_SKIPPED and rec.get("skip_reason"):
            detail_parts.append(rec["skip_reason"])
        elif st == STATUS_FAILED and rec.get("error"):
            err_short = str(rec["error"]).splitlines()[0][:100]
            detail_parts.append(err_short)
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"{spec.stage_id}: {st}{detail}")
    lines.append("--------------------------------")
    lines.append(
        f"Completed: {counts[STATUS_COMPLETED]} | "
        f"Failed: {counts[STATUS_FAILED]} | "
        f"Skipped: {counts[STATUS_SKIPPED]} | "
        f"Pending: {counts[STATUS_PENDING]}"
    )
    return "\n".join(lines)

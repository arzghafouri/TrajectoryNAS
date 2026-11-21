"""
Tests for Configurable and Resumable Research Evaluation Pipeline
==================================================================
Verifies all configuration, persistence, dependency, recovery, and resumability
behaviors in `configs/evaluate_config.py`, `experiments/execution_manager.py`,
and `experiments/research_gap_evaluation.py` using mocked evaluation runners.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_CURR_DIR = Path(__file__).resolve().parent
_PYTHON_SRC = _CURR_DIR.parent
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from configs import evaluate_config
from experiments import research_gap_evaluation as rge
from experiments.execution_manager import (
    ALL_STAGE_IDS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    ExecutionStatusStore,
    extract_config_dict,
)


def _base_cfg(**overrides) -> dict:
    cfg = extract_config_dict(evaluate_config)
    # Default all stages False in helper so individual tests enable exactly what they test
    for k in list(cfg.keys()):
        if k.startswith("RUN_"):
            cfg[k] = False
    cfg["SKIP_COMPLETED"] = True
    cfg["FORCE_RERUN"] = False
    cfg["CONTINUE_ON_ERROR"] = False
    cfg["REQUIRE_AUDIT"] = False
    cfg["SAVE_INTERMEDIATE_RESULTS"] = True
    cfg["SAVE_EXECUTION_STATUS"] = True
    cfg["GENERATE_PLOTS"] = False
    cfg["GENERATE_REPORT"] = True
    cfg.update(overrides)
    return cfg


def _make_fake_exec_fn(executed_calls: list[str], fail_on: set[str] | None = None, raise_on: set[str] | None = None, interrupt_on: set[str] | None = None):
    fail_on = fail_on or set()
    raise_on = raise_on or set()
    interrupt_on = interrupt_on or set()

    def _fake_exec(
        run_id: str,
        args,
        model_cfg,
        seed: int,
        checkpoint_dir: Path,
        branch_modes=None,
        fixed_indices=(3, 2, 0),
        enable_temp_annealing=True,
        warmup_epochs=None,
        lambda_flops=None,
    ):
        executed_calls.append(run_id)
        stage_tag = run_id.replace("factorized_", "").replace("wt_", "")
        if stage_tag in interrupt_on or run_id in interrupt_on:
            raise KeyboardInterrupt(f"simulated interrupt at {run_id}")
        if stage_tag in raise_on or run_id in raise_on:
            raise RuntimeError(f"simulated crash at {run_id}")
        if stage_tag in fail_on or run_id in fail_on:
            return {
                "status": "failed",
                "run_id": run_id,
                "seed": seed,
                "checkpoint_path": None,
                "error": f"simulated failure in {run_id}",
            }

        ckpt_path = Path(checkpoint_dir) / f"{run_id}_seed{seed}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(f"mock-ckpt-{run_id}-seed{seed}", encoding="utf-8")

        return {
            "status": "completed",
            "run_id": run_id,
            "seed": seed,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_size_mb": 0.01,
            "selected_indices": [3, 2, 0],
            "selected_temporal": "TemporalSelfAttention",
            "selected_interaction": "SocialPooling",
            "selected_map": "LaneAttention",
            "parameters": 123456,
            "parameters_m": 0.1235,
            "theoretical_flops_g": 0.05,
            "search_duration_s": 0.01,
            "final_training_duration_s": 0.02,
            "peak_search_gpu_memory_mb": None,
            "peak_train_gpu_memory_mb": None,
            "peak_gpu_memory_mb": None,
            "search_supernet_val_ade": 1.1,
            "search_supernet_val_fde": 2.2,
            "final_val_min_ade": 0.85,
            "final_val_min_fde": 1.75,
            "final_val_miss_rate_2m": 15.0,
            "k_modes": int(model_cfg.num_modes),
            "samples_evaluated": 16,
            "final_alphas": None,
            "stability": {"argmax_flips_total": 0, "argmax_flips_second_half": 0},
            "zero_cost_proxies": {},
            "temperature_schedule": {"annealing_enabled": enable_temp_annealing, "start": 1.0, "min": 0.2},
            "warmup_epochs": warmup_epochs if warmup_epochs is not None else 10,
            "lambda_flops": lambda_flops if lambda_flops is not None else 0.0005,
            "error": None,
        }

    return _fake_exec


def _fake_audit_ok(**kwargs):
    return {
        "experiment": "audit",
        "status": "completed",
        "timestamp_utc": "2026-11-21T00:00:00Z",
        "duration_seconds": 0.01,
        "dataset_name": kwargs.get("dataset_name", "synthetic"),
        "dataset_paths": {"dataset_ready": True},
        "warnings": [],
        "blocking_issues": [],
        "safe_to_compare_experiments": True,
    }


def _fake_hardware_fn(executed_hw: list[str]):
    def _fn(args, model_cfg, search_cfg, output_dir, prior_results=None, run_baselines=True, run_lambda_sweep=None, **kwargs):
        prior_by_cid = {r["candidate_id"]: r for r in (prior_results or []) if r.get("status") == "completed"}
        rows = []
        on_start = kwargs.get("on_stage_start")
        on_done = kwargs.get("on_stage_complete")

        if run_baselines:
            if on_start:
                on_start("HARDWARE")
            executed_hw.append("HARDWARE")
            b_rows = [
                {
                    "candidate_id": cid,
                    "category": "discovered_reference" if cid == "trajectory_lnas_reference" else "manual_baseline",
                    "lambda_flops": None,
                    "status": "completed",
                    "accuracy_status": "measured_trained",
                    "parameters": 1000,
                    "theoretical_flops_g": 0.05,
                    "final_val_min_ade": 0.9,
                    "final_val_min_fde": 1.8,
                    "final_val_miss_rate_2m": 16.0,
                }
                for cid in ("manual_lightweight", "manual_heavy_attention", "trajectory_lnas_reference")
            ]
            rows.extend(b_rows)
            if on_done:
                on_done("HARDWARE", b_rows, {"experiment": "hardware", "status": "completed", "rows": list(rows)})
        else:
            for cid in ("manual_lightweight", "manual_heavy_attention", "trajectory_lnas_reference"):
                if cid in prior_by_cid:
                    rows.append(prior_by_cid[cid])

        if run_lambda_sweep:
            if on_start:
                on_start("HARDWARE_LAMBDA_SEARCH")
            executed_hw.append("HARDWARE_LAMBDA_SEARCH")
            l_rows = [
                {
                    "candidate_id": "lambda_flops_0",
                    "category": "flops_penalty_sweep",
                    "lambda_flops": 0.0,
                    "status": "completed",
                    "accuracy_status": "measured_trained",
                    "final_val_min_ade": 0.88,
                }
            ]
            rows.extend(l_rows)
            if on_done:
                on_done("HARDWARE_LAMBDA_SEARCH", l_rows, {"experiment": "hardware", "status": "completed", "rows": list(rows)})
        else:
            for r in (prior_results or []):
                if r.get("category") == "flops_penalty_sweep" and r.get("status") == "completed":
                    rows.append(r)

        return {"experiment": "hardware", "status": "completed", "rows": rows}

    return _fn


def _fake_scene_fn(executed_sc: list[str], fail_scene: bool = False):
    def _fn(args, model_cfg, search_cfg, output_dir, run_stratification=True, run_prototype=None, prior_result=None, **kwargs):
        on_start = kwargs.get("on_stage_start")
        on_done = kwargs.get("on_stage_complete")
        active_proto = bool(args.run_scene_prototype) if run_prototype is None else bool(run_prototype)
        out = dict(prior_result) if prior_result else {
            "experiment": "scene",
            "status": "completed",
            "model_state": "trained_checkpoint",
            "total_val_samples": 10,
            "unavailable_dataset_fields": [],
            "rows": [],
            "experimental_scene_conditioned_prototype": {"experimental": True, "status": "not_run"},
        }
        if run_stratification:
            if on_start:
                on_start("SCENE")
            executed_sc.append("SCENE")
            if fail_scene:
                out = {"experiment": "scene", "status": "failed", "error": "mock scene failure", "rows": []}
                if on_done:
                    on_done("SCENE", out)
                return out
            out["status"] = "completed"
            out["rows"] = [{"group_name": "complexity_low", "criterion": "mock", "num_samples": 5, "sample_share_percent": 50.0, "min_ade_k": 0.8}]
            if on_done:
                on_done("SCENE", out)

        if active_proto:
            if on_start:
                on_start("SCENE_PROTOTYPE")
            executed_sc.append("SCENE_PROTOTYPE")
            out["experimental_scene_conditioned_prototype"] = {
                "experimental": True,
                "status": "completed",
                "val_min_ade_k": 0.82,
                "val_min_fde_k": 1.65,
                "val_miss_rate_2m_percent": 14.0,
            }
            if on_done:
                on_done("SCENE_PROTOTYPE", out)

        return out

    return _fn


class EvaluatePipelineResumeTests(unittest.TestCase):
    def test_evaluate_config_exposes_all_required_booleans(self):
        expected_flags = [
            "RUN_AUDIT",
            *(f"RUN_A{i}" for i in range(8)),
            "RUN_HARDWARE",
            "RUN_HARDWARE_LAMBDA_SEARCH",
            "RUN_SCENE",
            "RUN_SCENE_PROTOTYPE",
            *(f"RUN_T{i}" for i in range(4)),
            "SKIP_COMPLETED",
            "FORCE_RERUN",
            "CONTINUE_ON_ERROR",
            "REQUIRE_AUDIT",
            "SAVE_INTERMEDIATE_RESULTS",
            "SAVE_EXECUTION_STATUS",
            "GENERATE_PLOTS",
            "GENERATE_REPORT",
        ]
        for flag in expected_flags:
            self.assertTrue(hasattr(evaluate_config, flag), f"Missing flag: {flag}")
            self.assertIs(type(getattr(evaluate_config, flag)), bool, f"{flag} must be a bool")

    def test_boolean_toggles_control_exact_stages_and_disabled_never_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            executed: list[str] = []
            cfg = _base_cfg(RUN_A1=True, RUN_A5=True, RUN_T2=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed)):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(rc, 0)
            self.assertEqual(executed, ["factorized_A1", "factorized_A5", "wt_T2"])
            out = buf.getvalue()
            self.assertIn("A0: SKIP (disabled)", out)
            self.assertIn("A1: RUN", out)
            self.assertIn("Scheduled evaluations: 3", out)

            status_data = json.loads((Path(tmp) / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status_data["stages"]["A1"]["status"], STATUS_COMPLETED)
            self.assertEqual(status_data["stages"]["A5"]["status"], STATUS_COMPLETED)
            self.assertEqual(status_data["stages"]["T2"]["status"], STATUS_COMPLETED)
            self.assertEqual(status_data["stages"]["A0"]["status"], STATUS_SKIPPED)
            self.assertEqual(status_data["stages"]["A0"]["skip_reason"], "disabled")

    def test_critical_resume_a0_a3_then_disable_a0_a3_and_run_a4_a7(self):
        """Critical acceptance test: complete A0-A3, disable them, enable A4-A7, resume without repeating or losing A0-A3."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Run 1: A0-A3 enabled
            executed_run1: list[str] = []
            cfg1 = _base_cfg(RUN_A0=True, RUN_A1=True, RUN_A2=True, RUN_A3=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_run1)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc1 = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg1)

            self.assertEqual(rc1, 0)
            self.assertEqual(executed_run1, ["factorized_A0", "factorized_A1", "factorized_A2", "factorized_A3"])
            status1 = json.loads((tmp_path / "execution_status.json").read_text(encoding="utf-8"))
            a0_ts_run1 = status1["stages"]["A0"]["completion_timestamp"]
            self.assertIsNotNone(a0_ts_run1)

            # Run 2: A0-A3 disabled, A4-A7 enabled
            executed_run2: list[str] = []
            cfg2 = _base_cfg(RUN_A4=True, RUN_A5=True, RUN_A6=True, RUN_A7=True)
            buf2 = io.StringIO()
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_run2)):
                with contextlib.redirect_stdout(buf2):
                    rc2 = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg2)

            self.assertEqual(rc2, 0)
            self.assertEqual(executed_run2, ["factorized_A4", "factorized_A5", "factorized_A6", "factorized_A7"])
            plan_txt = buf2.getvalue()
            self.assertIn("A0: SKIP (disabled, previously completed)", plan_txt)
            self.assertIn("A4: RUN", plan_txt)

            # Verify A0-A3 COMPLETED status and timestamps were preserved, and A4-A7 are COMPLETED
            status2 = json.loads((tmp_path / "execution_status.json").read_text(encoding="utf-8"))
            for i in range(8):
                self.assertEqual(status2["stages"][f"A{i}"]["status"], STATUS_COMPLETED)
            self.assertEqual(status2["stages"]["A0"]["completion_timestamp"], a0_ts_run1)

            # Verify summary.json and factorized_nas_results.csv contain ALL 8 rows (A0-A7)
            summary2 = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
            fac_rows = summary2["experiments"]["factorized"]["rows"]
            self.assertEqual([r["config_id"] for r in fac_rows], [f"A{i}" for i in range(8)])
            with open(tmp_path / "factorized_nas_results.csv", "r", encoding="utf-8") as f:
                csv_ids = [row["config_id"] for row in csv.DictReader(f)]
            self.assertEqual(csv_ids, [f"A{i}" for i in range(8)])

    def test_skip_completed_and_missing_output_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            executed1: list[str] = []
            cfg = _base_cfg(RUN_A0=True, RUN_A1=True, SKIP_COMPLETED=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed1)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(executed1, ["factorized_A0", "factorized_A1"])

            # Second run with same config skips both A0 and A1
            executed2: list[str] = []
            buf2 = io.StringIO()
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed2)):
                with contextlib.redirect_stdout(buf2):
                    rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(executed2, [])
            self.assertIn("A0: SKIP (already completed)", buf2.getvalue())
            self.assertIn("Previously completed: 2", buf2.getvalue())

            # Delete A1's checkpoint output file -> A1 must be re-run while A0 remains skipped
            a1_ckpt = tmp_path / "checkpoints" / "factorized" / "factorized_A1_seed42.pt"
            a1_ckpt.unlink()
            executed3: list[str] = []
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed3)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(executed3, ["factorized_A1"])

            # SKIP_COMPLETED=False with FORCE_RERUN=False must rerun enabled completed stages
            executed4: list[str] = []
            cfg_no_skip = _base_cfg(RUN_A0=True, RUN_A1=True, SKIP_COMPLETED=False, FORCE_RERUN=False)
            buf4 = io.StringIO()
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed4)):
                with contextlib.redirect_stdout(buf4):
                    rc4 = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_no_skip)
            self.assertEqual(rc4, 0)
            self.assertEqual(executed4, ["factorized_A0", "factorized_A1"])
            self.assertIn("SKIP_COMPLETED=False", buf4.getvalue())

    def test_force_rerun_reruns_enabled_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            executed1: list[str] = []
            cfg1 = _base_cfg(RUN_A0=True, RUN_A1=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed1)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg1)

            # Force rerun ONLY A1; keep A0 disabled
            executed2: list[str] = []
            cfg2 = _base_cfg(RUN_A0=False, RUN_A1=True, FORCE_RERUN=True)
            buf2 = io.StringIO()
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed2)):
                with contextlib.redirect_stdout(buf2):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg2)
            self.assertEqual(rc, 0)
            self.assertEqual(executed2, ["factorized_A1"])
            self.assertIn("A1: RUN (force rerun)", buf2.getvalue())

            status = json.loads((tmp_path / "execution_status.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(status["stages"]["A1"]["previous_runs"]), 1)
            history_files = list((tmp_path / "history").glob("summary_before_rerun_*.json"))
            self.assertTrue(history_files, "Backup snapshot of previous summary should be preserved before rerun")

    def test_failed_stage_not_completed_and_continue_on_error_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # CONTINUE_ON_ERROR = False: A1 fails, A2 must NOT execute
            executed_stop: list[str] = []
            cfg_stop = _base_cfg(RUN_A0=True, RUN_A1=True, RUN_A2=True, CONTINUE_ON_ERROR=False)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_stop, fail_on={"A1"})):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_stop)
            self.assertEqual(rc, 1)
            self.assertEqual(executed_stop, ["factorized_A0", "factorized_A1"])
            status = json.loads((tmp_path / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stages"]["A0"]["status"], STATUS_COMPLETED)
            self.assertEqual(status["stages"]["A1"]["status"], STATUS_FAILED)
            self.assertEqual(status["stages"]["A2"]["status"], STATUS_PENDING)

        with tempfile.TemporaryDirectory() as tmp2:
            tmp2_path = Path(tmp2)
            # CONTINUE_ON_ERROR = True: A1 raises exception, A2 still executes
            executed_cont: list[str] = []
            cfg_cont = _base_cfg(RUN_A0=True, RUN_A1=True, RUN_A2=True, CONTINUE_ON_ERROR=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_cont, raise_on={"A1"})):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp2], evaluate_config_override=cfg_cont)
            self.assertEqual(rc, 1)
            self.assertEqual(executed_cont, ["factorized_A0", "factorized_A1", "factorized_A2"])
            status = json.loads((tmp2_path / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stages"]["A0"]["status"], STATUS_COMPLETED)
            self.assertEqual(status["stages"]["A1"]["status"], STATUS_FAILED)
            self.assertIn("simulated crash", status["stages"]["A1"]["error"])
            self.assertEqual(status["stages"]["A2"]["status"], STATUS_COMPLETED)

    def test_interrupted_running_state_is_safely_recovered_and_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Simulate a crashed run leaving A0 COMPLETED and A1 stuck in RUNNING
            store = ExecutionStatusStore(tmp_path / "execution_status.json", tmp_path)
            t0 = store.mark_running("A0")
            ckpt_a0 = tmp_path / "checkpoints" / "factorized" / "factorized_A0_seed42.pt"
            ckpt_a0.parent.mkdir(parents=True, exist_ok=True)
            ckpt_a0.write_text("ok", encoding="utf-8")
            (tmp_path / "summary.json").write_text(
                json.dumps({"experiments": {"factorized": {"rows": [{"config_id": "A0", "seed": 42, "status": "completed"}]}}}),
                encoding="utf-8",
            )
            store.mark_completed("A0", t0, ["checkpoints/factorized/factorized_A0_seed42.pt", "summary.json"])
            store.mark_running("A1")  # crashed mid-execution without completion

            # Next run must recover A1 (not treat as completed), skip A0, and run A1 & A2
            executed: list[str] = []
            cfg = _base_cfg(RUN_A0=True, RUN_A1=True, RUN_A2=True, SKIP_COMPLETED=True)
            buf = io.StringIO()
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed)):
                with contextlib.redirect_stdout(buf):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(rc, 0)
            self.assertEqual(executed, ["factorized_A1", "factorized_A2"])
            self.assertIn("recover interrupted run", buf.getvalue())

            status = json.loads((tmp_path / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stages"]["A1"]["status"], STATUS_COMPLETED)
            self.assertTrue(any(r.get("status") == STATUS_RUNNING for r in status["stages"]["A1"]["previous_runs"]))

        # Also verify KeyboardInterrupt mid-run saves status cleanly
        with tempfile.TemporaryDirectory() as tmp_kb:
            executed_kb: list[str] = []
            cfg_kb = _base_cfg(RUN_A0=True, RUN_A1=True, RUN_A2=True)
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_kb, interrupt_on={"A1"})):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_kb = rge.main(["--dataset", "synthetic", "--output-dir", tmp_kb], evaluate_config_override=cfg_kb)
            self.assertEqual(rc_kb, 130)
            status_kb = json.loads((Path(tmp_kb) / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status_kb["pipeline_state"], "INTERRUPTED")
            self.assertEqual(status_kb["stages"]["A0"]["status"], STATUS_COMPLETED)
            self.assertEqual(status_kb["stages"]["A1"]["status"], STATUS_FAILED)
            self.assertEqual(status_kb["stages"]["A2"]["status"], STATUS_PENDING)

    def test_dependencies_and_audit_gating(self):
        # 1. SCENE_PROTOTYPE enabled while SCENE disabled & incomplete -> ConfigValidationError (exit code 2)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_bad_dep = _base_cfg(RUN_SCENE=False, RUN_SCENE_PROTOTYPE=True)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_bad_dep)
            self.assertEqual(rc, 2)
            self.assertIn("SCENE_PROTOTYPE", err.getvalue())
            self.assertIn("SCENE", err.getvalue())

        # 2. SCENE completes in Run 1 -> SCENE_PROTOTYPE runs in Run 2 with RUN_SCENE=False
        with tempfile.TemporaryDirectory() as tmp:
            exec_sc: list[str] = []
            cfg_sc1 = _base_cfg(RUN_SCENE=True, RUN_SCENE_PROTOTYPE=False)
            with patch.object(rge, "run_scene_complexity_evaluation", side_effect=_fake_scene_fn(exec_sc)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc1 = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_sc1)
            self.assertEqual(rc1, 0)
            self.assertEqual(exec_sc, ["SCENE"])

            exec_sc2: list[str] = []
            cfg_sc2 = _base_cfg(RUN_SCENE=False, RUN_SCENE_PROTOTYPE=True)
            with patch.object(rge, "run_scene_complexity_evaluation", side_effect=_fake_scene_fn(exec_sc2)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc2 = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_sc2)
            self.assertEqual(rc2, 0)
            self.assertEqual(exec_sc2, ["SCENE_PROTOTYPE"])

        # 3. Runtime failure in SCENE skips SCENE_PROTOTYPE when CONTINUE_ON_ERROR=True
        with tempfile.TemporaryDirectory() as tmp:
            exec_sc_fail: list[str] = []
            cfg_sc_fail = _base_cfg(RUN_SCENE=True, RUN_SCENE_PROTOTYPE=True, CONTINUE_ON_ERROR=True)
            with patch.object(rge, "run_scene_complexity_evaluation", side_effect=_fake_scene_fn(exec_sc_fail, fail_scene=True)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_sc_fail)
            self.assertEqual(rc, 1)
            self.assertEqual(exec_sc_fail, ["SCENE"])
            st = json.loads((Path(tmp) / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["stages"]["SCENE"]["status"], STATUS_FAILED)
            self.assertEqual(st["stages"]["SCENE_PROTOTYPE"]["status"], STATUS_SKIPPED)
            self.assertIn("prerequisite SCENE failed", st["stages"]["SCENE_PROTOTYPE"]["skip_reason"])

        # 4. Blocking audit halts comparative experiments
        with tempfile.TemporaryDirectory() as tmp:
            executed_after_audit: list[str] = []
            cfg_aud = _base_cfg(RUN_AUDIT=True, RUN_A0=True, RUN_HARDWARE=True)

            def _bad_audit(**kwargs):
                return {"experiment": "audit", "status": "failed", "blocking_issues": ["scene leakage detected"], "warnings": []}

            with patch.object(rge, "run_data_audit", side_effect=_bad_audit), \
                 patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_after_audit)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg_aud)
            self.assertEqual(rc, 1)
            self.assertEqual(executed_after_audit, [])
            st = json.loads((Path(tmp) / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["stages"]["AUDIT"]["status"], STATUS_FAILED)
            self.assertEqual(st["stages"]["A0"]["status"], STATUS_SKIPPED)

        # 5. REQUIRE_AUDIT=True succeeds when AUDIT already completed previously
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rge, "run_data_audit", side_effect=_fake_audit_ok):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_a = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=_base_cfg(RUN_AUDIT=True))
            self.assertEqual(rc_a, 0)

            executed_req: list[str] = []
            with patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(executed_req)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_b = rge.main(
                        ["--dataset", "synthetic", "--output-dir", tmp],
                        evaluate_config_override=_base_cfg(REQUIRE_AUDIT=True, RUN_AUDIT=False, RUN_A0=True),
                    )
            self.assertEqual(rc_b, 0)
            self.assertEqual(executed_req, ["factorized_A0"])

    def test_configuration_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Non-boolean value (int 1 instead of bool True)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=_base_cfg(RUN_A0=1))
            self.assertEqual(rc, 2)
            self.assertIn("RUN_A0", err.getvalue())

            # Unrecognized stage identifier in STAGE_FLAGS
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                rc2 = rge.main(
                    ["--dataset", "synthetic", "--output-dir", tmp],
                    evaluate_config_override=_base_cfg(STAGE_FLAGS={"A99": True}),
                )
            self.assertEqual(rc2, 2)
            self.assertIn("A99", err2.getvalue())

            # Missing nuScenes files before running dataset-dependent stage A0
            err3 = io.StringIO()
            with contextlib.redirect_stderr(err3):
                rc3 = rge.main(
                    ["--dataset", "nuscenes", "--output-dir", tmp],
                    evaluate_config_override=_base_cfg(RUN_A0=True),
                )
            self.assertEqual(rc3, 2)
            self.assertIn("train_sequences", err3.getvalue())

            # Output directory points to an existing file
            file_out = Path(tmp) / "not_a_dir.txt"
            file_out.write_text("x", encoding="utf-8")
            err4 = io.StringIO()
            with contextlib.redirect_stderr(err4):
                rc4 = rge.main(
                    ["--dataset", "synthetic", "--output-dir", str(file_out)],
                    evaluate_config_override=_base_cfg(RUN_AUDIT=True),
                )
            self.assertEqual(rc4, 2)
            self.assertIn("not a directory", err4.getvalue())

            # REQUIRE_AUDIT=True when RUN_AUDIT=False and AUDIT incomplete
            err5 = io.StringIO()
            with contextlib.redirect_stderr(err5):
                rc5 = rge.main(
                    ["--dataset", "synthetic", "--output-dir", tmp],
                    evaluate_config_override=_base_cfg(REQUIRE_AUDIT=True, RUN_AUDIT=False, RUN_A0=True),
                )
            self.assertEqual(rc5, 2)
            self.assertIn("AUDIT", err5.getvalue())

    def test_all_sections_resumable_and_existing_math_untouched(self):
        # Verify existing mathematical helpers and matrices remain unchanged
        self.assertEqual([x["id"] for x in rge.FACTORIZED_ABLATION_MATRIX], [f"A{i}" for i in range(8)])
        self.assertEqual([x["id"] for x in rge.WARMUP_TEMP_MATRIX], [f"T{i}" for i in range(4)])
        self.assertAlmostEqual(rge._quantile([10.0, 20.0, 30.0], 0.5), 20.0, places=6)
        stats = rge._stats_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean"], 2.5)
        self.assertGreater(rge._shannon_entropy([0.25, 0.25, 0.25, 0.25]), 1.38)
        self.assertEqual(len(ALL_STAGE_IDS), 17)

        # End-to-end mocked run across AUDIT, HARDWARE, HARDWARE_LAMBDA_SEARCH, T0-T3
        with tempfile.TemporaryDirectory() as tmp:
            exec_calls: list[str] = []
            hw_calls: list[str] = []
            cfg = _base_cfg(
                RUN_AUDIT=True,
                RUN_HARDWARE=True,
                RUN_HARDWARE_LAMBDA_SEARCH=True,
                RUN_T0=True,
                RUN_T1=True,
            )
            with patch.object(rge, "run_data_audit", side_effect=_fake_audit_ok), \
                 patch.object(rge, "_execute_search_and_final_train", side_effect=_make_fake_exec_fn(exec_calls)), \
                 patch.object(rge, "run_hardware_profiling", side_effect=_fake_hardware_fn(hw_calls)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = rge.main(["--dataset", "synthetic", "--output-dir", tmp], evaluate_config_override=cfg)
            self.assertEqual(rc, 0)
            self.assertEqual(hw_calls, ["HARDWARE", "HARDWARE_LAMBDA_SEARCH"])
            self.assertEqual(exec_calls, ["wt_T0", "wt_T1"])

    def test_output_dir_resolves_inside_python_src_reports(self):
        expected_base = str((_PYTHON_SRC / "reports" / "research_gaps").resolve())

        buf_cfg = io.StringIO()
        with contextlib.redirect_stdout(buf_cfg):
            rc1 = rge.main(["--dry-run", "--dataset", "synthetic"], evaluate_config_override=_base_cfg())
        self.assertEqual(rc1, 0)
        self.assertIn(str((_PYTHON_SRC / "reports" / "research_gaps" / "pipeline").resolve()), buf_cfg.getvalue())

        buf_ts = io.StringIO()
        with contextlib.redirect_stdout(buf_ts):
            rc2 = rge.main(["--dry-run", "--dataset", "synthetic"], evaluate_config_override=_base_cfg(OUTPUT_DIR=None))
        self.assertEqual(rc2, 0)
        self.assertIn(expected_base, buf_ts.getvalue())


if __name__ == "__main__":
    unittest.main()

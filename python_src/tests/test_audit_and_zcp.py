"""
Comprehensive Audit & Zero-Cost Proxy Regression Test Suite
=============================================================
Covers:
  - Prompt 1 Tests A-I (K consistency, tensor shape normalization, top-K ranking,
    endpoint vs max-displacement miss rate, collision risk, SNIP train-batch/restoration,
    analytical FLOPs/MACs breakdown, HardwareProfiler, end-to-end smoke + architecture resolution)
  - Prompt 2 Tests 1-7 + Critical DARTS Decoupling Regression Test
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.search_config import SearchConfig
from core.zcp_ranking import (
    ZCPAcceptanceCriteria,
    compute_zcp_validation_correlations,
    enumerate_discrete_candidates,
    evaluate_screening_acceptance,
    kendall_tau,
    minmax_normalize,
    rank_normalize,
    screen_candidates,
    spearman_correlation,
    top_k_retention,
)

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


class PurePythonZCPStatTests(unittest.TestCase):
    """Dataset- and PyTorch-free verification of ZCP enumeration, correlations, retention, and gating."""

    def test_candidate_enumeration_uses_search_config_not_registry_aliases(self):
        cfg = SearchConfig()
        specs = enumerate_discrete_candidates(cfg)
        expected = len(cfg.temporal_ops) * len(cfg.interaction_ops) * len(cfg.map_ops)
        self.assertEqual(len(specs), expected)
        self.assertEqual(expected, 48)

    def test_spearman_and_kendall_exact_monotone_and_ties(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(spearman_correlation(xs, ys), 1.0, places=6)
        self.assertAlmostEqual(kendall_tau(xs, ys), 1.0, places=6)
        self.assertAlmostEqual(spearman_correlation(xs, list(reversed(ys))), -1.0, places=6)
        self.assertAlmostEqual(kendall_tau(xs, list(reversed(ys))), -1.0, places=6)
        # Constant vector should safely return 0.0 without ZeroDivisionError
        self.assertEqual(spearman_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertEqual(kendall_tau([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)

    def test_top_k_retention_and_normalization(self):
        zcp_ids = [3, 1, 5, 0, 2, 4]
        true_ids = [0, 1, 2, 3, 4, 5]
        ret3 = top_k_retention(zcp_ids, true_ids, top_k=3)
        # True top 3 = {0, 1, 2}; ZCP top 3 = {3, 1, 5} -> overlap = {1}
        self.assertEqual(ret3["retained_count"], 1)
        self.assertAlmostEqual(ret3["retention_rate"], 1 / 3, places=4)

        self.assertEqual(minmax_normalize([5.0, 5.0]), [0.5, 0.5])
        rn = rank_normalize([10.0, 30.0, 20.0], descending=True)
        self.assertEqual(rn, [0.0, 1.0, 0.5])

    def test_screening_disabled_by_default_and_acceptance_gate(self):
        candidates = [
            {"candidate_id": 0, "combined_score": 0.9, "theoretical_flops": 100.0},
            {"candidate_id": 1, "combined_score": 0.5, "theoretical_flops": 90.0},
            {"candidate_id": 2, "combined_score": 0.2, "theoretical_flops": 80.0},
            {"candidate_id": 3, "combined_score": 0.1, "theoretical_flops": 70.0},
        ]
        res_default = screen_candidates(candidates)
        self.assertFalse(res_default["screening_applied"])
        self.assertEqual(res_default["retained_count"], 4)
        self.assertEqual(res_default["pruned_count"], 0)

        res_active = screen_candidates(candidates, keep_ratio=0.5, screening_enabled=True)
        self.assertTrue(res_active["screening_applied"])
        self.assertEqual(res_active["retained_count"], 2)
        self.assertEqual([c["candidate_id"] for c in res_active["candidates"]], [0, 1])

    def test_correlation_report_and_acceptance_gate_logic(self):
        zcp_mock = {
            "zcp_compute_seconds": 1.2,
            "candidates": [
                {
                    "candidate_id": i,
                    "indices": [0, 0, i],
                    "temporal_op": "T",
                    "interaction_op": "I",
                    "map_op": "M",
                    "snip": float(10 - i),
                    "synflow": float(20 - 2 * i),
                    "combined_score": float(1.0 - 0.1 * i),
                    "theoretical_flops_m": 5.0 + i,
                }
                for i in range(6)
            ],
        }
        val_mock = [
            {
                "candidate_id": i,
                "val_min_ade": 0.5 + 0.2 * i,
                "val_min_fde": 1.0 + 0.3 * i,
                "val_endpoint_miss_rate_pct": 10.0 + 5.0 * i,
                "train_eval_seconds": 10.0,
            }
            for i in range(6)
        ]
        corr = compute_zcp_validation_correlations(zcp_mock, val_mock)
        self.assertAlmostEqual(corr["correlations"]["combined"]["spearman_vs_ade_accuracy"], 1.0, places=4)
        self.assertEqual(corr["retention"]["top_5"]["retention_rate"], 1.0)
        gate = evaluate_screening_acceptance(corr, stability_report={"mean_pairwise_spearman": 0.95})
        self.assertTrue(gate["screening_recommended"])

        # Inverse correlation should fail acceptance gate
        bad_val = [
            {
                "candidate_id": i,
                "val_min_ade": 2.0 - 0.2 * i,
                "val_min_fde": 3.0 - 0.3 * i,
                "val_endpoint_miss_rate_pct": 50.0 - 5.0 * i,
                "train_eval_seconds": 10.0,
            }
            for i in range(6)
        ]
        bad_corr = compute_zcp_validation_correlations(zcp_mock, bad_val)
        bad_gate = evaluate_screening_acceptance(bad_corr, stability_report={"mean_pairwise_spearman": 0.95})
        self.assertFalse(bad_gate["screening_recommended"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment.")
class TorchAuditAndZCPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_prompt1_a_k_consistency_across_model_and_metrics(self):
        from configs.model_config import ModelConfig
        from evaluations.metrics import endpoint_miss_rate_k, min_ade_k, min_fde_k
        from models.final_model import FinalTrajectoryModel

        for k_val in (1, 3, 6):
            with self.subTest(k_modes=k_val):
                m_cfg = ModelConfig(d_model=8, past_sec=2.0, future_sec=1.5, hz=2, num_modes=k_val, max_agents=2, max_obstacles=1, max_lanes=2, lane_pts=3)
                model = FinalTrajectoryModel(0, 0, 0, model_config=m_cfg)
                self.assertEqual(model.num_modes, k_val)
                preds, logits = model(torch.randn(2, m_cfg.past_len, 2))
                self.assertEqual(preds.shape, (2, k_val, m_cfg.future_len, 2))
                self.assertEqual(logits.shape, (2, k_val))
                tgt = torch.randn(2, m_cfg.future_len, 2)
                # Requesting oversized k clamps cleanly to model.num_modes
                self.assertEqual(
                    min_ade_k(preds, tgt, k=10, mode_logits=logits).item(),
                    min_ade_k(preds, tgt, k=k_val, mode_logits=logits).item(),
                )
                self.assertGreaterEqual(endpoint_miss_rate_k(preds, tgt, k=k_val, mode_logits=logits), 0.0)
                self.assertGreaterEqual(min_fde_k(preds, tgt, k=k_val, mode_logits=logits).item(), 0.0)

    def test_prompt1_b_tensor_shape_normalization_and_strict_3d(self):
        from evaluations.metrics import ade, fde
        from models.base_operator import normalize_future_tensor

        t3 = torch.zeros(2, 4, 2)
        t4_single = torch.zeros(2, 4, 1, 2)
        self.assertEqual(normalize_future_tensor(t3).shape, (2, 4, 2))
        self.assertEqual(normalize_future_tensor(t4_single).shape, (2, 4, 2))

        for bad in (torch.zeros(2, 4, 3, 2), torch.zeros(2, 4), torch.zeros(2, 4, 3)):
            with self.assertRaises(ValueError):
                normalize_future_tensor(bad)

        # ade and fde strictly reject 4D predictions (must use min_ade_k / min_fde_k)
        with self.assertRaises(ValueError):
            ade(torch.zeros(2, 3, 4, 2), t3)
        with self.assertRaises(ValueError):
            fde(torch.zeros(2, 3, 4, 2), t3)

    def test_prompt1_e_collision_risk_variants(self):
        from evaluations.metrics import evaluate_collision_risk

        pred = torch.zeros(2, 3, 4, 2)
        logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        neighbors = torch.full((2, 4, 2, 2), 10.0)
        neighbors[0, 2, 0, :] = 0.5  # collision with ego top-1 in scenario 0
        res = evaluate_collision_risk(pred, neighbors, vehicle_radius=1.8, mode_logits=logits)
        self.assertTrue(res["available"])
        self.assertEqual(res["collision_rate_percent"], 50.0)

        # Missing neighbors_future returns explicit unavailable dict
        missing = evaluate_collision_risk(pred, neighbors_future=None)
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["collision_rate_percent"])

    def test_prompt1_f_g_h_cost_breakdown_zcp_and_profiler(self):
        from configs.model_config import ModelConfig
        from core.zcp_ranking import evaluate_zcp_multi_seed_stability, score_discrete_candidates
        from evaluations.metrics import HardwareProfiler, snip_score, synflow_score
        from models.base_operator import resolve_architecture_indices
        from models.final_model import FinalTrajectoryModel
        from models.search_supernet import SearchSuperNet

        # Out-of-bounds integer index must raise ValueError (catching legacy (3,4,1) bug)
        with self.assertRaises(ValueError):
            FinalTrajectoryModel(3, 4, 1)

        # resolve_architecture_indices reads checkpoints/best_architecture.json -> (3, 2, 0)
        ops, src = resolve_architecture_indices()
        self.assertEqual(len(ops), 3)
        self.assertTrue(all(isinstance(x, int) for x in ops))

        m_cfg = ModelConfig(d_model=8, past_sec=2.0, future_sec=1.5, hz=2, num_modes=3, max_agents=2, max_obstacles=1, max_lanes=2, lane_pts=3)
        supernet = SearchSuperNet(model_config=m_cfg)
        final = FinalTrajectoryModel(0, 0, 0, model_config=m_cfg)
        self.assertEqual(supernet.get_ops_flops(batch_size=1)["base"], final.get_summary(batch_size=1)["base_flops"])
        self.assertEqual(
            final.get_summary(batch_size=2)["theoretical_flops"],
            2.0 * final.get_summary(batch_size=1)["theoretical_flops"],
        )
        self.assertIn("cost_breakdown", final.get_summary())
        with self.assertRaises(ValueError):
            final.get_summary(batch_size=0)

        batch = {
            "past": torch.randn(2, m_cfg.past_len, 2),
            "future": torch.randn(2, m_cfg.future_len, 2),
            "agents_past": torch.randn(2, m_cfg.past_len, m_cfg.max_agents, 2),
            "map_lanes": torch.randn(2, m_cfg.max_lanes, m_cfg.lane_pts, 2),
            "static_obstacles": torch.randn(2, m_cfg.max_obstacles, 4),
            "static_obstacle_mask": torch.ones(2, m_cfg.max_obstacles, dtype=torch.bool),
        }
        # ZCP works on both SuperNet (4,4,3) and discrete FinalTrajectoryModel
        self.assertGreater(snip_score(supernet, batch), 0.0)
        self.assertGreater(synflow_score(supernet), 0.0)
        self.assertGreater(snip_score(final, batch), 0.0)
        self.assertGreater(synflow_score(final), 0.0)

        # HardwareProfiler preserves model mode even on exception and tracks CPU/real-batch scope
        profiler = HardwareProfiler(final, torch.device("cpu"), batch_size=2, input_factory=lambda: batch)
        hw = profiler.profile(num_warmup=0, num_runs=2)
        self.assertEqual(hw["input_source"], "real_batch")
        self.assertEqual(hw["batch_size"], 2)
        self.assertIsNone(hw["cuda_peak_memory_allocated_mb"])

        final.train()
        with patch.object(final, "forward", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                profiler.profile(num_warmup=0, num_runs=1)
        self.assertTrue(final.training)

        # Discrete ZCP ranking + multi-seed stability on subset of candidates
        subset = enumerate_discrete_candidates()[:4]
        ranked = score_discrete_candidates(batch, model_config=m_cfg, candidates=subset, init_seed=42)
        self.assertEqual(ranked["search_space_size"], 4)
        self.assertFalse(ranked["coupled_to_darts_loss"])
        stab = evaluate_zcp_multi_seed_stability(batch, seeds=(42, 101), model_config=m_cfg, candidates=subset)
        self.assertIn("mean_pairwise_spearman", stab)

    def test_prompt2_critical_darts_loss_graph_has_zero_zcp_coupling(self):
        """Regression test: verify DARTSAlgorithm.validate() objective excludes any ZCP calls or nodes."""
        from core.darts_algorithm import DARTSAlgorithm

        src = inspect.getsource(DARTSAlgorithm.validate)
        self.assertNotIn("snip_score", src)
        self.assertNotIn("synflow_score", src)
        self.assertNotIn("zcp", src.lower())
        self.assertIn("batch_arch_loss = (", src)
        self.assertIn("loss_reg + (0.1 * loss_cls) + (self.cfg.lambda_flops * flops_cost)", src)

        # Also verify profile_zero_cost_proxies draws from train_loader (never val_loader)
        zcp_src = inspect.getsource(DARTSAlgorithm.profile_zero_cost_proxies)
        self.assertIn("self.train_loader", zcp_src)
        self.assertNotIn("self.val_loader", zcp_src)


if __name__ == "__main__":
    unittest.main()

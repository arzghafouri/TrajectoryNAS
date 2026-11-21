"""
Unit and Integration Tests for Hard Sample Mining (HSM)
======================================================
Tests mathematical correctness, weight monotonicity, sensitivity scaling,
config validation, and DataLoader integration.
"""

import math
import unittest
import sys
import os

# Ensure python_src in path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.train_config import TrainConfig
from data.hard_sample_mining import HardSampleMiningManager, HardSampleWeightedSampler


class HardSampleMiningTests(unittest.TestCase):
    def test_train_config_default_and_validation(self):
        # Default must be a valid sensitivity value between 0.0 and 1.0
        cfg = TrainConfig()
        self.assertTrue(0.0 <= cfg.hard_sample_mining <= 1.0)
        self.assertEqual(cfg.hard_sample_sensitivity, cfg.hard_sample_mining)
        self.assertEqual(cfg.hard_mining_sensitivity, cfg.hard_sample_mining)

        cfg_zero = TrainConfig(hard_sample_mining=0.0)
        self.assertEqual(cfg_zero.hard_sample_mining, 0.0)

        # Valid values between 0.0 and 1.0
        cfg_half = TrainConfig(hard_sample_mining=0.5)
        self.assertEqual(cfg_half.hard_sample_mining, 0.5)

        cfg_max = TrainConfig(hard_sample_mining=1.0)
        self.assertEqual(cfg_max.hard_sample_mining, 1.0)

        # Out-of-bounds must raise ValueError
        with self.assertRaises(ValueError):
            TrainConfig(hard_sample_mining=-0.1)

        with self.assertRaises(ValueError):
            TrainConfig(hard_sample_mining=1.05)

        with self.assertRaises(ValueError):
            TrainConfig(hard_sample_mining=10.0)

    def test_sensitivity_zero_is_strictly_uniform(self):
        # When gamma == 0, weights must be strictly identical across all samples
        manager = HardSampleMiningManager(num_samples=100, sensitivity=0.0)
        self.assertFalse(manager.is_active)

        # Update samples with disparate errors
        indices = [0, 1, 2, 3]
        errors = [0.01, 5.0, 10.0, 0.05]
        manager.update_sample_errors(indices, errors)

        try:
            import torch
            weights = manager.get_weights()
            # All weights must be exactly 1.0
            self.assertTrue(torch.all(weights == 1.0))
        except ImportError:
            self.assertTrue(all(w == 1.0 for w in manager.weights))

    def test_sensitivity_greater_than_zero_weights_hard_samples_higher(self):
        N = 50
        manager = HardSampleMiningManager(num_samples=N, sensitivity=0.8, ema_momentum=1.0)
        self.assertTrue(manager.is_active)

        # Sample 10 has very high error (hard curve / cut-in)
        # Sample 20 has very low error (straight line)
        manager.update_sample_errors(
            sample_indices=[10, 20],
            sample_errors=[2.5, 0.1]
        )

        weights = manager.get_weights()
        w10 = weights[10].item() if hasattr(weights[10], "item") else weights[10]
        w20 = weights[20].item() if hasattr(weights[20], "item") else weights[20]
        self.assertGreater(w10, w20)

        # Test ratio
        ratio = w10 / w20
        self.assertGreater(ratio, 2.0)

    def test_continuous_sensitivity_interpolation(self):
        # As sensitivity increases from 0 to 1, the weight disparity between hard and easy increases monotonically
        sensitivities = [0.0, 0.25, 0.5, 0.75, 1.0]
        ratios = []

        for s in sensitivities:
            m = HardSampleMiningManager(num_samples=10, sensitivity=s, ema_momentum=1.0)
            m.update_sample_errors([0, 1], [3.0, 0.3])
            w = m.get_weights()
            w0 = w[0].item() if hasattr(w[0], "item") else w[0]
            w1 = w[1].item() if hasattr(w[1], "item") else w[1]
            r = w0 / w1
            ratios.append(r)

        # Ratio at 0.0 must be exactly 1.0
        self.assertAlmostEqual(ratios[0], 1.0, places=5)

        # Ratios must strictly increase with sensitivity
        for i in range(len(ratios) - 1):
            self.assertLess(ratios[i], ratios[i + 1])

    def test_exponential_moving_average_smoothing(self):
        manager = HardSampleMiningManager(num_samples=10, sensitivity=0.5, ema_momentum=0.5)

        # Step 1: initial error = 1.0
        manager.update_sample_errors([0], [1.0])
        val0 = manager.errors[0].item() if hasattr(manager.errors[0], "item") else manager.errors[0]
        self.assertAlmostEqual(val0, 1.0, places=4)

        # Step 2: new error = 3.0 -> with beta=0.5: 0.5*1.0 + 0.5*3.0 = 2.0
        manager.update_sample_errors([0], [3.0])
        val1 = manager.errors[0].item() if hasattr(manager.errors[0], "item") else manager.errors[0]
        self.assertAlmostEqual(val1, 2.0, places=4)

    def test_sampler_integration(self):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            self.skipTest("PyTorch is required for DataLoader sampler tests")

        manager = HardSampleMiningManager(num_samples=20, sensitivity=0.7)
        manager.update_sample_errors([0, 1, 2], [10.0, 0.1, 0.1])

        sampler = manager.get_sampler(num_samples=40, replacement=True)
        self.assertEqual(len(sampler), 40)

        indices = list(sampler)
        self.assertEqual(len(indices), 40)

        # Sample 0 (error=10.0) should be sampled much more frequently than sample 1 (error=0.1)
        count_0 = indices.count(0)
        count_1 = indices.count(1)
        self.assertGreater(count_0, count_1)

    def test_summary_telemetry(self):
        manager = HardSampleMiningManager(num_samples=100, sensitivity=0.6)
        manager.update_sample_errors([0, 1, 2, 3, 4], [0.2, 0.5, 0.8, 1.2, 2.5])

        summary = manager.get_summary()
        self.assertTrue(summary["active"])
        self.assertEqual(summary["sensitivity"], 0.6)
        self.assertEqual(summary["total_samples"], 100)
        self.assertEqual(summary["samples_visited"], 5)
        self.assertGreater(summary["hard_to_easy_weight_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

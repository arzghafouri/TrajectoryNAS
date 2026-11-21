"""Dataset-free regression tests for trajectory metrics, operators and search math."""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluations.metrics import (
    HardwareProfiler,
    ade,
    best_of_k_loss,
    endpoint_miss_rate_k,
    evaluate_collision_risk,
    fde,
    get_supernet_costs,
    max_displacement_miss_rate_k,
    min_ade_k,
    min_fde_k,
    miss_rate,
    miss_rate_k,
    safe_norm,
    simple_complexity_proxy,
    snip_score,
    synflow_score,
)
from models.base_operator import safe_cross_attention, trajectory_kinematics
from models.final_model import FinalTrajectoryModel
from models.search_supernet import ArchParams, SearchSuperNet
from models.supernet_operators import PolylineEncoder


ORIGINAL_THREADS = torch.get_num_threads()
MODEL_OPTIONS = dict(
    d_model=8, past_len=4, future_len=3, num_modes=2,
    max_agents=2, max_obstacles=1, max_lanes=2, lane_pts=3,
)


def setUpModule():
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(ORIGINAL_THREADS)


class MetricTests(unittest.TestCase):
    def test_exact_predictions_have_zero_error_and_finite_gradients(self):
        prediction = torch.zeros(2, 3, 2, dtype=torch.double, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = ade(prediction, target) + fde(prediction, target)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 0)

    def test_low_precision_norm_uses_fp32_without_overflow(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = torch.tensor([[0.0, 0.0], [300.0, 0.0]], dtype=dtype, requires_grad=True)
                distances = safe_norm(values)
                torch.testing.assert_close(distances, torch.tensor([0.0, 300.0]))
                distances.sum().backward()
                self.assertTrue(torch.isfinite(values.grad).all())

    def test_winner_takes_all_matches_manual_loss_and_gradient_routing(self):
        prediction = torch.tensor(
            [[[[1.0, 0.0]] * 3, [[3.0, 0.0]] * 3],
             [[[4.0, 0.0]] * 3, [[2.0, 0.0]] * 3]], requires_grad=True
        )
        logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        regression, classification = best_of_k_loss(prediction, torch.zeros(2, 3, 2), logits)
        self.assertEqual(regression.item(), 2.25)
        torch.testing.assert_close(classification, nn.functional.cross_entropy(logits, torch.tensor([0, 1])))
        (regression + classification).backward()
        self.assertEqual(prediction.grad[0, 1].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[1, 0].abs().sum().item(), 0.0)
        self.assertGreater(prediction.grad[0, 0].abs().sum().item(), 0.0)

    def test_top_k_selection_uses_each_samples_confidence(self):
        prediction = torch.zeros(2, 2, 3, 2)
        prediction[0, 0, :, 0] = 10.0
        prediction[1, 1, :, 0] = 10.0
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        target = torch.zeros(2, 3, 2)
        for metric in (min_ade_k, min_fde_k, miss_rate_k):
            with self.subTest(metric=metric.__name__):
                self.assertEqual(float(metric(prediction, target, k=1, mode_logits=logits)), 0.0)
                self.assertGreater(float(metric(prediction, target, k=1)), 0.0)

    def test_top_k_validation_and_clamping(self):
        prediction = torch.zeros(1, 2, 3, 2)
        target = torch.zeros(1, 3, 2)
        self.assertEqual(min_ade_k(prediction, target, k=20).item(), 0.0)
        for metric in (min_ade_k, min_fde_k, miss_rate_k):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(ValueError):
                    metric(prediction, target, k=0)
                with self.assertRaises(ValueError):
                    metric(prediction, target, mode_logits=torch.zeros(1, 3))

    def test_miss_rate_checks_endpoint_and_max_displacement_separately(self):
        target = torch.zeros(1, 2, 2)
        for distance in (2.0, 3.0):
            with self.subTest(distance=distance):
                # Error only at t=0 (intermediate): endpoint_miss_rate == 0.0, max_displacement == 1.0
                mid_err_pred = torch.tensor([[[distance, 0.0], [0.0, 0.0]]])
                self.assertEqual(miss_rate(mid_err_pred, target), 0.0)
                self.assertEqual(miss_rate_k(mid_err_pred.unsqueeze(1), target), 0.0)
                self.assertEqual(endpoint_miss_rate_k(mid_err_pred.unsqueeze(1), target), 0.0)
                self.assertEqual(max_displacement_miss_rate_k(mid_err_pred.unsqueeze(1), target), 1.0)

                # Error at final step t=1: both endpoint_miss_rate and max_displacement == 1.0
                end_err_pred = torch.tensor([[[0.0, 0.0], [distance, 0.0]]])
                self.assertEqual(miss_rate(end_err_pred, target), 1.0)
                self.assertEqual(miss_rate_k(end_err_pred.unsqueeze(1), target), 1.0)
                self.assertEqual(endpoint_miss_rate_k(end_err_pred.unsqueeze(1), target), 1.0)
                self.assertEqual(max_displacement_miss_rate_k(end_err_pred.unsqueeze(1), target), 1.0)

    def test_miss_rate_requires_one_mode_to_match_endpoint_or_path(self):
        prediction = torch.tensor([[[[3.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]]]])
        # Mode 0 matches endpoint [0, 0], so endpoint miss rate is 0.0, while max-over-path MR is 1.0
        self.assertEqual(endpoint_miss_rate_k(prediction, torch.zeros(1, 2, 2)), 0.0)
        self.assertEqual(max_displacement_miss_rate_k(prediction, torch.zeros(1, 2, 2)), 1.0)
        prediction[:, 1] = 0.0
        self.assertEqual(endpoint_miss_rate_k(prediction, torch.zeros(1, 2, 2)), 0.0)
        self.assertEqual(max_displacement_miss_rate_k(prediction, torch.zeros(1, 2, 2)), 0.0)


class CollisionTests(unittest.TestCase):
    def test_invalid_timesteps_do_not_count_as_collisions(self):
        prediction = torch.zeros(1, 2, 2)
        neighbors = torch.tensor([[[[0.0, 0.0]], [[10.0, 0.0]]]])
        result = evaluate_collision_risk(prediction, neighbors, torch.tensor([[[False], [True]]]))
        self.assertTrue(result['available'])
        self.assertEqual(result['collision_rate_percent'], 0.0)

    def test_stationary_neighbor_at_origin_is_valid(self):
        for mask in (None, torch.ones(1, 1, dtype=torch.bool)):
            with self.subTest(mask=mask):
                result = evaluate_collision_risk(torch.zeros(1, 2, 2), torch.zeros(1, 2, 1, 2), mask)
                self.assertEqual(result['collision_rate_percent'], 100.0)

    def test_collision_denominator_excludes_samples_without_valid_neighbors(self):
        result = evaluate_collision_risk(
            torch.zeros(2, 2, 2), torch.zeros(2, 2, 1, 2), torch.tensor([[True], [False]])
        )
        self.assertEqual(result['scenarios_evaluated'], 1)
        self.assertEqual(result['collided_scenarios'], 1)
        self.assertEqual(result['collision_rate_percent'], 100.0)

    def test_absent_future_data_and_empty_neighbors_remain_unavailable(self):
        prediction = torch.zeros(1, 2, 2)
        for neighbors in (None, torch.zeros(1, 3, 1, 2), torch.zeros(1, 2, 0, 2)):
            with self.subTest(neighbors=neighbors):
                self.assertFalse(evaluate_collision_risk(prediction, neighbors)['available'])


class KinematicAndMapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_stationary_kinematics_have_finite_gradients(self):
        for channels in (2, 4):
            with self.subTest(channels=channels):
                tracks = torch.zeros(1, 4, channels, requires_grad=True)
                features = trajectory_kinematics(tracks)
                self.assertTrue(torch.equal(features[..., -1], torch.ones(1, 4)))
                features.sum().backward()
                self.assertTrue(torch.isfinite(tracks.grad).all())

    def test_velocity_acceleration_and_heading_units(self):
        tracks = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
        features = trajectory_kinematics(tracks, hz=2)
        torch.testing.assert_close(features[..., 2:4], torch.tensor([2.0, 4.0]).expand(1, 4, 2))
        self.assertEqual(features[..., 4:6].abs().sum().item(), 0.0)
        torch.testing.assert_close(features[..., 6:].square().sum(-1), torch.ones(1, 4))

    def test_fully_masked_polylines_remain_zero_after_projection(self):
        encoder = PolylineEncoder(d_model=8)
        cases = [
            (torch.randn(1, 2, 3, 2), torch.zeros(1, 2, 3, dtype=torch.bool)),
            (torch.randn(1, 2, 3, 2), torch.zeros(1, 2, dtype=torch.bool)),
            (torch.randn(1, 2, 4), torch.zeros(1, 2, dtype=torch.bool)),
        ]
        for values, mask in cases:
            with self.subTest(shape=values.shape, mask_shape=mask.shape):
                encoder.zero_grad()
                output = encoder(values, lane_types=torch.ones(1, 2, dtype=torch.long), mask=mask)
                self.assertEqual(output.abs().sum().item(), 0.0)
                output.sum().backward()
                self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                    for parameter in encoder.parameters()))

    def test_all_valid_point_mask_matches_unmasked_max_pooling(self):
        encoder = PolylineEncoder(d_model=8)
        lanes = torch.randn(1, 2, 3, 2)
        torch.testing.assert_close(encoder(lanes), encoder(lanes, mask=torch.ones(1, 2, 3, dtype=torch.bool)))

    def test_invalid_points_do_not_leak_through_tangents(self):
        encoder = PolylineEncoder(d_model=8)
        lanes = torch.randn(1, 2, 3, 2)
        mask = torch.tensor([[[True, False, True], [False, True, True]]])
        changed = lanes.masked_fill(~mask.unsqueeze(-1), 1000.0)
        torch.testing.assert_close(encoder(lanes, mask=mask), encoder(changed, mask=mask))

    def test_attention_all_masked_rows_have_finite_backward(self):
        attention = nn.MultiheadAttention(8, 2, batch_first=True)
        query = torch.randn(2, 4, 8, requires_grad=True)
        context = torch.randn(2, 3, 8, requires_grad=True)
        output = safe_cross_attention(
            attention, query, context, context, torch.tensor([[True, True, True], [False, True, False]])
        )
        self.assertEqual(output[0].abs().sum().item(), 0.0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_architecture_softmax_preserves_logit_differences_and_gradients(self):
        architecture = ArchParams(num_temporal=2)
        for temperature in (1.0, 0.25, 0.05):
            with self.subTest(temperature=temperature):
                architecture.zero_grad()
                with torch.no_grad():
                    architecture.alpha_temporal.copy_(torch.tensor([30.0, 31.0]))
                actual = architecture.softmax_all(temperature)[0]
                expected_logits = torch.tensor([30.0, 31.0], requires_grad=True)
                expected = torch.softmax(expected_logits / temperature, dim=-1)
                torch.testing.assert_close(actual, expected)
                actual[0].backward()
                expected[0].backward()
                torch.testing.assert_close(architecture.alpha_temporal.grad, expected_logits.grad)

    def test_invalid_architecture_values_raise_without_mutating_parameters(self):
        architecture = ArchParams()
        for temperature in (0.0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                architecture.softmax_all(temperature)
        with torch.no_grad():
            architecture.alpha_temporal[0] = float('inf')
        with self.assertRaises(ValueError):
            architecture.softmax_all()
        self.assertTrue(torch.isinf(architecture.alpha_temporal[0]))

    def test_expected_cost_proxy_is_not_constant_and_has_correct_gradient(self):
        architecture = ArchParams()
        costs = {
            'temporal': [1e6 * (i + 1) for i in range(architecture.num_temporal)],
            'interaction': [2e6 * (i + 1) for i in range(architecture.num_interaction)],
            'map': [1e6 * (i + 1) for i in range(architecture.num_map)],
            'base': 5e6,
        }
        weights = architecture.softmax_all()
        actual = simple_complexity_proxy(weights, costs)
        expected = sum((probability * torch.tensor(costs[branch])).sum()
                       for probability, branch in zip(weights, ('temporal', 'interaction', 'map')))
        torch.testing.assert_close(actual, (expected + costs['base']) / 1e9)
        torch.testing.assert_close(actual, get_supernet_costs(weights, costs))
        actual.backward()
        probability = weights[0].detach()
        temporal_cost = torch.tensor(costs['temporal'])
        expected_gradient = probability * (temporal_cost - (probability * temporal_cost).sum()) / 1e9
        torch.testing.assert_close(architecture.alpha_temporal.grad, expected_gradient)
        self.assertGreater(expected_gradient.abs().sum().item(), 0.0)

    def test_static_obstacles_affect_gcn_without_agent_history(self):
        for model in (SearchSuperNet(**MODEL_OPTIONS), FinalTrajectoryModel(0, 1, 0, **MODEL_OPTIONS)):
            with self.subTest(model=type(model).__name__):
                model.eval()
                obstacles = torch.tensor([[[8.0, 2.0, 1.0, 1.0]]], requires_grad=True)
                arguments = (torch.randn(1, 4, 2),)
                if isinstance(model, SearchSuperNet):
                    n_t = len(model.temporal_branch.ops)
                    n_i = len(model.interaction_branch.ops)
                    n_m = len(model.map_branch.ops)
                    arguments += (tuple(nn.functional.one_hot(torch.tensor(index), count).float()
                                        for index, count in ((0, n_t), (1, n_i), (0, n_m))),)
                prediction, _ = model(*arguments, static_obstacles=obstacles,
                                      static_obstacle_mask=torch.ones(1, 1, dtype=torch.bool))
                prediction.sum().backward()
                self.assertTrue(torch.isfinite(obstacles.grad).all())
                self.assertGreater(obstacles.grad.abs().sum().item(), 0.0)

    def test_forward_does_not_mutate_caller_agent_mask(self):
        for model in (SearchSuperNet(**MODEL_OPTIONS), FinalTrajectoryModel(0, 1, 0, **MODEL_OPTIONS)):
            with self.subTest(model=type(model).__name__):
                mask = torch.tensor([[False, True]])
                arguments = (torch.randn(1, 4, 2),)
                if isinstance(model, SearchSuperNet):
                    arguments += (ArchParams().softmax_all(),)
                model(*arguments, agents_past=torch.randn(1, 4, 2, 2), agent_mask=mask)
                self.assertFalse(mask[0, 0])

    def test_decoder_flops_include_input_and_recurrent_transforms(self):
        network = SearchSuperNet(**MODEL_OPTIONS)
        dimension = network.d_model
        expected_base = (
            2 * network.past_len * 8 * dimension
            + 2 * network.past_len * network.max_agents * 8 * dimension
            + 2 * network.max_obstacles * 4 * dimension
            + network.map_encoder.get_flops(network.max_lanes, network.lane_pts, dimension)
            + 2 * network.past_len * 3 * dimension * dimension
            + network.future_len * (12 * dimension * dimension + 18 * dimension)
            + 2 * network.future_len * dimension * (network.num_modes * 2)
            + 2 * dimension * dimension + 2 * dimension * network.num_modes
        )
        self.assertEqual(network.get_ops_flops()['base'], expected_base)
        final = FinalTrajectoryModel(0, 0, 0, **MODEL_OPTIONS)
        self.assertEqual(final.get_summary()['base_flops'], expected_base)
        self.assertEqual(final.get_summary()['theoretical_flops'], expected_base + final.get_summary()['op_flops'])

    def test_one_hot_supernet_matches_all_100_discrete_architectures(self):
        network = SearchSuperNet(**MODEL_OPTIONS).eval()
        past = torch.randn(2, 4, 2)
        context = dict(
            agents_past=torch.randn(2, 4, 2, 2), agent_mask=torch.ones(2, 2, dtype=torch.bool),
            map_lanes=torch.randn(2, 2, 3, 2), lane_mask=torch.ones(2, 2, dtype=torch.bool),
            static_obstacles=torch.randn(2, 1, 4), static_obstacle_mask=torch.ones(2, 1, dtype=torch.bool),
        )
        shared = ('input_proj', 'neighbor_proj', 'category_emb', 'obstacle_proj', 'obstacle_type_emb',
                  'map_encoder', 'fusion', 'decoder', 'out_proj', 'prob_head')
        n_t = len(network.temporal_branch.ops)
        n_i = len(network.interaction_branch.ops)
        n_m = len(network.map_branch.ops)
        for temporal in range(n_t):
            for interaction in range(n_i):
                for map_index in range(n_m):
                    with self.subTest(temporal=temporal, interaction=interaction, map_index=map_index):
                        final = FinalTrajectoryModel(temporal, interaction, map_index, **MODEL_OPTIONS).eval()
                        for name in shared:
                            getattr(final, name).load_state_dict(getattr(network, name).state_dict())
                        final.temporal_op.load_state_dict(network.temporal_branch.ops[temporal].state_dict())
                        final.interaction_op.load_state_dict(network.interaction_branch.ops[interaction].state_dict())
                        final.map_op.load_state_dict(network.map_branch.ops[map_index].state_dict())
                        weights = tuple(nn.functional.one_hot(torch.tensor(index), count).float()
                                        for index, count in ((temporal, n_t), (interaction, n_i), (map_index, n_m)))
                        with torch.no_grad():
                            actual = network(past, weights, **context)
                            expected = final(past, **context)
                        for actual_tensor, expected_tensor in zip(actual, expected):
                            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


class SaliencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.network = SearchSuperNet(**MODEL_OPTIONS).eval()
        self.network.temporal_branch.ops[3].norm.train()
        for parameter in self.network.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
        self.batch = dict(
            past=torch.randn(2, 4, 2), future=torch.randn(2, 3, 2),
            agents_past=torch.randn(2, 4, 2, 2), map_lanes=torch.randn(2, 2, 3, 2),
            static_obstacles=torch.randn(2, 1, 4), static_obstacle_mask=torch.ones(2, 1, dtype=torch.bool),
        )

    def assert_proxy_preserves_state(self, proxy, fail=False):
        state = {name: value.clone() for name, value in self.network.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in self.network.parameters()]
        modes = [module.training for module in self.network.modules()]
        cudnn = torch.backends.cudnn.enabled
        original_forward = self.network.forward

        def failing_forward(*args, **kwargs):
            original_forward(*args, **kwargs)
            raise RuntimeError('intentional scoring failure')

        def score():
            return proxy(self.network, self.batch) if proxy is snip_score else proxy(self.network)

        if fail:
            with patch.object(self.network, 'forward', side_effect=failing_forward):
                with self.assertRaisesRegex(RuntimeError, 'intentional scoring failure'):
                    score()
        else:
            result = score()
            self.assertTrue(math.isfinite(result))
            self.assertGreater(result, 0.0)
        for name, value in self.network.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        for parameter, gradient in zip(self.network.parameters(), gradients):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertEqual([module.training for module in self.network.modules()], modes)
        self.assertEqual(torch.backends.cudnn.enabled, cudnn)

    def test_snip_preserves_full_model_state(self):
        self.assert_proxy_preserves_state(snip_score)

    def test_synflow_preserves_full_model_state(self):
        self.assert_proxy_preserves_state(synflow_score)

    def test_proxy_failures_restore_full_state(self):
        for proxy in (snip_score, synflow_score):
            with self.subTest(proxy=proxy.__name__):
                self.assert_proxy_preserves_state(proxy, fail=True)


class _TinyModel(nn.Module):
    def __init__(self, summary=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.summary = summary or {}

    def forward(self, past, **kwargs):
        return past * self.weight

    def get_summary(self):
        return self.summary


class HardwareTests(unittest.TestCase):
    def profile(self, summary=None):
        return HardwareProfiler(
            _TinyModel(summary), torch.device('cpu'), past_len=2,
            max_agents=1, max_lanes=1, lane_pts=1, max_obstacles=0,
        ).profile(num_warmup=0, num_runs=1)

    def test_unknown_flops_are_not_fabricated(self):
        profile = self.profile()
        for name in ('theoretical_flops_g', 'theoretical_flops_m', 'theoretical_macs_m',
                     'base_flops_m', 'op_flops_m'):
            self.assertIsNone(profile[name], name)

    def test_gigaflops_and_megaflops_are_consistent(self):
        for summary in ({'theoretical_flops_g': 0.012}, {'theoretical_flops_m': 12.0}):
            with self.subTest(summary=summary):
                profile = self.profile(summary)
                self.assertEqual(profile['theoretical_flops_g'], 0.012)
                self.assertEqual(profile['theoretical_flops_m'], 12.0)
                self.assertEqual(profile['theoretical_macs_m'], 6.0)

    def test_zero_flops_are_preserved(self):
        profile = self.profile({'theoretical_flops_g': 0.0})
        self.assertEqual(profile['theoretical_flops_m'], 0.0)
        self.assertEqual(profile['theoretical_macs_m'], 0.0)


if __name__ == '__main__':
    unittest.main()

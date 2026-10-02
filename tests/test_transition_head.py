"""CPU-only Transition Head architecture and loss regression tests.

No frozen action model/CLIP initialization, CUDA work, or simulator is required.
Run from RAIN with: pytest tests/test_transition_head.py
"""
import ast
import inspect
import textwrap
import unittest

import torch
from torch import nn

from rainv2.configs.config import ProgressConfig
from rainv2.models.model import RAINModel
from rainv2.models.progress_heads import SingleViewProgressHead

from rain.transition_head import PoolingTCHead, VARIANTS, region_mean, tc_loss
from rain.model import PoolingModel


def _body(function):
    body = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def _prefix(function, stop_assignment):
    result = []
    for node in _body(function):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == stop_assignment:
            break
        result.append(ast.dump(node, include_attributes=False))
    else:
        raise AssertionError(f"Missing expected assignment: {stop_assignment}")
    return result


class TestPoolingTCHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20260913)
        self.third = torch.randn(4, 11, 1024)
        self.wrist = torch.randn_like(self.third)
        self.action = torch.randn(4, 768)
        self.third_mask = torch.zeros(4, 11)
        self.wrist_mask = torch.zeros(4, 11)
        # Both absent, third only, wrist only, and two differently sized regions.
        self.third_mask[1, :] = 1
        self.wrist_mask[2, [0, 3, 7]] = 1
        self.third_mask[3, [1, 2]] = 1
        self.wrist_mask[3, [0, 4, 5, 9]] = 1

    def inputs(self):
        return self.third, self.wrist, self.third_mask, self.wrist_mask, self.action

    def test_all_six_shapes_and_only_one_tc_output(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                head = PoolingTCHead(variant).eval()
                with torch.no_grad():
                    predictions, gate = head(*self.inputs())
                self.assertEqual(set(predictions), {"task_comp_logit", "task_comp_prob"})
                for value in predictions.values():
                    self.assertEqual(value.shape, (4,))
                    self.assertTrue(torch.isfinite(value).all())
                if variant.endswith("_gated"):
                    self.assertEqual(gate.shape, (4, 2))
                    self.assertTrue(((gate >= 0) & (gate <= 1)).all())
                    torch.testing.assert_close(gate.sum(-1), torch.ones(4))
                else:
                    self.assertIsNone(gate)
                    self.assertFalse(hasattr(head, "gate"))

    def test_region_mean_empty_full_ragged_and_threshold(self):
        tokens = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
        mask = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1],
                             [0, 1, 0, 1, 0], [0.5, 0.50001, 0, 0, 0]])
        actual = region_mean(tokens, mask)
        expected = torch.stack((tokens[0].mean(0), tokens[1].mean(0),
                                tokens[2, [1, 3]].mean(0), tokens[3, 1]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_missing_mask_uses_global_mean_not_zero(self):
        zero_mask = torch.zeros(4, 11)
        for variant in ("region_gated", "region_concat", "global_region_gated", "global_region_concat"):
            with self.subTest(variant=variant):
                head = PoolingTCHead(variant)
                pooled = head.pool_view(self.third, zero_mask)
                global_feature = head.input_proj(self.third.mean(1))
                expected = (torch.cat((global_feature, global_feature), -1)
                            if variant.startswith("global_region_") else global_feature)
                torch.testing.assert_close(pooled, expected)

    def test_global_only_ignores_head_mask_given_fixed_encoded_features(self):
        # Mask conditioning remains active inside the frozen encoder.
        for variant in ("global_gated", "global_concat"):
            with self.subTest(variant=variant):
                head = PoolingTCHead(variant).eval()
                first, _ = head(*self.inputs())
                second, _ = head(self.third, self.wrist, 1 - self.third_mask,
                                 1 - self.wrist_mask, self.action)
                self.assertTrue(torch.equal(first["task_comp_logit"], second["task_comp_logit"]))

    def test_pooling_precedes_shared_projection(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                head = PoolingTCHead(variant)
                projected_shapes = []
                hook = head.input_proj.register_forward_pre_hook(
                    lambda _, args: projected_shapes.append(tuple(args[0].shape)))
                head(*self.inputs())
                hook.remove()
                expected_calls = 4 if variant.startswith("global_region_") else 2
                self.assertEqual(projected_shapes, [(4, 1024)] * expected_calls)

    def test_global_region_order_and_shared_projection_exact(self):
        for variant in ("global_region_gated", "global_region_concat"):
            head = PoolingTCHead(variant)
            expected = torch.cat((head.input_proj(self.third.mean(1)),
                                  head.input_proj(region_mean(self.third, self.third_mask))), -1)
            self.assertTrue(torch.equal(head.pool_view(self.third, self.third_mask), expected))

    def test_fusion_and_action_concatenation_formulas_exact(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                head = PoolingTCHead(variant).eval()
                tc_inputs, action_calls = [], []
                hooks = [
                    head.tc.register_forward_pre_hook(lambda _, args: tc_inputs.append(args[0].detach().clone())),
                    head.action_proj.register_forward_pre_hook(lambda _, args: action_calls.append(args[0])),
                ]
                predictions, weights = head(*self.inputs())
                for hook in hooks:
                    hook.remove()
                self.assertEqual(len(action_calls), 1)
                self.assertIs(action_calls[0], self.action)
                third = head.pool_view(self.third, self.third_mask)
                wrist = head.pool_view(self.wrist, self.wrist_mask)
                concatenated = torch.cat((third, wrist), -1)
                if variant.endswith("_gated"):
                    gate = head.gate(concatenated)
                    fused = (1 - gate) * third + gate * wrist
                    self.assertTrue(torch.equal(weights, torch.cat((1 - gate, gate), -1)))
                else:
                    fused = concatenated
                    self.assertIsNone(weights)
                expected_input = torch.cat((fused, head.action_proj(self.action)), -1)
                self.assertTrue(torch.equal(tc_inputs[0], expected_input))
                self.assertTrue(torch.equal(predictions["task_comp_logit"], head.tc(expected_input).squeeze(-1)))

    def test_missing_regions_do_not_force_view_weights(self):
        for variant in ("global_gated", "region_gated", "global_region_gated"):
            head = PoolingTCHead(variant)
            with torch.no_grad():
                for parameter in head.gate.parameters():
                    parameter.zero_()
                _, weights = head(*self.inputs())
            self.assertTrue(torch.equal(weights, torch.full((4, 2), 0.5)))

    def test_no_attention_token_ffn_or_auxiliary_modules(self):
        for variant in VARIANTS:
            head = PoolingTCHead(variant)
            expected_children = {"input_proj", "action_proj", "tc"}
            if variant.endswith("_gated"):
                expected_children.add("gate")
            self.assertEqual(set(dict(head.named_children())), expected_children)
            for module in head.modules():
                self.assertNotIsInstance(module, (nn.MultiheadAttention, nn.Transformer,
                                                   nn.TransformerEncoder, nn.TransformerDecoder))
            self.assertFalse(any("attn" in key or "ffn" in key or "distance" in key or "alignment" in key
                                 for key in head.state_dict()))
            self.assertFalse(head.tc[0].elementwise_affine)
            if hasattr(head, "gate"):
                self.assertFalse(head.gate[0].elementwise_affine)

    def test_common_initialization_equal_for_same_seed(self):
        states = {}
        for variant in VARIANTS:
            torch.manual_seed(42)
            states[variant] = PoolingTCHead(variant).state_dict()
        for state in states.values():
            for key in ("input_proj.weight", "input_proj.bias", "action_proj.weight", "action_proj.bias"):
                self.assertTrue(torch.equal(state[key], states["global_gated"][key]))
        for first, second in (("global_gated", "region_gated"),
                              ("global_concat", "region_concat")):
            self.assertEqual(set(states[first]), set(states[second]))
            for key in states[first]:
                self.assertTrue(torch.equal(states[first][key], states[second][key]), key)
        # Both of these have a 576-input TC despite different fusion strategies.
        for key in states["global_concat"]:
            if key.startswith("tc."):
                self.assertTrue(torch.equal(states["global_concat"][key],
                                            states["global_region_gated"][key]), key)

    def test_all_parameters_receive_finite_gradients_for_all_mask_patterns(self):
        config = ProgressConfig(lambda_dist=0., lambda_align=0., lambda_tc=1.2,
                                release_task_comp_weight=1.5, tc_mask_dist_thresh=0.8)
        for variant in VARIANTS:
            for pattern in ("mixed", "empty", "full"):
                with self.subTest(variant=variant, mask=pattern):
                    head = PoolingTCHead(variant).train()
                    tm, wm = self.third_mask, self.wrist_mask
                    if pattern == "empty":
                        tm, wm = torch.zeros_like(tm), torch.zeros_like(wm)
                    elif pattern == "full":
                        tm, wm = torch.ones_like(tm), torch.ones_like(wm)
                    preds, _ = head(self.third, self.wrist, tm, wm, self.action)
                    loss = tc_loss(preds["task_comp_logit"], torch.tensor([0.2, 0.9, 0.7, 0.5]),
                                   torch.tensor([0., 1., 0., 1.]), torch.tensor([0, 1, 5, 1]), config)
                    loss["progress_loss"].backward()
                    for name, parameter in head.named_parameters():
                        self.assertIsNotNone(parameter.grad, (variant, pattern, name))
                        self.assertTrue(torch.isfinite(parameter.grad).all(), (variant, pattern, name))

    def test_all_ineligible_still_connects_every_parameter_with_zero_gradient(self):
        config = ProgressConfig(lambda_dist=0., lambda_align=0., tc_mask_dist_thresh=0.8)
        for variant in VARIANTS:
            head = PoolingTCHead(variant).train()
            predictions, _ = head(*self.inputs())
            loss = tc_loss(predictions["task_comp_logit"], torch.ones(4), torch.zeros(4), None, config)
            self.assertEqual(loss["progress_loss"].item(), 0.)
            loss["progress_loss"].backward()
            for name, parameter in head.named_parameters():
                self.assertIsNotNone(parameter.grad, (variant, name))
                self.assertEqual(parameter.grad.abs().sum().item(), 0., (variant, name))

    def test_state_dict_roundtrip_all_six(self):
        for variant in VARIANTS:
            first, second = PoolingTCHead(variant).eval(), PoolingTCHead(variant).eval()
            second.load_state_dict(first.state_dict(), strict=True)
            output_first, _ = first(*self.inputs())
            output_second, _ = second(*self.inputs())
            self.assertTrue(torch.equal(output_first["task_comp_logit"], output_second["task_comp_logit"]))

    def test_rejects_invalid_variant_and_input_shapes(self):
        with self.assertRaises(ValueError):
            PoolingTCHead("legacy")
        head = PoolingTCHead()
        with self.assertRaises(ValueError):
            head(self.third, self.wrist[:, :-1], self.third_mask, self.wrist_mask, self.action)
        with self.assertRaises(ValueError):
            head(self.third, self.wrist, self.third_mask[:, :-1], self.wrist_mask, self.action)
        with self.assertRaises(ValueError):
            head(self.third, self.wrist, self.third_mask, self.wrist_mask, self.action[:, :-1])


class TestOriginalTCProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_same_original_loss(self, values, distance, tc, actions, threshold):
        config = ProgressConfig(lambda_dist=0., lambda_align=0., lambda_tc=1.2,
                                release_task_comp_weight=1.5, tc_mask_dist_thresh=threshold)
        old_logits = values.detach().clone().requires_grad_()
        new_logits = values.detach().clone().requires_grad_()
        old = SingleViewProgressHead.compute_loss(
            {"pred_distance": torch.full_like(values, 0.25),
             "pred_alignment": torch.full_like(values, 0.75), "task_comp_logit": old_logits},
            distance, torch.zeros_like(values), tc, config, actions,
        )
        new = tc_loss(new_logits, distance, tc, actions, config)
        self.assertEqual(set(new), {"loss_task_comp", "progress_loss"})
        for key in new:
            self.assertTrue(torch.equal(old[key], new[key]), (key, old[key], new[key]))
        old_grad = torch.autograd.grad(old["progress_loss"], old_logits)[0]
        new_grad = torch.autograd.grad(new["progress_loss"], new_logits)[0]
        self.assertTrue(torch.equal(old_grad, new_grad))
        self.assertTrue(torch.isfinite(new_grad).all())
        return new_grad

    def test_original_tc_loss_and_gradient_match_bitwise(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float64):
            values = torch.randn(257, dtype=dtype) * 8
            distance = torch.rand(257, dtype=dtype)
            tc = torch.randint(0, 2, (257,)).to(dtype)
            actions = torch.randint(0, 7, (257,))
            for threshold in (0., 0.8):
                for action in (actions, None):
                    with self.subTest(dtype=dtype, threshold=threshold, actions=action is not None):
                        self.assert_same_original_loss(values, distance, tc, action, threshold)

    def test_eligibility_boundary_release_weight_and_zero_loss(self):
        gradient = self.assert_same_original_loss(
            torch.tensor([-0.7, 0.4, 0.1, -0.2, 0.8, -0.6]),
            torch.tensor([0.8, 0.80001, 0.99, 0.2, 0.2, 0.99]),
            torch.tensor([0., 0., 1., 1., 0., 1.]), torch.tensor([0, 0, 1, 1, 1, 0]), 0.8)
        self.assertNotEqual(gradient[0].item(), 0.)
        self.assertEqual(gradient[1].item(), 0.)
        self.assertNotEqual(gradient[2].item(), 0.)
        empty = self.assert_same_original_loss(torch.tensor([-100., 0., 100.]), torch.ones(3),
                                               torch.zeros(3), torch.tensor([0, 1, 5]), 0.8)
        self.assertTrue(torch.equal(empty, torch.zeros(3)))

    def test_auxiliary_losses_rejected(self):
        with self.assertRaises(AssertionError):
            tc_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), None,
                    ProgressConfig(lambda_dist=0.3, lambda_align=0.))

    def test_direct_original_subclass_and_inherited_frozen_action_methods(self):
        self.assertEqual(PoolingModel.__bases__, (RAINModel,))
        for name in ("forward_action", "_resolve_vision_scales", "_encode_views_split",
                     "_encode_vision_scales", "_build_condition_text_feat", "_progress_action_type",
                     "set_training_stage"):
            self.assertIs(getattr(PoolingModel, name), getattr(RAINModel, name), name)

    def test_frozen_training_feature_and_condition_prefix_is_ast_identical(self):
        original = _prefix(RAINModel.forward_progress, "progress_action_type")
        pooling = _prefix(PoolingModel.forward_progress, "(progress_preds, _)")
        self.assertEqual(original, pooling)

    def test_predict_action_multiscale_and_condition_prefix_is_ast_identical(self):
        original = _prefix(RAINModel.predict, "progress_action_type")
        pooling = _prefix(PoolingModel.predict, "(progress_preds, fusion_gate)")
        self.assertEqual(original, pooling)

    def test_no_dependency_on_previous_attention_experiments_or_settings(self):
        import rain.transition_head as head_module
        import rain.model as model_module
        for module in (head_module, model_module):
            imports = [node for node in ast.walk(ast.parse(inspect.getsource(module)))
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
            for node in imports:
                source = ast.unparse(node)
                self.assertNotIn("region_only_tc", source)
                self.assertNotIn("attention_full", source)
                self.assertNotIn("settings", source)


if __name__ == "__main__":
    unittest.main()

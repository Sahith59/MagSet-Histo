"""Small numerical and supervision checks; no cohort features or model fitting."""
import copy
import io
import math
import unittest

try:
    import torch
    from magset.case_mil import CaseMIL, all_subset_outputs, fixed_batch_case_loss
    from magset.baseline import SUBSETS
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch is required for frozen-field MIL checks')
class CaseMILTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        torch.manual_seed(812)
        self.x = torch.randn(3, 4, 5, 7)
        self.mask = torch.ones(3, 4, 5, dtype=torch.bool)
        self.mask[:, :, 3:] = False
        self.counts = [11, 25, 2, 5, 3]

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self, pool='mean', arm='dual'):
        return CaseMIL(7, 6, 3, .25, pool, arm).eval()

    def test_shared_initialization_is_exactly_paired_across_pooling_modes(self):
        for arm in ('direct', 'dual'):
            torch.manual_seed(47); mean = self.model('mean', arm)
            torch.manual_seed(47); attention = self.model('attention', arm)
            for name, value in mean.state_dict().items():
                torch.testing.assert_close(value, attention.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(set(attention.state_dict())-set(mean.state_dict()),
                             {'field_value.weight', 'field_value.bias', 'field_gate.weight', 'field_gate.bias', 'field_score.weight'})

    def test_padding_field_permutation_and_unavailable_view_values_do_not_affect_eval(self):
        for pool in ('mean', 'attention'):
            model = self.model(pool)
            base = model(self.x, self.mask, (0, 2))['joint_log_probabilities']
            changed = self.x.clone(); changed[~self.mask] = torch.nan
            changed[:, 1] = torch.inf; changed[:, 3] = -torch.inf
            # Requested-out views are hidden even when the supplied field mask marks them present.
            torch.testing.assert_close(base, model(changed, self.mask, (0, 2))['joint_log_probabilities'], rtol=0, atol=0)
            observed = self.mask.clone(); observed[:, 1] = False; observed[:, 3] = False
            candidate = model(changed, observed, (0, 2))['joint_log_probabilities']
            torch.testing.assert_close(base, candidate, rtol=0, atol=0)
            permutation = [2, 4, 0, 3, 1]
            result = model(self.x[:, :, permutation], self.mask[:, :, permutation], (0, 2))['joint_log_probabilities']
            torch.testing.assert_close(base, result, rtol=1e-6, atol=1e-6)
            longer = torch.cat((self.x, torch.full((3, 4, 2, 7), torch.nan)), 2)
            longer_mask = torch.cat((self.mask, torch.zeros(3, 4, 2, dtype=torch.bool)), 2)
            torch.testing.assert_close(base, model(longer, longer_mask, (0, 2))['joint_log_probabilities'], rtol=0, atol=0)

    def test_view_permutation_preserves_values_and_flags_have_canonical_positions(self):
        model = self.model('attention', 'direct')
        views, present = model.encode_views(self.x, self.mask)
        for subset in ((0,), (1, 3), (0, 1, 2, 3)):
            available = torch.zeros_like(present); available[:, list(subset)] = True
            value = torch.cat((views[:, subset].mean(1), available.float()), 1)
            manual = torch.log_softmax(model.heads['direct'](value), -1)
            torch.testing.assert_close(manual, model(self.x, self.mask, subset)['joint_log_probabilities'])
        # Relabel input view positions and the four corresponding head flag columns.
        permutation = [2, 0, 3, 1]; paired = copy.deepcopy(model)
        with torch.no_grad():
            paired.heads['direct'].weight[:, 6:] = model.heads['direct'].weight[:, 6:][:, permutation]
        before = model(self.x, self.mask, (0, 3))['joint_log_probabilities']
        after = paired(self.x[:, permutation], self.mask[:, permutation], (1, 2))['joint_log_probabilities']
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)

    def test_gated_attention_has_correct_scalar_weights_and_zero_padding_mass(self):
        model = self.model('attention', 'direct')
        pooled, _ = model.encode_views(self.x, self.mask)
        with torch.no_grad():
            h = model.project(self.x[0, 0, :3])
            scores = []
            for vector in h:
                score = sum(float(model.field_score.weight[0, k])
                            *math.tanh(float(model.field_value(vector)[k]))
                            /(1+math.exp(-float(model.field_gate(vector)[k]))) for k in range(3))
                scores.append(score)
            mass = [math.exp(s-max(scores)) for s in scores]
            expected = sum(h[i]*(mass[i]/sum(mass)) for i in range(3))
            torch.testing.assert_close(pooled[0, 0], expected, rtol=1e-6, atol=1e-6)

    def test_all_subsets_share_one_projection_and_match_separate_eval_predictions(self):
        for arm in ('direct', 'dual'):
            model = self.model('attention', arm); calls = []
            handle = model.project.register_forward_hook(lambda *args: calls.append(1))
            result = all_subset_outputs(model, self.x, self.mask)
            handle.remove(); self.assertEqual(len(calls), 1)
            self.assertEqual(result['joint_log_probabilities'].shape, (15, 3, 5))
            torch.testing.assert_close(result['joint_log_probabilities'].logsumexp(-1), torch.zeros(15, 3), rtol=0, atol=2e-7)
            for i, subset in enumerate(SUBSETS):
                expected = model(self.x, self.mask, subset)
                for name in expected:
                    # Selecting fields before GEMM can change its FP32 reduction path.
                    torch.testing.assert_close(result[name][i], expected[name], rtol=1e-6, atol=1e-6)

    def test_direct_and_dual_losses_match_manual_case_weights_and_subset_average(self):
        labels = torch.tensor([0, 2, 4]); number = sum(self.counts)
        for arm in ('direct', 'dual'):
            outputs = all_subset_outputs(self.model('mean', arm), self.x, self.mask)
            actual = fixed_batch_case_loss(outputs, labels, self.counts)
            terms = []
            for s in range(15):
                for i, label in enumerate(labels.tolist()):
                    if arm == 'direct':
                        logits = outputs['direct_logits'][s, i].tolist(); maximum = max(logits)
                        ce = math.log(sum(math.exp(v-maximum) for v in logits))+maximum-logits[label]
                        term = ce*number/(5*self.counts[label])
                    else:
                        logit = float(outputs['binary_logits'][s, i]); truth = int(label > 0)
                        bce = max(logit, 0)-logit*truth+math.log1p(math.exp(-abs(logit)))
                        broad = self.counts[0] if label == 0 else sum(self.counts[1:])
                        term = bce*number/(2*broad)
                        if label:
                            logits = outputs['malignant_logits'][s, i].tolist(); maximum = max(logits)
                            ce = math.log(sum(math.exp(v-maximum) for v in logits))+maximum-logits[label-1]
                            term += ce*number/(4*self.counts[label])
                        term *= .5
                    terms.append(term/15/4)
            self.assertAlmostEqual(float(actual), sum(terms), places=6)

    def test_last_batch_keeps_denominator_four_and_microbatch_accumulation_matches(self):
        for arm in ('direct', 'dual'):
            model = self.model('mean', arm); labels = torch.tensor([0, 3])
            whole = all_subset_outputs(model, self.x[:2], self.mask[:2])
            batch_loss = fixed_batch_case_loss(whole, labels, self.counts)
            summed = sum(fixed_batch_case_loss({k:v[:, i:i+1] for k,v in whole.items()}, labels[i:i+1], self.counts) for i in range(2))
            torch.testing.assert_close(batch_loss, summed)
            twice = {k:torch.cat((v, v), 1) for k,v in whole.items()}
            torch.testing.assert_close(2*batch_loss, fixed_batch_case_loss(twice, labels.repeat(2), self.counts))

    def test_benign_batch_has_no_expert_gradients_and_malignant_batch_does(self):
        model = self.model('attention', 'dual').train()
        for label in (0, 3):
            model.zero_grad(set_to_none=True)
            outputs = all_subset_outputs(model, self.x[:1], self.mask[:1])
            fixed_batch_case_loss(outputs, torch.tensor([label]), self.counts).backward()
            for parameter in model.heads['malignant'].parameters():
                if label == 0: self.assertIsNone(parameter.grad)
                else: self.assertGreater(float(parameter.grad.abs().max()), 0)
            self.assertGreater(float(model.heads['binary'].weight.grad.abs().max()), 0)
            self.assertGreater(float(model.project[0].weight.grad.abs().max()), 0)

    def test_actual_gradients_exclude_padding_and_one_small_update_reduces_case_loss(self):
        model = CaseMIL(7, 6, 3, 0., 'attention', 'dual').train()
        x = self.x.clone().requires_grad_(True); labels = torch.tensor([0, 2, 4])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = fixed_batch_case_loss(all_subset_outputs(model, x, self.mask), labels, self.counts)
        before.backward()
        self.assertEqual(float(x.grad[~self.mask].abs().max()), 0)
        self.assertGreater(float(x.grad[self.mask].abs().max()), 0)
        for module in (model.field_value, model.field_gate, model.field_score):
            self.assertGreater(float(module.weight.grad.abs().max()), 0)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
        optimizer.step()
        after = fixed_batch_case_loss(all_subset_outputs(model, x, self.mask), labels, self.counts)
        self.assertLess(float(after), float(before))

    def test_extreme_logits_are_finite_and_save_reload_preserves_all_outputs(self):
        model = self.model('attention', 'dual')
        with torch.no_grad():
            model.heads['binary'].bias.fill_(10000)
            model.heads['malignant'].bias.copy_(torch.tensor([10000., -10000., 5000., -5000.]))
        outputs = all_subset_outputs(model, self.x, self.mask)
        self.assertTrue(torch.isfinite(outputs['joint_log_probabilities']).all())
        self.assertTrue(torch.isfinite(fixed_batch_case_loss(outputs, torch.tensor([0, 2, 4]), self.counts)))
        saved = io.BytesIO(); torch.save(model.state_dict(), saved); saved.seek(0)
        loaded = self.model('attention', 'dual'); loaded.load_state_dict(torch.load(saved, weights_only=True))
        for name, value in outputs.items():
            torch.testing.assert_close(value, all_subset_outputs(loaded, self.x, self.mask)[name], rtol=0, atol=0)

    def test_invalid_actual_fields_roles_and_empty_subsets_are_rejected(self):
        model = self.model(); bad = self.x.clone(); bad[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError): model(bad, self.mask)
        missing = self.mask.clone(); missing[:, 1] = False
        with self.assertRaises(ValueError): all_subset_outputs(model, self.x, missing)
        with self.assertRaises(ValueError): model(self.x, missing, (1,))
        with self.assertRaises(ValueError): model(self.x, self.mask, (0, 0))
        with self.assertRaises(ValueError): model(self.x, torch.zeros_like(self.mask))
        outputs = all_subset_outputs(model, self.x, self.mask)
        for labels, counts, denominator in [(torch.tensor([0., 2., 4.]), self.counts, 4),
                                             (torch.tensor([0, 2, 4]), [11, 25, 0, 5, 3], 4),
                                             (torch.tensor([0, 2, 4]), self.counts, 2)]:
            with self.assertRaises(ValueError): fixed_batch_case_loss(outputs, labels, counts, denominator)


if __name__ == '__main__':
    unittest.main()

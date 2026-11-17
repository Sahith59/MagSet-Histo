"""Independent metric, pooling and paired uncertainty checks; no model execution."""
import importlib.util
import json
import math
import tempfile
import csv
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import f1_score

spec = importlib.util.spec_from_file_location('matched_collector', Path(__file__).resolve().parents[1]/'scripts/collect_matched_three_way.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MatchedCollectorMathTests(unittest.TestCase):
    def test_field_then_equal_view_pooling_and_threshold_tie(self):
        # More 40x fields must not make 40x outweigh 100x in a two-view mask.
        joint = np.array([[1., 0, 0, 0, 0], [1., 0, 0, 0, 0],
                          [0., 1, 0, 0, 0], [0., 0, 1, 0, 0], [0., 0, 0, 1, 0]])
        owners = np.array([7]*5)
        mags = np.array([40, 40, 100, 200, 400])
        views = m.field_views(joint, owners, mags, np.array([7]))
        pooled, pred, malignant = m.pool_subsets(views)
        mask = m.SUBSETS.index((0, 1))
        np.testing.assert_array_equal(pooled[0, mask], [.5, .5, 0, 0, 0])
        self.assertEqual(malignant[0, mask], .5)
        self.assertEqual(pred[0, mask], 1)
        self.assertEqual(pred[0, m.SUBSETS.index((0,))], 0)
        self.assertEqual(pred.shape, (1, 15))

    def test_pooling_field_permutation_and_inactive_view_invariance(self):
        rng = np.random.RandomState(9)
        joint = rng.dirichlet(np.ones(5), size=16)
        owners = np.repeat(np.array([2, 8]), 8)
        mags = np.tile(np.repeat(np.array(m.MAGNIFICATIONS), 2), 2)
        views = m.field_views(joint, owners, mags, np.array([2, 8]))
        permutation = rng.permutation(16)
        changed = m.field_views(joint[permutation], owners[permutation], mags[permutation], np.array([2, 8]))
        np.testing.assert_allclose(views, changed, rtol=0, atol=1e-15)
        original = m.pool_subsets(views)
        changed = views.copy(); changed[:, 1:] = np.array([1., 0, 0, 0, 0])
        perturbed = m.pool_subsets(changed)
        column = m.SUBSETS.index((0,))
        for left, right in zip(original, perturbed):
            np.testing.assert_array_equal(left[:, column], right[:, column])

    def test_pooling_rejects_incomplete_membership_or_invalid_joint_probabilities(self):
        joint = np.tile(np.array([.2]*5), (4, 1))
        owners = np.ones(4, dtype=int)
        mags = np.array(m.MAGNIFICATIONS)
        for kind in ('missing_view', 'extra_case', 'nan', 'unnormalized', 'integer'):
            with self.subTest(kind=kind):
                a, b, c = joint.copy(), owners.copy(), mags.copy()
                if kind == 'missing_view': c[-1] = 40
                if kind == 'extra_case': b[-1] = 9
                if kind == 'nan': a[0, 0] = np.nan
                if kind == 'unnormalized': a[0, 0] = .1
                if kind == 'integer': a = a.astype(int)
                with self.assertRaises(ValueError): m.field_views(a, b, c, np.array([1]))

    def test_explicit_five_class_metrics_and_conditional_denominators(self):
        truth = np.repeat(np.arange(5), 2)
        pred = np.array([0, 1, 1, 0, 2, 3, 3, 4, 4, 4])
        summary = m.case_subset_summary(truth, np.repeat(pred[:, None], 15, axis=1))
        self.assertAlmostEqual(summary['mean_subset_macro_f1'], f1_score(truth, pred, average='macro', labels=range(5)))
        self.assertAlmostEqual(summary['mean_subset_accuracy'], .6)
        all_four = summary['all_four']
        self.assertEqual(all_four['five_class']['class_support'], [2]*5)
        self.assertEqual(all_four['binary']['correct'], 8)
        self.assertEqual(all_four['end_to_end_malignant'], {'correct': 5., 'denominator': 8., 'recall': 5/8})
        self.assertEqual(all_four['conditional_malignant'], {'correct': 5., 'denominator': 7., 'accuracy': 5/7})
        self.assertEqual(summary['case_subset_events'], 150)
        # Fixed five labels also apply to sparse bootstrap draws.
        sparse = m.classification(np.array([0, 0]), np.array([0, 0]), 5)
        self.assertEqual(sparse['macro_f1'], .2)

    def test_paired_group_bootstrap_keeps_models_masks_and_case_groups_together(self):
        truth = np.arange(5)
        good = np.repeat(truth[:, None], 15, axis=1)
        wrong = (good+1) % 5
        predictions = {'cnn': good, 'midnight_p': good.copy(), 'swin': wrong}
        groups = ['cluster-a', 'cluster-a', 'cluster-b', 'cluster-c', 'cluster-d']
        result = m.paired_group_bootstrap(truth, predictions, groups, seed=812, replicates=50)
        self.assertEqual(result, m.paired_group_bootstrap(truth, predictions, groups, seed=812, replicates=50))
        self.assertEqual(result['groups'], 4)
        self.assertEqual(result['cases'], 5)
        self.assertGreater(result['draws_missing_any_truth_class'], 0)
        pair = result['pairwise']['cnn_minus_midnight_p']
        self.assertEqual(pair['mean_subset_macro_f1'], {'lower': 0., 'upper': 0.})
        self.assertEqual(pair['mean_subset_accuracy'], {'lower': 0., 'upper': 0.})
        pair = result['pairwise']['cnn_minus_swin']
        self.assertEqual(pair['mean_subset_accuracy'], {'lower': 1., 'upper': 1.})
        self.assertGreater(pair['mean_subset_macro_f1']['lower'], 0.)
        m.assert_public_safe(result)


    def test_saved_record_replay_matches_separate_producer_and_rejects_tampering(self):
        from magset import matched_data
        cases = [{'case_id': 'synthetic%d' % y, 'truth5': y, 'truth8': original, 'group': 'g%d' % y}
                 for y, original in enumerate((0, 1, 3, 4, 5))]
        fields = [{'index': i, 'case_index': i//4, 'view': i % 4} for i in range(20)]
        audit = {'cases': cases, 'fields': fields, 'roles': {'test': list(range(5))}}
        p5 = np.full((20, 5), .05)
        for index in range(20): p5[index, index//4] = .8
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            matched_data.save_predictions(folder/'fields.npz', np.arange(20), p5)
            records = matched_data.case_records(fields, cases, list(range(5)), np.arange(20), p5, 'test')
            matched_data.save_records(folder/'records.jsonl', records)
            summary = matched_data.summary(records)
            actual, pred = m.verify_prediction_set(folder, '', audit, 'test', summary)
            self.assertEqual(actual['mean_subset_macro_f1'], 1.)
            self.assertEqual(pred.shape, (5, 15))
            with self.assertRaisesRegex(ValueError, 'independent summary'):
                m.verify_prediction_set(folder, '', audit, 'test', {**summary, 'mean_subset_macro_f1': .9})
            records[0]['truth5'] = 1
            matched_data.save_records(folder/'records.jsonl', records)
            with self.assertRaisesRegex(ValueError, 'independent field-to-case'):
                m.verify_prediction_set(folder, '', audit, 'test', summary)

    def test_gate_binds_all_three_unchanged_validation_selections(self):
        from magset.data import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = {'midnight_p': {}, 'cnn': {}, 'swin': {}}
            for model in models:
                folder = root/'models'/model; folder.mkdir(parents=True)
                (folder/'selection.json').write_text(json.dumps({'model': model}))
            context = {'code_sha': 'a'*40, 'config_sha256': 'b'*64}
            gate = {'status': 'sealed all three validation selections', **context,
                    'input_audit_sha256': 'c'*64, 'test_predictions': 0,
                    'selections': {model: file_hash(root/'models'/model/'selection.json') for model in models}}
            (root/'test-gate.json').write_text(json.dumps(gate))
            args = SimpleNamespace(root=root)
            self.assertEqual(m.verify_gate(args, context, 'c'*64, models), file_hash(root/'test-gate.json'))
            (root/'models'/'cnn'/'selection.json').write_text('changed after gate')
            with self.assertRaisesRegex(ValueError, 'selection hash gate'):
                m.verify_gate(args, context, 'c'*64, models)

    def test_history_requires_initial_rate_fixed_schedule_and_full_budget(self):
        initial = 1e-4
        epochs = [{'epoch': i+1, 'learning_rate': initial*(.01+.99*.5*(1+math.cos(math.pi*i/39))),
                   'loss': 1., 'binary_loss': .5, 'malignant_loss': .5, 'optimizer_updates': 289*(i+1)}
                  for i in range(40)]
        history = {'epochs': epochs, 'candidate_epochs': [10, 20, 30, 40], 'field_batches_per_epoch': 289}
        config = {'training': {'learning_rates': {'cnn': initial}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'history.json'; path.write_text(json.dumps(history))
            with path.with_name('progress.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(epochs[0])); writer.writeheader(); writer.writerows(epochs)
            self.assertEqual(m.verify_history(path, config, 'cnn')['epochs'], 40)
            epochs[5]['optimizer_updates'] -= 1; path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'optimizer updates'): m.verify_history(path, config, 'cnn')
            epochs[5]['optimizer_updates'] += 1
            epochs[0]['learning_rate'] = 2e-4; path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'learning-rate'): m.verify_history(path, config, 'cnn')

    def test_labels_only_fold_replay_matches_producer_and_keeps_groups_disjoint(self):
        from magset import matched_data
        cases = [{'case_id': 'case%02d' % index, 'truth5': index//10, 'group': 'group%02d' % index}
                 for index in range(50)]
        roles, folds, choice = m.reconstruct_roles(cases, require_version=False)
        producer_roles, producer_folds, producer_choice = matched_data.split_cases(cases, require_version=False)
        np.testing.assert_array_equal(folds, producer_folds)
        self.assertEqual(choice, producer_choice)
        for role in roles:
            np.testing.assert_array_equal(roles[role], producer_roles[role])
            self.assertEqual(set(cases[i]['truth5'] for i in roles[role]), set(range(5)))
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            self.assertFalse({cases[i]['group'] for i in roles[left]} & {cases[i]['group'] for i in roles[right]})
        self.assertEqual(set().union(*(set(indices) for indices in roles.values())), set(range(50)))

    def test_validation_heads_must_compose_exact_saved_joint_probabilities(self):
        audit = {'fields': [{'index': i, 'case_index': i//2} for i in range(4)], 'roles': {'validation': [1]}}
        indices = np.array([2, 3], dtype=np.int64)
        binary = np.array([[.5, .5], [.2, .8]])
        malignant = np.array([[.4, .3, .2, .1], [.1, .2, .3, .4]])
        joint = np.column_stack((binary[:, 0], binary[:, 1, None]*malignant))
        joint /= joint.sum(1, keepdims=True)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez(folder/'validation-fields.npz', field_indices=indices, joint_probabilities5=joint)
            np.savez(folder/'validation-heads.npz', field_indices=indices,
                     binary_probabilities2=binary, malignant_probabilities4=malignant)
            m.verify_head_archive(folder, audit)
            malignant[0] = malignant[0, ::-1]
            np.savez(folder/'validation-heads.npz', field_indices=indices,
                     binary_probabilities2=binary, malignant_probabilities4=malignant)
            with self.assertRaisesRegex(ValueError, 'joint composition'): m.verify_head_archive(folder, audit)

    def test_cast_float32_heads_allow_roundoff_but_joint_probabilities_remain_strict(self):
        audit = {'fields': [{'index': i, 'case_index': 1} for i in range(2)], 'roles': {'validation': [1]}}
        indices = np.array([0, 1], dtype=np.int64)
        binary = np.array([[.1, .9], [.2, .8]], dtype=np.float32).astype(np.float64)
        malignant = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]], dtype=np.float32).astype(np.float64)
        self.assertGreater(abs(binary[0].sum()-1.), 1e-10)
        self.assertGreater(abs(malignant[0].sum()-1.), 1e-10)
        joint = np.column_stack((binary[:, 0], binary[:, 1, None]*malignant))
        joint /= joint.sum(1, keepdims=True)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez(folder/'validation-fields.npz', field_indices=indices, joint_probabilities5=joint)
            np.savez(folder/'validation-heads.npz', field_indices=indices,
                     binary_probabilities2=binary, malignant_probabilities4=malignant)
            m.verify_head_archive(folder, audit)
            binary[0, 0] += 1e-3
            np.savez(folder/'validation-heads.npz', field_indices=indices,
                     binary_probabilities2=binary, malignant_probabilities4=malignant)
            with self.assertRaisesRegex(ValueError, 'normalized probabilities'): m.verify_head_archive(folder, audit)
        joint[0, 0] += 1e-8
        with self.assertRaisesRegex(ValueError, 'normalized probabilities'):
            m.probabilities(joint, (2, 5), 'canonical joint')

    def test_artifact_hash_tamper_and_incomplete_report_never_publish_metrics(self):
        from magset.data import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); item = root/'artifact'; item.write_text('original')
            digest = file_hash(item)
            m.checked_hashes(root, {'artifact': digest}, required=('artifact',))
            item.write_text('tampered')
            with self.assertRaises(ValueError): m.checked_hashes(root, {'artifact': digest}, required=('artifact',))
            with self.assertRaises(ValueError): m.checked_hashes(root, {'../artifact': digest})
            args = SimpleNamespace(output=root/'checked')
            with patch.object(m, 'source_context', side_effect=ValueError('/private/path should stay private')):
                result = m.collect(args)
            self.assertNotIn('models', result)
            self.assertIn('failed or incomplete', result['status'])
            self.assertNotIn('/private/path', (args.output/'result.json').read_text())
            self.assertIn('/private/path', (args.output/'private_review.json').read_text())
            with self.assertRaisesRegex(ValueError, 'existing'): m.collect(args)


if __name__ == '__main__':
    unittest.main()

"""No-TensorFlow checks of independent Swin split/metric/artifact acceptance."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

spec = importlib.util.spec_from_file_location('swin_collector', Path(__file__).resolve().parents[1]/'scripts/collect_supplied_swin.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prediction_fixture():
    truth = np.repeat(np.arange(5), 2)
    pred = np.array([0, 1, 1, 0, 2, 3, 3, 4, 4, 4])
    pm = np.array([.2, .8, .8, .3, .7, .6, .9, .55, .8, .9])
    binary = np.column_stack((1-pm, pm))
    subtype = np.full((len(truth), 4), .05)
    subtype[np.arange(len(truth)), np.maximum(pred-1, 0)] = .85
    table = pd.DataFrame({'patient': ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                          'mag': ['40', '100']*5})
    arrays = {'test_indices': np.arange(len(truth)), 'y5_true': truth, 'y5_pred': pred,
              'p_mal': pm, 'binary_output': binary, 'subtype_output': subtype}
    split = {'test_indices': np.arange(len(truth)), 'y5': truth.copy()}
    return arrays, split, table


def write_progress(path, history):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', *sorted(history)])
        writer.writeheader()
        for epoch in range(len(history['loss'])):
            writer.writerow({'epoch': epoch, **{k: values[epoch] for k, values in history.items()}})


class SwinCollectorTests(unittest.TestCase):
    def test_metrics_have_explicit_five_labels_and_route_denominators(self):
        arrays, split, table = prediction_fixture()
        scalar, details, per_mag, voting = m.prediction_metrics(arrays, split, table)
        expected = {'binary_acc': .8, 'malignant_recall': 7/8, 'benign_recall': .5,
                    'mal_subtype_recall': 5/8, 'mal_subtype_macro_recall': 5/8,
                    'subtype_acc_given_malignant': 5/7, 'five_class_acc': .6,
                    'five_macro_f1': 89/150, 'five_cohen_kappa': .5}
        for key, value in expected.items():
            self.assertAlmostEqual(scalar[key], value, places=14)
        truth, pred = arrays['y5_true'], arrays['y5_pred']
        self.assertAlmostEqual(scalar['five_macro_f1'], f1_score(truth, pred, labels=range(5), average='macro', zero_division=0))
        self.assertAlmostEqual(scalar['five_cohen_kappa'], cohen_kappa_score(truth, pred))
        self.assertAlmostEqual(scalar['binary_auc'], roc_auc_score(truth > 0, arrays['p_mal']))
        self.assertEqual(details['five_class']['class_support'], [2]*5)
        self.assertEqual(details['five_class']['class_correct'], [1, 1, 1, 1, 2])
        self.assertEqual(details['five_class']['correct'], 6)
        self.assertEqual(details['end_to_end_malignant']['records'], 8)
        self.assertEqual(details['conditional_malignant_subtype']['records'], 7)
        self.assertEqual(set(per_mag), {'40', '100'})
        self.assertEqual(voting['groups'], 5)
        self.assertEqual(voting['bare_ids'], 4)
        self.assertEqual(voting['metrics']['five_class']['accuracy'], 1.0)

    def test_vote_ties_follow_first_seen_value_counts_and_group_true_target(self):
        arrays, split, table = prediction_fixture()
        # The tie [2,1] must vote 2, not the smallest numerical label.
        arrays['y5_pred'][:2] = [2, 1]
        arrays['p_mal'][:2] = [.7, .8]
        arrays['binary_output'][:2] = [[.3, .7], [.2, .8]]
        arrays['subtype_output'][:2] = [[.05, .85, .05, .05], [.85, .05, .05, .05]]
        _, _, _, voting = m.prediction_metrics(arrays, split, table)
        self.assertEqual(voting['metrics']['five_class']['confusion_matrix'][0], [0, 0, 1, 0, 0])
        self.assertEqual(voting['groups'], 5)  # one bare ID has two true target groups

    def test_probability_threshold_half_routes_malignant_even_with_argmax_binary_zero(self):
        arrays, split, table = prediction_fixture()
        arrays['p_mal'][0] = .5
        arrays['binary_output'][0] = [.5, .5]
        arrays['y5_pred'][0] = 1
        scalar, _, _, _ = m.prediction_metrics(arrays, split, table)
        self.assertEqual(scalar['benign_recall'], 0.0)

    def test_fail_closed_on_changed_prediction_order_truth_route_probability_or_schema(self):
        for kind in ('order', 'truth', 'route', 'nan', 'normalization', 'p_mal', 'shape', 'extra', 'integer_scores'):
            with self.subTest(kind=kind):
                arrays, split, table = prediction_fixture()
                if kind == 'order': arrays['test_indices'][[0, 1]] = arrays['test_indices'][[1, 0]]
                if kind == 'truth': arrays['y5_true'][0] = 1
                if kind == 'route': arrays['y5_pred'][0] = 4
                if kind == 'nan': arrays['subtype_output'][0, 0] = np.nan
                if kind == 'normalization': arrays['binary_output'][0] = [.2, .2]
                if kind == 'p_mal': arrays['p_mal'][0] = .25
                if kind == 'shape': arrays['subtype_output'] = arrays['subtype_output'][:-1]
                if kind == 'extra': arrays['unexpected'] = np.array([1])
                if kind == 'integer_scores': arrays['binary_output'] = arrays['binary_output'].astype(int)
                with self.assertRaises(ValueError): m.prediction_metrics(arrays, split, table)

    def test_missing_classes_still_penalize_five_class_macro_f1(self):
        scalar, details = m.metric_replay(np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(scalar['five_macro_f1'], .2)
        self.assertIsNone(scalar['five_cohen_kappa'])
        self.assertEqual(scalar['subtype_acc_given_malignant'], 0.0)
        self.assertIsNone(scalar['mal_subtype_recall'])
        self.assertEqual(details['five_class']['class_support'], [2, 0, 0, 0, 0])

    def test_split_rarity_keeps_duplicate_subtype_bare_id_together(self):
        rows = []
        # Enough patients for the registered quota draws, with two fitting IDs
        # beyond all quota assignments in each subtype.
        for sub in sorted(m.TEST_N):
            n = m.TEST_N[sub]+m.VAL_N[sub]+2
            for index in range(n):
                rows.append({'patient': f'{sub}{index:02}', 'subtype': sub})
        # DC patient DC00 also has LC images, and must be primarily LC
        # because there are fewer LC patients than DC patients.
        rows.append({'patient': 'DC00', 'subtype': 'LC'})
        table = pd.DataFrame(rows)
        split, primary = m.reconstruct_split(table)
        self.assertEqual(primary['DC00'], 'LC')
        self.assertEqual(set().union(*(set(v) for v in split.values())), set(range(len(table))))
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            self.assertFalse(set(table.iloc[split[a]].patient) & set(table.iloc[split[b]].patient))
        for role, quota in (('test', m.TEST_N), ('val', m.VAL_N)):
            patients = set(table.iloc[split[role]].patient)
            self.assertEqual({sub: sum(primary[p] == sub for p in patients) for sub in quota}, quota)
        again, _ = m.reconstruct_split(table)
        for role in split: np.testing.assert_array_equal(split[role], again[role])
        changed, _ = m.reconstruct_split(table, seed=1)
        self.assertFalse(np.array_equal(split['test'], changed['test']))

    def test_conservative_overlap_report_does_not_claim_bare_id_overlap(self):
        table = pd.DataFrame({'patient': ['a', 'b', 'c'], 'provider_key': ['p', 'q', 'r'],
                              'conservative_group': ['stem', 'other', 'stem'], 'split': list(m.ROLES)})
        public, private = m.overlap_counts(table)
        self.assertEqual(public['conservative_group']['overlap_keys']['train_test'], 1)
        self.assertEqual(public['patient']['overlap_keys']['train_test'], 0)
        self.assertEqual(private['conservative_group']['train_test'], ['stem'])
        m.assert_public_safe(public)

    def test_artifact_byte_tampering_missing_outputs_and_path_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = {}
            for name in m.REQUIRED_ARTIFACTS:
                (root/name).write_bytes(b'synthetic artifact')
                hashes[name] = m.file_hash(root/name)
            self.assertEqual(m.verify_artifacts(root, {'artifact_sha256': hashes}), hashes)
            (root/'predictions.npz').write_bytes(b'changed artifact')
            with self.assertRaisesRegex(ValueError, 'run artifact'): m.verify_artifacts(root, {'artifact_sha256': hashes})
            hashes['predictions.npz'] = m.file_hash(root/'predictions.npz')
            missing = hashes.copy(); del missing['history.json']
            with self.assertRaisesRegex(ValueError, 'lacks required'): m.verify_artifacts(root, {'artifact_sha256': missing})
            with self.assertRaisesRegex(ValueError, 'Invalid artifact'): m.verify_artifacts(root, {'artifact_sha256': {**hashes, '../secret': 'a'*64}})

    def test_history_selects_first_max_and_checks_original_score_budget_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'progress.csv'
            history = {'loss': [1.0]*12, 'val_score': [40., 60., 60.]+[50.]*9,
                       'val_binary_acc': [50., 70., 70.]+[60.]*9,
                       'val_mal_subtype_recall': [30., 50., 50.]+[40.]*9}
            metrics = {'epochs_completed': 12, 'selected_epoch_first_max_validation_score': 2,
                       'validation_max_score': 60.}
            config = {'training': {'maximum_epochs': 40, 'early_stopping_patience': 10}}
            write_progress(path, history)
            self.assertEqual(m.verify_history(history, metrics, config, path), (12, 2))
            with self.assertRaisesRegex(ValueError, 'selected epoch'):
                m.verify_history(history, {**metrics, 'selected_epoch_first_max_validation_score': 3}, config, path)
            with self.assertRaisesRegex(ValueError, 'maximum validation'):
                m.verify_history(history, {**metrics, 'validation_max_score': 61.}, config, path)
            bad = {**history, 'val_score': [50.]*12}
            with self.assertRaisesRegex(ValueError, 'formula'): m.verify_history(bad, metrics, config, path)
            bad = {key: value[:-1] for key, value in history.items()}
            with self.assertRaisesRegex(ValueError, 'early-stop'): m.verify_history(bad, metrics, config, path)
            path.write_text(path.read_text().replace('0,', '99,', 1))
            with self.assertRaisesRegex(ValueError, 'progress'): m.verify_history(history, metrics, config, path)

    def test_failure_report_has_no_accepted_metrics_or_private_path_message(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp)/'checked')
            with patch.object(m, 'source_context', side_effect=ValueError('/private/path/SOB_M_DC-14-1234 should stay private')):
                result = m.collect(args)
            self.assertIn('failed or incomplete', result['status'])
            self.assertNotIn('metrics', result)
            self.assertNotIn('original_code_metrics', result['run'])
            public_text = (args.output/'result.json').read_text()+(args.output/'summary.md').read_text()
            self.assertNotIn('/private/path', public_text)
            self.assertNotIn('SOB_M', public_text)
            self.assertIn('No accuracy', public_text)
            self.assertIn('/private/path', (args.output/'private_review.json').read_text())
            self.assertEqual(result['private_review_sha256'], m.file_hash(args.output/'private_review.json'))
            with self.assertRaisesRegex(ValueError, 'existing'): m.collect(args)

    def test_pretrained_load_rejects_uninitialized_backbone_and_allows_only_proven_indices(self):
        proof = {'checked_buffers': 12, 'window_size': [7, 7], 'shape': [49, 49],
                 'maximum_abs_error': 0, 'all_nontrainable': True}
        clean = {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': [],
                 'unexpected_keys': ['classifier'],
                 'deterministic_relative_position_index_check': proof}
        exact = [f'swin/encoder/layers.{stage}/blocks.{block}/attention/self/relative_position_index:0'
                 for stage, depth in enumerate((2, 2, 6, 2)) for block in range(depth)]
        clean['deterministic_relative_position_index_sha256'] = {key: '146d178a603c3a1c98a8100a70fe396369b281782f599312dde58016be6f7db2' for key in exact}
        self.assertEqual(m.verify_pretrained_load(clean)['unused_classifier_keys'], 1)
        native = {**clean, 'missing_keys': exact}
        accepted = m.verify_pretrained_load(native)
        self.assertEqual(accepted['missing_keys'], 12)
        self.assertEqual(accepted['unresolved_missing_keys'], 0)
        for missing in (['backbone learned weight'], exact[:-1], exact+[exact[0]],
                        exact+['swin/encoder/layers.0/blocks.0/attention/self/relative_position_bias_table:0'],
                        [key.replace('layers.0/', 'layers.4/') for key in exact]):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'loading failed'):
                m.verify_pretrained_load({**clean, 'missing_keys': missing})
        for key in ('mismatched_keys', 'error_msgs'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'loading failed'):
                m.verify_pretrained_load({**native, key: ['backbone problem']})
        for changed in ({'checked_buffers': 11}, {'window_size': [8, 8]}, {'shape': [48, 48]},
                        {'maximum_abs_error': 1}, {'all_nontrainable': False}):
            with self.subTest(proof=changed), self.assertRaisesRegex(ValueError, 'index proof'):
                m.verify_pretrained_load({**native, 'deterministic_relative_position_index_check': {**proof, **changed}})
        hashes = clean['deterministic_relative_position_index_sha256']
        for bad_hashes in ({key: 'a'*64 for key in exact}, {key: hashes[key] for key in exact[:-1]},
                           {**hashes, 'extra': next(iter(hashes.values()))}):
            with self.subTest(hashes=bad_hashes), self.assertRaisesRegex(ValueError, 'buffer hashes'):
                m.verify_pretrained_load({**native, 'deterministic_relative_position_index_sha256': bad_hashes})
        without_proof = {key: value for key, value in native.items() if key != 'deterministic_relative_position_index_check'}
        with self.assertRaisesRegex(ValueError, 'index proof'): m.verify_pretrained_load(without_proof)
        for unexpected in (['swin/embeddings/kernel:0'], ['notclassifier/kernel:0'], None):
            with self.subTest(unexpected=unexpected), self.assertRaisesRegex(ValueError, 'non-classifier'):
                m.verify_pretrained_load({**native, 'unexpected_keys': unexpected})

    def test_cli_contract_and_no_tensorflow_import(self):
        args = m.parser().parse_args(['--run', 'r', '--audit', 'a', '--config', 'c', '--code-sha', 'a'*40, '--output', 'o'])
        self.assertEqual(args.run, Path('r'))
        self.assertEqual(args.code_sha, 'a'*40)
        self.assertNotIn('tensorflow', sys.modules)
        self.assertNotIn('tf_keras', sys.modules)


if __name__ == '__main__':
    unittest.main()

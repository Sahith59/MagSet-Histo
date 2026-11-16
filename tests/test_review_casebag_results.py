"""Independent review guards using tiny saved arrays; no models or fitting."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / 'scripts/review_casebag_results.py'
_SPEC = importlib.util.spec_from_file_location('independent_casebag_review', _PATH)
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def archive_fixture(directory, arm='dual'):
    dimension = 3
    cases = [{'case_id': 'synthetic-' + str(y), 'label': y,
              'indices': {v: ['synthetic-field-' + str(y) + '-' + str(v)] for v in review.VIEWS}}
             for y in range(8)]
    features = np.random.default_rng(71).normal(size=(8, 4, dimension))
    descriptors = []
    for subset in review.SUBSETS:
        ix = [review.VIEWS.index(v) for v in subset]
        flags = np.zeros((8, 4)); flags[:, ix] = 1
        descriptors.append(np.concatenate((features[:, ix].mean(1), flags), axis=1).astype(np.float32))
    data = {'case_ids': np.asarray([c['case_id'] for c in cases]),
            'truth8': np.arange(8, dtype=np.int64), 'truth5': np.asarray(review.MAP, dtype=np.int64),
            'field_counts': np.ones((8, 4), dtype=np.int64), 'view_features': features,
            'descriptors': np.stack(descriptors, axis=1),
            'descriptor_subset_masks': np.asarray(review.MASKS, dtype=np.int64),
            'descriptor_magnifications': np.asarray(['+'.join(map(str, s)) for s in review.SUBSETS])}
    for name, count in ({'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}).items():
        data[name + '_coef'] = np.zeros((count, dimension + 4), dtype=np.float32)
        data[name + '_intercept'] = np.zeros(count, dtype=np.float32)
        data[name + '_classes'] = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
    if arm == 'dual':
        data['binary_intercept'][0] = np.log(.6 / .4)
        data['malignant_intercept'] = np.log([.3, .25, .25, .2]).astype(np.float32)
    rows = []
    for i, subset in enumerate(review.SUBSETS):
        values = review.scores(data, data['descriptors'][:, i], arm)
        for j, c in enumerate(cases):
            row = {'role': 'validation', 'case_id': c['case_id'], 'truth': c['label'],
                   'truth5': review.MAP[c['label']], 'magnifications': list(subset)}
            row.update({k: v[j].tolist() for k, v in values.items()}); rows.append(row)
    folder = Path(directory)
    np.savez_compressed(folder / 'fixture-features.npz', **data)
    (folder / 'fixture-predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return cases, data, rows


def hashes(directory):
    p = Path(directory)
    return {'features_sha256': review.file_hash(p / 'fixture-features.npz'),
            'predictions_sha256': review.file_hash(p / 'fixture-predictions.jsonl')}


class IndependentCasebagReviewTests(unittest.TestCase):
    def test_hard_route_uses_binary_group_even_when_benign_probability_wins(self):
        with tempfile.TemporaryDirectory() as d:
            cases, _, rows = archive_fixture(d)
            self.assertEqual(rows[0]['prediction'], 1)
            self.assertEqual(np.argmax(rows[0]['probabilities5']), 0)
            metric, _, _ = review.archive_review(d, 'fixture', cases, 'validation', 'dual', hashes(d), review.Audit(), 3)
            self.assertEqual(metric['all_view']['class_support'], [4, 1, 1, 1, 1])
            self.assertEqual(metric['pooled']['records'], 120)

    def test_direct_and_dual_saved_archives_are_replayed(self):
        for arm in ('direct', 'dual'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as d:
                cases, _, _ = archive_fixture(d, arm)
                audit = review.Audit()
                metric, _, _ = review.archive_review(d, 'fixture', cases, 'validation', arm, hashes(d), audit, 3)
                self.assertEqual(audit.archives, 1)
                self.assertEqual(audit.records, 120)
                self.assertEqual(len(metric['subsets']), 15)

    def test_modified_probabilities_fail_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            cases, _, rows = archive_fixture(d)
            rows[0]['probabilities5'][0] += .05
            p = Path(d) / 'fixture-predictions.jsonl'; p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(ValueError):
                review.archive_review(d, 'fixture', cases, 'validation', 'dual', hashes(d), review.Audit(), 3)

    def test_wrong_role_truth_and_noninteger_label_fail(self):
        for key, value in [('role', 'excluded'), ('truth', 1), ('truth5', 4), ('truth', 0.0)]:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as d:
                cases, _, rows = archive_fixture(d)
                rows[0][key] = value
                (Path(d) / 'fixture-predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.assertRaises(ValueError):
                    review.archive_review(d, 'fixture', cases, 'validation', 'dual', hashes(d), review.Audit(), 3)

    def test_wrong_descriptor_or_head_order_fails(self):
        for change in ('descriptor', 'classes', 'truth'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                cases, data, _ = archive_fixture(d)
                if change == 'descriptor':
                    data['descriptors'][0, 0, 0] += .25
                elif change == 'classes':
                    data['malignant_classes'] = np.asarray([4, 3, 2, 1])
                else:
                    data['truth8'][0] = 1
                np.savez_compressed(Path(d) / 'fixture-features.npz', **data)
                with self.assertRaises(ValueError):
                    review.archive_review(d, 'fixture', cases, 'validation', 'dual', hashes(d), review.Audit(), 3)

    def test_stable_extreme_loss_and_raw_balanced_difference(self):
        lp = np.asarray([[0., -1000., -2000., -3000., -4000.]])
        result = review.classification([4], [0], np.exp(lp), lp, 5)
        self.assertEqual(result['cross_entropy'], 4000)
        self.assertEqual(result['zero_true_probability_records'], 1)
        y = [0, 0, 0, 0, 1, 2, 3, 4]; p = np.tile([.8, .05, .05, .05, .05], (8, 1))
        result = review.classification(y, np.zeros(8, int), p, np.log(p), 5)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['balanced_accuracy'], .2)
        self.assertAlmostEqual(result['macro_f1'], 2 / 15)

    def test_rounded_negative_router_tie_is_malignant(self):
        arrays = {'binary_coef': np.zeros((1, 7), np.float32), 'binary_intercept': np.array([-1e-20], np.float32),
                  'malignant_coef': np.zeros((4, 7), np.float32), 'malignant_intercept': np.zeros(4, np.float32)}
        result = review.scores(arrays, np.zeros((1, 7), np.float32), 'dual')
        self.assertEqual(result['binary_score'][0], .5)
        self.assertEqual(result['prediction'][0], 1)

    def test_binary_ranking_ties_use_average_ranks_and_threshold_precision(self):
        rows = [{'truth5': y, 'binary_score': score} for y, score in [(1, .9), (0, .8), (1, .8), (0, .1)]]
        value = review.binary(rows)
        self.assertEqual(value['roc_auc'], .875)
        self.assertAlmostEqual(value['pr_average_precision'], (1 + 2 / 3) / 2)

    def test_earliest_selection_and_incomplete_candidate_rejection(self):
        candidates = [{'epoch': e, 'validation': {'mean_subset_macro_f1': f}} for e, f in zip(review.EPOCHS, [.3, .7, .7, .6])]
        self.assertEqual(review.select_candidate(candidates)['epoch'], 20)
        with self.assertRaises(ValueError):
            review.select_candidate(candidates[:-1])
        with self.assertRaises(ValueError):
            review.select_candidate(list(reversed(candidates)))

    def test_selection_identity_is_exact_but_independent_metrics_allow_roundoff(self):
        candidates = [
            {'epoch': e, 'validation': {'mean_subset_macro_f1': f,
                                      'binary_summary': {'cross_entropy': .7}},
             **{k: 'a' * 64 for k in ('state_sha256', 'checkpoint_sha256',
                                      'features_sha256', 'predictions_sha256')}}
            for e, f in zip(review.EPOCHS, [.3, .7, .7, .6])]
        saved = copy.deepcopy(candidates[1])
        saved['validation']['binary_summary']['cross_entropy'] += 2.220446049250313e-16
        self.assertEqual(review.verify_selection(saved, candidates, review.Audit())['epoch'], 20)
        for key, value in [('epoch', 35), ('epoch', 20.0), ('state_sha256', 'b' * 64)]:
            changed = copy.deepcopy(saved); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                review.verify_selection(changed, candidates, review.Audit())
        changed = copy.deepcopy(saved)
        changed['validation']['binary_summary']['cross_entropy'] += 1e-5
        with self.assertRaises(ValueError):
            review.verify_selection(changed, candidates, review.Audit())

    def test_partial_extra_and_complete_matrix_file_guards(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                review.matrix_paths(d, review.Audit())
            for arm in review.ARMS:
                for seed in (812, 813):
                    for trial in range(1, 6):
                        p = Path(d) / arm / f'split{seed}' / f'trial{trial}'; p.mkdir(parents=True)
                        for name in review.files_per_run():
                            (p / name).write_bytes(b'')
            self.assertEqual(len(review.matrix_paths(d, review.Audit())), 20)
            (Path(d) / 'outer-predictions.jsonl').write_text('{}')
            with self.assertRaises(ValueError):
                review.matrix_paths(d, review.Audit())

    def test_strict_json_and_public_privacy(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                review.decode(text)
        with self.assertRaises(ValueError):
            review.public_safe({'case_ids': ['synthetic']})
        with self.assertRaises(ValueError):
            review.public_safe({'scope': '/private/path'})
        review.public_safe({'verified_fits': 20, 'status': 'accepted'})

    def test_metric_summary_does_not_replace_mean_f1_with_pooled_f1(self):
        with tempfile.TemporaryDirectory() as d:
            cases, _, rows = archive_fixture(d, 'direct')
            # Different errors in one subset change nonlinear F1 averaging.
            first_subset = rows[:8]
            for row in first_subset:
                row['prediction'] = row['truth5']
            result = review.role_summary(rows, cases, 'direct', 'validation')
            self.assertNotAlmostEqual(result['mean_subset_macro_f1'], result['pooled']['macro_f1'])


if __name__ == '__main__':
    unittest.main()

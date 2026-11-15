"""Synthetic five-category metric checks, with no fitting or encoder execution."""
import copy
import json
import unittest

import numpy as np
from scipy.special import logsumexp

from magset.collect import SUBSETS
from magset.five_metrics import FINAL5, MAP8_TO5, classification_metrics, five_labels, summarize_records


def fixture(decision_rule='dual'):
    records = []
    malignant_labels = (1, 3, 4, 5)
    for original_label in range(8):
        truth = MAP8_TO5[original_label]
        binary_log = np.log([.9, .1] if truth == 0 else [.1, .9])
        malignant_logits = np.zeros(4)
        if truth:
            malignant_logits[malignant_labels.index(original_label)] = 3
        malignant_log = malignant_logits - logsumexp(malignant_logits)
        log_probabilities = np.r_[binary_log[0], binary_log[1] + malignant_log]
        probabilities = np.exp(log_probabilities)
        prediction = int(probabilities.argmax())
        if decision_rule == 'dual':
            prediction = 0 if np.exp(binary_log[1]) < .5 else 1 + int(malignant_log.argmax())
        for subset in SUBSETS:
            records.append({
                'role': 'validation', 'case_id': 'synthetic-case-' + str(original_label),
                'truth': original_label, 'truth5': truth, 'prediction': prediction,
                'magnifications': list(subset), 'probabilities5': probabilities.tolist(),
                'log_probabilities5': log_probabilities.tolist(),
                'binary_score': float(np.exp(binary_log[1])),
                'binary_log_probabilities': binary_log.tolist(),
                'malignant_probabilities4': np.exp(malignant_log).tolist(),
                'malignant_log_probabilities': malignant_log.tolist(),
            })
    return records


class FiveMetricsTests(unittest.TestCase):
    def test_canonical_pooling_mapping(self):
        self.assertEqual(len(FINAL5), 5)
        np.testing.assert_array_equal(five_labels(np.arange(8)), [0, 1, 0, 2, 3, 4, 0, 0])
        for invalid in ([8], [-1], [1.0], [[0, 1]], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                five_labels(invalid)

    def test_explicit_hard_prediction_differs_from_global_argmax(self):
        binary_log = np.log([.4, .6])
        malignant_log = np.log([.3, .25, .25, .2])
        log_probabilities = np.r_[binary_log[0], binary_log[1] + malignant_log]
        probabilities = np.exp(log_probabilities)
        self.assertEqual(int(probabilities.argmax()), 0)
        metric = classification_metrics(np.asarray([1]), np.asarray([1]),
                                        probabilities[None, :], log_probabilities[None, :])
        self.assertEqual(metric['accuracy'], 1)
        self.assertEqual(metric['confusion_matrix'][1][1], 1)
        self.assertAlmostEqual(metric['cross_entropy'], -np.log(.18))
        records = fixture()
        row = records[15]
        row.update(prediction=1, probabilities5=probabilities.tolist(),
                   log_probabilities5=log_probabilities.tolist(), binary_score=.6,
                   binary_log_probabilities=binary_log.tolist(),
                   malignant_probabilities4=np.exp(malignant_log).tolist(),
                   malignant_log_probabilities=malignant_log.tolist())
        self.assertEqual(summarize_records(records, 'dual')['mean_subset_accuracy'], 1)
        with self.assertRaises(ValueError):
            summarize_records(records, 'direct')

    def test_stable_cross_entropy_when_probability_underflows(self):
        logits = np.asarray([[1000., 0., -1000., -2000., -3000.]])
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        metric = classification_metrics(np.asarray([4]), np.asarray([0]),
                                        probabilities, log_probabilities)
        self.assertEqual(metric['cross_entropy'], 4000)
        self.assertEqual(metric['zero_true_probability_records'], 1)
        self.assertEqual(metric['brier'], 2)
        json.dumps(metric, allow_nan=False)

    def test_complete_subset_support_and_diagnostics(self):
        for decision in ('direct', 'dual'):
            result = summarize_records(fixture(decision), decision)
            self.assertEqual(result['distinct_cases'], 8)
            self.assertEqual(result['distinct_class_support'], [4, 1, 1, 1, 1])
            self.assertEqual(result['all_view']['class_support'], [4, 1, 1, 1, 1])
            self.assertEqual(result['pooled']['class_support'], [60, 15, 15, 15, 15])
            self.assertEqual(result['pooled']['records'], 120)
            self.assertEqual(len(result['subsets']), 15)
            self.assertEqual(result['mean_subset_accuracy'], 1)
            self.assertEqual(result['mean_subset_macro_f1'], 1)
            self.assertEqual(result['binary_summary']['full_view']['accuracy'], 1)
            self.assertAlmostEqual(result['binary_summary']['full_view']['cross_entropy'], -np.log(.9))
            self.assertEqual(result['malignant_summary']['distinct_class_support'], [1, 1, 1, 1])
            self.assertEqual(result['malignant_summary']['mean_subset_accuracy'], 1)
            json.dumps(result, allow_nan=False)

    def test_raw_and_balanced_accuracy_differ_with_pooled_benign(self):
        labels = np.asarray([0, 0, 0, 0, 1, 2, 3, 4])
        probabilities = np.tile([.8, .05, .05, .05, .05], (len(labels), 1))
        metric = classification_metrics(labels, np.zeros(len(labels), dtype=int),
                                        probabilities, np.log(probabilities))
        self.assertEqual(metric['accuracy'], .5)
        self.assertEqual(metric['balanced_accuracy'], .2)
        self.assertAlmostEqual(metric['macro_f1'], (2 / 3) / 5)
        balanced = classification_metrics(labels, np.zeros(len(labels), dtype=int),
                                          probabilities, np.log(probabilities),
                                          weights=[.25, .25, .25, .25, 1, 1, 1, 1])
        self.assertEqual(balanced['accuracy'], .2)

    def test_single_role_complete_coverage_and_truth_guards(self):
        records = fixture()
        for changed in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                summarize_records(changed, 'dual')
        changes = [('role', 'test'), ('role', 'train'), ('truth', 1), ('truth5', 4),
                   ('prediction', 2), ('magnifications', [40, 40]),
                   ('binary_score', float('nan')), ('binary_score', 1.1),
                   ('malignant_probabilities4', [.5] * 4),
                   ('binary_log_probabilities', np.log([.1, .9]).tolist()),
                   ('log_probabilities5', [float('-inf')] * 5)]
        for key, value in changes:
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize_records(changed, 'dual')
        changed = copy.deepcopy(records)
        del changed[0]['prediction']
        with self.assertRaises(ValueError):
            summarize_records(changed, 'dual')

    def test_incomplete_final_class_support_rejected(self):
        records = [row for row in fixture() if row['truth5'] != 4]
        with self.assertRaises(ValueError):
            summarize_records(records, 'dual')

    def test_record_order_invariance_and_true_group_diagnostic(self):
        records = fixture()
        binary_log = np.log([.9, .1])
        for row in records[15:30]:
            malignant_log = np.asarray(row['malignant_log_probabilities'])
            log_probabilities = np.r_[binary_log[0], binary_log[1] + malignant_log]
            row.update(binary_score=.1, binary_log_probabilities=binary_log.tolist(),
                       probabilities5=np.exp(log_probabilities).tolist(),
                       log_probabilities5=log_probabilities.tolist(), prediction=0)
        result = summarize_records(records, 'dual')
        reversed_result = summarize_records(list(reversed(records)), 'dual')
        for key in ('accuracy', 'macro_f1', 'balanced_accuracy', 'cross_entropy', 'brier'):
            self.assertAlmostEqual(result['mean_subset_' + key], reversed_result['mean_subset_' + key])
        self.assertEqual(result['mean_subset_accuracy'], 7 / 8)
        self.assertEqual(result['malignant_summary']['mean_subset_accuracy'], 1)
        self.assertEqual(result['binary_summary']['mean_all_subsets']['accuracy'], 7 / 8)

    def test_direct_binary_diagnostic_uses_group_probability(self):
        records = fixture('direct')
        binary_log = np.log([.4, .6])
        malignant_log = np.log([.3, .25, .25, .2])
        log_probabilities = np.r_[binary_log[0], binary_log[1] + malignant_log]
        for row in records[:15]:
            row.update(prediction=0, binary_score=.6, binary_log_probabilities=binary_log.tolist(),
                       malignant_probabilities4=np.exp(malignant_log).tolist(),
                       malignant_log_probabilities=malignant_log.tolist(),
                       probabilities5=np.exp(log_probabilities).tolist(),
                       log_probabilities5=log_probabilities.tolist())
        result = summarize_records(records, 'direct')
        self.assertEqual(result['mean_subset_accuracy'], 1)
        self.assertEqual(result['binary_summary']['mean_all_subsets']['accuracy'], 7 / 8)

    def test_binary_threshold_tie_uses_malignant_branch(self):
        records = fixture()
        binary_log = np.log([.5, .5])
        for row in records[:15]:
            malignant_log = np.asarray(row['malignant_log_probabilities'])
            log_probabilities = np.r_[binary_log[0], binary_log[1] + malignant_log]
            row.update(binary_score=.5, binary_log_probabilities=binary_log.tolist(),
                       probabilities5=np.exp(log_probabilities).tolist(),
                       log_probabilities5=log_probabilities.tolist(), prediction=1)
        result = summarize_records(records, 'dual')
        self.assertEqual(result['all_view']['confusion_matrix'][0][1], 1)
        self.assertEqual(result['binary_summary']['full_view']['confusion_matrix'][0][1], 1)

    def test_invalid_metric_arrays_rejected(self):
        probabilities = np.full((1, 5), .2)
        arguments = (np.asarray([0]), np.asarray([0]), probabilities, np.log(probabilities))
        invalid = [(np.asarray([0.]), *arguments[1:]),
                   (arguments[0], np.asarray([5]), *arguments[2:]),
                   (*arguments[:2], probabilities[:, :4], arguments[3]),
                   (*arguments[:3], np.zeros((1, 5)))]
        for args in invalid:
            with self.assertRaises(ValueError):
                classification_metrics(*args)
        for weights in ([0], [-1], [float('nan')], [float('inf')], [1, 1]):
            with self.assertRaises(ValueError):
                classification_metrics(*arguments, weights=weights)


if __name__ == '__main__':
    unittest.main()

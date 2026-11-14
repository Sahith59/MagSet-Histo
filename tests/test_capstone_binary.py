"""Synthetic binary secondary checks, never model execution or research performance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from magset import capstone_binary as binary
from magset.collect import SUBSETS
from magset.data import CLASS_NAMES


def fixture():
    rows = []; records = []; labels = {}
    for y, name in enumerate(CLASS_NAMES):
        cid = f'SOB_{y}-{y}'
        malignant = y in binary.MALIGNANT
        rows.append({'filename': f'SOB_{y}-image.png', 'case_id_raw': str(y),
                     'subtype': name, 'broad_class': 'malignant' if malignant else 'benign'})
        labels[cid] = int(malignant)
        # Deliberately wrong subtype within the right broad class.
        group = binary.MALIGNANT if malignant else binary.BENIGN
        wrong = group[(group.index(y) + 1) % 4]
        p = [0.] * 8; p[wrong] = 1.
        for subset in SUBSETS:
            records.append({'role': 'validation', 'case_id': cid, 'truth': y,
                            'magnifications': list(subset), 'probabilities': p})
    return rows, records, labels


class CapstoneBinaryTests(unittest.TestCase):
    def test_mapping_and_wrong_subtype_right_binary(self):
        rows, records, labels = fixture()
        self.assertEqual(binary.trusted_binary_labels(rows), labels)
        events = binary.binary_scores(records, labels)
        self.assertEqual(len(records), 120)
        for value in events.values():
            result = binary.binary_metrics(value)
            self.assertEqual(result['accuracy'], 1.)
            self.assertEqual(result['roc_auc'], 1.)
            self.assertEqual(result['pr_average_precision'], 1.)
            self.assertEqual(result['positive_class_brier'], 0.)
            self.assertEqual(result['class_support'], [4, 4])
        for row in records:
            self.assertNotEqual(int(np.argmax(row['probabilities'])), row['truth'])

    def test_fixed_threshold_ties_are_malignant(self):
        result = binary.binary_metrics([(0, .5), (1, .5)])
        self.assertEqual(result['confusion_matrix'], [[0, 1], [0, 1]])
        self.assertEqual(result['sensitivity'], 1.)
        self.assertEqual(result['specificity'], 0.)
        self.assertEqual(result['roc_auc'], .5)
        self.assertEqual(result['pr_average_precision'], .5)
        self.assertEqual(result['positive_class_brier'], .25)

    def test_known_rank_metrics_with_ties(self):
        result = binary.binary_metrics([(1, .9), (0, .8), (1, .8), (0, .1)])
        self.assertAlmostEqual(result['roc_auc'], .875)
        self.assertAlmostEqual(result['pr_average_precision'], (1 + 2 / 3) / 2)
        reverse = binary.binary_metrics([(0, .1), (1, .8), (0, .8), (1, .9)])
        for key in result:
            np.testing.assert_allclose(result[key], reverse[key], rtol=0, atol=1e-15)

    def test_complete_coverage_and_role_truth_guards(self):
        _, records, labels = fixture()
        for changed in [records[:-1], records + [records[0]]]:
            with self.assertRaises(ValueError):
                binary.binary_scores(changed, labels)
        for key, value in [('role', 'test'), ('truth', 1), ('case_id', 'outer'),
                           ('magnifications', [40, 40]), ('probabilities', [float('nan')] * 8),
                           ('probabilities', [.2] * 8)]:
            changed = copy.deepcopy(records); changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                binary.binary_scores(changed, labels)

    def test_manifest_broad_truth_guard(self):
        rows, _, _ = fixture()
        rows[0]['broad_class'] = 'malignant'
        with self.assertRaises(ValueError):
            binary.trusted_binary_labels(rows)

    def test_no_nan_or_undefined_auc(self):
        for bad in [[(0, float('nan')), (1, .5)], [(0, 1.1), (1, .4)],
                    [(0, .3)], [(2, .2), (1, .3)]]:
            with self.assertRaises(ValueError):
                binary.binary_metrics(bad)

    def test_parent_hash_class_order_and_coverage_tamper(self):
        trusted = {'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol'}
        parent = {'verified_runs': 10, 'class_names': CLASS_NAMES, **trusted,
                  'artifacts': [{'seed': s, 'trial': t} for s in [812, 813] for t in range(1, 6)]}
        self.assertEqual(len(binary.parent_entries(parent, trusted)), 10)
        for key, value in [('class_names', list(reversed(CLASS_NAMES))),
                           ('manifest_sha256', 'other'), ('artifacts', parent['artifacts'][:-1]),
                           ('artifacts', parent['artifacts'][:-1] + [parent['artifacts'][0]])]:
            changed = dict(parent); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                binary.parent_entries(changed, trusted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parent.json'; path.write_text(json.dumps(parent))
            digest = binary.file_hash(path)
            self.assertEqual(binary.checked_json(path, digest)['verified_runs'], 10)
            path.write_text('{}')
            with self.assertRaises(ValueError):
                binary.checked_json(path, digest)

    def test_fixed_full_run_summary_is_json_finite(self):
        _, records, labels = fixture()
        result = binary.run_summary(binary.binary_scores(records, labels))
        self.assertEqual(len(result['subsets']), 15)
        self.assertEqual(result['pooled_case_subset_events']['records'], 120)
        self.assertEqual(result['mean_all_subsets']['records'], 8)
        self.assertEqual(result['full_view']['accuracy'], 1.)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()

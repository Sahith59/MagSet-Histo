"""Toy checks of endpoint units and strict saved-score provenance; no model imports."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp

from magset import capstone_bridge as bridge
from magset.data import CLASS_NAMES, MAGNIFICATIONS, case_id


def fixture():
    rows, cases, logits = [], [], []
    for label, name in enumerate(CLASS_NAMES):
        indices = {}
        for m in MAGNIFICATIONS:
            indices[m] = []
            # One wrong case/view has nine fields; others have one.
            for field in range(9 if label == 0 and m == 40 else 1):
                path = f'{label}/{m}/{field}.png'
                row = {'filename': f'SOB_{label}-image.png', 'case_id_raw': str(label),
                       'relative_path': path, 'subtype': name, 'magnification': str(m)}
                rows.append(row); indices[m].append(path)
                z = np.full(8, -3., dtype=np.float64)
                z[1 if label == 0 and m == 40 else label] = 3.
                logits.append(z)
        cases.append({'case_id': case_id(row), 'label': label, 'indices': indices})
    z = np.asarray(logits); lp = z - logsumexp(z, axis=1, keepdims=True)
    fields = {'paths': np.asarray([r['relative_path'] for r in rows]), 'logits': z,
              'log_probabilities': lp, 'probabilities': np.exp(lp)}
    return rows, cases, fields


class CapstoneBridgeTests(unittest.TestCase):
    def test_image_and_balanced_units_differ(self):
        rows, cases, fields = fixture()
        bridge.validate_rows(rows, cases)
        slices = bridge.image_slices(rows, fields)
        self.assertAlmostEqual(slices['40']['raw_image']['accuracy'], 7 / 16)
        self.assertAlmostEqual(slices['40']['class_case_view_balanced']['accuracy'], 7 / 8)
        self.assertAlmostEqual(slices['all_fields']['raw_image']['accuracy'], 31 / 40)
        self.assertAlmostEqual(slices['all_fields']['class_case_view_balanced']['accuracy'], 31 / 32)
        self.assertEqual(slices['40']['raw_image']['class_support'], [9., 1., 1., 1., 1., 1., 1., 1.])
        self.assertEqual(list(slices), bridge.CONFIG['slices'])
        self.assertAlmostEqual(bridge.balanced_weights(rows).sum(), 1)

    def test_equal_cases_within_class_despite_unequal_case_counts(self):
        rows = []
        for label, identifier, count in [(0, 'a', 2), (0, 'b', 1), (1, 'c', 1)]:
            for i in range(count):
                rows.append({'filename': f'SOB_{label}-image.png', 'case_id_raw': identifier,
                             'relative_path': f'{identifier}/{i}.png',
                             'subtype': CLASS_NAMES[label], 'magnification': '40'})
        # Equal class mass, then equal cases within class, then equal fields.
        np.testing.assert_allclose(bridge.balanced_weights(rows), [.125, .125, .25, .5])

    def test_exact_case_bridge_and_tampering(self):
        rows, cases, fields = fixture()
        records = []
        for c in cases:
            ix = [i for i, r in enumerate(rows) if case_id(r) == c['case_id'] and int(r['magnification']) == 40]
            lp = logsumexp(fields['log_probabilities'][ix], axis=0) - np.log(len(ix))
            records.append({'case_id': c['case_id'], 'role': 'validation', 'truth': c['label'],
                            'magnifications': [40], 'probabilities': np.exp(lp).tolist(), 'log_probabilities': lp.tolist()})
        claimed = bridge.metrics([r['probabilities'] for r in records], np.arange(8),
                                 [r['log_probabilities'] for r in records])
        claimed['class_balanced_cross_entropy'] = claimed['cross_entropy']
        value = bridge.case_40_bridge(rows, fields, records, claimed)
        self.assertAlmostEqual(value['metrics']['accuracy'], 7 / 8)
        for key, bad in [('role', 'test'), ('truth', 7), ('probabilities', [0.125] * 8),
                         ('log_probabilities', [-np.log(8)] * 8)]:
            changed = json.loads(json.dumps(records)); changed[0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                bridge.case_40_bridge(rows, fields, changed, claimed)
        with self.assertRaises(ValueError):
            bridge.case_40_bridge(rows, fields, records[:-1], claimed)
        altered = dict(claimed, accuracy=0)
        with self.assertRaises(ValueError):
            bridge.case_40_bridge(rows, fields, records, altered)

    def test_field_path_dtype_and_math_guards(self):
        rows, _, fields = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fields.npz'
            np.savez(path, **fields)
            loaded = bridge.load_fields(path, rows)
            np.testing.assert_array_equal(loaded['probabilities'], fields['probabilities'])
            for key, bad in [('paths', fields['paths'][::-1]),
                             ('paths', np.repeat(fields['paths'][0], len(rows))),
                             ('logits', fields['logits'].astype(np.float32)),
                             ('probabilities', np.full_like(fields['probabilities'], np.nan)),
                             ('log_probabilities', fields['log_probabilities'] + 1)]:
                changed = dict(fields); changed[key] = bad; np.savez(path, **changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bridge.load_fields(path, rows)

    def test_trusted_row_truth_and_coverage_guards(self):
        rows, cases, _ = fixture()
        for changed in [rows[:-1], rows + [rows[0]],
                        [dict(rows[0], subtype=CLASS_NAMES[7])] + rows[1:],
                        [dict(rows[0], case_id_raw='outer')] + rows[1:]]:
            with self.assertRaises(ValueError):
                bridge.validate_rows(changed, cases)

    def test_hash_tamper_fails_before_context_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comparison.json'; path.write_text('{}')
            digest = bridge.check_hash(path, bridge.file_hash(path))
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                bridge.collect('absent', 'absent', path, digest, 'absent')

    def test_stable_underflow_and_no_probability_floor(self):
        z = np.zeros((1, 8), dtype=np.float64); z[0, 0] = -1000
        lp = z - logsumexp(z, axis=1, keepdims=True)
        value = bridge.metrics(np.exp(lp), np.array([0]), lp)
        self.assertGreater(value['cross_entropy'], 1000)
        self.assertEqual(value['zero_true_probability_records'], 1)
        with self.assertRaises(ValueError):
            bridge.metrics(np.exp(lp), np.array([0]), lp + 1)


if __name__ == '__main__':
    unittest.main()

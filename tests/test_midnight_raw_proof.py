"""Small, model-free tests for independent all-field Midnight proof math."""
import hashlib
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from scripts import replay_midnight_raw_features as raw
from scripts import collect_midnight_raw_proof as proof
from magset import matched_data, prior_centered_model


class MidnightRawProofTests(unittest.TestCase):
    def test_independent_png_crops_match_registered_both_geometries(self):
        for height in (460, 456):
            array = np.zeros((height, 700, 3), dtype=np.uint8)
            array[:, :, 0] = np.arange(700, dtype=np.uint16) % 256
            array[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
            image = Image.fromarray(array);buffer = BytesIO();image.save(buffer, format='PNG');data = buffer.getvalue()
            row = {'sha256': hashlib.sha256(data).hexdigest(), 'width': 700, 'height': height,
                   'relative_path': 'breast/benign/SOB_B_A-14-12345-40-001.png', 'subtype_code': 'A', 'view': 0, 'magnification': 40}
            actual = raw.checked_crops(data, row, {'truth5': 0})
            np.testing.assert_array_equal(actual, matched_data.crop_pixels(image))
            with self.assertRaises(ValueError): raw.checked_crops(data+b'changed', row, {'truth5': 0})
            with self.assertRaises(ValueError): raw.checked_crops(data, row, {'truth5': 1})

    def test_crop_fusion_is_exact_float64_mean_then_single_float32_cast(self):
        rng = np.random.RandomState(812)
        values = rng.normal(size=(3, 3, 3072)).astype(np.float32)
        expected = ((values[:, 0].astype(np.float64)+values[:, 1].astype(np.float64)+values[:, 2].astype(np.float64))/3).astype(np.float32)
        np.testing.assert_array_equal(raw.fuse_crop_features(values), expected)
        values[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): raw.fuse_crop_features(values)

    def test_independent_density_matches_reviewed_formula_all_four_temperatures(self):
        rng = np.random.RandomState(3)
        cases = [];fields = [];features = [];roles = {'train': [], 'validation': [], 'test': []}
        for label, fitting_count in enumerate((3, 4, 2, 3, 2)):
            for within in range(fitting_count+2):
                index = len(cases);cases.append({'truth5': label, 'truth8': (0, 1, 3, 4, 5)[label], 'case_id': 'synthetic%d' % index})
                role = 'train' if within < fitting_count else 'validation' if within == fitting_count else 'test'
                roles[role].append(index)
                for view in range(4):
                    for _ in range(1+(index+view) % 3):
                        fields.append({'index': len(fields), 'case_index': index, 'view': view})
                        features.append(rng.normal(size=7)+label*.1)
        values = np.asarray(features, dtype=np.float32);audit = {'cases': cases, 'fields': fields, 'roles': roles}
        actual, geometry = proof.fixed_field_probabilities(values, audit, .1)
        bags = [[values[[f['index'] for f in fields if f['case_index'] == case and f['view'] == view]] for view in range(4)] for case in range(len(cases))]
        labels = np.array([case['truth5'] for case in cases])
        expected_centers = prior_centered_model.centers(bags, labels, np.array(roles['train']))
        for view in range(4):
            for head in ('binary', 'malignant'):
                np.testing.assert_allclose(geometry[view]['centers'][head], expected_centers[head][view], rtol=0, atol=1e-14)
        for role in ('validation', 'test'):
            positions = {int(index): pos for pos, index in enumerate(actual[role]['indices'])}
            for query in roles[role]:
                prepared = prior_centered_model.prepare_case(bags, labels, roles['train'], query)
                for candidate, temperature in enumerate(actual[role]['settings']):
                    logs = prior_centered_model.field_log_probabilities(prepared, temperature, 'dual', 'head_specific')
                    for view, log in enumerate(logs):
                        indices = [f['index'] for f in fields if f['case_index'] == query and f['view'] == view]
                        expected = np.exp(log);expected /= expected.sum(1, keepdims=True)
                        np.testing.assert_allclose(actual[role]['probabilities'][candidate][[positions[index] for index in indices]], expected, rtol=0, atol=1e-12)
        self.assertEqual(actual['validation']['settings'], [.01, .03, .1, .3])
        self.assertEqual(actual['test']['settings'], [.1])  # no unused test candidates

    def test_independent_case_scores_match_registered_fixed_five_label_metrics(self):
        cases = [{'case_id': 'synthetic%d' % label, 'truth5': label, 'truth8': original}
                 for label, original in enumerate((0, 1, 3, 4, 5))]
        fields = [{'index': index, 'case_index': index//4, 'view': index % 4} for index in range(20)]
        audit = {'cases': cases, 'fields': fields, 'roles': {'test': list(range(5))}}
        indices = np.arange(20, dtype=np.int64);joint = np.asarray([np.eye(5)[index//4] for index in indices])
        predictions, summary, _ = proof.case_scores(audit, 'test', indices, joint)
        original = matched_data.summary(matched_data.case_records(fields, cases, list(range(5)), indices, joint, 'test'))
        self.assertEqual(summary['mean_subset_macro_f1'], original['mean_subset_macro_f1'])
        self.assertEqual(summary['mean_subset_accuracy'], 1.)
        np.testing.assert_array_equal(predictions, np.repeat(np.arange(5)[:, None], 15, axis=1))
        with self.assertRaises(ValueError): proof.case_scores(audit, 'test', indices[:-1], joint[:-1])


if __name__ == '__main__':
    unittest.main()

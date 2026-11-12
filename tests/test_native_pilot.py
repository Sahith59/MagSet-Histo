from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import softmax

from magset.data import file_hash
from magset.native_pilot import (ARMS, SUBSETS, CONFIG, allowed, audit_predictions, design,
    feature, fit_moments, metrics, peer_scores, predict_saved, prepare, projection, run)
from magset.tcga_protocol import make_protocol, RESOLUTIONS
from test_tcga_protocol import fixture


def toy_case(pid='case', role='fit', views=4, fields=3):
    rng = np.random.default_rng(12)
    return {'patient_id': pid, 'role': role, 'label': 0,
            'views': {m: {'paths': ['view%d/field%d.png' % (m, k) for k in range(fields)],
                          'features': rng.normal(size=(fields, 64))} for m in range(views)}}


def development_records():
    return [{'patient_id': 'dev%d' % label, 'role': 'development', 'truth': label,
             'subset': list(subset)} for subset in SUBSETS for label in range(16)]


class NativePilotTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros((4, 64)); self.std = np.ones((4, 64)); self.project = projection(64)

    def test_projection_is_fixed_orthogonal_and_label_blind(self):
        np.testing.assert_allclose(self.project.T @ self.project, np.eye(32), atol=1e-14)
        np.testing.assert_array_equal(self.project, projection(64))

    def test_equal_patient_moments_ignore_field_count_and_refuse_nonfit(self):
        a, b = toy_case('a', fields=1), toy_case('b', fields=5)
        for c, value in ((a, 0.), (b, 2.)):
            for view in c['views'].values(): view['features'][:] = value
        mean, std = fit_moments([a, b])
        np.testing.assert_allclose(mean, 1); np.testing.assert_allclose(std, 1)
        b['role'] = 'development'
        with self.assertRaisesRegex(ValueError, 'role'): fit_moments([a, b])
        with self.assertRaisesRegex(ValueError, 'Duplicate'): fit_moments([a, a])

    def test_missing_views_preserve_equal_patient_weight_without_row_balancing(self):
        cases = [toy_case('complete'), toy_case('incomplete', views=3)]
        x, y, w, records, seconds = design(cases, 'fit', self.mean, self.std, self.project)
        self.assertEqual(len(y), 22)
        mass = Counter()
        for r, v in zip(records, w): mass[r['patient_id']] += v
        self.assertAlmostEqual(mass['complete'], 1); self.assertAlmostEqual(mass['incomplete'], 1)
        self.assertEqual(x['mean'].shape, (22, 68)); self.assertEqual(x['affinity'].shape, (22, 100))
        self.assertIsNone(CONFIG['class_weight'])
        with self.assertRaisesRegex(ValueError, 'four views'):
            design([toy_case('incomplete', role='development', views=3)], 'development', self.mean, self.std, self.project)

    def test_permutation_and_missing_content_invariance_for_every_arm(self):
        case = toy_case(); changed = copy.deepcopy(case)
        changed['views'] = dict(reversed(list(changed['views'].items())))
        for view in changed['views'].values():
            view['paths'] = list(reversed(view['paths'])); view['features'] = view['features'][::-1]
        a = prepare(case, self.mean, self.std, self.project)
        b = prepare(changed, self.mean, self.std, self.project)
        for arm in ARMS:
            for subset in SUBSETS:
                np.testing.assert_allclose(feature(a, subset, arm, self.project),
                    feature(b, tuple(reversed(subset)), arm, self.project), atol=1e-13)
        changed['views'][2]['features'][:] = np.nan
        changed['views'][3]['features'][:] = 1e200
        b = prepare(changed, self.mean, self.std, self.project, subset=(0, 1))
        for arm in ARMS:
            np.testing.assert_allclose(feature(a, (0, 1), arm, self.project),
                feature(b, (0, 1), arm, self.project), atol=1e-13)
        for invalid in ((), (0, 0), (4,)):
            with self.assertRaises(ValueError): feature(a, invalid, 'mean', self.project)

    def test_baseline_block_one_view_boundary_and_signed_contrast(self):
        prepared = prepare(toy_case(), self.mean, self.std, self.project)
        for subset in SUBSETS:
            base = feature(prepared, subset, 'mean', self.project)
            for arm in ARMS[1:]:
                x = feature(prepared, subset, arm, self.project)
                np.testing.assert_array_equal(x[:len(base)], base)
                if arm != 'duplicate' and len(subset) == 1:
                    np.testing.assert_array_equal(x[-32:], 0)
            np.testing.assert_allclose(feature(prepared, subset, 'duplicate', self.project)[-32:], base[:-4] @ self.project)
        # Two opposite field directions and an agreeing peer. Affinity must
        # favor the agreeing direction. This is a signed contrast, not OT mass.
        e = np.eye(32)[0]
        manual = {0: {'mean': np.zeros(64), 'projected': np.stack([e, -e]), 'unit': np.stack([e, -e])},
                  1: {'mean': np.zeros(64), 'projected': e[None], 'unit': e[None]}}
        x = feature(manual, (0, 1), 'affinity', self.project)
        expected = .5 * np.tanh(5.)
        self.assertAlmostEqual(x[-32], expected)
        np.testing.assert_array_equal(x[-31:], 0)
        for m in (0, 1):
            weights = softmax(peer_scores(manual)[m, 1-m])
            matched = weights @ manual[m]['projected']
            residual = manual[m]['projected'].mean(axis=0) - matched
            np.testing.assert_allclose(matched + residual, manual[m]['projected'].mean(axis=0), atol=1e-15)

    def test_metric_denominators_brier_and_coverage_are_explicit16_class(self):
        records = development_records(); truth = np.asarray([r['truth'] for r in records])
        perfect = metrics(records, np.eye(16)[truth])
        self.assertEqual(perfect['mean_subset_macro_f1'], 1.0)
        self.assertEqual(perfect['mean_subset_brier'], 0.0)
        uniform = metrics(records, np.full((240, 16), 1/16))
        self.assertAlmostEqual(uniform['mean_subset_brier'], 15/16)
        one = next(iter(uniform['subsets'].values()))
        self.assertAlmostEqual(one['accuracy'], 1/16)
        self.assertAlmostEqual(one['balanced_accuracy'], 1/16)
        self.assertAlmostEqual(one['macro_f1'], (2/17)/16)
        with self.assertRaises(ValueError): metrics(records, np.full((240, 8), 1/8))
        with self.assertRaisesRegex(ValueError, 'exactly one'): metrics(records[:-1], np.full((239, 16), 1/16))
        bad = np.full((240, 16), 1/16); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): metrics(records, bad)
        bad = np.full((240, 16), 1/16); bad[0, 0] += 1e-7
        with self.assertRaises(ValueError): metrics(records, bad)

    def test_sealed_scoring_refused_before_model_access_and_reload_is_numeric(self):
        sealed = [toy_case('sealed', role='sealed_test')]
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            predict_saved('/nonexistent', np.zeros((1, 68)), sealed, 'sealed_test')
        with self.assertRaisesRegex(ValueError, 'role'):
            design(sealed, 'development', self.mean, self.std, self.project)
        rng = np.random.default_rng(7); x = rng.normal(size=(5, 68)); coef = rng.normal(size=(16, 68))
        intercept = rng.normal(size=16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'
            np.savez_compressed(path, coef=coef, intercept=intercept, classes=np.arange(16))
            p = predict_saved(path, x, [toy_case('dev', role='development')], 'development')
            expected = np.exp(x @ coef.T + intercept - (x @ coef.T + intercept).max(axis=1, keepdims=True))
            expected /= expected.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(p, expected, atol=1e-15)

    def test_prediction_audit_rejects_truth_role_and_membership_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = fixture(root/'inventory'); out = root/'protocol'
            p = make_protocol(root/'inventory', out, source); protocol = out/'protocol.json'
            records = [{'arm': a, 'patient_id': pid, 'role': 'development',
                        'truth': p['patient_records'][pid]['label'], 'subset': list(s), 'probabilities': [1/16]*16}
                       for a in ARMS for pid, role in p['roles'].items() if role == 'development' for s in SUBSETS]
            predictions = root/'predictions.jsonl'
            def save(rows): predictions.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save(records); self.assertEqual(set(audit_predictions(predictions, protocol, file_hash(protocol))), set(ARMS))
            for kind in ('truth', 'role', 'membership'):
                bad = copy.deepcopy(records)
                if kind == 'truth': bad[0]['truth'] = (bad[0]['truth']+1)%16
                if kind == 'role': bad[0]['patient_id'] = next(k for k,v in p['roles'].items() if v == 'sealed_test')
                if kind == 'membership': bad[0] = bad[1]
                save(bad)
                with self.assertRaises(ValueError): audit_predictions(predictions, protocol, file_hash(protocol))

    @unittest.skipIf(importlib.util.find_spec('torch') is None, 'Native cache adapter requires pinned Torch/Pillow')
    def test_full_synthetic_pipeline_four_fits_reload_and_no_reserved_predictions(self):
        from magset.verify_native import verify
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = fixture(root/'inventory'); out = root/'registered'
            p = make_protocol(root/'inventory', out, source); manifest = out/'manifest.csv'; protocol = out/'protocol.json'
            from magset.tcga_features import native_rows
            rows, _ = native_rows(manifest, protocol, file_hash(protocol)); cache = root/'cache'; cache.mkdir()
            encoder = json.loads(Path('configs/encoders.json').read_text())['resnet50']
            for resolution in RESOLUTIONS:
                keys = [r['relative_path'] for r in rows if r['resolution_mpp'] == resolution]
                target = cache/('resnet50-mpp%s.npz'%resolution)
                np.savez_compressed(target, paths=np.asarray(keys), features=np.zeros((len(keys),2048), dtype=np.float32))
                target.with_suffix('.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest),
                    'protocol_sha256':file_hash(protocol), 'feature_sha256':file_hash(target), 'images':len(keys),
                    'encoder':encoder, 'smoke_limit':0, 'label_blind':True, 'code_sha':'synthetic'}))
            audit = root/'audit.json'; audit.write_text(json.dumps(verify(manifest,protocol,cache,file_hash(protocol))))
            result = run(manifest,protocol,cache,file_hash(protocol),audit,file_hash(audit),root/'run','synthetic-test')
            self.assertEqual(result['fits_completed'],4); self.assertEqual(result['training_rows'],1152)
            self.assertEqual(result['sealed_predictions_computed'],0)
            self.assertTrue(result['preprocessing_reload_verified'])
            self.assertAlmostEqual(result['training_weight'],80)
            for value in result['class_training_weight'].values(): self.assertAlmostEqual(value,5)
            self.assertEqual(result['arms']['mean']['stored_coefficient_count'],32848)
            self.assertEqual(result['arms']['affinity']['stored_coefficient_count'],33360)
            with self.assertRaisesRegex(ValueError,'exists'):
                run(manifest,protocol,cache,file_hash(protocol),audit,file_hash(audit),root/'run','synthetic-test')


if __name__ == '__main__': unittest.main()

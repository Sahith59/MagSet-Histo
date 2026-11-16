"""3072-feature contract/coordinator fixtures; no fits, images or encoder builds."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import midnight_five as m


def configuration():
    config = json.loads(Path('configs/breakhis-five-class-v1.json').read_text())
    extraction = json.loads(Path('configs/midnight-12k-v1.json').read_text())
    config.update(name='breakhis-midnight-five-v1', schema_version=1,
                  encoder_name='midnight-12k', encoder_dimension=3072, expected_images=7909,
                  encoder=extraction['encoder'], readouts=['case', 'field'], max_iter=2000,
                  total_production_runs=20, total_head_fits=240, new_encoder_forwards=0,
                  head_weight_mass={'direct': 46, 'binary': 46, 'malignant': 35},
                  feature_producer_code_sha=m.FEATURE_PRODUCER_SHA,
                  feature_extraction_config_sha256=m.file_hash('configs/midnight-12k-v1.json'))
    return config


def fixture():
    config = configuration(); cases = []; bags = []; outer = {}
    for label, (count, held) in enumerate(zip([4, 38, 10, 5, 9, 6, 3, 7], [2, 12, 3, 2, 3, 2, 1, 3])):
        for index in range(count):
            cid = 'fixture-%d-%d' % (label, index)
            cases.append({'case_id': cid, 'label': label})
            outer[cid] = 'test' if index < held else 'train'
            views = []
            for view in range(4):
                fields = np.zeros((1+(index+view) % 2, 3072), np.float32)
                fields[:, 0] = label/8; fields[:, 1] = index/40
                fields[:, 2] = np.arange(len(fields))/10+view/8
                views.append(fields)
            bags.append(views)
    y5 = m.five_labels(np.array([case['label'] for case in cases]))
    roles = {(seed, trial): m.stratified_inner_split(cases, outer, seed)
             for seed in config['seeds'] for trial in config['trials']}
    z = np.asarray([[view.mean(0) for view in case] for case in bags], dtype=np.float32)
    trusted = {'cases': cases, 'manifest_sha256': config['manifest_sha256'],
               'protocol_sha256': config['protocol_sha256'], 'unit': 'synthetic provider case',
               'trials': {str(trial): {'case_roles': outer} for trial in config['trials']}}
    context = {'code_sha': 'a'*40, 'source_archive_sha256': 'b'*64,
               'implementation_hashes': {}, 'config_sha256': 'c'*64}
    return config, context, trusted, cases, bags, z, y5, roles


def models(arm):
    result = {}
    names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
    for name in names:
        model = LogisticRegression()
        count = 1 if name == 'binary' else 5 if name == 'direct' else 4
        model.coef_ = np.zeros((count, 3076), dtype=np.float64)
        model.coef_[:, :3] = np.arange(count*3).reshape(count, 3)/100
        model.intercept_ = np.linspace(-.3, .3, count, dtype=np.float64)
        model.classes_ = np.arange(2) if name == 'binary' else np.arange(5) if name == 'direct' else np.arange(1, 5)
        model.n_features_in_ = 3076; model.n_iter_ = np.array([2])
        result[name] = model
    return result


class MidnightFiveTests(unittest.TestCase):
    def test_fixed_recipe_rejects_scope_grid_label_budget_and_missing_cache_changes(self):
        config = configuration(); m.validate_config(config)
        for key, changed in [('encoder_dimension', 1024), ('C_grid', [.1]), ('max_iter', 1000),
                             ('readouts', ['case']), ('total_head_fits', 120), ('technical_head_fits', 1),
                             ('map8_to5', list(range(8))), ('feature_producer_code_sha', 'a'*40),
                             ('cache_provenance', {}), ('feature_extraction_config_sha256', None)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                candidate = copy.deepcopy(config); candidate[key] = changed
                m.validate_config(candidate)

    def test_actual_3072_case_and_field_saved_heads_replay_without_fitting(self):
        config, context, trusted, cases, bags, z, y5, roles = fixture()
        train, val, excluded = roles[812, 1]
        mean, std = m.fit_normalization(bags, train)
        with tempfile.TemporaryDirectory() as tmp, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')):
            for arm in m.ARMS:
                heads = models(arm); path = Path(tmp)/(arm+'.npz')
                m.save_model(path, mean, std, heads)
                arrays = m.load_model(path, arm, dimension=3072)
                self.assertEqual(arrays['mean'].shape, (4, 3072))
                for readout in m.READOUTS:
                    scorer, inputs = (m.case.score_records, z) if readout == 'case' else (m.field.score_records, bags)
                    rows, replay = scorer(arrays, inputs, val, cases, 'validation', arm, heads)
                    self.assertEqual(len(rows), 120); self.assertLess(replay, 1e-12)
                    self.assertEqual(m.summarize_records(rows, arm)['distinct_class_support'], [4, 1, 1, 1, 1])
                with self.assertRaises(ValueError):
                    m.load_model(path, arm, dimension=1024)

    def test_shared_audit_and_both_readout_budgets_with_earliest_ties(self):
        loaded = fixture()
        def fake_case(x, y, arm, C, config):
            return models(arm)
        def fake_field(bags, train, y5, mean, std, arm, C, config):
            return models(arm)
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'load_inputs', return_value=loaded), patch.object(m.case, 'fit_arm', side_effect=fake_case) as case_fit, patch.object(m.field, 'fit_arm', side_effect=fake_field) as field_fit, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')), patch('sys.stdout', new_callable=io.StringIO):
            root = Path(tmp); audit = root/'audit'
            m.audit_input(SimpleNamespace(output=audit))
            receipt = json.loads((audit/'audit.json').read_text())
            self.assertNotIn('readout', receipt)
            self.assertEqual(len(receipt['checks']), 10)
            for readout in m.READOUTS:
                args = SimpleNamespace(output=root/readout/'split812/trial1', audit=audit,
                                       seed=812, trial=1, readout=readout)
                m.run(args)
                report = json.loads((args.output/'run.json').read_text())
                expected_status = 'complete five-class validation run' if readout == 'case' else 'complete field-supervised five-class validation run'
                self.assertEqual(report['status'], expected_status)
                self.assertEqual(report['readout'], readout)
                self.assertEqual(report['role_counts'], [46, 8, 28])
                self.assertEqual(report['head_fits'], 12)
                for arm in m.ARMS:
                    self.assertEqual(report['arms'][arm]['selected_index'], 0)
                    self.assertEqual(len(report['arms'][arm]['candidates']), 4)
                    self.assertEqual('field_receipts' in report['arms'][arm]['candidates'][0], readout == 'field')
                self.assertEqual(len(list(args.output.rglob('model.npz'))), 8)
                with self.assertRaises(ValueError):
                    m.run(args)
            self.assertEqual(case_fit.call_count, 8)
            self.assertEqual(field_fit.call_count, 8)

    def test_missing_readout_and_changed_common_audit_cannot_fit(self):
        loaded = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'load_inputs', return_value=loaded), patch.object(m.case, 'fit_arm', side_effect=AssertionError('No fitting')), patch.object(m.field, 'fit_arm', side_effect=AssertionError('No fitting')):
            root = Path(tmp); args = SimpleNamespace(output=root/'invalid', readout=None)
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                m.run(args)
            m.audit_input(SimpleNamespace(output=root/'audit'))
            path = root/'audit/audit.json'; record = json.loads(path.read_text())
            record['config_sha256'] = 'd'*64; m.write_json(path, record)
            args = SimpleNamespace(output=root/'run', readout='case', seed=812, trial=1, audit=root/'audit')
            with self.assertRaisesRegex(ValueError, 'audit/context'):
                m.run(args)

    def test_canonical_source_config_is_required_before_cache_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'magset').mkdir(); (root/'configs').mkdir()
            canonical = root/'configs/breakhis-midnight-five-v1.json'; canonical.write_text('{}')
            other = root/'different.json'; other.write_text('{"changed":true}')
            with patch.object(m, '__file__', str(root/'magset/midnight_five.py')), self.assertRaisesRegex(ValueError, 'immutable source registration'):
                m.load_inputs(SimpleNamespace(config=other))

    def test_loader_binds_actual_cache_hashes_producer_and_full_3072_coverage(self):
        config, context, trusted, cases, bags, z, y5, roles = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'magset').mkdir(); (root/'configs').mkdir(); cache = root/'cache'; cache.mkdir()
            source = root/'magset/midnight_features.py'; source.write_text('synthetic source fixture')
            extraction = json.loads(Path('configs/midnight-12k-v1.json').read_text())
            extraction_path = root/'configs/midnight-12k-v1.json'; m.write_json(extraction_path, extraction)
            config['feature_extraction_config_sha256'] = m.file_hash(extraction_path)
            for scale in m.MAGNIFICATIONS:
                path = cache/('midnight-12k-%d.npz'%scale); path.write_bytes(b'synthetic cache fixture')
                metadata = {'code_sha': m.FEATURE_PRODUCER_SHA,
                            'config_sha256': config['feature_extraction_config_sha256'],
                            'source_file_sha256': m.file_hash(source), 'encoder': config['encoder']}
                m.write_json(path.with_suffix('.json'), metadata)
                config['cache_provenance'][str(scale)] = {'npz_sha256': m.file_hash(path),
                                                         'json_sha256': m.file_hash(path.with_suffix('.json'))}
            config_path = root/'configs/breakhis-midnight-five-v1.json'; m.write_json(config_path, config)
            args = SimpleNamespace(config=config_path, cache=cache, manifest=None, protocol=None, code_sha='a'*40)
            broad = {case['case_id']: int(label > 0) for case, label in zip(cases, y5)}
            with patch.object(m, '__file__', str(root/'magset/midnight_five.py')), patch.object(m, 'source_context', return_value=context.copy()), patch.object(m.importlib.metadata, 'version', side_effect=lambda name: config['dependencies'][name]), patch.object(m, 'load_trusted_context', return_value=trusted), patch.object(m, 'read_manifest', return_value=[None]*7909), patch.object(m, 'trusted_binary_labels', return_value=broad), patch.object(m, 'load_bags', return_value=(cases, bags, [])):
                actual = m.load_inputs(args)
                self.assertEqual(actual[5].shape, (82, 4, 3072))
                self.assertEqual(actual[1]['config_sha256'], m.file_hash(config_path))
                self.assertEqual(actual[1]['feature_producer_code_sha'], m.FEATURE_PRODUCER_SHA)
                self.assertEqual(set(actual[7]), set(roles))
                path = cache/'midnight-12k-40.npz'; path.write_bytes(b'changed cache')
                with self.assertRaisesRegex(ValueError, 'cache bytes'):
                    m.load_inputs(args)
                path.write_bytes(b'synthetic cache fixture')
                meta_path = path.with_suffix('.json'); metadata = json.loads(meta_path.read_text())
                metadata['code_sha'] = 'f'*40; m.write_json(meta_path, metadata)
                config['cache_provenance']['40']['json_sha256'] = m.file_hash(meta_path); m.write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, 'producer/encoder'):
                    m.load_inputs(args)


if __name__ == '__main__':
    unittest.main()

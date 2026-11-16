"""Bounded metadata and explicit native-bag delegation fixtures, no real cache/model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import native_five as m, kernel_five as kernel
from magset.data import MAGNIFICATIONS
from magset.five_collect import read_json
from test_kernel_five import fixture


def bound_config():
    config = json.loads(Path('configs/breakhis-native-kernel-v1.json').read_text())
    config.update(feature_producer_code_sha='a'*40, feature_audit_sha256='b'*64,
                  cache_provenance={str(v): {'npz_sha256': 'c'*64, 'json_sha256': 'd'*64, 'images': n}
                                    for v, n in zip(MAGNIFICATIONS, (1995, 2081, 2013, 1820))})
    return config


class NativeKDETests(unittest.TestCase):
    def test_unbound_or_different_recipe_cannot_reach_any_classifier_math(self):
        config = bound_config(); self.assertIs(m.validate_bindings(config), config)
        for key, value in [('feature_producer_code_sha', None), ('feature_audit_sha256', None), ('cache_provenance', {}), ('temperatures', [.001, .01, .1, 1.])]:
            changed = copy.deepcopy(config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_bindings(changed)
        changed = copy.deepcopy(config); changed['cache_provenance']['40']['images'] += 1
        with self.assertRaisesRegex(ValueError, '7909'): m.validate_bindings(changed)
        changed = copy.deepcopy(config); changed['cache_provenance']['40']['components_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'binding'): m.validate_bindings(changed)

    def test_native_bags_are_explicitly_delegated_without_center_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = fixture(root)
            native = [[x*2+.3 for x in case] for case in f.bags]
            inputs = (*f.inputs[:4], native, f.z, f.y5, f.roles)
            args = SimpleNamespace(output=root/'native', audit=f.audit_dir, seed=812, trial=1)
            # Use an audit generated from these supplied native bags.
            with patch.object(m, 'load_inputs', return_value=inputs):
                args.output = root/'native-audit'; m.audit_input(args)
                args.audit = args.output; args.output = root/'native'
                with patch.object(kernel, 'load_inputs', side_effect=AssertionError('No original-reader fallback')):
                    m.run(args)
            report = read_json(args.output/'run.json')
            from magset.kernel_model import support_summary
            expected = support_summary(native, f.y5, f.roles[812, 1][0])['centers_sha256']
            old = support_summary(f.bags, f.y5, f.roles[812, 1][0])['centers_sha256']
            model = read_json(args.output/'dual/candidate0/model.json')
            self.assertEqual(model['support_summary']['centers_sha256'], expected)
            self.assertNotEqual(expected, old)
            self.assertEqual(report['optimized_head_fits'], 0)
            self.assertEqual(len(report['arms']['dual']['candidates']), 4)

    def test_loader_binds_audit_geometry_dependencies_and_replaces_all_old_features(self):
        own = bound_config(); extraction = json.loads(Path('configs/midnight-native-v1.json').read_text())
        old_config = json.loads(Path('configs/breakhis-midnight-five-v1.json').read_text())
        cases = [{'case_id': 'synthetic%d'%i, 'label': 0} for i in range(82)]
        old = [[np.zeros((1, 3072), np.float32) for _ in MAGNIFICATIONS] for _ in cases]
        native = [[np.full((1, 3072), 3, np.float32) for _ in MAGNIFICATIONS] for _ in cases]
        context = {'implementation_hashes': {}}
        trusted = {'manifest_sha256': extraction['manifest_sha256']}
        inputs = (old_config, context, trusted, cases, old, None, np.zeros(82, int), {})
        metadata = {}
        for v in MAGNIFICATIONS:
            metadata[v] = {'code_sha': own['feature_producer_code_sha'], 'config_sha256': own['native_extraction_config_sha256'],
                'manifest_sha256': trusted['manifest_sha256'], 'source_file_sha256': 'e'*64, 'encoder': extraction['encoder'],
                'smoke_limit': 0, 'images': own['cache_provenance'][str(v)]['images'], 'input_shape': [3, 448, 700],
                'token_shape': [1601, 1536], 'frozen_base': True, 'model_image_size': 518, 'position_grid': [37, 37],
                'patch_grid': [32, 50], 'dependencies': extraction['dependencies'], 'native_forwards': own['cache_provenance'][str(v)]['images'],
                'attention_implementation': 'eager', 'precision': 'FP32; TF32 disabled', 'transform': extraction['encoder']['transform'],
                'base_extraction_config_sha256': extraction['base_extraction_config_sha256'], 'base_source_file_sha256': 'e'*64}
        audit = {'status': 'accepted all four frozen native-input shards', 'code_sha': own['feature_producer_code_sha'],
                 'config_sha256': own['native_extraction_config_sha256'], 'manifest_sha256': trusted['manifest_sha256'],
                 'images': 7909, 'encoder': extraction['encoder'], 'shards': own['cache_provenance'], 'source_file_sha256': 'e'*64}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)/'native'; cache.mkdir()
            for v in MAGNIFICATIONS: (cache/('midnight-native-v1-%d.npz'%v)).touch()
            args = SimpleNamespace(config=Path('native-config.json'), center_cache=Path('center-cache'), cache=cache,
                                   feature_audit=Path('feature-audit.json'), manifest=Path('fixture.csv'))
            def digest(path):
                path = Path(path)
                if path == args.config or path.name == 'breakhis-native-kernel-v1.json': return 'f'*64
                if path.name == 'breakhis-midnight-five-v1.json': return own['base_input_config_sha256']
                if path.name == 'midnight-native-v1.json': return own['native_extraction_config_sha256']
                if path == args.feature_audit: return own['feature_audit_sha256']
                if path.suffix == '.npz': return 'c'*64
                if path.suffix == '.json': return 'd'*64
                return 'e'*64
            def read(path):
                path = Path(path)
                if path == args.config: return own
                if path.name == 'midnight-native-v1.json': return extraction
                if path == args.feature_audit: return audit
                return metadata[int(path.stem.rsplit('-', 1)[1])]
            with patch.object(m, 'file_hash', side_effect=digest), patch.object(m, 'read_json', side_effect=read), patch.object(m.base, 'load_inputs', return_value=inputs) as loader, patch.object(m, 'read_manifest', return_value=[]), patch.object(m, 'load_bags', return_value=(cases, native, [])):
                result = m.load_inputs(args)
                self.assertIs(result[4], native)
                self.assertTrue(np.all(result[5] == 3))
                self.assertEqual(loader.call_args.args[0].cache, args.center_cache)
                self.assertEqual(result[1]['cache_provenance'], own['cache_provenance'])
                for field, bad in [('smoke_limit', 4), ('token_shape', [257, 1536]), ('dependencies', {}), ('frozen_base', False), ('source_file_sha256', '0'*64)]:
                    previous = metadata[40][field]; metadata[40][field] = bad
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'lineage'): m.load_inputs(args)
                    metadata[40][field] = previous
                audit['code_sha'] = '0'*40
                with self.assertRaisesRegex(ValueError, 'audit binding'): m.load_inputs(args)


if __name__ == '__main__': unittest.main()

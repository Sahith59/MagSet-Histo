"""Bounded metadata and explicit complementary-bag delegation fixtures, no real cache/model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import approved_five as m, kernel_five as kernel
from magset.data import MAGNIFICATIONS
from magset.five_collect import read_json
from test_kernel_five import fixture


def bound_config():
    config = json.loads(Path('configs/breakhis-uni2-h-v1-kernel.json').read_text())
    config.update(feature_producer_code_sha='a'*40, feature_audit_sha256='b'*64,
                  cache_provenance={str(v): {'npz_sha256': 'c'*64, 'json_sha256': 'd'*64, 'images': n}
                                    for v, n in zip(MAGNIFICATIONS, (1995, 2081, 2013, 1820))})
    return config


class ApprovedKDETests(unittest.TestCase):
    def test_all_readers_keep_the_registered_dimension_and_four_setting_budget(self):
        for name, dimension in [('uni2-h-v1', 1536), ('virchow2-v1', 2560), ('h-optimus-0-v1', 1536)]:
            config = bound_config()
            config.update(name='breakhis-' + name + '-kernel', encoder_name=name, dimension=dimension)
            self.assertIs(m.validate_bindings(config), config)
            config['dimension'] = dimension + 1
            with self.assertRaises(ValueError):
                m.validate_bindings(config)

    def test_unbound_or_different_recipe_cannot_reach_any_classifier_math(self):
        config = bound_config(); self.assertIs(m.validate_bindings(config), config)
        for key, value in [('feature_producer_code_sha', None), ('feature_audit_sha256', None), ('cache_provenance', {}), ('temperatures', [.001, .01, .1, 1.])]:
            changed = copy.deepcopy(config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_bindings(changed)
        changed = copy.deepcopy(config); changed['cache_provenance']['40']['images'] += 1
        with self.assertRaisesRegex(ValueError, '7909'): m.validate_bindings(changed)
        changed = copy.deepcopy(config); changed['cache_provenance']['40']['components_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'binding'): m.validate_bindings(changed)

    def test_complementary_bags_are_explicitly_delegated_without_center_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = fixture(root)
            complementary = [[x*2+.3 for x in case] for case in f.bags]
            inputs = (*f.inputs[:4], complementary, f.z, f.y5, f.roles)
            args = SimpleNamespace(output=root/'complementary', audit=f.audit_dir, seed=812, trial=1)
            # Use an audit generated from these supplied complementary bags.
            with patch.object(m, 'load_inputs', return_value=inputs):
                args.output = root/'complementary-audit'; m.audit_input(args)
                args.audit = args.output; args.output = root/'complementary'
                with patch.object(kernel, 'load_inputs', side_effect=AssertionError('No original-reader fallback')):
                    m.run(args)
            report = read_json(args.output/'run.json')
            from magset.kernel_model import support_summary
            expected = support_summary(complementary, f.y5, f.roles[812, 1][0])['centers_sha256']
            old = support_summary(f.bags, f.y5, f.roles[812, 1][0])['centers_sha256']
            model = read_json(args.output/'dual/candidate0/model.json')
            self.assertEqual(model['support_summary']['centers_sha256'], expected)
            self.assertNotEqual(expected, old)
            self.assertEqual(report['optimized_head_fits'], 0)
            self.assertEqual(len(report['arms']['dual']['candidates']), 4)


class ApprovedLoaderTests(unittest.TestCase):
    def test_loader_preserves_roles_and_binds_every_new_field_archive(self):
        from magset import approved_features as producer
        own=bound_config(); extraction=json.loads(Path('configs/uni2-h-v1.json').read_text())
        original=json.loads(Path('configs/breakhis-midnight-five-v1.json').read_text())
        cases=[{'case_id':'synthetic%d'%i,'label':0} for i in range(82)]
        bags=[[np.full((1,1536),3,np.float32) for _ in MAGNIFICATIONS] for _ in cases]
        inputs=(original,{'implementation_hashes':{}},{'manifest_sha256':extraction['manifest_sha256']},cases,None,None,np.zeros(82,int),{})
        audit={'status':'accepted all four frozen approved encoder shards','encoder_name':own['encoder_name'],'code_sha':own['feature_producer_code_sha'],
               'config_sha256':own['extraction_config_sha256'],'manifest_sha256':extraction['manifest_sha256'],
               'images':7909,'encoder':extraction['encoder'],'shards':own['cache_provenance'],'source_file_sha256':'e'*64}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir()
            for mag in MAGNIFICATIONS:(cache/('uni2-h-v1-%d.npz'%mag)).touch()
            args=SimpleNamespace(config=root/'own.json',center_cache=root/'old',cache=cache,feature_audit=root/'audit.json',manifest=root/'manifest.csv')
            def hashed(path):
                path=Path(path)
                if path==args.config or path.name=='breakhis-uni2-h-v1-kernel.json':return '0'*64
                if path==args.feature_audit:return own['feature_audit_sha256']
                if path.name=='breakhis-midnight-five-v1.json':return own['base_input_config_sha256']
                if path.name=='uni2-h-v1.json':return own['extraction_config_sha256']
                if path.parent==cache:return 'c'*64 if path.suffix=='.npz' else 'd'*64
                return 'e'*64
            rows=[{'magnification':str(m),'relative_path':'synthetic/%d.png'%m} for m in MAGNIFICATIONS]
            def verify(path, paths, config, limit, sha):
                mag=int(Path(path).stem.rsplit('-',1)[1]);self.assertEqual(paths,['synthetic/%d.png'%mag]);self.assertEqual(limit,0)
                self.assertEqual(sha,own['feature_producer_code_sha']);self.assertEqual(config,extraction)
                return {'magnification':mag,'images':own['cache_provenance'][str(mag)]['images']}
            def read(path):return audit if Path(path)==args.feature_audit else own
            with patch.object(m,'file_hash',side_effect=hashed),patch.object(m,'read_json',side_effect=read),patch.object(m.base,'load_inputs',return_value=inputs) as loader,patch.object(producer,'read_config',return_value=extraction),patch.object(producer,'verify_output',side_effect=verify) as shard,patch.object(m,'read_manifest',return_value=rows),patch.object(m,'load_bags',return_value=(cases,bags,[])):
                result=m.load_inputs(args)
                self.assertIs(result[4],bags);self.assertEqual(result[5].shape,(82,4,1536));self.assertEqual(result[0]['encoder_dimension'],1536)
                self.assertEqual(loader.call_args.args[0].cache,args.center_cache);self.assertEqual(shard.call_count,4)
                audit['encoder_name'] = 'h-optimus-0-v1'
                with self.assertRaisesRegex(ValueError, 'lineage'):
                    m.load_inputs(args)
                audit['encoder_name'] = own['encoder_name']
                extraction['encoder_name'] = 'h-optimus-0-v1'
                with self.assertRaisesRegex(ValueError, 'identity or dimension'):
                    m.load_inputs(args)
                extraction['encoder_name'] = own['encoder_name']
                audit['source_file_sha256']='f'*64
                with self.assertRaisesRegex(ValueError,'lineage'):m.load_inputs(args)


if __name__ == '__main__': unittest.main()

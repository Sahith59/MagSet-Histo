import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from magset.data import file_hash
from magset.tcga_protocol import make_protocol
from test_tcga_protocol import fixture
try:
    from magset.tcga_features import native_rows
    from magset.verify_native import verify
except ImportError:
    native_rows=None


@unittest.skipIf(native_rows is None,'Pinned Torch/Pillow needed for native feature adapter')
class NativeFeatureTests(unittest.TestCase):
    def test_all_cache_keys_values_and_provenance_are_verified_without_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');out=root/'registered'
            make_protocol(root/'inventory',out,source);manifest=out/'manifest.csv';protocol=out/'protocol.json'
            ph=file_hash(protocol);rows,p=native_rows(manifest,protocol,ph);cache=root/'cache';cache.mkdir()
            encoder=json.loads(Path('configs/encoders.json').read_text())['resnet50']
            for resolution in p['resolution_mpp']:
                keys=[r['relative_path'] for r in rows if r['resolution_mpp']==resolution]
                path=cache/('resnet50-mpp%s.npz'%resolution)
                np.savez_compressed(path,paths=np.asarray(keys),features=np.zeros((len(keys),2048),dtype=np.float32))
                path.with_suffix('.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest),
                    'protocol_sha256':ph,'feature_sha256':file_hash(path),'images':len(keys),'encoder':encoder,
                    'smoke_limit':0,'label_blind':True,'code_sha':'synthetic'}))
            report=verify(manifest,protocol,cache,ph)
            self.assertEqual(report['images'],p['fields']);self.assertEqual(report['classifier_predictions_computed'],0)
            path=cache/'resnet50-mpp0.25.npz'
            with np.load(path,allow_pickle=False) as saved:keys=saved['paths'];z=saved['features']
            z[0,0]=np.nan;np.savez_compressed(path,paths=keys,features=z)
            meta=json.loads(path.with_suffix('.json').read_text());meta['feature_sha256']=file_hash(path)
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'values invalid'):verify(manifest,protocol,cache,ph)

    def test_role_path_patient_and_registered_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');out=root/'registered'
            make_protocol(root/'inventory',out,source)
            manifest=out/'manifest.csv';protocol=out/'protocol.json';ph=file_hash(protocol)
            rows,p=native_rows(manifest,protocol,ph)
            self.assertEqual(len(rows),p['fields'])
            with self.assertRaisesRegex(ValueError,'registered job hash'):
                native_rows(manifest,protocol,'other')
            rows[0]['resolution_mpp']='2.0' if rows[0]['resolution_mpp']!='2.0' else '0.5'
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            p['manifest_sha256']=file_hash(manifest);protocol.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'path/patient'):
                native_rows(manifest,protocol,file_hash(protocol))


import copy
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from magset import native_features as midnight_native
from magset import midnight_features as midnight_base
from magset import coverage_features as coverage_source
from magset.data import MAGNIFICATIONS
try:
    import torch
    import torchvision
except ImportError:
    torch = None

MIDNIGHT_NATIVE_ROOT = Path(__file__).resolve().parents[1]
_native_spec = importlib.util.spec_from_file_location('native_input_audit_fixture', MIDNIGHT_NATIVE_ROOT/'scripts/audit_native_features.py')
native_input_audit = importlib.util.module_from_spec(_native_spec)
_native_spec.loader.exec_module(native_input_audit)


def _native_configuration():
    return json.loads((MIDNIGHT_NATIVE_ROOT/'configs/midnight-native-v1.json').read_text())


def _native_record(config, scale, count, limit=0):
    return {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
            'config_sha256': midnight_native.CONFIG_SHA256, 'code_sha': 'a'*40,
            'source_file_sha256': file_hash(midnight_native.__file__), 'base_source_file_sha256': file_hash(midnight_base.__file__),
            'magnification': scale, 'images': count, 'smoke_limit': limit,
            'dependencies': config['dependencies'], 'base_extraction_config_sha256': config['base_extraction_config_sha256'],
            'input_shape': [3, 448, 700], 'token_shape': [1601, 1536], 'model_image_size': 518,
            'position_grid': [37, 37], 'patch_grid': [32, 50], 'frozen_base': True,
            'attention_implementation': 'eager', 'precision': 'FP32; TF32 disabled', 'transform': config['encoder']['transform'],
            'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'], 'parity': {},
            'native_forwards': count if not limit else 16, 'geometry_counts': {'700x460': count//2, '700x456': count-count//2}}


class MidnightNativeFeatureTests(unittest.TestCase):
    def test_native_registered_configuration_and_frozen_base_pins(self):
        config = midnight_native.read_config(MIDNIGHT_NATIVE_ROOT/'configs/midnight-native-v1.json')
        self.assertEqual(config['expected_token_shape'], [1601, 1536])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'different.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration bytes'):
                midnight_native.read_config(path)

    def test_native_manifest_complete_geometry_and_mixed_height_smoke(self):
        config = _native_configuration()
        rows = [{'relative_path': 'fixture/%d.png'%i, 'width': 700,
                 'height': 460 if i < 7835 else 456, 'magnification': 40} for i in range(7909)]
        with patch.object(coverage_source, 'file_hash', return_value=config['manifest_sha256']), patch.object(coverage_source, 'read_manifest', return_value=rows):
            _, selected = midnight_native.select_rows(Path('fixture.csv'), config, 40, 4)
            self.assertEqual([row['height'] for row in selected], [460, 460, 456, 456])
            self.assertEqual([row['relative_path'] for row in selected], ['fixture/0.png', 'fixture/1.png', 'fixture/7835.png', 'fixture/7836.png'])
            rows[0]['height'] = 448
            with self.assertRaisesRegex(ValueError, 'geometry'):
                midnight_native.select_rows(Path('fixture.csv'), config, 40, 0)

    @unittest.skipIf(torch is None, 'Torch/Torchvision are needed for native pixel checks')
    def test_native_pixels_preserve_source_values_and_correct_height_width(self):
        with patch.object(midnight_base, 'load_model', side_effect=AssertionError('No model construction')):
            for height, top in [(460, 6), (456, 4)]:
                raw = np.arange(height*700*3, dtype=np.uint32).astype(np.uint8).reshape(height, 700, 3)
                pixels, geometry = midnight_native.native_pixels(Image.fromarray(raw, 'RGB'))
                expected = (torch.from_numpy(raw[top:top+448].copy()).permute(2, 0, 1).float()/255-.5)/.5
                self.assertEqual(pixels.shape, (3, 448, 700)); self.assertTrue(torch.equal(pixels, expected))
                self.assertEqual(geometry['crop_box'], [0, top, 700, top+448])
                self.assertAlmostEqual(pixels.numel()/(height*700*3), 448/height)
            with self.assertRaises(ValueError):
                midnight_native.native_pixels(Image.new('RGB', (448, 700)))

    @unittest.skipIf(torch is None, 'Torch is needed for token-array checks')
    def test_native_token_helper_requires_1601_and_preserves_old_257_helper(self):
        tokens = torch.zeros(1, 1601, 1536); tokens[:, 0] = 3; tokens[:, 1:] = 7
        result = midnight_native.classification_features(tokens)
        self.assertEqual(result.shape, (1, 3072))
        self.assertTrue(torch.equal(result[:, :1536], torch.full((1, 1536), 3.)))
        self.assertTrue(torch.equal(result[:, 1536:], torch.full((1, 1536), 7.)))
        for invalid in (tokens[:, :257], tokens.double(), tokens[:, :, :100]):
            with self.assertRaises(ValueError): midnight_native.classification_features(invalid)
        with self.assertRaises(ValueError): midnight_base.classification_features(tokens)
        self.assertEqual(midnight_base.classification_features(tokens[:, :257]).shape, (1, 3072))
        tokens[0, 10, 10] = float('nan')
        with self.assertRaises(ValueError): midnight_native.classification_features(tokens)

    def test_native_model_contract_preserves_518_configuration_and_frozen_flags(self):
        config = SimpleNamespace(image_size=518, patch_size=14, hidden_size=1536,
                                 num_attention_heads=24, num_hidden_layers=40, _attn_implementation='eager')
        parameter = SimpleNamespace(requires_grad=False)
        model = SimpleNamespace(training=False, config=config, parameters=lambda: iter([parameter]))
        midnight_native.check_model(model)
        config.image_size = 448
        with self.assertRaises(ValueError): midnight_native.check_model(model)
        config.image_size = 518; parameter.requires_grad = True
        with self.assertRaises(ValueError): midnight_native.check_model(model)

    def test_native_injected_encoder_uses_ordered_batches_of_four(self):
        calls = []
        def tiny(values):
            calls.append(len(values)); return np.repeat(values.astype(np.float32), 3072, axis=1)
        output = midnight_native.encode_batches(tiny, np.arange(9, dtype=np.float32)[:, None])
        self.assertEqual(calls, [4, 4, 1]); np.testing.assert_array_equal(output[:, 0], np.arange(9))
        with self.assertRaises(ValueError): midnight_native.encode_batches(tiny, np.zeros((2, 1)), 8)
        with self.assertRaises(ValueError): midnight_native.encode_batches(lambda x: np.zeros((len(x), 1601), np.float32), np.zeros((2, 1)))

    def test_native_roundtrip_overwrite_and_rehashed_provenance_tampering(self):
        config = _native_configuration(); paths = ['fixture/a', 'fixture/b']; features = np.ones((2, 3072), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'midnight-native-v1-40.npz'
            midnight_native.write_output(target, paths, features, _native_record(config, 40, 2))
            midnight_native.verify_output(target, paths, config, code_sha='a'*40)
            original = target.read_bytes()
            with self.assertRaises(ValueError): midnight_native.write_output(target, paths, features*2, _native_record(config, 40, 2))
            self.assertEqual(target.read_bytes(), original)
            meta = target.with_suffix('.json'); record = json.loads(meta.read_text()); record['token_shape'] = [257, 1536]
            meta.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                midnight_native.verify_output(target, paths, config, code_sha='a'*40)
            record['token_shape'] = [1601, 1536]; record['parity_atol'] = .1
            meta.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                midnight_native.verify_output(target, paths, config, code_sha='a'*40)

    def test_native_atomic_failure_preserves_foreign_partial_file(self):
        config = _native_configuration(); features = np.zeros((1, 3072), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'midnight-native-v1-40.npz'; link = midnight_native.os.link; counter = []
            def fail_second(source, destination):
                counter.append(destination)
                if len(counter) == 2: raise OSError('synthetic interrupted publication')
                link(source, destination)
            with patch.object(midnight_native.os, 'link', side_effect=fail_second), self.assertRaises(OSError):
                midnight_native.write_output(target, ['fixture'], features, _native_record(config, 40, 1))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            partial = target.with_suffix('.npz.part'); partial.write_bytes(b'other attempt')
            with self.assertRaises(FileExistsError):
                midnight_native.write_output(target, ['fixture'], features, _native_record(config, 40, 1))
            self.assertEqual(partial.read_bytes(), b'other attempt')

    def test_native_cpu_auditor_all_shards_and_extra_file_rejection(self):
        config = _native_configuration(); config['expected_images'] = 8; config['expected_shapes'] = {'700x460': 4, '700x456': 4}
        rows = [{'relative_path': 'fixture/%d/%d.png'%(mag, index), 'magnification': mag,
                 'subtype': 'adenosis', 'width': 700, 'height': height}
                for mag in MAGNIFICATIONS for index, height in enumerate((460, 456))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            config['manifest_sha256'] = file_hash(manifest); cache = root/'cache'; cache.mkdir()
            for mag in MAGNIFICATIONS:
                paths = [row['relative_path'] for row in rows if row['magnification'] == mag]
                midnight_native.write_output(cache/('midnight-native-v1-%d.npz'%mag), paths, np.full((2, 3072), mag, np.float32), _native_record(config, mag, 2))
            with patch.object(native_input_audit.producer, 'read_config', return_value=config):
                result = native_input_audit.audit(MIDNIGHT_NATIVE_ROOT/'configs/midnight-native-v1.json', manifest, cache, 'a'*40)
                self.assertEqual(result['images'], 8); self.assertEqual(len(result['shards']), 4)
                self.assertEqual(result['status'], 'accepted all four frozen native-input shards')
                np.savez(cache/'extra.npz', features=np.zeros(1))
                with self.assertRaisesRegex(ValueError, 'artifact matrix'):
                    native_input_audit.audit(MIDNIGHT_NATIVE_ROOT/'configs/midnight-native-v1.json', manifest, cache, 'a'*40)


if __name__=='__main__':unittest.main()

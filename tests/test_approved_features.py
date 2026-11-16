import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from magset import approved_features as producer
from magset.data import MAGNIFICATIONS, file_hash
try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('approved_feature_audit_fixture', ROOT/'scripts/audit_approved_features.py')
auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
NAMES = ('uni2-h-v1', 'virchow2-v1', 'h-optimus-0-v1')


def config(name='uni2-h-v1'):
    return json.loads((ROOT/'configs'/(name+'.json')).read_text())


def record(cfg, paths, scale=40, limit=0):
    out = producer.metadata(cfg, paths, scale, limit, 'a'*40)
    out.update({'loader_receipt': {'strict_full_state': True, 'exact_parameter_copy': True,
                'state_tensors': 2, 'parameter_count': 3, 'checkpoint_parameter_dtypes': ['torch.float32']},
                'geometry_counts': {'700x460': len(paths)//2, '700x456': len(paths)-len(paths)//2},
                'parity': {} if not limit else dict.fromkeys(('repeat_max_abs', 'single_batch_max_abs', 'reconstructed_model_max_abs'), 0.),
                'elapsed_seconds': 1., 'steady_batch_seconds': .1 if limit else None,
                'peak_gpu_bytes': 1, 'peak_reserved_gpu_bytes': 2})
    return out


class ApprovedArtifactTests(unittest.TestCase):
    def test_all_three_registered_configs_and_metadata(self):
        for name in NAMES:
            cfg = producer.read_config(ROOT/'configs'/(name+'.json'))
            for limit, count in ((0, 2), (4, 16)):
                out = producer.metadata(cfg, ['a', 'b'], 40, limit, 'a'*40)
                self.assertEqual(out['image_forwards'], count)
                self.assertEqual(out['output_dimension'], cfg['encoder']['dimension'])
                self.assertEqual(out['host_batch_size'], 4)
                self.assertEqual(out['model_image_batch_size'], 1)
                self.assertEqual(out['precision'], producer.PRECISION)
                self.assertEqual(out['readout'], cfg['readout'])
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)/'tamper.json'; altered = copy.deepcopy(cfg)
                altered['parity']['atol'] = 1.; target.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, 'configuration bytes'):
                    producer.read_config(target)
        self.assertEqual(config('h-optimus-0-v1')['architecture']['kwargs']['img_size'], 224)

    def test_hashed_smoke_selection_preserves_order_and_geometry(self):
        cfg = config(); rows = [{'relative_path': 'fixture/%d'%i, 'width': 700,
            'height': 460 if i < 7835 else 456, 'magnification': 40} for i in range(7909)]
        selected = [rows[i] for i in (1, 3, 7836, 7838)]
        cfg['smoke']['relative_path_sha256'] = [hashlib.sha256(row['relative_path'].encode()).hexdigest() for row in selected]
        with patch.object(producer, 'file_hash', return_value=cfg['manifest_sha256']), patch.object(producer, 'read_manifest', return_value=rows):
            _, actual = producer.select_rows('manifest', cfg, 40, 4)
            self.assertEqual(actual, selected)
            self.assertNotIn('relative_paths', cfg['smoke'])
            cfg['smoke']['relative_path_sha256'][0] = '0'*64
            with self.assertRaisesRegex(ValueError, 'smoke path is absent'):
                producer.select_rows('manifest', cfg, 40, 4)
            rows[0]['relative_path'] = '../escape'
            with self.assertRaisesRegex(ValueError, 'escapes root'):
                producer.select_rows('manifest', cfg, 40, 0)
        with self.assertRaises(ValueError): producer.select_rows('manifest', cfg, 100, 4)

    def test_local_artifact_verification_rejects_missing_changed_and_escaping_pins(self):
        cfg = config(); data = b'private checkpoint bytes'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'pytorch_model.bin'; path.write_bytes(data)
            pin = {'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            cfg['artifacts'] = {'pytorch_model.bin': pin}; cfg['encoder'].update(weights_size_bytes=len(data), weights_sha256=pin['sha256'])
            self.assertTrue(producer.verify_files(root, cfg)); path.write_bytes(data+b'bad')
            with self.assertRaisesRegex(ValueError, 'changed pinned'): producer.verify_files(root, cfg)
            cfg['artifacts']['../outside'] = pin
            with self.assertRaisesRegex(ValueError, 'Invalid local artifact pin'): producer.artifact_pins(cfg)

    def test_roundtrip_and_rehashed_provenance_and_payload_tamper(self):
        for name in NAMES:
            cfg = config(name); paths = ['fixture/a', 'fixture/b']; dim = cfg['output_dimension']
            features = np.arange(2*dim, dtype=np.float32).reshape(2, dim)
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)/(name+'-40.npz')
                producer.write_output(target, paths, features, record(cfg, paths))
                producer.verify_output(target, paths, cfg, code_sha='a'*40)
                with self.assertRaisesRegex(ValueError, 'existing'): producer.write_output(target, paths, features, record(cfg, paths))
                meta = target.with_suffix('.json'); original = json.loads(meta.read_text())
                for key, value in (('readout', 'incorrect'), ('model_image_batch_size', 4), ('precision', 'FP16'),
                                   ('elapsed_seconds', 0), ('code_sha', 'b'*40), ('image_forwards', 8)):
                    altered = dict(original); altered[key] = value; meta.write_text(json.dumps(altered))
                    with self.assertRaises(ValueError): producer.verify_output(target, paths, cfg, code_sha='a'*40)
                changed = features.copy(); changed[0, 0] = np.nan
                np.savez_compressed(target, paths=np.asarray(paths), features=changed)
                original['feature_sha256'] = file_hash(target); meta.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, 'finite coverage'): producer.verify_output(target, paths, cfg)

    def test_interrupted_publication_rolls_back_owned_files_and_preserves_foreign_partial(self):
        cfg = config(); paths = ['fixture']; features = np.zeros((1, 1536), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'approved.npz'; real_link = producer.os.link; calls = []
            def fail_second(source, destination):
                calls.append(destination)
                if len(calls) == 2: raise OSError('interrupted')
                real_link(source, destination)
            with patch.object(producer.os, 'link', side_effect=fail_second), self.assertRaises(OSError):
                producer.write_output(target, paths, features, record(cfg, paths))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            partial = target.with_suffix('.npz.part'); partial.write_bytes(b'foreign')
            with self.assertRaises(FileExistsError): producer.write_output(target, paths, features, record(cfg, paths))
            self.assertEqual(partial.read_bytes(), b'foreign')

    def test_cpu_audit_requires_complete_exact_four_shards_and_counts(self):
        cfg = config('virchow2-v1'); cfg['expected_images'] = 8; cfg['expected_shapes'] = {'700x460': 4, '700x456': 4}
        rows = [{'relative_path': 'fixture/%d/%d'%(mag, i), 'magnification': mag, 'subtype': 'adenosis',
                 'width': 700, 'height': h} for mag in MAGNIFICATIONS for i, h in enumerate((460, 456))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            cfg['manifest_sha256'] = file_hash(manifest); cache = root/'cache'; cache.mkdir()
            for mag in MAGNIFICATIONS:
                paths = [row['relative_path'] for row in rows if row['magnification'] == mag]
                producer.write_output(cache/('%s-%d.npz'%(cfg['name'], mag)), paths, np.full((2, 2560), mag, np.float32), record(cfg, paths, mag))
            with patch.object(producer, 'read_config', return_value=cfg):
                out = auditor.audit(ROOT/'configs/virchow2-v1.json', manifest, cache, 'a'*40)
                self.assertEqual(out['status'], auditor.STATUS); self.assertEqual(out['images'], 8)
                self.assertEqual(out['image_forwards'], 8); self.assertEqual(out['new_encoder_forwards'], 0)
                self.assertEqual(out['output_dimension'], 2560); self.assertEqual(out['image_decodes'], 0)
                np.savez(cache/'extra.npz', features=np.zeros(1))
                with self.assertRaisesRegex(ValueError, 'artifact matrix'): auditor.audit(ROOT/'configs/virchow2-v1.json', manifest, cache, 'a'*40)


@unittest.skipIf(torch is None, 'Torch and Torchvision needed for tensor checks')
class ApprovedTensorTests(unittest.TestCase):
    def test_transform_matches_pil_geometry_and_normalization_for_every_encoder(self):
        for name in NAMES:
            cfg = config(name); spec = cfg['encoder']['transform']
            interpolation = Image.Resampling.BILINEAR if spec['interpolation'] == 'bilinear' else Image.Resampling.BICUBIC
            for height in (460, 456):
                raw = np.arange(height*700*3, dtype=np.uint32).astype(np.uint8).reshape(height, 700, 3)
                image = Image.fromarray(raw, 'RGB'); pixels, geometry = producer.approved_pixels(image, cfg)
                width = int(700*224/height); resized = image.resize((width, 224), interpolation)
                left = int(round((width-224)/2)); cropped = resized.crop((left, 0, left+224, 224))
                expected = transforms.Normalize(spec['mean'], spec['std'])(transforms.ToTensor()(cropped))
                self.assertTrue(torch.equal(pixels, expected)); self.assertTrue(pixels.is_contiguous())
                self.assertEqual(geometry['resized'], [width, 224]); self.assertEqual(geometry['crop_box'], [left, 0, left+224, 224])
            with self.assertRaises(ValueError): producer.approved_pixels(Image.new('L', (700, 460)), cfg)

    def test_official_virchow_readout_excludes_registers(self):
        cfg = config('virchow2-v1')
        class Tokens(torch.nn.Module):
            def forward(self, pixels):
                out = torch.full((len(pixels), 261, 1280), 7., dtype=torch.float32)
                out[:, 0] = 2.; out[:, 1:5] = 1000000.
                return out
        actual = producer.classification_features(Tokens(), torch.zeros(2, 3, 224, 224), cfg)
        self.assertTrue(torch.equal(actual[:, :1280], torch.full((2, 1280), 2.)))
        self.assertTrue(torch.equal(actual[:, 1280:], torch.full((2, 1280), 7.)))
        with self.assertRaises(ValueError): producer.classification_features(lambda _: torch.zeros(1, 257, 1280), torch.zeros(1, 3, 224, 224), cfg)

    def test_canonical_calls_are_host_partition_invariant_for_all_readouts(self):
        pixels = torch.stack([torch.full((3, 224, 224), float(i)) for i in range(4)])
        for name in NAMES:
            cfg = config(name); calls = []
            def model(value):
                calls.append(len(value)); output = value.mean((1, 2, 3))
                shape = (len(value), *cfg['expected_token_shape']) if cfg['expected_token_shape'] else (len(value), cfg['output_dimension'])
                return output.reshape(len(value), *([1]*(len(shape)-1))).expand(shape).clone() + (1. if len(value) > 1 else 0.)
            expected = producer.classification_features(model, pixels, cfg)
            for partition in ((1, 3), (2, 2), (3, 1), (1, 1, 1, 1)):
                rows = []; start = 0
                for count in partition:
                    rows.append(producer.classification_features(model, pixels[start:start+count], cfg)); start += count
                self.assertTrue(torch.equal(torch.cat(rows), expected))
            self.assertEqual(set(calls), {1}); self.assertEqual(expected[:, 0].tolist(), [0., 1., 2., 3.])
            with self.assertRaises(ValueError): producer.classification_features(model, pixels.double(), cfg)
            with self.assertRaises(ValueError): producer.classification_features(model, pixels.transpose(2, 3), cfg)

    def test_strict_load_materializes_meta_casts_fp32_and_rejects_missing_shape_and_nan(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(2)); self.register_buffer('integer', torch.zeros(1, dtype=torch.int64))
        state = {'weight': torch.tensor([2., 4.], dtype=torch.float16), 'integer': torch.tensor([3])}
        with torch.device('meta'): model = Tiny()
        loaded, receipt = producer.strict_load(model, state)
        self.assertEqual(loaded.weight.tolist(), [2, 4]); self.assertEqual(loaded.integer.tolist(), [3])
        self.assertFalse(loaded.weight.requires_grad); self.assertFalse(loaded.training)
        self.assertEqual(loaded.weight.dtype, torch.float32); self.assertEqual(receipt['checkpoint_parameter_dtypes'], ['torch.float16'])
        for broken in ({'weight': state['weight']}, dict(state, weight=torch.zeros(3)), dict(state, weight=torch.tensor([1., float('nan')]))):
            with torch.device('meta'): model = Tiny()
            with self.assertRaises(ValueError): producer.strict_load(model, broken)

    def test_safe_local_bin_loader_supports_meta_scalar_schedule_and_freezes_attention(self):
        class Tiny(torch.nn.Module):
            num_features = 1536
            def __init__(self):
                super().__init__(); self.schedule = [v.item() for v in torch.linspace(0., .2, 3)]
                self.weight = torch.nn.Parameter(torch.zeros(2)); self.fused_attn = True
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pytorch_model.bin'; torch.save({'weight': torch.tensor([3., 7.])}, path)
            with patch.dict('sys.modules', timm=SimpleNamespace()), patch.object(producer, 'verify_files', return_value=True), patch.object(producer, '_construction', side_effect=lambda _: Tiny()):
                model = producer.load_model(tmp, torch.device('cpu'), cfg)
            self.assertEqual(model.weight.tolist(), [3, 7]); self.assertFalse(model.fused_attn)
            self.assertFalse(model.training); self.assertEqual(model.schedule, [v.item() for v in torch.linspace(0., .2, 3)])
            self.assertTrue(model._approved_loader_receipt['strict_full_state'])
            self.assertFalse(model.weight.is_meta)
        self.assertEqual(torch.linspace(0, 1, 2).device.type, 'cpu')

    def test_trusted_constructor_never_uses_hub_or_remote_code(self):
        cfg = config('virchow2-v1'); sentinel = object(); calls = []
        fake = SimpleNamespace(layers=SimpleNamespace(SwiGLUPacked=sentinel),
            create_model=lambda *args, **kwargs: calls.append((args, kwargs)) or 'constructed')
        with patch.dict('sys.modules', timm=fake):
            self.assertEqual(producer._construction(cfg), 'constructed')
            args, kwargs = calls[0]
            self.assertEqual(args, ('vit_huge_patch14_224',)); self.assertIs(kwargs['mlp_layer'], sentinel)
            self.assertIs(kwargs['act_layer'], torch.nn.SiLU); self.assertFalse(kwargs['pretrained'])
            self.assertEqual(kwargs['global_pool'], '')
            cfg['architecture']['kwargs']['checkpoint_path'] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'must not fetch'): producer._construction(cfg)

    def test_encoder_order_and_predeclared_numeric_drift_rejection(self):
        cfg = config(); calls = []
        def encode(value):
            calls.append(len(value)); return np.repeat(value.astype(np.float32), 1536, axis=1)
        out = producer.encode_batches(encode, np.arange(9, dtype=np.float32)[:, None], cfg)
        self.assertEqual(calls, [4, 4, 1]); np.testing.assert_array_equal(out[:, 0], np.arange(9))
        self.assertEqual(producer.assert_parity(out, out.copy(), cfg), 0.)
        with self.assertRaisesRegex(ValueError, 'tolerances'): producer.assert_parity(out, out+1, cfg)
        with self.assertRaises(ValueError): producer.encode_batches(encode, np.zeros((1, 1)), cfg, 8)


if __name__ == '__main__':
    unittest.main()

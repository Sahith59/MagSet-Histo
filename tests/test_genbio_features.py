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
from magset import genbio_features as producer
from magset import coverage_features as coverage
from magset.data import MAGNIFICATIONS, file_hash
try:
    import torch
    import torchvision
    from safetensors.torch import save_file
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/genbio-pathfm-v1.json'
spec = importlib.util.spec_from_file_location('genbio_audit_fixture', ROOT/'scripts/audit_genbio_features.py')
auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)


def configuration():
    return json.loads(CONFIG.read_text())


def record(config, paths, scale=40, limit=0):
    out = producer.metadata(config, paths, scale, limit, 'a'*40)
    out.update({'loader_receipt': {'strict_full_state': True, 'exact_parameter_copy': True,
                                  'rope_periods_dtype': 'torch.bfloat16', 'state_tensors': 2, 'parameter_count': 1},
                'geometry_counts': {'700x460': len(paths)//2, '700x456': len(paths)-len(paths)//2},
                'parity': {} if not limit else dict.fromkeys(['repeat_max_abs', 'single_batch_max_abs', 'reconstructed_model_max_abs'], 0.),
                'elapsed_seconds': 1., 'steady_batch_seconds': .1 if limit else None,
                'peak_gpu_bytes': 1, 'peak_reserved_gpu_bytes': 2})
    return out


class GenBioArtifactTests(unittest.TestCase):
    def test_registered_config_and_metadata_dimension_forward_accounting(self):
        config = producer.read_config(CONFIG)
        self.assertEqual(config['encoder']['dimension'], 4608)
        for limit, count in [(0, 2), (4, 16)]:
            out = producer.metadata(config, ['a', 'b'], 40, limit, 'a'*40)
            self.assertEqual(out['image_forwards'], count)
            self.assertEqual(out['channel_forwards'], count*3)
            self.assertEqual(out['token_shape_per_channel'], [201, 1536])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'other.json'; p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration bytes'): producer.read_config(p)

    def test_complete_manifest_and_two_height_label_blind_smoke(self):
        config = configuration()
        rows = [{'relative_path': 'fixture/%d'%i, 'width': 700, 'height': 460 if i < 7835 else 456, 'magnification': 40} for i in range(7909)]
        with patch.object(coverage, 'file_hash', return_value=config['manifest_sha256']), patch.object(coverage, 'read_manifest', return_value=rows):
            _, selected = producer.select_rows(Path('fixture'), config, 40, 4)
            self.assertEqual([x['height'] for x in selected], [460, 460, 456, 456])
            rows[0]['relative_path'] = '../escaped'
            with self.assertRaises(ValueError): producer.select_rows(Path('fixture'), config, 40, 0)

    def test_byte_verified_local_import_supports_relative_import_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bodies = {'configuration_genbio_pathfm.py': b'VALUE=17\n',
                      'modeling_genbio_pathfm.py': b'from .configuration_genbio_pathfm import VALUE\nRESULT=VALUE+1\n'}
            pins = {}
            for name, body in bodies.items():
                (root/name).write_bytes(body); pins[name] = (len(body), hashlib.sha256(body).hexdigest())
            with patch.dict(producer.ARTIFACTS, pins):
                a, b = producer.import_local_sources(root)
                self.assertEqual(a.VALUE, 17); self.assertEqual(b.RESULT, 18)
                (root/'modeling_genbio_pathfm.py').write_bytes(b'raise AssertionError("must not execute")\n')
                with self.assertRaisesRegex(ValueError, 'source differs'): producer.import_local_sources(root)

    def test_roundtrip_overwrite_and_rehashed_provenance_corruptions(self):
        config = configuration(); paths = ['fixture/a', 'fixture/b']; values = np.arange(2*4608, dtype=np.float32).reshape(2, 4608)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'genbio-pathfm-v1-40.npz'
            producer.write_output(path, paths, values, record(config, paths))
            producer.verify_output(path, paths, config, code_sha='a'*40)
            with self.assertRaises(ValueError): producer.write_output(path, paths, values, record(config, paths))
            meta = path.with_suffix('.json'); original = json.loads(meta.read_text())
            for key, value in [('channels_per_image', 1), ('output_dimension', 1536), ('official_rope_dtype', 'float32'), ('channel_forwards', 2), ('elapsed_seconds', 0.)]:
                altered = dict(original); altered[key] = value; meta.write_text(json.dumps(altered))
                with self.assertRaises(ValueError): producer.verify_output(path, paths, config, code_sha='a'*40)
            meta.write_text(json.dumps(original)); changed = values.copy(); changed[0, 0] = np.nan
            np.savez_compressed(path, paths=np.asarray(paths), features=changed)
            original['feature_sha256'] = file_hash(path); meta.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'finite coverage'): producer.verify_output(path, paths, config)

    def test_atomic_interrupted_publication_and_foreign_partial_preserved(self):
        config = configuration(); paths = ['fixture']; values = np.zeros((1, 4608), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'genbio.npz'; link = producer.os.link; calls = []
            def fail_second(src, dst):
                calls.append(dst)
                if len(calls) == 2: raise OSError('interrupted')
                link(src, dst)
            with patch.object(producer.os, 'link', side_effect=fail_second), self.assertRaises(OSError):
                producer.write_output(target, paths, values, record(config, paths))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            partial = target.with_suffix('.npz.part'); partial.write_bytes(b'foreign')
            with self.assertRaises(FileExistsError): producer.write_output(target, paths, values, record(config, paths))
            self.assertEqual(partial.read_bytes(), b'foreign')

    def test_complete_cpu_auditor_checks_all_four_archives_and_no_extra_file(self):
        config = configuration(); config['expected_images'] = 8; config['expected_shapes'] = {'700x460': 4, '700x456': 4}
        rows = [{'relative_path': 'fixture/%d/%d'%(mag, i), 'magnification': mag, 'subtype': 'adenosis', 'width': 700, 'height': h} for mag in MAGNIFICATIONS for i, h in enumerate((460, 456))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as stream:
                w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            config['manifest_sha256'] = file_hash(manifest); cache = root/'cache'; cache.mkdir()
            for mag in MAGNIFICATIONS:
                paths = [r['relative_path'] for r in rows if r['magnification'] == mag]
                producer.write_output(cache/('genbio-pathfm-v1-%d.npz'%mag), paths, np.full((2, 4608), mag, np.float32), record(config, paths, mag))
            with patch.object(producer, 'read_config', return_value=config):
                out = auditor.audit(CONFIG, manifest, cache, 'a'*40)
                self.assertEqual(out['images'], 8); self.assertEqual(out['channel_forwards'], 24)
                self.assertEqual(out['new_encoder_forwards'], 0)
                np.savez(cache/'extra.npz', features=np.zeros(1))
                with self.assertRaisesRegex(ValueError, 'artifact matrix'): auditor.audit(CONFIG, manifest, cache, 'a'*40)


@unittest.skipIf(torch is None, 'Pinned Torch/Torchvision/Safetensors needed for tiny GenBio checks')
class GenBioTensorTests(unittest.TestCase):
    def test_whole_square_transform_matches_official_pil_resize_and_channel_norm(self):
        from torchvision import transforms
        for height in (460, 456):
            raw = np.arange(height*700*3, dtype=np.uint32).astype(np.uint8).reshape(height, 700, 3)
            image = Image.fromarray(raw, 'RGB'); pixels, geometry = producer.genbio_pixels(image)
            resized = image.resize((224, 224), Image.Resampling.BILINEAR)
            reference = transforms.Normalize((.697, .575, .728), (.188, .240, .187))(transforms.ToTensor()(resized))
            self.assertTrue(torch.equal(pixels, reference)); self.assertTrue(pixels.is_contiguous())
            self.assertIsNone(geometry['crop_box']); self.assertEqual(geometry['original'], [700, height])

    def test_official_channel_reshape_order_and_output_validation(self):
        class TinyOfficialForward(torch.nn.Module):
            def forward(self, x):
                b, _, h, w = x.shape
                channels = x.view(b*3, 1, h, w)
                cls = channels.mean((1, 2, 3))[:, None].repeat(1, 1536).view(b, 3, 1536)
                return torch.cat([cls[:, 0], cls[:, 1], cls[:, 2]], -1)
        pixels = torch.stack([torch.stack([torch.full((224, 224), float(v)) for v in triple]) for triple in [(1, 2, 3), (4, 5, 6)]])
        values = producer.classification_features(TinyOfficialForward(), pixels)
        self.assertEqual(values[:, ::1536].tolist(), [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError): producer.classification_features(lambda _: torch.zeros(2, 1536), pixels)
        with self.assertRaises(ValueError): producer.classification_features(lambda _: torch.full((2, 4608), float('nan')), pixels)
        with self.assertRaises(ValueError): producer.classification_features(TinyOfficialForward(), pixels.transpose(2, 3))
        with self.assertRaises(ValueError): producer.classification_features(TinyOfficialForward(), pixels.double())

    def test_tiny_strict_loader_copies_all_parameters_preserves_bf16_buffer_and_rejects_keys(self):
        class Tiny(torch.nn.Module):
            def __init__(self, config):
                super().__init__(); self.config = config; self.weight = torch.nn.Parameter(torch.zeros(2))
                self.backbone = torch.nn.Module(); self.backbone.rope_embed = torch.nn.Module()
                self.backbone.rope_embed.register_buffer('periods', torch.zeros(2, dtype=torch.bfloat16))
        cfg = dict(producer.ARCHITECTURE); ctor = lambda **x: SimpleNamespace(**x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.safetensors'; tiny = Tiny(ctor(**cfg)); state = tiny.state_dict()
            state['weight'] = torch.tensor([3., 7.]); state['backbone.rope_embed.periods'] = torch.tensor([2., 4.], dtype=torch.bfloat16)
            save_file(state, path)
            with patch.object(producer, 'verify_files', return_value=cfg), patch.object(producer, 'import_local_sources', return_value=(SimpleNamespace(GenBioPathFMConfig=ctor), SimpleNamespace(GenBioPathFMModel=Tiny))):
                loaded = producer.load_model(tmp, torch.device('cpu'))
                self.assertEqual(loaded.weight.tolist(), [3, 7]); self.assertFalse(loaded.weight.requires_grad)
                self.assertEqual(loaded.backbone.rope_embed.periods.dtype, torch.bfloat16)
                self.assertTrue(loaded._genbio_loader_receipt['exact_parameter_copy'])
                self.assertFalse(loaded.training)
                del state['weight']; save_file(state, path)
                with self.assertRaisesRegex(ValueError, 'full-state keys'): producer.load_model(tmp, torch.device('cpu'))

    def test_batch_order_and_parity_rejects_actual_numeric_drift(self):
        calls = []
        def tiny(x):
            calls.append(len(x)); return np.repeat(x.astype(np.float32), 4608, axis=1)
        values = producer.encode_batches(tiny, np.arange(9, dtype=np.float32)[:, None])
        self.assertEqual(calls, [4, 4, 1]); np.testing.assert_array_equal(values[:, 0], np.arange(9))
        with self.assertRaises(ValueError): producer.base.assert_parity(values, values+1.)
        with self.assertRaises(ValueError): producer.encode_batches(tiny, np.zeros((1, 1)), 8)


if __name__ == '__main__':
    unittest.main()

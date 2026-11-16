"""Small fixtures only: no real checkpoint, network call, or billion-parameter model."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from magset import midnight_features as mf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_midnight_fixture', ROOT / 'scripts/prepare_midnight.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@unittest.skipIf(torch is None, 'PyTorch is not installed in the lightweight environment')
class MidnightFixtures(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/midnight-12k-v1.json').read_text())

    def test_canonical_schema_and_critical_mutations(self):
        self.assertEqual(mf.validate_config(self.config), self.config)
        mutations = [('revision', 'main'), ('dimension', 1536), ('weights_size_bytes', 0),
                     ('config_sha256', '0' * 64), ('transform', {})]
        for name, value in mutations:
            config = copy.deepcopy(self.config)
            config['encoder'][name] = value
            with self.assertRaises(ValueError):
                mf.validate_config(config)
        config = copy.deepcopy(self.config)
        config['expected_images'] = 7908
        with self.assertRaises(ValueError):
            mf.validate_config(config)

    def test_official_rgb_normalization_and_center_crop(self):
        operation = mf.build_transform()
        image = Image.new('RGB', (700, 460), color=(255, 0, 127))
        pixels = operation(image)
        self.assertEqual(pixels.shape, (3, 224, 224))
        self.assertEqual(pixels.dtype, torch.float32)
        self.assertTrue(torch.allclose(pixels[:, 0, 0], torch.tensor([1., -1., -1. / 255]), atol=1e-7))
        ramp = np.zeros((460, 700, 3), dtype=np.uint8)
        ramp[:, :, 0] = np.linspace(0, 255, 700, dtype=np.uint8)
        cropped = operation(Image.fromarray(ramp))
        # A centered square excludes the horizontal edges; a distorted square resize does not.
        self.assertGreater(float(cropped[0, 0, 0]), -.8)
        self.assertLess(float(cropped[0, 0, 0]), -.5)

    def test_cls_plus_patch_mean_and_rejection(self):
        tokens = torch.zeros(2, 257, 1536)
        tokens[:, 0] = 3
        tokens[:, 1:] = 7
        features = mf.classification_features(tokens)
        self.assertEqual(features.shape, (2, 3072))
        self.assertTrue(torch.equal(features[:, :1536], torch.full((2, 1536), 3.)))
        self.assertTrue(torch.equal(features[:, 1536:], torch.full((2, 1536), 7.)))
        for wrong in (tokens[:, :256], tokens.double(), tokens[:, :, :1024]):
            with self.assertRaises(ValueError):
                mf.classification_features(wrong)
        tokens[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            mf.classification_features(tokens)

    def test_strict_state_assignment_tiny_meta_fixture(self):
        original = torch.nn.Linear(3, 2)
        with torch.device('meta'):
            model = torch.nn.Linear(3, 2)
        mf.strict_load(model, original.state_dict())
        self.assertFalse(model.training)
        self.assertTrue(all(not p.is_meta and not p.requires_grad for p in model.parameters()))
        self.assertTrue(torch.equal(model.weight, original.weight))
        with self.assertRaises(RuntimeError):
            mf.strict_load(torch.nn.Linear(3, 2), {'weight': original.weight})
        with self.assertRaises(RuntimeError):
            mf.strict_load(torch.nn.Linear(4, 2), original.state_dict())
        with self.assertRaises(ValueError):
            mf.strict_load(torch.nn.Linear(3, 2), {k: v.double() for k, v in original.state_dict().items()})

    def test_fixed_batch_parity(self):
        x = np.ones((4, 3072), dtype=np.float32)
        self.assertEqual(mf.assert_parity(x, x.copy()), 0)
        with self.assertRaises(ValueError):
            mf.assert_parity(x, x + .01)
        with self.assertRaises(ValueError):
            mf.assert_parity(x, x[:1])

    def test_archive_roundtrip_coverage_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'midnight-12k-40.npz'
            features = np.arange(2 * 3072, dtype=np.float32).reshape(2, 3072)
            paths = ['fixture/a.png', 'fixture/b.png']
            record = {'encoder': self.config['encoder'], 'manifest_sha256': 'a' * 64,
                      'config_sha256': 'b' * 64, 'smoke_limit': 4, 'images': 2}
            mf.write_output(target, paths, features, record)
            mf.verify_output(target, target.with_suffix('.json'), paths, self.config['encoder'],
                             'a' * 64, 4, 'b' * 64)
            with self.assertRaises(ValueError):
                mf.write_output(target, paths, features, record)
            with self.assertRaises(ValueError):
                mf.verify_output(target, target.with_suffix('.json'), paths[::-1], self.config['encoder'],
                                 'a' * 64, 4, 'b' * 64)
            with self.assertRaises(ValueError):
                mf.verify_output(target, target.with_suffix('.json'), paths, self.config['encoder'],
                                 'a' * 64, 0, 'b' * 64)

    def test_manifest_order_and_path_confinement(self):
        rows = [{'relative_path': 'fixture/%d.png' % i, 'magnification': 40} for i in range(7909)]
        with patch.object(mf, 'read_manifest', return_value=rows), patch.object(mf, 'file_hash', return_value=self.config['manifest_sha256']):
            selected = mf.select_rows(Path('fixture.csv'), self.config, 40, 4)
            self.assertEqual(selected, rows[:4])
            with self.assertRaises(ValueError):
                mf.select_rows(Path('fixture.csv'), self.config, 40, 5)
            rows[0]['relative_path'] = '../escape.png'
            with self.assertRaises(ValueError):
                mf.select_rows(Path('fixture.csv'), self.config, 40, 4)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(data))}


class DownloadFixtures(unittest.TestCase):
    def test_resume_validated_range_and_skip_verified_file(self):
        payload = b'fixture checkpoint bytes'
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'model.safetensors'
            target.with_name(target.name + '.part').write_bytes(payload[:5])
            def opener(request, timeout):
                self.assertEqual(request.get_header('Range'), 'bytes=5-')
                return Response(payload[5:], 206, {'Content-Range': 'bytes 5-%d/%d' % (len(payload)-1, len(payload)),
                                                   'Content-Length': str(len(payload)-5)})
            result = prepare.download('https://example.invalid/model', target, len(payload), digest, opener)
            self.assertFalse(result['already_present'])
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(target.with_name(target.name + '.part').exists())
            result = prepare.download('https://example.invalid/model', target, len(payload), digest,
                                      lambda *args, **kwargs: self.fail('verified artifact contacted network'))
            self.assertTrue(result['already_present'])

    def test_ignored_range_restarts_instead_of_appending(self):
        payload = b'abcdef'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'model'
            target.with_name('model.part').write_bytes(b'abc')
            prepare.download('https://example.invalid/model', target, 6, hashlib.sha256(payload).hexdigest(),
                             lambda *args, **kwargs: Response(payload))
            self.assertEqual(target.read_bytes(), payload)

    def test_bad_range_and_wrong_hash_never_publish(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'model'
            target.with_name('model.part').write_bytes(b'ab')
            with self.assertRaises(ValueError):
                prepare.download('https://example.invalid/model', target, 6, '0' * 64,
                                 lambda *args, **kwargs: Response(b'cdef', 206,
                                    {'Content-Range': 'bytes 1-4/6', 'Content-Length': '4'}))
            self.assertFalse(target.exists())
            target.with_name('model.part').unlink()
            with self.assertRaises(ValueError):
                prepare.download('https://example.invalid/model', target, 6, '0' * 64,
                                 lambda *args, **kwargs: Response(b'abcdef'))
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()

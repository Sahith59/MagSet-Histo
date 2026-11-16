"""Synthetic pixel/cache fixtures without a Midnight model or network access."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from magset import coverage_features as m
from magset import midnight_features as base
from magset.data import MAGNIFICATIONS, file_hash

try:
    import torch
    import torchvision
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage_audit_fixture', ROOT/'scripts/audit_coverage_features.py')
audit_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit_module)


def configuration():
    return json.loads((ROOT/'configs/midnight-coverage-v1.json').read_text())


def record(config, scale, count, limit=0):
    receipt = config['base_cache_provenance'][str(scale)]
    return {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
            'config_sha256': m.CONFIG_SHA256, 'code_sha': 'a'*40,
            'source_file_sha256': file_hash(m.__file__), 'magnification': scale,
            'images': count, 'smoke_limit': limit, 'dependencies': config['dependencies'],
            'base_center_npz_sha256': receipt['npz_sha256'], 'base_center_json_sha256': receipt['json_sha256'],
            'base_feature_producer_code_sha': config['base_feature_producer_code_sha'],
            'base_extraction_config_sha256': config['base_extraction_config_sha256'],
            'base_classifier_config_sha256': config['base_classifier_config_sha256'],
            'transform': config['encoder']['transform'], 'frozen_base': True,
            'geometry_examples': config['geometry_examples'], 'new_crop_forwards': count*2,
            'geometry_counts': {'700x460': count//2, '700x456': count-count//2}}


class CoverageCacheTests(unittest.TestCase):
    def test_registered_configuration_and_base_bindings(self):
        config = m.read_config(ROOT/'configs/midnight-coverage-v1.json')
        self.assertEqual(config['encoder_name'], 'midnight-coverage-v1')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'changed.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration bytes'):
                m.read_config(path)

    def test_manifest_geometry_and_mixed_height_smoke_are_exact(self):
        config = configuration()
        rows = [{'relative_path': 'fixture/%d.png'%i, 'width': '700', 'height': '460' if i < 7835 else '456', 'magnification': '40'} for i in range(7909)]
        with patch.object(m, 'file_hash', return_value=config['manifest_sha256']), patch.object(m, 'read_manifest', return_value=rows):
            full, smoke = m.select_rows(Path('fixture.csv'), config, 40, 4)
            self.assertEqual([row['height'] for row in smoke], ['460', '460', '456', '456'])
            self.assertEqual([row['relative_path'] for row in smoke], ['fixture/0.png', 'fixture/1.png', 'fixture/7835.png', 'fixture/7836.png'])
            with self.assertRaises(ValueError): m.select_rows(Path('fixture.csv'), config, 100, 4)
            rows[0]['relative_path'] = '../escape.png'
            with self.assertRaises(ValueError): m.select_rows(Path('fixture.csv'), config, 40, 0)

    def test_all_base_shards_are_pinned_before_center_reuse(self):
        config = configuration(); extraction = json.loads((ROOT/'configs/midnight-12k-v1.json').read_text())
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scale in MAGNIFICATIONS:
                paths = ['fixture/%d/%d.png'%(scale, i) for i in range(2)]
                rows.extend({'relative_path': path, 'magnification': scale} for path in paths)
                path = root/('midnight-12k-%d.npz'%scale)
                np.savez_compressed(path, paths=np.asarray(paths), features=np.full((2, 3072), scale, np.float32))
                meta = {'feature_sha256': file_hash(path), 'encoder': extraction['encoder'],
                        'manifest_sha256': config['manifest_sha256'], 'config_sha256': config['base_extraction_config_sha256'],
                        'smoke_limit': 0, 'images': 2, 'magnification': scale,
                        'code_sha': config['base_feature_producer_code_sha'], 'source_file_sha256': file_hash(base.__file__),
                        'dependencies': config['dependencies']}
                path.with_suffix('.json').write_text(json.dumps(meta))
                config['base_cache_provenance'][str(scale)] = {'npz_sha256': file_hash(path), 'json_sha256': file_hash(path.with_suffix('.json'))}
            paths, values, meta = m.load_base_centers(config, root, rows, 40)
            self.assertEqual(values.shape, (2, 3072)); self.assertEqual(len(paths), 2)
            (root/'midnight-12k-400.npz').write_bytes(b'changed other view')
            with self.assertRaisesRegex(ValueError, 'center cache bytes'):
                m.load_base_centers(config, root, rows, 40)

    def test_fusion_has_one_float64_average_and_rejects_invalid_components(self):
        left = np.full((2, 3072), 1e20, np.float32); center = np.full_like(left, -1e20); right = np.ones_like(left)
        actual = m.fuse(left, center, right)
        np.testing.assert_array_equal(actual, np.full_like(left, np.float32(1/3)))
        for bad in (right.astype(np.float64), right[:, :3], np.full_like(right, np.nan)):
            with self.assertRaises(ValueError): m.fuse(left, center, bad)

    def test_atomic_roundtrip_rehash_cannot_hide_fused_arithmetic_change(self):
        config = configuration(); paths = ['fixture/a', 'fixture/b']; center = np.ones((2, 3072), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'midnight-coverage-v1-40.npz'
            m.write_output(target, paths, center*2, center*3, center, record(config, 40, 2))
            m.verify_output(target, paths, center, config, code_sha='a'*40)
            with self.assertRaises(ValueError): m.write_output(target, paths, center, center, center, record(config, 40, 2))
            np.savez_compressed(target, paths=np.asarray(paths), features=np.full_like(center, 99))
            meta_path = target.with_suffix('.json'); meta = json.loads(meta_path.read_text()); meta['feature_sha256'] = file_hash(target)
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'arithmetic'):
                m.verify_output(target, paths, center, config, code_sha='a'*40)

    def test_failed_publication_rolls_back_new_files_and_never_overwrites(self):
        config = configuration(); center = np.zeros((1, 3072), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'midnight-coverage-v1-40.npz'; original_link = m.os.link; calls = []
            def fail_second(source, destination):
                calls.append(destination)
                if len(calls) == 2: raise OSError('synthetic publication interruption')
                original_link(source, destination)
            with patch.object(m.os, 'link', side_effect=fail_second), self.assertRaises(OSError):
                m.write_output(target, ['fixture'], center, center, center, record(config, 40, 1))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            other_writer = m.component_path(target).with_suffix('.npz.part')
            other_writer.write_bytes(b'owned by another attempt')
            with self.assertRaises(FileExistsError):
                m.write_output(target, ['fixture'], center, center, center, record(config, 40, 1))
            self.assertEqual(other_writer.read_bytes(), b'owned by another attempt')

    def test_injected_tiny_extractor_preserves_order_and_maximum_batch(self):
        calls = []
        def tiny(values):
            calls.append(len(values)); return np.repeat(values.astype(np.float32), 3072, axis=1)
        result = m.encode_batches(tiny, np.arange(17, dtype=np.float32)[:, None])
        self.assertEqual(calls, [8, 8, 1]); np.testing.assert_array_equal(result[:, 0], np.arange(17))
        with self.assertRaises(ValueError): m.encode_batches(tiny, np.ones((2, 1)), 4)
        with self.assertRaises(ValueError): m.encode_batches(lambda values: np.zeros((len(values), 2), np.float32), np.zeros((2, 1)))

    def test_all_shard_cpu_audit_independently_reconstructs_fused_vectors(self):
        config = configuration(); config['expected_images'] = 8; config['expected_shapes'] = {'700x460': 4, '700x456': 4}
        rows = [{'relative_path': 'fixture/%d/%d.png'%(scale, index), 'magnification': scale,
                 'subtype': 'adenosis', 'width': 700, 'height': height}
                for scale in MAGNIFICATIONS for index, height in enumerate((460, 456))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            config['manifest_sha256'] = file_hash(manifest); cache = root/'cache'; cache.mkdir()
            def centers(cfg, center_cache, all_rows, scale):
                return [row['relative_path'] for row in rows if row['magnification'] == scale], np.ones((2, 3072), np.float32), {}
            for scale in MAGNIFICATIONS:
                paths, center, _ = centers(config, None, rows, scale)
                m.write_output(cache/('midnight-coverage-v1-%d.npz'%scale), paths, center*2, center*3, center, record(config, scale, 2))
            with patch.object(audit_module.producer, 'read_config', return_value=config), patch.object(audit_module.producer, 'load_base_centers', side_effect=centers):
                result = audit_module.audit(ROOT/'configs/midnight-coverage-v1.json', manifest, cache, root, 'a'*40)
                self.assertEqual(result['images'], 8); self.assertEqual(len(result['shards']), 4)
                self.assertEqual(result['new_encoder_forwards'], 0)
                np.savez(cache/'extra.npz', features=np.zeros(1))
                with self.assertRaisesRegex(ValueError, 'archive matrix'):
                    audit_module.audit(ROOT/'configs/midnight-coverage-v1.json', manifest, cache, root, 'a'*40)

    @unittest.skipIf(torch is None, 'Torch/Torchvision are required for pixel transform fixtures')
    def test_both_geometries_have_exact_center_tensor_and_distinct_windows(self):
        with patch.object(base, 'load_model', side_effect=AssertionError('No model construction')):
            for height, resized, offsets in [(460, [340, 224], [0, 58, 116]), (456, [343, 224], [0, 60, 119])]:
                values = np.arange(700*height*3, dtype=np.uint32).astype(np.uint8).reshape(height, 700, 3)
                image = Image.fromarray(values, 'RGB'); tensors, geometry = m.crop_tensors(image)
                self.assertEqual(geometry['resized'], resized); self.assertEqual(geometry['lefts'], offsets)
                self.assertTrue(torch.equal(tensors[1], base.build_transform()(image)))
                self.assertEqual(tensors.shape, (3, 3, 224, 224)); self.assertEqual(len(set(offsets)), 3)


if __name__ == '__main__':
    unittest.main()

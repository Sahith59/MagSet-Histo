import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import csv
import platform
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from magset.data import CLASS_NAMES, file_hash

try:
    import torch
    from magset.multicrop_features import (CheckedFields, SOURCE_FILES, blend_features,
        canonical_hash, encode_batches, load_thumbnails, quadrant_bounds, quadrants,
        select_rows, validate_config, verify_source, run)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Pinned Torch needed for multicrop feature checks')
class MulticropFeatureTests(unittest.TestCase):
    def test_real_and_odd_geometry_covers_every_pixel_once(self):
        for width, height in ((700, 460), (7, 5), (2, 2)):
            coverage = np.zeros((height, width), dtype=np.uint8)
            pixels = np.arange(height * width, dtype=np.int32).reshape(height, width)
            image = Image.fromarray(pixels)
            for crop, bounds in zip(quadrants(image), quadrant_bounds(width, height)):
                x0, y0, x1, y1 = bounds
                coverage[y0:y1, x0:x1] += 1
                np.testing.assert_array_equal(np.asarray(crop), pixels[y0:y1, x0:x1])
            np.testing.assert_array_equal(coverage, np.ones_like(coverage))
        for size in ((1, 3), (3, 0)):
            with self.assertRaises(ValueError):
                quadrant_bounds(*size)

    def test_blend_uses_float64_accumulation_and_exact_replay(self):
        g = np.asarray([[2, -3]], dtype=np.float32)
        q = np.asarray([[[1e8, 1], [1, 3], [-1e8, 5], [1, 7]]], dtype=np.float32)
        expected = (.5 * g.astype(np.float64) + .5 * q.mean(axis=1, dtype=np.float64)).astype(np.float32)
        np.testing.assert_array_equal(blend_features(g, q), expected)
        self.assertEqual(blend_features(g, q).dtype, np.float32)
        for badg, badq in ((g.astype(np.float64), q), (g, q[:, :3]),
                            (g, np.full_like(q, np.nan)), (np.full_like(g, np.inf), q)):
            with self.assertRaises(ValueError):
                blend_features(badg, badq)

    def test_mixed_forward_batch_sizes_preserve_field_quadrant_order(self):
        images = []
        for base in (10, 80, 160):
            x = np.empty((5, 7, 3), dtype=np.uint8)
            for i, (x0, y0, x1, y1) in enumerate(quadrant_bounds(7, 5)):
                x[y0:y1, x0:x1] = base + i
            images.append(Image.fromarray(x))
        def transform(batch):
            return torch.tensor([[np.asarray(im).mean(), np.asarray(im).min()] for im in batch], dtype=torch.float32)
        def encode(x):
            return x * 2
        lookup = {str(i): np.asarray([i, i + 1], dtype=np.float32) for i in range(3)}
        loader = [[('0', images[0]), ('1', images[1])], [('2', images[2])]]
        outputs = [encode_batches(loader, transform, encode, lookup, 'cpu', 2, b, smoke=True)
                   for b in (1, 3, 8)]
        for out in outputs:
            self.assertEqual(out['paths'].tolist(), ['0', '1', '2'])
            for i, base in enumerate((10, 80, 160)):
                np.testing.assert_array_equal(out['quadrant_features'][i, :, 0],
                                              np.arange(base, base + 4) * 2)
            np.testing.assert_array_equal(out['features'], blend_features(out['thumbnail_features'], out['quadrant_features']))
            np.testing.assert_array_equal(out['features'], outputs[0]['features'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            encode_batches([loader[0], loader[0]], transform, encode, lookup, 'cpu', 2, 2)
        with self.assertRaisesRegex(ValueError, 'finite float32'):
            encode_batches(loader, lambda xs: transform(xs).double(), encode, lookup, 'cpu', 2, 2)

    def test_dataset_role_guard_runs_before_any_image_open(self):
        row = {'relative_path': 'fit.png'}
        with patch('magset.multicrop_features.load_checked_image') as opened:
            with self.assertRaisesRegex(ValueError, 'role path guard'):
                CheckedFields([row], '/not-real', {'validation.png'})
            dataset = CheckedFields([row], '/not-real', {'fit.png'})
            dataset.rows[0] = {'relative_path': 'outer.png'}
            with self.assertRaisesRegex(ValueError, 'role path guard'):
                dataset[0]
            opened.assert_not_called()

    def test_smoke_selects_only_trusted_fit_before_decoding(self):
        rows, cases, roles = [], [], {}
        for label, subtype in enumerate(CLASS_NAMES):
            for i in range(3):
                key = 'PREFIX-%d-%d' % (label, i)
                cases.append({'case_id': key, 'label': label})
                roles[key] = 'train' if i < 2 else 'test'
                for n in range(4):
                    rows.append({'relative_path': '%s/%d.png' % (key, n), 'magnification': '40',
                                 'filename': 'PREFIX-x', 'case_id_raw': '%d-%d' % (label, i)})
        trusted = {'cases': cases, 'trials': {'1': {'case_roles': roles}}}
        from magset.baseline import stratified_inner_split
        fit, val, test = stratified_inner_split(cases, roles, 812)
        selected = select_rows(rows, 40, True, trusted)
        self.assertEqual(len(selected), 4)
        allowed = {cases[i]['case_id'] for i in fit}
        from magset.data import case_id
        self.assertTrue(all(case_id(r) in allowed for r in selected))
        self.assertFalse({case_id(r) for r in selected} & {cases[i]['case_id'] for i in list(val) + list(test)})
        for mag, context in ((100, trusted), (40, None)):
            with self.assertRaises(ValueError):
                select_rows(rows, mag, True, context)

    def thumbnail_fixture(self, root):
        path = root / 'phikon-v2-40.npz'
        # Reverse cache order deliberately: extraction must map paths, not row indices.
        np.savez_compressed(path, paths=np.asarray(['b', 'a']), features=np.asarray([[2, 3], [4, 5]], dtype=np.float32))
        encoder = {'dimension': 2, 'revision': 'a' * 40}
        meta = {'manifest_sha256': 'manifest', 'encoder': encoder, 'feature_sha256': file_hash(path),
                'magnification': 40, 'smoke_limit': 0, 'precision': 'float32; TF32 disabled',
                'code_sha': 'b' * 40, 'transform': {'size': {'height': 224, 'width': 224}}, 'images': 2}
        path.with_suffix('.json').write_text(json.dumps(meta))
        config = {'encoder': encoder, 'thumbnail_provenance': {'40': {
            'npz_sha256': file_hash(path), 'json_sha256': file_hash(path.with_suffix('.json'))}}}
        rows = [{'relative_path': p, 'magnification': '40'} for p in ('a', 'b')]
        return path, config, rows

    def test_thumbnail_exact_hash_provenance_and_path_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, config, rows = self.thumbnail_fixture(root)
            lookup, meta, hashes = load_thumbnails(root, 40, rows, 'manifest', config)
            np.testing.assert_array_equal(lookup['a'], [4, 5])
            self.assertEqual(hashes, config['thumbnail_provenance']['40'])
            for changed_rows, manifest in ((rows[:1], 'manifest'), (rows, 'other')):
                with self.assertRaises(ValueError):
                    load_thumbnails(root, 40, changed_rows, manifest, config)
            path.with_suffix('.json').write_text(json.dumps({**meta, 'transform': {'altered': True}}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_thumbnails(root, 40, rows, 'manifest', config)
            path, config, rows = self.thumbnail_fixture(root)
            with path.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_thumbnails(root, 40, rows, 'manifest', config)

    def test_thumbnail_finite_dtype_and_duplicate_paths_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for values, keys in ((np.ones((2, 2), dtype=np.float64), ['a', 'b']),
                                  (np.full((2, 2), np.nan, dtype=np.float32), ['a', 'b']),
                                  (np.ones((2, 2), dtype=np.float32), ['a', 'a'])):
                path, config, rows = self.thumbnail_fixture(root)
                np.savez_compressed(path, paths=np.asarray(keys), features=values)
                meta = json.loads(path.with_suffix('.json').read_text())
                meta['feature_sha256'] = file_hash(path)
                path.with_suffix('.json').write_text(json.dumps(meta))
                config['thumbnail_provenance']['40'] = {'npz_sha256': file_hash(path), 'json_sha256': file_hash(path.with_suffix('.json'))}
                with self.assertRaisesRegex(ValueError, 'coverage/shape/values'):
                    load_thumbnails(root, 40, rows, 'manifest', config)

    def test_registered_source_marker_config_and_actual_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in SOURCE_FILES:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(name)
            (root / '.source-sha').write_text('a' * 40 + '\n')
            (root / '.archive-sha256').write_text('b' * 64 + '\n')
            config = root / 'configs/breakhis-multicrop-v1.json'
            config.parent.mkdir()
            config.write_text('{}\n')
            with patch('magset.multicrop_features.__file__', str(root / SOURCE_FILES[0])):
                hashes, archive = verify_source('a' * 40, config)
                self.assertEqual(hashes[SOURCE_FILES[0]], file_hash(root / SOURCE_FILES[0]))
                self.assertEqual(archive, 'b' * 64)
                with self.assertRaises(ValueError):
                    verify_source('c' * 40, config)
                other = root / 'other.json'; other.write_text('{"changed":1}')
                with self.assertRaisesRegex(ValueError, 'Configuration differs'):
                    verify_source('a' * 40, other)
                (root / '.archive-sha256').write_text('invalid')
                with self.assertRaisesRegex(ValueError, 'archive checksum'):
                    verify_source('a' * 40, config)

    def test_complete_toy_run_hashes_images_replays_archive_and_refuses_processor_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir()
            images = root / 'images'; images.mkdir()
            encoder = {'repository': 'owkin/phikon-v2', 'revision': 'a' * 40, 'dimension': 1024}
            description = {'do_resize': True, 'size': {'shortest_edge': 224},
                           'do_center_crop': True, 'crop_size': {'height': 224, 'width': 224}}
            def transform(batch):
                return torch.tensor([[float(np.asarray(im).mean())] for im in batch], dtype=torch.float32)
            def encode(x):
                return x.repeat(1, 1024)
            model = torch.nn.Identity().eval().requires_grad_(False)
            rows = []
            for i in range(2):
                p = images / ('field%d.png' % i)
                x = np.empty((5, 7, 3), dtype=np.uint8)
                for j, (x0, y0, x1, y1) in enumerate(quadrant_bounds(7, 5)):
                    x[y0:y1, x0:x1] = i * 10 + j
                Image.fromarray(x).save(p)
                rows.append({'relative_path': p.name, 'magnification': '40', 'subtype': CLASS_NAMES[0],
                             'sha256': file_hash(p), 'width': '7', 'height': '5'})
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            path = cache / 'phikon-v2-40.npz'
            original = [Image.open(images / r['relative_path']).convert('RGB') for r in rows]
            g = encode(transform(original)).numpy()
            np.savez_compressed(path, paths=np.asarray([r['relative_path'] for r in rows]), features=g)
            meta = {'manifest_sha256': file_hash(manifest), 'encoder': encoder, 'feature_sha256': file_hash(path),
                    'magnification': 40, 'smoke_limit': 0, 'precision': 'float32; TF32 disabled',
                    'code_sha': 'b' * 40, 'transform': description, 'images': 2,
                    'torch': torch.__version__, 'python': platform.python_version()}
            path.with_suffix('.json').write_text(json.dumps(meta))
            config = {'encoder': encoder, 'thumbnail_weight': .5, 'quadrant_weight': .5, 'smoke_fields': 4,
                      'smoke_magnification': 40, 'thumbnail_parity_atol': 1e-4,
                      'manifest_sha256': file_hash(manifest), 'thumbnail_provenance': {'40': {
                          'npz_sha256': file_hash(path), 'json_sha256': file_hash(path.with_suffix('.json'))}}}
            config_path = root / 'config.json'; config_path.write_text(json.dumps(config))
            args = SimpleNamespace(config=config_path, code_sha='c' * 40, batch_size=3, workers=0,
                                   output=root / 'output', manifest=manifest, smoke=False, protocol=None,
                                   magnification=40, thumbnail_cache=cache, images_root=images, device='cpu')
            with patch('magset.multicrop_features.verify_source', return_value=({'source': 'd' * 64}, 'e' * 64)):
                with patch('magset.multicrop_features.load_encoder', return_value=(model, transform, encode, {'different': True})):
                    with patch('magset.multicrop_features.load_checked_image') as opened:
                        with self.assertRaisesRegex(ValueError, 'Actual processor differs'):
                            run(args)
                        opened.assert_not_called()
                        self.assertFalse(args.output.exists())
                with patch('magset.multicrop_features.load_encoder', return_value=(model, transform, encode, description)):
                    record = run(args)
                    self.assertEqual(record['images'], 2)
                    self.assertEqual(record['quadrant_forwards'], 8)
                    self.assertEqual(record['blend_replay_max_abs'], 0)
                    artifact = args.output / 'phikon-v2-40.npz'
                    self.assertEqual(file_hash(artifact), record['feature_sha256'])
                    with np.load(artifact, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['thumbnail_features'], g)
                        np.testing.assert_array_equal(saved['features'], blend_features(g, saved['quadrant_features']))
                        np.testing.assert_array_equal(saved['raw_sha256'], [r['sha256'] for r in rows])
                    with self.assertRaisesRegex(ValueError, 'never overwrite'):
                        run(args)
                    args.output = root / 'tampered-output'
                    (images / rows[0]['relative_path']).write_bytes(b'tampered original')
                    with self.assertRaisesRegex(ValueError, 'checksum differs'):
                        run(args)
                    self.assertFalse(args.output.exists())

    def test_recipe_is_fixed_and_processor_hash_is_order_independent(self):
        config = {'encoder': {'repository': 'owkin/phikon-v2', 'revision': 'a' * 40, 'dimension': 1024},
                  'thumbnail_weight': .5, 'quadrant_weight': .5, 'smoke_fields': 4,
                  'smoke_magnification': 40, 'thumbnail_parity_atol': 1e-4, 'thumbnail_provenance': {}}
        validate_config(config)
        for key, value in (('thumbnail_weight', .7), ('quadrant_weight', .3),
                           ('smoke_fields', 5), ('thumbnail_parity_atol', 1e-3)):
            bad = copy.deepcopy(config); bad[key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)
        self.assertEqual(canonical_hash({'a': 1, 'b': 2}), canonical_hash({'b': 2, 'a': 1}))
        self.assertNotEqual(canonical_hash({'size': 224}), canonical_hash({'size': 448}))


if __name__ == '__main__':
    unittest.main()

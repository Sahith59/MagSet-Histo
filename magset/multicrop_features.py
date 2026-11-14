"""Fixed frozen Phikon thumbnail/quadrant extraction; no classifier or selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .baseline import stratified_inner_split
from .breakhis_predict import load_checked_image
from .collect import load_trusted_context
from .data import MAGNIFICATIONS, case_id, file_hash, read_manifest
from .features import identity_collate, load_encoder, worker_init

SOURCE_FILES = ('magset/multicrop_features.py', 'magset/features.py', 'magset/data.py',
                'magset/breakhis_predict.py', 'magset/baseline.py', 'magset/collect.py')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_source(code_sha, config_path):
    """Validate the immutable snapshot markers and its registered configuration."""
    if not re.fullmatch(r'[0-9a-f]{40}', code_sha):
        raise ValueError('Need a full committed source SHA')
    root = Path(__file__).resolve().parents[1]
    if (root / '.source-sha').read_text().strip() != code_sha:
        raise ValueError('Immutable execution source marker differs')
    archive_sha = (root / '.archive-sha256').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', archive_sha):
        raise ValueError('Invalid immutable archive checksum marker')
    registered = root / 'configs/breakhis-multicrop-v1.json'
    if Path(config_path).read_bytes() != registered.read_bytes():
        raise ValueError('Configuration differs from immutable source registration')
    return {name: file_hash(root / name) for name in SOURCE_FILES}, archive_sha


def validate_config(config):
    if (config.get('thumbnail_weight') != .5 or config.get('quadrant_weight') != .5
            or config.get('smoke_fields') != 4 or config.get('smoke_magnification') != 40):
        raise ValueError('Fixed thumbnail/quadrant recipe differs')
    encoder = config.get('encoder', {})
    if (encoder.get('repository') != 'owkin/phikon-v2'
            or not re.fullmatch(r'[0-9a-f]{40}', encoder.get('revision', ''))
            or encoder.get('dimension') != 1024):
        raise ValueError('Need pinned frozen Phikon encoder configuration')
    if not isinstance(config.get('thumbnail_provenance'), dict):
        raise ValueError('Need registered accepted thumbnail provenance')
    tolerance = config.get('thumbnail_parity_atol')
    if not isinstance(tolerance, (int, float)) or tolerance != 1e-4:
        raise ValueError('Need a registered bounded thumbnail parity tolerance')
    return encoder


def quadrant_bounds(width, height):
    if width < 2 or height < 2:
        raise ValueError('Fields need positive nonempty quadrants')
    x, y = width // 2, height // 2
    return ((0, 0, x, y), (x, 0, width, y),
            (0, y, x, height), (x, y, width, height))


def quadrants(image):
    return [image.crop(bounds) for bounds in quadrant_bounds(*image.size)]


def blend_features(thumbnails, quarters):
    g, q = np.asarray(thumbnails), np.asarray(quarters)
    if (g.ndim != 2 or q.shape != (len(g), 4, g.shape[1])
            or g.dtype != np.float32 or q.dtype != np.float32
            or not np.isfinite(g).all() or not np.isfinite(q).all()):
        raise ValueError('Need finite float32 thumbnail and four-quadrant features')
    result = np.asarray(.5 * g.astype(np.float64) + .5 * q.mean(axis=1, dtype=np.float64),
                        dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite blended features')
    return result


def select_rows(rows, magnification, smoke=False, trusted=None):
    """Resolve the entire allowed path set before any original image is opened."""
    if magnification not in MAGNIFICATIONS:
        raise ValueError('Unsupported magnification')
    if not rows or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Need unique nonempty manifest paths')
    selected = [r for r in rows if int(r['magnification']) == magnification]
    if smoke:
        if magnification != 40 or trusted is None:
            raise ValueError('Smoke requires trusted fitting-only 40x roles')
        cases = trusted['cases']
        fitting, _, _ = stratified_inner_split(cases, trusted['trials']['1']['case_roles'], 812)
        allowed = {cases[i]['case_id'] for i in fitting}
        selected = sorted((r for r in selected if case_id(r) in allowed),
                          key=lambda r: r['relative_path'])[:4]
        if len(selected) != 4 or any(case_id(r) not in allowed for r in selected):
            raise ValueError('Smoke needs exactly four trusted fitting fields')
    if not selected:
        raise ValueError('Empty magnification shard')
    return selected


def load_thumbnails(cache, magnification, rows, manifest_sha, config):
    """Check registered files and full magnification coverage, including for smoke."""
    target = Path(cache) / ('phikon-v2-%d.npz' % magnification)
    meta_path = target.with_suffix('.json')
    pins = config['thumbnail_provenance'].get(str(magnification), {})
    hashes = {'npz_sha256': file_hash(target), 'json_sha256': file_hash(meta_path)}
    if hashes != pins:
        raise ValueError('Accepted thumbnail file/provenance checksum differs')
    meta = json.loads(meta_path.read_text())
    if (meta['manifest_sha256'] != manifest_sha or meta['encoder'] != config['encoder']
            or meta['feature_sha256'] != hashes['npz_sha256']
            or meta['magnification'] != magnification or meta['smoke_limit'] != 0
            or meta['precision'] != 'float32; TF32 disabled'
            or not re.fullmatch(r'[0-9a-f]{40}', meta['code_sha'])
            or not isinstance(meta['transform'], dict)):
        raise ValueError('Original thumbnail metadata differs')
    expected = [r['relative_path'] for r in rows if int(r['magnification']) == magnification]
    with np.load(target, allow_pickle=False) as saved:
        keys, values = saved['paths'].tolist(), saved['features']
    if (len(keys) != len(set(keys)) or set(keys) != set(expected)
            or meta['images'] != len(keys)
            or values.shape != (len(keys), config['encoder']['dimension'])
            or values.dtype != np.float32 or not np.isfinite(values).all()):
        raise ValueError('Original thumbnail coverage/shape/values differ')
    return {key: values[i].copy() for i, key in enumerate(keys)}, meta, hashes


class CheckedFields(Dataset):
    def __init__(self, rows, root, allowed_paths):
        self.rows, self.root = rows, Path(root)
        expected = [r['relative_path'] for r in rows]
        if len(expected) != len(set(expected)) or set(expected) != set(allowed_paths):
            raise ValueError('Image source/role path guard differs')
        self.allowed_paths = frozenset(allowed_paths)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row['relative_path'] not in self.allowed_paths:
            raise ValueError('Image source/role path guard differs')
        image = load_checked_image(self.root, row)
        return row['relative_path'], image


def encode_batches(loader, transform, encode, thumbnail_lookup, device, dimension, batch_size,
                   smoke=False):
    """Keep field-major quadrant order independently of loader/forward batch sizes."""
    paths, quarters, reencoded = [], [], []
    with torch.inference_mode():
        for batch in loader:
            keys, images = zip(*batch)
            if len(set(keys)) != len(keys) or any(k in paths or k not in thumbnail_lookup for k in keys):
                raise ValueError('Duplicate or unknown inference path')
            crops = [q for image in images for q in quadrants(image)]
            chunks = []
            for offset in range(0, len(crops), batch_size):
                pixels = transform(crops[offset:offset + batch_size])
                if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.float32
                        or not torch.isfinite(pixels).all()):
                    raise ValueError('Processor must return finite float32 tensor')
                output = encode(pixels.to(device))
                if (output.dtype != torch.float32 or output.shape != (len(pixels), dimension)
                        or not torch.isfinite(output).all()):
                    raise ValueError('Encoder must return finite float32 embeddings')
                chunks.append(output.cpu().numpy())
            quarters.append(np.concatenate(chunks).reshape(len(batch), 4, dimension))
            if smoke:
                chunks = []
                for offset in range(0, len(images), batch_size):
                    pixels = transform(list(images[offset:offset + batch_size]))
                    if pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
                        raise ValueError('Invalid thumbnail processor result')
                    output = encode(pixels.to(device))
                    if (output.dtype != torch.float32 or output.shape != (len(pixels), dimension)
                            or not torch.isfinite(output).all()):
                        raise ValueError('Invalid thumbnail parity embedding')
                    chunks.append(output.cpu().numpy())
                reencoded.append(np.concatenate(chunks))
            paths.extend(keys)
    if not paths:
        raise ValueError('Empty extraction')
    q = np.concatenate(quarters).astype(np.float32, copy=False)
    g = np.stack([thumbnail_lookup[k] for k in paths])
    result = {'paths': np.asarray(paths), 'features': blend_features(g, q),
              'quadrant_features': q, 'thumbnail_features': g}
    if smoke:
        result['reencoded_thumbnail_features'] = np.concatenate(reencoded)
    return result


def run(args):
    config_sha = file_hash(args.config)
    config = json.loads(args.config.read_text())
    encoder = validate_config(config)
    source_hashes, archive_sha = verify_source(args.code_sha, args.config)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid inference batch/worker budget')
    if args.output.exists():
        raise ValueError('Output directory exists; never overwrite extraction artifacts')
    manifest_sha = file_hash(args.manifest)
    rows = read_manifest(args.manifest)
    trusted = None
    if args.smoke:
        if args.protocol is None:
            raise ValueError('Smoke needs trusted protocol')
        trusted = load_trusted_context(args.manifest, args.protocol)
        if (config.get('protocol_sha256') != trusted['protocol_sha256']
                or config.get('manifest_sha256') != manifest_sha):
            raise ValueError('Smoke protocol/manifest differs from registration')
    elif config.get('manifest_sha256') != manifest_sha:
        raise ValueError('Manifest differs from registration')
    selected = select_rows(rows, args.magnification, args.smoke, trusted)
    lookup, thumbnail_meta, thumbnail_hashes = load_thumbnails(
        args.thumbnail_cache, args.magnification, rows, manifest_sha, config)
    if (thumbnail_meta.get('torch') != torch.__version__
            or thumbnail_meta.get('python') != platform.python_version()):
        raise ValueError('Runtime differs from accepted thumbnail torch/Python environment')
    # Guard geometry/dimensions and path membership before DataLoader opens images.
    for row in selected:
        quadrant_bounds(int(row['width']), int(row['height']))
    allowed = [r['relative_path'] for r in selected]
    dataset = CheckedFields(selected, args.images_root, allowed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation/framework not available')
    torch.manual_seed(812)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, transform, encode, description = load_encoder('phikon-v2', {'phikon-v2': encoder}, device)
    if canonical_hash(description) != canonical_hash(thumbnail_meta['transform']):
        raise ValueError('Actual processor differs from accepted thumbnail processor')
    if (model.training or any(p.requires_grad for p in model.parameters())
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('Encoder must be frozen eval float32')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=identity_collate, worker_init_fn=worker_init,
                        persistent_workers=args.workers > 0,
                        multiprocessing_context='spawn' if args.workers else None)
    start = time.monotonic()
    result = encode_batches(loader, transform, encode, lookup, device, encoder['dimension'],
                            args.batch_size, args.smoke)
    if result['paths'].tolist() != allowed:
        raise ValueError('Extraction order/coverage differs from selected paths')
    replay = blend_features(result['thumbnail_features'], result['quadrant_features'])
    replay_error = float(np.max(np.abs(result['features'] - replay)))
    if replay_error != 0:
        raise ValueError('Combined feature replay differs')
    parity = None
    if args.smoke:
        parity = float(np.max(np.abs(result['reencoded_thumbnail_features'] - result['thumbnail_features'])))
        if parity > config['thumbnail_parity_atol']:
            raise ValueError('Thumbnail re-encoding parity gate failed: %.9g' % parity)
    result.update(raw_sha256=np.asarray([r['sha256'] for r in selected]),
                  widths=np.asarray([int(r['width']) for r in selected], dtype=np.int64),
                  heights=np.asarray([int(r['height']) for r in selected], dtype=np.int64),
                  bounds=np.asarray([quadrant_bounds(int(r['width']), int(r['height'])) for r in selected],
                                    dtype=np.int64))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    # Exclusive directory creation prevents concurrent writers, including smoke/full collision.
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('phikon-v2-%d.npz' % args.magnification)
    with target.open('xb') as stream:
        np.savez_compressed(stream, **result)
    with np.load(target, allow_pickle=False) as saved:
        if (saved['paths'].tolist() != allowed
                or not np.array_equal(saved['features'], blend_features(saved['thumbnail_features'], saved['quadrant_features']))):
            raise ValueError('Serialized feature replay differs')
    record = {'manifest_sha256': manifest_sha, 'encoder': encoder, 'code_sha': args.code_sha,
              'source_sha256': source_hashes, 'implementation_sha256': file_hash(__file__),
              'archive_sha256': archive_sha,
              'config': config, 'config_sha256': config_sha,
              'magnification': args.magnification, 'images': len(selected),
              'smoke_limit': 4 if args.smoke else 0, 'smoke': bool(args.smoke),
              'smoke_roles': {'trial': 1, 'seed': 812, 'role': 'train',
                              'protocol_sha256': trusted['protocol_sha256']} if args.smoke else None,
              'thumbnail_provenance': thumbnail_hashes, 'thumbnail_metadata': thumbnail_meta,
              'thumbnail_feature_sha256': thumbnail_hashes['npz_sha256'],
              'transform': description, 'transform_sha256': canonical_hash(description),
              'quadrant_order': ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
              'blend': 'float32(0.5*float64(g) + 0.5*mean(q1..q4,float64))',
              'thumbnail_parity_max_abs': parity, 'blend_replay_max_abs': replay_error,
              'quadrant_forwards': 4 * len(selected), 'technical_thumbnail_forwards': len(selected) if args.smoke else 0,
              'precision': 'float32; TF32 disabled', 'elapsed_seconds': elapsed,
              'python': platform.python_version(), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'feature_sha256': file_hash(target),
              'fit_status': 'frozen external pretrained encoder; no cohort fitting; labels used only for smoke role reconstruction'}
    target.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'images': len(selected), 'feature_sha256': record['feature_sha256'],
                      'thumbnail_parity_max_abs': parity, 'blend_replay_max_abs': replay_error}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'images-root', 'thumbnail-cache', 'output', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    parser.add_argument('--code-sha', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

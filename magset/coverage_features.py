"""Frozen Midnight left/center/right coverage descriptors with cached centers."""
import argparse
import gc
import json
import os
from pathlib import Path, PurePosixPath
import re
import time

import numpy as np

from . import midnight_features as base
from .breakhis_predict import load_checked_image
from .data import MAGNIFICATIONS, file_hash, read_manifest

CONFIG_SHA256 = '6caada1cb238a41d483e3e2cd9eb3f2f362c44b9073b07a96456a99599e81e98'
GEOMETRY = {(700, 460): ((340, 224), (0, 58, 116), 0),
            (700, 456): ((343, 224), (0, 60, 119), 0)}


def read_config(path):
    if file_hash(path) != CONFIG_SHA256:
        raise ValueError('Registered coverage configuration bytes differ')
    config = json.loads(Path(path).read_text())
    root = Path(__file__).resolve().parents[1]
    for key in ('base_extraction_config', 'base_classifier_config'):
        if file_hash(root/config[key]) != config[key+'_sha256']:
            raise ValueError('Accepted base configuration differs')
    extraction = base.read_config(root/config['base_extraction_config'])
    classifier = json.loads((root/config['base_classifier_config']).read_text())
    if (config['base_cache_provenance'] != classifier['cache_provenance']
            or config['dependencies'] != extraction['dependencies']
            or config['encoder']['dimension'] != 3072
            or any(config['encoder'][key] != extraction['encoder'][key]
                   for key in extraction['encoder'] if key not in ('features', 'transform'))):
        raise ValueError('Coverage base cache/encoder/environment lineage differs')
    return config


def select_rows(manifest, config, magnification, limit):
    if magnification not in MAGNIFICATIONS or limit not in (0, 4) or (limit and magnification != 40):
        raise ValueError('Use a full shard or the registered four-image 40x smoke')
    if file_hash(manifest) != config['manifest_sha256']:
        raise ValueError('Registered manifest differs')
    rows = read_manifest(manifest)
    counts = {}
    for row in rows:
        path = PurePosixPath(row['relative_path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Manifest path escapes the image root')
        shape = '%sx%s' % (row['width'], row['height'])
        counts[shape] = counts.get(shape, 0)+1
    if len(rows) != config['expected_images'] or counts != config['expected_shapes']:
        raise ValueError('Full release image count or geometry differs')
    chosen = [row for row in rows if int(row['magnification']) == magnification]
    if limit:
        smoke_rows = []
        for height in (460, 456):
            smoke_rows.extend([row for row in chosen if int(row['height']) == height][:2])
        chosen = smoke_rows
        if len(chosen) != 4 or [int(row['height']) for row in chosen] != [460, 460, 456, 456]:
            raise ValueError('Technical smoke requires two fields of each audited height')
    return rows, chosen


def load_base_centers(config, center_cache, rows, magnification):
    root = Path(__file__).resolve().parents[1]
    extraction = base.read_config(root/config['base_extraction_config'])
    for scale in MAGNIFICATIONS:
        path = Path(center_cache)/('midnight-12k-%d.npz'%scale)
        receipt = config['base_cache_provenance'][str(scale)]
        if file_hash(path) != receipt['npz_sha256'] or file_hash(path.with_suffix('.json')) != receipt['json_sha256']:
            raise ValueError('Accepted center cache bytes differ')
        metadata = json.loads(path.with_suffix('.json').read_text())
        if (metadata.get('code_sha') != config['base_feature_producer_code_sha']
                or metadata.get('config_sha256') != config['base_extraction_config_sha256']
                or metadata.get('source_file_sha256') != file_hash(root/'magset/midnight_features.py')
                or metadata.get('encoder') != extraction['encoder']
                or metadata.get('dependencies') != extraction['dependencies']):
            raise ValueError('Center cache producer or encoder lineage differs')
    target = Path(center_cache)/('midnight-12k-%d.npz'%magnification)
    paths = [row['relative_path'] for row in rows if int(row['magnification']) == magnification]
    base.verify_output(target, target.with_suffix('.json'), paths, extraction['encoder'],
                       config['manifest_sha256'], 0, config['base_extraction_config_sha256'])
    with np.load(target, allow_pickle=False) as saved:
        values = saved['features'].copy()
    return paths, values, json.loads(target.with_suffix('.json').read_text())


def crop_tensors(image):
    import torch
    from torchvision.transforms import v2, InterpolationMode
    if image.mode != 'RGB' or image.size not in GEOMETRY:
        raise ValueError('Expected an audited RGB image geometry')
    resized = v2.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True)(image)
    shape, lefts, top = GEOMETRY[image.size]
    actual_lefts = (0, int(round((resized.width-224)/2)), resized.width-224)
    if resized.size != shape or actual_lefts != lefts or int(round((resized.height-224)/2)) != top:
        raise ValueError('Pinned resize/crop integer geometry differs')
    crops = [resized.crop((left, top, left+224, top+224)) for left in lefts]
    if not np.array_equal(np.asarray(crops[1]), np.asarray(v2.CenterCrop(224)(resized))):
        raise ValueError('Explicit center pixels differ from official crop')
    after = v2.Compose([v2.ToTensor(), v2.Normalize((.5,)*3, (.5,)*3)])
    tensors = torch.stack([after(crop) for crop in crops])
    if not torch.equal(tensors[1], base.build_transform()(image)):
        raise ValueError('Explicit center tensor differs from official transform')
    return tensors, {'original': list(image.size), 'resized': list(shape), 'lefts': list(lefts), 'top': top}


def fuse(left, center, right):
    arrays = [np.asarray(value) for value in (left, center, right)]
    if (any(value.ndim != 2 or value.shape[1] != 3072 or value.dtype != np.float32
            or not np.isfinite(value).all() for value in arrays)
            or len({value.shape for value in arrays}) != 1):
        raise ValueError('Expected equal finite FP32 crop descriptors')
    return ((arrays[0].astype(np.float64)+arrays[1].astype(np.float64)+arrays[2].astype(np.float64))/3).astype(np.float32)


def component_path(target):
    target = Path(target)
    return target.with_name(target.stem+'-components.npz')


def verify_output(target, paths, centers, config, limit=0, code_sha=None):
    target = Path(target); components = component_path(target)
    record = json.loads(target.with_suffix('.json').read_text())
    scale = record.get('magnification'); receipt = config['base_cache_provenance'].get(str(scale), {})
    expected = {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
                'config_sha256': CONFIG_SHA256, 'smoke_limit': limit, 'images': len(paths),
                'source_file_sha256': file_hash(__file__), 'dependencies': config['dependencies'],
                'feature_sha256': file_hash(target), 'components_sha256': file_hash(components),
                'base_center_npz_sha256': receipt.get('npz_sha256'), 'base_center_json_sha256': receipt.get('json_sha256'),
                'base_feature_producer_code_sha': config['base_feature_producer_code_sha'],
                'base_extraction_config_sha256': config['base_extraction_config_sha256'],
                'base_classifier_config_sha256': config['base_classifier_config_sha256'],
                'transform': config['encoder']['transform'], 'frozen_base': True,
                'geometry_examples': config['geometry_examples']}
    if code_sha is not None: expected['code_sha'] = code_sha
    if scale not in MAGNIFICATIONS or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Coverage cache provenance differs')
    with np.load(target, allow_pickle=False) as saved, np.load(components, allow_pickle=False) as parts:
        if (set(saved.files) != {'paths', 'features'} or set(parts.files) != {'paths', 'left_features', 'right_features'}
                or saved['paths'].tolist() != paths or parts['paths'].tolist() != paths or len(set(paths)) != len(paths)):
            raise ValueError('Coverage cache schema or ordered path coverage differs')
        reconstructed = fuse(parts['left_features'], centers, parts['right_features'])
        values = saved['features']
        if values.dtype != np.float32 or not np.isfinite(values).all() or not np.array_equal(values, reconstructed):
            raise ValueError('Fused descriptors differ from exact cached-center arithmetic')
    return record


def write_output(target, paths, left, right, centers, record):
    target = Path(target); components = component_path(target); meta = target.with_suffix('.json')
    if any(path.exists() for path in (target, components, meta)):
        raise ValueError('Refusing existing coverage artifacts')
    values = fuse(left, centers, right)
    if values.shape[0] != len(paths) or len(set(paths)) != len(paths):
        raise ValueError('Crop descriptors lack unique field paths')
    target.parent.mkdir(parents=True, exist_ok=True); pending = []; published = []
    try:
        for path, arrays in [(components, {'paths': np.asarray(paths), 'left_features': left, 'right_features': right}),
                             (target, {'paths': np.asarray(paths), 'features': values})]:
            temp = path.with_suffix('.npz.part')
            with temp.open('xb') as stream:
                pending.append(temp)
                np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        record = dict(record, feature_sha256=file_hash(pending[1]), components_sha256=file_hash(pending[0]))
        temporary_meta = meta.with_suffix('.json.part')
        with temporary_meta.open('x') as stream:
            pending.append(temporary_meta)
            stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        for temporary, destination in zip(pending, (components, target, meta)):
            os.link(temporary, destination); published.append(destination)
        with np.load(target, allow_pickle=False) as saved, np.load(components, allow_pickle=False) as parts:
            if not np.array_equal(saved['features'], values) or not np.array_equal(parts['left_features'], left) or not np.array_equal(parts['right_features'], right):
                raise ValueError('Actual coverage archive reload differs')
    except Exception:
        for path in published: path.unlink()
        raise
    finally:
        for path in pending:
            if path.exists(): path.unlink()
    return record


def encode_batches(encoder, pixels, batch_size=8):
    if batch_size != 8:
        raise ValueError('Use the registered maximum crop batch of eight')
    values = []
    for begin in range(0, len(pixels), batch_size):
        output = np.asarray(encoder(pixels[begin:begin+batch_size]))
        if output.shape != (min(batch_size, len(pixels)-begin), 3072) or output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError('Invalid frozen crop encoder output')
        values.append(output)
    return np.concatenate(values)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cache-root', 'center-cache', 'manifest', 'images-root', 'output-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--limit', type=int, choices=(0, 4), default=0)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run coverage extraction in a CUDA scheduler allocation')
    if args.batch_size != 8 or args.workers != 0 or not re.fullmatch('[0-9a-f]{40}', args.code_sha):
        raise ValueError('Registered batch/workers/source definition differs')
    root = Path(__file__).resolve().parents[1]
    if (root/'.source-sha').read_text().strip() != args.code_sha:
        raise ValueError('Immutable extraction source receipt differs')
    config = read_config(args.config); base.runtime_versions(config)
    rows, selected = select_rows(args.manifest, config, args.magnification, args.limit)
    paths, cached, _ = load_base_centers(config, args.center_cache, rows, args.magnification)
    lookup = dict(zip(paths, cached)); selected_paths = [row['relative_path'] for row in selected]
    centers = np.asarray([lookup[path] for path in selected_paths], np.float32)
    target = args.output_root/('%s-%d.npz'%(config['encoder_name'], args.magnification))
    if any(path.exists() for path in (target, component_path(target), target.with_suffix('.json'))):
        verify_output(target, selected_paths, centers, config, args.limit, args.code_sha)
        print('Verified existing coverage shard'); return
    torch.manual_seed(812); torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    model = base.load_model(args.cache_root, torch.device('cuda')); started = time.monotonic()
    def encode(pixels):
        return base.classification_features(model(pixel_values=pixels.to('cuda')).last_hidden_state).cpu().numpy()
    left, right, geometries = [], [], []; smoke_pixels = smoke_features = None
    with torch.inference_mode():
        for start in range(0, len(selected), 4):
            chunk = selected[start:start+4]
            transformed = [crop_tensors(load_checked_image(args.images_root, row)) for row in chunk]
            pixels = torch.stack([item[0] for item in transformed])
            geometries.extend(item[1] for item in transformed)
            if args.limit:
                smoke_pixels = pixels.flatten(0, 1)
                smoke_features = encode_batches(encode, smoke_pixels).reshape(len(chunk), 3, 3072)
                left.append(smoke_features[:, 0]); right.append(smoke_features[:, 2])
            else:
                values = encode_batches(encode, pixels[:, [0, 2]].flatten(0, 1)).reshape(len(chunk), 2, 3072)
                left.append(values[:, 0]); right.append(values[:, 1])
    left, right = np.concatenate(left), np.concatenate(right); parity = {}
    if args.limit:
        original = smoke_features.reshape(-1, 3072)
        with torch.inference_mode():
            parity['center_cache_max_abs'] = base.assert_parity(centers, smoke_features[:, 1])
            parity['repeat_max_abs'] = base.assert_parity(original, encode_batches(encode, smoke_pixels))
            singles = np.concatenate([encode(smoke_pixels[index:index+1]) for index in range(len(smoke_pixels))])
            parity['single_batch_max_abs'] = base.assert_parity(original, singles)
        del model; gc.collect(); torch.cuda.empty_cache()
        model = base.load_model(args.cache_root, torch.device('cuda'))
        with torch.inference_mode():
            fresh = encode_batches(encode, smoke_pixels).reshape(-1, 3, 3072)
        parity['reconstructed_model_max_abs'] = base.assert_parity(original, fresh.reshape(-1, 3072))
        parity['fused_max_abs'] = base.assert_parity(fuse(left, centers, right), fuse(fresh[:, 0], fresh[:, 1], fresh[:, 2]))
    torch.cuda.synchronize()
    receipt = config['base_cache_provenance'][str(args.magnification)]
    record = {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'], 'config_sha256': CONFIG_SHA256,
              'code_sha': args.code_sha, 'source_file_sha256': file_hash(__file__), 'magnification': args.magnification,
              'images': len(selected_paths), 'smoke_limit': args.limit, 'dependencies': config['dependencies'],
              'base_center_npz_sha256': receipt['npz_sha256'], 'base_center_json_sha256': receipt['json_sha256'],
              'base_feature_producer_code_sha': config['base_feature_producer_code_sha'],
              'base_extraction_config_sha256': config['base_extraction_config_sha256'],
              'base_classifier_config_sha256': config['base_classifier_config_sha256'],
              'precision': 'FP32; eager attention; TF32 disabled', 'crop_order': config['crop_labels'],
              'transform': config['encoder']['transform'], 'frozen_base': True,
              'geometry_examples': config['geometry_examples'],
              'geometry_counts': {'%dx%d'%shape: sum(tuple(item['original']) == shape for item in geometries) for shape in GEOMETRY},
              'parity': parity, 'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'],
              'new_crop_forwards': len(selected_paths)*2 if not args.limit else 48,
              'elapsed_seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(),
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    write_output(target, selected_paths, left, right, centers, record)
    verify_output(target, selected_paths, centers, config, args.limit, args.code_sha)
    print(json.dumps({'status': 'accepted frozen three-crop coverage shard', 'images': len(selected_paths), 'parity': parity}))


if __name__ == '__main__':
    main()

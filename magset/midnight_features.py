"""Pinned, frozen FP32 Midnight-12k image features; no network access at inference."""
import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import time

import numpy as np

from .data import MAGNIFICATIONS, file_hash, read_manifest

REPOSITORY = 'kaiko-ai/midnight'
REVISION = 'adc6b15679c981cce6f9b018bbad09d16eeeda9f'
WEIGHTS_SHA256 = '52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1'
WEIGHTS_SIZE = 4546005432
MODEL_CONFIG_SHA256 = '0362e014a3f3b1ad2f2faf115d21cbc05ed48b3a6114bdb4ddec897bc83cc0fd'
CARD_SHA256 = '7595d9ac11154d804a9e920abae185bd98798ebd4ba375490f1a845dc32d2001'
TRANSFORM = {'resize_shorter_side': 224, 'center_crop': 224,
             'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5],
             'interpolation': 'bilinear', 'antialias': True}
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4


def validate_config(config):
    if config.get('name') != 'midnight-12k-v1' or config.get('schema_version') != 1:
        raise ValueError('Unexpected Midnight configuration identity')
    if not re.fullmatch('[0-9a-f]{64}', config.get('manifest_sha256', '')):
        raise ValueError('Pinned manifest hash required')
    if config.get('expected_images') != 7909:
        raise ValueError('The full 7909-image release is required')
    encoder = config.get('encoder', {})
    expected = {'repository': REPOSITORY, 'revision': REVISION, 'dimension': 3072,
                'weights_file': 'model.safetensors', 'weights_sha256': WEIGHTS_SHA256,
                'weights_size_bytes': WEIGHTS_SIZE, 'config_file': 'config.json',
                'config_sha256': MODEL_CONFIG_SHA256, 'config_size_bytes': 879,
                'model_card_file': 'README.md', 'model_card_sha256': CARD_SHA256,
                'model_card_size_bytes': 8080, 'license': 'MIT', 'transform': TRANSFORM}
    if any(encoder.get(k) != v for k, v in expected.items()):
        raise ValueError('Pinned encoder, files, dimension or transform differs')
    if not isinstance(encoder.get('features'), str) or not encoder['features']:
        raise ValueError('Feature definition required')
    dependencies = config.get('dependencies', {})
    if not all(isinstance(dependencies.get(k), str) for k in
               ('torch', 'torchvision', 'transformers', 'safetensors', 'Pillow')):
        raise ValueError('Pinned runtime dependencies required')
    if dependencies['transformers'] != '4.51.3' or dependencies['safetensors'] != '0.5.3':
        raise ValueError('Use the existing pinned Transformers/safetensors environment')
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def runtime_versions(config):
    actual = {k: importlib.metadata.version(k) for k in config['dependencies']}
    if actual != config['dependencies']:
        raise ValueError('Pinned runtime versions differ: ' + repr(actual))
    return actual


def verify_files(cache_root):
    root = Path(cache_root)
    for name, size, digest in [('config.json', 879, MODEL_CONFIG_SHA256),
                               ('README.md', 8080, CARD_SHA256),
                               ('model.safetensors', WEIGHTS_SIZE, WEIGHTS_SHA256)]:
        path = root / name
        if not path.is_file() or path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError('Missing or changed pinned artifact: ' + name)
    cfg = json.loads((root / 'config.json').read_text())
    if any(cfg.get(k) != v for k, v in {'model_type': 'dinov2', 'hidden_size': 1536,
           'num_hidden_layers': 40, 'patch_size': 14, 'use_swiglu_ffn': True}.items()):
        raise ValueError('Official model architecture differs')
    return cfg


def build_transform():
    # Use the exact operator order and defaults of the pinned official model card.
    from torchvision.transforms import v2, InterpolationMode
    return v2.Compose([v2.Resize(224, interpolation=InterpolationMode.BILINEAR,
                                 antialias=True), v2.CenterCrop(224), v2.ToTensor(),
                       v2.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])


def classification_features(last_hidden_state):
    import torch
    if (last_hidden_state.ndim != 3 or last_hidden_state.shape[1:] != (257, 1536)
            or last_hidden_state.dtype != torch.float32
            or not torch.isfinite(last_hidden_state).all()):
        raise ValueError('Expected finite FP32 224/14 CLS+256-patch tokens')
    features = torch.cat([last_hidden_state[:, 0], last_hidden_state[:, 1:].mean(1)], dim=-1)
    if features.shape != (last_hidden_state.shape[0], 3072):
        raise ValueError('Invalid classification feature dimension')
    return features


def strict_load(model, state):
    import torch
    if not state or any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError('The official checkpoint must contain only FP32 tensors')
    receipt = model.load_state_dict(state, strict=True, assign=True)
    if receipt.missing_keys or receipt.unexpected_keys:
        raise ValueError('Full state was not loaded')
    return model.eval().requires_grad_(False)


def load_model(cache_root, device):
    import torch
    from safetensors.torch import load_file
    from transformers import Dinov2Config, Dinov2Model
    cfg = verify_files(cache_root)
    model_config = Dinov2Config.from_dict(cfg)
    model_config._attn_implementation = 'eager'
    # Meta construction avoids allocating and randomly initializing a second 4.5GB model.
    with torch.device('meta'):
        model = Dinov2Model(model_config)
    state = load_file(str(Path(cache_root) / 'model.safetensors'), device='cpu')
    model = strict_load(model, state)
    del state
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Model has unmaterialized or non-FP32 parameters')
    return model.to(device=device, dtype=torch.float32)


def select_rows(manifest, config, magnification, limit):
    if magnification not in MAGNIFICATIONS or limit not in (0, 4):
        raise ValueError('Use a full shard or a separate four-image technical smoke')
    if file_hash(manifest) != config['manifest_sha256']:
        raise ValueError('Manifest hash differs')
    rows = read_manifest(manifest)
    if len(rows) != config['expected_images']:
        raise ValueError('Full manifest coverage differs')
    for row in rows:
        path = PurePosixPath(row['relative_path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Manifest image path must remain relative to the image root')
    chosen = [r for r in rows if int(r['magnification']) == magnification]
    if len(chosen) < 4:
        raise ValueError('Magnification shard is incomplete')
    return chosen[:limit] if limit else chosen


def verify_output(target, metadata, expected_paths, encoder, manifest_hash, limit, config_hash):
    record = json.loads(Path(metadata).read_text())
    if (record.get('encoder') != encoder or record.get('manifest_sha256') != manifest_hash
            or record.get('config_sha256') != config_hash or record.get('smoke_limit') != limit
            or record.get('images') != len(expected_paths)
            or record.get('feature_sha256') != file_hash(target)):
        raise ValueError('Existing feature provenance differs')
    with np.load(target, allow_pickle=False) as saved:
        if (set(saved.files) != {'paths', 'features'} or saved['paths'].tolist() != expected_paths
                or saved['features'].shape != (len(expected_paths), 3072)
                or saved['features'].dtype != np.float32 or not np.isfinite(saved['features']).all()):
            raise ValueError('Saved feature coverage, shape, dtype or values differ')
    return record


def write_output(target, paths, features, record):
    target = Path(target)
    meta = target.with_suffix('.json')
    if target.exists() or meta.exists():
        raise ValueError('Refusing to overwrite an existing feature artifact')
    if (len(paths) != len(set(paths)) or features.shape != (len(paths), 3072)
            or features.dtype != np.float32 or not np.isfinite(features).all()):
        raise ValueError('Invalid feature archive')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.npz.part')
    with temp.open('xb') as stream:
        np.savez_compressed(stream, paths=np.asarray(paths), features=features)
        stream.flush()
        os.fsync(stream.fileno())
    # Linking is an atomic no-overwrite publication on the same filesystem.
    os.link(temp, target)
    temp.unlink()
    record = dict(record, feature_sha256=file_hash(target))
    temp_meta = meta.with_suffix('.json.part')
    with temp_meta.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    os.link(temp_meta, meta)
    temp_meta.unlink()
    # A saved artifact is accepted only after an actual NPZ reload.
    with np.load(target, allow_pickle=False) as saved:
        if saved['paths'].tolist() != paths or not np.array_equal(saved['features'], features):
            raise ValueError('Feature save/reload mismatch')
    return record


def assert_parity(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError('Parity shape/values differ')
    error = float(np.max(np.abs(reference - candidate)))
    if not np.allclose(reference, candidate, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise ValueError('Frozen encoder batch/reconstruction parity failed: ' + str(error))
    return error


def main():
    import torch
    from torch.utils.data import DataLoader
    from .features import Fields, identity_collate, worker_init
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cache-root', 'manifest', 'images-root', 'output-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--code-sha', required=True)
    p.add_argument('--magnification', choices=MAGNIFICATIONS, type=int, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--limit', type=int, default=0, choices=(0, 4))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run registered encoder extraction inside a scheduler allocation')
    if not re.fullmatch('[0-9a-f]{40}', args.code_sha) or args.batch_size < 1 or args.workers < 0:
        raise ValueError('Valid source SHA, batch size and worker count required')
    config = read_config(args.config)
    versions = runtime_versions(config)
    rows = select_rows(args.manifest, config, args.magnification, args.limit)
    paths = [r['relative_path'] for r in rows]
    target = args.output_root / ('midnight-12k-%d.npz' % args.magnification)
    meta = target.with_suffix('.json')
    cfg_hash = file_hash(args.config)
    if target.exists() or meta.exists():
        if not (target.exists() and meta.exists()):
            raise ValueError('Incomplete cache exists; use a separate output directory')
        verify_output(target, meta, paths, config['encoder'], config['manifest_sha256'], args.limit, cfg_hash)
        print('Verified existing feature archive', flush=True)
        return
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('The registered encoder extraction uses CUDA, not CPU model inference')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation is unavailable')
    torch.manual_seed(812)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    transform = build_transform()
    model = load_model(args.cache_root, device)
    loader = DataLoader(Fields(rows, args.images_root), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=identity_collate, worker_init_fn=worker_init,
                        persistent_workers=args.workers > 0,
                        multiprocessing_context='spawn' if args.workers else None)
    start = time.monotonic()
    outputs = []
    processed_paths = []
    parity = {}
    first_pixels = first_features = None
    def encode(pixels):
        return classification_features(model(pixel_values=pixels).last_hidden_state).cpu().numpy()
    with torch.inference_mode():
        for batch in loader:
            keys, images = zip(*batch)
            pixels = torch.stack([transform(im) for im in images]).to(device)
            features = encode(pixels)
            if not outputs:
                parity['repeat_batch_max_abs'] = assert_parity(features, encode(pixels))
                parity['single_batch_max_abs'] = assert_parity(features[:1], encode(pixels[:1]))
                if args.limit:
                    first_pixels, first_features = pixels.cpu(), features.copy()
            processed_paths.extend(keys)
            outputs.append(features)
            if len(processed_paths) % 256 < len(batch):
                print(json.dumps({'encoded': len(processed_paths), 'total': len(paths)}), flush=True)
        if args.limit:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            model = load_model(args.cache_root, device)
            parity['reconstructed_model_max_abs'] = assert_parity(first_features, encode(first_pixels.to(device)))
    if processed_paths != paths:
        raise ValueError('Data loader changed image order or coverage')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    record = {'manifest_sha256': config['manifest_sha256'], 'encoder': config['encoder'],
              'config_sha256': cfg_hash, 'code_sha': args.code_sha,
              'source_file_sha256': file_hash(__file__), 'magnification': args.magnification,
              'model_implementation': {'class': type(model).__name__, 'package': 'transformers',
                  'model_source_sha256': file_hash(inspect.getfile(type(model))),
                  'configuration_source_sha256': file_hash(inspect.getfile(type(model.config)))},
              'images': len(paths), 'smoke_limit': args.limit, 'elapsed_seconds': elapsed,
              'images_per_second': len(paths) / elapsed, 'precision': 'float32; AMP and TF32 disabled',
              'attention_implementation': 'eager', 'transform': TRANSFORM,
              'parity_atol': PARITY_ATOL, 'parity_rtol': PARITY_RTOL, 'parity': parity,
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
              'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
              'python': platform.python_version(), 'dependencies': versions,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'fit_status': 'frozen external pretrained encoder; no cohort fitting or labels used'}
    write_output(target, paths, np.concatenate(outputs), record)
    verify_output(target, meta, paths, config['encoder'], config['manifest_sha256'], args.limit, cfg_hash)
    print(json.dumps({'status': 'accepted frozen feature shard', 'images': len(paths), 'parity': parity}), flush=True)


if __name__ == '__main__':
    main()

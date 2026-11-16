"""Frozen Midnight native 448x700 inputs with automatic positional interpolation."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from . import midnight_features as base
from .coverage_features import select_rows
from .breakhis_predict import load_checked_image
from .data import MAGNIFICATIONS, file_hash

CONFIG_SHA256 = '71528e9b654b94bcfd880267b1640d882637580c17d19f00c8d161d2756cd1dc'
CROP_BOXES = {(700, 460): (0, 6, 700, 454), (700, 456): (0, 4, 700, 452)}


def read_config(path):
    if file_hash(path) != CONFIG_SHA256:
        raise ValueError('Registered native-input configuration bytes differ')
    config = json.loads(Path(path).read_text())
    root = Path(__file__).resolve().parents[1]
    source = root/config['base_extraction_config']
    if file_hash(source) != config['base_extraction_config_sha256']:
        raise ValueError('Accepted base extraction configuration differs')
    original = base.read_config(source)
    if (config['dependencies'] != original['dependencies']
            or any(config['encoder'][key] != original['encoder'][key]
                   for key in original['encoder'] if key not in ('features', 'transform'))):
        raise ValueError('Native-input frozen encoder/environment lineage differs')
    return config


def native_pixels(image):
    import torch
    from torchvision.transforms import v2
    if image.mode != 'RGB' or image.size not in CROP_BOXES:
        raise ValueError('Expected one of the audited native RGB geometries')
    box = CROP_BOXES[image.size]; cropped = image.crop(box)
    if cropped.size != (700, 448) or not np.array_equal(np.asarray(cropped), np.asarray(v2.CenterCrop((448, 700))(image))):
        raise ValueError('Native crop pixels differ from the exact registered box')
    value = v2.Compose([v2.ToTensor(), v2.Normalize((.5,)*3, (.5,)*3)])(cropped)
    if value.shape != (3, 448, 700) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Invalid native input tensor')
    return value, {'original': list(image.size), 'crop_box': list(box)}


def classification_features(last_hidden_state):
    """Keep this 1601-token contract separate from the accepted 224-pixel helper."""
    import torch
    if (last_hidden_state.ndim != 3 or last_hidden_state.shape[1:] != (1601, 1536)
            or last_hidden_state.dtype != torch.float32 or not torch.isfinite(last_hidden_state).all()):
        raise ValueError('Expected finite FP32 native CLS plus 1600 patch tokens')
    return torch.cat((last_hidden_state[:, 0], last_hidden_state[:, 1:].mean(1)), dim=1)


def check_model(model):
    expected = {'image_size': 518, 'patch_size': 14, 'hidden_size': 1536,
                'num_attention_heads': 24, 'num_hidden_layers': 40}
    if (model.training or any(parameter.requires_grad for parameter in model.parameters())
            or any(getattr(model.config, key) != value for key, value in expected.items())
            or model.config._attn_implementation != 'eager'):
        raise ValueError('Frozen original model configuration or execution state differs')


def encode_batches(encoder, pixels, batch_size=4):
    if batch_size != 4:
        raise ValueError('Use the registered native crop batch of four')
    result = []
    for start in range(0, len(pixels), batch_size):
        value = np.asarray(encoder(pixels[start:start+batch_size]))
        if value.shape != (min(batch_size, len(pixels)-start), 3072) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid native field descriptor')
        result.append(value)
    return np.concatenate(result)


def verify_output(target, paths, config, limit=0, code_sha=None):
    target = Path(target); record = json.loads(target.with_suffix('.json').read_text())
    expected = {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
                'config_sha256': CONFIG_SHA256, 'images': len(paths), 'smoke_limit': limit,
                'source_file_sha256': file_hash(__file__), 'base_source_file_sha256': file_hash(base.__file__),
                'dependencies': config['dependencies'], 'feature_sha256': file_hash(target),
                'base_extraction_config_sha256': config['base_extraction_config_sha256'],
                'input_shape': config['input_shape'], 'token_shape': config['expected_token_shape'],
                'model_image_size': 518, 'position_grid': [37, 37], 'patch_grid': [32, 50],
                'frozen_base': True, 'attention_implementation': 'eager',
                'precision': 'FP32; TF32 disabled', 'transform': config['encoder']['transform'],
                'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'],
                'native_forwards': len(paths) if not limit else config['smoke']['native_forwards']}
    if code_sha is not None: expected['code_sha'] = code_sha
    if record.get('magnification') not in MAGNIFICATIONS or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Native feature provenance differs')
    parity = record.get('parity')
    if limit:
        names = {'repeat_max_abs', 'single_batch_max_abs', 'reconstructed_model_max_abs'}
        if (not isinstance(parity, dict) or set(parity) != names
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) or value < 0 for value in parity.values())):
            raise ValueError('Registered native smoke parity receipt differs')
    elif parity != {}:
        raise ValueError('Full native extraction must carry its zero-extra-forward parity receipt')
    with np.load(target, allow_pickle=False) as saved:
        if (set(saved.files) != {'paths', 'features'} or saved['paths'].tolist() != paths
                or len(paths) != len(set(paths)) or saved['features'].shape != (len(paths), 3072)
                or saved['features'].dtype != np.float32 or not np.isfinite(saved['features']).all()):
            raise ValueError('Native feature path/shape/dtype/finite coverage differs')
    return record


def write_output(target, paths, features, record):
    target = Path(target); meta = target.with_suffix('.json')
    if target.exists() or meta.exists():
        raise ValueError('Refusing existing native feature artifacts')
    if (features.shape != (len(paths), 3072) or features.dtype != np.float32
            or not np.isfinite(features).all() or len(paths) != len(set(paths))):
        raise ValueError('Invalid native feature archive')
    target.parent.mkdir(parents=True, exist_ok=True); owned = []; published = []
    try:
        temporary = target.with_suffix('.npz.part')
        with temporary.open('xb') as stream:
            owned.append(temporary); np.savez_compressed(stream, paths=np.asarray(paths), features=features)
            stream.flush(); os.fsync(stream.fileno())
        record = dict(record, feature_sha256=file_hash(temporary))
        temporary_meta = meta.with_suffix('.json.part')
        with temporary_meta.open('x') as stream:
            owned.append(temporary_meta); stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        for source, destination in zip(owned, (target, meta)):
            os.link(source, destination); published.append(destination)
        with np.load(target, allow_pickle=False) as saved:
            if saved['paths'].tolist() != paths or not np.array_equal(saved['features'], features):
                raise ValueError('Actual native archive reload differs')
    except Exception:
        for path in published: path.unlink()
        raise
    finally:
        for path in owned:
            if path.exists(): path.unlink()
    return record


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cache-root', 'manifest', 'images-root', 'output-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--limit', type=int, choices=(0, 4), default=0)
    p.add_argument('--batch-size', type=int, default=4); p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run native extraction in a CUDA scheduler allocation')
    if args.batch_size != 4 or args.workers != 0 or not re.fullmatch('[0-9a-f]{40}', args.code_sha):
        raise ValueError('Registered native batch/workers/source definition differs')
    if (Path(__file__).resolve().parents[1]/'.source-sha').read_text().strip() != args.code_sha:
        raise ValueError('Immutable native extraction source receipt differs')
    config = read_config(args.config); base.runtime_versions(config)
    rows, selected = select_rows(args.manifest, config, args.magnification, args.limit)
    paths = [row['relative_path'] for row in selected]
    target = args.output_root/('%s-%d.npz'%(config['encoder_name'], args.magnification))
    if target.exists() or target.with_suffix('.json').exists():
        verify_output(target, paths, config, args.limit, args.code_sha)
        print('Verified existing native shard'); return
    torch.manual_seed(812); torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    model = base.load_model(args.cache_root, torch.device('cuda')); check_model(model)
    started = time.monotonic(); output = []; geometries = []; smoke_pixels = None
    def encode(pixels):
        tokens = model(pixel_values=pixels.to('cuda'), output_attentions=False,
                       output_hidden_states=False, return_dict=True).last_hidden_state
        return classification_features(tokens).cpu().numpy()
    with torch.inference_mode():
        for start in range(0, len(selected), 4):
            values = [native_pixels(load_checked_image(args.images_root, row)) for row in selected[start:start+4]]
            pixels = torch.stack([value[0] for value in values]); geometries.extend(value[1] for value in values)
            output.append(encode_batches(encode, pixels))
            if args.limit: smoke_pixels = pixels
            processed = start + len(values)
            if not args.limit and (processed % 200 == 0 or processed == len(selected)):
                print(json.dumps({'magnification': args.magnification, 'processed': processed,
                                  'total': len(selected), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    features = np.concatenate(output); parity = {}; steady_batch_seconds = None
    if args.limit:
        with torch.inference_mode():
            repeated_start = time.monotonic()
            repeated = encode_batches(encode, smoke_pixels)
            torch.cuda.synchronize()
            steady_batch_seconds = time.monotonic()-repeated_start
            parity['repeat_max_abs'] = base.assert_parity(features, repeated)
            singles = np.concatenate([encode(smoke_pixels[index:index+1]) for index in range(len(smoke_pixels))])
            parity['single_batch_max_abs'] = base.assert_parity(features, singles)
        del model; gc.collect(); torch.cuda.empty_cache()
        model = base.load_model(args.cache_root, torch.device('cuda')); check_model(model)
        with torch.inference_mode():
            fresh = encode_batches(encode, smoke_pixels)
        parity['reconstructed_model_max_abs'] = base.assert_parity(features, fresh)
    torch.cuda.synchronize()
    count = len(paths) if not args.limit else config['smoke']['native_forwards']
    elapsed = time.monotonic()-started
    record = {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'], 'config_sha256': CONFIG_SHA256,
              'code_sha': args.code_sha, 'source_file_sha256': file_hash(__file__),
              'base_source_file_sha256': file_hash(base.__file__), 'magnification': args.magnification,
              'images': len(paths), 'smoke_limit': args.limit, 'dependencies': config['dependencies'],
              'base_extraction_config_sha256': config['base_extraction_config_sha256'],
              'input_shape': config['input_shape'], 'token_shape': config['expected_token_shape'],
              'model_image_size': 518, 'position_grid': [37, 37], 'patch_grid': [32, 50], 'frozen_base': True,
              'transform': config['encoder']['transform'], 'attention_implementation': 'eager', 'precision': 'FP32; TF32 disabled',
              'geometry_counts': {'%dx%d'%shape: sum(tuple(item['original']) == shape for item in geometries) for shape in CROP_BOXES},
              'native_forwards': count, 'parity': parity, 'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'],
              'elapsed_seconds': elapsed, 'native_forwards_per_second': count/elapsed, 'gpu': torch.cuda.get_device_name(),
              'steady_batch_seconds': steady_batch_seconds,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    write_output(target, paths, features, record); verify_output(target, paths, config, args.limit, args.code_sha)
    print(json.dumps({'status': 'accepted frozen native-input shard', 'images': len(paths), 'parity': parity}))


if __name__ == '__main__':
    main()

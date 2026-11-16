"""Frozen, hash-pinned GenBio RGB-channel CLS features; no remote code loader."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

from . import midnight_features as base
from .coverage_features import select_rows
from .breakhis_predict import load_checked_image
from .data import MAGNIFICATIONS, file_hash

CONFIG_SHA256 = '4ab5fbcaa5714871d49ec41c70f7e3f667ae3802d14585a014a5ca24e3e720ca'
DIMENSION = 4608
PRECISION = 'FP32 parameters/inputs; official BF16 RoPE; no autocast; TF32 disabled'
ATTENTION = 'torch_scaled_dot_product_attention'
ARTIFACTS = {
    'model.safetensors': (4534810816, '1bb298ba3431352ad6d7b85a2c34874b42db05ef8749079a7999fa8c37cd4461'),
    'config.json': (993, 'ed5f1ecf200bc3bfcef37270ac73560d2633a7dbe31ec143743c25c120705c1f'),
    'configuration_genbio_pathfm.py': (2222, '30b13fd9e13f1795f640268ed0b72600762c17e65de897489cfd69e638c2d4f0'),
    'modeling_genbio_pathfm.py': (25633, 'c46e835b83475c86f7e87a11458d35ed55a44c0335916d3467bae57f69c9d6c5'),
    'LICENSE.txt': (10678, 'd94523de4cd4ce315aec6bcc2efc2ad034cc987a1a36f95dcc57e4986450c2ba'),
    'README.md': (3228, '262129d34b6c630d3cd53ae8ba7d71893b28852e795552581eba503ad7a110b3'),
}
ARCHITECTURE = {'img_size': 224, 'patch_size': 16, 'in_chans': 1, 'embed_dim': 1536,
                'depth': 40, 'num_heads': 24, 'n_storage_tokens': 4,
                'ffn_layer': 'swiglu64', 'pos_embed_rope_dtype': 'bf16'}


def read_config(path):
    if file_hash(path) != CONFIG_SHA256:
        raise ValueError('Registered GenBio configuration bytes differ')
    config = json.loads(Path(path).read_text())
    encoder = config['encoder']
    if (encoder['repository'] != 'genbio-ai/genbio-pathfm'
            or encoder['revision'] != '6eab2a4ea6fbaee16e5f193187952042ebe5d3ec'
            or encoder['dimension'] != DIMENSION or config['output_dimension'] != DIMENSION
            or config['input_shape'] != [3, 224, 224]
            or config['expected_token_shape_per_channel'] != [201, 1536]
            or config['channels_per_image'] != 3
            or config['parity'] != {'atol': 1e-4, 'rtol': 1e-4}):
        raise ValueError('Registered GenBio representation differs')
    return config


def verify_files(cache_root):
    root = Path(cache_root)
    for name, (size, digest) in ARTIFACTS.items():
        path = root/name
        if not path.is_file() or path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError('Missing or changed pinned GenBio artifact: '+name)
    config = json.loads((root/'config.json').read_text())
    if any(config.get(key) != value for key, value in ARCHITECTURE.items()):
        raise ValueError('Official GenBio architecture differs')
    return config


def import_local_sources(cache_root):
    """Import only the two already byte-verified official modules as a local package."""
    root = Path(cache_root).resolve()
    for name in ('configuration_genbio_pathfm.py', 'modeling_genbio_pathfm.py'):
        size, digest = ARTIFACTS[name]
        if (root/name).stat().st_size != size or file_hash(root/name) != digest:
            raise ValueError('Pinned local GenBio source differs: '+name)
    package_name = '_magset_genbio_'+ARTIFACTS['modeling_genbio_pathfm.py'][1][:16]
    package = types.ModuleType(package_name); package.__path__ = [str(root)]
    sys.modules[package_name] = package
    modules = []
    try:
        for stem in ('configuration_genbio_pathfm', 'modeling_genbio_pathfm'):
            name = package_name+'.'+stem
            spec = importlib.util.spec_from_file_location(name, root/(stem+'.py'))
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module
            # Compile verified source bytes directly, bypassing any unverified .pyc.
            source = (root/(stem+'.py')).read_bytes()
            if hashlib.sha256(source).hexdigest() != ARTIFACTS[stem+'.py'][1]:
                raise ValueError('Local GenBio source changed before import')
            exec(compile(source, str(root/(stem+'.py')), 'exec'), module.__dict__)
            modules.append(module)
    except Exception:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name+'.'):
                del sys.modules[name]
        raise
    return modules


def check_model(model):
    import torch
    if (model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters())
            or any(getattr(model.config, key) != value for key, value in ARCHITECTURE.items())
            or model.backbone.rope_embed.periods.dtype != torch.bfloat16):
        raise ValueError('Frozen GenBio configuration/parameter/RoPE precision differs')


def load_model(cache_root, device):
    import torch
    from safetensors.torch import load_file
    raw = verify_files(cache_root)
    configuration, modeling = import_local_sources(cache_root)
    model = modeling.GenBioPathFMModel(configuration.GenBioPathFMConfig(**raw))
    state = load_file(str(Path(cache_root)/'model.safetensors'), device='cpu')
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[key].shape != value.shape for key, value in expected.items()):
        raise ValueError('GenBio strict full-state keys/shapes differ')
    if any(state[name].dtype != torch.float32 for name, _ in model.named_parameters()):
        raise ValueError('GenBio checkpoint parameters must be FP32')
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('GenBio strict state load was incomplete')
    if any(not torch.equal(parameter.detach(), state[name]) for name, parameter in model.named_parameters()):
        raise ValueError('GenBio exact FP32 parameter copy differs')
    receipt = {'strict_full_state': True, 'exact_parameter_copy': True,
               'state_tensors': len(state), 'parameter_count': sum(p.numel() for p in model.parameters()),
               'rope_periods_dtype': str(model.backbone.rope_embed.periods.dtype)}
    del state, expected; gc.collect()
    # Preserve the official BF16 rotary buffer: never call model.float().
    model.eval().requires_grad_(False).to(device)
    check_model(model); model._genbio_loader_receipt = receipt
    return model


def genbio_pixels(image):
    import torch
    from torchvision import transforms
    if image.mode != 'RGB' or image.size not in ((700, 460), (700, 456)):
        raise ValueError('Expected audited GenBio RGB source geometry')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize((.697, .575, .728), (.188, .240, .187))])
    pixels = transform(image).contiguous()
    if pixels.shape != (3, 224, 224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
        raise ValueError('Invalid GenBio input tensor')
    return pixels, {'original': list(image.size), 'resized': [224, 224], 'crop_box': None}


def classification_features(model, pixels):
    import torch
    if (pixels.ndim != 4 or pixels.shape[1:] != (3, 224, 224) or not pixels.is_contiguous()
            or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all()):
        raise ValueError('GenBio input must be finite contiguous FP32 B3HW')
    value = model(pixels)
    if value.shape != (len(pixels), DIMENSION) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Expected finite FP32 official RGB-channel CLS features')
    return value


def encode_batches(encoder, pixels, batch_size=4):
    if batch_size != 4:
        raise ValueError('Use the registered GenBio batch of four')
    result = []
    for start in range(0, len(pixels), batch_size):
        value = np.asarray(encoder(pixels[start:start+batch_size]))
        if value.shape != (min(batch_size, len(pixels)-start), DIMENSION) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid GenBio field descriptor')
        result.append(value)
    return np.concatenate(result)


def metadata(config, paths, magnification, limit, code_sha):
    image_forwards = len(paths) if not limit else config['smoke']['image_forwards']
    return {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
            'config_sha256': CONFIG_SHA256, 'code_sha': code_sha, 'source_file_sha256': file_hash(__file__),
            'dependencies': config['dependencies'], 'magnification': magnification, 'images': len(paths),
            'smoke_limit': limit, 'input_shape': config['input_shape'],
            'token_shape_per_channel': config['expected_token_shape_per_channel'],
            'patch_grid': [14, 14], 'channels_per_image': 3, 'output_dimension': DIMENSION,
            'official_rope_dtype': 'bfloat16', 'frozen_base': True, 'attention_implementation': ATTENTION,
            'precision': PRECISION, 'transform': config['encoder']['transform'],
            'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'],
            'image_forwards': image_forwards, 'channel_forwards': 3*image_forwards}


def verify_output(target, paths, config, limit=0, code_sha=None):
    target = Path(target); record = json.loads(target.with_suffix('.json').read_text())
    expected = metadata(config, paths, record.get('magnification'), limit, record.get('code_sha') if code_sha is None else code_sha)
    expected['feature_sha256'] = file_hash(target)
    if record.get('magnification') not in MAGNIFICATIONS or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('GenBio feature provenance differs')
    receipt = record.get('loader_receipt', {})
    if (receipt.get('strict_full_state') is not True or receipt.get('exact_parameter_copy') is not True
            or receipt.get('rope_periods_dtype') != 'torch.bfloat16'
            or any(isinstance(receipt.get(key), bool) or not isinstance(receipt.get(key), int)
                   or receipt[key] <= 0 for key in ('state_tensors', 'parameter_count'))):
        raise ValueError('GenBio strict-load receipt differs')
    elapsed = record.get('elapsed_seconds')
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed <= 0
            or any(isinstance(record.get(k), bool) or not isinstance(record.get(k), int) or record[k] <= 0
                   for k in ('peak_gpu_bytes', 'peak_reserved_gpu_bytes'))):
        raise ValueError('GenBio runtime/memory receipt differs')
    steady = record.get('steady_batch_seconds')
    if (limit and (isinstance(steady, bool) or not isinstance(steady, (int, float)) or not np.isfinite(steady) or steady <= 0)) or (not limit and steady is not None):
        raise ValueError('GenBio steady batch timing receipt differs')
    parity = record.get('parity')
    if limit:
        names = {'repeat_max_abs', 'single_batch_max_abs', 'reconstructed_model_max_abs'}
        if not isinstance(parity, dict) or set(parity) != names or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in parity.values()):
            raise ValueError('GenBio smoke parity receipt differs')
    elif parity != {}:
        raise ValueError('Full GenBio extraction must not claim extra replay forwards')
    with np.load(target, allow_pickle=False) as saved:
        if set(saved.files) != {'paths', 'features'} or saved['paths'].tolist() != paths or len(set(paths)) != len(paths) or saved['features'].shape != (len(paths), DIMENSION) or saved['features'].dtype != np.float32 or not np.isfinite(saved['features']).all():
            raise ValueError('GenBio feature path/shape/dtype/finite coverage differs')
    return record


def write_output(target, paths, features, record):
    target = Path(target); meta = target.with_suffix('.json')
    if target.exists() or meta.exists():
        raise ValueError('Refusing existing GenBio feature artifacts')
    if features.shape != (len(paths), DIMENSION) or features.dtype != np.float32 or not np.isfinite(features).all() or len(set(paths)) != len(paths):
        raise ValueError('Invalid GenBio feature archive')
    target.parent.mkdir(parents=True, exist_ok=True); owned = []; published = []
    try:
        temporary = target.with_suffix('.npz.part')
        with temporary.open('xb') as stream:
            owned.append(temporary); np.savez_compressed(stream, paths=np.asarray(paths), features=features)
            stream.flush(); os.fsync(stream.fileno())
        record = dict(record, feature_sha256=file_hash(temporary)); temporary_meta = meta.with_suffix('.json.part')
        with temporary_meta.open('x') as stream:
            owned.append(temporary_meta); stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        for source, destination in zip(owned, (target, meta)):
            os.link(source, destination); published.append(destination)
        with np.load(target, allow_pickle=False) as saved:
            if saved['paths'].tolist() != paths or not np.array_equal(saved['features'], features):
                raise ValueError('Actual GenBio archive reload differs')
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
        raise RuntimeError('Run GenBio extraction in a CUDA scheduler allocation')
    if args.batch_size != 4 or args.workers != 0 or not re.fullmatch('[0-9a-f]{40}', args.code_sha):
        raise ValueError('Registered GenBio batch/workers/source definition differs')
    if (Path(__file__).resolve().parents[1]/'.source-sha').read_text().strip() != args.code_sha:
        raise ValueError('Immutable GenBio extraction source receipt differs')
    config = read_config(args.config); base.runtime_versions(config)
    _, selected = select_rows(args.manifest, config, args.magnification, args.limit)
    paths = [row['relative_path'] for row in selected]
    target = args.output_root/('%s-%d.npz'%(config['encoder_name'], args.magnification))
    if target.exists() or target.with_suffix('.json').exists():
        verify_output(target, paths, config, args.limit, args.code_sha); print('Verified existing GenBio shard'); return
    torch.manual_seed(812); torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    model = load_model(args.cache_root, torch.device('cuda')); check_model(model)
    receipt = model._genbio_loader_receipt; started = time.monotonic(); output = []; geometries = []; smoke_pixels = None
    def encode(pixels):
        return classification_features(model, pixels.to('cuda')).cpu().numpy()
    with torch.inference_mode():
        for start in range(0, len(selected), 4):
            values = [genbio_pixels(load_checked_image(args.images_root, row)) for row in selected[start:start+4]]
            pixels = torch.stack([value[0] for value in values]); geometries.extend(value[1] for value in values)
            output.append(encode_batches(encode, pixels))
            if args.limit: smoke_pixels = pixels
            processed = start+len(values)
            if not args.limit and (processed % 200 == 0 or processed == len(selected)):
                print(json.dumps({'magnification': args.magnification, 'processed': processed, 'total': len(selected), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    features = np.concatenate(output); parity = {}; steady_batch_seconds = None
    if args.limit:
        with torch.inference_mode():
            repeated_start = time.monotonic(); repeated = encode_batches(encode, smoke_pixels); torch.cuda.synchronize()
            steady_batch_seconds = time.monotonic()-repeated_start
            parity['repeat_max_abs'] = base.assert_parity(features, repeated)
            singles = np.concatenate([encode(smoke_pixels[index:index+1]) for index in range(len(smoke_pixels))])
            parity['single_batch_max_abs'] = base.assert_parity(features, singles)
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_model(args.cache_root, torch.device('cuda')); check_model(model)
        if model._genbio_loader_receipt != receipt: raise ValueError('Fresh GenBio loader receipt differs')
        with torch.inference_mode(): fresh = encode_batches(encode, smoke_pixels)
        parity['reconstructed_model_max_abs'] = base.assert_parity(features, fresh)
    torch.cuda.synchronize(); elapsed = time.monotonic()-started
    record = metadata(config, paths, args.magnification, args.limit, args.code_sha)
    record.update({'loader_receipt': receipt, 'geometry_counts': {shape: sum('%dx%d'%tuple(item['original']) == shape for item in geometries) for shape in config['expected_shapes']},
                   'parity': parity, 'elapsed_seconds': elapsed, 'steady_batch_seconds': steady_batch_seconds,
                   'channel_forwards_per_second': record['channel_forwards']/elapsed, 'gpu': torch.cuda.get_device_name(),
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(),
                   'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    write_output(target, paths, features, record); verify_output(target, paths, config, args.limit, args.code_sha)
    print(json.dumps({'status': 'accepted frozen GenBio shard', 'images': len(paths), 'parity': parity}))


if __name__ == '__main__':
    main()

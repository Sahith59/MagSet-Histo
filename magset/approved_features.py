"""Frozen local-timm features for three approved, independently pinned encoders.

Weights and model-card receipts remain in private caches. Construction never uses
an HF hub loader, downloaded Python, or trust_remote_code. One image per model
call makes descriptors independent of host I/O batch partitions.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time

import numpy as np

from . import midnight_features as base
from .breakhis_predict import load_checked_image
from .data import MAGNIFICATIONS, file_hash, read_manifest

REGISTRATIONS = {
    'MahmoodLab/UNI2-h': ('uni2-h-v1', 'd517a8dd47902dd7c308b3c36f63bce47e7b9a43', 1536, 'cls'),
    'paige-ai/Virchow2': ('virchow2-v1', '3158645804b69e3f3bc4439d4116edddf0840a72', 2560, 'cls_plus_mean_patch_excluding_registers'),
    'bioptimus/H-optimus-0': ('h-optimus-0-v1', 'b145cc1e6c6b30d3251aa8b1f844e6974188a743', 1536, 'cls'),
}
HOST_BATCH_SIZE = 4
MODEL_IMAGE_BATCH_SIZE = 1
BATCH_POLICY = 'one original image per model call; host batches4 only I/O'
BATCH_PARITY_SCOPE = 'outer host-batch partition invariance under canonical one-image model calls'
PRECISION = 'FP32 parameters/inputs; no autocast; TF32 disabled'
PARITY = {'atol': 1e-4, 'rtol': 1e-4}
ATTENTION = 'eager'


def read_config(path):
    """Accept only an exact copy of one immutable source registration."""
    config = json.loads(Path(path).read_text())
    repository = config.get('encoder', {}).get('repository')
    if repository not in REGISTRATIONS:
        raise ValueError('Unregistered approved encoder repository')
    name, revision, dimension, readout = REGISTRATIONS[repository]
    canonical = Path(__file__).resolve().parents[1]/'configs'/(name+'.json')
    if file_hash(path) != file_hash(canonical):
        raise ValueError('Registered approved-encoder configuration bytes differ')
    encoder = config['encoder']; execution = config['execution']
    if (config.get('name') != name or config.get('encoder_name') != name
            or config.get('schema_version') != 1 or encoder.get('revision') != revision
            or encoder.get('dimension') != dimension or config.get('output_dimension') != dimension
            or config.get('readout') != readout or config.get('input_shape') != [3, 224, 224]
            or config.get('expected_images') != 7909 or config.get('expected_shapes') != {'700x460': 7835, '700x456': 74}
            or config.get('frozen_base') is not True
            or execution.get('batch_size') != HOST_BATCH_SIZE or execution.get('workers') != 0
            or execution.get('model_image_batch_size') != MODEL_IMAGE_BATCH_SIZE
            or execution.get('model_batch_policy') != BATCH_POLICY
            or execution.get('attention_implementation') != ATTENTION
            or execution.get('dtype') != 'float32' or execution.get('tf32') is not False
            or execution.get('training') is not False or config.get('parity') != PARITY):
        raise ValueError('Registered approved representation/execution differs')
    if not re.fullmatch('[0-9a-f]{64}', config.get('manifest_sha256', '')):
        raise ValueError('Pinned manifest hash required')
    transform = encoder['transform']
    if (type(transform.get('resize_shorter_side')) is not int or transform['resize_shorter_side'] < 224
            or transform.get('center_crop') != 224 or transform.get('antialias') is not True
            or transform.get('interpolation') not in ('bilinear', 'bicubic')
            or any(not isinstance(transform.get(k), list) or len(transform[k]) != 3
                   or not np.isfinite(transform[k]).all() for k in ('mean', 'std'))
            or any(x <= 0 for x in transform['std'])):
        raise ValueError('Registered transform is incomplete')
    if (not isinstance(config.get('architecture'), dict)
            or not isinstance(config['architecture'].get('name'), str)
            or not isinstance(config['architecture'].get('kwargs'), dict)):
        raise ValueError('Explicit local timm construction required')
    if config.get('expected_token_shape') != ([261, 1280] if repository == 'paige-ai/Virchow2' else []):
        raise ValueError('Registered official model-output shape differs')
    artifact_pins(config)
    if not all(isinstance(config.get('dependencies', {}).get(k), str)
               for k in ('torch', 'torchvision', 'timm', 'safetensors', 'Pillow')):
        raise ValueError('Pinned approved-encoder dependencies required')
    smoke = config.get('smoke', {})
    if (smoke.get('images') != 4 or smoke.get('magnification') != 40 or smoke.get('image_forwards') != 16
            or not isinstance(smoke.get('relative_path_sha256'), list) or len(smoke['relative_path_sha256']) != 4
            or len(set(smoke['relative_path_sha256'])) != 4
            or any(not re.fullmatch('[0-9a-f]{64}', x) for x in smoke['relative_path_sha256']) or not isinstance(smoke.get('role'), dict)
            or smoke['role'].get('seed') != 812 or smoke['role'].get('trial') != 1
            or not re.fullmatch('[0-9a-f]{64}', smoke['role'].get('protocol_sha256', ''))):
        raise ValueError('Predeclared fitting-role technical smoke required')
    return config


def artifact_pins(config):
    pins = config.get('artifacts')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Pinned local artifacts required')
    for name, pin in pins.items():
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..')
                or not isinstance(pin, dict) or type(pin.get('size_bytes')) is not int or pin['size_bytes'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', pin.get('sha256', ''))):
            raise ValueError('Invalid local artifact pin')
    encoder = config['encoder']; weights = encoder['weights_file']
    if weights not in ('model.safetensors', 'pytorch_model.bin') or pins.get(weights) != {
            'size_bytes': encoder['weights_size_bytes'], 'sha256': encoder['weights_sha256']}:
        raise ValueError('Weight artifact pin differs from encoder definition')
    return pins


def verify_files(cache_root, config):
    root = Path(cache_root)
    for name, pin in artifact_pins(config).items():
        path = root/name
        if not path.is_file() or path.stat().st_size != pin['size_bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Missing or changed pinned approved artifact: '+name)
    return True


def select_rows(manifest, config, magnification, limit):
    if magnification not in MAGNIFICATIONS or limit not in (0, 4) or (limit and magnification != 40):
        raise ValueError('Use a full shard or the registered fitting-only four-image 40x smoke')
    if file_hash(manifest) != config['manifest_sha256']:
        raise ValueError('Registered manifest differs')
    rows = read_manifest(manifest); counts = {}; paths = set()
    for row in rows:
        key = row['relative_path']; path = PurePosixPath(key)
        if path.is_absolute() or '..' in path.parts or key in paths:
            raise ValueError('Manifest image path escapes root or is duplicated')
        paths.add(key); shape = '%sx%s'%(row['width'], row['height'])
        counts[shape] = counts.get(shape, 0)+1
        if int(row['magnification']) not in MAGNIFICATIONS:
            raise ValueError('Unexpected manifest magnification')
    if len(rows) != config['expected_images'] or counts != config['expected_shapes']:
        raise ValueError('Full release image count or geometry differs')
    chosen = [row for row in rows if int(row['magnification']) == magnification]
    if not chosen:
        raise ValueError('Required magnification is absent')
    if limit:
        lookup = {hashlib.sha256(row['relative_path'].encode()).hexdigest(): row for row in chosen}
        try:
            chosen = [lookup[key] for key in config['smoke']['relative_path_sha256']]
        except KeyError as error:
            raise ValueError('Registered fitting-only smoke path is absent') from error
        if sorted(int(row['height']) for row in chosen) != [456, 456, 460, 460]:
            raise ValueError('Technical smoke requires two fields of each audited height')
    return rows, chosen


def _construction(config):
    import torch
    import timm
    kwargs = dict(config['architecture']['kwargs'])
    if 'pretrained' in kwargs or any(key in kwargs for key in ('pretrained_cfg', 'pretrained_cfg_overlay', 'checkpoint_path')):
        raise ValueError('Local construction must not fetch or replace pinned weights')
    for key, value in (('mlp_layer', 'SwiGLUPacked'), ('act_layer', 'SiLU')):
        if key in kwargs:
            if kwargs[key] != value:
                raise ValueError('Unregistered executable timm construction argument')
            kwargs[key] = timm.layers.SwiGLUPacked if key == 'mlp_layer' else torch.nn.SiLU
    return timm.create_model(config['architecture']['name'], pretrained=False, **kwargs)


def strict_load(model, state):
    """Exact full-state load on a meta construction, then explicit FP32 evaluation."""
    import torch
    expected = model.state_dict()
    if (not isinstance(state, dict) or not state or set(state) != set(expected)
            or any(not isinstance(state[key], torch.Tensor) or state[key].shape != value.shape
                   for key, value in expected.items())):
        raise ValueError('Approved encoder strict full-state keys/shapes differ')
    parameter_names = {name for name, _ in model.named_parameters()}
    if any(not state[name].is_floating_point() or not torch.isfinite(state[name]).all() for name in parameter_names):
        raise ValueError('Approved encoder checkpoint parameters must be finite floating tensors')
    receipt = model.load_state_dict(state, strict=True, assign=True)
    if receipt.missing_keys or receipt.unexpected_keys:
        raise ValueError('Approved encoder strict state load was incomplete')
    model.to(dtype=torch.float32)
    if any(parameter.is_meta or not torch.equal(parameter.detach(), state[name].to(dtype=torch.float32))
           for name, parameter in model.named_parameters()):
        raise ValueError('Approved encoder exact FP32 parameter copy differs')
    result = {'strict_full_state': True, 'exact_parameter_copy': True,
              'state_tensors': len(state), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'checkpoint_parameter_dtypes': sorted({str(state[name].dtype) for name in parameter_names})}
    return model.eval().requires_grad_(False), result


def check_model(model, config):
    import torch
    if (model.training or any(p.requires_grad or p.is_meta or p.dtype != torch.float32 for p in model.parameters())
            or any(value.is_meta or (value.is_floating_point() and value.dtype != torch.float32)
                   for value in model.buffers())
            or any(getattr(module, 'fused_attn', False) for module in model.modules())):
        raise ValueError('Frozen FP32 eager-attention approved model differs')
    # The registered local constructor and complete pinned state establish the
    # architecture; these checks catch runtime changes in pooling/prefix layout.
    expected_features = 1280 if config['encoder']['repository'] == 'paige-ai/Virchow2' else 1536
    if getattr(model, 'num_features', None) != expected_features:
        raise ValueError('Approved model embedding dimension differs')
    if config['encoder']['repository'] == 'paige-ai/Virchow2' and getattr(model, 'num_prefix_tokens', None) != 5:
        raise ValueError('Virchow2 requires one CLS plus four register tokens')


def load_model(cache_root, device, config):
    import torch
    # Import optional runtimes before the meta context: import-time Torch
    # scalar probes in torchvision/Dynamo must execute on ordinary CPU tensors.
    import timm
    verify_files(cache_root, config)
    # timm1.0.15 creates the small stochastic-depth scalar schedule with
    # torch.linspace(...).item(), which cannot run on meta. Keep only that
    # schedule on CPU; all model tensors stay allocation-free meta tensors.
    # This mode is scoped to construction and never changes encoder arithmetic.
    class CpuScalarSchedule(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.linspace:
                kwargs['device'] = 'cpu'
            return func(*args, **kwargs)
    with torch.device('meta'), CpuScalarSchedule():
        model = _construction(config)
    weights = Path(cache_root)/config['encoder']['weights_file']
    if weights.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(weights), device='cpu')
    else:
        state = torch.load(str(weights), map_location='cpu', weights_only=True, mmap=True)
    model, receipt = strict_load(model, state)
    del state; gc.collect()
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False
    model.to(device=device)
    check_model(model, config); model._approved_loader_receipt = receipt
    return model


def approved_pixels(image, config):
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    if image.mode != 'RGB' or image.size not in ((700, 460), (700, 456)):
        raise ValueError('Expected audited approved-encoder RGB source geometry')
    spec = config['encoder']['transform']
    interpolation = {'bilinear': InterpolationMode.BILINEAR, 'bicubic': InterpolationMode.BICUBIC}[spec['interpolation']]
    resized = transforms.Resize(spec['resize_shorter_side'], interpolation=interpolation, antialias=True)(image)
    left = int(round((resized.width-224)/2)); top = int(round((resized.height-224)/2))
    cropped = transforms.CenterCrop(224)(resized)
    pixels = transforms.Normalize(spec['mean'], spec['std'])(transforms.ToTensor()(cropped)).contiguous()
    if pixels.shape != (3, 224, 224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
        raise ValueError('Invalid approved-encoder input tensor')
    return pixels, {'original': list(image.size), 'resized': list(resized.size), 'crop_box': [left, top, left+224, top+224]}


def classification_features(model, pixels, config):
    """Use the official representation with canonical one-image model calls."""
    import torch
    if (pixels.ndim != 4 or not len(pixels) or pixels.shape[1:] != (3, 224, 224) or not pixels.is_contiguous()
            or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all()):
        raise ValueError('Approved input must be finite contiguous FP32 B3HW')
    rows = []; dimension = config['output_dimension']
    for index in range(len(pixels)):
        output = model(pixels[index:index+1])
        expected = (1, *config['expected_token_shape']) if config['expected_token_shape'] else (1, dimension)
        if not isinstance(output, torch.Tensor) or output.shape != expected or output.dtype != torch.float32 or not torch.isfinite(output).all():
            raise ValueError('Official approved model-output shape/dtype/finite contract differs')
        if config['readout'] == 'cls':
            value = output
        elif config['readout'] == 'cls_plus_mean_patch_excluding_registers':
            value = torch.cat((output[:, 0], output[:, 5:].mean(1)), dim=1)
        else:
            raise ValueError('Unregistered approved encoder readout')
        if value.shape != (1, dimension) or not torch.isfinite(value).all():
            raise ValueError('Invalid official approved field descriptor')
        rows.append(value)
    return torch.cat(rows, dim=0)


def encode_batches(encoder, pixels, config, batch_size=HOST_BATCH_SIZE):
    if batch_size != HOST_BATCH_SIZE or not len(pixels):
        raise ValueError('Use the registered nonempty host batch partition')
    rows = []
    for start in range(0, len(pixels), batch_size):
        value = np.asarray(encoder(pixels[start:start+batch_size]))
        if value.shape != (min(batch_size, len(pixels)-start), config['output_dimension']) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid approved field descriptor')
        rows.append(value)
    return np.concatenate(rows)


def assert_parity(original, replay, config):
    if original.shape != replay.shape or not np.isfinite(original).all() or not np.isfinite(replay).all():
        raise ValueError('Approved parity shape or finite contract differs')
    if not np.allclose(original, replay, **config['parity']):
        raise ValueError('Approved technical parity exceeds predeclared tolerances')
    return float(np.abs(original-replay).max())


def metadata(config, paths, magnification, limit, code_sha):
    return {'encoder': config['encoder'], 'manifest_sha256': config['manifest_sha256'],
            'config_sha256': file_hash(Path(__file__).resolve().parents[1]/'configs'/(config['name']+'.json')),
            'code_sha': code_sha, 'source_file_sha256': file_hash(__file__),
            'dependencies': config['dependencies'], 'magnification': magnification, 'images': len(paths),
            'smoke_limit': limit, 'input_shape': config['input_shape'], 'token_shape': config['expected_token_shape'],
            'output_dimension': config['output_dimension'], 'frozen_base': True,
            'architecture': config['architecture'], 'artifacts': config['artifacts'], 'readout': config['readout'],
            'attention_implementation': ATTENTION, 'precision': PRECISION,
            'model_image_batch_size': MODEL_IMAGE_BATCH_SIZE, 'host_batch_size': HOST_BATCH_SIZE,
            'model_batch_policy': BATCH_POLICY, 'batch_parity_scope': BATCH_PARITY_SCOPE,
            'transform': config['encoder']['transform'],
            'parity_atol': config['parity']['atol'], 'parity_rtol': config['parity']['rtol'],
            'image_forwards': len(paths) if not limit else config['smoke']['image_forwards'],
            'technical_smoke': config['smoke'] if limit else None}


def verify_output(target, paths, config, limit=0, code_sha=None):
    target = Path(target); record = json.loads(target.with_suffix('.json').read_text())
    expected = metadata(config, paths, record.get('magnification'), limit, record.get('code_sha') if code_sha is None else code_sha)
    expected['feature_sha256'] = file_hash(target)
    if (record.get('magnification') not in MAGNIFICATIONS or not re.fullmatch('[0-9a-f]{40}', record.get('code_sha', ''))
            or any(record.get(key) != value for key, value in expected.items())):
        raise ValueError('Approved feature provenance differs')
    receipt = record.get('loader_receipt', {})
    if (receipt.get('strict_full_state') is not True or receipt.get('exact_parameter_copy') is not True
            or any(type(receipt.get(k)) is not int or receipt[k] <= 0 for k in ('state_tensors', 'parameter_count'))
            or not isinstance(receipt.get('checkpoint_parameter_dtypes'), list)
            or not receipt['checkpoint_parameter_dtypes']
            or any(value not in ('torch.float32', 'torch.float16', 'torch.bfloat16') for value in receipt['checkpoint_parameter_dtypes'])):
        raise ValueError('Approved strict-load receipt differs')
    if (not _positive_number(record.get('elapsed_seconds'))
            or any(type(record.get(k)) is not int or record[k] <= 0 for k in ('peak_gpu_bytes', 'peak_reserved_gpu_bytes'))):
        raise ValueError('Approved runtime/memory receipt differs')
    if (limit and not _positive_number(record.get('steady_batch_seconds'))) or (not limit and record.get('steady_batch_seconds') is not None):
        raise ValueError('Approved steady timing receipt differs')
    parity = record.get('parity')
    if limit:
        names = {'repeat_max_abs', 'single_batch_max_abs', 'reconstructed_model_max_abs'}
        if (not isinstance(parity, dict) or set(parity) != names
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in parity.values())):
            raise ValueError('Approved smoke parity receipt differs')
    elif parity != {}:
        raise ValueError('Full approved extraction must not claim extra replay forwards')
    with np.load(target, allow_pickle=False) as saved:
        if (set(saved.files) != {'paths', 'features'} or saved['paths'].tolist() != paths or len(set(paths)) != len(paths)
                or saved['features'].shape != (len(paths), config['output_dimension'])
                or saved['features'].dtype != np.float32 or not np.isfinite(saved['features']).all()):
            raise ValueError('Approved feature path/shape/dtype/finite coverage differs')
    return record


def _positive_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value) and value > 0


def write_output(target, paths, features, record):
    target = Path(target); meta = target.with_suffix('.json')
    if target.exists() or meta.exists():
        raise ValueError('Refusing existing approved feature artifacts')
    if (not paths or features.shape != (len(paths), record['output_dimension']) or features.dtype != np.float32
            or not np.isfinite(features).all() or len(set(paths)) != len(paths)):
        raise ValueError('Invalid approved feature archive')
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
                raise ValueError('Actual approved feature archive reload differs')
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
    p.add_argument('--batch-size', type=int, default=HOST_BATCH_SIZE); p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run approved extraction in a CUDA scheduler allocation')
    if args.batch_size != HOST_BATCH_SIZE or args.workers != 0 or not re.fullmatch('[0-9a-f]{40}', args.code_sha):
        raise ValueError('Registered approved batch/workers/source definition differs')
    if (Path(__file__).resolve().parents[1]/'.source-sha').read_text().strip() != args.code_sha:
        raise ValueError('Immutable approved extraction source receipt differs')
    config = read_config(args.config); base.runtime_versions(config)
    _, selected = select_rows(args.manifest, config, args.magnification, args.limit)
    paths = [row['relative_path'] for row in selected]
    target = args.output_root/('%s-%d.npz'%(config['encoder_name'], args.magnification))
    if target.exists() or target.with_suffix('.json').exists():
        verify_output(target, paths, config, args.limit, args.code_sha); print('Verified existing approved shard'); return
    torch.manual_seed(812); torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    model = load_model(args.cache_root, torch.device('cuda'), config); check_model(model, config)
    receipt = model._approved_loader_receipt; started = time.monotonic(); output = []; geometries = []; smoke_pixels = None
    def encode(pixels):
        return classification_features(model, pixels.to('cuda'), config).cpu().numpy()
    with torch.inference_mode():
        for start in range(0, len(selected), HOST_BATCH_SIZE):
            values = [approved_pixels(load_checked_image(args.images_root, row), config) for row in selected[start:start+HOST_BATCH_SIZE]]
            pixels = torch.stack([value[0] for value in values]); geometries.extend(value[1] for value in values)
            output.append(encode_batches(encode, pixels, config))
            if args.limit: smoke_pixels = pixels
            processed = start+len(values)
            if not args.limit and (processed % 200 == 0 or processed == len(selected)):
                print(json.dumps({'magnification': args.magnification, 'processed': processed, 'total': len(selected), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    features = np.concatenate(output); parity = {}; steady_batch_seconds = None
    if args.limit:
        with torch.inference_mode():
            repeated_start = time.monotonic(); repeated = encode_batches(encode, smoke_pixels, config); torch.cuda.synchronize()
            steady_batch_seconds = time.monotonic()-repeated_start
            parity['repeat_max_abs'] = assert_parity(features, repeated, config)
            singles = np.concatenate([encode(smoke_pixels[index:index+1]) for index in range(len(smoke_pixels))])
            parity['single_batch_max_abs'] = assert_parity(features, singles, config)
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_model(args.cache_root, torch.device('cuda'), config); check_model(model, config)
        if model._approved_loader_receipt != receipt:
            raise ValueError('Fresh approved loader receipt differs')
        with torch.inference_mode(): fresh = encode_batches(encode, smoke_pixels, config)
        parity['reconstructed_model_max_abs'] = assert_parity(features, fresh, config)
    torch.cuda.synchronize(); elapsed = time.monotonic()-started
    record = metadata(config, paths, args.magnification, args.limit, args.code_sha)
    record.update({'loader_receipt': receipt,
                   'geometry_counts': {shape: sum('%dx%d'%tuple(item['original']) == shape for item in geometries) for shape in config['expected_shapes']},
                   'geometry_examples': geometries if args.limit else None,
                   'parity': parity, 'elapsed_seconds': elapsed, 'steady_batch_seconds': steady_batch_seconds,
                   'image_forwards_per_second': record['image_forwards']/elapsed, 'gpu': torch.cuda.get_device_name(),
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(),
                   'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    write_output(target, paths, features, record); verify_output(target, paths, config, args.limit, args.code_sha)
    print(json.dumps({'status': 'accepted frozen approved shard', 'encoder_name': config['encoder_name'], 'images': len(paths), 'parity': parity}))


if __name__ == '__main__':
    main()

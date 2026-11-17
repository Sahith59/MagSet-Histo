"""Re-extract one full frozen Midnight view from original PNGs; never fit."""
import argparse
import hashlib
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

PRODUCER = 'a6e6a54e4fac090051837df17b0dea14efca3bb3'
RESULT_SHA = '3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62'
LOADER_SHA = 'cffc7891b617fa2bbac15f4972d925dfab0d76ad6a838dbbb4235cf5699ee80c'
MIDNIGHT_CONFIG_SHA = '71da2f7ee8bbfb27876a212f7a2c6dd765ba23f4ca86d7d80e4b2982e7b017ce'
COVERAGE_CONFIG_SHA = '6caada1cb238a41d483e3e2cd9eb3f2f362c44b9073b07a96456a99599e81e98'
MAGNIFICATIONS = (40, 100, 200, 400)
CODES = {'A': 0, 'F': 0, 'PT': 0, 'TA': 0, 'DC': 1, 'LC': 2, 'MC': 3, 'PC': 4}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_crops(data, row, case):
    """Independent PNG/identity/pixel path; no registered crop helper is called."""
    require(hashlib.sha256(data).hexdigest() == row['sha256'], 'Original PNG bytes differ')
    filename = Path(row['relative_path']).name
    match = re.fullmatch(r'SOB_([BM])_(A|F|PT|TA|DC|LC|MC|PC)-.+-(40|100|200|400)-\d+\.png', filename)
    require(match is not None, 'Original filename grammar differs')
    broad, code, mag = match.groups()
    label = CODES[code]
    require((broad == 'M') == (label > 0) and label == case['truth5'] and code == row['subtype_code'],
            'Filename/case/manifest label differs')
    require(int(mag) == int(row['magnification']) == MAGNIFICATIONS[row['view']], 'Original magnification differs')
    require(('/malignant/' if label else '/benign/') in ('/'+row['relative_path']), 'Directory/filename label differs')
    with Image.open(BytesIO(data)) as image:
        image.load()
        require(image.mode == 'RGB' and image.format == 'PNG', 'Expected original RGB PNG')
        require(image.size == (int(row['width']), int(row['height'])) and image.size in ((700, 460), (700, 456)),
                'Original image geometry differs')
        width = int(image.width*224/image.height)
        resized = image.resize((width, 224), resample=Image.Resampling.BILINEAR)
        lefts = (0, int(round((width-224)/2)), width-224)
        return np.stack([np.asarray(resized.crop((left, 0, left+224, 224)), dtype=np.uint8) for left in lefts])


def fuse_crop_features(crops):
    crops = np.asarray(crops)
    require(crops.ndim == 3 and crops.shape[1:] == (3, 3072) and crops.dtype == np.float32
            and np.isfinite(crops).all(), 'Expected all three finite FP32 crop descriptors')
    return ((crops[:, 0].astype(np.float64)+crops[:, 1].astype(np.float64)+crops[:, 2].astype(np.float64))/3).astype(np.float32)


def execute(args):
    require((args.producer_source/'.source-sha').read_text().strip() == PRODUCER, 'Frozen producer identity differs')
    checked = args.study/'checked/result.json'
    require(digest(checked) == RESULT_SHA, 'Accepted comparison bytes differ')
    baseline = read(checked)
    for name, expected in baseline['implementation_sha256'].items():
        require(digest(args.producer_source/name) == expected, 'Frozen implementation changed')
    require(digest(args.producer_source/'magset/midnight_features.py') == LOADER_SHA, 'Frozen encoder loader differs')
    config_path = args.producer_source/'configs/midnight-12k-v1.json'
    coverage_path = args.producer_source/'configs/midnight-coverage-v1.json'
    require(digest(config_path) == MIDNIGHT_CONFIG_SHA and digest(coverage_path) == COVERAGE_CONFIG_SHA,
            'Frozen raw-feature recipe differs')
    config = read(config_path)
    versions = {name: importlib.metadata.version(name) for name in config['dependencies']}
    require(versions == config['dependencies'], 'Pinned Midnight runtime differs')
    audit_path = args.study/'input-audit/private.json'
    require(digest(audit_path) == baseline['input_audit_sha256'], 'Input audit bytes differ')
    audit = read(audit_path)
    indices = np.asarray([row['index'] for row in audit['fields'] if int(row['magnification']) == args.magnification], dtype=np.int64)
    require(len(indices) > 0 and len(set(indices.tolist())) == len(indices), 'Invalid full-view field coverage')
    pixels = np.load(args.study/'pixels/pixels.npy', mmap_mode='r', allow_pickle=False)
    require(pixels.shape == (7909, 3, 224, 224, 3) and pixels.dtype == np.uint8, 'Shared pixel cache schema differs')
    sys.path.insert(0, str(args.producer_source.resolve()))
    import torch
    from magset import midnight_features as frozen
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One allocated CUDA device required')
    require(torch.cuda.get_device_name(0) == 'NVIDIA A40', 'Use registered A40 encoder hardware')
    torch.manual_seed(812)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    model = frozen.load_model(args.model_cache, torch.device('cuda'))
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()), 'Frozen encoder state differs')
    original_root = Path(audit['raw_root']).resolve()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            selected = indices[start:start+8]
            crop_arrays = []
            for index in selected:
                row = audit['fields'][int(index)]
                path = (original_root/row['relative_path']).resolve()
                require(path.is_relative_to(original_root), 'Original image path escaped root')
                crops = checked_crops(path.read_bytes(), row, audit['cases'][row['case_index']])
                require(np.array_equal(crops, pixels[index]), 'Independent raw crop pixels differ from shared cache')
                crop_arrays.append(crops)
            array = np.stack(crop_arrays).reshape(-1, 224, 224, 3)
            # Match ToTensor then Normalize(.5,.5), without importing the producer transform.
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32).div(255.)
            tensor = tensor.sub(.5).div(.5)
            encoded = []
            for offset in range(0, len(tensor), 8):
                tokens = model(pixel_values=tensor[offset:offset+8].to('cuda')).last_hidden_state
                require(tokens.dtype == torch.float32 and tuple(tokens.shape[1:]) == (257, 1536)
                        and bool(torch.isfinite(tokens).all()), 'Raw encoder token shape/dtype/value differs')
                # Independent official readout: final CLS concatenated with mean final patch tokens.
                descriptor = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
                encoded.append(descriptor.cpu().numpy())
            outputs.append(np.concatenate(encoded).reshape(len(selected), 3, 3072))
            if start % 128 == 0:
                print(json.dumps({'magnification': args.magnification, 'raw_fields_completed': start+len(selected)}), flush=True)
    torch.cuda.synchronize()
    crops = np.concatenate(outputs).astype(np.float32, copy=False)
    fused = fuse_crop_features(crops)
    path = args.output/'raw_features.npz'
    np.savez_compressed(path, field_indices=indices, features=fused, crop_features=crops)
    with np.load(path, allow_pickle=False) as reload:
        require(np.array_equal(reload['field_indices'], indices) and np.array_equal(reload['features'], fused)
                and np.array_equal(reload['crop_features'], crops), 'Raw feature archive reload differs')
    return {'status': 'accepted all-field frozen raw Midnight feature shard', 'magnification': args.magnification,
            'producer_code_sha': PRODUCER, 'producer_archive_sha256': baseline['source_archive_sha256'],
            'original_comparison_sha256': RESULT_SHA, 'input_audit_sha256': digest(audit_path),
            'frozen_model_loader_sha256': LOADER_SHA, 'raw_feature_config_sha256': MIDNIGHT_CONFIG_SHA,
            'coverage_config_sha256': COVERAGE_CONFIG_SHA, 'encoder_revision': config['encoder']['revision'],
            'encoder_weights_sha256': config['encoder']['weights_sha256'], 'images': len(indices),
            'original_png_hashes_checked': len(indices), 'filename_directory_labels_checked': len(indices),
            'pixel_cache_equal_fields': len(indices), 'raw_crop_forwards': 3*len(indices), 'crop_forward_batch': 8,
            'features': 'FP32 CLS1536 plus mean-patch1536, then FP64 mean3 cast once FP32',
            'precision': 'FP32; eager attention; TF32 disabled; no AMP', 'environment': versions,
            'gpu': torch.cuda.get_device_name(0), 'new_fits': 0, 'new_selections': 0,
            'artifact_sha256': {'raw_features.npz': digest(path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-source', 'study', 'model-cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing existing proof output')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = {'status': 'raw Midnight feature shard failed', 'magnification': args.magnification,
              'verifier_sha256': digest(__file__), 'new_fits': 0, 'new_selections': 0}
    try:
        result.update(execute(args), elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        import traceback
        result['error_type'] = type(error).__name__
        (args.output/'private-traceback.txt').write_text(traceback.format_exc())
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'].startswith('accepted ') else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""CPU audit of one approved encoder's complete four-shard feature archive."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import approved_features as producer
from magset.data import MAGNIFICATIONS, file_hash

STATUS = 'accepted all four frozen approved encoder shards'


def audit(config_path, manifest, cache, code_sha):
    if not re.fullmatch('[0-9a-f]{40}', code_sha):
        raise ValueError('A frozen feature producer source revision is required')
    config = producer.read_config(config_path)
    rows, _ = producer.select_rows(manifest, config, 40, 0)
    images = 0; shards = {}
    for scale in MAGNIFICATIONS:
        selected = [row for row in rows if int(row['magnification']) == scale]
        paths = [row['relative_path'] for row in selected]
        target = Path(cache)/('%s-%d.npz'%(config['encoder_name'], scale))
        record = producer.verify_output(target, paths, config, 0, code_sha)
        if record['magnification'] != scale:
            raise ValueError('Approved shard magnification differs')
        counts = {shape: sum('%sx%s'%(row['width'], row['height']) == shape for row in selected) for shape in config['expected_shapes']}
        if record.get('geometry_counts') != counts:
            raise ValueError('Approved shard raw geometry counts differ')
        with np.load(target, allow_pickle=False) as saved:
            features = saved['features']
            if (saved['paths'].tolist() != paths or features.shape != (len(paths), config['output_dimension'])
                    or features.dtype != np.float32 or not np.isfinite(features).all()):
                raise ValueError('Independent approved feature coverage check failed')
        shards[str(scale)] = {'npz_sha256': file_hash(target), 'json_sha256': file_hash(target.with_suffix('.json')), 'images': len(paths)}
        images += len(paths)
    expected = {Path(cache)/('%s-%d.npz'%(config['encoder_name'], scale)) for scale in MAGNIFICATIONS}
    if (set(Path(cache).glob('*.npz')) != expected
            or set(Path(cache).glob('*.json')) != {path.with_suffix('.json') for path in expected}
            or images != config['expected_images']):
        raise ValueError('Complete four-shard approved artifact matrix differs')
    return {'status': STATUS, 'encoder_name': config['encoder_name'], 'code_sha': code_sha,
            'config_sha256': file_hash(config_path), 'manifest_sha256': config['manifest_sha256'],
            'encoder': config['encoder'], 'dependencies': config['dependencies'],
            'expected_images': config['expected_images'], 'images': images, 'shards': shards,
            'source_file_sha256': file_hash(producer.__file__),
            'implementation_hashes': {'approved_features.py': file_hash(producer.__file__), 'audit_approved_features.py': file_hash(__file__)},
            'architecture': config['architecture'], 'artifacts': config['artifacts'], 'readout': config['readout'],
            'input_shape': config['input_shape'], 'token_shape': config['expected_token_shape'],
            'output_dimension': config['output_dimension'], 'image_forwards': images,
            'model_image_batch_size': producer.MODEL_IMAGE_BATCH_SIZE, 'host_batch_size': producer.HOST_BATCH_SIZE,
            'model_batch_policy': producer.BATCH_POLICY, 'batch_parity_scope': producer.BATCH_PARITY_SCOPE,
            'new_fits': 0, 'new_encoder_forwards': 0, 'image_decodes': 0}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run the complete approved-feature audit through the scheduler')
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'cache', 'code-sha', 'output'):
        p.add_argument('--'+name, type=str if name == 'code-sha' else Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing approved feature audit output')
    result = audit(args.config, args.manifest, args.cache, args.code_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True); temporary = args.output.with_suffix('.json.part')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink()
    print(json.dumps({'status': result['status'], 'images': result['images'], 'audit_sha256': file_hash(args.output)}))


if __name__ == '__main__':
    main()

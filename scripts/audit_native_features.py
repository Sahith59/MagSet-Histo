#!/usr/bin/env python3
"""Audit all full native-input shards without constructing an image encoder."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import native_features as producer
from magset.data import MAGNIFICATIONS, file_hash


def audit(config_path, manifest, cache, code_sha):
    config = producer.read_config(config_path)
    rows, _ = producer.select_rows(manifest, config, 40, 0)
    images = 0; shards = {}
    for scale in MAGNIFICATIONS:
        paths = [row['relative_path'] for row in rows if int(row['magnification']) == scale]
        target = Path(cache)/('%s-%d.npz'%(config['encoder_name'], scale))
        record = producer.verify_output(target, paths, config, 0, code_sha)
        if record['magnification'] != scale:
            raise ValueError('Native shard magnification differs')
        counts = {shape: sum(int(row['magnification']) == scale and '%sx%s'%(row['width'], row['height']) == shape for row in rows)
                  for shape in config['expected_shapes']}
        if record.get('geometry_counts') != counts:
            raise ValueError('Native shard raw geometry counts differ')
        with np.load(target, allow_pickle=False) as saved:
            features = saved['features']
            if (saved['paths'].tolist() != paths or features.shape != (len(paths), 3072)
                    or features.dtype != np.float32 or not np.isfinite(features).all()):
                raise ValueError('Independent native feature coverage check failed')
        shards[str(scale)] = {'npz_sha256': file_hash(target), 'json_sha256': file_hash(target.with_suffix('.json')), 'images': len(paths)}
        images += len(paths)
    expected = {Path(cache)/('%s-%d.npz'%(config['encoder_name'], scale)) for scale in MAGNIFICATIONS}
    if set(Path(cache).glob('*.npz')) != expected or images != config['expected_images']:
        raise ValueError('Complete four-shard native artifact matrix differs')
    return {'status': 'accepted all four frozen native-input shards', 'code_sha': code_sha,
            'config_sha256': file_hash(config_path), 'manifest_sha256': config['manifest_sha256'],
            'encoder': config['encoder'], 'expected_images': config['expected_images'], 'images': images,
            'shards': shards, 'source_file_sha256': file_hash(producer.__file__),
            'implementation_hashes': {'native_features.py': file_hash(producer.__file__), 'audit_native_features.py': file_hash(__file__)},
            'input_shape': config['input_shape'], 'token_shape': config['expected_token_shape'],
            'new_fits': 0, 'new_encoder_forwards': 0, 'image_decodes': 0}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run the complete native feature audit through the scheduler')
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'cache', 'code-sha', 'output'):
        p.add_argument('--'+name, type=str if name == 'code-sha' else Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing native audit output')
    result = audit(args.config, args.manifest, args.cache, args.code_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True); temporary = args.output.with_suffix('.json.part')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, args.output); temporary.unlink()
    print(json.dumps({'status': result['status'], 'images': result['images'], 'audit_sha256': file_hash(args.output)}))


if __name__ == '__main__':
    main()

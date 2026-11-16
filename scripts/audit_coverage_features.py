#!/usr/bin/env python3
"""Independently audit all frozen coverage components and exact fused vectors."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import coverage_features as producer
from magset.data import MAGNIFICATIONS, file_hash


def audit(config_path, manifest, cache, center_cache, code_sha):
    config = producer.read_config(config_path)
    rows, _ = producer.select_rows(manifest, config, 40, 0)
    shards = {}; images = 0
    for scale in MAGNIFICATIONS:
        paths, centers, _ = producer.load_base_centers(config, center_cache, rows, scale)
        target = Path(cache)/('%s-%d.npz'%(config['encoder_name'], scale))
        component = producer.component_path(target)
        metadata = producer.verify_output(target, paths, centers, config, 0, code_sha)
        if metadata.get('magnification') != scale or metadata.get('new_crop_forwards') != 2*len(paths):
            raise ValueError('Full coverage shard magnification/forward budget differs')
        counts = {shape: sum(int(row['magnification']) == scale and '%sx%s'%(row['width'], row['height']) == shape for row in rows)
                  for shape in config['expected_shapes']}
        if metadata.get('geometry_counts') != counts:
            raise ValueError('Coverage shard input shape counts differ')
        with np.load(target, allow_pickle=False) as saved, np.load(component, allow_pickle=False) as parts:
            if parts['paths'].tolist() != paths or saved['paths'].tolist() != paths:
                raise ValueError('Independent ordered original-field coverage differs')
            left, right, actual = parts['left_features'], parts['right_features'], saved['features']
            for values in (left, right, actual):
                if values.shape != (len(paths), 3072) or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError('Independent component shape/dtype/finite check failed')
            # This replay intentionally does not call the producer fusion helper.
            expected = ((left.astype(np.float64)+centers.astype(np.float64)+right.astype(np.float64))/3.).astype(np.float32)
            if not np.array_equal(actual, expected):
                raise ValueError('Independent exact fused-feature reconstruction failed')
        shards[str(scale)] = {'npz_sha256': file_hash(target), 'json_sha256': file_hash(target.with_suffix('.json')),
                              'components_sha256': file_hash(component), 'images': len(paths)}
        images += len(paths)
    expected_npz = {Path(cache)/('%s-%d%s.npz'%(config['encoder_name'], scale, suffix))
                    for scale in MAGNIFICATIONS for suffix in ('', '-components')}
    if set(Path(cache).glob('*.npz')) != expected_npz or images != config['expected_images']:
        raise ValueError('Complete coverage archive matrix differs')
    return {'status': 'accepted all four frozen three-crop coverage shards', 'code_sha': code_sha,
            'config_sha256': file_hash(config_path), 'manifest_sha256': config['manifest_sha256'],
            'expected_images': config['expected_images'], 'images': images, 'encoder': config['encoder'],
            'shards': shards, 'source_file_sha256': file_hash(producer.__file__),
            'implementation_hashes': {'coverage_features.py': file_hash(producer.__file__),
                                      'audit_coverage_features.py': file_hash(__file__)},
            'base_cache_provenance': config['base_cache_provenance'],
            'base_feature_producer_code_sha': config['base_feature_producer_code_sha'],
            'new_fits': 0, 'new_encoder_forwards': 0, 'image_decodes': 0}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run the complete cache audit through the scheduler')
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'cache', 'center-cache', 'code-sha', 'output'):
        p.add_argument('--'+name, type=str if name == 'code-sha' else Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing coverage audit output')
    result = audit(args.config, args.manifest, args.cache, args.center_cache, args.code_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.part')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, args.output); temporary.unlink()
    print(json.dumps({'status': result['status'], 'images': result['images'], 'audit_sha256': file_hash(args.output)}))


if __name__ == '__main__':
    main()

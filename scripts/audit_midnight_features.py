#!/usr/bin/env python3
"""Verify all four frozen Midnight feature shards before classifier fitting."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.bags import load_bags
from magset.data import MAGNIFICATIONS, file_hash, read_manifest
from magset.midnight_features import read_config, verify_output


def audit(manifest, cache, config_path, feature_source):
    config = read_config(config_path)
    manifest_hash = file_hash(manifest)
    if manifest_hash != config['manifest_sha256']:
        raise ValueError('Registered manifest bytes differ')
    rows = read_manifest(manifest)
    if len(rows) != config['expected_images']:
        raise ValueError('Full release image count differs')
    source_sha = (feature_source / '.source-sha').read_text().strip()
    module_hash = file_hash(feature_source / 'magset/midnight_features.py')
    config_hash = file_hash(config_path)
    provenance = {}
    counts = {}
    for scale in MAGNIFICATIONS:
        target = cache / ('midnight-12k-%d.npz' % scale)
        metadata = target.with_suffix('.json')
        paths = [row['relative_path'] for row in rows if int(row['magnification']) == scale]
        verify_output(target, metadata, paths, config['encoder'], manifest_hash, 0, config_hash)
        record = json.loads(metadata.read_text())
        if (record['code_sha'] != source_sha or record['source_file_sha256'] != module_hash
                or record['dependencies'] != config['dependencies']
                or record['transform'] != config['encoder']['transform']):
            raise ValueError('Feature producer, dependency or transform lineage differs')
        provenance[str(scale)] = {'npz_sha256': file_hash(target),
                                  'json_sha256': file_hash(metadata)}
        counts[str(scale)] = len(paths)
    cases, bags, _ = load_bags(rows, cache, 'midnight-12k', manifest_hash)
    if len(cases) != 82 or any(len(views) != 4 for views in bags):
        raise ValueError('Full provider-case coverage differs')
    return {'status': 'accepted complete frozen Midnight feature cache',
            'manifest_sha256': manifest_hash, 'feature_producer_code_sha': source_sha,
            'feature_module_sha256': module_hash,
            'feature_extraction_config_sha256': config_hash,
            'cache_provenance': provenance, 'images': len(rows), 'qualified_cases': len(cases),
            'images_by_magnification': counts, 'dimension': 3072,
            'encoder': config['encoder'], 'new_fits': 0, 'new_encoder_forwards': 0,
            'image_decodes': 0, 'script_sha256': file_hash(__file__)}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run the complete cache audit through the scheduler')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache', 'config', 'feature-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing feature audit output')
    result = audit(args.manifest, args.cache, args.config, args.feature_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': result['status'], 'images': result['images'],
                      'audit_sha256': file_hash(args.output)}))


if __name__ == '__main__':
    main()

"""Same registered KDE computation on independently audited three-crop fields."""
import argparse
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from . import kernel_five as kernel, midnight_five as base
from .bags import fit_normalization, load_bags
from .data import MAGNIFICATIONS, file_hash, read_manifest
from .five_collect import read_json


def validate_bindings(config):
    expected = {'name': 'breakhis-coverage-kernel-v1', 'schema_version': 1,
                'encoder_name': 'midnight-coverage-v1', 'dimension': 3072,
                'temperatures': kernel.TEMPERATURES, 'arms': list(kernel.ARMS),
                'production_runs': 10, 'candidate_pipelines': 80,
                'support_classifier_instances': 30, 'optimized_head_fits': 0,
                'new_encoder_forwards': 0, 'outer_predictions': 0,
                'coverage_new_crop_forwards': 15818, 'normalization': kernel.NORMALIZATION,
                'training_diagnostic': kernel.FITTING_POLICY,
                'routing_threshold': .5, 'routing_tie': 'malignant'}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered coverage KDE recipe differs')
    if not re.fullmatch('[0-9a-f]{40}', config.get('feature_producer_code_sha') or ''):
        raise ValueError('Actual coverage feature producer must be bound before fitting')
    for key in ('feature_audit_sha256', 'coverage_extraction_config_sha256', 'base_input_config_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config.get(key) or ''):
            raise ValueError('Accepted coverage input hashes are required before fitting')
    pins = config.get('cache_provenance')
    if not isinstance(pins, dict) or set(pins) != set(map(str, MAGNIFICATIONS)):
        raise ValueError('All four audited coverage caches must be bound')
    for receipt in pins.values():
        if (set(receipt) != {'npz_sha256', 'json_sha256', 'components_sha256', 'images'}
                or type(receipt['images']) is not int or receipt['images'] <= 0
                or any(not re.fullmatch('[0-9a-f]{64}', receipt[k]) for k in ('npz_sha256', 'json_sha256', 'components_sha256'))):
            raise ValueError('Invalid actual coverage shard binding')
    if sum(r['images'] for r in pins.values()) != 7909:
        raise ValueError('Coverage cache must contain all7909 original fields')
    return config


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    canonical = root/'configs/breakhis-coverage-kernel-v1.json'
    if file_hash(args.config) != file_hash(canonical):
        raise ValueError('Coverage KDE config differs from immutable registration')
    own = validate_bindings(read_json(args.config))
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']:
        raise ValueError('Original Midnight input registration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path; forwarded.cache = args.center_cache
    original, context, trusted, cases, _, _, y5, roles = base.load_inputs(forwarded)
    extraction_path = root/own['coverage_extraction_config']
    if file_hash(extraction_path) != own['coverage_extraction_config_sha256']:
        raise ValueError('Coverage extraction recipe differs')
    extraction = read_json(extraction_path)
    audit_path = Path(args.feature_audit)
    if file_hash(audit_path) != own['feature_audit_sha256']:
        raise ValueError('Accepted full coverage feature audit differs')
    audit = read_json(audit_path)
    for key, value in {'status': 'accepted all four frozen three-crop coverage shards',
                       'code_sha': own['feature_producer_code_sha'],
                       'config_sha256': own['coverage_extraction_config_sha256'],
                       'manifest_sha256': trusted['manifest_sha256'], 'images': 7909,
                       'encoder': extraction['encoder'], 'shards': own['cache_provenance']}.items():
        if audit.get(key) != value:
            raise ValueError('Coverage feature audit binding differs: '+key)
    for mag in MAGNIFICATIONS:
        path = Path(args.cache)/('midnight-coverage-v1-%d.npz'%mag)
        pin = own['cache_provenance'][str(mag)]; meta_path = path.with_suffix('.json')
        component = path.with_name(path.stem+'-components.npz')
        if any(file_hash(p) != pin[k] for p, k in ((path, 'npz_sha256'), (meta_path, 'json_sha256'), (component, 'components_sha256'))):
            raise ValueError('Accepted coverage cache bytes differ')
        metadata = read_json(meta_path)
        wanted = {'code_sha': own['feature_producer_code_sha'], 'config_sha256': own['coverage_extraction_config_sha256'],
                  'source_file_sha256': file_hash(root/'magset/coverage_features.py'),
                  'encoder': extraction['encoder'], 'smoke_limit': 0, 'images': pin['images'],
                  'components_sha256': pin['components_sha256'],
                  'base_center_npz_sha256': original['cache_provenance'][str(mag)]['npz_sha256'],
                  'base_center_json_sha256': original['cache_provenance'][str(mag)]['json_sha256']}
        if any(metadata.get(k) != v for k, v in wanted.items()):
            raise ValueError('Coverage cache feature lineage differs')
    new_cases, bags, _ = load_bags(read_manifest(args.manifest), args.cache, own['encoder_name'], trusted['manifest_sha256'])
    if new_cases != cases:
        raise ValueError('Coverage changed the original field/provider membership')
    z = np.asarray([[x.mean(0) for x in views] for views in bags], dtype=np.float32)
    if z.shape != (82, 4, 3072) or not np.isfinite(z).all():
        raise ValueError('Coverage case representation differs')
    context = dict(context, config_sha256=file_hash(args.config), base_config_sha256=own['base_input_config_sha256'],
                   coverage_config_sha256=file_hash(args.config), feature_audit_sha256=own['feature_audit_sha256'],
                   encoder=extraction['encoder'], cache_provenance=own['cache_provenance'],
                   center_cache_provenance=original['cache_provenance'],
                   feature_producer_code_sha=own['feature_producer_code_sha'],
                   feature_extraction_config_sha256=own['coverage_extraction_config_sha256'])
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('kernel_five.py', 'kernel_collect.py', 'kernel_model.py',
                  'coverage_features.py', 'coverage_five.py', 'coverage_collect.py')))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in names}
    config = {**original, **own, 'encoder': extraction['encoder']}
    return config, context, trusted, cases, bags, z, y5, roles


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing coverage audit output')
    checks = []
    for (seed, trial), (train, val, excluded) in roles.items():
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
                       'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
                       'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                       'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output.mkdir(parents=True)
    base.write_json(output/'audit.json', {'status': 'zero-fit inputs accepted', 'new_fits': 0,
                    'new_forwards': 0, 'outer_predictions': 0, 'checks': checks, **context})


def run(args):
    return kernel.run(args, inputs=load_inputs(args))


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('audit', 'run', 'collect'))
    for key in ('manifest', 'protocol', 'cache', 'center-cache', 'feature-audit', 'config', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None: raise ValueError('Coverage run requires audited role')
        run(args)
    else:
        from .coverage_collect import collect
        collect(args)

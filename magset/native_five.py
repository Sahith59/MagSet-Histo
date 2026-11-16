"""Unchanged reviewed KDE on audited frozen native-global Midnight fields."""
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
    expected = {'name': 'breakhis-native-kernel-v1', 'schema_version': 1,
                'encoder_name': 'midnight-native-v1', 'dimension': 3072,
                'temperatures': kernel.TEMPERATURES, 'arms': list(kernel.ARMS),
                'production_runs': 10, 'candidate_pipelines': 80, 'support_classifier_instances': 30,
                'optimized_head_fits': 0, 'new_encoder_forwards': 0, 'outer_predictions': 0,
                'native_full_image_forwards': 7909, 'normalization': kernel.NORMALIZATION,
                'training_diagnostic': kernel.FITTING_POLICY, 'routing_threshold': .5, 'routing_tie': 'malignant'}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered native-input KDE recipe differs')
    if not isinstance(config.get('feature_producer_code_sha'), str) or not re.fullmatch('[0-9a-f]{40}', config['feature_producer_code_sha']):
        raise ValueError('Actual native feature producer must be bound before scoring')
    for key in ('feature_audit_sha256', 'native_extraction_config_sha256', 'base_input_config_sha256'):
        if not isinstance(config.get(key), str) or not re.fullmatch('[0-9a-f]{64}', config[key]):
            raise ValueError('Accepted native input hashes must be bound before scoring')
    pins = config.get('cache_provenance')
    if not isinstance(pins, dict) or set(pins) != set(map(str, MAGNIFICATIONS)):
        raise ValueError('All four audited native caches must be bound')
    for receipt in pins.values():
        if (not isinstance(receipt, dict) or set(receipt) != {'npz_sha256', 'json_sha256', 'images'}
                or type(receipt['images']) is not int or receipt['images'] <= 0
                or any(not isinstance(receipt[k], str) or not re.fullmatch('[0-9a-f]{64}', receipt[k]) for k in ('npz_sha256', 'json_sha256'))):
            raise ValueError('Invalid native shard byte/count binding')
    if sum(r['images'] for r in pins.values()) != 7909:
        raise ValueError('Native cache must contain all 7909 original fields')
    return config


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]; canonical = root/'configs/breakhis-native-kernel-v1.json'
    if file_hash(args.config) != file_hash(canonical):
        raise ValueError('Native KDE config differs from immutable registration')
    own = validate_bindings(read_json(args.config))
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']:
        raise ValueError('Original Midnight role/input registration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path; forwarded.cache = args.center_cache
    original, context, trusted, cases, _, _, y5, roles = base.load_inputs(forwarded)
    extraction_path = root/own['native_extraction_config']
    if file_hash(extraction_path) != own['native_extraction_config_sha256']:
        raise ValueError('Native extraction configuration differs')
    extraction = read_json(extraction_path)
    if (extraction.get('encoder_name') != own['encoder_name'] or extraction.get('input_shape') != [3, 448, 700]
            or extraction.get('expected_token_shape') != [1601, 1536] or extraction.get('frozen_base') is not True):
        raise ValueError('Native input/token/frozen encoder contract differs')
    audit_path = Path(args.feature_audit)
    if file_hash(audit_path) != own['feature_audit_sha256']:
        raise ValueError('Accepted complete native feature audit differs')
    audit = read_json(audit_path)
    wanted = {'status': 'accepted all four frozen native-input shards', 'code_sha': own['feature_producer_code_sha'],
              'config_sha256': own['native_extraction_config_sha256'], 'manifest_sha256': trusted['manifest_sha256'],
              'images': 7909, 'encoder': extraction['encoder'], 'shards': own['cache_provenance'],
              'source_file_sha256': file_hash(root/'magset/native_features.py')}
    if any(audit.get(k) != v for k, v in wanted.items()):
        raise ValueError('Native feature audit binding differs')
    for mag in MAGNIFICATIONS:
        path = Path(args.cache)/('%s-%d.npz'%(own['encoder_name'], mag)); pin = own['cache_provenance'][str(mag)]
        if file_hash(path) != pin['npz_sha256'] or file_hash(path.with_suffix('.json')) != pin['json_sha256']:
            raise ValueError('Accepted native cache bytes differ')
        metadata = read_json(path.with_suffix('.json'))
        expected = {'code_sha': own['feature_producer_code_sha'], 'config_sha256': own['native_extraction_config_sha256'],
                    'manifest_sha256': trusted['manifest_sha256'], 'source_file_sha256': file_hash(root/'magset/native_features.py'),
                    'encoder': extraction['encoder'], 'smoke_limit': 0, 'images': pin['images'],
                    'input_shape': [3, 448, 700], 'token_shape': [1601, 1536], 'frozen_base': True,
                    'model_image_size': 518, 'position_grid': [37, 37], 'patch_grid': [32, 50],
                    'dependencies': extraction['dependencies'], 'native_forwards': pin['images'],
                    'attention_implementation': 'eager', 'precision': 'FP32; TF32 disabled',
                    'transform': extraction['encoder']['transform'],
                    'base_extraction_config_sha256': extraction['base_extraction_config_sha256'],
                    'base_source_file_sha256': file_hash(root/'magset/midnight_features.py')}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Native cache source/geometry/runtime lineage differs')
    if set(Path(args.cache).glob('*.npz')) != {Path(args.cache)/('%s-%d.npz'%(own['encoder_name'], m)) for m in MAGNIFICATIONS}:
        raise ValueError('Native cache must contain exactly four full global archives, without components')
    new_cases, bags, _ = load_bags(read_manifest(args.manifest), args.cache, own['encoder_name'], trusted['manifest_sha256'])
    if new_cases != cases:
        raise ValueError('Native cache changed original field/provider membership')
    z = np.asarray([[x.mean(0) for x in views] for views in bags], dtype=np.float32)
    if z.shape != (82, 4, 3072) or not np.isfinite(z).all():
        raise ValueError('Native case representation shape/finiteness differs')
    context = dict(context, config_sha256=file_hash(args.config), base_config_sha256=own['base_input_config_sha256'],
                   native_config_sha256=file_hash(args.config), feature_audit_sha256=own['feature_audit_sha256'],
                   encoder=extraction['encoder'], cache_provenance=own['cache_provenance'],
                   center_cache_provenance=original['cache_provenance'], feature_producer_code_sha=own['feature_producer_code_sha'],
                   feature_extraction_config_sha256=own['native_extraction_config_sha256'])
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('kernel_five.py', 'kernel_collect.py', 'kernel_model.py',
                                                 'native_features.py', 'native_five.py', 'native_collect.py')))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in names}
    return {**original, **own, 'encoder': extraction['encoder']}, context, trusted, cases, bags, z, y5, roles


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args); output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing native input audit')
    checks = []
    for (seed, trial), (train, val, excluded) in roles.items():
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
            'fitting_support': np.bincount(y5[train], minlength=5).tolist(), 'validation_support': np.bincount(y5[val], minlength=5).tolist(),
            'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output.mkdir(parents=True); base.write_json(output/'audit.json', {'status': 'zero-fit inputs accepted', 'new_fits': 0,
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
        if args.audit is None or args.seed is None or args.trial is None: raise ValueError('Native run requires audited role')
        run(args)
    else:
        from .native_collect import collect
        collect(args)

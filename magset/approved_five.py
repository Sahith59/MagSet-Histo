"""Unchanged case-centered KDE on hash-bound frozen approved reader field descriptions."""
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
    names = {'uni2-h-v1': 1536, 'virchow2-v1': 2560, 'h-optimus-0-v1': 1536}
    name = config.get('encoder_name') if isinstance(config, dict) else None
    if name not in names:
        raise ValueError('Unregistered approved encoder classifier')
    expected = {'name': 'breakhis-' + name + '-kernel', 'schema_version': 1,
                'encoder_name': name, 'dimension': names[name],
                'temperatures': kernel.TEMPERATURES, 'arms': list(kernel.ARMS),
                'production_runs': 10, 'candidate_pipelines': 80, 'support_classifier_instances': 30,
                'optimized_head_fits': 0, 'new_encoder_forwards': 0, 'outer_predictions': 0,
                'image_forwards': 7909,
                'normalization': kernel.NORMALIZATION, 'training_diagnostic': kernel.FITTING_POLICY,
                'routing_threshold': .5, 'routing_tie': 'malignant'}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered approved reader KDE recipe differs')
    if not re.fullmatch('[0-9a-f]{40}', config.get('feature_producer_code_sha') or ''):
        raise ValueError('Actual feature producer must be bound before scoring')
    for key in ('feature_audit_sha256', 'extraction_config_sha256', 'base_input_config_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config.get(key) or ''):
            raise ValueError('Accepted full feature input hashes must be bound')
    pins = config.get('cache_provenance')
    if not isinstance(pins, dict) or set(pins) != set(map(str, MAGNIFICATIONS)):
        raise ValueError('Exactly four accepted feature shards are required')
    for pin in pins.values():
        if (not isinstance(pin, dict) or set(pin) != {'npz_sha256', 'json_sha256', 'images'}
                or type(pin['images']) is not int or pin['images'] <= 0
                or any(not re.fullmatch('[0-9a-f]{64}', pin.get(k) or '') for k in ('npz_sha256', 'json_sha256'))):
            raise ValueError('Invalid feature shard byte/count binding')
    if sum(p['images'] for p in pins.values()) != 7909:
        raise ValueError('Complete7909-field cache required')
    return config


def load_inputs(args):
    from . import approved_features as producer
    root = Path(__file__).resolve().parents[1]
    own = validate_bindings(read_json(args.config))
    canonical = root/'configs'/('breakhis-' + own['encoder_name'] + '-kernel.json')
    if file_hash(args.config) != file_hash(canonical):
        raise ValueError('Classifier config differs from frozen registration')
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']:
        raise ValueError('Original accepted role/input config differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path; forwarded.cache = args.center_cache
    original, context, trusted, cases, _, _, y5, roles = base.load_inputs(forwarded)
    path = root/own['extraction_config']
    if file_hash(path) != own['extraction_config_sha256']:
        raise ValueError('Registered extractor config differs')
    extraction = producer.read_config(path)
    if (extraction['encoder_name'] != own['encoder_name']
            or extraction['output_dimension'] != own['dimension']):
        raise ValueError('Classifier and extractor identity or dimension differs')
    audit = read_json(args.feature_audit)
    if file_hash(args.feature_audit) != own['feature_audit_sha256']:
        raise ValueError('Accepted full feature audit bytes differ')
    wanted = {'status': 'accepted all four frozen approved encoder shards',
              'encoder_name': own['encoder_name'],
              'code_sha': own['feature_producer_code_sha'], 'config_sha256': own['extraction_config_sha256'],
              'manifest_sha256': trusted['manifest_sha256'], 'images': 7909,
              'encoder': extraction['encoder'], 'shards': own['cache_provenance'],
              'source_file_sha256': file_hash(root/'magset/approved_features.py')}
    if any(audit.get(k) != v for k, v in wanted.items()):
        raise ValueError('Full feature audit lineage differs')
    rows = read_manifest(args.manifest)
    expected = {Path(args.cache)/('%s-%d.npz'%(own['encoder_name'], m)) for m in MAGNIFICATIONS}
    if set(Path(args.cache).glob('*.npz')) != expected:
        raise ValueError('Unexpected or incomplete feature archive matrix')
    for mag in MAGNIFICATIONS:
        path = Path(args.cache)/('%s-%d.npz'%(own['encoder_name'], mag)); pin = own['cache_provenance'][str(mag)]
        if file_hash(path) != pin['npz_sha256'] or file_hash(path.with_suffix('.json')) != pin['json_sha256']:
            raise ValueError('Accepted feature shard bytes differ')
        paths = [r['relative_path'] for r in rows if int(r['magnification']) == mag]
        metadata = producer.verify_output(path, paths, extraction, 0, own['feature_producer_code_sha'])
        if metadata.get('images') != pin['images'] or metadata.get('magnification') != mag:
            raise ValueError('Full shard membership/count differs')
    new_cases, bags, _ = load_bags(rows, args.cache, own['encoder_name'], trusted['manifest_sha256'])
    if new_cases != cases:
        raise ValueError('New reader changed original case/image membership')
    z = np.asarray([[x.mean(0) for x in views] for views in bags], dtype=np.float32)
    if z.shape != (82, 4, own['dimension']) or not np.isfinite(z).all():
        raise ValueError('New case descriptions have invalid shape or values')
    context = dict(context, config_sha256=file_hash(args.config), base_config_sha256=own['base_input_config_sha256'],
                   approved_config_sha256=file_hash(args.config), feature_audit_sha256=own['feature_audit_sha256'],
                   encoder=extraction['encoder'], cache_provenance=own['cache_provenance'],
                   center_cache_provenance=original['cache_provenance'], feature_producer_code_sha=own['feature_producer_code_sha'],
                   feature_extraction_config_sha256=own['extraction_config_sha256'])
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('kernel_five.py','kernel_collect.py','kernel_model.py',
                                                 'native_collect.py','genbio_collect.py','approved_features.py','approved_five.py','approved_collect.py')))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in names}
    return {**original, **own, 'encoder': extraction['encoder']}, context, trusted, cases, bags, z, y5, roles


def audit_input(args):
    config, context, _, _, bags, _, y5, roles = load_inputs(args); output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing approved reader classifier audit')
    checks = []
    for (seed, trial), (train, val, excluded) in roles.items():
        mean, std = fit_normalization(bags, train)
        checks.append({'seed':seed,'trial':trial,'role_counts':[len(train),len(val),len(excluded)],
                       'fitting_support':np.bincount(y5[train],minlength=5).tolist(),
                       'validation_support':np.bincount(y5[val],minlength=5).tolist(),
                       'normalization_sha256':hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output.mkdir(parents=True); base.write_json(output/'audit.json',dict(context,status='zero-fit inputs accepted',
        new_fits=0,new_forwards=0,outer_predictions=0,checks=checks))


def run(args):
    return kernel.run(args, inputs=load_inputs(args))


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','run','collect'))
    for key in ('manifest','protocol','cache','center-cache','feature-audit','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--audit',type=Path);p.add_argument('--runs',type=Path)
    p.add_argument('--seed',type=int,choices=(812,813));p.add_argument('--trial',type=int,choices=range(1,6))
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.action=='audit':audit_input(args)
    elif args.action=='run':
        if args.audit is None or args.seed is None or args.trial is None:raise ValueError('Run requires audited role')
        run(args)
    else:
        from .approved_collect import collect
        collect(args)

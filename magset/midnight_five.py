"""Two predeclared frozen Midnight readouts on the unchanged five-label roles."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from . import field_five as field
from . import five_class as case
from .bags import fit_normalization, load_bags
from .baseline import stratified_inner_split, vectorize, SUBSETS
from .capstone_binary import trusted_binary_labels
from .collect import load_trusted_context
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest
from .five_collect import read_json, verify_audit
from .five_metrics import FINAL5, MAP8_TO5, five_labels, summarize_records

READOUTS = ('case', 'field')
ARMS = case.ARMS
FEATURE_PRODUCER_SHA = 'a9a2fda4461aca74e6596a747ab191aefa7a96da'
DEPENDENCIES = tuple(dict.fromkeys(case.DEPENDENCIES + field.DEPENDENCIES +
                                 ('midnight_five.py', 'midnight_collect.py', 'midnight_features.py')))
save_model = case.save_model
load_model = case.load_model
write_json = case.write_json


def validate_config(config):
    """Only cache hashes are generated after extraction; the recipe is fixed."""
    if not isinstance(config, dict):
        raise ValueError('Expected a registered classifier configuration object')
    expected = {'name': 'breakhis-midnight-five-v1', 'schema_version': 1,
                'encoder_name': 'midnight-12k', 'encoder_dimension': 3072,
                'expected_images': 7909, 'seeds': [812, 813], 'trials': [1, 2, 3, 4, 5],
                'arms': list(ARMS), 'readouts': list(READOUTS), 'C_grid': [.01, .1, 1., 10.],
                'training_seed': 812, 'solver': 'lbfgs', 'penalty': 'l2', 'tol': 1e-4,
                'max_iter': 2000, 'routing_threshold': .5, 'routing_tie': 'malignant',
                'expected_role_counts': [46, 8, 28], 'fitting_five_support': [11, 25, 2, 5, 3],
                'validation_five_support': [4, 1, 1, 1, 1], 'excluded_five_support': [9, 12, 2, 3, 2],
                'production_runs': 10, 'production_head_fits': 120,
                'total_production_runs': 20, 'total_head_fits': 240, 'technical_head_fits': 0,
                'new_encoder_forwards': 0, 'outer_predictions': 0,
                'head_weight_mass': {'direct': 46, 'binary': 46, 'malignant': 35},
                'final_labels': list(FINAL5), 'map8_to5': list(MAP8_TO5),
                'feature_producer_code_sha': FEATURE_PRODUCER_SHA,
                'dependencies': {'numpy': '1.26.4', 'scipy': '1.15.2', 'scikit-learn': '1.6.1'}}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Registered Midnight classifier recipe differs')
    for key in ('manifest_sha256', 'protocol_sha256', 'feature_extraction_config_sha256'):
        if not isinstance(config.get(key), str) or not re.fullmatch('[0-9a-f]{64}', config[key]):
            raise ValueError('Required classifier provenance hash missing: '+key)
    provenance = config.get('cache_provenance', {})
    if not isinstance(provenance, dict) or set(provenance) != set(map(str, MAGNIFICATIONS)):
        raise ValueError('All four audited Midnight cache bindings are required')
    for receipt in provenance.values():
        if (not isinstance(receipt, dict) or set(receipt) != {'npz_sha256', 'json_sha256'}
                or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                       for value in receipt.values())):
            raise ValueError('Final audited feature cache hashes are required before fitting')
    return config


def source_context(code_sha):
    context = case.source_context(code_sha)
    root = Path(__file__).resolve().parents[1]
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in DEPENDENCIES}
    return context


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    canonical = root/'configs/breakhis-midnight-five-v1.json'
    config_hash = file_hash(args.config)
    if config_hash != file_hash(canonical):
        raise ValueError('Classifier config differs from the immutable source registration')
    config = validate_config(read_json(args.config))
    extraction_path = root/'configs/midnight-12k-v1.json'
    extraction = read_json(extraction_path)
    if (file_hash(extraction_path) != config['feature_extraction_config_sha256']
            or config.get('encoder') != extraction['encoder']
            or config['encoder']['dimension'] != 3072):
        raise ValueError('Pinned Midnight encoder/extraction configuration differs')
    context = source_context(args.code_sha)
    versions = {name: importlib.metadata.version(name) for name in config['dependencies']}
    if versions != config['dependencies']:
        raise ValueError('Pinned classifier numerical environment differs')
    trusted = load_trusted_context(args.manifest, args.protocol)
    if any(trusted[key] != config[key] for key in ('manifest_sha256', 'protocol_sha256')):
        raise ValueError('Registered manifest/protocol differs')
    rows = read_manifest(args.manifest); broad = trusted_binary_labels(rows)
    for scale in MAGNIFICATIONS:
        path = Path(args.cache)/('midnight-12k-%d.npz' % scale)
        receipt = config['cache_provenance'][str(scale)]
        if (file_hash(path) != receipt['npz_sha256']
                or file_hash(path.with_suffix('.json')) != receipt['json_sha256']):
            raise ValueError('Audited Midnight cache bytes differ')
        metadata = read_json(path.with_suffix('.json'))
        if (metadata.get('code_sha') != config['feature_producer_code_sha']
                or metadata.get('config_sha256') != config['feature_extraction_config_sha256']
                or metadata.get('source_file_sha256') != file_hash(root/'magset/midnight_features.py')
                or metadata.get('encoder') != config['encoder']):
            raise ValueError('Audited Midnight cache producer/encoder lineage differs')
    cases, bags, metadata = load_bags(rows, args.cache, config['encoder_name'], trusted['manifest_sha256'])
    if cases != trusted['cases'] or len(cases) != 82 or len(rows) != config['expected_images']:
        raise ValueError('Trusted case ordering/full release coverage differs')
    y5 = five_labels(np.asarray([c['label'] for c in cases]))
    if any(broad[c['case_id']] != int(label > 0) for c, label in zip(cases, y5)):
        raise ValueError('Original broad labels and final-five map disagree')
    roles = {}
    for seed in config['seeds']:
        for trial in config['trials']:
            indices = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
            if ([len(ix) for ix in indices] != config['expected_role_counts']
                    or any(set(indices[i]) & set(indices[j]) for i in range(3) for j in range(i))):
                raise ValueError('Original-eight-derived roles/counts/disjointness differ')
            supports = [np.bincount(y5[ix], minlength=5).tolist() for ix in indices]
            if supports != [config[key] for key in ('fitting_five_support', 'validation_five_support', 'excluded_five_support')]:
                raise ValueError('Registered five-class role supports differ')
            roles[seed, trial] = indices
    z = np.asarray([[fields.mean(axis=0) for fields in views] for views in bags], dtype=np.float32)
    if z.shape != (82, 4, 3072) or not np.isfinite(z).all():
        raise ValueError('Expected finite 3072-dimensional four-view representation')
    context.update(config_sha256=config_hash, manifest_sha256=trusted['manifest_sha256'],
                   protocol_sha256=trusted['protocol_sha256'], unit=trusted['unit'], dependencies=versions,
                   python=platform.python_version(), encoder=config['encoder'], cache_provenance=config['cache_provenance'],
                   feature_producer_code_sha=config['feature_producer_code_sha'],
                   feature_extraction_config_sha256=config['feature_extraction_config_sha256'])
    return config, context, trusted, cases, bags, z, y5, roles


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing output')
    checks = []
    for (seed, trial), (train, val, excluded) in roles.items():
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
                       'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
                       'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                       'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output.mkdir(parents=True)
    write_json(output/'audit.json', {'status': 'zero-fit inputs accepted', 'new_fits': 0,
               'new_forwards': 0, 'outer_predictions': 0, 'checks': checks, **context})


def run(args):
    started = time.monotonic(); output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing output')
    if args.readout not in READOUTS:
        raise ValueError('A predeclared case/field readout is required')
    config, common, trusted, cases, bags, z, y5, roles = load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), common, config, bags, y5, roles)
    context = dict(common, readout=args.readout)
    train, val, excluded = roles[args.seed, args.trial]
    mean, std = fit_normalization(bags, train)
    if args.readout == 'case':
        normalized = (z[train]-mean)/std
        x = np.concatenate([vectorize(normalized, subset, 'mean') for subset in SUBSETS])
        target = np.tile(y5[train], 15)
    output.mkdir(parents=True); reports = {}
    for arm in ARMS:
        candidates = []
        for index, C in enumerate(config['C_grid']):
            folder = output/arm/('candidate%d' % index); folder.mkdir(parents=True)
            begin = time.monotonic(); extra = {}
            if args.readout == 'case':
                models = case.fit_arm(x, target, arm, C, config)
            else:
                models = field.fit_arm(bags, train, y5, mean, std, arm, C, config)
                receipts, objectives = field.fitting_diagnostics(models, bags, train, y5, mean, std, C)
                if any(not np.isclose(mass, config['head_weight_mass'][name], rtol=0, atol=1e-10)
                       for name, mass in receipts['weight_mass_per_head'].items()):
                    raise ValueError('Registered field head training mass differs')
                extra = {'field_receipts': receipts, 'head_objectives': objectives}
            save_model(folder/'model.npz', mean, std, models)
            arrays = load_model(folder/'model.npz', arm, dimension=config['encoder_dimension'])
            values = {}; hashes = {}; replay = 0.
            for role, indices in (('train', train), ('validation', val)):
                scorer, inputs = (case.score_records, z) if args.readout == 'case' else (field.score_records, bags)
                records, error = scorer(arrays, inputs, indices, cases, role, arm, models)
                values[role] = summarize_records(records, arm); replay = max(replay, error)
                path = folder/(role+'-predictions.jsonl')
                path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
                hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = {'C': C, 'index': index, 'head_fits': len(models),
                         'n_iter': {key: model.n_iter_.tolist() for key, model in models.items()},
                         'sklearn_replay_max_abs': replay, 'elapsed_seconds': time.monotonic()-begin,
                         'model_sha256': file_hash(folder/'model.npz'), 'fitting': values['train'],
                         'validation': values['validation'], **extra, **hashes}
            write_json(folder/'metrics.json', candidate); candidates.append(candidate)
            print(json.dumps({'readout': args.readout, 'arm': arm, 'seed': args.seed, 'trial': args.trial,
                              'C': C, 'validation_F1': candidate['validation']['mean_subset_macro_f1']}), flush=True)
        selected = max(range(len(candidates)), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
        reports[arm] = {'selected_index': selected, 'selected_C': config['C_grid'][selected],
                        'candidates': candidates, 'fitting': candidates[selected]['fitting'],
                        'validation': candidates[selected]['validation']}
    status = 'complete five-class validation run' if args.readout == 'case' else 'complete field-supervised five-class validation run'
    result = {'status': status, 'seed': args.seed, 'trial': args.trial, 'class_names': list(FINAL5),
              'original_class_names': list(CLASS_NAMES),
              'case_ids': {name: [cases[i]['case_id'] for i in indices] for name, indices in
                           (('train', train), ('validation', val), ('excluded', excluded))},
              'role_counts': [46, 8, 28], 'head_fits': 12, 'outer_predictions': 0, 'arms': reports,
              'audit_sha256': file_hash(audit_path), 'elapsed_seconds': time.monotonic()-started,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **context}
    write_json(output/'run.json', result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path)
    p.add_argument('--runs', type=Path); p.add_argument('--seed', type=int, choices=(812, 813))
    p.add_argument('--trial', type=int, choices=range(1, 6)); p.add_argument('--readout', choices=READOUTS)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit':
        audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None or args.readout is None:
            raise ValueError('Run needs accepted audit, readout, seed and trial')
        run(args)
    else:
        from .midnight_collect import collect
        collect(args)

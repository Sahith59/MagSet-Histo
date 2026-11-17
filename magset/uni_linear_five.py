"""Registered UNI2-h field logistic baseline using the reviewed field learner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import time

import numpy as np

from . import approved_five as approved, field_five as field
from .bags import fit_normalization
from .data import CLASS_NAMES, file_hash
from .five_collect import read_json, require_equal, verify_audit
from .five_metrics import FINAL5, summarize_records

ARMS = ('direct', 'dual')
CONFIG_NAME = 'breakhis-uni-linear-v1.json'
FITTING_POLICY = 'in-sample resubstitution; optimizer and normalizer use fitting cases only'
write_json = field.write_json


def validate_config(config):
    expected = {'name': 'breakhis-uni-linear-v1', 'schema_version': 1,
        'base_input_config': 'configs/breakhis-uni2-h-v1-kernel.json',
        'encoder_name': 'uni2-h-v1', 'encoder_dimension': 1536, 'arms': list(ARMS),
        'C_grid': [.01, .1, 1., 10.], 'solver': 'lbfgs', 'penalty': 'l2', 'tol': 1e-4,
        'max_iter': 2000, 'training_seed': 812, 'production_runs': 10,
        'production_head_fits': 120, 'technical_head_fits': 12, 'candidate_pipelines': 80,
        'head_weight_mass': {'direct': 46, 'binary': 46, 'malignant': 35}}
    if not isinstance(config, dict): raise ValueError('Expected registered UNI linear configuration')
    for key, value in expected.items(): require_equal(config.get(key), value, 'registered UNI field learner/'+key, 0)
    for key in ('base_input_config_sha256', 'reference_comparison_sha256', 'secondary_reference_comparison_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config.get(key) or ''):
            raise ValueError('Required UNI linear provenance hash missing')
    return config


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    own = validate_config(read_json(args.config))
    if file_hash(args.config) != file_hash(root/'configs'/CONFIG_NAME):
        raise ValueError('UNI linear configuration differs from immutable registration')
    base_path = root/own['base_input_config']
    if file_hash(base_path) != own['base_input_config_sha256']:
        raise ValueError('Accepted UNI2-h classifier configuration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = base_path
    original, context, trusted, cases, bags, z, y5, roles = approved.load_inputs(forwarded)
    names = tuple(context['implementation_hashes'])+('uni_linear_five.py', 'uni_linear_collect.py')
    context = dict(context, parent_approved_config_sha256=context['config_sha256'],
        config_sha256=file_hash(args.config), uni_linear_config_sha256=file_hash(args.config))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in dict.fromkeys(names)}
    return {**original, **own}, context, trusted, cases, bags, z, y5, roles


def smoke_query_indices(y5, train):
    labels = np.asarray(y5); indices = np.asarray(train)
    result = []
    for label in range(5):
        eligible = indices[labels[indices] == label]
        if not len(eligible): raise ValueError('Technical smoke requires every final class in fitting support')
        result.append(int(eligible[0]))
    return np.asarray(result, dtype=int)


def audit_input(args):
    config, context, _, _, bags, _, y5, roles = load_inputs(args)
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing UNI linear audit')
    checks = []
    for (seed, trial), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
            'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
            'validation_support': np.bincount(y5[val], minlength=5).tolist(),
            'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output.mkdir(parents=True)
    write_json(output/'audit.json', dict(context, status='zero-fit inputs accepted', new_fits=0,
        new_forwards=0, outer_predictions=0, checks=checks))


def verify_smoke_acceptance(args, config, context, audit_sha):
    path = getattr(args, 'smoke_acceptance', None)
    if path is None: raise ValueError('Production requires the independently verified fitting-only smoke receipt')
    value = read_json(path)
    for key, expected in context.items(): require_equal(value.get(key), expected, 'smoke acceptance context/'+key, 0)
    expected = {'status': 'accepted UNI2-h field logistic fitting-only smoke', 'head_fits': 12,
        'verified_prediction_archives': 8, 'verified_case_subset_records': 600,
        'audit_sha256': audit_sha, 'validation_records': 0, 'collector_new_fits': 0}
    for key, wanted in expected.items(): require_equal(value.get(key), wanted, 'smoke acceptance/'+key, 0)
    if getattr(args, 'smokes', None) is None or file_hash(Path(args.smokes)/'run.json') != value.get('smoke_run_sha256'):
        raise ValueError('Accepted smoke output bytes differ')
    return file_hash(path)


def run(args, smoke=False):
    started = time.monotonic(); output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing UNI linear output')
    config, context, _, cases, bags, z, y5, roles = load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    if smoke and (args.seed, args.trial) != (812, 1):
        raise ValueError('Registered fitting-only smoke is812/trial1')
    acceptance = None if smoke else verify_smoke_acceptance(args, config, context, file_hash(audit_path))
    train, val, excluded = roles[args.seed, args.trial]
    queries = smoke_query_indices(y5, train) if smoke else train
    mean, std = fit_normalization(bags, train)
    output.mkdir(parents=True); reports = {}
    for arm in ARMS:
        candidates = []
        for index, C in enumerate(config['C_grid']):
            folder = output/arm/('candidate%d'%index); folder.mkdir(parents=True)
            begin = time.monotonic()
            models = field.fit_arm(bags, train, y5, mean, std, arm, C, config)
            receipts, objectives = field.fitting_diagnostics(models, bags, train, y5, mean, std, C)
            for name in models:
                require_equal(receipts['weight_mass_per_head'][name], config['head_weight_mass'][name], 'registered fitting weight mass', 1e-10)
            path = folder/'model.npz'; field.save_model(path, mean, std, models)
            arrays = field.load_model(path, arm, dimension=config['encoder_dimension'])
            values, hashes, replay = {}, {}, 0.
            items = [('train', queries)] if smoke else [('train', train), ('validation', val)]
            for role, indices in items:
                records, error = field.score_records(arrays, bags, indices, cases, role, arm, models)
                values[role] = summarize_records(records, arm); replay = max(replay, error)
                path = folder/(role+'-predictions.jsonl')
                path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
                hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = {'C': C, 'index': index, 'head_fits': len(models),
                'n_iter': {name: model.n_iter_.tolist() for name, model in models.items()},
                'sklearn_replay_max_abs': replay, 'elapsed_seconds': time.monotonic()-begin,
                'model_sha256': file_hash(folder/'model.npz'), 'fitting': values['train'],
                'field_receipts': receipts, 'head_objectives': objectives, **hashes}
            if not smoke: candidate['validation'] = values['validation']
            write_json(folder/'metrics.json', candidate); candidates.append(candidate)
        if smoke:
            reports[arm] = {'candidates': candidates}
        else:
            selected = max(range(4), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
            reports[arm] = {'selected_index': selected, 'selected_C': config['C_grid'][selected], 'candidates': candidates,
                'fitting': candidates[selected]['fitting'], 'validation': candidates[selected]['validation']}
    report = dict(context, status='complete UNI2-h field fitting-only technical smoke' if smoke else 'complete field-supervised five-class validation run',
        seed=args.seed, trial=args.trial, class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
        case_ids={name: [cases[i]['case_id'] for i in indices] for name, indices in [('train', train), ('validation', val), ('excluded', excluded)]},
        role_counts=[len(train), len(val), len(excluded)], head_fits=12, outer_predictions=0,
        fitting_evaluation=FITTING_POLICY, arms=reports, audit_sha256=file_hash(audit_path),
        smoke_acceptance_sha256=acceptance, elapsed_seconds=time.monotonic()-started,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(output/'run.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'smoke', 'verify-smoke', 'run', 'collect'))
    for key in ('manifest', 'protocol', 'cache', 'center-cache', 'feature-audit', 'config', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--smokes', type=Path); p.add_argument('--smoke-acceptance', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action in ('verify-smoke', 'collect'):
        from .uni_linear_collect import collect, verify_smoke
        (verify_smoke if args.action == 'verify-smoke' else collect)(args)
    else:
        if args.audit is None or args.seed is None or args.trial is None:
            raise ValueError('Fitting requires audited inputs and original role')
        run(args, smoke=args.action == 'smoke')

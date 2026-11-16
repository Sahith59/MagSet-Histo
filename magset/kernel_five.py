"""Registered centered-cosine Midnight KDE; labels construct supports, no optimized coefficients."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import midnight_five as base
from .baseline import SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .field_five import pool_log_probabilities
from .five_collect import read_json, verify_audit
from .five_metrics import FINAL5, MAP8_TO5, summarize_records
from .kernel_model import NORMALIZATION

ARMS = ('direct', 'dual')
TEMPERATURES = [.01, .03, .1, .3]
FITTING_POLICY = 'leave-one-case-out support and centering'



def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    canonical = root/'configs/breakhis-midnight-kernel-v1.json'
    if file_hash(args.config) != file_hash(canonical):
        raise ValueError('Kernel config differs from immutable source registration')
    kernel = read_json(args.config)
    expected = {'name': 'breakhis-midnight-kernel-v1', 'schema_version': 1,
                'temperatures': TEMPERATURES, 'production_runs': 10,
                'candidate_pipelines': 80, 'support_classifier_instances': 30,
                'optimized_head_fits': 0, 'new_encoder_forwards': 0, 'outer_predictions': 0,
                'encoder_name': 'midnight-12k', 'dimension': 3072, 'arms': list(ARMS),
                'normalization': NORMALIZATION, 'training_diagnostic': FITTING_POLICY,
                'reference_readout': 'case', 'routing_threshold': .5, 'routing_tie': 'malignant'}
    if any(kernel.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered kernel recipe differs')
    for key, value in {'each_arrangement_macro_f1_delta': .02, 'accuracy_not_lower': True,
                       'positive_recall_all_five_classes': True}.items():
        if kernel.get('improvement_screen', {}).get(key) != value:
            raise ValueError('Registered kernel improvement screen differs')
    path = root/kernel['base_input_config']
    if file_hash(path) != kernel['base_input_config_sha256']:
        raise ValueError('Accepted Midnight base config differs')
    base_args = SimpleNamespace(**vars(args)); base_args.config = path
    config, context, trusted, cases, bags, z, y5, roles = base.load_inputs(base_args)
    context = dict(context, kernel_config_sha256=file_hash(args.config))
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('kernel_model.py', 'kernel_five.py', 'kernel_collect.py')))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in names}
    return {**config, **kernel}, context, trusted, cases, bags, z, y5, roles


def prepare_queries(bags, y5, train, validation):
    from .kernel_model import prepare_case
    return {role: [prepare_case(bags, y5, train, int(i), leave_query_case_out=(role == 'train')) for i in indices]
            for role, indices in [('train', train), ('validation', validation)]}


def score_records(prepared, cases, role, arm, temperature):
    from .kernel_model import field_log_probabilities
    if role not in ('train', 'validation') or arm not in ARMS or temperature not in TEMPERATURES:
        raise ValueError('Only registered roles, arms and temperatures may be scored')
    records = []
    for query in prepared:
        index = query.query_index
        fields = field_log_probabilities(query, temperature, arm)
        for subset in SUBSETS:
            values = pool_log_probabilities(fields, subset, arm)
            original = int(cases[index]['label'])
            row = {'role': role, 'case_id': cases[index]['case_id'], 'truth': original,
                   'truth5': int(MAP8_TO5[original]), 'magnifications': [MAGNIFICATIONS[k] for k in subset]}
            row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()})
            records.append(row)
    return records


def model_record(context, cases, train, support, arm, temperature):
    return {'schema_version': 1, 'method': 'centered-cosine class-balanced KDE',
            'arm': arm, 'temperature': temperature, 'support_indices': list(map(int, train)),
            'support_case_ids': [cases[i]['case_id'] for i in train], 'support_summary': support,
            'normalization': NORMALIZATION, 'fitting_evaluation': FITTING_POLICY,
            'prediction_policy': 'same-view uniform class/case/field KDE; joint field probabilities before equal field/view means; hard malignant >=0.5',
            **context}


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    from .bags import fit_normalization
    import hashlib
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing audit output')
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


def run(args, inputs=None):
    from .kernel_model import support_summary
    started = time.monotonic(); output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing run output')
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args) if inputs is None else inputs
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    train, val, excluded = roles[args.seed, args.trial]
    prepared = prepare_queries(bags, y5, train, val)
    support = support_summary(bags, y5, train)
    output.mkdir(parents=True); arms = {}
    for arm in ARMS:
        candidates = []
        for index, temperature in enumerate(TEMPERATURES):
            folder = output/arm/('candidate%d'%index); folder.mkdir(parents=True)
            begin = time.monotonic()
            base.write_json(folder/'model.json', model_record(context, cases, train, support, arm, temperature))
            summaries, hashes = {}, {}
            for role in ('train', 'validation'):
                records = score_records(prepared[role], cases, role, arm, temperature)
                summaries[role] = summarize_records(records, arm)
                path = folder/(role+'-predictions.jsonl')
                path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
                hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = {'index': index, 'temperature': temperature, 'optimized_head_fits': 0,
                         'support_heads': ['direct'] if arm == 'direct' else ['binary', 'malignant'],
                         'model_sha256': file_hash(folder/'model.json'), 'fitting_evaluation': FITTING_POLICY,
                         'fitting_leave_case_out': summaries['train'], 'validation': summaries['validation'],
                         'elapsed_seconds': time.monotonic()-begin, **hashes}
            base.write_json(folder/'metrics.json', candidate); candidates.append(candidate)
        selected = max(range(4), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
        arms[arm] = {'selected_index': selected, 'selected_temperature': TEMPERATURES[selected],
                     'candidates': candidates, 'fitting_leave_case_out': candidates[selected]['fitting_leave_case_out'],
                     'validation': candidates[selected]['validation']}
    report = {'status': 'complete centered-cosine KDE validation run', 'seed': args.seed, 'trial': args.trial,
              'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
              'case_ids': {r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]},
              'role_counts': [46, 8, 28], 'optimized_head_fits': 0, 'support_classifier_instances': 3,
              'outer_predictions': 0, 'fitting_evaluation': FITTING_POLICY, 'fitting_uses_support_labels': True,
              'support_instances_scope': config['support_instances_scope'], 'arms': arms,
              'audit_sha256': file_hash(audit_path), 'elapsed_seconds': time.monotonic()-started,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **context}
    base.write_json(output/'run.json', report)


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('audit', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None: raise ValueError('Run needs audit, seed and trial')
        run(args)
    else:
        from .kernel_collect import collect
        collect(args)

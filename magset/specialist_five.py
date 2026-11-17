"""Fixed binary/malignant UNI2-h metric factorial with inherited temperatures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import time

import numpy as np
from scipy.special import logsumexp

from . import approved_five as approved, kernel_five as kernel
from .bags import fit_normalization
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .field_five import pool_log_probabilities
from .five_collect import read_json, require_equal, verify_audit, verify_records
from .five_metrics import FINAL5, MAP8_TO5, summarize_records
from .kernel_model import (EPS, HEAD_SLICES, class_weights, density_logs,
                           field_log_probabilities, normalize_centered, prepare_case)

ARMS = ('identity', 'binary_only', 'malignant_only', 'both')
FITTING_POLICY = 'in-sample learned metric with whole query omitted from density support'
IDENTITY_FITTING_POLICY = 'leave-one-case-out support and centering; no learned metric'
CONFIG_NAME = 'breakhis-specialist-metric-v1.json'
write_json = approved.base.write_json


def validate_config(config):
    expected = {'name': 'breakhis-specialist-metric-v1', 'schema_version': 1,
                'base_input_config': 'configs/breakhis-uni2-h-v1-kernel.json',
                'arms': list(ARMS), 'production_runs': 10, 'metric_fits': 20,
                'candidate_pipelines': 40}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered specialist metric recipe differs')
    for key in ('base_input_config_sha256', 'reference_comparison_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config.get(key) or ''):
            raise ValueError('Specialist input/reference hash missing')
    if config.get('reference_file') != 'docs/evidence/BREAKHIS_UNI2_H_RESULTS.json':
        raise ValueError('Specialist must inherit the accepted UNI2-h comparison')
    from .specialist_metric import TRAINING_SPEC
    metric = config.get('metric', {})
    expected_metric = {'steps': TRAINING_SPEC['steps'], 'optimizer': TRAINING_SPEC['optimizer'],
        'learning_rate': TRAINING_SPEC['learning_rate'], 'weight_decay': TRAINING_SPEC['weight_decay'],
        'identity_penalty': TRAINING_SPEC['regularization'], 'log_factor_bound': TRAINING_SPEC['log_factor_bound'],
        'training_seed': TRAINING_SPEC['seed'], 'training_dtype': TRAINING_SPEC['dtype'],
        'tf32': False, 'deterministic': True, 'dimension': 1536, 'heads': ['binary', 'malignant']}
    for key, value in expected_metric.items():
        require_equal(metric.get(key), value, 'registered metric/'+key, 0)
    return config


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    own = validate_config(read_json(args.config))
    if file_hash(args.config) != file_hash(root/'configs'/CONFIG_NAME):
        raise ValueError('Specialist config differs from immutable registration')
    base_path = root/own['base_input_config']
    if file_hash(base_path) != own['base_input_config_sha256']:
        raise ValueError('Accepted UNI2-h classifier configuration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = base_path
    original, context, trusted, cases, bags, z, y5, roles = approved.load_inputs(forwarded)
    old_hash = context['config_sha256']
    names = tuple(context['implementation_hashes']) + ('specialist_metric.py', 'specialist_five.py', 'specialist_collect.py')
    context = dict(context, config_sha256=file_hash(args.config), specialist_config_sha256=file_hash(args.config),
                   parent_approved_config_sha256=old_hash,
                   specialist_reference_sha256=own['reference_comparison_sha256'])
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in dict.fromkeys(names)}
    config = {**original, **own}
    reference = load_reference(config, context, roles)
    return config, context, trusted, cases, bags, z, y5, roles, reference


def load_reference(config, context, roles):
    path = Path(__file__).resolve().parents[1]/config['reference_file']
    if file_hash(path) != config['reference_comparison_sha256']:
        raise ValueError('Accepted UNI2-h reference bytes differ')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance', 'feature_audit_sha256'):
        require_equal(reference.get(key), context[key], 'accepted UNI2-h/'+key, 0)
    require_equal(reference.get('config_sha256'), context['parent_approved_config_sha256'], 'parent classifier configuration', 0)
    if (reference.get('verified_runs') != 10 or reference.get('outer_predictions') != 0
            or reference.get('class_names') != list(FINAL5)
            or {(r['seed'], r['trial']) for r in reference.get('runs', [])} != set(roles)
            or len(reference.get('runs', [])) != 10):
        raise ValueError('Accepted UNI2-h original role matrix differs')
    for seed, trial in roles:
        _, temperature, _ = selected_reference(reference, seed, trial)
        require_equal(config.get('inherited_temperatures', {}).get(str(seed), {}).get(str(trial)),
                      temperature, 'registered inherited temperature', 0)
    return reference


def selected_reference(reference, seed, trial):
    matches = [r for r in reference['runs'] if (r['seed'], r['trial']) == (seed, trial)]
    if len(matches) != 1:
        raise ValueError('Exactly one accepted original role required')
    arm = matches[0]['arms']['dual']; index = arm['selected_index']
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('Accepted selected temperature index differs')
    temperature = kernel.TEMPERATURES[index]
    require_equal(arm['selected_temperature'], temperature, 'inherited temperature', 0)
    candidate = arm['candidates'][index]
    require_equal(candidate['temperature'], temperature, 'inherited candidate temperature', 0)
    return index, temperature, candidate


def _metric_fields(prepared, bags, y5, temperature, factors):
    """Independently replace binary and/or malignant geometry, preserving other logs."""
    dimension = bags[0][0].shape[1]
    if not isinstance(factors, dict) or set(factors)-{'binary', 'malignant'}:
        raise ValueError('Expected declared binary/malignant factors')
    for value in factors.values():
        if (value.shape != (dimension,) or value.dtype != np.float64
                or not np.isfinite(value).all() or np.any(value <= 0)):
            raise ValueError('Expected finite positive float64 metric factors')
    active = {head: value for head, value in factors.items() if not np.array_equal(value, np.ones(dimension))}
    if not active:
        return field_log_probabilities(prepared, temperature, 'dual')
    from .specialist_metric import transform_vectors
    original = density_logs(prepared, temperature)
    indices = np.asarray(prepared.support_indices)
    result = []
    for view, old in enumerate(prepared.views):
        center = prepared.centers[view]
        heads = {head: original[head][view] for head in ('binary', 'malignant')}
        for head, value in active.items():
            support = np.concatenate([transform_vectors(normalize_centered(bags[i][view], center), value) for i in indices])
            query = transform_vectors(normalize_centered(bags[prepared.query_index][view], center), value)
            cosine = np.clip(query @ support.T, -1., 1.)
            scaled = cosine/float(temperature); maximum = scaled.max(1, keepdims=True)
            mass = np.exp(scaled-maximum) @ old.weights[:, HEAD_SLICES[head]]
            if not np.isfinite(mass).all() or np.any(mass <= 0):
                raise ValueError('Nonfinite learned head density')
            logs = np.log(mass)+maximum; logs -= logsumexp(logs, axis=1, keepdims=True)
            heads[head] = logs
        binary, malignant = heads['binary'], heads['malignant']
        result.append(np.column_stack((binary[:, 0], binary[:, 1, None]+malignant)))
    return tuple(result)


def score_records(bags, y5, train, indices, cases, role, temperature, factors):
    if role not in ('train', 'validation'):
        raise ValueError('Specialist scoring permits only fitting/validation roles')
    allowed = np.asarray(indices)
    if (allowed.ndim != 1 or len(set(allowed.tolist())) != len(allowed)
            or any((i in train) != (role == 'train') for i in allowed)):
        raise ValueError('Specialist query membership differs')
    records = []
    for index in allowed:
        prepared = prepare_case(bags, y5, train, int(index), leave_query_case_out=(role == 'train'))
        old_fields = field_log_probabilities(prepared, temperature, 'dual')
        fields = _metric_fields(prepared, bags, y5, temperature, factors)
        for subset in kernel.SUBSETS:
            old = pool_log_probabilities(old_fields, subset, 'dual')
            values = pool_log_probabilities(fields, subset, 'dual')
            if 'binary' not in factors:
                # Preserve accepted router even at a floating-point .5 tie.
                values['binary_score'] = old['binary_score']
                values['binary_log_probabilities'] = old['binary_log_probabilities']
                values['prediction'] = (0 if old['binary_score'] < .5 else
                                        1+int(values['malignant_log_probabilities'].argmax()))
            original = int(cases[index]['label'])
            row = {'role': role, 'case_id': cases[index]['case_id'], 'truth': original,
                   'truth5': int(MAP8_TO5[original]),
                   'magnifications': [MAGNIFICATIONS[v] for v in subset]}
            row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()})
            records.append(row)
    return records


def verify_identity(args, reference, seed, trial, cases, bags, y5, train, val):
    index, temperature, candidate = selected_reference(reference, seed, trial)
    path = Path(args.reference_runs)/('split%d'%seed)/('trial%d'%trial)/'dual'/('candidate%d'%index)/'validation-predictions.jsonl'
    if file_hash(path) != candidate['validation_predictions_sha256']:
        raise ValueError('Accepted selected UNI2-h validation prediction bytes differ')
    expected = score_records(bags, y5, train, val, cases, 'validation', temperature, {})
    summary = verify_records(path, expected, cases, val, 'validation', 'dual')
    require_equal(summary, candidate['validation'], 'accepted identity validation summary')
    return {'seed': seed, 'trial': trial, 'temperature': temperature, 'selected_index': index,
            'validation_predictions_sha256': file_hash(path), 'verified_records': len(expected)}


def audit_input(args):
    config, context, _, cases, bags, _, y5, roles, reference = load_inputs(args)
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing specialist audit')
    checks, replay = [], []
    for (seed, trial), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
                       'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
                       'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                       'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
        replay.append(verify_identity(args, reference, seed, trial, cases, bags, y5, train, val))
    output.mkdir(parents=True)
    write_json(output/'audit.json', dict(context, status='zero-fit inputs accepted', new_fits=0,
        new_forwards=0, outer_predictions=0, checks=checks, identity_replay=replay))


def verify_input_audit(path, context, config, bags, y5, roles, reference):
    audit = read_json(path); verify_audit(audit, context, config, bags, y5, roles)
    wanted = []
    for seed, trial in sorted(roles):
        index, temperature, candidate = selected_reference(reference, seed, trial)
        wanted.append({'seed': seed, 'trial': trial, 'temperature': temperature, 'selected_index': index,
                       'validation_predictions_sha256': candidate['validation_predictions_sha256'], 'verified_records': 120})
    require_equal(audit.get('identity_replay'), wanted, 'zero-fit identity replay matrix', 0)
    return audit


def save_factors(path, result):
    np.savez_compressed(path, **{key: np.asarray(result[key]) for key in ('factors', 'log_factors', 'theta')})


def load_factors(path, dimension=1536):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'factors', 'log_factors', 'theta'}:
            raise ValueError('Unexpected metric artifact arrays')
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, dtype in (('factors', np.float64), ('log_factors', np.float64), ('theta', np.float32)):
        if arrays[key].shape != (dimension,) or arrays[key].dtype != dtype or not np.isfinite(arrays[key]).all():
            raise ValueError('Metric factor artifact shape/dtype/finiteness differs')
    logs = .5*np.tanh(arrays['theta'].astype(np.float64))
    if (not np.allclose(logs, arrays['log_factors'], rtol=0, atol=1e-6)
            or not np.allclose(np.exp(logs), arrays['factors'], rtol=0, atol=1e-6)
            or np.any(np.abs(arrays['log_factors']) > .5+2e-7)
            or any(not np.array_equal(arrays[k], arrays[k].astype(np.float32).astype(np.float64))
                   for k in ('factors', 'log_factors'))):
        raise ValueError('Metric factor parameterization differs')
    return arrays


def run(args, smoke=False):
    started = time.monotonic(); output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing specialist run output')
    config, context, _, cases, bags, z, y5, roles, reference = load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_input_audit(audit_path, context, config, bags, y5, roles, reference)
    key = (args.seed, args.trial)
    if smoke and key != (812, 1): raise ValueError('Technical smoke is fitting812/trial1 only')
    train, val, excluded = roles[key]
    centroids = fitting_centroids(bags, train)
    if centroids.shape != (46, 4, 1536) or np.bincount(y5[train], minlength=5).tolist() != [11, 25, 2, 5, 3]:
        raise ValueError('Registered fitting tensor/support differs')
    if not smoke:
        verify_identity(args, reference, *key, cases, bags, y5, train, val)
    index, temperature, _ = selected_reference(reference, *key)
    smoke_indices = smoke_query_indices(y5, train) if smoke else None
    # The optimizer receives only the fitting cases; no validation/excluded tensors.
    from .specialist_metric import fit_metric
    output.mkdir(parents=True); states = {}; receipts = {}; hashes = {}
    for head in ('binary', 'malignant'):
        result = fit_metric(centroids, y5[train], temperature, head=head, device=args.device)
        path = output/(head+'-metric.npz'); save_factors(path, result)
        states[head] = load_factors(path, z.shape[-1])
        training = output/(head+'-training.json')
        write_json(training, {'history': result['history'], 'receipt': result['receipt']})
        receipts[head] = result['receipt']
        hashes[head] = {'metric_sha256': file_hash(path), 'training_sha256': file_hash(training)}
    arms = {}
    for arm in ARMS:
        folder = output/arm; folder.mkdir()
        names = () if arm == 'identity' else ('binary', 'malignant') if arm == 'both' else (arm.removesuffix('_only'),)
        factors = {head: states[head]['factors'] for head in names}
        summaries, prediction_hashes = {}, {}
        items = [('train', smoke_indices if smoke else train, 'fitting')]
        if not smoke: items.append(('validation', val, 'validation'))
        for role, indices, label in items:
            records = score_records(bags, y5, train, indices, cases, role, temperature, factors)
            path = folder/(role+'-predictions.jsonl')
            path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
            summaries[label] = summarize_records(records, 'dual')
            prediction_hashes[role+'_predictions_sha256'] = file_hash(path)
        arms[arm] = dict(summaries, **prediction_hashes)
    model = dict(context, schema_version=1, temperature=temperature, inherited_selected_index=index,
        head_artifacts=hashes,
        fitting_evaluation=FITTING_POLICY, identity_fitting_evaluation=IDENTITY_FITTING_POLICY,
        training_indices=list(map(int, train)), support_policy='same accepted fitting cases; whole query omitted for fitting density support',
        metric_changes='separate binary/malignant diagonals after old support-case centering and L2, then reL2',
        binary_policy='identity and malignant_only reuse accepted binary records exactly; other arms use learned binary metric')
    write_json(output/'model.json', model)
    report = dict(context, status='complete specialist fitting-only technical smoke' if smoke else 'complete specialist metric validation run',
        seed=args.seed, trial=args.trial, class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
        role_counts=[len(train), len(val), len(excluded)], metric_fits=2, outer_predictions=0,
        inherited_selected_index=index, temperature=temperature,
        fitting_evaluation=FITTING_POLICY, identity_fitting_evaluation=IDENTITY_FITTING_POLICY,
        audit_sha256=file_hash(audit_path), head_artifacts=hashes, model_sha256=file_hash(output/'model.json'),
        case_ids={name: [cases[i]['case_id'] for i in ix] for name, ix in
                  [('train', train), ('validation', val), ('excluded', excluded)]},
        training_receipts=receipts, arms=arms,
        elapsed_seconds=time.monotonic()-started, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(output/'run.json', report)
    return report


def smoke_query_indices(y5, train):
    """Use the first fitting case of each final class for complete smoke metrics."""
    labels = np.asarray(y5)
    train = np.asarray(train)
    if (labels.ndim != 1 or labels.dtype.kind not in 'iu' or train.ndim != 1
            or train.dtype.kind not in 'iu' or len(set(train.tolist())) != len(train)
            or np.any((train < 0) | (train >= len(labels)))):
        raise ValueError('Technical smoke needs valid unique fitting indices')
    result = []
    for label in range(5):
        eligible = train[labels[train] == label]
        if not len(eligible):
            raise ValueError('Technical smoke needs fitting support for every final class')
        result.append(int(eligible[0]))
    return np.asarray(result, dtype=int)


def fitting_centroids(bags, train):
    """Preserve the KDE's FP64-before-mean center, rather than approved loader z."""
    values = np.asarray([[np.asarray(fields, dtype=np.float64).mean(0) for fields in bags[i]] for i in train], dtype=np.float64)
    from .kernel_model import _centers
    if not np.array_equal(values.mean(0), np.stack(_centers(bags, train))):
        raise ValueError('Training centroid center differs from accepted KDE center')
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'smoke', 'run', 'collect'))
    for key in ('manifest', 'protocol', 'cache', 'center-cache', 'feature-audit', 'config', 'output', 'reference-runs'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'collect':
        from .specialist_collect import collect
        collect(args)
    else:
        if args.audit is None or args.seed is None or args.trial is None:
            raise ValueError('Run/smoke requires accepted input audit and original role')
        run(args, smoke=args.action == 'smoke')

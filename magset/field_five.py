"""Frozen-Phikon field supervision and joint-probability case pooling.

Dual inference composes each field's joint five-class distribution before any
averaging. Case routing and conditional typing are derived from that pooled
joint distribution, rather than independently averaging router and expert.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from . import five_class as base
from .bags import fit_normalization
from .baseline import SUBSETS as INDEX_SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .five_metrics import FINAL5, MAP8_TO5, summarize_records

FIELD_CONFIG_SHA256 = 'b70bf70726dd733c2ac357198e5047ed27aa8e35454e172b5797af3300349e15'
CONFIG_SHA256 = FIELD_CONFIG_SHA256
ARMS = base.ARMS
DEPENDENCIES = base.DEPENDENCIES + ('field_five.py', 'field_collect.py')
save_model = base.save_model
load_model = base.load_model
linear_log_probabilities = base.linear_log_probabilities
write_json = base.write_json


def load_inputs(args):
    """Extend the immutable accepted input contract without changing its config."""
    if file_hash(args.config) != FIELD_CONFIG_SHA256:
        raise ValueError('Registered field config bytes differ')
    field_config = json.loads(Path(args.config).read_text())
    root = Path(__file__).resolve().parents[1]
    base_path = getattr(args, 'base_config', None) or root/field_config['base_inputs_config']
    if file_hash(base_path) != field_config['base_inputs_config_sha256']:
        raise ValueError('Accepted base input config differs')
    original, context, trusted, cases, bags, z, y5, roles = base.load_inputs(
        args.manifest, args.protocol, args.cache, base_path, args.code_sha)
    for key in ('seeds', 'trials', 'arms', 'C_grid', 'dependencies'):
        if field_config[key] != original[key]:
            raise ValueError('Field/base matched contract differs: '+key)
    if field_config['encoder_dimension'] != original['encoder']['dimension']:
        raise ValueError('Field encoder dimension differs')
    config = {**original, **field_config}
    context = dict(context, field_config_sha256=FIELD_CONFIG_SHA256)
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name)
                                        for name in DEPENDENCIES}
    return config, context, trusted, cases, bags, z, y5, roles


def _indices(indices, count):
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values)
            or not np.issubdtype(values.dtype, np.integer)
            or len(set(values.tolist())) != len(values)
            or np.any((values < 0) | (values >= count))):
        raise ValueError('Expected unique nonempty allowed case indices')
    return values.astype(int)


def field_input(fields, mean, std, view):
    """Preserve accepted float32 standardization and append one-hot view flags."""
    fields = np.asarray(fields)
    if (view not in range(4) or mean.ndim != 2 or mean.shape[0] != 4
            or std.shape != mean.shape or np.any(std <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or fields.ndim != 2 or not len(fields) or fields.shape[1] != mean.shape[1]
            or not np.isfinite(fields).all()):
        raise ValueError('Invalid field features or fitting-only normalizer')
    normalized = (fields.astype(np.float32)-mean[view])/std[view]
    flags = np.zeros((len(fields), 4), dtype=np.float32)
    flags[:, view] = 1
    x = np.concatenate((normalized, flags), axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite standardized field input')
    return x


def field_training_rows(bags, indices, y5, mean, std, head):
    """Class/case/view-balanced fields from fitting indices only.

    Returns x, target, sample_weight. Specialist target order remains 1..4.
    Head weight masses are N, N and N_malignant, not field counts.
    """
    indices = _indices(indices, len(bags))
    y5 = np.asarray(y5)
    if (y5.shape != (len(bags),) or not np.issubdtype(y5.dtype, np.integer)
            or np.any((y5 < 0) | (y5 > 4)) or head not in ('direct', 'binary', 'malignant')):
        raise ValueError('Invalid field head or final-five targets')
    eligible = indices[y5[indices] > 0] if head == 'malignant' else indices
    labels = (y5 > 0).astype(int) if head == 'binary' else y5
    classes = np.arange(2) if head == 'binary' else np.arange(1, 5) if head == 'malignant' else np.arange(5)
    counts = {int(label): int(np.sum(labels[eligible] == label)) for label in classes}
    if not len(eligible) or any(count == 0 for count in counts.values()):
        raise ValueError('Every fitting head class requires at least one case')
    inputs = []; targets = []; weights = []
    for index in eligible:
        if len(bags[index]) != 4:
            raise ValueError('Expected four fitting view bags')
        label = int(labels[index])
        for view in range(4):
            x = field_input(bags[index][view], mean, std, view)
            inputs.append(x)
            targets.append(np.full(len(x), label, dtype=np.int64))
            weights.append(np.full(len(x), len(eligible)/(len(classes)*counts[label]*4*len(x)), dtype=np.float64))
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(weights)


def head_fit(x, y, weights, C, config):
    model = LogisticRegression(C=C, class_weight=None, solver=config['solver'],
                               penalty=config['penalty'], tol=config['tol'],
                               max_iter=config['max_iter'], random_state=config['training_seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    return model


def fit_arm(bags, indices, y5, mean, std, arm, C, config):
    if arm not in ARMS:
        raise ValueError('Unknown field five-class arm')
    names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
    return {name: head_fit(*field_training_rows(bags, indices, y5, mean, std, name), C, config)
            for name in names}


def fitting_diagnostics(models, bags, indices, y5, mean, std, C):
    """Record finite weighted objectives/gradients; this performs no fitting."""
    receipts = {name: {} for name in ('fields_per_head', 'weight_mass_per_head', 'class_mass_per_head')}
    objectives = {}
    for name, model in models.items():
        x, y, weights = field_training_rows(bags, indices, y5, mean, std, name)
        x = x.astype(np.float64)
        logp = linear_log_probabilities(x, model.coef_, model.intercept_)
        encoded = np.searchsorted(model.classes_, y)
        mass = float(weights.sum())
        data_loss = float(-np.dot(weights, logp[np.arange(len(y)), encoded]))
        penalty = float(np.square(model.coef_).sum()/(2*C))
        if name == 'binary':
            residual = (np.exp(logp[:, 1])-encoded)[:, None]
        else:
            residual = np.exp(logp)-np.eye(len(model.classes_))[encoded]
        weighted = weights[:, None]*residual
        gradient = (weighted.T @ x + model.coef_/C)/mass
        intercept_gradient = weighted.sum(axis=0)/mass
        maximum = float(max(np.max(np.abs(gradient)), np.max(np.abs(intercept_gradient))))
        objectives[name] = {'weight_mass': mass, 'data_loss_sum': data_loss,
                            'penalty': penalty, 'objective_sum': data_loss+penalty,
                            'normalized_max_abs_gradient': maximum}
        if not all(np.isfinite(value) for value in objectives[name].values()):
            raise ValueError('Nonfinite fitting objective or gradient diagnostic')
        receipts['fields_per_head'][name] = len(y)
        receipts['weight_mass_per_head'][name] = mass
        receipts['class_mass_per_head'][name] = [float(weights[y == label].sum()) for label in model.classes_]
    return receipts, objectives


def field_log_probabilities(arrays, x, arm):
    if arm == 'direct':
        return linear_log_probabilities(x, arrays['direct_coef'], arrays['direct_intercept'])
    if arm != 'dual':
        raise ValueError('Unknown field five-class arm')
    binary = linear_log_probabilities(x, arrays['binary_coef'], arrays['binary_intercept'])
    malignant = linear_log_probabilities(x, arrays['malignant_coef'], arrays['malignant_intercept'])
    return np.column_stack((binary[:, 0], binary[:, 1, None]+malignant))


def pool_log_probabilities(field_logs, subset, arm):
    """Mean joint probabilities within fields, then equal available-view mean."""
    if (arm not in ARMS or len(field_logs) != 4 or not subset
            or len(set(subset)) != len(subset) or any(view not in range(4) for view in subset)):
        raise ValueError('Expected a valid nonempty magnification subset and arm')
    views = []
    for view in subset:
        logp = np.asarray(field_logs[view], dtype=np.float64)
        if (logp.ndim != 2 or logp.shape[1] != 5 or not len(logp)
                or not np.isfinite(logp).all()
                or not np.allclose(logsumexp(logp, axis=1), 0, rtol=0, atol=1e-10)):
            raise ValueError('Invalid normalized finite field log probabilities')
        views.append(logsumexp(logp, axis=0)-np.log(len(logp)))
    log5 = logsumexp(np.asarray(views), axis=0)-np.log(len(views))
    log_malignant = logsumexp(log5[1:])
    binary_log = np.asarray([log5[0], log_malignant])
    malignant_log = log5[1:]-log_malignant
    score = float(np.clip(np.exp(log_malignant), 0, 1))
    prediction = int(log5.argmax()) if arm == 'direct' else 0 if score < .5 else 1+int(malignant_log.argmax())
    return {'probabilities5': np.exp(log5), 'log_probabilities5': log5, 'prediction': prediction,
            'binary_score': score, 'binary_log_probabilities': binary_log,
            'malignant_probabilities4': np.exp(malignant_log), 'malignant_log_probabilities': malignant_log}


def score_records(arrays, bags, indices, cases, role, arm, models=None):
    if role not in ('train', 'validation') or len(bags) != len(cases) or arm not in ARMS:
        raise ValueError('Only fitting/validation cases in a declared arm may be scored')
    indices = _indices(indices, len(cases))
    logs = {}; oracle_views = {}; max_error = 0.
    for index in indices:
        if len(bags[index]) != 4:
            raise ValueError('Expected four allowed case/view bags')
        logs[index] = []; oracle_views[index] = []
        for view in range(4):
            x = field_input(bags[index][view], arrays['mean'], arrays['std'], view)
            log5 = field_log_probabilities(arrays, x.astype(np.float64), arm)
            logs[index].append(log5)
            if models is not None:
                if arm == 'direct':
                    expected = models['direct'].predict_proba(x)
                else:
                    r = models['binary'].predict_proba(x)
                    q = models['malignant'].predict_proba(x)
                    expected = np.column_stack((r[:, 0], r[:, 1, None]*q))
                error = float(np.max(np.abs(np.exp(log5)-expected)))
                max_error = max(max_error, error)
                if error > 1e-12:
                    raise ValueError('Saved field coefficient/sklearn probability replay differs')
                oracle_views[index].append(expected.mean(axis=0, dtype=np.float64))
    result = []
    for subset in INDEX_SUBSETS:
        for index in indices:
            values = pool_log_probabilities(logs[index], subset, arm)
            if models is not None:
                expected = np.asarray(oracle_views[index])[list(subset)].mean(axis=0)
                error = float(np.max(np.abs(values['probabilities5']-expected)))
                max_error = max(max_error, error)
                if error > 1e-12:
                    raise ValueError('Saved case probability/sklearn replay differs')
            original = int(cases[index]['label'])
            row = {'role': role, 'case_id': cases[index]['case_id'], 'truth': original,
                   'truth5': int(MAP8_TO5[original]), 'magnifications': [MAGNIFICATIONS[k] for k in subset]}
            row.update({key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in values.items()})
            result.append(row)
    return result, max_error


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
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    from .five_collect import read_json, verify_audit
    audit_path = Path(args.audit)/'audit.json'; audit = read_json(audit_path)
    verify_audit(audit, context, config, bags, y5, roles)
    train, val, excluded = roles[args.seed, args.trial]
    mean, std = fit_normalization(bags, train)
    output.mkdir(parents=True)
    reports = {}
    for arm in ARMS:
        candidates = []
        for index, C in enumerate(config['C_grid']):
            folder = output/arm/('candidate%d' % index); folder.mkdir(parents=True)
            begin = time.monotonic()
            models = fit_arm(bags, train, y5, mean, std, arm, C, config)
            receipts, objectives = fitting_diagnostics(models, bags, train, y5, mean, std, C)
            expected_names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
            if any(not np.isclose(receipts['weight_mass_per_head'][name], config['head_weight_mass'][name],
                                  rtol=0, atol=1e-10) for name in expected_names):
                raise ValueError('Registered fitting head mass differs')
            save_model(folder/'model.npz', mean, std, models)
            arrays = load_model(folder/'model.npz', arm, dimension=config['encoder_dimension'])
            values = {}; hashes = {}; replay = 0.
            for role, ix in (('train', train), ('validation', val)):
                records, error = score_records(arrays, bags, ix, cases, role, arm, models)
                values[role] = summarize_records(records, arm); replay = max(replay, error)
                path = folder/(role+'-predictions.jsonl')
                path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
                hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = {'C': C, 'index': index, 'head_fits': len(models),
                         'n_iter': {key: model.n_iter_.tolist() for key, model in models.items()},
                         'sklearn_replay_max_abs': replay, 'elapsed_seconds': time.monotonic()-begin,
                         'model_sha256': file_hash(folder/'model.npz'), 'fitting': values['train'],
                         'validation': values['validation'], 'field_receipts': receipts,
                         'head_objectives': objectives, **hashes}
            write_json(folder/'metrics.json', candidate); candidates.append(candidate)
            print(json.dumps({'arm': arm, 'seed': args.seed, 'trial': args.trial, 'C': C,
                              'validation_F1': candidate['validation']['mean_subset_macro_f1']}), flush=True)
        selected = max(range(len(candidates)), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
        reports[arm] = {'selected_index': selected, 'selected_C': config['C_grid'][selected],
                        'candidates': candidates, 'fitting': candidates[selected]['fitting'],
                        'validation': candidates[selected]['validation']}
    result = {'status': 'complete field-supervised five-class validation run', 'seed': args.seed, 'trial': args.trial,
              'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
              'case_ids': {name: [cases[i]['case_id'] for i in ix] for name, ix in
                           (('train', train), ('validation', val), ('excluded', excluded))},
              'role_counts': [46, 8, 28], 'head_fits': 12, 'outer_predictions': 0,
              'arms': reports, 'audit_sha256': file_hash(audit_path),
              'elapsed_seconds': time.monotonic()-started, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **context}
    write_json(output/'run.json', result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--base-config', type=Path)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path)
    p.add_argument('--runs', type=Path); p.add_argument('--seed', type=int, choices=(812, 813))
    p.add_argument('--trial', type=int, choices=range(1, 6))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit':
        audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None:
            raise ValueError('Run needs accepted audit, seed and trial')
        run(args)
    else:
        from .field_collect import collect
        collect(args)

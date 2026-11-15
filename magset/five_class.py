"""Registered direct-five versus hard-routed binary/malignant-four comparison."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .bags import load_bags, fit_normalization
from .baseline import SUBSETS as INDEX_SUBSETS, stratified_inner_split, vectorize
from .capstone_binary import trusted_binary_labels
from .collect import load_trusted_context
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest
from .five_metrics import FINAL5, MAP8_TO5, five_labels, summarize_records

CONFIG_SHA256 = 'd3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b'
ARMS = ('direct', 'dual')
DEPENDENCIES = ('five_class.py', 'five_metrics.py', 'bags.py', 'baseline.py',
                'capstone_binary.py', 'collect.py', 'data.py', 'five_collect.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_context(code_sha):
    root = Path(__file__).resolve().parents[1]
    if (not re.fullmatch('[0-9a-f]{40}', code_sha)
            or (root/'.source-sha').read_text().strip() != code_sha):
        raise ValueError('Production requires an immutable commit-bound source snapshot')
    archive = (root/'.archive-sha256').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}', archive):
        raise ValueError('Immutable archive receipt missing')
    return {'code_sha': code_sha, 'source_archive_sha256': archive,
            'implementation_hashes': {name: file_hash(root/'magset'/name) for name in DEPENDENCIES}}


def load_inputs(manifest, protocol, cache, config_path, code_sha):
    if file_hash(config_path) != CONFIG_SHA256:
        raise ValueError('Registered config bytes differ')
    config = json.loads(Path(config_path).read_text())
    if tuple(config['final_labels']) != FINAL5 or tuple(config['map8_to5']) != MAP8_TO5:
        raise ValueError('Canonical five-label contract differs')
    context = source_context(code_sha)
    versions = {name: importlib.metadata.version(name) for name in config['dependencies']}
    if versions != config['dependencies']:
        raise ValueError('Pinned numerical environment differs')
    trusted = load_trusted_context(manifest, protocol)
    if any(trusted[k] != config[k] for k in ('manifest_sha256', 'protocol_sha256')):
        raise ValueError('Registered manifest/protocol differs')
    rows = read_manifest(manifest)
    broad = trusted_binary_labels(rows)
    for scale in MAGNIFICATIONS:
        path = Path(cache)/('phikon-v2-%d.npz' % scale)
        expected = config['cache_provenance'][str(scale)]
        if (file_hash(path) != expected['npz_sha256']
                or file_hash(path.with_suffix('.json')) != expected['json_sha256']):
            raise ValueError('Accepted original cache bytes differ')
    cases, bags, metadata = load_bags(rows, cache, 'phikon-v2', trusted['manifest_sha256'])
    if cases != trusted['cases'] or len(cases) != 82 or len(rows) != 7909:
        raise ValueError('Trusted case ordering/release coverage differs')
    if any(any(meta['encoder'].get(k) != v for k, v in config['encoder'].items()) for meta in metadata):
        raise ValueError('Pinned encoder definition differs')
    y8 = np.asarray([c['label'] for c in cases]); y5 = five_labels(y8)
    if any(broad[c['case_id']] != int(y > 0) for c, y in zip(cases, y5)):
        raise ValueError('Provider broad labels and five-label map disagree')
    assignments = {}
    for seed in config['seeds']:
        for trial in config['trials']:
            # Reconstruct ORIGINAL EIGHT labels before pooling the targets.
            indices = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
            if [len(ix) for ix in indices] != config['expected_role_counts']:
                raise ValueError('Retained role counts differ')
            if any(set(indices[i]) & set(indices[j]) for i in range(3) for j in range(i)):
                raise ValueError('Case roles overlap')
            supports = [np.bincount(y5[ix], minlength=5).tolist() for ix in indices]
            if supports != [config[k] for k in ('fitting_five_support', 'validation_five_support', 'excluded_five_support')]:
                raise ValueError('Retained five-class support differs')
            assignments[seed, trial] = indices
    z = np.asarray([[field.mean(axis=0) for field in case] for case in bags], dtype=np.float32)
    if z.shape != (82, 4, 1024) or not np.isfinite(z).all():
        raise ValueError('Mean view representation differs')
    context.update(config_sha256=CONFIG_SHA256, manifest_sha256=trusted['manifest_sha256'],
                   protocol_sha256=trusted['protocol_sha256'], unit=trusted['unit'],
                   dependencies=versions, python=platform.python_version(),
                   encoder=config['encoder'], cache_provenance=config['cache_provenance'])
    return config, context, trusted, cases, bags, z, y5, assignments


def head_fit(x, y, C, config):
    model = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs',
                               penalty='l2', tol=config['tol'], max_iter=config['max_iter'],
                               random_state=config['training_seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y, sample_weight=np.full(len(y), 1/15, dtype=np.float64))
    return model


def fit_arm(x, y5, arm, C, config):
    if arm == 'direct':
        models = {'direct': head_fit(x, y5, C, config)}
    elif arm == 'dual':
        malignant = y5 > 0
        if set(y5[malignant]) != {1, 2, 3, 4}:
            raise ValueError('All malignant fitting subtypes required')
        models = {'binary': head_fit(x, malignant.astype(int), C, config),
                  'malignant': head_fit(x[malignant], y5[malignant], C, config)}
    else:
        raise ValueError('Unknown five-class arm')
    return models


def save_model(path, mean, std, models):
    arrays = {'mean': mean, 'std': std}
    for name, model in models.items():
        for field in ('coef_', 'intercept_', 'classes_'):
            arrays[name+'_'+field.rstrip('_')] = getattr(model, field)
    np.savez_compressed(path, **arrays)


def load_model(path, arm, dimension=1024):
    names = ('direct',) if arm == 'direct' else ('binary', 'malignant') if arm == 'dual' else ()
    if not names:
        raise ValueError('Unknown saved arm')
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    expected = {'mean', 'std'} | {name+'_'+key for name in names for key in ('coef', 'intercept', 'classes')}
    if (set(arrays) != expected or arrays['mean'].shape != (4, dimension)
            or arrays['std'].shape != (4, dimension) or np.any(arrays['std'] <= 0)
            or arrays['mean'].dtype != np.float32 or arrays['std'].dtype != np.float32
            or not all(np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('Saved normalizer/schema/finite values differ')
    for name in names:
        classes = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
        count = 1 if name == 'binary' else len(classes)
        if (arrays[name+'_coef'].shape != (count, dimension+4)
                or arrays[name+'_intercept'].shape != (count,)
                or not np.array_equal(arrays[name+'_classes'], classes)
                or not np.issubdtype(arrays[name+'_classes'].dtype, np.integer)
                or arrays[name+'_coef'].dtype != np.float64
                or arrays[name+'_intercept'].dtype != np.float64):
            raise ValueError('Saved head shape/class order/dtype differs')
    return arrays


def linear_log_probabilities(x, coef, intercept):
    logits = np.asarray(x, dtype=np.float64) @ coef.T + intercept
    if coef.shape[0] == 1:
        a = logits[:, 0]
        return np.column_stack((-np.logaddexp(0, a), -np.logaddexp(0, -a)))
    return logits-logsumexp(logits, axis=1, keepdims=True)


def scores_from_arrays(arrays, x, arm):
    if arm not in ARMS:
        raise ValueError('Unknown scoring arm')
    if arm == 'direct':
        log5 = linear_log_probabilities(x, arrays['direct_coef'], arrays['direct_intercept'])
        log_malignant = logsumexp(log5[:, 1:], axis=1)
        binary_log = np.column_stack((log5[:, 0], log_malignant))
        malignant_log = log5[:, 1:]-log_malignant[:, None]
        probabilities = np.exp(log5)
        pred = probabilities.argmax(axis=1)
    else:
        binary_log = linear_log_probabilities(x, arrays['binary_coef'], arrays['binary_intercept'])
        malignant_log = linear_log_probabilities(x, arrays['malignant_coef'], arrays['malignant_intercept'])
        log5 = np.column_stack((binary_log[:, 0], binary_log[:, 1, None]+malignant_log))
        probabilities = np.exp(log5)
        pred = np.where(np.exp(binary_log[:, 1]) >= .5, 1+malignant_log.argmax(axis=1), 0)
    return {'probabilities5': probabilities, 'log_probabilities5': log5, 'prediction': pred,
            'binary_score': np.clip(np.exp(binary_log[:, 1]), 0., 1.), 'binary_log_probabilities': binary_log,
            'malignant_probabilities4': np.exp(malignant_log), 'malignant_log_probabilities': malignant_log}


def score_records(arrays, z, indices, cases, role, arm, models=None):
    raw_indices = np.asarray(indices)
    if (role not in ('train', 'validation') or raw_indices.ndim != 1 or not len(raw_indices)
            or not np.issubdtype(raw_indices.dtype, np.integer)
            or len(set(raw_indices.tolist())) != len(raw_indices)
            or np.any((raw_indices < 0) | (raw_indices >= len(cases)))):
        raise ValueError('Only valid unique fitting/validation indices may be scored')
    indices = raw_indices.astype(int)
    normalized = (z[indices]-arrays['mean'])/arrays['std']
    result = []; max_error = 0.
    for subset in INDEX_SUBSETS:
        x = vectorize(normalized, subset, 'mean')
        values = scores_from_arrays(arrays, x, arm)
        if models is not None:
            if arm == 'direct':
                expected = models['direct'].predict_proba(x)
                decisions = models['direct'].predict(x)
            else:
                binary = models['binary'].predict_proba(x)
                malignant = models['malignant'].predict_proba(x)
                expected = np.column_stack((binary[:, 0], binary[:, 1, None]*malignant))
                decisions = np.where(binary[:, 1] >= .5, 1+malignant.argmax(axis=1), 0)
            error = float(np.max(np.abs(values['probabilities5']-expected)))
            max_error = max(max_error, error)
            if error > 1e-12 or not np.array_equal(values['prediction'], decisions):
                raise ValueError('Saved coefficient/sklearn probability or explicit decision replay differs')
        for j, i in enumerate(indices):
            original = int(cases[i]['label'])
            row = {'role': role, 'case_id': cases[i]['case_id'], 'truth': original,
                   'truth5': int(MAP8_TO5[original]), 'magnifications': [MAGNIFICATIONS[k] for k in subset]}
            row.update({key: value[j].tolist() for key, value in values.items()})
            result.append(row)
    return result, max_error


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(
        args.manifest, args.protocol, args.cache, args.config, args.code_sha)
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
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(
        args.manifest, args.protocol, args.cache, args.config, args.code_sha)
    audit_path = Path(args.audit)/'audit.json'
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'zero-fit inputs accepted' or audit['new_fits'] != 0
            or any(audit[k] != context[k] for k in context)):
        raise ValueError('Accepted zero-fit audit/source context differs')
    from .five_collect import verify_audit
    verify_audit(audit, context, config, bags, y5, roles)
    train, val, excluded = roles[args.seed, args.trial]
    mean, std = fit_normalization(bags, train)
    normalized = (z[train]-mean)/std
    x = np.concatenate([vectorize(normalized, s, 'mean') for s in INDEX_SUBSETS])
    y = np.tile(y5[train], 15)
    output.mkdir(parents=True)
    reports = {}
    for arm in ARMS:
        candidates = []
        for index, C in enumerate(config['C_grid']):
            folder = output/arm/('candidate%d' % index); folder.mkdir(parents=True)
            begin = time.monotonic(); models = fit_arm(x, y, arm, C, config)
            save_model(folder/'model.npz', mean, std, models)
            arrays = load_model(folder/'model.npz', arm)
            values = {}; hashes = {}; replay = 0.
            for role, ix in (('train', train), ('validation', val)):
                records, error = score_records(arrays, z, ix, cases, role, arm, models)
                values[role] = summarize_records(records, arm); replay = max(replay, error)
                path = folder/(role+'-predictions.jsonl')
                path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
                hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = {'C': C, 'index': index, 'head_fits': len(models),
                         'n_iter': {k: v.n_iter_.tolist() for k, v in models.items()},
                         'sklearn_replay_max_abs': replay, 'elapsed_seconds': time.monotonic()-begin,
                         'model_sha256': file_hash(folder/'model.npz'), 'fitting': values['train'],
                         'validation': values['validation'], **hashes}
            write_json(folder/'metrics.json', candidate)
            candidates.append(candidate)
            print(json.dumps({'arm': arm, 'seed': args.seed, 'trial': args.trial,
                              'C': C, 'validation_F1': candidate['validation']['mean_subset_macro_f1']}), flush=True)
        selected = max(range(4), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
        reports[arm] = {'selected_index': selected, 'selected_C': config['C_grid'][selected],
                        'candidates': candidates, 'fitting': candidates[selected]['fitting'],
                        'validation': candidates[selected]['validation']}
    result = {'status': 'complete five-class validation run', 'seed': args.seed, 'trial': args.trial,
              'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
              'case_ids': {name: [cases[i]['case_id'] for i in ix] for name, ix in
                           (('train', train), ('validation', val), ('excluded', excluded))},
              'role_counts': [46, 8, 28], 'head_fits': 12, 'outer_predictions': 0,
              'arms': reports, 'audit_sha256': file_hash(audit_path),
              'elapsed_seconds': time.monotonic()-started, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              **context}
    write_json(output/'run.json', result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
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
        from .five_collect import collect
        collect(args)

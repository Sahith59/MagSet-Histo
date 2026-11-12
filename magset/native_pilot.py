"""Fixed four-fit native development pilot; sealed patients cannot be scored."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from scipy.special import logsumexp, softmax
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import file_hash
from .tcga_protocol import RESOLUTIONS

ARMS = ('mean', 'duplicate', 'random', 'affinity')
SUBSETS = tuple(s for n in range(1, 5) for s in combinations(range(4), n))
CONFIG = {
    'name': 'native-development-v1', 'projection_seed': 812, 'projection_dimension': 32,
    'affinity_temperature': 0.2, 'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs',
    'max_iter': 1000, 'tol': 1e-4, 'class_weight': None,
    'random_partition_salt': 'magset-native-random-v1',
    'brier_convention': 'mean across patients of sum across16 classes of squared error',
    'class_order': list(range(16)),
}


def allowed(cases, role):
    """Role guards run before any statistic fitting or classifier prediction."""
    if role not in ('fit', 'development'):
        raise ValueError('Sealed/unknown roles are forbidden in the development pilot')
    if not cases or any(c['role'] != role for c in cases):
        raise ValueError('Unexpected patient role for ' + role)
    if len({c['patient_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate patient')


def load_cohort(manifest, protocol, cache, protocol_sha, audit, audit_sha):
    # A full frozen-cache audit may read sealed embeddings. No sealed bag is
    # retained below, and no fitted statistic or prediction uses those rows.
    from .tcga_features import native_rows
    from .verify_native import verify
    if file_hash(audit) != audit_sha:
        raise ValueError('Cache audit differs from the submission-pinned digest')
    verified = verify(manifest, protocol, cache, protocol_sha)
    if json.loads(Path(audit).read_text()) != verified:
        raise ValueError('Frozen cache audit no longer matches the actual caches')
    rows, p = native_rows(manifest, protocol, protocol_sha)
    cases = {pid: {'patient_id': pid, 'role': role,
                  'label': p['patient_records'][pid]['label'], 'views': {}}
             for pid, role in p['roles'].items() if role in ('fit', 'development')}
    if Counter(c['role'] for c in cases.values()) != {'fit': 80, 'development': 16}:
        raise ValueError('Registered native role counts differ')
    for role, count in (('fit', 5), ('development', 1)):
        if Counter(c['label'] for c in cases.values() if c['role'] == role) != {k: count for k in range(16)}:
            raise ValueError('Registered native class counts differ')
    for m, resolution in enumerate(RESOLUTIONS):
        with np.load(Path(cache) / ('resnet50-mpp%s.npz' % resolution), allow_pickle=False) as saved:
            lookup = {path: i for i, path in enumerate(saved['paths'].tolist())}
            values = saved['features']  # NpzFile otherwise decompresses on every access.
            by_patient = {pid: [] for pid in cases}
            for row in rows:
                if row['resolution_mpp'] == resolution and row['patient_id'] in cases:
                    by_patient[row['patient_id']].append(row['relative_path'])
            for pid, paths in by_patient.items():
                if paths:
                    paths.sort()
                    cases[pid]['views'][m] = {'paths': paths,
                        'features': values[[lookup[k] for k in paths]].astype(np.float64)}
    result = sorted(cases.values(), key=lambda c: c['patient_id'])
    return [c for c in result if c['role'] == 'fit'], [c for c in result if c['role'] == 'development'], verified


def fit_moments(cases):
    allowed(cases, 'fit')
    means, stds = [], []
    for m in range(4):
        first, second = [], []
        for c in cases:
            if m not in c['views']:
                continue
            x = np.asarray(c['views'][m]['features'], dtype=np.float64)
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError('Invalid fitting fields')
            first.append(x.mean(axis=0)); second.append(np.square(x).mean(axis=0))
        if not first:
            raise ValueError('No fitting patients at one resolution')
        mu = np.mean(first, axis=0)
        std = np.sqrt(np.maximum(np.mean(second, axis=0) - mu * mu, 0))
        means.append(mu); stds.append(np.where(std > 1e-6, std, 1.0))
    return np.asarray(means), np.asarray(stds)


def projection(dimension):
    if dimension < CONFIG['projection_dimension']:
        raise ValueError('Input dimension is too small for the fixed32D projection')
    rng = np.random.default_rng(CONFIG['projection_seed'])
    q, r = np.linalg.qr(rng.standard_normal((dimension, CONFIG['projection_dimension'])))
    return q * np.where(np.diag(r) >= 0, 1.0, -1.0)


def prepare(case, mean, std, project, subset=None):
    """Order by immutable paths. Excluded view contents never enter preparation."""
    views = tuple(sorted(case['views'])) if subset is None else valid_subset(case, subset)
    prepared = {}
    for m in views:
        view = case['views'][m]; paths = view['paths']
        x = np.asarray(view['features'], dtype=np.float64)
        if (x.ndim != 2 or x.shape != (len(paths), project.shape[0]) or not len(x)
                or len(set(paths)) != len(paths) or not np.isfinite(x).all()):
            raise ValueError('Invalid observed field bag')
        order = np.argsort(paths); paths = [paths[i] for i in order]
        z = (x[order] - mean[m]) / std[m]
        p = z @ project
        unit = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        selected = np.asarray([hashlib.sha256((CONFIG['random_partition_salt'] + '|' + k).encode()).digest()[0] & 1
                               for k in paths], dtype=np.float64)
        if not selected.any():
            selected[:] = 1
        prepared[m] = {'mean': z.mean(axis=0), 'projected': p, 'unit': unit,
                       'random_weights': selected / selected.sum()}
    return prepared


def valid_subset(case, subset):
    values = tuple(subset)
    if not values or len(set(values)) != len(values) or any(m not in case['views'] for m in values):
        raise ValueError('Need a nonempty subset of actually available views')
    return tuple(sorted(values))


def peer_scores(prepared):
    scores = {}
    for i, j in combinations(sorted(prepared), 2):
        sim = prepared[i]['unit'] @ prepared[j]['unit'].T / CONFIG['affinity_temperature']
        scores[i, j] = logsumexp(sim, axis=1) - np.log(sim.shape[1])
        scores[j, i] = logsumexp(sim, axis=0) - np.log(sim.shape[0])
    return scores


def feature(prepared, subset, arm, project, peers=None):
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    subset = valid_subset({'views': prepared}, subset)
    base = np.mean([prepared[m]['mean'] for m in subset], axis=0)
    mask = np.zeros(4); mask[list(subset)] = 1
    block = np.concatenate([base, mask])
    if arm == 'mean':
        return block
    if arm == 'duplicate':
        extra = base @ project
    elif len(subset) == 1:
        extra = np.zeros(CONFIG['projection_dimension'])
    else:
        if arm == 'affinity' and peers is None:
            peers = peer_scores({m: prepared[m] for m in subset})
        contrasts = []
        for m in subset:
            p = prepared[m]['projected']
            if arm == 'random':
                weights = prepared[m]['random_weights']
            else:
                scores = np.mean([peers[m, j] for j in subset if j != m], axis=0)
                weights = softmax(scores)
            contrasts.append(weights @ p - p.mean(axis=0))
        extra = np.mean(contrasts, axis=0)
    return np.concatenate([block, extra])


def design(cases, role, mean, std, project):
    allowed(cases, role)
    x = {arm: [] for arm in ARMS}; labels, weights, records = [], [], []
    started = time.perf_counter()
    for c in cases:
        prepared = prepare(c, mean, std, project); peers = peer_scores(prepared)
        subsets = [s for s in SUBSETS if set(s).issubset(c['views'])]
        if not subsets or (role == 'development' and len(subsets) != 15):
            raise ValueError('Development patients must have all four views')
        for subset in subsets:
            for arm in ARMS:
                x[arm].append(feature(prepared, subset, arm, project, peers))
            labels.append(c['label']); weights.append(1.0 / len(subsets))
            records.append({'patient_id': c['patient_id'], 'role': role,
                            'truth': c['label'], 'subset': list(subset)})
    return ({a: np.asarray(v) for a, v in x.items()}, np.asarray(labels),
            np.asarray(weights), records, time.perf_counter() - started)


def predict_saved(path, x, cases, role):
    allowed(cases, role)
    if role != 'development':
        raise ValueError('Prediction is development-only')
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['classes'], np.arange(16)):
            raise ValueError('Classifier class order is not canonical16')
        return softmax(x @ saved['coef'].T + saved['intercept'], axis=1)


def metrics(records, probabilities):
    truth = np.asarray([r['truth'] for r in records])
    p = np.asarray(probabilities)
    if (p.shape != (len(records), 16) or not len(records) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
            or np.any(truth < 0) or np.any(truth > 15)):
        raise ValueError('Invalid native truth/probabilities')
    result = {}
    for subset in SUBSETS:
        indices = [i for i, r in enumerate(records) if tuple(r['subset']) == subset]
        if not indices:
            raise ValueError('Incomplete development subset coverage')
        y, prob = truth[indices], p[indices]
        if Counter(y.tolist()) != {i: 1 for i in range(16)}:
            raise ValueError('Need exactly one development patient per class/subset')
        pred = prob.argmax(axis=1); cm = confusion_matrix(y, pred, labels=np.arange(16))
        recall = np.diag(cm) / cm.sum(axis=1)
        name = '+'.join(RESOLUTIONS[m] for m in subset)
        result[name] = {'macro_f1': float(f1_score(y, pred, labels=np.arange(16), average='macro', zero_division=0)),
                        'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(recall.mean()),
                        'class_recall': recall.tolist(), 'confusion_matrix': cm.tolist(),
                        'brier': float(np.square(prob - np.eye(16)[y]).sum(axis=1).mean())}
    values = list(result.values())
    return {'mean_subset_macro_f1': float(np.mean([v['macro_f1'] for v in values])),
            'worst_subset_macro_f1': float(min(v['macro_f1'] for v in values)),
            'full_view_macro_f1': result['+'.join(RESOLUTIONS)]['macro_f1'],
            'mean_subset_brier': float(np.mean([v['brier'] for v in values])), 'subsets': result}


def audit_predictions(path, protocol, protocol_sha):
    if file_hash(protocol) != protocol_sha:
        raise ValueError('Registered protocol hash differs')
    p = json.loads(Path(protocol).read_text())
    expected = {(pid, s) for pid, role in p['roles'].items() if role == 'development' for s in SUBSETS}
    by_arm = {a: [] for a in ARMS}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r['arm'] not in ARMS or p['roles'].get(r['patient_id']) != 'development' or r['role'] != 'development':
            raise ValueError('Unregistered arm or forbidden prediction role')
        if r['truth'] != p['patient_records'][r['patient_id']]['label']:
            raise ValueError('Prediction truth differs from trusted protocol')
        by_arm[r['arm']].append(r)
    result = {}
    for arm, records in by_arm.items():
        keys = [(r['patient_id'], tuple(r['subset'])) for r in records]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Prediction memberships/subsets duplicate or incomplete')
        result[arm] = metrics(records, [r['probabilities'] for r in records])
    return result


def run(manifest, protocol, cache, protocol_sha, audit, audit_sha, output, code_sha):
    output = Path(output)
    if output.exists():
        raise ValueError('Pilot output exists; do not rerun/overwrite this fixed budget')
    fit, dev, verified = load_cohort(manifest, protocol, cache, protocol_sha, audit, audit_sha)
    output.mkdir(parents=True)
    # Marker intentionally survives failures. A failed run requires inspection,
    # never an automatic additional configuration or overwrite.
    (output / 'started.json').write_text(json.dumps({'config': CONFIG, 'code_sha': code_sha}, indent=2) + '\n')
    mean, std = fit_moments(fit); project = projection(mean.shape[1])
    np.savez_compressed(output / 'preprocessing.npz', mean=mean, std=std, projection=project)
    fit_x, y, weight, fit_records, fit_feature_seconds = design(fit, 'fit', mean, std, project)
    patient_weight = Counter()
    for r, w in zip(fit_records, weight):
        patient_weight[r['patient_id']] += float(w)
    if not all(np.isclose(w, 1.0) for w in patient_weight.values()):
        raise ValueError('Training weights do not preserve equal patient mass')
    models, info = {}, {}
    for arm in ARMS:
        start = time.perf_counter()
        model = LogisticRegression(C=CONFIG['C'], penalty=CONFIG['penalty'], solver=CONFIG['solver'],
                    max_iter=CONFIG['max_iter'], tol=CONFIG['tol'], class_weight=None)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(fit_x[arm], y, sample_weight=weight)
        if not np.array_equal(model.classes_, np.arange(16)):
            raise ValueError('Fitted classifier lost a class')
        models[arm] = model
        np.savez_compressed(output / (arm + '.npz'), coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
        info[arm] = {'fit_seconds': time.perf_counter() - start, 'iterations': model.n_iter_.tolist(),
                     'input_dimension': fit_x[arm].shape[1],
                     'fit_base_block_rms': float(np.sqrt(np.mean(np.square(fit_x[arm][:, :mean.shape[1]+4])))),
                     'fit_extra_block_rms': None if arm == 'mean' else float(np.sqrt(np.mean(np.square(fit_x[arm][:, -32:])))),
                     'stored_coefficient_count': int(model.coef_.size + model.intercept_.size)}
    # All four fixed fits must converge before any development score is computed.
    with np.load(output / 'preprocessing.npz', allow_pickle=False) as saved:
        for name, original in (('mean', mean), ('std', std), ('projection', project)):
            if not np.array_equal(saved[name], original):
                raise ValueError('Preprocessing save/reload differs')
        mean, std, project = saved['mean'], saved['std'], saved['projection']
    dev_x, _, _, records, dev_feature_seconds = design(dev, 'development', mean, std, project)
    with (output / 'predictions.jsonl').open('w') as stream:
        for arm in ARMS:
            prob = predict_saved(output / (arm + '.npz'), dev_x[arm], dev, 'development')
            reference = models[arm].predict_proba(dev_x[arm])
            error = float(np.max(np.abs(prob - reference)))
            if error > 1e-12:
                raise ValueError('Save/reload prediction differs')
            info[arm]['reload_max_probability_difference'] = error
            for r, values in zip(records, prob):
                stream.write(json.dumps({'arm': arm, **r, 'probabilities': values.tolist()}) + '\n')
    verified_metrics = audit_predictions(output / 'predictions.jsonl', protocol, protocol_sha)
    report = {'schema': 'magset.native-development.v1', 'config': CONFIG, 'code_sha': code_sha,
              'specification_sha256': file_hash(Path(__file__).resolve().parents[1] / 'docs/NATIVE_DEVELOPMENT_PILOT_SPEC.md'),
              'protocol_sha256': protocol_sha, 'manifest_sha256': file_hash(manifest),
              'cache_audit_sha256': audit_sha, 'cache': verified, 'fits_completed': len(models),
              'preprocessing_reload_verified': True,
              'fit_patients': len(fit), 'development_patients': len(dev), 'sealed_predictions_computed': 0,
              'training_rows': len(y), 'training_weight': float(weight.sum()),
              'class_training_weight': {str(k): float(weight[y == k].sum()) for k in range(16)},
              'feature_seconds': {'fit_all_arms_shared': fit_feature_seconds, 'development_all_arms_shared': dev_feature_seconds},
              'arms': {a: {**info[a], 'development': verified_metrics[a]} for a in ARMS},
              'artifacts': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                              'slurm_job_id': os.environ.get('SLURM_JOB_ID')},
              'interpretation': 'Four fixed development fits;16 independent development patients, no confirmation/significance/novelty claim'}
    (output / 'pilot.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({a: verified_metrics[a]['mean_subset_macro_f1'] for a in ARMS}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'cache', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-protocol-sha', 'expected-audit-sha', 'code-sha'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    run(a.manifest, a.protocol, a.cache, a.expected_protocol_sha, a.audit,
        a.expected_audit_sha, a.output, a.code_sha)


if __name__ == '__main__':
    main()

"""Independent NumPy/SciPy review of the complete registered case-bag matrix.

No production-package imports, Torch, model construction, image decoding or
fitting. Checkpoint tensor/lifecycle checks remain the separate collector's job.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import tarfile

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

SOURCE = 'b5f0df18a47c1607be492bdaab7ab6cd2171226b'
ARCHIVE = 'd4655c0290997adc5eaf19aef69c2c5f3e6ab460df66e443e98ff0532613e9ab'
IMPLEMENTATION = 'e34aba8a2112e67b90960cafbeb5aaca1e4c7c8b0568c6f78f4d7af75178663d'
CONFIG = '3f633d0e43f66f51c4e27b3324a17180cda2bf7f2a91c9a757d7b1187da7d7b9'
WEIGHTS = 'e134f5fc07f28cff44a0601a13d79d2eff6304fa7ed7d64332984e227c2785b3'
MANIFEST = '958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31'
PROTOCOL = 'dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134'
REFERENCE = 'bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795'
COLLECTOR = '650c14cb05e75d9b4c1846e86150b372c5faf4a3'
LABELS8 = ('adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma',
           'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma')
LABELS5 = ('benign', 'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma')
MAP = (0, 1, 0, 2, 3, 4, 0, 0)
VIEWS = (40, 100, 200, 400)
SUBSETS = tuple(s for n in range(1, 5) for s in itertools.combinations(VIEWS, n))
MASKS = [sum(1 << VIEWS.index(v) for v in s) for s in SUBSETS]
EPOCHS = (10, 20, 35, 50)
ARMS = ('direct', 'dual')
MEASURES = ('macro_f1', 'accuracy', 'balanced_accuracy', 'cross_entropy', 'brier')
MODULES = ('five_casebag.py', 'five_metrics.py', 'baseline.py', 'data.py', 'collect.py',
           'capstone_binary.py', 'breakhis_predict.py', 'supervised_breakhis.py',
           'adapt_breakhis.py', 'collect_adaptation.py', 'features.py')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decode(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return decode(Path(path).read_text())


class Audit:
    def __init__(self):
        self.checks = 0
        self.max_errors = {}
        self.archives = 0
        self.records = 0

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def same(self, actual, expected, category='metrics', tolerance=1e-10):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), category + ': keys differ')
            for key in expected:
                self.same(actual[key], expected[key], category, tolerance)
        elif isinstance(expected, (list, tuple)):
            self.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), category + ': list differs')
            for left, right in zip(actual, expected):
                self.same(left, right, category, tolerance)
        elif isinstance(expected, (str, bool)) or expected is None:
            self.require(type(actual) is type(expected) and actual == expected, category + ': identity differs')
        else:
            if type(expected) is int:
                self.require(type(actual) is int, category + ': integer type differs')
            left, right = np.asarray(actual), np.asarray(expected)
            self.require(left.shape == right.shape and left.dtype.kind in 'iuf' and right.dtype.kind in 'iuf', category + ': numeric shape/type')
            self.require(np.isfinite(left).all() and np.isfinite(right).all(), category + ': nonfinite value')
            error = float(np.max(np.abs(left - right))) if left.size else 0.
            self.max_errors[category] = max(self.max_errors.get(category, 0.), error)
            self.require(error <= tolerance, category + ': numeric disagreement')


def classification(y, prediction, probabilities, logs, count):
    y = np.asarray(y, dtype=int); prediction = np.asarray(prediction, dtype=int)
    p = np.asarray(probabilities); lp = np.asarray(logs)
    matrix = np.zeros((count, count), dtype=float)
    np.add.at(matrix, (y, prediction), 1)
    support = matrix.sum(1); called = matrix.sum(0); correct = matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(count), where=support > 0)
    f1 = np.divide(2 * correct, support + called, out=np.zeros(count), where=support + called > 0)
    return {'accuracy': float(correct.sum() / len(y)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall[support > 0].mean()),
            'cross_entropy': float(np.mean(-lp[np.arange(len(y)), y])),
            'brier': float(np.mean(np.sum((p - np.eye(count)[y]) ** 2, axis=1))),
            'confusion_matrix': matrix.tolist(), 'class_recall': recall.tolist(),
            'class_support': support.tolist(), 'records': len(y), 'weight_sum': float(len(y)),
            'zero_true_probability_records': int(np.count_nonzero(p[np.arange(len(y)), y] == 0))}


def binary(rows, with_ce=False):
    y = np.asarray([r['truth5'] > 0 for r in rows], dtype=int)
    score = np.asarray([r['binary_score'] for r in rows]); pred = (score >= .5).astype(int)
    matrix = np.zeros((2, 2), dtype=int); np.add.at(matrix, (y, pred), 1)
    support = matrix.sum(1); correct = matrix.diagonal(); called = matrix.sum(0)
    f1 = np.divide(2 * correct, support + called, out=np.zeros(2), where=support + called > 0)
    positive = int(y.sum()); negative = len(y) - positive
    auc = (rankdata(score)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    previous = 0; average_precision = 0.
    for threshold in np.unique(score)[::-1]:
        called_rows = score >= threshold; tp = int(y[called_rows].sum())
        average_precision += (tp - previous) / positive * tp / int(called_rows.sum())
        previous = tp
    result = {'accuracy': float(correct.sum() / len(y)), 'macro_f1': float(f1.mean()),
              'balanced_accuracy': float(np.mean(correct / support)),
              'sensitivity': float(correct[1] / support[1]), 'specificity': float(correct[0] / support[0]),
              'confusion_matrix': matrix.tolist(), 'class_support': support.tolist(),
              'roc_auc': float(auc), 'pr_average_precision': float(average_precision),
              'positive_class_brier': float(np.mean((score - y) ** 2)), 'records': len(y)}
    if with_ce:
        result['cross_entropy'] = float(np.mean([-r['binary_log_probabilities'][t] for r, t in zip(rows, y)]))
    return result


def row_metrics(rows, malignant=False):
    values = [r for r in rows if r['truth5'] > 0] if malignant else rows
    p = np.asarray([r['malignant_probabilities4' if malignant else 'probabilities5'] for r in values])
    lp = np.asarray([r['malignant_log_probabilities' if malignant else 'log_probabilities5'] for r in values])
    return classification([r['truth5'] - int(malignant) for r in values],
                          p.argmax(1) if malignant else [r['prediction'] for r in values], p, lp, 4 if malignant else 5)


def role_summary(rows, cases, arm, role):
    support = [sum(MAP[c['label']] == k for c in cases) for k in range(5)]
    def base(malignant=False):
        subsets = {'+'.join(map(str, s)): row_metrics([r for r in rows if tuple(r['magnifications']) == s], malignant) for s in SUBSETS}
        counts = support[1:] if malignant else support
        result = {'subsets': subsets, 'pooled': row_metrics(rows, malignant),
                  'all_view': subsets['40+100+200+400'], 'distinct_cases': sum(counts),
                  'distinct_class_support': counts, 'worst_subset_macro_f1': min(v['macro_f1'] for v in subsets.values())}
        result.update({'mean_subset_' + k: float(np.mean([v[k] for v in subsets.values()])) for k in MEASURES})
        return result
    result = base(); bs = {'+'.join(map(str, s)): binary([r for r in rows if tuple(r['magnifications']) == s]) for s in SUBSETS}
    mean = {k: np.mean([v[k] for v in bs.values()], axis=0).tolist() for k in next(iter(bs.values()))}
    higher = ('accuracy', 'macro_f1', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'pr_average_precision')
    worst = {k: min(v[k] for v in bs.values()) for k in higher}
    worst['positive_class_brier'] = max(v['positive_class_brier'] for v in bs.values())
    for s in SUBSETS:
        name = '+'.join(map(str, s)); bs[name]['cross_entropy'] = binary([r for r in rows if tuple(r['magnifications']) == s], True)['cross_entropy']
    ce = [v['cross_entropy'] for v in bs.values()]; mean['cross_entropy'] = float(np.mean(ce)); worst['cross_entropy'] = max(ce)
    result.update(class_names=list(LABELS5), role=role, decision_rule=arm,
                  binary_summary={'subsets': bs, 'mean_all_subsets': mean, 'full_view': bs['40+100+200+400'],
                                  'worst_subset_by_metric': worst, 'pooled_case_subset_events': binary(rows, True)},
                  malignant_summary=base(True))
    result['malignant_summary']['class_names'] = list(LABELS5[1:])
    return result


def scores(heads, descriptor, arm):
    x = np.asarray(descriptor, dtype=np.float64)
    if arm == 'direct':
        z = x @ heads['direct_coef'].astype(np.float64).T + heads['direct_intercept'].astype(np.float64)
        lp = z - logsumexp(z, axis=1, keepdims=True); lm = logsumexp(lp[:, 1:], axis=1)
        bl = np.column_stack((lp[:, 0], lm)); ml = lp[:, 1:] - lm[:, None]; pred = lp.argmax(1)
    elif arm == 'dual':
        a = (x @ heads['binary_coef'].astype(np.float64).T + heads['binary_intercept'].astype(np.float64))[:, 0]
        bl = np.column_stack((-np.logaddexp(0, a), -np.logaddexp(0, -a)))
        z = x @ heads['malignant_coef'].astype(np.float64).T + heads['malignant_intercept'].astype(np.float64)
        ml = z - logsumexp(z, axis=1, keepdims=True); lp = np.column_stack((bl[:, 0], bl[:, 1, None] + ml))
        pred = np.where(np.exp(bl[:, 1]) >= .5, 1 + ml.argmax(1), 0)
    else:
        raise ValueError('Unknown arm')
    return {'prediction': pred, 'probabilities5': np.exp(lp), 'log_probabilities5': lp,
            'binary_score': np.clip(np.exp(bl[:, 1]), 0, 1), 'binary_log_probabilities': bl,
            'malignant_probabilities4': np.exp(ml), 'malignant_log_probabilities': ml}


def archive_review(folder, prefix, cases, role, arm, hashes, audit, dimension=1280):
    feature = Path(folder) / (prefix + '-features.npz'); prediction = Path(folder) / (prefix + '-predictions.jsonl')
    audit.require(file_hash(feature) == hashes['features_sha256'] and file_hash(prediction) == hashes['predictions_sha256'], 'Producer archive/prediction hash mismatch')
    with np.load(feature, allow_pickle=False) as loaded:
        data = {k: loaded[k] for k in loaded.files}
        audit.require(len(loaded.files) == len(data), 'Duplicate NPZ key')
    heads = {'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}
    expected = {'case_ids', 'truth8', 'truth5', 'view_features', 'field_counts', 'descriptors', 'descriptor_subset_masks', 'descriptor_magnifications'} | {n + k for n in heads for k in ('_coef', '_intercept', '_classes')}
    audit.require(set(data) == expected, 'NPZ schema differs')
    audit.require(data['case_ids'].dtype.kind == 'U' and data['case_ids'].tolist() == [c['case_id'] for c in cases], 'Trusted role membership/order differs')
    for key, values in [('truth8', [c['label'] for c in cases]), ('truth5', [MAP[c['label']] for c in cases])]:
        audit.require(data[key].dtype == np.int64 and np.array_equal(data[key], values), 'Trusted original/final label differs')
    audit.require(data['field_counts'].dtype == np.int64 and np.array_equal(data['field_counts'], [[len(c['indices'][v]) for v in VIEWS] for c in cases]), 'Trusted field counts differ')
    views = data['view_features']; audit.require(views.dtype == np.float64 and views.shape == (len(cases), 4, dimension) and np.isfinite(views).all(), 'Invalid view means')
    audit.require(data['descriptor_subset_masks'].dtype == np.int64 and np.array_equal(data['descriptor_subset_masks'], MASKS) and data['descriptor_magnifications'].dtype.kind == 'U' and data['descriptor_magnifications'].tolist() == ['+'.join(map(str, s)) for s in SUBSETS], 'Subset axis differs')
    descriptors = []
    for subset in SUBSETS:
        ix = [VIEWS.index(v) for v in subset]; flags = np.zeros((len(cases), 4)); flags[:, ix] = 1
        descriptors.append(np.concatenate((views[:, ix].mean(1, dtype=np.float64), flags), axis=1).astype(np.float32))
    descriptors = np.stack(descriptors, axis=1)
    audit.require(data['descriptors'].dtype == np.float32 and np.array_equal(data['descriptors'], descriptors), 'Descriptor arithmetic differs')
    for name, count in heads.items():
        for suffix, shape in [('_coef', (count, dimension + 4)), ('_intercept', (count,))]:
            a = data[name + suffix]; audit.require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'Invalid head array')
        labels = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
        audit.require(data[name + '_classes'].dtype == np.int64 and np.array_equal(data[name + '_classes'], labels), 'Canonical head order differs')
    expected_rows = []
    for i, subset in enumerate(SUBSETS):
        replay = scores(data, descriptors[:, i], arm)
        for j, case in enumerate(cases):
            row = {'role': role, 'case_id': case['case_id'], 'truth': case['label'], 'truth5': MAP[case['label']], 'magnifications': list(subset)}
            row.update({k: v[j].tolist() for k, v in replay.items()}); expected_rows.append(row)
    actual = [decode(line) for line in prediction.read_text().splitlines()]
    audit.same(actual, expected_rows, 'coefficient_replay')
    summary = role_summary(actual, cases, arm, role)
    if arm == 'dual':
        for name, metric in summary['subsets'].items():
            audit.require(metric['accuracy'] <= summary['binary_summary']['subsets'][name]['accuracy'] + 1e-14, 'Hard-route upper bound failed')
    audit.archives += 1; audit.records += len(actual)
    return summary, data, actual


def select_candidate(candidates):
    if [v['epoch'] for v in candidates] != list(EPOCHS):
        raise ValueError('Incomplete or reordered checkpoint selection matrix')
    return max(candidates, key=lambda v: v['validation']['mean_subset_macro_f1'])


def verify_selection(saved, candidates, audit):
    """Keep checkpoint identity exact while allowing independent math roundoff."""
    best = select_candidate(candidates)
    audit.require(type(saved.get('epoch')) is int and saved['epoch'] == best['epoch'],
                  'Selected epoch is not the earliest maximum')
    for key in ('state_sha256', 'checkpoint_sha256', 'features_sha256', 'predictions_sha256'):
        audit.require(isinstance(saved.get(key), str) and re.fullmatch('[0-9a-f]{64}', saved[key])
                      and saved[key] == best[key], 'Selected artifact identity differs')
    audit.same(saved, best, 'selection', 1e-10)
    return best


def aggregate(values):
    out = {'mean_subset_' + k: float(np.mean([v['mean_subset_' + k] for v in values])) for k in MEASURES}
    out['minimum_trial_worst_subset_macro_f1'] = min(v['worst_subset_macro_f1'] for v in values)
    out['mean_trial_worst_subset_macro_f1'] = float(np.mean([v['worst_subset_macro_f1'] for v in values]))
    out['all_view'] = {k: float(np.mean([v['all_view'][k] for v in values])) for k in MEASURES}
    cm = np.sum([v['pooled']['confusion_matrix'] for v in values], axis=0); support = cm.sum(1)
    out.update(pooled_confusion_matrix=cm.tolist(), pooled_class_support=support.tolist(), pooled_class_recall=(cm.diagonal() / support).tolist(), pooled_case_subset_records=sum(v['pooled']['records'] for v in values))
    keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'pr_average_precision', 'positive_class_brier', 'cross_entropy')
    out['binary_mean_metrics'] = {k: float(np.mean([v['binary_summary']['mean_all_subsets'][k] for v in values])) for k in keys}
    out['malignant_mean_metrics'] = {'mean_subset_' + k: float(np.mean([v['malignant_summary']['mean_subset_' + k] for v in values])) for k in MEASURES}
    return out


def files_per_run():
    result = {'started.json', 'metrics.json', 'last.pt', 'selected.pt'}
    for e in EPOCHS:
        result.update({f'candidate-epoch{e}.pt', f'candidate-epoch{e}-features.npz', f'candidate-epoch{e}-predictions.jsonl'})
    for role in ('train', 'validation'):
        result.update({f'selected-{role}-features.npz', f'selected-{role}-predictions.jsonl'})
    return result


def matrix_paths(root, audit):
    root = Path(root)
    folders = {(a, s, t): root / a / f'split{s}' / f'trial{t}' for a in ARMS for s in (812, 813) for t in range(1, 6)}
    expected = {folder / n for folder in folders.values() for n in files_per_run()}
    paths = list(root.rglob('*'))
    audit.require(not any(p.is_symlink() for p in paths) and {p for p in paths if p.is_file()} == expected, 'Partial or extra production artifact matrix')
    return folders


def trusted_cases(manifest, protocol, config, audit):
    audit.require(file_hash(manifest) == MANIFEST and file_hash(protocol) == PROTOCOL, 'Trusted data hashes differ')
    with Path(manifest).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cases = {}
    for r in rows:
        identity = r['filename'].split('-')[0] + '-' + r['case_id_raw']; label = LABELS8.index(r['subtype']); view = int(r['magnification'])
        audit.require(view in VIEWS and r['broad_class'] == ('benign' if MAP[label] == 0 else 'malignant'), 'Manifest broad/view truth differs')
        c = cases.setdefault(identity, {'case_id': identity, 'label': label, 'indices': {v: [] for v in VIEWS}})
        audit.require(c['label'] == label, 'Inconsistent case truth'); c['indices'][view].append(r['relative_path'])
    audit.require(len(rows) == 7909 and len(cases) == 82 and len({r['relative_path'] for r in rows}) == 7909 and all(all(c['indices'].values()) for c in cases.values()), 'Release coverage differs')
    source = read_json(protocol); roles = {}; fingerprint = []
    for seed in (812, 813):
        for trial in range(1, 6):
            outer = source['trials'][str(trial)]['case_roles']; audit.require(set(outer) == set(cases) and set(outer.values()) == {'train', 'test'}, 'Outer roles differ')
            rng = random.Random(seed); val = [rng.choice(sorted(k for k, c in cases.items() if c['label'] == label and outer[k] == 'train')) for label in range(8)]
            train = sorted(k for k, r in outer.items() if r == 'train' and k not in val); val = sorted(val); excluded = sorted(k for k, r in outer.items() if r == 'test')
            ids = {'train': train, 'validation': val, 'excluded': excluded}
            audit.require([len(v) for v in ids.values()] == [46, 8, 28] and sum(map(len, ids.values())) == len(set().union(*(set(v) for v in ids.values()))) == 82, 'Role disjointness/counts differ')
            for role, support in [('train', [11, 25, 2, 5, 3]), ('validation', [4, 1, 1, 1, 1]), ('excluded', [9, 12, 2, 3, 2])]:
                audit.require([sum(MAP[cases[k]['label']] == i for k in ids[role]) for i in range(5)] == support, 'Pooled support differs')
            roles[seed, trial] = ids
            fingerprint.append({'seed': seed, 'trial': trial, 'roles': {'train': train, 'validation': val, 'test': excluded}})
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    audit.require(digest == config['membership_fingerprint_sha256'], 'Original-eight role fingerprint differs')
    return cases, roles


def review(args):
    audit = Audit(); source = Path(args.training_source_root)
    audit.require((source / '.source-sha').read_text().strip() == SOURCE and (source / '.archive-sha256').read_text().strip() == ARCHIVE, 'Training snapshot receipts differ')
    config_path = source / 'configs/breakhis-casebag-v1.json'; config = read_json(config_path)
    audit.require(file_hash(config_path) == CONFIG and file_hash(source / 'magset/five_casebag.py') == IMPLEMENTATION, 'Frozen config/producer differs')
    implementation = {n: file_hash(source / 'magset' / n) for n in MODULES}
    archive_checked = False
    if args.training_source_archive:
        audit.require(file_hash(args.training_source_archive) == ARCHIVE, 'Training archive bytes differ')
        with tarfile.open(args.training_source_archive) as bundle:
            audit.require(bundle.pax_headers.get('comment') == SOURCE, 'Embedded commit differs')
            seen = set()
            for member in bundle.getmembers():
                p = PurePosixPath(member.name)
                audit.require(not p.is_absolute() and '..' not in p.parts and member.name not in seen and (member.isfile() or member.isdir()), 'Unsafe/duplicate archive member'); seen.add(member.name)
                if member.isfile():
                    target = source.joinpath(*p.parts)
                    audit.require(not target.is_symlink() and target.read_bytes() == bundle.extractfile(member).read(), 'Extracted source byte mismatch')
        archive_checked = True
    cases, roles = trusted_cases(args.manifest, args.protocol, config, audit)
    comparison = read_json(args.comparison); verification = read_json(args.verification)
    audit.require(file_hash(args.comparison) == verification['comparison_sha256'], 'Comparison receipt hash differs')
    for k, v in {'producer_code_sha': SOURCE, 'producer_archive_sha256': ARCHIVE, 'producer_implementation_sha256': IMPLEMENTATION, 'config_sha256': CONFIG, 'collector_code_sha': COLLECTOR, 'verified_fits': 20, 'verified_candidate_checkpoints': 80, 'verified_selected_role_records': 16200, 'verified_candidate_validation_records': 9600, 'new_fits': 0, 'cnn_forwards': 0, 'image_decodes': 0, 'outer_predictions': 0}.items():
        audit.same(verification[k], v, 'verification', 0)
    audit.require(verification['status'] == 'complete case-bag output verification accepted' and verification['checker_execution_device'] == 'cpu', 'Collector verification scope differs')
    common = {'code_sha': SOURCE, 'source_archive_sha256': ARCHIVE, 'implementation_sha256': implementation, 'config_sha256': CONFIG, 'manifest_sha256': MANIFEST, 'protocol_sha256': PROTOCOL, 'weights_sha256': WEIGHTS, 'membership_fingerprint_sha256': config['membership_fingerprint_sha256']}
    for k, v in common.items():
        audit.same(comparison[k], v, 'comparison_context', 0)
    env = comparison['training_environment']; audit.require(env['dependencies'] == config['dependencies'] and env['gpu_family'] == 'A40' and env['device_type'] == 'cuda' and env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'Training environment differs')
    audit.same(verification['checked_training_environment'], env, 'environment', 0)
    paths = matrix_paths(args.runs, audit)
    reported = {(r['arm'], r['seed'], r['trial']): r for r in comparison['runs']}
    audit.require(len(reported) == len(comparison['runs']) == 20 and set(reported) == set(paths), 'Comparison run matrix differs')
    runs = []
    for key, folder in sorted(paths.items()):
        arm, seed, trial = key; report = read_json(folder / 'metrics.json'); context = report['context']; proof = report['proof']; ids = roles[seed, trial]
        audit.require(report['status'] == 'complete case-bag validation run' and report['outer_predictions'] == 0, 'Incomplete/outer production report')
        expected_context = dict(common, config=config, environment=env, arm=arm, seed=seed, trial=trial, smoke=False, case_ids=ids, role_sha256=hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(), smoke_acceptance_sha256=comparison['smoke_acceptance_sha256'])
        audit.same(context, expected_context, 'run_context', 0); audit.same(read_json(folder / 'started.json'), context, 'started_context', 0)
        audit.require([r['epoch'] for r in proof['history']] == list(range(1, 51)) and all(r['steps'] == 12 and r['cases'] == 46 for r in proof['history']), 'Incomplete 50-epoch/600-step history')
        audit.require(file_hash(folder / 'last.pt') == proof['last_checkpoint_sha256'], 'Last checkpoint hash differs')
        candidates = []
        for c in proof['candidates']:
            epoch = c['epoch']; audit.require(epoch in EPOCHS and file_hash(folder / f'candidate-epoch{epoch}.pt') == c['checkpoint_sha256'], 'Candidate checkpoint binding differs')
            metric, _, _ = archive_review(folder, f'candidate-epoch{epoch}', [cases[i] for i in ids['validation']], 'validation', arm, c, audit)
            audit.same(c['validation'], metric, 'candidate_metrics')
            candidates.append({'epoch': epoch, 'validation': metric, **{n: c[n] for n in ('state_sha256', 'checkpoint_sha256', 'features_sha256', 'predictions_sha256')}})
        best = verify_selection(proof['best'], candidates, audit)
        summaries = {}; selected_data = {}; selected_rows = {}
        for role in ('train', 'validation'):
            metric, data, rows = archive_review(folder, 'selected-' + role, [cases[i] for i in ids[role]], role, arm, report[role + '_artifacts'], audit)
            audit.same(report[role], metric, 'selected_metrics'); summaries['fitting' if role == 'train' else role] = metric; selected_data[role] = data; selected_rows[role] = rows
        with np.load(folder / f"candidate-epoch{best['epoch']}-features.npz", allow_pickle=False) as b:
            for name in heads_for(arm):
                for suffix in ('_coef', '_intercept', '_classes'):
                    audit.require(np.array_equal(b[name + suffix], selected_data['train'][name + suffix]) and np.array_equal(b[name + suffix], selected_data['validation'][name + suffix]), 'Selected head differs from chosen candidate')
            feature_error = float(np.max(np.abs(b['view_features'] - selected_data['validation']['view_features'])))
        cr = [decode(s) for s in (folder / f"candidate-epoch{best['epoch']}-predictions.jsonl").read_text().splitlines()]; sr = selected_rows['validation']
        probability_error = float(np.max(np.abs(np.array([r['probabilities5'] for r in cr]) - np.array([r['probabilities5'] for r in sr]))))
        audit.require(feature_error <= 1e-6 and probability_error <= 1e-6 and [r['prediction'] for r in cr] == [r['prediction'] for r in sr], 'Selected raw-output receipt replay differs')
        audit.same(report['selected_feature_replay_max_abs'], feature_error, 'selected_feature_error', 1e-12); audit.same(report['selected_probability_replay_max_abs'], probability_error, 'selected_probability_error', 1e-12)
        audit.same(summaries['validation']['mean_subset_macro_f1'], best['validation']['mean_subset_macro_f1'], 'selected_f1', 1e-12)
        row = reported[key]
        for name, value in [('selected_epoch', best['epoch']), ('candidates', candidates), *summaries.items(), ('metrics_sha256', file_hash(folder / 'metrics.json')), ('last_checkpoint_sha256', file_hash(folder / 'last.pt')), ('selected_checkpoint_sha256', file_hash(folder / 'selected.pt')), ('selected_state_sha256', best['state_sha256']), ('selected_artifacts', {r: report[r + '_artifacts'] for r in ('train', 'validation')}), ('committed_epochs', 50), ('committed_optimizer_updates', 600)]:
            audit.same(row[name], value, 'comparison_runs')
        runs.append({'arm': arm, 'seed': seed, 'trial': trial, **summaries})
    audit.require(audit.archives == 120 and audit.records == 25800, 'Complete numerical archive/record budget differs')
    reference_path = source / 'docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json'; audit.require(file_hash(reference_path) == REFERENCE, 'Frozen direct-five reference differs')
    old = read_json(reference_path); reference = {str(s): old['summary'][str(s)]['direct']['validation'] for s in (812, 813)}
    audit.same(comparison['reference_direct_validation'], reference, 'reference')
    summary = {}; matched = {}; gate = {}
    for seed in (812, 813):
        group = {}
        for arm in ARMS:
            values = [r for r in runs if r['arm'] == arm and r['seed'] == seed]
            group[arm] = {role: aggregate([r[role] for r in values]) for role in ('fitting', 'validation')}
            group[arm]['fitting_minus_validation'] = {k: group[arm]['fitting'][k] - group[arm]['validation'][k] for k in group[arm]['validation'] if k.startswith('mean_subset_')}
        direct, dual = (group[a]['validation'] for a in ARMS)
        matched[str(seed)] = {'delta_mean_subset_macro_f1': dual['mean_subset_macro_f1'] - direct['mean_subset_macro_f1'], 'delta_mean_subset_accuracy': dual['mean_subset_accuracy'] - direct['mean_subset_accuracy']}
        f1 = dual['mean_subset_macro_f1'] - reference[str(seed)]['mean_subset_macro_f1']; acc = dual['mean_subset_accuracy'] - reference[str(seed)]['mean_subset_accuracy']; positive = all(v > 0 for v in dual['pooled_class_recall'])
        gate[str(seed)] = {'delta_mean_subset_macro_f1': f1, 'delta_mean_subset_accuracy': acc, 'all_five_recall_positive': positive, 'gate_passed': bool(f1 >= .02 and acc >= 0 and positive)}; summary[str(seed)] = group
    audit.same(comparison['summary'], summary, 'aggregate_summary'); audit.same(comparison['matched_direct_vs_dual'], matched, 'matched_comparison'); audit.same(comparison['comparison_against_frozen_direct'], gate, 'gate')
    audit.require(comparison['gate_passed'] is all(g['gate_passed'] for g in gate.values()), 'Global gate differs')
    audit.require(comparison['verified_fits'] == 20 and comparison['candidate_checkpoints'] == 80 and comparison['production_head_fits'] == 30 and comparison['class_names'] == list(LABELS5) and comparison['original_class_names'] == list(LABELS8) and comparison['subset_masks'] == MASKS and comparison['routing_threshold'] == .5 and comparison['routing_tie'] == 'malignant', 'Comparison task/counts differ')
    audit.require(all(comparison[k] == 0 for k in ('collector_new_fits', 'collector_cnn_forwards', 'collector_image_decodes', 'outer_predictions')), 'Collector no-work scope differs')
    union = {c for ids in roles.values() for c in ids['validation']}; audit.require(len(union) == comparison['distinct_validation_cases'] == 31, 'Distinct provider-case support differs')
    public_safe(comparison); public_safe(verification)
    result = {'status': 'independent complete case-bag numerical review accepted', 'schema_version': 1, 'review_code_sha256': file_hash(__file__), 'producer_code_sha': SOURCE, 'producer_archive_sha256': ARCHIVE, 'producer_implementation_sha256': IMPLEMENTATION, 'config_sha256': CONFIG, 'manifest_sha256': MANIFEST, 'protocol_sha256': PROTOCOL, 'pretrained_weights_sha256': WEIGHTS, 'comparison_sha256': file_hash(args.comparison), 'verification_sha256': file_hash(args.verification), 'assertions': audit.checks, 'verified_fits': 20, 'candidate_archives': 80, 'selected_role_archives': 40, 'verified_saved_score_records': audit.records, 'distinct_validation_provider_cases': 31, 'training_archive_bytes_checked': archive_checked, 'max_absolute_errors': audit.max_errors, 'gate_passed': comparison['gate_passed'], 'new_fits': 0, 'cnn_forwards': 0, 'image_decodes': 0, 'outer_predictions': 0, 'scope': 'Independent saved NPZ/JSONL numerical/role/hash review. Checkpoint tensor/optimizer and recorded GPU-forward receipt acceptance remain the separate hash-bound collector scope; no CNN image output reproduction.'}
    public_safe(result)
    return result


def heads_for(arm):
    return ('direct',) if arm == 'direct' else ('binary', 'malignant')


def public_safe(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in {'case_id', 'case_ids', 'relative_path', 'filename', 'slurm_job_id'}:
                raise ValueError('Private field in public output')
            public_safe(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            public_safe(item)
    elif isinstance(value, str) and (value.startswith('/') or re.search(r'SOB_[BM]_|\b\d{2}-\d{3,}[A-Za-z]*\b|\\', value)):
        raise ValueError('Private identifier/path in public output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'comparison', 'verification', 'manifest', 'protocol', 'training-source-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--training-source-archive', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing review output')
    result = review(args); args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); temporary.replace(args.output)
    print(json.dumps({k: result[k] for k in ('status', 'verified_fits', 'verified_saved_score_records', 'max_absolute_errors')}))


if __name__ == '__main__':
    main()

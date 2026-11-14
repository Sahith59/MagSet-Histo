"""Read-only saved-validation image/case endpoint bridge; never fit or forward a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .collect import load_trusted_context, verify_run
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest

CONFIG = {
    'analysis': 'saved-validation-image-case-endpoint-bridge-v1',
    'seeds': [812, 813], 'trials': [1, 2, 3, 4, 5],
    'slices': ['all_fields', '40', '100', '200', '400'],
    'readout': 'accepted supervised direct image softmax head',
    'weighted_unit': 'equal class; equal case within class; equal view; uniform fields within case/view',
    'case_40_readout': 'equal-field arithmetic mean probabilities',
    'fit': False, 'forward': False, 'outer_scoring': False,
    'selection': 'none; all five slices and all ten accepted runs required',
}


def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_hash(path, expected):
    actual = file_hash(path)
    if actual != expected:
        raise ValueError('Accepted artifact hash differs')
    return actual


def metrics(probabilities, labels, log_probabilities, weights=None):
    """Eight-class metrics; macro F1 always includes all eight classes."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    lp = np.asarray(log_probabilities, dtype=np.float64)
    if (y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y > 7)) or p.shape != (len(y), 8)
            or lp.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(lp).all()
            or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)
            or not np.allclose(np.exp(lp), p, rtol=0, atol=1e-12)
            or not np.allclose(logsumexp(lp, axis=1), 0, rtol=0, atol=1e-12)):
        raise ValueError('Invalid labels/probabilities/log probabilities')
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0) or w.sum() <= 0:
        raise ValueError('Invalid metric weights')
    matrix = np.zeros((8, 8))
    np.add.at(matrix, (y, p.argmax(1)), w)
    support = matrix.sum(1)
    called = matrix.sum(0)
    tp = matrix.diagonal()
    recall = np.divide(tp, support, out=np.zeros(8), where=support > 0)
    f1 = np.divide(2 * tp, support + called, out=np.zeros(8), where=(support + called) > 0)
    return {'accuracy': float(tp.sum() / w.sum()),
            'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall[support > 0].mean()),
            'cross_entropy': float(np.average(-lp[np.arange(len(y)), y], weights=w)),
            'brier': float(np.average(((p - np.eye(8)[y]) ** 2).sum(1), weights=w)),
            'confusion_matrix': matrix.tolist(), 'class_recall': recall.tolist(),
            'class_support': support.tolist(), 'records': len(y),
            'weight_sum': float(w.sum()),
            'zero_true_probability_records': int(np.sum(p[np.arange(len(y)), y] == 0))}


def equal_values(actual, claimed):
    """Compare independently recomputed values, accepting only tiny floating error."""
    for key, value in actual.items():
        if key not in claimed:
            raise ValueError('Saved metric missing: ' + key)
        a, b = np.asarray(value), np.asarray(claimed[key])
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9):
            raise ValueError('Saved metric differs: ' + key)


def load_fields(path, rows):
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in ('paths', 'logits', 'probabilities', 'log_probabilities')}
    paths = values['paths'].tolist()
    expected = [r['relative_path'] for r in rows]
    if paths != expected or len(set(paths)) != len(expected):
        raise ValueError('Validation field path order/coverage differs')
    for key in ('logits', 'probabilities', 'log_probabilities'):
        a = values[key]
        if a.dtype != np.float64 or a.shape != (len(rows), 8) or not np.isfinite(a).all():
            raise ValueError('Invalid float64 field array')
    lp = values['logits'] - logsumexp(values['logits'], axis=1, keepdims=True)
    if (not np.allclose(lp, values['log_probabilities'], rtol=0, atol=1e-12)
            or not np.allclose(np.exp(lp), values['probabilities'], rtol=0, atol=1e-12)):
        raise ValueError('Field log/probability arithmetic differs')
    return values


def validate_rows(rows, cases):
    labels = {c['case_id']: c['label'] for c in cases}
    if len(labels) != 8 or Counter(labels.values()) != Counter(range(8)):
        raise ValueError('Expected one validation case per class')
    paths = [r['relative_path'] for r in rows]
    if len(paths) != len(set(paths)) or not rows:
        raise ValueError('Missing/duplicate validation fields')
    for r in rows:
        if case_id(r) not in labels or CLASS_NAMES.index(r['subtype']) != labels[case_id(r)]:
            raise ValueError('Field case/truth differs from trusted validation')
    counts = Counter((case_id(r), int(r['magnification'])) for r in rows)
    if set(counts) != {(c, m) for c in labels for m in MAGNIFICATIONS}:
        raise ValueError('Validation case/view coverage differs')
    expected = {p for c in cases for m in MAGNIFICATIONS for p in c['indices'][m]}
    if set(paths) != expected:
        raise ValueError('Validation manifest paths differ from trusted cases')


def balanced_weights(rows):
    """Normalize the predeclared class/case/view/field hierarchy for one slice."""
    case_labels = {case_id(r): CLASS_NAMES.index(r['subtype']) for r in rows}
    classes = Counter(case_labels.values())
    views = {c: {int(r['magnification']) for r in rows if case_id(r) == c} for c in case_labels}
    counts = Counter((case_id(r), int(r['magnification'])) for r in rows)
    return np.asarray([1 / (len(classes) * classes[case_labels[case_id(r)]]
                           * len(views[case_id(r)]) * counts[case_id(r), int(r['magnification'])])
                       for r in rows], dtype=np.float64)


def image_slices(rows, fields):
    result = {}
    for name in CONFIG['slices']:
        indices = [i for i, r in enumerate(rows)
                   if name == 'all_fields' or int(r['magnification']) == int(name)]
        selected = [rows[i] for i in indices]
        y = np.asarray([CLASS_NAMES.index(r['subtype']) for r in selected])
        p = fields['probabilities'][indices]
        lp = fields['log_probabilities'][indices]
        result[name] = {'raw_image': metrics(p, y, lp),
                        'class_case_view_balanced': metrics(p, y, lp, balanced_weights(selected))}
    return result


def case_40_bridge(rows, fields, records, claimed):
    """Reconstruct saved 40x case scores; no field predictions leave this function."""
    selected = [r for r in records if tuple(r['magnifications']) == (40,)]
    groups = {}
    for i, row in enumerate(rows):
        if int(row['magnification']) == 40:
            groups.setdefault(case_id(row), []).append(i)
    if len(selected) != len(groups) or {r['case_id'] for r in selected} != set(groups):
        raise ValueError('Saved 40x case coverage differs')
    ys, ps, lps = [], [], []
    for record in sorted(selected, key=lambda r: r['case_id']):
        indices = groups[record['case_id']]
        truth = CLASS_NAMES.index(rows[indices[0]]['subtype'])
        lp = logsumexp(fields['log_probabilities'][indices], axis=0) - np.log(len(indices))
        p = np.exp(lp)
        direct_mean = fields['probabilities'][indices].mean(0)
        if (record['role'] != 'validation' or record['truth'] != truth
                or not np.allclose(p, direct_mean, rtol=0, atol=1e-12)
                or not np.allclose(p, record['probabilities'], rtol=0, atol=1e-12)
                or not np.allclose(lp, record['log_probabilities'], rtol=0, atol=1e-12)):
            raise ValueError('Saved 40x case probability/truth/role bridge differs')
        ys.append(truth); ps.append(p); lps.append(lp)
    value = metrics(ps, np.asarray(ys), lps)
    compare = {k: v for k, v in value.items() if k not in ('class_support', 'weight_sum')}
    equal_values(compare, claimed)
    counts = Counter(ys)
    balanced_ce = metrics(ps, np.asarray(ys), lps, [1 / (8 * counts[y]) for y in ys])['cross_entropy']
    equal_values({'class_balanced_cross_entropy': balanced_ce}, claimed)
    return {'verified_saved_scores': True, 'verified_saved_metrics': True, 'metrics': value}


def mean_metrics(values):
    return {k: np.asarray([v[k] for v in values]).mean(axis=0).tolist() for k in values[0]}


def collect(manifest, protocol, comparison, expected_comparison_sha256, artifacts):
    """Require the complete accepted 2x5 matrix, emitting aggregates only."""
    comparison_hash = check_hash(comparison, expected_comparison_sha256)
    accepted = json.loads(Path(comparison).read_text())
    trusted = load_trusted_context(manifest, protocol)
    if (accepted['verified_runs'] != 10 or tuple(accepted['class_names']) != CLASS_NAMES
            or any(accepted[k] != trusted[k] for k in ('manifest_sha256', 'protocol_sha256'))):
        raise ValueError('Accepted comparison differs from trusted context')
    expected = {(s, t) for s in CONFIG['seeds'] for t in CONFIG['trials']}
    entries = {(a['seed'], a['trial']): a for a in accepted['artifacts']}
    if len(entries) != len(accepted['artifacts']) or set(entries) != expected:
        raise ValueError('Accepted comparison must contain exactly ten runs')
    paths = list(Path(artifacts).glob('**/metrics.json'))
    if len(paths) != 10:
        raise ValueError('Exactly ten saved run reports required')
    manifest_rows = read_manifest(manifest)
    runs, hashes, case_sets, field_sets = {}, [], {}, {}
    for path in sorted(paths):
        report, verified_hashes = verify_run(path, trusted=trusted)
        key = (report['seed'], report['trial'])
        if key not in expected or key in runs:
            raise ValueError('Unexpected/duplicate run identity')
        entry = entries[key]
        for name, digest in verified_hashes.items():
            if digest != entry[name]:
                raise ValueError('Run differs from accepted comparison hash')
        if (report['encoder'] != 'resnet50-supervised-v1' or report['method'] != 'mean_probability'
                or report.get('outer_predictions_computed') != 0
                or report['config_sha256'] != accepted['config_sha256']
                or report['code_sha'] != accepted['training_source_sha']
                or report['field_artifact_sha256'] != entry['field_artifact_sha256']):
            raise ValueError('Accepted supervised identity/role/provenance differs')
        validation = set(report['case_ids']['validation'])
        cases = [c for c in trusted['cases'] if c['case_id'] in validation]
        rows = [r for r in manifest_rows if case_id(r) in validation]
        validate_rows(rows, cases)
        field_path = path.parent / 'validation-fields.npz'
        field_hash = check_hash(field_path, entry['field_artifact_sha256']['validation'])
        fields = load_fields(field_path, rows)
        records = [json.loads(line) for line in (path.parent / 'predictions.jsonl').read_text().splitlines()]
        slices = image_slices(rows, fields)
        # These are the already accepted direct-image endpoints over all views.
        for kind, saved in (('raw_image', 'raw'), ('class_case_view_balanced', 'balanced')):
            equal_values({k: v for k, v in slices['all_fields'][kind].items()
                          if k not in ('class_support', 'weight_sum')}, report['field_metrics']['validation'][saved])
        bridge = case_40_bridge(rows, fields, records, report['validation']['subsets']['40'])
        runs[key] = {'seed': key[0], 'trial': key[1], 'slices': slices, 'case_40_bridge': bridge}
        hashes.append({'seed': key[0], 'trial': key[1], **verified_hashes,
                       'validation_fields_sha256': field_hash})
        case_sets[key] = validation
        field_sets[key] = {str(m): {r['relative_path'] for r in rows if int(r['magnification']) == m}
                           for m in MAGNIFICATIONS}
    if set(runs) != expected:
        raise ValueError('Complete accepted run matrix required')
    summaries = []
    for seed in CONFIG['seeds']:
        keys = [(seed, t) for t in CONFIG['trials']]
        summaries.append({'seed': seed, 'trials': 5,
            'distinct_validation_cases': len(set.union(*(case_sets[k] for k in keys))),
            'distinct_images_by_view': {str(m): len(set.union(*(field_sets[k][str(m)] for k in keys))) for m in MAGNIFICATIONS},
            'distinct_images_all_views': len(set.union(*(v for k in keys for v in field_sets[k].values()))),
            'repeated_field_evaluations': sum(len(v) for k in keys for v in field_sets[k].values()),
            'five_trial_mean': {name: {kind: mean_metrics([runs[k]['slices'][name][kind] for k in keys])
                                      for kind in ('raw_image', 'class_case_view_balanced')} for name in CONFIG['slices']},
            'case_40_five_trial_mean': mean_metrics([runs[k]['case_40_bridge']['metrics'] for k in keys])})
    counts = {s['seed']: s['distinct_validation_cases'] for s in summaries}
    union_cases = len(set.union(*case_sets.values()))
    if counts != {812: 22, 813: 20} or union_cases != 31:
        raise ValueError('Accepted validation distinct-case counts differ')
    return {'status': 'complete saved-validation endpoint bridge', 'class_names': CLASS_NAMES,
        'analysis_config': CONFIG, 'analysis_config_sha256': config_hash(),
        'analysis_code_sha256': file_hash(__file__),
        'source_hashes': {'accepted_comparison_sha256': comparison_hash,
                          'manifest_sha256': trusted['manifest_sha256'], 'protocol_sha256': trusted['protocol_sha256'],
                          'training_source_sha': accepted['training_source_sha'], 'training_config_sha256': accepted['config_sha256']},
        'verified_runs': 10, 'distinct_validation_cases_union': union_cases,
        'distinct_images_by_view_union': {str(m): len(set.union(*(v[str(m)] for v in field_sets.values()))) for m in MAGNIFICATIONS},
        'distinct_images_all_views_union': len(set.union(*(v for views in field_sets.values() for v in views.values()))),
        'repeated_field_evaluations': sum(len(v) for views in field_sets.values() for v in views.values()),
        'summary': summaries, 'runs': [runs[k] for k in sorted(runs)], 'artifact_hashes': hashes,
        'interpretation': 'Validation only. Repeated fields are evaluations, not independent patients. Same accepted weights and data, different prediction units; no exact notebook reproduction. ResNet trained on all zooms and selected by case macro F1; historical EfficientNet used 40x image validation. No selection among these slices.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'comparison', 'artifacts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-comparison-sha256', required=True)
    args = parser.parse_args()
    result = collect(args.manifest, args.protocol, args.comparison, args.expected_comparison_sha256, args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'verified_runs': result['verified_runs']}))


if __name__ == '__main__':
    main()

"""Retrospective binary secondary diagnostic from accepted eight-class validation scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .collect import SUBSETS, load_trusted_context, verify_run
from .data import CLASS_NAMES, case_id, file_hash, read_manifest

BENIGN = (0, 2, 6, 7)
MALIGNANT = (1, 3, 4, 5)
CONFIG = {
    'analysis': 'retrospective-binary-secondary-v1',
    'seeds': [812, 813], 'trials': [1, 2, 3, 4, 5],
    'pipelines': ['frozen', 'adapted', 'supervised'],
    'benign_indices': list(BENIGN), 'malignant_indices': list(MALIGNANT),
    'positive_class': 'malignant', 'positive_threshold': 0.5, 'tie': 'malignant',
    'subsets': [list(s) for s in SUBSETS],
    'score': 'sum canonical malignant subtype probabilities',
    'selection': 'none; all accepted pipelines, trials, seeds and subsets',
    'cross_entropy': 'omitted; old frozen archives lack stable log probabilities',
    'fit': False, 'forward': False, 'outer_scoring': False,
}


def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_json(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Accepted comparison hash differs')
    return json.loads(Path(path).read_text())


def trusted_binary_labels(rows):
    """Verify provider broad labels agree with the fixed canonical subtype mapping."""
    labels = {}
    for row in rows:
        index = CLASS_NAMES.index(row['subtype'])
        expected = 'malignant' if index in MALIGNANT else 'benign'
        if row['broad_class'] != expected:
            raise ValueError('Manifest broad/subtype labels disagree')
        cid = case_id(row); truth = int(index in MALIGNANT)
        if cid in labels and labels[cid] != truth:
            raise ValueError('Case broad truth changed')
        labels[cid] = truth
    if not labels or set(labels.values()) != {0, 1}:
        raise ValueError('Both trusted broad classes required')
    return labels


def binary_scores(records, labels):
    """Validate full 8x15 coverage again, then collapse only saved P8 vectors."""
    seen = set(); truths = {}; result = {s: [] for s in SUBSETS}
    for row in records:
        subset = tuple(row['magnifications']); cid = row['case_id']
        truth = row['truth']; p = np.asarray(row['probabilities'], dtype=np.float64)
        if (row['role'] != 'validation' or cid not in labels or subset not in SUBSETS
                or (cid, subset) in seen or type(truth) is not int or truth not in range(8)
                or int(truth in MALIGNANT) != labels[cid]
                or p.shape != (8,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError('Invalid saved role/truth/probability/coverage')
        if cid in truths and truths[cid] != truth:
            raise ValueError('Subtype truth changed across subsets')
        truths[cid] = truth; seen.add((cid, subset))
        pm = float(p[list(MALIGNANT)].sum()); pb = float(p[list(BENIGN)].sum())
        if not np.isclose(pm + pb, 1, rtol=0, atol=1e-10):
            raise ValueError('Collapsed probabilities differ')
        result[subset].append((labels[cid], pm))
    if (set(truths) != set(labels) or len(labels) != 8
            or Counter(truths.values()) != Counter(range(8))
            or any(len(v) != 8 for v in result.values())):
        raise ValueError('Expected exactly eight cases and all fifteen subsets')
    return result


def binary_metrics(events):
    array = np.asarray(events, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 2 or not len(array) or not np.isfinite(array).all()
            or not np.isin(array[:, 0], [0, 1]).all() or np.any((array[:, 1] < 0) | (array[:, 1] > 1))):
        raise ValueError('Invalid binary events')
    y = array[:, 0].astype(int); score = array[:, 1]; pred = (score >= .5).astype(int)
    matrix = np.zeros((2, 2), dtype=int); np.add.at(matrix, (y, pred), 1)
    support = matrix.sum(1); called = matrix.sum(0); tp = matrix.diagonal()
    if np.any(support == 0):
        raise ValueError('Both binary classes required for AUC and recall')
    recall = tp / support
    f1 = np.divide(2 * tp, support + called, out=np.zeros(2), where=(support + called) > 0)
    # Mann-Whitney probability: exact score ties contribute one half.
    positive = score[y == 1]; negative = score[y == 0]
    auc = ((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative))
    # AP sums recall increments times precision at each distinct score threshold.
    order = np.argsort(-score, kind='stable'); sy = y[order]; ss = score[order]
    endpoints = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    cumulative_tp = np.cumsum(sy)[endpoints]
    increments = np.diff(np.r_[0, cumulative_tp]) / len(positive)
    ap = np.sum(increments * cumulative_tp / (endpoints + 1))
    return {'accuracy': float(tp.sum() / len(y)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean()), 'sensitivity': float(recall[1]),
            'specificity': float(recall[0]), 'confusion_matrix': matrix.tolist(),
            'class_support': support.tolist(), 'roc_auc': float(auc),
            'pr_average_precision': float(ap), 'positive_class_brier': float(np.mean((score - y) ** 2)),
            'records': len(y)}


def mean_metrics(values):
    return {k: np.asarray([v[k] for v in values]).mean(axis=0).tolist() for k in values[0]}


def run_summary(events):
    subsets = {'+'.join(map(str, s)): binary_metrics(events[s]) for s in SUBSETS}
    values = list(subsets.values())
    # Worst is separately defined for each metric; these are descriptive summaries, not selected readouts.
    higher = ('accuracy', 'macro_f1', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'pr_average_precision')
    worst = {k: min(v[k] for v in values) for k in higher}
    worst['positive_class_brier'] = max(v['positive_class_brier'] for v in values)
    return {'subsets': subsets, 'mean_all_subsets': mean_metrics(values),
            'full_view': subsets['40+100+200+400'], 'worst_subset_by_metric': worst,
            'pooled_case_subset_events': binary_metrics([e for s in SUBSETS for e in events[s]])}


def parent_entries(parent, trusted):
    expected = {(s, t) for s in CONFIG['seeds'] for t in CONFIG['trials']}
    if (parent['verified_runs'] != 10 or tuple(parent['class_names']) != CLASS_NAMES
            or any(parent[k] != trusted[k] for k in ('manifest_sha256', 'protocol_sha256'))):
        raise ValueError('Accepted parent context differs')
    entries = {(a['seed'], a['trial']): a for a in parent['artifacts']}
    if len(entries) != len(parent['artifacts']) or set(entries) != expected:
        raise ValueError('Accepted parent matrix must contain all ten unique runs')
    return entries


def collect(manifest, protocol, adaptation_comparison, expected_adaptation_sha256,
            supervised_comparison, expected_supervised_sha256,
            frozen_artifacts, adapted_artifacts, supervised_artifacts):
    adapted_parent = checked_json(adaptation_comparison, expected_adaptation_sha256)
    supervised_parent = checked_json(supervised_comparison, expected_supervised_sha256)
    trusted = load_trusted_context(manifest, protocol)
    broad = trusted_binary_labels(read_manifest(manifest))
    ae = parent_entries(adapted_parent, trusted); se = parent_entries(supervised_parent, trusted)
    roots = {'frozen': Path(frozen_artifacts), 'adapted': Path(adapted_artifacts), 'supervised': Path(supervised_artifacts)}
    outputs = []; all_cases = set(); shared_membership = {}; artifact_hashes = []
    for pipeline in CONFIG['pipelines']:
        runs = {}; per_seed_cases = {s: set() for s in CONFIG['seeds']}
        for seed in CONFIG['seeds']:
            for trial in CONFIG['trials']:
                key = (seed, trial)
                folder = roots[pipeline] / ('split%d' % seed)
                folder /= ('phikon-v2-mean-trial%d-seed%d' % (trial, seed)) if pipeline == 'frozen' else ('trial%d' % trial)
                report, hashes = verify_run(folder / 'metrics.json', trusted=trusted)
                if (report['seed'], report['trial']) != key or report.get('outer_predictions_computed', 0) != 0:
                    raise ValueError('Run identity/outer role differs')
                expected = ae[key]['frozen_reference'] if pipeline == 'frozen' else (ae[key] if pipeline == 'adapted' else se[key])
                if pipeline == 'frozen' and any(expected[k] != se[key]['frozen_reference'][k] for k in hashes):
                    raise ValueError('Accepted parents disagree on frozen prediction provenance')
                if any(hashes[k] != expected[k] for k in hashes):
                    raise ValueError('Saved run differs from accepted parent hashes')
                identity = {'frozen': ('phikon-v2', 'mean'),
                            'adapted': ('phikon-v2-lastblock-v1', 'mean'),
                            'supervised': ('resnet50-supervised-v1', 'mean_probability')}[pipeline]
                if (report['encoder'], report['method']) != identity:
                    raise ValueError('Accepted pipeline class/readout identity differs')
                if pipeline != 'frozen':
                    parent = adapted_parent if pipeline == 'adapted' else supervised_parent
                    source_key = 'source_sha' if pipeline == 'adapted' else 'training_source_sha'
                    if report['config_sha256'] != parent['config_sha256'] or report['code_sha'] != parent[source_key]:
                        raise ValueError('Accepted training source/config differs')
                cases = set(report['case_ids']['validation'])
                if key in shared_membership and shared_membership[key] != cases:
                    raise ValueError('Pipelines disagree on trusted validation membership')
                shared_membership[key] = cases
                labels = {c: broad[c] for c in cases}
                records = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
                events = binary_scores(records, labels)
                runs[key] = {'seed': seed, 'trial': trial, **run_summary(events)}
                per_seed_cases[seed].update(cases); all_cases.update(cases)
                artifact_hashes.append({'pipeline': pipeline, 'seed': seed, 'trial': trial, **hashes})
        summaries = []
        for seed in CONFIG['seeds']:
            values = [runs[(seed, t)] for t in CONFIG['trials']]
            cases = per_seed_cases[seed]
            summaries.append({'seed': seed, 'trials': 5, 'distinct_validation_cases': len(cases),
                'distinct_case_broad_support': [sum(broad[c] == k for c in cases) for k in (0, 1)],
                'case_subset_events': 600,
                'five_trial_mean_subsets': {name: mean_metrics([v['subsets'][name] for v in values]) for name in values[0]['subsets']},
                'five_trial_mean_all_subsets': mean_metrics([v['mean_all_subsets'] for v in values]),
                'five_trial_mean_full_view': mean_metrics([v['full_view'] for v in values]),
                'five_trial_mean_worst_subset_by_metric': mean_metrics([v['worst_subset_by_metric'] for v in values]),
                'five_trial_mean_pooled_case_subset_events': mean_metrics([v['pooled_case_subset_events'] for v in values])})
        if {s['seed']: s['distinct_validation_cases'] for s in summaries} != {812: 22, 813: 20}:
            raise ValueError('Distinct validation support differs from accepted context')
        outputs.append({'pipeline': pipeline, 'runs': [runs[k] for k in sorted(runs)],
                        'summary': summaries, 'case_subset_events': 1200})
    if len(all_cases) != 31:
        raise ValueError('Distinct validation case union differs')
    return {'status': 'complete retrospective binary secondary diagnostic',
        'analysis_config': CONFIG, 'analysis_config_sha256': config_hash(), 'analysis_code_sha256': file_hash(__file__),
        'source_hashes': {'adaptation_comparison_sha256': expected_adaptation_sha256,
                         'supervised_comparison_sha256': expected_supervised_sha256,
                         'manifest_sha256': trusted['manifest_sha256'], 'protocol_sha256': trusted['protocol_sha256']},
        'subtype_class_order': CLASS_NAMES, 'binary_class_order': ['benign', 'malignant'],
        'distinct_validation_cases_union': len(all_cases),
        'distinct_case_broad_support_union': [sum(broad[c] == k for c in all_cases) for k in (0, 1)],
        'verified_runs': 30, 'pipelines': outputs, 'artifact_hashes': artifact_hashes,
        'analytic_always_malignant_reference': {'kind': 'analytic reference, not a trained model',
            'validation_case_support_per_run': [4, 4], 'accuracy': .5, 'balanced_accuracy': .5,
            'sensitivity': 1., 'specificity': 0., 'macro_f1': 1 / 3},
        'interpretation': 'Retrospective secondary diagnostic using reused inner validation. All three eight-class-selected pipelines are reported without binary optimization, calibration or threshold search. No new fits/forwards or outer scoring. Case-subset events repeat cases, not independent patients; no significance claim. Binary scores cannot replace the eight-class primary gate. The historical paper 86.22% binary image endpoint uses a different protocol and is not directly comparable.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'adaptation-comparison', 'supervised-comparison',
                 'frozen-artifacts', 'adapted-artifacts', 'supervised-artifacts', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-adaptation-sha256', required=True)
    parser.add_argument('--expected-supervised-sha256', required=True)
    a = parser.parse_args()
    result = collect(a.manifest, a.protocol, a.adaptation_comparison, a.expected_adaptation_sha256,
                     a.supervised_comparison, a.expected_supervised_sha256,
                     a.frozen_artifacts, a.adapted_artifacts, a.supervised_artifacts)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'verified_runs': result['verified_runs']}))


if __name__ == '__main__':
    main()

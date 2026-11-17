#!/usr/bin/env python3
"""Read-only NumPy review of the two-head, fitting-only metric smoke."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import specialist_five as runner, specialist_collect as collector
from magset.data import file_hash
from magset.five_collect import assert_public_safe, read_json, require_equal, verify_records
from magset.five_metrics import FINAL5


def fitting_queries(train, labels):
    """Independently choose the first fitting key per final label for metrics."""
    values = [next((int(i) for i in train if labels[i] == label), None) for label in range(5)]
    if any(value is None for value in values):
        raise ValueError('Technical fitting queries must cover all five final labels')
    return np.asarray(values, dtype=int)


def review(args):
    if Path(args.out).exists():
        raise ValueError('Refusing an existing smoke-review output')
    config, context, _, cases, bags, _, labels, roles, reference = runner.load_inputs(args)
    require_equal(config.get('technical_objective_atol'), 1e-5, 'registered technical objective tolerance', 0)
    require_equal(config.get('technical_metric_fits'), 4, 'registered cumulative technical fits', 0)
    audit = Path(args.audit)/'audit.json'
    runner.verify_input_audit(audit, context, config, bags, labels, roles, reference)
    root = Path(args.smoke_root)
    report = read_json(root/'run.json')
    for key, value in context.items():
        require_equal(report.get(key), value, 'smoke source/context/'+key, 0)
    expected = {'status': 'complete specialist fitting-only technical smoke', 'seed': 812, 'trial': 1,
                'class_names': list(FINAL5), 'role_counts': [46, 8, 28], 'metric_fits': 2,
                'outer_predictions': 0, 'audit_sha256': file_hash(audit),
                'fitting_evaluation': runner.FITTING_POLICY,
                'identity_fitting_evaluation': runner.IDENTITY_FITTING_POLICY}
    for key, value in expected.items():
        require_equal(report.get(key), value, 'smoke/'+key, 0)
    train, validation, excluded = roles[812, 1]
    queries = fitting_queries(train, labels)
    wanted_ids = {name: [cases[i]['case_id'] for i in indices] for name, indices in
                  [('train', train), ('validation', validation), ('excluded', excluded)]}
    require_equal(report.get('case_ids'), wanted_ids, 'original smoke roles', 0)
    index, temperature, _ = runner.selected_reference(reference, 812, 1)
    require_equal(report.get('temperature'), temperature, 'fixed inherited temperature', 0)
    require_equal(report.get('inherited_selected_index'), index, 'inherited index', 0)
    paths = {root/arm/'train-predictions.jsonl' for arm in runner.ARMS}
    if set(root.rglob('*predictions*.jsonl')) != paths or list(root.rglob('*validation*')):
        raise ValueError('Smoke must contain fitting-only predictions for exactly four arms')
    if (set(root.rglob('*.npz')) != {root/(head+'-metric.npz') for head in ('binary', 'malignant')}
            or set(root.rglob('*-training.json')) != {root/(head+'-training.json') for head in ('binary', 'malignant')}):
        raise ValueError('Smoke metric/history artifact matrix differs')
    if set(report.get('arms', {})) != set(runner.ARMS) or set(report.get('head_artifacts', {})) != {'binary', 'malignant'}:
        raise ValueError('Incomplete smoke arm/head matrix')
    if file_hash(root/'model.json') != report.get('model_sha256'):
        raise ValueError('Smoke model bytes differ')
    model = read_json(root/'model.json')
    for key, value in {**context, 'temperature': temperature, 'inherited_selected_index': index,
                       'head_artifacts': report['head_artifacts'], 'training_indices': list(map(int, train))}.items():
        require_equal(model.get(key), value, 'smoke model/'+key, 0)
    centroids = runner.fitting_centroids(bags, train)
    if centroids.shape != (46, 4, 1536):
        raise ValueError('Smoke fitting tensor dimensions differ')
    states, receipts, diagnostics = {}, {}, {}
    for head in ('binary', 'malignant'):
        states[head], receipts[head] = collector.verify_training(root, head, centroids, labels[train], temperature,
                                                               report['head_artifacts'][head])
        receipt = receipts[head]
        history = read_json(root/(head+'-training.json'))['history']
        if not any(row['gradient_l2'] > 0 for row in history) or not np.any(states[head]['theta'] != 0):
            raise ValueError('Real technical metric has no active learning update')
        differences = [abs(receipt[phase+'_torch_objective'][key]-receipt[phase+'_numpy_objective'][key])
                       for phase in ('initial', 'final') for key in ('loss', 'ce', 'penalty')]
        if max(differences) > config['technical_objective_atol']:
            raise ValueError('Technical FP32/FP64 objective parity failed')
        diagnostics[head] = {'updates': len(history), 'maximum_gradient_l2': max(row['gradient_l2'] for row in history),
                             'maximum_objective_parity_error': max(differences),
                             'theta_sha256': receipt['theta_sha256'], 'centers_sha256': receipt['centers_sha256'],
                             'factor_min': receipt['factor_min'], 'factor_max': receipt['factor_max']}
    require_equal(report.get('training_receipts'), receipts, 'embedded smoke training receipts', 0)
    count, identity = 0, None
    for arm in runner.ARMS:
        saved = report['arms'][arm]
        if set(saved) != {'fitting', 'train_predictions_sha256'}:
            raise ValueError('Smoke arm contains validation or unexpected output')
        names = () if arm == 'identity' else ('binary', 'malignant') if arm == 'both' else (arm.removesuffix('_only'),)
        factors = {head: states[head]['factors'] for head in names}
        wanted = collector.replay_records(bags, labels, train, queries, cases, 'train', temperature, factors)
        path = root/arm/'train-predictions.jsonl'
        require_equal(file_hash(path), saved['train_predictions_sha256'], 'smoke prediction bytes', 0)
        summary = verify_records(path, wanted, cases, queries, 'train', 'dual')
        require_equal(saved['fitting'], summary, 'smoke fitting summary')
        actual = [collector.read_json_line(line) for line in path.read_text().splitlines()]
        if arm == 'identity':
            identity = {(row['case_id'], tuple(row['magnifications'])): row for row in actual}
        if arm == 'malignant_only':
            for row in actual:
                old = identity[row['case_id'], tuple(row['magnifications'])]
                for key in ('binary_score', 'binary_log_probabilities'):
                    require_equal(row[key], old[key], 'exact smoke router preservation', 0)
                require_equal(row['prediction'] == 0, old['prediction'] == 0, 'unchanged smoke route', 0)
        count += len(wanted)
    require_equal(count, 300, 'complete fitting-only smoke record count', 0)
    result = {key: context[key] for key in ('code_sha', 'source_archive_sha256', 'implementation_hashes',
              'config_sha256', 'manifest_sha256', 'protocol_sha256', 'feature_audit_sha256')}
    result.update(status='accepted independent NumPy fitting-only specialist technical smoke', seed=812, trial=1,
                  metric_fits=2, fitting_queries=5, fitting_prediction_records=count, validation_prediction_records=0,
                  registered_cumulative_technical_metric_fits=config['technical_metric_fits'],
                  reviewer_metric_fits=0, outer_predictions=0, new_encoder_forwards=0, dimension=1536, temperature=temperature,
                  objective_atol=config['technical_objective_atol'], prediction_atol=1e-12,
                  audit_sha256=file_hash(audit), smoke_run_sha256=file_hash(root/'run.json'),
                  model_sha256=file_hash(root/'model.json'), head_artifacts=report['head_artifacts'],
                  technical_diagnostics=diagnostics,
                  reviewer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert_public_safe(result)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(args.out, result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'cache', 'center-cache', 'feature-audit', 'config', 'reference-runs',
                 'audit', 'smoke-root', 'out'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--code-sha', required=True)
    return result


if __name__ == '__main__':
    receipt = review(parser().parse_args())
    print(receipt['status'])

"""Independently verify every registered frozen-Phikon field candidate without fitting."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .bags import fit_normalization
from .baseline import SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .five_class import ARMS, load_model, write_json
from .five_collect import (PUBLIC_CONTEXT, aggregate_summaries, assert_public_safe,
                           read_json, require_equal, verify_audit, verify_records)
from .five_metrics import FINAL5, MAP8_TO5


def training_rows(bags, indices, y5, mean, std, head):
    """Reconstruct weights independently; every head class/case/view has equal mass."""
    indices = [int(i) for i in indices if head != 'malignant' or y5[i] > 0]
    labels = np.asarray([int(y5[i] > 0) if head == 'binary' else y5[i] for i in indices])
    classes = np.arange(2) if head == 'binary' else np.arange(1, 5) if head == 'malignant' else np.arange(5)
    if set(labels) != set(classes) or len(set(indices)) != len(indices):
        raise ValueError('Fitting head classes/indices differ')
    counts = {label: int(np.sum(labels == label)) for label in classes}
    blocks, targets, weights = [], [], []
    for i, label in zip(indices, labels):
        for view in range(4):
            fields = np.asarray(bags[i][view], dtype=np.float32)
            if fields.ndim != 2 or not len(fields) or not np.isfinite(fields).all():
                raise ValueError('Invalid fitting field bag')
            mask = np.zeros((len(fields), 4)); mask[:, view] = 1
            blocks.append(np.column_stack(((fields-mean[view])/std[view], mask)))
            targets.extend([label]*len(fields))
            weights.extend([len(indices)/(len(classes)*counts[label]*4*len(fields))]*len(fields))
    return np.concatenate(blocks), np.asarray(targets), np.asarray(weights)


def field_receipts(rows):
    return {'fields_per_head': {h: len(y) for h, (_, y, _) in rows.items()},
            'weight_mass_per_head': {h: float(w.sum()) for h, (_, _, w) in rows.items()},
            'class_mass_per_head': {h: [float(w[y == label].sum()) for label in np.unique(y)]
                                    for h, (_, y, w) in rows.items()}}


def logits(x, arrays, head):
    # Avoid platform BLAS reductions in this independent numerical replay.
    return np.einsum('nd,kd->nk', x, arrays[head+'_coef'], optimize=False)+arrays[head+'_intercept']


def log_probabilities(value):
    if value.shape[1] == 1:
        a = value[:, 0]
        return np.column_stack((-np.logaddexp(0, a), -np.logaddexp(0, -a)))
    return value-logsumexp(value, axis=1, keepdims=True)


def head_objectives(arrays, rows, C, config):
    """Weighted sklearn L2 objective and its independently calculated gradient."""
    result = {}
    for head, (x, labels, weights) in rows.items():
        coefficient = arrays[head+'_coef']; classes = arrays[head+'_classes']
        logs = log_probabilities(logits(x, arrays, head)); probability = np.exp(logs)
        target = np.searchsorted(classes, labels); mass = float(weights.sum())
        loss = float(-np.sum(weights*logs[np.arange(len(labels)), target]))
        penalty = float(np.square(coefficient).sum()/(2*C))
        residual = probability.copy(); residual[np.arange(len(labels)), target] -= 1
        if len(classes) == 2:
            residual = residual[:, 1, None]
        weighted = residual*weights[:, None]
        gradient = np.einsum('nk,nd->kd', weighted, x, optimize=False)/mass+coefficient/(C*mass)
        intercept_gradient = weighted.sum(axis=0)/mass
        maximum = float(max(np.max(np.abs(gradient)), np.max(np.abs(intercept_gradient))))
        values = {'weight_mass': mass, 'data_loss_sum': loss, 'penalty': penalty,
                  'objective_sum': loss+penalty, 'normalized_max_abs_gradient': maximum}
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('Nonfinite/negative fitted head objective')
        result[head] = values
    return result


def replay_records(arrays, bags, indices, cases, role, arm):
    """Independent log-space field-joint pooling; never call the trainer scorer."""
    result = []
    for i in indices:
        views = []
        for view in range(4):
            fields = np.asarray(bags[i][view], dtype=np.float32)
            mask = np.zeros((len(fields), 4)); mask[:, view] = 1
            x = np.column_stack(((fields-arrays['mean'][view])/arrays['std'][view], mask))
            if arm == 'direct':
                joint = log_probabilities(logits(x, arrays, 'direct'))
            else:
                binary = log_probabilities(logits(x, arrays, 'binary'))
                malignant = log_probabilities(logits(x, arrays, 'malignant'))
                joint = np.column_stack((binary[:, 0], binary[:, 1, None]+malignant))
            views.append(logsumexp(joint, axis=0)-np.log(len(fields)))
        for subset in SUBSETS:
            log5 = logsumexp(np.asarray(views)[list(subset)], axis=0)-np.log(len(subset))
            log_r = logsumexp(log5[1:]); binary = np.asarray([log5[0], log_r])
            conditional = log5[1:]-log_r
            probability = np.exp(log5); routing = float(np.clip(np.exp(log_r), 0, 1))
            decision = int(probability.argmax()) if arm == 'direct' else (1+int(conditional.argmax()) if routing >= .5 else 0)
            row = {'role': role, 'case_id': cases[i]['case_id'], 'truth': int(cases[i]['label']),
                   'truth5': int(MAP8_TO5[cases[i]['label']]),
                   'magnifications': [MAGNIFICATIONS[k] for k in subset],
                   'probabilities5': probability.tolist(), 'log_probabilities5': log5.tolist(),
                   'prediction': decision, 'binary_score': routing, 'binary_log_probabilities': binary.tolist(),
                   'malignant_probabilities4': np.exp(conditional).tolist(),
                   'malignant_log_probabilities': conditional.tolist()}
            if not np.isfinite(log5).all() or not np.isfinite(conditional).all():
                raise ValueError('Nonfinite independent pooled scores')
            result.append(row)
    return result


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha256):
    """Verify one complete two-arm/four-candidate run and return public-safe fields."""
    folder = Path(folder)
    report = read_json(folder/'run.json')
    for key, value in context.items():
        require_equal(report.get(key), value, 'run/context/'+key, 0)
    if (report.get('status') != 'complete field-supervised five-class validation run'
            or (report.get('seed'), report.get('trial')) != (seed, trial)
            or type(report.get('head_fits')) is not int or report['head_fits'] != 12
            or type(report.get('outer_predictions')) is not int or report['outer_predictions'] != 0
            or report.get('audit_sha256') != audit_sha256
            or report.get('class_names') != list(FINAL5)
            or report.get('original_class_names') != list(CLASS_NAMES)
            or set(report.get('arms', {})) != set(ARMS)):
        raise ValueError('Run identity/head budget/audit/label contract differs')
    train, val, excluded = roles[seed, trial]
    require_equal(report.get('role_counts'), [len(train), len(val), len(excluded)], 'role counts', 0)
    expected_ids = {name: [cases[i]['case_id'] for i in ix] for name, ix in
                    (('train', train), ('validation', val), ('excluded', excluded))}
    require_equal(report.get('case_ids'), expected_ids, 'original-eight role membership', 0)
    mean, std = fit_normalization(bags, train)
    result = {'seed': seed, 'trial': trial, 'head_fits': report['head_fits'],
              'run_sha256': file_hash(folder/'run.json'), 'arms': {}}
    expected_models = set()
    expected_scores = set()
    for arm in ARMS:
        heads = ('direct',) if arm == 'direct' else ('binary', 'malignant')
        rows = {h: training_rows(bags, train, y5, mean, std, h) for h in heads}
        receipt = field_receipts(rows)
        saved_arm = report['arms'][arm]
        candidates = saved_arm.get('candidates')
        if not isinstance(candidates, list) or len(candidates) != len(config['C_grid']):
            raise ValueError('Incomplete candidate matrix')
        public_candidates = []
        for index, C in enumerate(config['C_grid']):
            directory = folder/arm/('candidate%d' % index)
            candidate = read_json(directory/'metrics.json')
            require_equal(candidates[index], candidate, 'embedded candidate', 0)
            names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
            if (type(candidate.get('index')) is not int or candidate['index'] != index
                    or candidate.get('C') != C or isinstance(candidate.get('C'), bool)
                    or type(candidate.get('head_fits')) is not int or candidate['head_fits'] != len(names)
                    or set(candidate.get('n_iter', {})) != set(names)):
                raise ValueError('Candidate index/C/head budget differs')
            for iterations in candidate['n_iter'].values():
                if (not isinstance(iterations, list) or len(iterations) != 1
                        or type(iterations[0]) is not int or not 1 <= iterations[0] <= config['max_iter']):
                    raise ValueError('Candidate convergence/iteration metadata differs')
            for name in ('elapsed_seconds', 'sklearn_replay_max_abs'):
                value = candidate.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Candidate time/replay metadata differs')
            if candidate['sklearn_replay_max_abs'] > 1e-12:
                raise ValueError('Candidate sklearn replay exceeds registered tolerance')
            model_path = directory/'model.npz'; expected_models.add(model_path)
            if file_hash(model_path) != candidate.get('model_sha256'):
                raise ValueError('Saved model hash differs')
            arrays = load_model(model_path, arm, dimension=config['encoder']['dimension'])
            if not np.array_equal(arrays['mean'], mean) or not np.array_equal(arrays['std'], std):
                raise ValueError('Saved normalizer differs bitwise from fitting-only moments')
            require_equal(candidate.get('field_receipts'), receipt, 'field receipt', 1e-10)
            objective = head_objectives(arrays, rows, C, config)
            require_equal(candidate.get('head_objectives'), objective, 'head objective', 1e-8)
            summaries = {}
            for role, indices, summary_name in (('train', train, 'fitting'), ('validation', val, 'validation')):
                path = directory/(role+'-predictions.jsonl'); expected_scores.add(path)
                if file_hash(path) != candidate.get(role+'_predictions_sha256'):
                    raise ValueError('Saved predictions hash differs')
                expected = replay_records(arrays, bags, indices, cases, role, arm)
                summary = verify_records(path, expected, cases, indices, role, arm)
                require_equal(candidate.get(summary_name), summary, 'candidate/'+summary_name)
                summaries[summary_name] = summary
            public_candidates.append({key: candidate[key] for key in
                ('C', 'index', 'head_fits', 'n_iter', 'sklearn_replay_max_abs', 'elapsed_seconds',
                 'model_sha256', 'train_predictions_sha256', 'validation_predictions_sha256',
                 'field_receipts', 'head_objectives')})
            public_candidates[-1].update(summaries, metrics_sha256=file_hash(directory/'metrics.json'))
        selected = max(range(len(public_candidates)),
                       key=lambda i: public_candidates[i]['validation']['mean_subset_macro_f1'])
        if (type(saved_arm.get('selected_index')) is not int or saved_arm['selected_index'] != selected
                or saved_arm.get('selected_C') != config['C_grid'][selected]):
            raise ValueError('Selected candidate differs from earliest maximum validation hard-label F1')
        for name in ('fitting', 'validation'):
            require_equal(saved_arm.get(name), public_candidates[selected][name], 'selected/'+name)
        result['arms'][arm] = {'selected_index': selected, 'selected_C': config['C_grid'][selected],
                              'candidates': public_candidates,
                              'fitting': public_candidates[selected]['fitting'],
                              'validation': public_candidates[selected]['validation']}
    if (set(folder.rglob('*.npz')) != expected_models
            or set(folder.rglob('*predictions*.jsonl')) != expected_scores):
        raise ValueError('Unexpected model/prediction artifacts, including forbidden extra roles')
    return result



def reference_comparison(config, context, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json'
    if file_hash(path) != config['reference_comparison_sha256']:
        raise ValueError('Accepted frozen case-head reference hash differs')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    if (reference.get('verified_runs') != 10 or reference.get('head_fits') != 120
            or reference.get('outer_predictions') != 0
            or reference.get('class_names') != list(FINAL5)):
        raise ValueError('Reference comparison scope differs')
    return reference


def collect(args):
    """Publish only a complete hash-bound matrix and its fixed reference comparison."""
    from .field_five import load_inputs
    if args.runs is None or args.audit is None:
        raise ValueError('Collector requires --runs and accepted --audit')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing comparison output')
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    reference = reference_comparison(config, context, getattr(args, 'reference', None))
    audit_sha = file_hash(audit_path); root = Path(args.runs)
    keys = {(s, t) for s in config['seeds'] for t in config['trials']}
    paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in keys}
    if set(root.rglob('run.json')) != paths:
        raise ValueError('Complete registered run matrix is missing or has extra runs')
    runs = [verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context,
                       cases, bags, z, y5, roles, audit_sha) for s, t in sorted(keys)]
    models = {p.parent/a/('candidate%d'%i)/'model.npz' for p in paths for a in ARMS for i in range(4)}
    scores = {p.parent/a/('candidate%d'%i)/(r+'-predictions.jsonl')
              for p in paths for a in ARMS for i in range(4) for r in ('train', 'validation')}
    if set(root.rglob('*.npz')) != models or set(root.rglob('*predictions*.jsonl')) != scores:
        raise ValueError('Complete matrix has unexpected/missing model or prediction artifacts')
    heads = sum(r['head_fits'] for r in runs)
    if heads != config['production_head_fits'] or len(runs) != config['production_runs']:
        raise ValueError('Complete production head/run budget differs')
    record_count = sum(c[role]['pooled']['records'] for r in runs for a in ARMS
                       for c in r['arms'][a]['candidates'] for role in ('fitting', 'validation'))
    if record_count != 64800 or len(scores) != 160:
        raise ValueError('Complete candidate score-record/archive budget differs')
    summary, paired, against_reference = {}, {}, {}
    screen = config['improvement_screen']
    for seed in config['seeds']:
        selected = [r for r in runs if r['seed'] == seed]; ss = {}
        for arm in ARMS:
            values = {role: aggregate_summaries([r['arms'][arm][role] for r in selected])
                      for role in ('fitting', 'validation')}
            values['fitting_minus_validation'] = {k: values['fitting'][k]-values['validation'][k]
                                                 for k in values['validation'] if k.startswith('mean_subset_')}
            ss[arm] = values
        ss['distinct_validation_cases'] = len({cases[i]['case_id']
            for (s, _), (_, val, _) in roles.items() if s == seed for i in val})
        direct, dual = ss['direct']['validation'], ss['dual']['validation']
        paired[str(seed)] = {'delta_mean_subset_macro_f1': dual['mean_subset_macro_f1']-direct['mean_subset_macro_f1'],
                             'delta_mean_subset_accuracy': dual['mean_subset_accuracy']-direct['mean_subset_accuracy']}
        comparisons = {}
        old = reference['summary'][str(seed)]['dual']['validation']
        for arm in ARMS:
            value = ss[arm]['validation']; df = value['mean_subset_macro_f1']-old['mean_subset_macro_f1']
            da = value['mean_subset_accuracy']-old['mean_subset_accuracy']
            positive = all(r > 0 for r in value['pooled_class_recall'])
            comparisons[arm] = {'delta_mean_subset_macro_f1': df, 'delta_mean_subset_accuracy': da,
                'all_five_recall_positive': positive,
                'gate_passed': bool(df >= screen['each_arrangement_macro_f1_delta'] and da >= 0 and positive)}
        against_reference[str(seed)] = comparisons; summary[str(seed)] = ss
    result = {k: context[k] for k in PUBLIC_CONTEXT}
    result.update({k: context[k] for k in ('field_config_sha256', 'base_inputs_config_sha256') if k in context})
    result.update(status='complete exploratory field-supervised five-class development comparison',
        class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=len(runs),
        candidate_pipelines=80, verified_prediction_archives=len(scores), verified_case_subset_records=record_count,
        head_fits=heads, production_head_fits=heads, collector_new_fits=0,
        new_encoder_forwards=0, outer_predictions=0, audit_sha256=audit_sha,
        reference_comparison_sha256=config['reference_comparison_sha256'],
        reference_code_sha=reference['code_sha'], reference_dual_validation={s: reference['summary'][s]['dual']['validation'] for s in summary},
        gate_passed=all(v['dual']['gate_passed'] for v in against_reference.values()),
        gate_scope='descriptive registered screen against frozen case-mean dual on reused provider-case validation; not a fresh test',
        summary=summary, paired=paired, comparison_against_frozen_case_dual=against_reference, runs=runs,
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}),
        selection=config['selection'], C_grid=config['C_grid'], routing_threshold=config['routing_threshold'],
        routing_tie=config['routing_tie'], aggregation=config['aggregation'], interpretation=config['interpretation'])
    assert_public_safe(result)
    output.mkdir(parents=True); write_json(output/'comparison.json', result)
    return result

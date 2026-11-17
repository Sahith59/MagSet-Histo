"""NumPy-only final-state and complete prediction replay for the fixed metric factorial."""
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import specialist_five as runner
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .five_collect import (aggregate_summaries, assert_public_safe, read_json,
                           require_equal, verify_records)
from .five_metrics import FINAL5, MAP8_TO5
from .genbio_collect import comparisons


def _unit(value):
    value = np.asarray(value, dtype=np.float64)
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _all_weights(labels, counts):
    outputs = []
    for head in ('direct', 'binary', 'malignant'):
        values = (labels > 0).astype(int) if head == 'binary' else labels
        classes = range(2) if head == 'binary' else range(1, 5) if head == 'malignant' else range(5)
        cases = np.zeros((len(labels), len(classes)), dtype=np.float64)
        for column, label in enumerate(classes):
            selected = values == label
            cases[selected, column] = 1./(int(selected.sum())*counts[selected])
        outputs.append(np.repeat(cases, counts, axis=0))
    return np.concatenate(outputs, axis=1)


def _density(query, support, labels, counts, classes, temperature, weights=None, normalize=True):
    cosine = np.clip(query @ support.T, -1., 1.)
    scaled = cosine/float(temperature)
    maximum = scaled.max(axis=1, keepdims=True)
    if weights is None:
        weights = []
        for label in classes:
            case = np.where(labels == label, 1./(int((labels == label).sum())*counts), 0.)
            weights.append(np.repeat(case, counts))
        weights = np.stack(weights, axis=1)
    mass = np.exp(scaled-maximum) @ weights
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError('Nonfinite independent density replay')
    logs = np.log(mass)+maximum
    return logs-logsumexp(logs, axis=1, keepdims=True) if normalize else logs


def replay_records(bags, y5, train, indices, cases, role, temperature, factors):
    """Literal field math independent of runner preparation/transformation helpers."""
    if role not in ('train', 'validation') or set(factors)-{'binary', 'malignant'}:
        raise ValueError('Independent replay role/head differs')
    records = []
    for query_index in indices:
        if (query_index in train) != (role == 'train'):
            raise ValueError('Independent replay support/query roles overlap')
        support_indices = np.asarray([i for i in train if i != query_index], dtype=int)
        labels = np.asarray(y5)[support_indices]
        views, old_joint_views = [], []
        for view in range(4):
            center = np.mean([np.asarray(bags[i][view], dtype=np.float64).mean(0) for i in support_indices], axis=0)
            support = np.concatenate([_unit(np.asarray(bags[i][view], dtype=np.float64)-center) for i in support_indices])
            query = _unit(np.asarray(bags[query_index][view], dtype=np.float64)-center)
            counts = np.asarray([len(bags[i][view]) for i in support_indices], dtype=int)
            binary_labels = (labels > 0).astype(int)
            weights = _all_weights(labels, counts)
            # Preserve the accepted joint matrix multiplication order at .5 ties.
            all_logs = _density(query, support, labels, counts, range(11), temperature, weights, normalize=False)
            old_binary = all_logs[:, 5:7]-logsumexp(all_logs[:, 5:7], axis=1, keepdims=True)
            old_malignant = all_logs[:, 7:11]-logsumexp(all_logs[:, 7:11], axis=1, keepdims=True)
            heads = {}
            for head, head_labels, classes in (('binary', binary_labels, range(2)), ('malignant', labels, range(1, 5))):
                value = factors.get(head)
                if value is None or np.array_equal(value, np.ones(support.shape[1])):
                    heads[head] = old_binary if head == 'binary' else old_malignant
                else:
                    columns = slice(5, 7) if head == 'binary' else slice(7, 11)
                    heads[head] = _density(_unit(query*value), _unit(support*value), head_labels, counts, classes, temperature, weights[:, columns])
            binary, malignant = heads['binary'], heads['malignant']
            joint = np.column_stack((binary[:, 0], binary[:, 1, None]+malignant))
            views.append(logsumexp(joint, axis=0)-np.log(len(joint)))
            old_joint = np.column_stack((old_binary[:, 0], old_binary[:, 1, None]+old_malignant))
            old_joint_views.append(logsumexp(old_joint, axis=0)-np.log(len(old_joint)))
        for subset in runner.kernel.SUBSETS:
            log5 = logsumexp(np.asarray(views)[list(subset)], axis=0)-np.log(len(subset))
            malignant_log = logsumexp(log5[1:])
            binary_log = np.asarray([log5[0], malignant_log])
            score = float(np.clip(np.exp(malignant_log), 0, 1))
            if 'binary' not in factors:
                old_log5 = logsumexp(np.asarray(old_joint_views)[list(subset)], axis=0)-np.log(len(subset))
                binary_log = np.asarray([old_log5[0], logsumexp(old_log5[1:])])
                score = float(np.clip(np.exp(binary_log[1]), 0, 1))
            q_log = log5[1:]-malignant_log
            prediction = 0 if score < .5 else 1+int(q_log.argmax())
            original = int(cases[query_index]['label'])
            records.append({'role': role, 'case_id': cases[query_index]['case_id'], 'truth': original,
                'truth5': int(MAP8_TO5[original]), 'magnifications': [MAGNIFICATIONS[v] for v in subset],
                'probabilities5': np.exp(log5).tolist(), 'log_probabilities5': log5.tolist(),
                'prediction': prediction, 'binary_score': score, 'binary_log_probabilities': binary_log.tolist(),
                'malignant_probabilities4': np.exp(q_log).tolist(), 'malignant_log_probabilities': q_log.tolist()})
    return records


def verify_training(folder, head, centroids, labels, temperature, hashes):
    from .specialist_metric import FITTING_DIAGNOSTIC, TRAINING_SPEC, numpy_objective
    path = Path(folder)/(head+'-metric.npz')
    training_path = Path(folder)/(head+'-training.json')
    if file_hash(path) != hashes['metric_sha256'] or file_hash(training_path) != hashes['training_sha256']:
        raise ValueError('Final metric/training artifact bytes differ')
    state = runner.load_factors(path, centroids.shape[-1])
    training = read_json(training_path)
    if set(training) != {'history', 'receipt'} or len(training['history']) != 200:
        raise ValueError('Fixed 200-update history is incomplete')
    for step, row in enumerate(training['history'], 1):
        if set(row) != {'step', 'loss_before_update', 'ce_before_update', 'penalty_before_update', 'gradient_l2'}:
            raise ValueError('Metric history columns differ')
        require_equal(row['step'], step, 'optimizer update sequence', 0)
        for name in ('loss_before_update', 'ce_before_update', 'penalty_before_update', 'gradient_l2'):
            if isinstance(row[name], bool) or not isinstance(row[name], (int, float)) or not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError('Nonfinite/negative metric optimization diagnostic')
        require_equal(row['loss_before_update'], row['ce_before_update']+row['penalty_before_update'], 'history objective sum', 2e-5)
    receipt = training['receipt']
    center = np.asarray(centroids, dtype=np.float64).mean(0)
    expected = {'recipe': TRAINING_SPEC, 'cases': 46, 'head': head,
        'head_cases': 46 if head == 'binary' else 35, 'views': 4, 'dimension': 1536,
        'fitting_counts': [11, 25, 2, 5, 3], 'temperature': temperature,
        'centers_sha256': hashlib.sha256(center.tobytes()).hexdigest(),
        'theta_sha256': hashlib.sha256(state['theta'].tobytes()).hexdigest(),
        'factor_min': float(state['factors'].min()), 'factor_max': float(state['factors'].max()),
        'validation_selection_opportunities': 0, 'deterministic_algorithms': True, 'tf32': False}
    for key, value in expected.items(): require_equal(receipt.get(key), value, 'training receipt/'+key, 0)
    for phase, factors in [('initial', None), ('final', state['factors'])]:
        objective = numpy_objective(centroids, labels, temperature, head=head, factors=factors)
        if phase == 'initial':
            eligible = np.flatnonzero(labels > 0) if head == 'malignant' else np.arange(len(labels))
            vectors = np.asarray(centroids, dtype=np.float64)[eligible]-center
            vectors = _unit(vectors).astype(np.float32)
            head_labels = labels[eligible]-1 if head == 'malignant' else (labels > 0).astype(int)
            require_equal(receipt.get('training_vectors_sha256'), hashlib.sha256(vectors.tobytes()).hexdigest(), 'actual training-vector digest', 0)
            require_equal(receipt.get('head_counts'), np.bincount(head_labels).tolist(), 'head class case counts', 0)
        scalars = {name: float(objective[name]) for name in ('loss', 'ce', 'penalty')}
        require_equal(receipt.get(phase+'_numpy_objective'), scalars, 'independent '+phase+' metric objective', 1e-10)
        require_equal(receipt.get(phase+'_torch_objective'), scalars, 'FP32/NumPy '+phase+' objective', 2e-4)
    initial = receipt['initial_torch_objective']
    require_equal(training['history'][0]['loss_before_update'], initial['loss'], 'initial history objective', 0)
    if receipt.get('device') not in ('cpu', 'cuda', 'cuda:0') or not isinstance(receipt.get('torch_version'), str):
        raise ValueError('Training device/version receipt missing')
    require_equal(receipt.get('factor_storage'), 'actual Torch FP32 log-factors and factors cast to float64', 'canonical factor storage', 0)
    require_equal(receipt.get('fitting_diagnostic'), FITTING_DIAGNOSTIC, 'honest fitting diagnostic', 0)
    return state, receipt


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, reference, audit_sha):
    folder = Path(folder); report = read_json(folder/'run.json')
    for key, value in context.items(): require_equal(report.get(key), value, 'run context/'+key, 0)
    index, temperature, candidate = runner.selected_reference(reference, seed, trial)
    expected = {'status': 'complete specialist metric validation run', 'seed': seed, 'trial': trial,
        'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'role_counts': [46, 8, 28],
        'metric_fits': 2, 'outer_predictions': 0, 'inherited_selected_index': index, 'temperature': temperature,
        'fitting_evaluation': runner.FITTING_POLICY, 'identity_fitting_evaluation': runner.IDENTITY_FITTING_POLICY,
        'audit_sha256': audit_sha}
    for key, value in expected.items(): require_equal(report.get(key), value, 'run/'+key, 0)
    train, val, excluded = roles[seed, trial]
    ids = {name: [cases[i]['case_id'] for i in ix] for name, ix in [('train', train), ('validation', val), ('excluded', excluded)]}
    require_equal(report.get('case_ids'), ids, 'original provider roles', 0)
    if set(report.get('arms', {})) != set(runner.ARMS) or set(report.get('head_artifacts', {})) != {'binary', 'malignant'}:
        raise ValueError('Incomplete factorial/head artifact matrix')
    if file_hash(folder/'model.json') != report.get('model_sha256'):
        raise ValueError('Saved model hash differs')
    model = read_json(folder/'model.json')
    wanted_model = dict(context, schema_version=1, temperature=temperature, inherited_selected_index=index,
        head_artifacts=report['head_artifacts'], fitting_evaluation=runner.FITTING_POLICY,
        identity_fitting_evaluation=runner.IDENTITY_FITTING_POLICY, training_indices=list(map(int, train)),
        support_policy='same accepted fitting cases; whole query omitted for fitting density support',
        metric_changes='separate binary/malignant diagonals after old support-case centering and L2, then reL2',
        binary_policy='identity and malignant_only reuse accepted binary records exactly; other arms use learned binary metric')
    require_equal(model, wanted_model, 'saved model', 0)
    states, receipts = {}, {}
    for head in ('binary', 'malignant'):
        states[head], receipts[head] = verify_training(folder, head, runner.fitting_centroids(bags, train), y5[train], temperature, report['head_artifacts'][head])
    require_equal(report.get('training_receipts'), receipts, 'embedded training receipts', 0)
    result = {'seed': seed, 'trial': trial, 'temperature': temperature, 'inherited_selected_index': index,
              'metric_fits': 2, 'outer_predictions': 0, 'run_sha256': file_hash(folder/'run.json'),
              'model_sha256': report['model_sha256'], 'head_artifacts': report['head_artifacts'],
              'training_receipts': receipts, 'arms': {}}
    identity_records = {}
    for arm in runner.ARMS:
        saved = report['arms'][arm]
        names = () if arm == 'identity' else ('binary', 'malignant') if arm == 'both' else (arm.removesuffix('_only'),)
        factors = {head: states[head]['factors'] for head in names}
        summaries, hashes = {}, {}
        for role, indices, summary_name in [('train', train, 'fitting'), ('validation', val, 'validation')]:
            path = folder/arm/(role+'-predictions.jsonl')
            if file_hash(path) != saved.get(role+'_predictions_sha256'):
                raise ValueError('Saved factorial prediction bytes differ')
            wanted = replay_records(bags, y5, train, indices, cases, role, temperature, factors)
            summary = verify_records(path, wanted, cases, indices, role, 'dual')
            require_equal(saved.get(summary_name), summary, 'factorial '+arm+'/'+summary_name)
            actual = [read_json_line(line) for line in path.read_text().splitlines()]
            if arm == 'identity': identity_records[role] = actual
            if arm == 'malignant_only':
                old = {(r['case_id'], tuple(r['magnifications'])): r for r in identity_records[role]}
                for row in actual:
                    source = old[row['case_id'], tuple(row['magnifications'])]
                    for key in ('binary_score', 'binary_log_probabilities'):
                        require_equal(row[key], source[key], 'exact malignant-only binary preservation', 0)
                    if (row['prediction'] == 0) != (source['prediction'] == 0):
                        raise ValueError('Malignant-only intervention changed a binary route')
            summaries[summary_name] = summary; hashes[role+'_predictions_sha256'] = file_hash(path)
        result['arms'][arm] = dict(summaries, **hashes)
    require_equal(result['arms']['identity']['validation'], candidate['validation'], 'accepted selected identity result', 1e-10)
    elapsed = report.get('elapsed_seconds')
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid elapsed-time receipt')
    result['elapsed_seconds'] = elapsed
    return result


def read_json_line(line):
    from .five_collect import _unique_object, _invalid_constant
    import json
    return json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def collect(args):
    if args.runs is None or args.audit is None:
        raise ValueError('Specialist collector requires complete runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing specialist comparison')
    config, context, _, cases, bags, z, y5, roles, reference = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    runner.verify_input_audit(audit_path, context, config, bags, y5, roles, reference)
    root = Path(args.runs)
    paths = {root/('split%d'%seed)/('trial%d'%trial)/'run.json' for seed, trial in roles}
    models = {path.parent/'model.json' for path in paths}
    arrays = {path.parent/(head+'-metric.npz') for path in paths for head in ('binary', 'malignant')}
    histories = {path.parent/(head+'-training.json') for path in paths for head in ('binary', 'malignant')}
    scores = {path.parent/arm/(role+'-predictions.jsonl') for path in paths for arm in runner.ARMS for role in ('train', 'validation')}
    if (set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != models
            or set(root.rglob('*.npz')) != arrays or set(root.rglob('*-training.json')) != histories
            or set(root.rglob('*predictions*.jsonl')) != scores):
        raise ValueError('Incomplete or extra specialist artifact/role/arm matrix')
    runs = [verify_run(root/('split%d'%seed)/('trial%d'%trial), seed, trial, config, context,
                       cases, bags, z, y5, roles, reference, file_hash(audit_path)) for seed, trial in sorted(roles)]
    records = sum(run['arms'][arm][role]['pooled']['records'] for run in runs for arm in runner.ARMS for role in ('fitting', 'validation'))
    if len(runs) != 10 or records != 32400 or sum(run['metric_fits'] for run in runs) != 20:
        raise ValueError('Specialist fit/record budget differs')
    summary, paired = {}, {}
    for seed in config['seeds']:
        values = {arm: {role: aggregate_summaries([run['arms'][arm][role] for run in runs if run['seed'] == seed])
                       for role in ('fitting', 'validation')} for arm in runner.ARMS}
        summary[str(seed)] = values
        paired[str(seed)] = {arm: comparisons(values[arm]['validation'], values['identity']['validation'], config['improvement_screen'])
                             for arm in runner.ARMS if arm != 'identity'}
    result = dict(context, status='complete exploratory frozen UNI2-h head-specific metric factorial',
        class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10,
        metric_fits=20, collector_metric_fits=0, candidate_pipelines=40, verified_prediction_archives=80,
        verified_case_subset_records=records, new_encoder_forwards=0, outer_predictions=0,
        arms=list(runner.ARMS), primary_arm='both', primary_comparison='both versus identity on each validation arrangement',
        fitting_evaluation=runner.FITTING_POLICY, identity_fitting_evaluation=runner.IDENTITY_FITTING_POLICY,
        inherited_temperatures=config['inherited_temperatures'], validation_selection_opportunities=0,
        audit_sha256=file_hash(audit_path), reference_comparison_sha256=config['reference_comparison_sha256'],
        family_scope='four prespecified arms; no new validation setting selection; adaptive arm choice remains exploratory',
        summary=summary, paired_against_identity=paired,
        gate_passed=all(paired[str(seed)]['both']['gate_passed'] for seed in config['seeds']),
        interpretation=config['interpretation'], distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}),
        runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.write_json(output/'comparison.json', result)
    return result

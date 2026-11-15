"""Verify the complete registered five-class matrix without fitting or prediction on outer roles."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .bags import fit_normalization
from .data import CLASS_NAMES, file_hash
from .five_metrics import FINAL5, MAP8_TO5, MEAN_METRICS, summarize_records


PUBLIC_CONTEXT = ('code_sha', 'source_archive_sha256', 'implementation_hashes',
                  'config_sha256', 'manifest_sha256', 'protocol_sha256', 'unit',
                  'dependencies', 'python', 'encoder', 'cache_provenance')


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: '+key)
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def require_equal(actual, expected, name, tolerance=2e-12):
    """Compare full nested artifacts, with bounded floating replay tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(name+': object keys differ')
        for key in expected:
            require_equal(actual[key], expected[key], name+'/'+key, tolerance)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(name+': sequence differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            require_equal(left, right, name+'/'+str(i), tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(name+': value differs')
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float, np.integer, np.floating))
                or not math.isfinite(float(actual)) or not math.isfinite(float(expected))
                or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance)):
            raise ValueError(name+': numerical value differs')
    elif actual != expected:
        raise ValueError(name+': unsupported value differs')


def verify_audit(audit, context, config, bags, y5, roles):
    """Pure validation of the accepted zero-classifier-fit audit and all ten normalizers."""
    if (audit.get('status') != 'zero-fit inputs accepted'
            or any(type(audit.get(k)) is not int or audit[k] != 0
                   for k in ('new_fits', 'new_forwards', 'outer_predictions'))):
        raise ValueError('Accepted audit status/zero-work counters differ')
    for key, value in context.items():
        require_equal(audit.get(key), value, 'audit/context/'+key, 0)
    expected_keys = {(seed, trial) for seed in config['seeds'] for trial in config['trials']}
    if set(roles) != expected_keys or len(expected_keys) != config['production_runs']:
        raise ValueError('Accepted audit role matrix differs')
    checks = audit.get('checks')
    if not isinstance(checks, list) or len(checks) != len(expected_keys):
        raise ValueError('Accepted audit is incomplete')
    indexed = {}
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError('Accepted audit check is invalid')
        key = (check.get('seed'), check.get('trial'))
        if key not in expected_keys or key in indexed:
            raise ValueError('Accepted audit has unexpected/duplicate run')
        indexed[key] = check
    for key in sorted(expected_keys):
        train, val, excluded = roles[key]
        mean, std = fit_normalization(bags, train)
        expected = {'seed': key[0], 'trial': key[1],
                    'role_counts': [len(train), len(val), len(excluded)],
                    'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
                    'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                    'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()}
        require_equal(indexed[key], expected, 'audit/check', 0)
        if (expected['role_counts'] != config['expected_role_counts']
                or expected['fitting_support'] != config['fitting_five_support']
                or expected['validation_support'] != config['validation_five_support']):
            raise ValueError('Audit role counts/support differ from registration')


def _record_key(record):
    return record['case_id'], tuple(record['magnifications'])


def verify_records(path, expected, cases, indices, role, arm):
    """Check trusted original labels/roles and replay every explicit decision and score."""
    labels = {cases[i]['case_id']: int(cases[i]['label']) for i in indices}
    records = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
               for line in Path(path).read_text().splitlines()]
    wanted = {_record_key(row): row for row in expected}
    actual = {}
    for row in records:
        if (not isinstance(row, dict) or row.get('role') != role
                or not isinstance(row.get('case_id'), str)
                or row.get('case_id') not in labels
                or type(row.get('truth')) is not int or row['truth'] != labels[row['case_id']]
                or type(row.get('truth5')) is not int or row['truth5'] != MAP8_TO5[row['truth']]):
            raise ValueError('Saved prediction differs from trusted role/original truth')
        try:
            key = _record_key(row)
        except (KeyError, TypeError) as error:
            raise ValueError('Saved prediction subset differs') from error
        if key in actual or key not in wanted:
            raise ValueError('Saved predictions duplicate or unexpected case/subset')
        require_equal(row, wanted[key], 'coefficient replay', 1e-12)
        actual[key] = row
    if set(actual) != set(wanted) or len(records) != len(indices)*15:
        raise ValueError('Saved predictions lack complete trusted fifteen-subset coverage')
    return summarize_records(records, arm)


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha256):
    """Verify one complete two-arm/four-candidate run and return public-safe fields."""
    from .five_class import ARMS, load_model, score_records
    folder = Path(folder)
    report = read_json(folder/'run.json')
    for key, value in context.items():
        require_equal(report.get(key), value, 'run/context/'+key, 0)
    if (report.get('status') != 'complete five-class validation run'
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
            summaries = {}
            for role, indices, summary_name in (('train', train, 'fitting'), ('validation', val, 'validation')):
                path = directory/(role+'-predictions.jsonl'); expected_scores.add(path)
                if file_hash(path) != candidate.get(role+'_predictions_sha256'):
                    raise ValueError('Saved predictions hash differs')
                expected, _ = score_records(arrays, z, indices, cases, role, arm)
                summary = verify_records(path, expected, cases, indices, role, arm)
                require_equal(candidate.get(summary_name), summary, 'candidate/'+summary_name)
                summaries[summary_name] = summary
            public_candidates.append({key: candidate[key] for key in
                ('C', 'index', 'head_fits', 'n_iter', 'sklearn_replay_max_abs', 'elapsed_seconds',
                 'model_sha256', 'train_predictions_sha256', 'validation_predictions_sha256')})
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


def aggregate_summaries(values):
    metrics = tuple('mean_subset_'+key for key in MEAN_METRICS)
    result = {key: float(np.mean([value[key] for value in values])) for key in metrics}
    result['minimum_trial_worst_subset_macro_f1'] = float(min(value['worst_subset_macro_f1'] for value in values))
    result['mean_trial_worst_subset_macro_f1'] = float(np.mean([value['worst_subset_macro_f1'] for value in values]))
    result['all_view'] = {key: float(np.mean([value['all_view'][key] for value in values])) for key in MEAN_METRICS}
    matrix = np.sum([value['pooled']['confusion_matrix'] for value in values], axis=0)
    support = matrix.sum(axis=1)
    result.update(pooled_confusion_matrix=matrix.tolist(), pooled_class_support=support.tolist(),
                  pooled_class_recall=np.divide(matrix.diagonal(), support,
                      out=np.zeros(5), where=support > 0).tolist(),
                  pooled_case_subset_records=sum(value['pooled']['records'] for value in values))
    binary_keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'sensitivity', 'specificity',
                   'roc_auc', 'pr_average_precision', 'positive_class_brier', 'cross_entropy')
    result['binary_mean_metrics'] = {key: float(np.mean([
        value['binary_summary']['mean_all_subsets'][key] for value in values])) for key in binary_keys}
    result['malignant_mean_metrics'] = {key: float(np.mean([
        value['malignant_summary'][key] for value in values])) for key in metrics}
    return result


def assert_public_safe(value):
    """Fail closed if an aggregate accidentally retains IDs or machine paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'case_id', 'case_ids', 'relative_path', 'filename', 'slurm_job_id'}:
                raise ValueError('Private field in public comparison')
            assert_public_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_public_safe(item)
    elif isinstance(value, str):
        if (re.search(r'SOB_[BM]_|\b\d{2}-\d{3,}[A-Za-z]*\b', value)
                or value.startswith('/') or '\\' in value):
            raise ValueError('Private identifier/path in public comparison')


def collect(args):
    """Accept only the full hash-bound 10-run/120-head matrix, then publish aggregates."""
    from .five_class import ARMS, load_inputs, write_json
    if args.runs is None or args.audit is None:
        raise ValueError('Collector requires --runs and accepted --audit')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing comparison output')
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(
        args.manifest, args.protocol, args.cache, args.config, args.code_sha)
    audit_path = Path(args.audit)/'audit.json'; audit = read_json(audit_path)
    verify_audit(audit, context, config, bags, y5, roles)
    audit_sha = file_hash(audit_path)
    root = Path(args.runs)
    expected = {(seed, trial) for seed in config['seeds'] for trial in config['trials']}
    expected_paths = {root/('split%d' % seed)/('trial%d' % trial)/'run.json' for seed, trial in expected}
    if set(root.rglob('run.json')) != expected_paths:
        raise ValueError('Complete registered run matrix is missing or has extra runs')
    runs = [verify_run(root/('split%d' % seed)/('trial%d' % trial), seed, trial,
                       config, context, cases, bags, z, y5, roles, audit_sha)
            for seed, trial in sorted(expected)]
    expected_models = {path.parent/arm/('candidate%d' % index)/'model.npz'
                       for path in expected_paths for arm in ARMS for index in range(len(config['C_grid']))}
    expected_scores = {path.parent/arm/('candidate%d' % index)/(role+'-predictions.jsonl')
                       for path in expected_paths for arm in ARMS for index in range(len(config['C_grid']))
                       for role in ('train', 'validation')}
    if (set(root.rglob('*.npz')) != expected_models
            or set(root.rglob('*predictions*.jsonl')) != expected_scores):
        raise ValueError('Complete matrix has unexpected/missing model or prediction artifacts')
    head_fits = sum(run['head_fits'] for run in runs)
    if head_fits != config['production_head_fits'] or len(runs) != config['production_runs']:
        raise ValueError('Complete production head/run budget differs')
    summary = {}; paired = {}
    for seed in config['seeds']:
        seed_runs = [run for run in runs if run['seed'] == seed]
        seed_summary = {}
        for arm in ARMS:
            values = {role: aggregate_summaries([run['arms'][arm][role] for run in seed_runs])
                      for role in ('fitting', 'validation')}
            values['fitting_minus_validation'] = {key: values['fitting'][key]-values['validation'][key]
                                                 for key in values['validation'] if key.startswith('mean_subset_')}
            seed_summary[arm] = values
        seed_summary['distinct_validation_cases'] = len({cases[i]['case_id']
            for (s, _), (_, val, _) in roles.items() if s == seed for i in val})
        direct = seed_summary['direct']['validation']; dual = seed_summary['dual']['validation']
        delta_f1 = dual['mean_subset_macro_f1']-direct['mean_subset_macro_f1']
        delta_accuracy = dual['mean_subset_accuracy']-direct['mean_subset_accuracy']
        positive = all(value > 0 for value in dual['pooled_class_recall'])
        paired[str(seed)] = {'delta_mean_subset_macro_f1': delta_f1,
                             'delta_mean_subset_accuracy': delta_accuracy,
                             'all_five_recall_positive': positive,
                             'gate_passed': bool(delta_f1 >= config['gate_margin'] and delta_accuracy >= 0 and positive)}
        summary[str(seed)] = seed_summary
    result = {key: context[key] for key in PUBLIC_CONTEXT}
    result.update(status='complete exploratory five-class development comparison',
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
                  verified_runs=len(runs), head_fits=head_fits, production_head_fits=head_fits,
                  collector_new_fits=0, new_encoder_forwards=0,
                  outer_predictions=0, audit_sha256=audit_sha,
                  gate_passed=all(value['gate_passed'] for value in paired.values()),
                  gate_scope='descriptive screen on reused provider-case validation; not a fresh test',
                  summary=summary, paired=paired, runs=runs,
                  distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}),
                  selection='earliest maximum end-to-end hard-label mean15-subset macro F1',
                  C_grid=config['C_grid'], routing_threshold=config['routing_threshold'],
                  routing_tie=config['routing_tie'])
    assert_public_safe(result)
    output.mkdir(parents=True)
    write_json(output/'comparison.json', result)
    return result

"""Complete UNI2-h field linear matrix with reviewed native coefficient replay."""
from pathlib import Path

import numpy as np

from . import uni_linear_five as runner, field_collect as replay
from .bags import fit_normalization
from .data import CLASS_NAMES, file_hash
from .five_class import load_model
from .five_collect import (aggregate_summaries, assert_public_safe, read_json,
                           require_equal, verify_audit, verify_records)
from .five_metrics import FINAL5
from .genbio_collect import comparisons


def _check_convergence(candidate, config):
    for values in candidate['n_iter'].values():
        if len(values) != 1 or type(values[0]) is not int or not 1 <= values[0] < config['max_iter']:
            raise ValueError('UNI linear solver convergence/iteration receipt differs')
    for value in candidate['head_objectives'].values():
        if value['normalized_max_abs_gradient'] > config['tol']+1e-8:
            raise ValueError('UNI linear independent fitted gradient exceeds registered tolerance')


def verify_smoke(args):
    """Fit nothing; reconstruct all 12 head objectives and 600 fitting-only scores."""
    if args.audit is None or args.smokes is None:
        raise ValueError('Smoke review requires accepted inputs and saved fitting-only smoke')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing UNI linear smoke acceptance')
    config, context, _, cases, bags, _, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    root = Path(args.smokes); report = read_json(root/'run.json')
    for key, value in context.items(): require_equal(report.get(key), value, 'smoke context/'+key, 0)
    expected = {'status': 'complete UNI2-h field fitting-only technical smoke', 'seed': 812, 'trial': 1,
        'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'head_fits': 12,
        'outer_predictions': 0, 'audit_sha256': file_hash(audit_path), 'fitting_evaluation': runner.FITTING_POLICY}
    for key, value in expected.items(): require_equal(report.get(key), value, 'smoke/'+key, 0)
    train, val, excluded = roles[812, 1]
    require_equal(report.get('case_ids'), {name: [cases[i]['case_id'] for i in indices] for name, indices in
                  [('train', train), ('validation', val), ('excluded', excluded)]}, 'smoke original roles', 0)
    require_equal(report.get('role_counts'), [46, 8, 28], 'smoke role counts', 0)
    queries = []
    for label in range(5):
        eligible = [int(i) for i in train if int(y5[i]) == label]
        if not eligible: raise ValueError('Smoke lacks fitting support for a final label')
        queries.append(eligible[0])
    mean, std = fit_normalization(bags, train)
    models, scores, metric_paths = set(), set(), set(); heads = records = 0
    if set(report.get('arms', {})) != set(runner.ARMS): raise ValueError('Incomplete smoke arm matrix')
    for arm in runner.ARMS:
        candidates = report['arms'][arm].get('candidates')
        if not isinstance(candidates, list) or len(candidates) != 4 or set(report['arms'][arm]) != {'candidates'}:
            raise ValueError('Smoke must report all four Cs without selection')
        names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
        rows = {head: replay.training_rows(bags, train, y5, mean, std, head) for head in names}
        receipts = replay.field_receipts(rows)
        for index, C in enumerate(config['C_grid']):
            folder = root/arm/('candidate%d'%index)
            path = folder/'metrics.json'; metric_paths.add(path); candidate = read_json(path)
            require_equal(candidate, candidates[index], 'embedded smoke candidate', 0)
            for key, value in {'C': C, 'index': index, 'head_fits': len(names)}.items():
                require_equal(candidate.get(key), value, 'smoke candidate/'+key, 0)
            if 'validation' in candidate or 'validation_predictions_sha256' in candidate:
                raise ValueError('Technical smoke may not include validation results')
            path = folder/'model.npz'; models.add(path)
            if file_hash(path) != candidate.get('model_sha256'): raise ValueError('Smoke model bytes differ')
            arrays = load_model(path, arm, dimension=config['encoder_dimension'])
            if not np.array_equal(arrays['mean'], mean) or not np.array_equal(arrays['std'], std):
                raise ValueError('Smoke normalizer differs from fitting-only field moments')
            require_equal(candidate.get('field_receipts'), receipts, 'smoke head mass', 1e-10)
            objectives = replay.head_objectives(arrays, rows, C, config)
            require_equal(candidate.get('head_objectives'), objectives, 'independent smoke head objective', 1e-8)
            if set(candidate.get('n_iter', {})) != set(names): raise ValueError('Smoke head iteration matrix differs')
            _check_convergence(candidate, config)
            error = candidate.get('sklearn_replay_max_abs')
            if isinstance(error, bool) or not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 1e-12:
                raise ValueError('Smoke sklearn coefficient replay exceeds tolerance')
            path = folder/'train-predictions.jsonl'; scores.add(path)
            if file_hash(path) != candidate.get('train_predictions_sha256'): raise ValueError('Smoke fitting prediction bytes differ')
            wanted = replay.replay_records(arrays, bags, queries, cases, 'train', arm)
            summary = verify_records(path, wanted, cases, queries, 'train', arm)
            require_equal(candidate.get('fitting'), summary, 'smoke fitting summary')
            heads += len(names); records += summary['pooled']['records']
    if (set(root.rglob('*.npz')) != models or set(root.rglob('*predictions*.jsonl')) != scores
            or set(root.rglob('metrics.json')) != metric_paths or set(root.rglob('run.json')) != {root/'run.json'}
            or heads != 12 or records != 600):
        raise ValueError('Smoke complete fitting-only artifact/head/record matrix differs')
    receipt = dict(context, status='accepted UNI2-h field logistic fitting-only smoke', head_fits=12,
        verified_prediction_archives=8, verified_case_subset_records=600, validation_records=0,
        collector_new_fits=0, audit_sha256=file_hash(audit_path), smoke_run_sha256=file_hash(root/'run.json'),
        technical_scope='all four C settings; full fitting head support; first fitting query per final class; no validation selection')
    assert_public_safe(receipt); output.mkdir(parents=True); runner.write_json(output/'acceptance.json', receipt)
    return receipt


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha, smoke_sha):
    report = read_json(Path(folder)/'run.json')
    require_equal(report.get('fitting_evaluation'), runner.FITTING_POLICY, 'plain in-sample fitting diagnostic', 0)
    require_equal(report.get('smoke_acceptance_sha256'), smoke_sha, 'accepted technical smoke', 0)
    result = replay.verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha)
    for arm in runner.ARMS:
        for candidate in result['arms'][arm]['candidates']: _check_convergence(candidate, config)
    result['fitting_evaluation'] = runner.FITTING_POLICY
    return result


def load_reference(config, context, roles, secondary=False):
    prefix = 'secondary_' if secondary else ''
    path = Path(__file__).resolve().parents[1]/config[prefix+'reference_file']
    if file_hash(path) != config[prefix+'reference_comparison_sha256']:
        raise ValueError('Accepted UNI linear comparison reference bytes differ')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    if reference.get('class_names') != list(FINAL5) or reference.get('outer_predictions') != 0:
        raise ValueError('Reference five-category scope differs')
    if not secondary:
        for key in ('encoder', 'cache_provenance', 'feature_audit_sha256'):
            require_equal(reference.get(key), context[key], 'same-UNI reference/'+key, 0)
        if (len(reference.get('runs', [])) != 10
                or {(r['seed'], r['trial']) for r in reference['runs']} != set(roles)):
            raise ValueError('UNI KDE reference role matrix differs')
    return reference


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('Collector needs complete runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing UNI linear comparison')
    config, context, _, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'; verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    smoke_sha = runner.verify_smoke_acceptance(args, config, context, file_hash(audit_path))
    reference = load_reference(config, context, roles); retained = load_reference(config, context, roles, True)
    root = Path(args.runs); paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    models = {p.parent/arm/('candidate%d'%i)/'model.npz' for p in paths for arm in runner.ARMS for i in range(4)}
    scores = {p.parent/arm/('candidate%d'%i)/(role+'-predictions.jsonl') for p in paths for arm in runner.ARMS for i in range(4) for role in ('train', 'validation')}
    metrics = {p.parent/arm/('candidate%d'%i)/'metrics.json' for p in paths for arm in runner.ARMS for i in range(4)}
    if (set(root.rglob('run.json')) != paths or set(root.rglob('*.npz')) != models
            or set(root.rglob('*predictions*.jsonl')) != scores or set(root.rglob('metrics.json')) != metrics):
        raise ValueError('Complete UNI field linear artifact matrix differs')
    runs = [verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context, cases, bags, z, y5, roles,
                       file_hash(audit_path), smoke_sha) for s, t in sorted(roles)]
    heads = sum(r['head_fits'] for r in runs)
    records = sum(c[role]['pooled']['records'] for r in runs for arm in runner.ARMS for c in r['arms'][arm]['candidates'] for role in ('fitting', 'validation'))
    if len(runs) != 10 or heads != 120 or len(scores) != 160 or records != 64800:
        raise ValueError('UNI linear production/candidate/head/record budget differs')
    summary, paired, vs_uni, vs_retained = {}, {}, {}, {}
    for seed in config['seeds']:
        values = {arm: {role: aggregate_summaries([r['arms'][arm][role] for r in runs if r['seed'] == seed])
                       for role in ('fitting', 'validation')} for arm in runner.ARMS}
        summary[str(seed)] = values
        paired[str(seed)] = {key: values['dual']['validation'][key]-values['direct']['validation'][key]
                            for key in values['dual']['validation'] if key.startswith('mean_subset_')}
        uni = reference['summary'][str(seed)]['dual']['validation']
        old = retained['summary']['head_specific'][str(seed)]['dual']['validation']
        vs_uni[str(seed)] = {arm: comparisons(values[arm]['validation'], uni, config['improvement_screen']) for arm in runner.ARMS}
        vs_retained[str(seed)] = {arm: comparisons(values[arm]['validation'], old, config['improvement_screen']) for arm in runner.ARMS}
    result = dict(context, status='complete exploratory UNI2-h field logistic comparison', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), verified_runs=10, candidate_pipelines=80, head_fits=120,
        production_head_fits=120, collector_new_fits=0, new_encoder_forwards=0, outer_predictions=0,
        verified_prediction_archives=160, verified_case_subset_records=64800,
        fitting_evaluation=runner.FITTING_POLICY, audit_sha256=file_hash(audit_path), smoke_acceptance_sha256=smoke_sha,
        reference_comparison_sha256=config['reference_comparison_sha256'], secondary_reference_comparison_sha256=config['secondary_reference_comparison_sha256'],
        summary=summary, paired_direct_vs_dual=paired, comparison_against_same_UNI_KDE_dual=vs_uni,
        comparison_against_retained_P_dual=vs_retained,
        gate_passed=all(vs_uni[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        retained_gate_passed=all(vs_retained[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        C_grid=config['C_grid'], selection=config['selection'], interpretation=config['interpretation'],
        family_selection_scope='four C settings per arm/role; heads share C; reader was adaptively chosen after prior results; exploratory reused validation',
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.write_json(output/'comparison.json', result)
    return result

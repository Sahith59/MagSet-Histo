"""Complete-matrix provenance, support and prediction replay for Midnight KDE."""
import math
from pathlib import Path

from . import kernel_five as runner
from .data import CLASS_NAMES, file_hash
from .five_collect import (aggregate_summaries, assert_public_safe, read_json,
                           require_equal, verify_audit, verify_records)
from .five_metrics import FINAL5


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha):
    from .kernel_model import support_summary
    folder = Path(folder); report = read_json(folder/'run.json')
    for key, value in context.items(): require_equal(report.get(key), value, 'run context/'+key, 0)
    expected = {'status': 'complete centered-cosine KDE validation run', 'seed': seed, 'trial': trial,
                'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                'role_counts': [46, 8, 28], 'optimized_head_fits': 0, 'support_classifier_instances': 3,
                'outer_predictions': 0, 'fitting_evaluation': runner.FITTING_POLICY, 'fitting_uses_support_labels': True,
                'support_instances_scope': config['support_instances_scope'], 'audit_sha256': audit_sha}
    for key, value in expected.items(): require_equal(report.get(key), value, 'run/'+key, 0)
    train, val, excluded = roles[seed, trial]
    ids = {r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]}
    require_equal(report.get('case_ids'), ids, 'original role membership', 0)
    if set(report.get('arms', {})) != set(runner.ARMS): raise ValueError('Incomplete arm matrix')
    prepared = runner.prepare_queries(bags, y5, train, val); support = support_summary(bags, y5, train)
    result = {'seed': seed, 'trial': trial, 'run_sha256': file_hash(folder/'run.json'),
              'optimized_head_fits': 0, 'support_classifier_instances': 3, 'arms': {}}
    models, scores = set(), set()
    for arm in runner.ARMS:
        saved = report['arms'][arm]; candidates = saved.get('candidates')
        if not isinstance(candidates, list) or len(candidates) != 4: raise ValueError('Incomplete temperature matrix')
        accepted = []
        for index, temperature in enumerate(runner.TEMPERATURES):
            directory = folder/arm/('candidate%d'%index); candidate = read_json(directory/'metrics.json')
            require_equal(candidates[index], candidate, 'embedded candidate', 0)
            for key, value in {'index': index, 'temperature': temperature, 'optimized_head_fits': 0,
                               'support_heads': ['direct'] if arm == 'direct' else ['binary', 'malignant'],
                               'fitting_evaluation': runner.FITTING_POLICY}.items():
                require_equal(candidate.get(key), value, 'candidate/'+key, 0)
            elapsed = candidate.get('elapsed_seconds')
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid candidate elapsed time')
            path = directory/'model.json'; models.add(path)
            if file_hash(path) != candidate.get('model_sha256'): raise ValueError('Support model hash differs')
            require_equal(read_json(path), runner.model_record(context, cases, train, support, arm, temperature), 'support model', 0)
            for role, indices, key in [('train', train, 'fitting_leave_case_out'), ('validation', val, 'validation')]:
                path = directory/(role+'-predictions.jsonl'); scores.add(path)
                if file_hash(path) != candidate.get(role+'_predictions_sha256'): raise ValueError('Prediction hash differs')
                wanted = runner.score_records(prepared[role], cases, role, arm, temperature)
                summary = verify_records(path, wanted, cases, indices, role, arm)
                require_equal(candidate.get(key), summary, 'candidate/'+key)
            accepted.append(dict(candidate, metrics_sha256=file_hash(directory/'metrics.json')))
        selected = max(range(4), key=lambda i: accepted[i]['validation']['mean_subset_macro_f1'])
        require_equal(saved.get('selected_index'), selected, 'earliest temperature selection', 0)
        require_equal(saved.get('selected_temperature'), runner.TEMPERATURES[selected], 'selected temperature', 0)
        for key in ('fitting_leave_case_out', 'validation'):
            require_equal(saved.get(key), accepted[selected][key], 'selected/'+key)
        result['arms'][arm] = {'selected_index': selected, 'selected_temperature': runner.TEMPERATURES[selected],
                              'candidates': accepted, 'fitting_leave_case_out': accepted[selected]['fitting_leave_case_out'],
                              'validation': accepted[selected]['validation']}
    if set(folder.rglob('model.json')) != models or set(folder.rglob('*predictions*.jsonl')) != scores or list(folder.rglob('*.npz')):
        raise ValueError('Unexpected support or prediction artifacts')
    return result


def reference_comparison(config, context, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'docs/evidence/BREAKHIS_MIDNIGHT_RESULTS.json'
    if file_hash(path) != config['reference_comparison_sha256']: raise ValueError('Accepted Midnight comparison hash differs')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance', 'config_sha256'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    for key, value in {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                       'verified_runs': 20, 'head_fits': 240, 'readouts': ['case', 'field'], 'outer_predictions': 0}.items():
        require_equal(reference.get(key), value, 'reference/'+key, 0)
    case_runs = [r for r in reference.get('runs', []) if r.get('readout') == 'case']
    wanted = {(s, t) for s in config['seeds'] for t in config['trials']}
    if len(case_runs) != 10 or {(r['seed'], r['trial']) for r in case_runs} != wanted: raise ValueError('Reference roles differ')
    for run in case_runs:
        for arm in runner.ARMS:
            for role, support in [('fitting', config['fitting_five_support']), ('validation', config['validation_five_support'])]:
                require_equal(run['arms'][arm][role]['distinct_class_support'], support, 'reference support', 0)
    return reference


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('Collector requires runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing comparison output')
    config, context, trusted, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'; verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    reference = reference_comparison(config, context, getattr(args, 'reference', None))
    root = Path(args.runs); audit_sha = file_hash(audit_path)
    paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    models = {p.parent/a/('candidate%d'%i)/'model.json' for p in paths for a in runner.ARMS for i in range(4)}
    scores = {p.parent/a/('candidate%d'%i)/(r+'-predictions.jsonl') for p in paths for a in runner.ARMS for i in range(4) for r in ('train', 'validation')}
    if set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != models or set(root.rglob('*predictions*.jsonl')) != scores or list(root.rglob('*.npz')):
        raise ValueError('Complete kernel artifact matrix differs')
    runs = [verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context, cases, bags, z, y5, roles, audit_sha) for s, t in sorted(roles)]
    records = sum(c[role]['pooled']['records'] for r in runs for a in runner.ARMS for c in r['arms'][a]['candidates'] for role in ('fitting_leave_case_out', 'validation'))
    if len(runs) != 10 or records != 64800 or sum(r['support_classifier_instances'] for r in runs) != 30 or any(r['optimized_head_fits'] != 0 for r in runs):
        raise ValueError('Kernel support/candidate budget differs')
    summary, paired, comparisons = {}, {}, {}
    for seed in config['seeds']:
        values = {}
        for arm in runner.ARMS:
            values[arm] = {role: aggregate_summaries([r['arms'][arm][role] for r in runs if r['seed'] == seed]) for role in ('fitting_leave_case_out', 'validation')}
        direct, dual = (values[a]['validation'] for a in runner.ARMS)
        paired[str(seed)] = {k: dual[k]-direct[k] for k in ('mean_subset_macro_f1', 'mean_subset_accuracy')}
        old = reference['summary']['case'][str(seed)]['dual']['validation']; comparison = {}
        for arm in runner.ARMS:
            value = values[arm]['validation']; df = value['mean_subset_macro_f1']-old['mean_subset_macro_f1']; da = value['mean_subset_accuracy']-old['mean_subset_accuracy']
            positive = all(x > 0 for x in value['pooled_class_recall'])
            comparison[arm] = {'delta_mean_subset_macro_f1': df, 'delta_mean_subset_accuracy': da, 'all_five_recall_positive': positive,
                               'gate_passed': bool(df >= config['improvement_screen']['each_arrangement_macro_f1_delta'] and da >= 0 and positive)}
        summary[str(seed)] = values; comparisons[str(seed)] = comparison
    result = dict(context, status='complete exploratory centered-cosine KDE development comparison',
        class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10, candidate_pipelines=80,
        support_classifier_instances=30, optimized_head_fits=0, collector_optimized_head_fits=0,
        fitted_coefficients=0, fitting_uses_support_labels=True, support_instances_scope=config['support_instances_scope'],
        new_encoder_forwards=0, outer_predictions=0,
        verified_prediction_archives=160, verified_case_subset_records=records, fitting_evaluation=runner.FITTING_POLICY,
        temperatures=runner.TEMPERATURES, normalization=runner.NORMALIZATION, audit_sha256=audit_sha,
        reference_comparison_sha256=config['reference_comparison_sha256'], reference_code_sha=reference['code_sha'],
        reference_readout='case', reference_arm='dual', summary=summary, paired_direct_vs_dual=paired,
        comparison_against_midnight_case_dual=comparisons, gate_passed=all(x['dual']['gate_passed'] for x in comparisons.values()),
        gate_scope='descriptive registered screen on reused provider-case validation; both arrangements required; not a fresh test', runs=runs,
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}))
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

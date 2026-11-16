"""Full native-input matrix replay with primary three-crop and secondary center controls."""
from pathlib import Path

from . import native_five as runner, kernel_collect as replay
from .data import CLASS_NAMES, file_hash
from .five_collect import aggregate_summaries, assert_public_safe, read_json, require_equal, verify_audit
from .five_metrics import FINAL5

METHOD_FILES = ('data.py', 'baseline.py', 'five_metrics.py', 'bags.py', 'kernel_model.py', 'field_five.py')


def reference_comparison(config, context, roles, secondary=False):
    root = Path(__file__).resolve().parents[1]
    path = root/config['secondary_reference_file' if secondary else 'reference_file']
    digest = config['secondary_reference_comparison_sha256' if secondary else 'reference_comparison_sha256']
    if file_hash(path) != digest: raise ValueError('Accepted native comparison reference bytes differ')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    for name in METHOD_FILES:
        require_equal(reference['implementation_hashes'][name], context['implementation_hashes'][name], 'unchanged method/split/'+name, 0)
    for key, value in {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                       'verified_runs': 10, 'candidate_pipelines': 80, 'outer_predictions': 0}.items():
        require_equal(reference.get(key), value, 'reference/'+key, 0)
    if len(reference['runs']) != 10 or {(r['seed'], r['trial']) for r in reference['runs']} != set(roles):
        raise ValueError('Reference original role matrix differs')
    for run in reference['runs']:
        for arm in runner.kernel.ARMS:
            for role, support in [('fitting_leave_case_out', config['fitting_five_support']), ('validation', config['validation_five_support'])]:
                require_equal(run['arms'][arm][role]['distinct_class_support'], support, 'reference support', 0)
    return reference


def comparisons(value, old, screen):
    delta = {key: value[key]-old[key] for key in value if key.startswith('mean_subset_')}
    positive = all(r > 0 for r in value['pooled_class_recall'])
    return {'deltas': delta, 'all_view_deltas': {k: value['all_view'][k]-old['all_view'][k] for k in value['all_view']},
        'binary_deltas': {k: value['binary_mean_metrics'][k]-old['binary_mean_metrics'][k] for k in value['binary_mean_metrics']},
        'malignant_deltas': {k: value['malignant_mean_metrics'][k]-old['malignant_mean_metrics'][k] for k in value['malignant_mean_metrics']},
        'class_recall_deltas': [a-b for a, b in zip(value['pooled_class_recall'], old['pooled_class_recall'])],
        'all_five_recall_positive': positive,
        'gate_passed': bool(delta['mean_subset_macro_f1'] >= screen['each_arrangement_macro_f1_delta'] and delta['mean_subset_accuracy'] >= 0 and positive)}


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('Native collector needs complete runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing native comparison output')
    config, context, trusted, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'; verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    primary = reference_comparison(config, context, roles); secondary = reference_comparison(config, context, roles, True)
    root = Path(args.runs); paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    models = {p.parent/a/('candidate%d'%i)/'model.json' for p in paths for a in runner.kernel.ARMS for i in range(4)}
    scores = {p.parent/a/('candidate%d'%i)/(r+'-predictions.jsonl') for p in paths for a in runner.kernel.ARMS for i in range(4) for r in ('train', 'validation')}
    if set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != models or set(root.rglob('*predictions*.jsonl')) != scores or list(root.rglob('*.npz')):
        raise ValueError('Incomplete or extra native run artifact matrix')
    runs = [replay.verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context, cases, bags, z, y5, roles, file_hash(audit_path)) for s, t in sorted(roles)]
    records = sum(c[role]['pooled']['records'] for r in runs for a in runner.kernel.ARMS for c in r['arms'][a]['candidates'] for role in ('fitting_leave_case_out', 'validation'))
    if len(runs) != 10 or records != 64800 or sum(r['support_classifier_instances'] for r in runs) != 30 or any(r['optimized_head_fits'] != 0 for r in runs):
        raise ValueError('Native candidate/support budget differs')
    summary, paired, vs_primary, vs_secondary = {}, {}, {}, {}
    for seed in config['seeds']:
        values = {a: {role: aggregate_summaries([r['arms'][a][role] for r in runs if r['seed'] == seed]) for role in ('fitting_leave_case_out', 'validation')} for a in runner.kernel.ARMS}
        summary[str(seed)] = values
        paired[str(seed)] = {k: values['dual']['validation'][k]-values['direct']['validation'][k] for k in values['dual']['validation'] if k.startswith('mean_subset_')}
        for reference, target in ((primary, vs_primary), (secondary, vs_secondary)):
            old = reference['summary'][str(seed)]['dual']['validation']
            target[str(seed)] = {a: comparisons(values[a]['validation'], old, config['improvement_screen']) for a in runner.kernel.ARMS}
    result = dict(context, status='complete exploratory native-global Midnight KDE comparison', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), verified_runs=10, candidate_pipelines=80, support_classifier_instances=30,
        optimized_head_fits=0, collector_optimized_head_fits=0, new_encoder_forwards=0, native_feature_forwards=7909,
        feature_input_shape=[3, 448, 700], feature_token_shape=[1601, 1536],
        compute_scope='Native extraction uses 6.25 times the pixels of a 224-square input and 1601 rather than 257 tokens; forward counts are not comparable measures of compute or runtime',
        outer_predictions=0, verified_prediction_archives=160, verified_case_subset_records=records,
        fitting_uses_support_labels=True, fitting_evaluation=runner.kernel.FITTING_POLICY,
        support_instances_scope=config['support_instances_scope'], audit_sha256=file_hash(audit_path),
        reference_comparison_sha256=config['reference_comparison_sha256'], reference_code_sha=primary['code_sha'],
        secondary_reference_comparison_sha256=config['secondary_reference_comparison_sha256'], secondary_reference_code_sha=secondary['code_sha'],
        summary=summary, paired_direct_vs_dual=paired, comparison_against_threecrop_kernel_dual=vs_primary,
        comparison_against_single_center_kernel_dual=vs_secondary,
        gate_passed=all(vs_primary[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        secondary_gate_passed=all(vs_secondary[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        gate_scope='descriptive reused provider-case validation; primary three-crop reference on both arrangements; same four temperatures',
        interpretation=config['interpretation'],
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

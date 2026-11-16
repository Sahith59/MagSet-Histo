"""Complete matrix replay of the fixed coverage intervention using reviewed KDE."""
from pathlib import Path

from . import coverage_five as runner
from . import kernel_collect as replay
from .data import CLASS_NAMES, file_hash
from .five_collect import aggregate_summaries, assert_public_safe, read_json, require_equal, verify_audit
from .five_metrics import FINAL5
from .midnight_collect import gate_comparison, delta


def reference_comparison(config, context, roles):
    path = Path(__file__).resolve().parents[1]/'docs/evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json'
    if file_hash(path) != config['reference_comparison_sha256']:
        raise ValueError('Accepted single-center comparison bytes differ')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    for name in ('data.py', 'baseline.py', 'five_metrics.py', 'bags.py', 'kernel_model.py', 'field_five.py'):
        require_equal(reference['implementation_hashes'][name], context['implementation_hashes'][name], 'unchanged method/split/'+name, 0)
    for key, value in {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                       'verified_runs': 10, 'candidate_pipelines': 80, 'outer_predictions': 0}.items():
        require_equal(reference.get(key), value, 'reference/'+key, 0)
    pairs = {(r['seed'], r['trial']) for r in reference['runs']}
    if len(reference['runs']) != 10 or pairs != set(roles): raise ValueError('Reference original roles differ')
    for run in reference['runs']:
        for arm in runner.kernel.ARMS:
            for role, support in [('fitting_leave_case_out', config['fitting_five_support']), ('validation', config['validation_five_support'])]:
                require_equal(run['arms'][arm][role]['distinct_class_support'], support, 'reference support', 0)
    return reference


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('Coverage collector needs runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing coverage comparison output')
    config, context, trusted, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'; verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    reference = reference_comparison(config, context, roles); root = Path(args.runs)
    paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    models = {p.parent/a/('candidate%d'%i)/'model.json' for p in paths for a in runner.kernel.ARMS for i in range(4)}
    scores = {p.parent/a/('candidate%d'%i)/(r+'-predictions.jsonl') for p in paths for a in runner.kernel.ARMS for i in range(4) for r in ('train', 'validation')}
    if set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != models or set(root.rglob('*predictions*.jsonl')) != scores or list(root.rglob('*.npz')):
        raise ValueError('Incomplete or extra coverage run artifact matrix')
    runs = [replay.verify_run(p.parent, s, t, config, context, cases, bags, z, y5, roles, file_hash(audit_path))
            for s, t in sorted(roles) for p in [root/('split%d'%s)/('trial%d'%t)/'run.json']]
    records = sum(c[role]['pooled']['records'] for r in runs for a in runner.kernel.ARMS for c in r['arms'][a]['candidates'] for role in ('fitting_leave_case_out', 'validation'))
    if len(runs) != 10 or records != 64800: raise ValueError('Coverage candidate budget differs')
    summary, paired, comparisons = {}, {}, {}
    for seed in config['seeds']:
        values = {a: {role: aggregate_summaries([r['arms'][a][role] for r in runs if r['seed'] == seed])
                       for role in ('fitting_leave_case_out', 'validation')} for a in runner.kernel.ARMS}
        summary[str(seed)] = values
        paired[str(seed)] = delta(values['dual']['validation'], values['direct']['validation'])
        old = reference['summary'][str(seed)]['dual']['validation']
        comparisons[str(seed)] = {a: gate_comparison(values[a]['validation'], old, config['improvement_screen']) for a in runner.kernel.ARMS}
    result = dict(context, status='complete exploratory three-crop coverage KDE comparison',
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10,
                  candidate_pipelines=80, support_classifier_instances=30, optimized_head_fits=0,
                  collector_optimized_head_fits=0, new_encoder_forwards=0, coverage_feature_new_crop_forwards=15818,
                  outer_predictions=0, verified_prediction_archives=160, verified_case_subset_records=records,
                  fitting_uses_support_labels=True, fitting_evaluation=runner.kernel.FITTING_POLICY,
                  support_instances_scope=config['support_instances_scope'], audit_sha256=file_hash(audit_path),
                  reference_comparison_sha256=config['reference_comparison_sha256'], reference_code_sha=reference['code_sha'],
                  summary=summary, paired_direct_vs_dual=paired, comparison_against_single_center_kernel_dual=comparisons,
                  gate_passed=all(comparisons[str(s)]['dual']['gate_passed'] for s in config['seeds']),
                  gate_scope='descriptive reused provider-case validation; same four temperatures; threecrop reader cost disclosed',
                  distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

"""Replay every fixed-reader KDE candidate and preserve matched reference comparisons."""
from pathlib import Path

from . import approved_five as runner, kernel_collect as replay
from .data import CLASS_NAMES, file_hash
from .five_collect import aggregate_summaries, assert_public_safe, read_json, verify_audit
from .five_metrics import FINAL5
from .genbio_collect import reference_comparison, comparisons


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('approved encoder collector needs complete runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing approved encoder comparison output')
    config, context, trusted, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'; verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    primary = reference_comparison(config, context, roles); secondary = reference_comparison(config, context, roles, True)
    root = Path(args.runs); paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    models = {p.parent/a/('candidate%d'%i)/'model.json' for p in paths for a in runner.kernel.ARMS for i in range(4)}
    scores = {p.parent/a/('candidate%d'%i)/(r+'-predictions.jsonl') for p in paths for a in runner.kernel.ARMS for i in range(4) for r in ('train', 'validation')}
    if set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != models or set(root.rglob('*predictions*.jsonl')) != scores or list(root.rglob('*.npz')):
        raise ValueError('Incomplete or extra approved encoder run artifact matrix')
    runs = [replay.verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context, cases, bags, z, y5, roles, file_hash(audit_path)) for s, t in sorted(roles)]
    records = sum(c[role]['pooled']['records'] for r in runs for a in runner.kernel.ARMS for c in r['arms'][a]['candidates'] for role in ('fitting_leave_case_out', 'validation'))
    if len(runs) != 10 or records != 64800 or sum(r['support_classifier_instances'] for r in runs) != 30 or any(r['optimized_head_fits'] != 0 for r in runs):
        raise ValueError('approved encoder candidate/support budget differs')
    summary, paired, vs_primary, vs_secondary = {}, {}, {}, {}
    for seed in config['seeds']:
        values = {a: {role: aggregate_summaries([r['arms'][a][role] for r in runs if r['seed'] == seed]) for role in ('fitting_leave_case_out', 'validation')} for a in runner.kernel.ARMS}
        summary[str(seed)] = values
        paired[str(seed)] = {k: values['dual']['validation'][k]-values['direct']['validation'][k] for k in values['dual']['validation'] if k.startswith('mean_subset_')}
        for reference, target in ((primary, vs_primary), (secondary, vs_secondary)):
            old = reference['summary'][str(seed)]['dual']['validation']
            target[str(seed)] = {a: comparisons(values[a]['validation'], old, config['improvement_screen']) for a in runner.kernel.ARMS}
    result = dict(context, status='complete exploratory frozen approved encoder KDE comparison', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), verified_runs=10, candidate_pipelines=80, support_classifier_instances=30,
        optimized_head_fits=0, collector_optimized_head_fits=0, new_encoder_forwards=0, image_forwards=7909,
        feature_input_shape=[3, 224, 224], encoder_dimension=config['dimension'],
        compute_scope='7909 original RGB fields; one frozen model call per original image; runtime and actual GPU memory reported separately',
        outer_predictions=0, verified_prediction_archives=160, verified_case_subset_records=records,
        fitting_uses_support_labels=True, fitting_evaluation=runner.kernel.FITTING_POLICY,
        support_instances_scope=config['support_instances_scope'], audit_sha256=file_hash(audit_path),
        reference_comparison_sha256=config['reference_comparison_sha256'], reference_code_sha=primary['code_sha'],
        secondary_reference_comparison_sha256=config['secondary_reference_comparison_sha256'], secondary_reference_code_sha=secondary['code_sha'],
        summary=summary, paired_direct_vs_dual=paired, comparison_against_single_center_kernel_dual=vs_primary,
        comparison_against_threecrop_kernel_dual=vs_secondary,
        gate_passed=all(vs_primary[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        secondary_gate_passed=all(vs_secondary[str(s)]['dual']['gate_passed'] for s in config['seeds']),
        gate_scope='descriptive reused provider-case validation; primary single-center Midnight reference on both arrangements; same four temperatures',
        interpretation=config['interpretation'],
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

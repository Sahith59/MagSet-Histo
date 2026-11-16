"""Replay the complete paired centering matrix, including duplicated direct controls."""
import math
from pathlib import Path

from . import prior_centered_five as runner
from .data import CLASS_NAMES, file_hash
from .five_collect import aggregate_summaries, assert_public_safe, read_json, require_equal, verify_records
from .five_metrics import FINAL5
from .midnight_collect import delta, gate_comparison


def verify_direct_duplicates(modes):
    left, right = (modes[m]['arms']['direct'] for m in runner.MODES)
    if any(len(value.get('candidates', [])) != 4 for value in (left, right)):
        raise ValueError('Duplicated direct candidate coverage differs')
    for key in ('selected_index', 'selected_temperature', 'fitting_leave_case_out', 'validation'):
        require_equal(right.get(key), left.get(key), 'duplicated direct/'+key, 0)
    for a, b in zip(left['candidates'], right['candidates']):
        for key in ('index', 'temperature', 'train_predictions_sha256', 'validation_predictions_sha256',
                    'fitting_leave_case_out', 'validation'):
            require_equal(b.get(key), a.get(key), 'duplicated direct candidate/'+key, 0)


def verify_run(folder, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha):
    from .prior_centered_model import density_cache, support_summary
    folder = Path(folder); report = read_json(folder/'run.json')
    for key, value in context.items(): require_equal(report.get(key), value, 'run context/'+key, 0)
    expected = dict(status='complete paired prior-centered KDE validation run', seed=seed, trial=trial,
                    class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
                    role_counts=config['expected_role_counts'], optimized_head_fits=0, new_encoder_forwards=0,
                    outer_predictions=0, fitting_evaluation=runner.FITTING_POLICY, fitting_labels_used_for_centers=True,
                    logical_full_fit_head_definitions=6, structurally_unique_full_fit_head_definitions=5,
                    audit_sha256=audit_sha)
    for key, value in expected.items(): require_equal(report.get(key), value, 'run/'+key, 0)
    train, val, excluded = roles[seed, trial]
    require_equal(report.get('case_ids'), {r: [cases[i]['case_id'] for i in ix] for r, ix in
                  [('train', train), ('validation', val), ('excluded', excluded)]}, 'original role membership', 0)
    if set(report.get('modes', {})) != set(runner.MODES): raise ValueError('Incomplete mode matrix')
    for mode in runner.MODES:
        arms = report['modes'][mode].get('arms', {})
        if set(arms) != set(runner.ARMS) or any(len(arms[a].get('candidates', [])) != 4 for a in runner.ARMS):
            raise ValueError('Incomplete mode/arm/temperature matrix')
    prepared = runner.prepare_queries(bags, y5, train, val)
    supports = {m: support_summary(bags, y5, train, mode=m) for m in runner.MODES}
    result = dict(seed=seed, trial=trial, run_sha256=file_hash(folder/'run.json'), optimized_head_fits=0,
                  logical_full_fit_head_definitions=6, structurally_unique_full_fit_head_definitions=5,
                  modes={m: {'arms': {a: {'candidates': []} for a in runner.ARMS}} for m in runner.MODES})
    for index, temperature in enumerate(runner.TEMPERATURES):
        caches = {r: [density_cache(q, temperature) for q in qs] for r, qs in prepared.items()}
        for mode in runner.MODES:
            for arm in runner.ARMS:
                directory = folder/mode/arm/('candidate%d'%index); candidate = read_json(directory/'metrics.json')
                require_equal(report['modes'][mode]['arms'][arm]['candidates'][index], candidate, 'embedded candidate', 0)
                for key, value in dict(index=index, temperature=temperature, mode=mode, arm=arm,
                                       optimized_head_fits=0, fitting_evaluation=runner.FITTING_POLICY).items():
                    require_equal(candidate.get(key), value, 'candidate/'+key, 0)
                elapsed = candidate.get('elapsed_seconds')
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('Invalid candidate elapsed time')
                path = directory/'model.json'
                if file_hash(path) != candidate.get('model_sha256'): raise ValueError('Support model hash differs')
                require_equal(read_json(path), runner.model_record(context, cases, train, supports[mode], arm, temperature, mode), 'support model', 0)
                for role, indices, key in [('train', train, 'fitting_leave_case_out'), ('validation', val, 'validation')]:
                    path = directory/(role+'-predictions.jsonl')
                    if file_hash(path) != candidate.get(role+'_predictions_sha256'): raise ValueError('Prediction hash differs')
                    wanted = runner.score_records(prepared[role], cases, role, arm, temperature, mode, caches[role])
                    require_equal(candidate.get(key), verify_records(path, wanted, cases, indices, role, arm), 'candidate/'+key)
                result['modes'][mode]['arms'][arm]['candidates'].append(dict(candidate, metrics_sha256=file_hash(directory/'metrics.json')))
    for mode in runner.MODES:
        for arm in runner.ARMS:
            saved = report['modes'][mode]['arms'][arm]; accepted = result['modes'][mode]['arms'][arm]; cs = accepted['candidates']
            index = max(range(4), key=lambda i: cs[i]['validation']['mean_subset_macro_f1'])
            values = dict(selected_index=index, selected_temperature=runner.TEMPERATURES[index],
                          fitting_leave_case_out=cs[index]['fitting_leave_case_out'], validation=cs[index]['validation'])
            for key, value in values.items():
                require_equal(saved.get(key), value, 'earliest selection/'+key, 0 if key.startswith('selected_') else 2e-12)
            accepted.update(values)
    verify_direct_duplicates(result['modes'])
    return result


def references(config, context, roles):
    root = Path(__file__).resolve().parents[1]; accepted = {}
    for name, prefix in [('coverage', ''), ('single_center', 'secondary_')]:
        path = root/config[prefix+'reference_file']
        if file_hash(path) != config[prefix+'reference_comparison_sha256']: raise ValueError('Reference comparison bytes differ')
        reference = read_json(path)
        for key in ('manifest_sha256', 'protocol_sha256'):
            require_equal(reference.get(key), context[key], 'reference/'+key, 0)
        for name_module in ('data.py', 'baseline.py', 'five_metrics.py', 'bags.py', 'kernel_model.py', 'field_five.py'):
            require_equal(reference['implementation_hashes'][name_module], context['implementation_hashes'][name_module], 'unchanged method/'+name_module, 0)
        for key, value in dict(class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10,
                               candidate_pipelines=80, outer_predictions=0).items():
            require_equal(reference.get(key), value, 'reference/'+key, 0)
        if len(reference['runs']) != 10 or {(r['seed'], r['trial']) for r in reference['runs']} != set(roles):
            raise ValueError('Reference role matrix differs')
        for run in reference['runs']:
            for arm in runner.ARMS:
                for role, support in [('fitting_leave_case_out', config['fitting_five_support']), ('validation', config['validation_five_support'])]:
                    require_equal(run['arms'][arm][role]['distinct_class_support'], support, 'reference class support', 0)
        if name == 'coverage':
            for key in ('encoder', 'cache_provenance', 'feature_audit_sha256'):
                require_equal(reference.get(key), context[key], 'same coverage features/'+key, 0)
            require_equal(reference.get('config_sha256'), config['base_input_config_sha256'], 'same coverage configuration', 0)
        else:
            require_equal(reference.get('cache_provenance'), context['center_cache_provenance'], 'original center features', 0)
        accepted[name] = reference
    return accepted


def collect(args):
    if args.runs is None or args.audit is None: raise ValueError('Collector requires runs and accepted audit')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing prior-centering comparison output')
    config, context, _, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit = Path(args.audit)/'audit.json'; runner.verify_input_audit(read_json(audit), context, config, bags, y5, roles)
    old = references(config, context, roles); root = Path(args.runs)
    paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    directories = {p.parent/m/a/('candidate%d'%i) for p in paths for m in runner.MODES for a in runner.ARMS for i in range(4)}
    if (set(root.rglob('run.json')) != paths or set(root.rglob('model.json')) != {p/'model.json' for p in directories}
            or set(root.rglob('metrics.json')) != {p/'metrics.json' for p in directories}
            or set(root.rglob('*predictions*.jsonl')) != {p/(r+'-predictions.jsonl') for p in directories for r in ('train', 'validation')}
            or list(root.rglob('*.npz')) or list(root.rglob('*.pt'))):
        raise ValueError('Incomplete or extra paired centering artifact matrix')
    runs = [verify_run(root/('split%d'%s)/('trial%d'%t), s, t, config, context, cases, bags, z, y5, roles, file_hash(audit)) for s, t in sorted(roles)]
    def count(unique):
        return sum(c[r]['pooled']['records'] for x in runs for m in runner.MODES for a in runner.ARMS
                   if not unique or m == runner.MODES[0] or a == 'dual'
                   for c in x['modes'][m]['arms'][a]['candidates'] for r in ('fitting_leave_case_out', 'validation'))
    if len(runs) != 10 or count(False) != 129600 or count(True) != 97200: raise ValueError('Paired centering record budget differs')
    summary, paired, comparisons, passed = {}, {}, {}, {}
    for mode in runner.MODES:
        summary[mode], paired[mode], comparisons[mode] = {}, {}, {}
        for seed in config['seeds']:
            values = {a: {r: aggregate_summaries([x['modes'][mode]['arms'][a][r] for x in runs if x['seed'] == seed])
                           for r in ('fitting_leave_case_out', 'validation')} for a in runner.ARMS}
            summary[mode][str(seed)] = values; paired[mode][str(seed)] = delta(values['dual']['validation'], values['direct']['validation'])
            comparisons[mode][str(seed)] = {name: {a: gate_comparison(values[a]['validation'], ref['summary'][str(seed)]['dual']['validation'], config['improvement_screen']) for a in runner.ARMS} for name, ref in old.items()}
        passed[mode] = all(comparisons[mode][str(s)]['coverage']['dual']['gate_passed'] for s in config['seeds'])
    result = dict(context, status='complete exploratory paired supervised-centering comparison',
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_run_pairs=10,
                  candidate_pipelines=160, verified_prediction_archives=320, verified_case_subset_records=count(False),
                  **runner.COUNTS, optimized_head_fits=0, collector_optimized_head_fits=0, new_encoder_forwards=0,
                  outer_predictions=0, fitting_labels_used_for_centers=True, fitting_evaluation=runner.FITTING_POLICY,
                  normalization=config['normalization'], temperatures=runner.TEMPERATURES, modes=list(runner.MODES),
                  direct_controls_identical=True, selection_opportunities_per_named_mode_arm=4,
                  selection_opportunities_if_choosing_between_modes_per_arm=8, audit_sha256=file_hash(audit),
                  structurally_unique_family_selection_opportunities=config['structurally_unique_family_selection_opportunities'],
                  family_opportunity_scope=config['family_opportunity_scope'],
                  duplicate_scope=config['duplicate_scope'], interpretation=config['interpretation'],
                  reference_comparison_sha256=config['reference_comparison_sha256'],
                  secondary_reference_comparison_sha256=config['secondary_reference_comparison_sha256'],
                  summary=summary, paired_direct_vs_dual=paired, comparisons_against_reference_dual=comparisons,
                  gate_passed_by_mode=passed, gate_scope='descriptive reused provider-case validation; both modes reported; duplicated direct controls are not independent observations',
                  distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

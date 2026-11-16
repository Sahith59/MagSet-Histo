"""Verify the complete two-readout Midnight classifier matrix without fitting."""
from pathlib import Path

from . import field_collect, five_collect
from .data import CLASS_NAMES, file_hash
from .five_class import ARMS, CONFIG_SHA256 as PHIKON_CONFIG_SHA256, write_json
from .five_collect import (aggregate_summaries, assert_public_safe, read_json,
                           require_equal, verify_audit)
from .five_metrics import FINAL5

READOUTS = ('case', 'field')
REFERENCE_COMPARISON_SHA256 = 'bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795'


def validate_matrix_contract(config, context):
    """The two registered readouts share all ten original roles and budgets."""
    expected = {'readouts': list(READOUTS), 'seeds': [812, 813], 'trials': [1, 2, 3, 4, 5],
                'arms': list(ARMS), 'C_grid': [.01, .1, 1., 10.],
                'expected_role_counts': [46, 8, 28], 'fitting_five_support': [11, 25, 2, 5, 3],
                'validation_five_support': [4, 1, 1, 1, 1], 'excluded_five_support': [9, 12, 2, 3, 2],
                'production_runs': 10, 'production_head_fits': 120,
                'total_production_runs': 20, 'total_head_fits': 240}
    for key, value in expected.items():
        require_equal(config.get(key), value, 'registered matrix/'+key, 0)
    require_equal(config.get('reference_comparison_sha256'), REFERENCE_COMPARISON_SHA256, 'registered reference', 0)
    screen = config.get('improvement_screen', {})
    for key, value in {'each_arrangement_macro_f1_delta': .02, 'accuracy_not_lower': True,
                       'positive_recall_all_five_classes': True}.items():
        require_equal(screen.get(key), value, 'registered screen/'+key, 0)
    if context.get('encoder', {}).get('dimension') != 3072 or 'readout' in context:
        raise ValueError('Common Midnight context must be 3072-dimensional and readout-independent')


def reference_comparison(config, context, path=None):
    """Bind the old task/roles; its different encoder and cache are intentional."""
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json'
    if file_hash(path) != config['reference_comparison_sha256']:
        raise ValueError('Accepted Phikon reference hash differs')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256'):
        require_equal(reference.get(key), context[key], 'reference/'+key, 0)
    expected = {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                'config_sha256': PHIKON_CONFIG_SHA256, 'verified_runs': 10, 'head_fits': 120,
                'outer_predictions': 0, 'C_grid': config['C_grid'],
                'routing_threshold': config['routing_threshold'], 'routing_tie': config['routing_tie']}
    for key, value in expected.items():
        require_equal(reference.get(key), value, 'reference/'+key, 0)
    wanted = {(seed, trial) for seed in config['seeds'] for trial in config['trials']}
    runs = reference.get('runs')
    if not isinstance(runs, list) or len(runs) != len(wanted):
        raise ValueError('Reference run matrix differs')
    seen = set()
    for run in runs:
        key = (run.get('seed'), run.get('trial'))
        if key not in wanted or key in seen or set(run.get('arms', {})) != set(ARMS):
            raise ValueError('Reference original role matrix differs')
        seen.add(key)
        for arm in ARMS:
            for role, support, count in [('fitting', config['fitting_five_support'], 46),
                                          ('validation', config['validation_five_support'], 8)]:
                value = run['arms'][arm][role]
                require_equal(value.get('distinct_class_support'), support, 'reference role support', 0)
                require_equal(value.get('distinct_cases'), count, 'reference role cases', 0)
                require_equal(value.get('pooled', {}).get('records'), count*15, 'reference subset coverage', 0)
    return reference


def expected_inventory(root, config):
    paths = {root/readout/('split%d'%seed)/('trial%d'%trial)/'run.json'
             for readout in READOUTS for seed in config['seeds'] for trial in config['trials']}
    models = {path.parent/arm/('candidate%d'%i)/'model.npz'
              for path in paths for arm in ARMS for i in range(len(config['C_grid']))}
    scores = {path.parent/arm/('candidate%d'%i)/(role+'-predictions.jsonl')
              for path in paths for arm in ARMS for i in range(len(config['C_grid']))
              for role in ('train', 'validation')}
    return paths, models, scores


def verify_run(folder, readout, seed, trial, config, context, cases, bags, z, y5, roles, audit_sha):
    if readout not in READOUTS:
        raise ValueError('Only registered case/field readouts may be verified')
    verifier = five_collect.verify_run if readout == 'case' else field_collect.verify_run
    result = verifier(folder, seed, trial, config, dict(context, readout=readout),
                      cases, bags, z, y5, roles, audit_sha)
    return dict(result, readout=readout)


def delta(value, reference):
    keys = ('mean_subset_macro_f1', 'mean_subset_accuracy', 'mean_subset_balanced_accuracy')
    result = {'delta_'+key: value[key]-reference[key] for key in keys}
    result['delta_all_view_accuracy'] = value['all_view']['accuracy']-reference['all_view']['accuracy']
    return result


def gate_comparison(value, reference, screen):
    result = delta(value, reference)
    positive = all(recall > 0 for recall in value['pooled_class_recall'])
    result['all_five_recall_positive'] = positive
    result['gate_passed'] = bool(result['delta_mean_subset_macro_f1'] >= screen['each_arrangement_macro_f1_delta']
                                  and result['delta_mean_subset_accuracy'] >= 0 and positive)
    return result


def aggregate(runs, config, context, cases, roles, reference, audit_sha):
    summary, paired, comparisons = {}, {}, {}
    for readout in READOUTS:
        rs, rp, rc = {}, {}, {}
        for seed in config['seeds']:
            selected = [r for r in runs if r['readout'] == readout and r['seed'] == seed]
            ss = {}
            for arm in ARMS:
                value = {role: aggregate_summaries([r['arms'][arm][role] for r in selected])
                         for role in ('fitting', 'validation')}
                value['fitting_minus_validation'] = {k: value['fitting'][k]-value['validation'][k]
                                                     for k in value['validation'] if k.startswith('mean_subset_')}
                ss[arm] = value
            ss['distinct_validation_cases'] = len({cases[i]['case_id']
                for (s, _), (_, val, _) in roles.items() if s == seed for i in val})
            rp[str(seed)] = delta(ss['dual']['validation'], ss['direct']['validation'])
            old = reference['summary'][str(seed)]['dual']['validation']
            rc[str(seed)] = {arm: gate_comparison(ss[arm]['validation'], old, config['improvement_screen'])
                             for arm in ARMS}
            rs[str(seed)] = ss
        summary[readout], paired[readout], comparisons[readout] = rs, rp, rc
    field_case = {str(seed): {arm: delta(summary['field'][str(seed)][arm]['validation'],
                                        summary['case'][str(seed)][arm]['validation']) for arm in ARMS}
                  for seed in config['seeds']}
    gates = {r: all(v['dual']['gate_passed'] for v in comparisons[r].values()) for r in READOUTS}
    records = sum(c[role]['pooled']['records'] for r in runs for arm in ARMS
                  for c in r['arms'][arm]['candidates'] for role in ('fitting', 'validation'))
    if records != 129600:
        raise ValueError('Complete Midnight candidate record budget differs')
    result = dict(context)
    result.update(status='complete exploratory two-readout Midnight five-class development comparison',
        class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), readouts=list(READOUTS),
        verified_runs=len(runs), verified_runs_per_readout={r: sum(x['readout'] == r for x in runs) for r in READOUTS},
        candidate_pipelines=160, head_fits=sum(r['head_fits'] for r in runs), production_head_fits=config['total_head_fits'],
        verified_prediction_archives=320, verified_case_subset_records=records,
        collector_new_fits=0, new_encoder_forwards=0, outer_predictions=0, audit_sha256=audit_sha,
        reference_comparison_sha256=config['reference_comparison_sha256'], reference_code_sha=reference['code_sha'],
        reference_encoder=reference['encoder'],
        reference_dual_validation={str(s): reference['summary'][str(s)]['dual']['validation'] for s in config['seeds']},
        summary=summary, paired_direct_vs_dual=paired, paired_field_vs_case=field_case,
        comparison_against_phikon_case_dual=comparisons, gate_passed_by_readout=gates,
        gate_scope='Separate predeclared readout screens on reused provider-case validation; both arrangements required per readout; not a fresh test',
        runs=runs, distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}),
        selection=config['selection'], C_grid=config['C_grid'], routing_threshold=config['routing_threshold'],
        routing_tie=config['routing_tie'], readout_definitions=config['readout_definitions'])
    assert_public_safe(result)
    return result


def collect(args):
    from .midnight_five import load_inputs
    if args.runs is None or args.audit is None:
        raise ValueError('Collector requires --runs and accepted --audit')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing comparison output')
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    validate_matrix_contract(config, context)
    audit_path = Path(args.audit)/'audit.json'
    verify_audit(read_json(audit_path), context, config, bags, y5, roles)
    reference = reference_comparison(config, context, getattr(args, 'reference', None))
    root = Path(args.runs); audit_sha = file_hash(audit_path)
    paths, models, scores = expected_inventory(root, config)
    if set(root.rglob('run.json')) != paths:
        raise ValueError('Complete two-readout run matrix is missing or has extra runs')
    if set(root.rglob('*.npz')) != models or set(root.rglob('*predictions*.jsonl')) != scores:
        raise ValueError('Complete two-readout artifact matrix differs')
    runs = [verify_run(root/readout/('split%d'%seed)/('trial%d'%trial), readout, seed, trial,
                       config, context, cases, bags, z, y5, roles, audit_sha)
            for readout in READOUTS for seed in config['seeds'] for trial in config['trials']]
    if len(runs) != config['total_production_runs'] or sum(r['head_fits'] for r in runs) != config['total_head_fits']:
        raise ValueError('Complete Midnight run/head budget differs')
    result = aggregate(runs, config, context, cases, roles, reference, audit_sha)
    output.mkdir(parents=True); write_json(output/'comparison.json', result)
    return result

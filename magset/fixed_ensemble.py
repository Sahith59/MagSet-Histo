"""One fixed equal joint-probability mixture of accepted Midnight case and KDE outputs."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from . import midnight_five as base
from .bags import fit_normalization
from .collect import SUBSETS
from .data import CLASS_NAMES, file_hash
from .five_collect import (_unique_object, _invalid_constant, aggregate_summaries,
                           assert_public_safe, read_json, require_equal, verify_audit)
from .five_metrics import FINAL5, MAP8_TO5, summarize_records

ARMS = ('direct', 'dual')
IDENTITY = ('case_id', 'role', 'truth', 'truth5', 'magnifications')


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]; canonical = root/'configs/breakhis-fixed-ensemble-v1.json'
    if file_hash(args.config) != file_hash(canonical): raise ValueError('Ensemble config differs from immutable registration')
    own = read_json(args.config)
    fixed = {'name': 'breakhis-fixed-ensemble-v1', 'schema_version': 1, 'midnight_readout': 'case',
             'weight_midnight': .5, 'weight_kernel': .5, 'arms': list(ARMS), 'production_run_pairs': 10,
             'source_selected_prediction_archives': 40, 'source_case_subset_records': 4800,
             'prediction_archives': 20, 'case_subset_records': 2400, 'new_optimized_coefficients': 0,
             'new_encoder_forwards': 0, 'new_weight_selections': 0,
             'inherited_selection_opportunities_per_role_per_arm': {'midnight_case_C': 4, 'kernel_temperature': 4},
             'routing_threshold': .5, 'routing_tie': 'malignant'}
    for key, value in fixed.items(): require_equal(own.get(key), value, 'registered ensemble/'+key, 0)
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']: raise ValueError('Accepted Midnight base config differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path
    config, context, trusted, cases, bags, z, y5, roles = base.load_inputs(forwarded)
    context = dict(context, ensemble_config_sha256=file_hash(args.config))
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('fixed_ensemble.py',)))
    context['implementation_hashes'] = {n: file_hash(root/'magset'/n) for n in names}
    return {**config, **own}, context, trusted, cases, bags, z, y5, roles


def references(config, context, roles):
    root = Path(__file__).resolve().parents[1]; result = {}
    expected_roles = set(roles)
    for name in ('midnight', 'kernel'):
        path = root/config[name+'_reference_file']
        if file_hash(path) != config[name+'_reference_sha256']: raise ValueError('Accepted '+name+' ledger hash differs')
        document = read_json(path)
        for key in ('config_sha256', 'manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance'):
            require_equal(document.get(key), context[key], name+' reference/'+key, 0)
        for key, value in {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'outer_predictions': 0}.items():
            require_equal(document.get(key), value, name+' reference/'+key, 0)
        require_equal(document.get('verified_runs'), 20 if name == 'midnight' else 10, name+' run budget', 0)
        selected = [r for r in document.get('runs', []) if name == 'kernel' or r.get('readout') == 'case']
        if len(selected) != 10 or {(r.get('seed'), r.get('trial')) for r in selected} != expected_roles:
            raise ValueError('Accepted '+name+' role matrix differs')
        for run in selected:
            if set(run.get('arms', {})) != set(ARMS): raise ValueError('Reference arm coverage differs')
            for arm in ARMS:
                fit = run['arms'][arm]['fitting' if name == 'midnight' else 'fitting_leave_case_out']
                require_equal(fit.get('distinct_class_support'), config['fitting_five_support'], 'reference fitting support', 0)
                require_equal(fit.get('distinct_cases'), 46, 'reference fitting cases', 0)
                value = run['arms'][arm]['validation']
                require_equal(value.get('distinct_class_support'), config['validation_five_support'], 'reference class support', 0)
                require_equal(value.get('distinct_cases'), 8, 'reference validation cases', 0)
                require_equal(value.get('pooled', {}).get('records'), 120, 'reference subset coverage', 0)
        result[name] = {'document': document, 'runs': {(r['seed'], r['trial']): r for r in selected}}
    return result


def checked_records(path, cases, validation, arm):
    records = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
               for line in Path(path).read_text().splitlines()]
    labels = {cases[i]['case_id']: int(cases[i]['label']) for i in validation}; indexed = {}
    keys = [(cases[i]['case_id'], tuple(subset)) for subset in SUBSETS for i in validation]
    for row in records:
        if (not isinstance(row, dict) or row.get('role') != 'validation' or row.get('case_id') not in labels
                or type(row.get('truth')) is not int or row['truth'] != labels[row['case_id']]
                or type(row.get('truth5')) is not int or row['truth5'] != MAP8_TO5[row['truth']]):
            raise ValueError('Input differs from original trusted validation role/labels')
        try: key = row['case_id'], tuple(row['magnifications'])
        except (KeyError, TypeError) as error: raise ValueError('Invalid input subset') from error
        if key in indexed or key not in keys: raise ValueError('Duplicate or unexpected input case/subset')
        indexed[key] = row
    if len(records) != 120 or set(indexed) != set(keys): raise ValueError('Incomplete eight-case fifteen-subset input')
    ordered = [indexed[k] for k in keys]
    return ordered, summarize_records(ordered, arm)


def load_components(args, config, context, cases, roles):
    refs = references(config, context, roles); sources = {}; bindings = []
    for (seed, trial), (_, validation, _) in sorted(roles.items()):
        for name in ('midnight', 'kernel'):
            public = refs[name]['runs'][seed, trial]
            root = Path(args.midnight_runs)/'case' if name == 'midnight' else Path(args.kernel_runs)
            folder = root/('split%d'%seed)/('trial%d'%trial); path = folder/'run.json'
            if file_hash(path) != public['run_sha256']: raise ValueError('Actual '+name+' run hash differs from accepted ledger')
            actual = read_json(path)
            for key, value in {'seed': seed, 'trial': trial, 'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES)}.items():
                require_equal(actual.get(key), value, 'source identity/'+key, 0)
            for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance', 'config_sha256'):
                require_equal(actual.get(key), context[key], 'source context/'+key, 0)
            identities = {r: [cases[i]['case_id'] for i in ix] for r, ix in zip(('train', 'validation', 'excluded'), roles[seed, trial])}
            require_equal(actual.get('case_ids'), identities, 'source original roles', 0)
            require_equal(actual.get('role_counts'), [46, 8, 28], 'source role counts', 0)
            for arm in ARMS:
                accepted = public['arms'][arm]; private = actual['arms'][arm]; index = accepted['selected_index']
                if type(index) is not int or len(accepted.get('candidates', [])) != 4:
                    raise ValueError('Accepted component candidate budget differs')
                chosen = max(range(4), key=lambda i: accepted['candidates'][i]['validation']['mean_subset_macro_f1'])
                require_equal(index, chosen, 'accepted earliest component selection', 0)
                require_equal(private.get('selected_index'), index, 'source selected index', 0)
                parameter = 'selected_C' if name == 'midnight' else 'selected_temperature'
                require_equal(private.get(parameter), accepted[parameter], 'source selected parameter', 0)
                directory = folder/arm/('candidate%d'%index); metrics_path = directory/'metrics.json'
                candidate = accepted['candidates'][index]
                require_equal(candidate.get('index'), index, 'selected candidate identity', 0)
                require_equal(accepted[parameter], candidate['C' if name == 'midnight' else 'temperature'], 'selected candidate parameter', 0)
                if file_hash(metrics_path) != candidate['metrics_sha256']: raise ValueError('Source candidate metrics hash differs')
                metric = read_json(metrics_path)
                require_equal(private['candidates'][index], metric, 'embedded source candidate', 0)
                for key, value in metric.items(): require_equal(candidate.get(key), value, 'accepted source candidate/'+key, 0)
                prediction = directory/'validation-predictions.jsonl'; digest = file_hash(prediction)
                if digest != candidate['validation_predictions_sha256']: raise ValueError('Source selected predictions hash differs')
                ordered, summary = checked_records(prediction, cases, validation, arm)
                require_equal(summary, accepted['validation'], 'accepted source metrics')
                sources[name, seed, trial, arm] = ordered
                bindings.append({'source': name, 'seed': seed, 'trial': trial, 'arm': arm,
                    'run_sha256': public['run_sha256'], 'selected_index': index, 'metrics_sha256': file_hash(metrics_path),
                    'prediction_sha256': digest, 'records': len(ordered)})
        for arm in ARMS:
            a, b = (sources[n, seed, trial, arm] for n in ('midnight', 'kernel'))
            require_equal([{k: r[k] for k in IDENTITY} for r in a], [{k: r[k] for k in IDENTITY} for r in b], 'joined component identity', 0)
    if len(bindings) != 40 or sum(b['records'] for b in bindings) != 4800: raise ValueError('Complete forty-archive input budget differs')
    return refs, sources, bindings


def blend_records(left, right, arm, independent=False):
    if arm not in ARMS or len(left) != len(right): raise ValueError('Declared arm and paired records required')
    result = []
    for a, b in zip(left, right):
        require_equal({k: a[k] for k in IDENTITY}, {k: b[k] for k in IDENTITY}, 'blend identity', 0)
        x, y = np.asarray(a['log_probabilities5']), np.asarray(b['log_probabilities5'])
        if independent:
            maximum = np.maximum(x, y)
            logs = maximum+np.log(.5*np.exp(x-maximum)+.5*np.exp(y-maximum))
            probability = np.asarray([.5*float(u)+.5*float(v) for u, v in zip(a['probabilities5'], b['probabilities5'])])
        else:
            logs = np.logaddexp(x, y)-np.log(2.)
            probability = .5*np.asarray(a['probabilities5'], dtype=np.float64)+.5*np.asarray(b['probabilities5'], dtype=np.float64)
        log_r = logsumexp(logs[1:]); conditional = logs[1:]-log_r
        score = float(np.clip(probability[1:].sum(), 0, 1))
        conditional_probability = probability[1:]/score if score > 0 else np.exp(conditional)
        prediction = int(probability.argmax()) if arm == 'direct' else (1+int(conditional_probability.argmax()) if score >= .5 else 0)
        row = {k: a[k] for k in IDENTITY}
        row.update(probabilities5=probability.tolist(), log_probabilities5=logs.tolist(), prediction=prediction,
                   binary_score=score, binary_log_probabilities=[float(logs[0]), float(log_r)],
                   malignant_probabilities4=conditional_probability.tolist(), malignant_log_probabilities=conditional.tolist())
        result.append(row)
    return result


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    refs, sources, bindings = load_components(args, config, context, cases, roles)
    checks = []
    for (seed, trial), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
            'fitting_support': np.bincount(y5[train], minlength=5).tolist(), 'validation_support': np.bincount(y5[val], minlength=5).tolist(),
            'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing audit output')
    output.mkdir(parents=True); base.write_json(output/'audit.json', {'status': 'zero-fit inputs accepted', 'new_fits': 0,
        'new_forwards': 0, 'outer_predictions': 0, 'checks': checks, 'source_bindings': bindings, **context})


def audited_inputs(args):
    values = load_inputs(args); config, context, trusted, cases, bags, z, y5, roles = values
    refs, sources, bindings = load_components(args, config, context, cases, roles)
    path = Path(args.audit)/'audit.json'; audit = read_json(path)
    verify_audit(audit, context, config, bags, y5, roles)
    require_equal(audit.get('source_bindings'), bindings, 'accepted forty-source input audit', 0)
    return values, refs, sources, bindings, file_hash(path)


def run(args):
    (config, context, trusted, cases, bags, z, y5, roles), refs, sources, bindings, audit_sha = audited_inputs(args)
    train, val, excluded = roles[args.seed, args.trial]; output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing ensemble run')
    output.mkdir(parents=True); arms = {}
    for arm in ARMS:
        records = blend_records(sources['midnight', args.seed, args.trial, arm], sources['kernel', args.seed, args.trial, arm], arm)
        folder = output/arm; folder.mkdir(); path = folder/'validation-predictions.jsonl'
        path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
        metric = {'validation': summarize_records(records, arm), 'prediction_sha256': file_hash(path)}
        base.write_json(folder/'metrics.json', metric); arms[arm] = metric
    report = {'status': 'complete fixed equal-probability ensemble validation run', 'seed': args.seed, 'trial': args.trial,
        'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'role_counts': [46, 8, 28],
        'case_ids': {r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]},
        'source_bindings': [b for b in bindings if (b['seed'], b['trial']) == (args.seed, args.trial)],
        'audit_sha256': audit_sha, 'arms': arms, 'new_optimized_coefficients': 0, 'new_encoder_forwards': 0, 'new_weight_selections': 0, **context}
    base.write_json(output/'run.json', report)


def collect(args):
    (config, context, trusted, cases, bags, z, y5, roles), refs, sources, bindings, audit_sha = audited_inputs(args)
    output = Path(args.output); root = Path(args.runs)
    if output.exists(): raise ValueError('Refusing existing ensemble comparison')
    paths = {root/('split%d'%s)/('trial%d'%t)/'run.json' for s, t in roles}
    scores = {p.parent/a/'validation-predictions.jsonl' for p in paths for a in ARMS}
    if set(root.rglob('run.json')) != paths or set(root.rglob('*predictions*.jsonl')) != scores or list(root.rglob('*.npz')) or list(root.rglob('*.pt')):
        raise ValueError('Complete ten-pair ensemble artifact matrix differs')
    runs = []
    for seed, trial in sorted(roles):
        folder = root/('split%d'%seed)/('trial%d'%trial); report = read_json(folder/'run.json')
        expected = dict(context, status='complete fixed equal-probability ensemble validation run', seed=seed, trial=trial,
            class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), role_counts=[46, 8, 28], audit_sha256=audit_sha,
            new_optimized_coefficients=0, new_encoder_forwards=0, new_weight_selections=0)
        for key, value in expected.items(): require_equal(report.get(key), value, 'ensemble run/'+key, 0)
        require_equal(report.get('case_ids'), {r: [cases[i]['case_id'] for i in ix] for r, ix in zip(('train', 'validation', 'excluded'), roles[seed, trial])}, 'ensemble roles', 0)
        require_equal(report.get('source_bindings'), [b for b in bindings if (b['seed'], b['trial']) == (seed, trial)], 'ensemble source bindings', 0)
        if set(report.get('arms', {})) != set(ARMS): raise ValueError('Incomplete ensemble arms')
        arms = {}
        for arm in ARMS:
            path = folder/arm/'validation-predictions.jsonl'; metric = read_json(folder/arm/'metrics.json')
            require_equal(report['arms'][arm], metric, 'embedded ensemble metrics', 0)
            if file_hash(path) != metric.get('prediction_sha256'): raise ValueError('Ensemble prediction hash differs')
            actual, summary = checked_records(path, cases, roles[seed, trial][1], arm)
            replay = blend_records(sources['midnight', seed, trial, arm], sources['kernel', seed, trial, arm], arm, independent=True)
            require_equal(actual, replay, 'independent linear-probability replay', 2e-12)
            require_equal(metric.get('validation'), summary, 'ensemble validation metrics')
            arms[arm] = dict(metric, metrics_sha256=file_hash(folder/arm/'metrics.json'))
        runs.append({'seed': seed, 'trial': trial, 'run_sha256': file_hash(folder/'run.json'), 'arms': arms})
    summary, comparisons, paired = {}, {}, {}
    for seed in config['seeds']:
        values = {a: {'validation': aggregate_summaries([r['arms'][a]['validation'] for r in runs if r['seed'] == seed])} for a in ARMS}
        comparisons[str(seed)] = {}
        for name in ('midnight', 'kernel'):
            reference = refs[name]['document']['summary']; old = reference['case'][str(seed)]['dual']['validation'] if name == 'midnight' else reference[str(seed)]['dual']['validation']
            comparisons[str(seed)][name] = {}
            for arm in ARMS:
                current = values[arm]['validation']; delta = {key: current[key]-old[key] for key in current if key.startswith('mean_subset_')}
                positive = all(v > 0 for v in current['pooled_class_recall'])
                comparisons[str(seed)][name][arm] = {'deltas': delta, 'all_five_recall_positive': positive,
                    'all_view_deltas': {k: current['all_view'][k]-old['all_view'][k] for k in current['all_view']},
                    'binary_deltas': {k: current['binary_mean_metrics'][k]-old['binary_mean_metrics'][k] for k in current['binary_mean_metrics']},
                    'malignant_deltas': {k: current['malignant_mean_metrics'][k]-old['malignant_mean_metrics'][k] for k in current['malignant_mean_metrics']},
                    'class_recall_deltas': (np.asarray(current['pooled_class_recall'])-old['pooled_class_recall']).tolist(),
                    'gate_passed': bool(delta['mean_subset_macro_f1'] >= config['improvement_screen']['each_arrangement_macro_f1_delta'] and delta['mean_subset_accuracy'] >= 0 and positive)}
        paired[str(seed)] = {k: values['dual']['validation'][k]-values['direct']['validation'][k] for k in values['dual']['validation'] if k.startswith('mean_subset_')}
        summary[str(seed)] = values
    result = dict(context, status='complete exploratory fixed equal-probability ensemble comparison', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), verified_runs=10, verified_prediction_archives=20, verified_case_subset_records=2400,
        verified_source_archives=40, verified_source_case_subset_records=4800, new_optimized_coefficients=0, new_encoder_forwards=0,
        new_weight_selections=0, outer_predictions=0, weights={'midnight_case': .5, 'kernel': .5},
        inherited_selection_opportunities_per_role_per_arm=config['inherited_selection_opportunities_per_role_per_arm'],
        audit_sha256=audit_sha, source_bindings=bindings, summary=summary, paired_direct_vs_dual=paired, comparisons=comparisons,
        gate_passed=all(v['kernel']['dual']['gate_passed'] for v in comparisons.values()),
        gate_scope='descriptive screen against KDE dual on both reused validation arrangements; not a fresh test',
        midnight_reference_sha256=config['midnight_reference_sha256'], kernel_reference_sha256=config['kernel_reference_sha256'],
        selection=config['selection'], interpretation=config['interpretation'], runs=runs,
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}))
    assert_public_safe(result); output.mkdir(parents=True); base.write_json(output/'comparison.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('audit', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output', 'midnight-runs', 'kernel-runs'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None: raise ValueError('Run needs accepted audit, seed and trial')
        run(args)
    else:
        if args.audit is None or args.runs is None: raise ValueError('Collect needs accepted audit and complete runs')
        collect(args)

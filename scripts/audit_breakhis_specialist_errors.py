"""Hash-bound, descriptive provider-case error audit; no model fitting or selection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magset.baseline import stratified_inner_split
from magset.collect import SUBSETS, load_trusted_context
from magset.data import case_id, file_hash, read_manifest
from magset.five_class import source_context
from magset.five_collect import (assert_public_safe, read_json, require_equal,
                                 _invalid_constant, _unique_object)
from magset.five_metrics import FINAL5, MAP8_TO5, summarize_records

PIPELINES = ('identity', 'binary_only', 'malignant_only', 'both', 'uni_field_linear_dual')
ROLE_KEYS = {(seed, trial) for seed in (812, 813) for trial in range(1, 6)}
ALL_FOUR = (40, 100, 200, 400)
PRODUCER_CONTEXT = ('code_sha', 'source_archive_sha256', 'implementation_hashes',
                    'config_sha256', 'manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bound_comparison(path, expected_sha, trusted, kind):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha or '') or file_hash(path) != expected_sha:
        raise ValueError('Accepted '+kind+' comparison bytes differ')
    value = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256'):
        require_equal(value.get(key), trusted[key], kind+' comparison/'+key, 0)
    expected_status = ('complete exploratory frozen UNI2-h head-specific metric factorial' if kind == 'metric'
                       else 'complete exploratory UNI2-h field logistic comparison')
    if (value.get('status') != expected_status or value.get('verified_runs') != 10
            or value.get('class_names') != list(FINAL5) or value.get('outer_predictions') != 0
            or len(value.get('runs', [])) != 10
            or {(r['seed'], r['trial']) for r in value['runs']} != ROLE_KEYS):
        raise ValueError('Accepted '+kind+' comparison scope differs')
    for key in ('code_sha', 'source_archive_sha256', 'config_sha256'):
        length = 40 if key == 'code_sha' else 64
        if not re.fullmatch('[0-9a-f]{%d}'%length, value.get(key) or ''):
            raise ValueError('Accepted producer provenance incomplete')
    return value


def read_bound_predictions(path, digest, cases, validation, wanted_summary):
    if file_hash(path) != digest: raise ValueError('Selected validation archive bytes differ')
    rows = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            for line in Path(path).read_text().splitlines()]
    labels = {cases[i]['case_id']: int(cases[i]['label']) for i in validation}
    seen = set()
    for row in rows:
        if (row.get('role') != 'validation' or row.get('case_id') not in labels
                or row.get('truth') != labels[row['case_id']]
                or type(row.get('truth')) is not int or row.get('truth5') != MAP8_TO5[row['truth']]):
            raise ValueError('Selected validation archive differs from trusted original truth/role')
        key = (row['case_id'], tuple(row.get('magnifications', [])))
        if key in seen: raise ValueError('Duplicate selected case/subset would corrupt joining')
        seen.add(key)
    expected = {(identity, subset) for identity in labels for subset in SUBSETS}
    if seen != expected or len(rows) != 120:
        raise ValueError('Selected archive lacks exact eight-case/fifteen-subset coverage')
    # This checks all normalized/stable probabilities, joint/conditional
    # decomposition and the actual fixed .5 then malignant-argmax decision.
    summary = summarize_records(rows, 'dual')
    require_equal(summary, wanted_summary, 'selected validation summary', 1e-10)
    return rows


def load_joined(metric, linear, metric_root, linear_root, trusted):
    cases = trusted['cases']; joined = {name: {} for name in PIPELINES}; receipts = []
    role_membership = {}
    for kind, comparison, root in [('metric', metric, Path(metric_root)), ('linear', linear, Path(linear_root))]:
        for public_run in sorted(comparison['runs'], key=lambda r: (r['seed'], r['trial'])):
            seed, trial = public_run['seed'], public_run['trial']
            train, validation, excluded = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
            if [len(train), len(validation), len(excluded)] != [46, 8, 28]:
                raise ValueError('Original role reconstruction changed')
            expected_ids = {role: [cases[i]['case_id'] for i in indices] for role, indices in
                            [('train', train), ('validation', validation), ('excluded', excluded)]}
            folder = root/('split%d'%seed)/('trial%d'%trial)
            path = folder/'run.json'
            if file_hash(path) != public_run['run_sha256']: raise ValueError('Accepted original run bytes differ')
            actual = read_json(path)
            for key in PRODUCER_CONTEXT:
                require_equal(actual.get(key), comparison[key], kind+' producer/'+key, 0)
            require_equal(actual.get('case_ids'), expected_ids, 'original role membership', 0)
            require_equal(actual.get('role_counts'), [46, 8, 28], 'role counts', 0)
            require_equal([actual.get('seed'), actual.get('trial')], [seed, trial], 'source role identity', 0)
            role_membership[(seed, trial)] = expected_ids
            if kind == 'metric':
                selected = [(name, folder/name/'validation-predictions.jsonl', public_run['arms'][name], actual['arms'][name])
                            for name in PIPELINES[:4]]
            else:
                public_arm = public_run['arms']['dual']; actual_arm = actual['arms']['dual']
                require_equal(actual_arm['selected_index'], public_arm['selected_index'], 'saved linear selection', 0)
                candidates = public_arm['candidates']
                selected_index = max(range(4), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
                require_equal(public_arm['selected_index'], selected_index, 'original earliest maximum selection', 0)
                directory = folder/'dual'/('candidate%d'%selected_index)
                candidate = candidates[selected_index]
                if file_hash(directory/'metrics.json') != candidate['metrics_sha256']:
                    raise ValueError('Accepted selected linear candidate bytes differ')
                embedded = read_json(directory/'metrics.json')
                require_equal(embedded, actual_arm['candidates'][selected_index], 'embedded selected candidate', 0)
                require_equal(embedded['validation'], candidate['validation'], 'accepted selected candidate summary', 1e-10)
                selected = [('uni_field_linear_dual', directory/'validation-predictions.jsonl', candidate, embedded)]
            for pipeline, path, accepted, saved in selected:
                for key in ('validation_predictions_sha256', 'validation'):
                    require_equal(saved.get(key), accepted[key], 'accepted '+pipeline+'/'+key, 1e-10)
                rows = read_bound_predictions(path, accepted['validation_predictions_sha256'], cases, validation, accepted['validation'])
                for row in rows:
                    key = (seed, trial, row['case_id'], tuple(row['magnifications']))
                    if key in joined[pipeline]: raise ValueError('Duplicate joined role/case/subset key')
                    joined[pipeline][key] = row
                receipts.append({'pipeline': pipeline, 'seed': seed, 'trial': trial,
                    'source_run_sha256': public_run['run_sha256'],
                    'validation_predictions_sha256': file_hash(path), 'records': len(rows)})
    keys = set(joined['identity'])
    if len(keys) != 1200 or any(set(value) != keys for value in joined.values()):
        raise ValueError('Five pipelines do not share the exact complete 1200-key join')
    for key in keys:
        original = joined['identity'][key]
        for pipeline in PIPELINES:
            row = joined[pipeline][key]
            require_equal([row['truth'], row['truth5']], [original['truth'], original['truth5']], 'cross-pipeline trusted truth', 0)
        row = joined['malignant_only'][key]
        for name in ('binary_score', 'binary_log_probabilities'):
            require_equal(row[name], original[name], 'exact malignant-only router identity', 0)
    return joined, receipts, role_membership


def event_flags(row):
    malignant = row['truth5'] > 0
    routed = row['binary_score'] >= .5
    error = row['prediction'] != row['truth5']
    route_error = routed != malignant
    subtype_error = malignant and routed and error
    if int(error) != int(route_error)+int(subtype_error):
        raise ValueError('Final errors do not decompose into routing and post-route typing')
    return {'final_error': error, 'routing_error': route_error,
        'malignant_false_negative': malignant and not routed, 'benign_false_positive': not malignant and routed,
        'subtype_error_after_correct_route': subtype_error, 'true_malignant': malignant,
        'conditional_type_error_on_true_malignant': malignant and 1+int(np.argmax(row['malignant_probabilities4'])) != row['truth5']}


def counts(rows):
    rows = list(rows); totals = Counter(); matrix = np.zeros((5, 5), dtype=int)
    per_key = defaultdict(lambda: [0, 0])
    for row in rows:
        flags = event_flags(row)
        totals.update({key: int(value) for key, value in flags.items()})
        matrix[row['truth5'], row['prediction']] += 1
        per_key[row['case_id']][0] += 1
        per_key[row['case_id']][1] += int(flags['final_error'])
    result = {'dependent_case_subset_events': len(rows), 'distinct_provider_keys': len({r['case_id'] for r in rows}),
        'final_correct': len(rows)-totals['final_error'], 'final_errors': totals['final_error'],
        'event_weighted_error_rate': totals['final_error']/len(rows) if rows else None,
        'equal_provider_key_mean_error_rate': float(np.mean([errors/exposures for exposures, errors in per_key.values()])) if per_key else None,
        'routing_errors': totals['routing_error'], 'malignant_false_negatives': totals['malignant_false_negative'],
        'benign_false_positives': totals['benign_false_positive'],
        'subtype_errors_after_correct_route': totals['subtype_error_after_correct_route'],
        'true_malignant_events': totals['true_malignant'],
        'conditional_type_errors_on_true_malignant': totals['conditional_type_error_on_true_malignant'],
        'confusion_matrix': matrix.tolist()}
    if result['final_errors'] != result['routing_errors']+result['subtype_errors_after_correct_route']:
        raise ValueError('Aggregate error accounting differs')
    return result


def class_counts(rows):
    rows = list(rows)
    return {name: counts(r for r in rows if r['truth5'] == label) for label, name in enumerate(FINAL5)}


def concentration(rows):
    per_key = defaultdict(lambda: [0, 0])
    for row in rows:
        per_key[row['case_id']][0] += 1
        per_key[row['case_id']][1] += int(event_flags(row)['final_error'])
    ordered = sorted((errors, exposures) for exposures, errors in per_key.values())[::-1]
    errors = sum(value[0] for value in ordered)
    return {'distinct_provider_keys': len(ordered), 'dependent_final_errors': errors,
        'ranking_policy': 'descending raw error-event burden; unequal validation exposures retained; not per-key failure propensity',
        'keys_with_no_observed_errors': sum(e == 0 for e, _ in ordered),
        'keys_wrong_in_every_observed_event': sum(e == n for e, n in ordered),
        'error_event_counts_descending': [e for e, _ in ordered],
        'exposures_in_same_error_sorted_order': [n for _, n in ordered],
        'top_key_error_concentration': {str(k): {'error_events': sum(e for e, _ in ordered[:k]),
            'fraction_of_errors': sum(e for e, _ in ordered[:k])/errors if errors else 0.}
            for k in (1, 3, 5, 10)}}


def correction_counts(identity, candidate, keys):
    total = Counter()
    for key in keys:
        old, new = identity[key], candidate[key]
        before, after = event_flags(old), event_flags(new)
        total['events'] += 1
        total['corrections'] += int(before['final_error'] and not after['final_error'])
        total['regressions'] += int(not before['final_error'] and after['final_error'])
        total['both_wrong'] += int(before['final_error'] and after['final_error'])
        total['both_correct'] += int(not before['final_error'] and not after['final_error'])
        total['changed_final_predictions'] += int(old['prediction'] != new['prediction'])
        total['changed_routes'] += int((old['binary_score'] >= .5) != (new['binary_score'] >= .5))
    result = dict(total)
    result['net_correct_change'] = total['corrections']-total['regressions']
    if sum(total[k] for k in ('corrections', 'regressions', 'both_wrong', 'both_correct')) != total['events']:
        raise ValueError('Paired correction/regression partition differs')
    expected = sum(int(candidate[k]['prediction'] == candidate[k]['truth5'])-int(identity[k]['prediction'] == identity[k]['truth5']) for k in keys)
    if result['net_correct_change'] != expected: raise ValueError('Paired net correction accounting differs')
    return result


def public_audit(joined):
    result = {'pipelines': {}, 'paired_against_identity': {}}
    for pipeline, indexed in joined.items():
        rows = list(indexed.values())
        value = {'all_subsets': counts(rows), 'all_four_only': counts(r for r in rows if tuple(r['magnifications']) == ALL_FOUR),
            'per_class_all_subsets': class_counts(rows), 'by_arrangement': {},
            'case_error_concentration': concentration(rows),
            'malignant_case_error_concentration': concentration(r for r in rows if r['truth5'] > 0)}
        for seed in (812, 813):
            selected = [r for key, r in indexed.items() if key[0] == seed]
            value['by_arrangement'][str(seed)] = {'all_subsets': counts(selected), 'per_class': class_counts(selected),
                'all_four_only': counts(r for r in selected if tuple(r['magnifications']) == ALL_FOUR),
                'all_four_per_class': class_counts(r for r in selected if tuple(r['magnifications']) == ALL_FOUR),
                'case_error_concentration': concentration(selected)}
        result['pipelines'][pipeline] = value
        if pipeline != 'identity':
            paired = {'all_subsets': correction_counts(joined['identity'], indexed, sorted(indexed)),
                      'all_four_only': correction_counts(joined['identity'], indexed, [k for k in sorted(indexed) if k[3] == ALL_FOUR]),
                      'by_arrangement': {}}
            for seed in (812, 813):
                keys = [k for k in sorted(indexed) if k[0] == seed]
                paired['by_arrangement'][str(seed)] = {
                    'all_subsets': correction_counts(joined['identity'], indexed, keys),
                    'per_class': {name: correction_counts(joined['identity'], indexed, [k for k in keys if indexed[k]['truth5'] == label])
                                  for label, name in enumerate(FINAL5)},
                    'all_four_only': correction_counts(joined['identity'], indexed, [k for k in keys if k[3] == ALL_FOUR]),
                    'all_four_per_class': {name: correction_counts(joined['identity'], indexed,
                        [k for k in keys if k[3] == ALL_FOUR and indexed[k]['truth5'] == label])
                        for label, name in enumerate(FINAL5)}}
            result['paired_against_identity'][pipeline] = paired
    return result


def private_case_audit(joined, manifest_rows):
    fields = defaultdict(lambda: defaultdict(list))
    for row in manifest_rows:
        fields[case_id(row)][str(row['magnification'])].append(row['relative_path'])
    keys = sorted({key[2] for key in joined['identity']})
    result = []
    for identity in keys:
        selected = {pipeline: [(key, row) for key, row in indexed.items() if key[2] == identity]
                    for pipeline, indexed in joined.items()}
        truth = selected['identity'][0][1]['truth5']
        entry = {'case_id': identity, 'truth5': truth, 'final_class': FINAL5[truth],
            'observed_validation_roles': sorted({(key[0], key[1]) for key, _ in selected['identity']}),
            'pipelines': {pipeline: {'all_subsets': counts(row for _, row in rows),
                'all_four_only': counts(row for key, row in rows if key[3] == ALL_FOUR)} for pipeline, rows in selected.items()},
            'inspection_index': {}}
        for magnification in ('40', '100', '200', '400'):
            paths = sorted(fields[identity][magnification])
            if not paths: raise ValueError('Validation case lacks a required original field view')
            positions = sorted({0, (len(paths)-1)//2, len(paths)-1})
            entry['inspection_index'][magnification] = {'original_field_count': len(paths),
                'all_original_manifest_paths': paths,
                'fixed_first_middle_last_paths': [paths[position] for position in positions]}
        result.append(entry)
    # Inspection order is descriptive and fixed: most identity errors first,
    # ties resolved by provider key; no model is selected by this order.
    return sorted(result, key=lambda row: (-row['pipelines']['identity']['all_subsets']['final_errors'], row['case_id']))


def audit(args):
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing error-audit output')
    context = source_context(args.code_sha)
    context['audit_script_sha256'] = file_hash(Path(__file__))
    trusted = load_trusted_context(args.manifest, args.protocol)
    if len(trusted['cases']) != 82: raise ValueError('Expected all82 qualified provider keys')
    metric = bound_comparison(args.metric_comparison, args.metric_comparison_sha256, trusted, 'metric')
    linear = bound_comparison(args.linear_comparison, args.linear_comparison_sha256, trusted, 'linear')
    for key in ('encoder', 'cache_provenance'):
        require_equal(metric[key], linear[key], 'same frozen UNI2-h input/'+key, 0)
    joined, receipts, role_membership = load_joined(metric, linear, args.metric_runs, args.linear_runs, trusted)
    details = public_audit(joined)
    if details['pipelines']['identity']['all_subsets']['distinct_provider_keys'] != 31:
        raise ValueError('Expected31 distinct reused validation provider keys')
    provenance = dict(context, metric_comparison_sha256=args.metric_comparison_sha256,
        linear_comparison_sha256=args.linear_comparison_sha256,
        metric_producer={key: metric[key] for key in ('code_sha', 'source_archive_sha256', 'config_sha256')},
        linear_producer={key: linear[key] for key in ('code_sha', 'source_archive_sha256', 'config_sha256')},
        manifest_sha256=trusted['manifest_sha256'], protocol_sha256=trusted['protocol_sha256'])
    public = dict(provenance, status='complete descriptive selected-validation specialist error audit',
        class_names=list(FINAL5), pipelines=list(PIPELINES), verified_original_roles=10,
        verified_selected_prediction_archives=50, verified_records_per_pipeline=1200,
        verified_dependent_records=6000, distinct_validation_provider_keys=31,
        new_fits=0, new_encoder_forwards=0, new_predictions=0, outer_predictions=0,
        selection_opportunities_added=0, inspection_images_copied=0,
        provider_key_average_scope='equal_provider_key_mean_error_rate averages each distinct provider key\'s observed error-events/exposure-events equally within the reported scope; descriptive repeated-validation average, not a majority-vote outcome or fresh independent-case score',
        interpretation='post-hoc descriptive audit of reused validation; dependent roles and magnification subsets; provider identity is not independently verified clinical identity; no morphological or label-defect conclusion',
        source_archive_receipts=receipts, summary=details)
    assert_public_safe(public)
    private = dict(provenance, status='complete private provider-case error and original-field inspection index',
        role_membership=[{'seed': seed, 'trial': trial, **ids} for (seed, trial), ids in sorted(role_membership.items())],
        inspection_policy='all original paths retained; fixed first/middle/last lexicographic field paths per view; no image decoding/copying or morphology interpretation',
        cases=private_case_audit(joined, read_manifest(args.manifest)))
    output.mkdir(parents=True)
    write_json(output/'private.json', private)
    public['private_audit_sha256'] = file_hash(output/'private.json')
    write_json(output/'public.json', public)
    return public


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('metric-comparison', 'metric-runs', 'linear-comparison', 'linear-runs', 'manifest', 'protocol', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--metric-comparison-sha256', required=True)
    p.add_argument('--linear-comparison-sha256', required=True)
    p.add_argument('--code-sha', required=True)
    return p


if __name__ == '__main__': audit(parser().parse_args())

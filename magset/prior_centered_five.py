"""Paired supervised class-prior centering ablation on accepted coverage features."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import coverage_five as source, kernel_five as kernel, midnight_five as base
from .bags import fit_normalization
from .baseline import SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .field_five import pool_log_probabilities
from .five_collect import read_json, require_equal, verify_audit as generic_verify_audit
from .five_metrics import FINAL5, MAP8_TO5, summarize_records

MODES = ('class_shared', 'head_specific')
ARMS = kernel.ARMS
TEMPERATURES = kernel.TEMPERATURES
FITTING_POLICY = 'leave entire query case out before centers, weights and support counts; query truth does not determine centers'
COUNTS = dict(production_run_pairs=10, logical_candidate_pipelines=160,
              structurally_unique_candidate_definitions=120, logical_prediction_archives=320,
              structurally_nonduplicated_prediction_archives=240, logical_case_subset_records=129600,
              structurally_nonduplicated_candidate_records=97200, logical_full_fit_head_definitions=60,
              structurally_unique_full_fit_head_definitions=50)


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    if file_hash(args.config) != file_hash(root/'configs/breakhis-prior-centered-v1.json'):
        raise ValueError('Prior-centering config differs from source registration')
    own = read_json(args.config)
    expected = dict(COUNTS, name='breakhis-prior-centered-v1', schema_version=1,
                    modes=list(MODES), arms=list(ARMS), temperatures=TEMPERATURES,
                    new_optimized_coefficients=0, new_encoder_forwards=0,
                    fitting_labels_used_for_centers=True, fitting_diagnostic=FITTING_POLICY,
                    selection_opportunities_per_named_mode_arm=4,
                    selection_opportunities_if_choosing_between_modes_per_arm=8,
                    structurally_unique_family_selection_opportunities={'direct': 4, 'dual': 8},
                    routing_threshold=.5, routing_tie='malignant')
    for key, value in expected.items(): require_equal(own.get(key), value, 'registered centering/'+key, 0)
    from .prior_centered_model import NORMALIZATION
    require_equal(own.get('normalization'), NORMALIZATION, 'registered centering/normalization', 0)
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']:
        raise ValueError('Accepted coverage configuration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path
    inherited, context, trusted, cases, bags, z, y5, roles = source.load_inputs(forwarded)
    context = dict(context, config_sha256=file_hash(args.config),
                   prior_centered_config_sha256=file_hash(args.config),
                   prior_centered_base_config_sha256=own['base_input_config_sha256'])
    context['implementation_hashes'] = dict(context['implementation_hashes'])
    for name in ('prior_centered_model.py', 'prior_centered_five.py', 'prior_centered_collect.py'):
        context['implementation_hashes'][name] = file_hash(root/'magset'/name)
    return {**inherited, **own}, context, trusted, cases, bags, z, y5, roles


def center_checks(bags, y5, roles):
    from .prior_centered_model import support_summary
    return [{'seed': s, 'trial': t, 'support_summary': support_summary(bags, y5, train)}
            for (s, t), (train, _, _) in sorted(roles.items())]


def verify_input_audit(audit, context, config, bags, y5, roles):
    generic_verify_audit(audit, context, config, bags, y5, roles)
    require_equal(audit.get('head_center_checks'), center_checks(bags, y5, roles), 'audit/head centers', 0)


def audit_input(args):
    config, context, _, _, bags, _, y5, roles = load_inputs(args)
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing prior-centering audit output')
    checks = []
    for (s, t), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append(dict(seed=s, trial=t, role_counts=[len(train), len(val), len(excluded)],
                           fitting_support=np.bincount(y5[train], minlength=5).tolist(),
                           validation_support=np.bincount(y5[val], minlength=5).tolist(),
                           normalization_sha256=hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()))
    audit = dict(context, status='zero-fit inputs accepted', new_fits=0, new_forwards=0,
                 outer_predictions=0, checks=checks, head_center_checks=center_checks(bags, y5, roles))
    verify_input_audit(audit, context, config, bags, y5, roles)
    output.mkdir(parents=True); base.write_json(output/'audit.json', audit)


def prepare_queries(bags, y5, train, validation):
    from .prior_centered_model import prepare_case
    return {role: [prepare_case(bags, y5, train, int(i), leave_query_case_out=(role == 'train')) for i in indices]
            for role, indices in [('train', train), ('validation', validation)]}


def score_records(prepared, cases, role, arm, temperature, mode, caches):
    from .prior_centered_model import field_log_probabilities
    if role not in ('train', 'validation') or arm not in ARMS or mode not in MODES or temperature not in TEMPERATURES:
        raise ValueError('Only registered roles, modes, arms and temperatures may be scored')
    if len(prepared) != len(caches): raise ValueError('Query/cache coverage differs')
    records = []
    for query, cache in zip(prepared, caches):
        fields = field_log_probabilities(query, temperature, arm, mode, cache=cache)
        original = int(cases[query.query_index]['label'])
        for subset in SUBSETS:
            row = dict(role=role, case_id=cases[query.query_index]['case_id'], truth=original,
                       truth5=int(MAP8_TO5[original]), magnifications=[MAGNIFICATIONS[k] for k in subset])
            values = pool_log_probabilities(fields, subset, arm)
            row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()})
            records.append(row)
    return records


def model_record(context, cases, train, support, arm, temperature, mode):
    mapping = support['mode_used_center_mapping']
    heads = ['direct'] if arm == 'direct' else ['binary', 'malignant']
    return dict(context, schema_version=1, method='supervised class-prior centered cosine KDE',
                arm=arm, mode=mode, temperature=temperature, support_indices=list(map(int, train)),
                support_case_ids=[cases[i]['case_id'] for i in train], support_summary=support,
                used_centers_sha256={h: support['centers_sha256'][mapping[h]] for h in heads},
                normalization=support['normalization'], fitting_evaluation=FITTING_POLICY,
                fitting_labels_used_for_centers=True)


def run(args):
    from .prior_centered_model import density_cache, support_summary
    started = time.monotonic(); output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing prior-centering run output')
    config, context, _, cases, bags, _, y5, roles = load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    verify_input_audit(read_json(audit_path), context, config, bags, y5, roles)
    train, val, excluded = roles[args.seed, args.trial]
    prepared = prepare_queries(bags, y5, train, val)
    supports = {mode: support_summary(bags, y5, train, mode=mode) for mode in MODES}
    output.mkdir(parents=True); modes = {m: {'arms': {a: {'candidates': []} for a in ARMS}} for m in MODES}
    for index, temperature in enumerate(TEMPERATURES):
        caches = {r: [density_cache(q, temperature) for q in queries] for r, queries in prepared.items()}
        for mode in MODES:
            for arm in ARMS:
                begin = time.monotonic(); folder = output/mode/arm/('candidate%d'%index); folder.mkdir(parents=True)
                base.write_json(folder/'model.json', model_record(context, cases, train, supports[mode], arm, temperature, mode))
                summaries, hashes = {}, {}
                for role in ('train', 'validation'):
                    records = score_records(prepared[role], cases, role, arm, temperature, mode, caches[role])
                    summaries[role] = summarize_records(records, arm); path = folder/(role+'-predictions.jsonl')
                    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
                    hashes[role+'_predictions_sha256'] = file_hash(path)
                candidate = dict(index=index, temperature=temperature, mode=mode, arm=arm,
                                 optimized_head_fits=0, model_sha256=file_hash(folder/'model.json'),
                                 fitting_evaluation=FITTING_POLICY, fitting_leave_case_out=summaries['train'],
                                 validation=summaries['validation'], elapsed_seconds=time.monotonic()-begin, **hashes)
                base.write_json(folder/'metrics.json', candidate); modes[mode]['arms'][arm]['candidates'].append(candidate)
    for mode in MODES:
        for arm in ARMS:
            saved = modes[mode]['arms'][arm]; cs = saved['candidates']
            index = max(range(4), key=lambda i: cs[i]['validation']['mean_subset_macro_f1'])
            saved.update(selected_index=index, selected_temperature=TEMPERATURES[index],
                         fitting_leave_case_out=cs[index]['fitting_leave_case_out'], validation=cs[index]['validation'])
    report = dict(context, status='complete paired prior-centered KDE validation run', seed=args.seed, trial=args.trial,
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
                  case_ids={r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]},
                  role_counts=[len(train), len(val), len(excluded)], optimized_head_fits=0, new_encoder_forwards=0,
                  outer_predictions=0, fitting_evaluation=FITTING_POLICY, fitting_labels_used_for_centers=True,
                  logical_full_fit_head_definitions=6, structurally_unique_full_fit_head_definitions=5,
                  modes=modes, audit_sha256=file_hash(audit_path), elapsed_seconds=time.monotonic()-started,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    base.write_json(output/'run.json', report)


def parser():
    return source.parser()


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'run':
        if args.audit is None or args.seed is None or args.trial is None: raise ValueError('Run needs audited seed/trial')
        run(args)
    else:
        from .prior_centered_collect import collect
        collect(args)

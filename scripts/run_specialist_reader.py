"""Run the registered frozen-specialist validation family and sealed follow-up."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import matched_data as md
from magset import prior_centered_model as parent_model
from magset import specialist_reader as core
from magset.bags import load_bags
from magset.data import MAGNIFICATIONS, file_hash

REPO = Path(__file__).resolve().parents[1]
ARMS = ('p_control', 'uni2_specialist')
CORE_ARMS = {'p_control': 'p_control', 'uni2_specialist': 'uni2_c'}


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()).hexdigest()


def registered_config(path):
    fixed = REPO / 'configs/breakhis-specialist-reader-v1.json'
    require(file_hash(path) == file_hash(fixed), 'Use the exact registered specialist configuration')
    cfg = read_json(path)
    require(cfg['arms'] == list(ARMS) and cfg['primary_arm'] == 'uni2_specialist'
            and tuple(cfg['density']['malignant_temperatures']) == core.TEMPERATURES
            and cfg['router']['temperature'] == core.ROUTER_TEMPERATURE,
            'Registered specialist family differs')
    return cfg


def source_context(code_sha, config_path):
    require(re.fullmatch('[0-9a-f]{40}', code_sha or '') is not None
            and (REPO / '.source-sha').read_text().strip() == code_sha, 'Immutable specialist source required')
    archive = (REPO / '.archive-sha256').read_text().strip()
    require(re.fullmatch('[0-9a-f]{64}', archive) is not None, 'Immutable source archive digest required')
    files = ('scripts/run_specialist_reader.py', 'magset/specialist_reader.py', 'magset/matched_data.py',
             'magset/prior_centered_model.py', 'magset/kernel_model.py', 'magset/bags.py',
             'magset/data.py', 'magset/five_metrics.py')
    return {'code_sha': code_sha, 'source_archive_sha256': archive, 'config_sha256': file_hash(config_path),
            'implementation_sha256': {name: file_hash(REPO / name) for name in files}}


def verify_references(cfg):
    refs = cfg['references']; hashes = {}
    for name in ('p_feature_config', 'p_readout_config', 'uni_feature_config', 'uni_extraction_config'):
        path = REPO / refs[name]
        require(file_hash(path) == refs[name + '_sha256'], 'Registered reference configuration changed: ' + name)
        hashes[refs[name]] = file_hash(path)
    for section, name, digest in ((cfg['matched_reference'], 'file', 'sha256'),
                                  (cfg['matched_reference'], 'validation_file', 'validation_sha256'),
                                  (refs, 'uni_previous_result', 'uni_previous_result_sha256'),
                                  (cfg['input_audit'], 'public_file', 'public_sha256')):
        require(file_hash(REPO / section[name]) == section[digest], 'Registered reference evidence changed')
    matched = REPO / cfg['matched_reference']['config']
    require(file_hash(matched) == cfg['matched_reference']['config_sha256'], 'Parent matched configuration changed')
    return hashes


def load_cache(cache, feature_cfg, extraction, fields, cases, manifest_sha, dimension, components):
    """Validate raw cache paths first; load by canonical path, never cache position."""
    cache = Path(cache); source = {row['relative_path']: row for row in fields}; seen = set()
    require(len(source) == len(fields), 'Canonical manifest paths duplicate')
    for mag in MAGNIFICATIONS:
        stem = cache / ('%s-%d' % (feature_cfg['encoder_name'], mag))
        npz = stem.with_suffix('.npz'); meta_path = stem.with_suffix('.json')
        pin = feature_cfg['cache_provenance'][str(mag)]
        required = [(npz, 'npz_sha256'), (meta_path, 'json_sha256')]
        if components:
            required.append((cache / (stem.name + '-components.npz'), 'components_sha256'))
        require(all(file_hash(path) == pin[key] for path, key in required), 'Frozen feature cache bytes changed')
        meta = read_json(meta_path)
        require(meta['encoder'] == extraction['encoder'] and meta['manifest_sha256'] == manifest_sha
                and meta['feature_sha256'] == pin['npz_sha256'] and meta['images'] == pin['images']
                and meta['smoke_limit'] == 0 and meta['magnification'] == mag
                and meta['code_sha'] == feature_cfg['feature_producer_code_sha'], 'Feature extraction lineage differs')
        if components:
            expected_config = feature_cfg['coverage_extraction_config_sha256']
            require(meta['components_sha256'] == pin['components_sha256'], 'Coverage components lineage differs')
        else:
            expected_config = feature_cfg['extraction_config_sha256']
        require(meta['config_sha256'] == expected_config, 'Feature extraction configuration differs')
        with np.load(npz, allow_pickle=False) as saved:
            paths = saved['paths'].tolist()
        wanted = {f['relative_path'] for f in fields if int(f['magnification']) == mag}
        require(len(paths) == len(set(paths)) and set(paths) == wanted and not seen.intersection(paths),
                'Feature path/view coverage differs')
        seen.update(paths)
    require(seen == set(source), 'Incomplete canonical cache field bijection')
    ordered = [source[path] for path in sorted(seen)]
    loaded_cases, bags, _ = load_bags(ordered, cache, feature_cfg['encoder_name'], manifest_sha)
    require(len(loaded_cases) == len(cases), 'Cached case count differs')
    for i, (cached, audited) in enumerate(zip(loaded_cases, cases)):
        require(cached['case_id'] == audited['case_id'] and cached['label'] == audited['truth8']
                and cached['stem_group'] == audited['group'], 'Cached provider/label/group order differs')
        for view in range(4):
            count = sum(f['case_index'] == i and f['view'] == view for f in fields)
            require(bags[i][view].shape == (count, dimension), 'Cached field dimensions/order coverage differs')
    return bags, ordered


def load_inputs(args):
    cfg = registered_config(args.config); refs = verify_references(cfg)
    versions = {name: importlib.metadata.version(name) for name in cfg['runtime']}
    require(versions == cfg['runtime'], 'Pinned specialist CPU runtime differs')
    context = source_context(args.code_sha, args.config)
    parent = Path(args.parent)
    require(file_hash(parent / 'checked/result.json') == cfg['matched_reference']['sha256'], 'Accepted parent result changed')
    accepted = read_json(parent / 'checked/result.json')
    require(accepted['status'] == 'complete accepted matched three-way comparison'
            and accepted['code_sha'] == cfg['matched_reference']['producer_code_sha'], 'Parent comparison is not accepted')
    audit_dir = parent / 'input-audit'
    require(file_hash(audit_dir / 'private.json') == cfg['input_audit']['private_sha256']
            and file_hash(audit_dir / 'indices.npz') == cfg['input_audit']['indices_sha256']
            and file_hash(audit_dir / 'public.json') == cfg['input_audit']['public_sha256'], 'Parent input audit changed')
    audit = md.load_audit(audit_dir, REPO / cfg['matched_reference']['config'])
    require(audit['manifest_sha256'] == cfg['manifest_sha256'], 'Parent canonical manifest differs')
    for role, counts in cfg['expected_roles'].items():
        require(all(audit['role_summary'][role][name] == value for name, value in counts.items()), 'Registered role counts differ')
    p_selection = parent / 'models/midnight_p/selection.json'
    require(file_hash(p_selection) == accepted['models']['midnight_p']['selection_sha256'], 'Parent P selection changed')
    selected = read_json(p_selection)
    require(selected['selected_setting'] == .1 and selected['support_case_indices'] == audit['roles']['train']
            and selected['input_audit_sha256'] == cfg['input_audit']['private_sha256'], 'Parent router support/setting differs')
    p_cfg = read_json(REPO / cfg['references']['p_feature_config'])
    uni_cfg = read_json(REPO / cfg['references']['uni_feature_config'])
    p_extract_path = REPO / p_cfg['coverage_extraction_config']
    require(file_hash(p_extract_path) == p_cfg['coverage_extraction_config_sha256'], 'P extraction registration changed')
    p_extract = read_json(p_extract_path); uni_extract = read_json(REPO / cfg['references']['uni_extraction_config'])
    p_bags, p_rows = load_cache(args.p_cache, p_cfg, p_extract, audit['fields'], audit['cases'], cfg['manifest_sha256'], 3072, True)
    uni_bags, uni_rows = load_cache(args.uni_cache, uni_cfg, uni_extract, audit['fields'], audit['cases'], cfg['manifest_sha256'], 1536, False)
    require(core.assert_field_alignment(p_rows, uni_rows) == len(audit['fields']), 'Feature alignment incomplete')
    labels = np.asarray([case['truth5'] for case in audit['cases']], dtype=np.int64)
    fit = list(audit['roles']['train']); malignant = [i for i in fit if labels[i] > 0]
    alignment = [(f['relative_path'], f['sha256'], audit['cases'][f['case_index']]['case_id'],
                  audit['cases'][f['case_index']]['group'], int(f['magnification']),
                  audit['cases'][f['case_index']]['truth8']) for f in p_rows]
    context.update(parent_result_sha256=cfg['matched_reference']['sha256'],
                   parent_config_sha256=cfg['matched_reference']['config_sha256'],
                   input_audit_sha256=cfg['input_audit']['private_sha256'], manifest_sha256=cfg['manifest_sha256'],
                   reference_config_sha256=refs,
                   cache_provenance={'p': p_cfg['cache_provenance'], 'uni': uni_cfg['cache_provenance']},
                   p_cache=str(Path(args.p_cache).resolve()), uni_cache=str(Path(args.uni_cache).resolve()),
                   field_alignment_sha256=canonical_hash(alignment), support_case_indices=fit,
                   malignant_support_case_indices=malignant, expected_counts=cfg['counts'], runtime_versions=versions)
    uni_support = core.prepare_support(uni_bags, labels, fit)
    summary = {'p_control': parent_model.support_summary(p_bags, labels, fit, mode='head_specific'),
               'uni2_specialist': core.support_summary(uni_support)}
    return dict(config=cfg, context=context, audit=audit, parent=parent, parent_selection=selected,
                parent_result=accepted, p_bags=p_bags, uni_bags=uni_bags, labels=labels,
                uni_support=uni_support, support_summary=summary)


def parent_archive(state, role):
    parent = state['parent']; selected = state['parent_selection']; cfg = state['config']
    if role == 'validation':
        folder = parent / 'models/midnight_p/candidates' / ('candidate%d' % selected['selected_index'])
        path = folder / 'validation-fields.npz'
        expected_sha = selected['candidate_artifact_sha256'][selected['selected_index']]['validation-fields.npz']
    elif role == 'test':
        folder = parent / 'models/midnight_p/test'; receipt = read_json(folder / 'test.json')
        require(receipt['status'] == 'complete selected test inference'
                and receipt['selection_sha256'] == state['parent_result']['models']['midnight_p']['selection_sha256']
                and receipt['code_sha'] == cfg['matched_reference']['producer_code_sha']
                and receipt['input_audit_sha256'] == cfg['input_audit']['private_sha256'], 'Canonical parent follow-up context differs')
        path = folder / 'fields.npz'; expected_sha = receipt['artifact_sha256']['fields.npz']
    else:
        raise ValueError('Only validation and retrospective test references are allowed')
    require(file_hash(path) == expected_sha, 'Canonical parent probability archive changed')
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'field_indices', 'joint_probabilities5'}, 'Parent archive schema differs')
        ix = saved['field_indices'].copy(); joint = saved['joint_probabilities5'].copy()
    expected = np.array([f['index'] for f in state['audit']['fields'] if f['case_index'] in set(state['audit']['roles'][role])], np.int64)
    require(ix.dtype == np.int64 and joint.dtype == np.float64 and np.array_equal(ix, expected)
            and joint.shape == (len(ix), 5) and np.isfinite(joint).all() and np.all(joint > 0)
            and np.allclose(joint.sum(1), 1., rtol=0, atol=1e-12), 'Parent field coverage/probabilities differ')
    return ix, joint, path


def role_predictions(state, role, settings, parent_indices, parent_joint):
    """Eight validation pipelines or exactly two sealed selected follow-up pipelines."""
    audit = state['audit']; fields = audit['fields']; indices = parent_indices
    lookup = {int(i): n for n, i in enumerate(indices)}
    outputs = {arm: [np.full((len(indices), 5), np.nan, np.float64) for _ in temperatures]
               for arm, temperatures in settings.items()}
    for query in sorted(audit['roles'][role]):
        rows = [sorted((f for f in fields if f['case_index'] == query and f['view'] == view),
                       key=lambda f: f['relative_path']) for view in range(4)]
        reference = tuple(np.log(parent_joint[[lookup[f['index']] for f in view]]) for view in rows)
        binary, _ = core.decompose_joint_logs(reference)
        kwargs = {'parent_reference_joint_logs': reference} if role == 'test' else {}
        prepared = core.prepare_query(state['p_bags'], state['uni_bags'], state['labels'],
                                      audit['roles']['train'], query, state['uni_support'], **kwargs)
        for arm, temperatures in settings.items():
            for candidate, temperature in enumerate(temperatures):
                probabilities = core.field_probabilities(prepared, temperature, CORE_ARMS[arm], fixed_binary_logs=binary)
                for view, values in enumerate(probabilities):
                    outputs[arm][candidate][[lookup[f['index']] for f in rows[view]]] = values
    require(all(np.isfinite(values).all() for candidates in outputs.values() for values in candidates), 'Incomplete specialist role fields')
    return outputs


def records(state, role, indices, joint):
    audit = state['audit']
    return md.case_records(audit['fields'], audit['cases'], audit['roles'][role], indices, joint, role)


def binary_checks(reference_joint, candidate_joint, reference_records, candidate_records):
    marginal = lambda q: np.column_stack((q[:, 0], q[:, 1:].sum(1)))
    delta = float(np.max(np.abs(marginal(reference_joint) - marginal(candidate_joint))))
    by = {(r['case_id'], tuple(r['magnifications'])): r for r in reference_records}
    require(len(candidate_records) == len(by), 'Case/subset binary coverage differs')
    case_delta = 0.; disagreements = 0
    for r in candidate_records:
        old = by[(r['case_id'], tuple(r['magnifications']))]
        case_delta = max(case_delta, abs(r['binary_score'] - old['binary_score']))
        disagreements += int((r['prediction'] > 0) != (old['prediction'] > 0))
    require(delta <= 1e-12 and case_delta <= 1e-12 and disagreements == 0, 'Fixed parent binary marginals/decisions changed')
    return dict(maximum_field_binary_abs=delta, maximum_case_binary_abs=case_delta, binary_decision_disagreements=disagreements)


def save_prediction(path, indices, joint):
    md.save_predictions(path, indices, joint)
    with np.load(path, allow_pickle=False) as saved:
        require(saved['field_indices'].dtype == np.int64 and saved['joint_probabilities5'].dtype == np.float64
                and np.array_equal(saved['field_indices'], indices)
                and np.array_equal(saved['joint_probabilities5'], joint), 'Specialist archive save/reload changed values')


def verify_gate(output):
    output = Path(output); gate = read_json(output / 'family-gate.json')
    context = read_json(output / 'context.json')
    require(gate['status'] == 'sealed both specialist selections' and gate['test_queries'] == 0
            and gate['context_sha256'] == file_hash(output / 'context.json')
            and gate['support_summary_sha256'] == file_hash(output / 'support-summary.json')
            and gate['input_audit_sha256'] == context['input_audit_sha256'], 'Specialist gate context changed')
    require(set(gate['selections']) == set(ARMS), 'Both specialist selections must be sealed')
    for arm in ARMS:
        folder = output / 'arms' / arm; path = folder / 'selection.json'
        require(file_hash(path) == gate['selections'][arm], 'Specialist selection changed after sealing')
        selection = read_json(path); scores = selection['candidate_validation_scores']
        require(selection['status'] == 'selected validation candidate' and selection['arm'] == arm
                and selection['test_predictions'] == 0 and len(scores) == 4 and np.isfinite(scores).all()
                and type(selection['selected_index']) is int and selection['selected_index'] == int(np.argmax(scores))
                and selection['selected_setting'] == core.TEMPERATURES[selection['selected_index']]
                and selection['context_sha256'] == gate['context_sha256']
                and selection['support_summary_sha256'] == gate['support_summary_sha256'], 'Sealed validation selection differs')
        require(all(selection[key] == context[key] for key in
                    ('code_sha', 'source_archive_sha256', 'config_sha256', 'parent_result_sha256', 'input_audit_sha256')),
                'Selection source/input context differs')
        require(len(selection['candidate_artifact_sha256']) == 4, 'All four candidate artifact receipts required')
        for index, receipt in enumerate(selection['candidate_artifact_sha256']):
            require(set(receipt) == {'validation-fields.npz', 'validation-records.jsonl', 'metrics.json'}, 'Candidate artifact receipt differs')
            candidate = folder / 'candidates' / ('candidate%d' % index)
            require(all(file_hash(candidate / name) == digest for name, digest in receipt.items()), 'Candidate artifacts changed after sealing')
            metric = read_json(candidate / 'metrics.json')
            require(metric['setting'] == core.TEMPERATURES[index]
                    and metric['validation']['mean_subset_macro_f1'] == scores[index], 'Sealed candidate score/setting differs')
    return gate


def run(args):
    output = Path(args.output)
    require(not output.exists(), 'Refusing to overwrite a specialist study')
    state = load_inputs(args); output.mkdir(parents=True)
    context = state['context']; cfg = state['config']
    md.write_json(output / 'context.json', context); md.write_json(output / 'support-summary.json', state['support_summary'])
    context_sha = file_hash(output / 'context.json'); support_sha = file_hash(output / 'support-summary.json')
    indices, reference, reference_path = parent_archive(state, 'validation')
    reference_records = records(state, 'validation', indices, reference)
    settings = {arm: core.TEMPERATURES for arm in ARMS}
    probabilities = role_predictions(state, 'validation', settings, indices, reference)
    identity = probabilities['p_control'][2]; identity_records = records(state, 'validation', indices, identity)
    field_delta = float(np.max(np.abs(identity - reference)))
    case_delta = max(float(np.max(np.abs(np.asarray(a['probabilities5']) - np.asarray(b['probabilities5']))))
                     for a, b in zip(identity_records, reference_records))
    require(field_delta <= cfg['identity']['field_probability_atol'] and case_delta <= cfg['identity']['case_probability_atol']
            and [r['prediction'] for r in identity_records] == [r['prediction'] for r in reference_records], 'Parent P validation identity failed')
    identity_folder = output / 'parent-identity'; identity_folder.mkdir()
    save_prediction(identity_folder / 'validation-fields.npz', indices, identity)
    md.write_json(identity_folder / 'validation.json', dict(status='accepted rederived parent P validation identity',
        parent_archive_sha256=file_hash(reference_path), maximum_field_probability_abs=field_delta,
        maximum_case_probability_abs=case_delta, decision_disagreements=0, test_queries=0))
    selections = {}; validation_count = 0
    for arm in ARMS:
        scores = []; hashes = []
        for candidate, (temperature, joint) in enumerate(zip(core.TEMPERATURES, probabilities[arm])):
            folder = output / 'arms' / arm / 'candidates' / ('candidate%d' % candidate); folder.mkdir(parents=True)
            rows = records(state, 'validation', indices, joint); checks = binary_checks(reference, joint, reference_records, rows)
            metric = md.summary(rows); save_prediction(folder / 'validation-fields.npz', indices, joint)
            md.save_records(folder / 'validation-records.jsonl', rows)
            md.write_json(folder / 'metrics.json', {'setting': temperature, 'validation': metric, 'binary_invariance': checks})
            scores.append(float(metric['mean_subset_macro_f1'])); validation_count += len(rows)
            hashes.append({name: file_hash(folder / name) for name in ('validation-fields.npz', 'validation-records.jsonl', 'metrics.json')})
        index = int(np.argmax(scores)); require(np.isfinite(scores).all(), 'Nonfinite specialist validation score')
        selected = dict(context, status='selected validation candidate', arm=arm, context_sha256=context_sha,
                        support_summary_sha256=support_sha, selected_index=index, selected_setting=core.TEMPERATURES[index],
                        candidate_validation_scores=scores, candidate_artifact_sha256=hashes, test_predictions=0,
                        validation_identity_sha256=file_hash(identity_folder / 'validation.json'))
        md.write_json(output / 'arms' / arm / 'selection.json', selected); selections[arm] = selected
    require(validation_count == cfg['counts']['validation_case_subset_records'], 'Validation record budget differs')
    gate = dict(status='sealed both specialist selections', context_sha256=context_sha, support_summary_sha256=support_sha,
                input_audit_sha256=context['input_audit_sha256'], test_queries=0,
                selections={arm: file_hash(output / 'arms' / arm / 'selection.json') for arm in ARMS})
    md.write_json(output / 'family-gate.json', gate); verify_gate(output)
    # Canonical parent test probabilities are opened only after both seals. They
    # supply the fixed router/reference, never a newly evaluated T=.1 specialist.
    indices, reference, reference_path = parent_archive(state, 'test')
    reference_records = records(state, 'test', indices, reference)
    shutil.copyfile(reference_path, identity_folder / 'test-fields.npz')
    md.write_json(identity_folder / 'test.json', dict(status='canonical parent router reference only',
        role='test', parent_archive_sha256=file_hash(reference_path), copied_archive_sha256=file_hash(identity_folder / 'test-fields.npz'),
        new_p_specialist_queries=0, selection_gate_sha256=file_hash(output / 'family-gate.json')))
    selected_settings = {arm: (selections[arm]['selected_setting'],) for arm in ARMS}
    verify_gate(output)
    probabilities = role_predictions(state, 'test', selected_settings, indices, reference)
    test_count = 0
    for arm in ARMS:
        verify_gate(output)
        folder = output / 'arms' / arm / 'test'; folder.mkdir()
        joint = probabilities[arm][0]; rows = records(state, 'test', indices, joint)
        checks = binary_checks(reference, joint, reference_records, rows); metric = md.summary(rows)
        save_prediction(folder / 'fields.npz', indices, joint); md.save_records(folder / 'records.jsonl', rows)
        md.write_json(folder / 'metrics.json', metric)
        md.write_json(folder / 'test.json', dict(status='complete selected retrospective inference', arm=arm,
            selected_index=selections[arm]['selected_index'], selected_setting=selections[arm]['selected_setting'],
            context_sha256=context_sha, support_summary_sha256=support_sha,
            selection_sha256=file_hash(output / 'arms' / arm / 'selection.json'), gate_sha256=file_hash(output / 'family-gate.json'),
            input_audit_sha256=context['input_audit_sha256'], binary_invariance=checks,
            artifact_sha256={name: file_hash(folder / name) for name in ('fields.npz', 'records.jsonl', 'metrics.json')}))
        test_count += len(rows)
    require(test_count == cfg['counts']['retrospective_case_subset_records'], 'Selected follow-up record budget differs')
    result = dict(status='complete specialist reader family', context_sha256=context_sha,
                  support_summary_sha256=support_sha, gate_sha256=file_hash(output / 'family-gate.json'),
                  validation_case_subset_records=validation_count, retrospective_case_subset_records=test_count,
                  validation_prediction_archives=8, selected_retrospective_prediction_archives=2,
                  new_encoder_forwards=0, optimized_head_fits=0, new_image_decodes=0,
                  artifact_sha256={str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob('*'))
                                   if path.is_file() and path.name != 'result.json'})
    md.write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'p-cache', 'uni-cache', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-sha', required=True); args = parser.parse_args()
    existed = Path(args.output).exists()
    try:
        value = run(args)
    except Exception as error:
        if not existed and Path(args.output).is_dir() and not (Path(args.output) / 'result.json').exists():
            md.write_json(Path(args.output) / 'failure.json', {'status': 'failed specialist reader execution',
                'error_type': type(error).__name__, 'message': str(error)})
        raise
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()

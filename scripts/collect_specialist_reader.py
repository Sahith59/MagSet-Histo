"""Independently check the frozen specialist family; no fitting or encoder calls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.five_class import write_json
from magset.five_collect import assert_public_safe, read_json, require_equal
from scripts import collect_matched_three_way as metrics

ARMS = ('p_control', 'uni2_specialist')
TEMPERATURES = (.01, .03, .1, .3)
CONTEXT_KEYS = ('code_sha', 'source_archive_sha256', 'config_sha256',
                'parent_result_sha256', 'input_audit_sha256')
CANDIDATE_FILES = {'validation-fields.npz', 'validation-records.jsonl', 'metrics.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_digest(arrays):
    return hashlib.sha256(b''.join(np.asarray(a, dtype='<f8').tobytes(order='C') for a in arrays)).hexdigest()


def alignment_digest(audit):
    rows = []
    for field in audit['fields']:
        case = audit['cases'][field['case_index']]
        rows.append([field['relative_path'], field['sha256'], case['case_id'], case['group'],
                     int(field['magnification']), int(case['truth8'])])
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def load_field_archive(path, audit, role):
    owners = set(audit['roles'][role])
    wanted = np.asarray([f['index'] for f in audit['fields'] if f['case_index'] in owners], dtype=np.int64)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'field_indices', 'joint_probabilities5'}, 'Field archive schema differs')
        indices, joint = saved['field_indices'].copy(), saved['joint_probabilities5'].copy()
    require(indices.dtype == np.dtype('int64') and np.array_equal(indices, wanted), 'Field archive order/membership differs')
    require(joint.dtype == np.dtype('float64'), 'Joint probabilities must be float64')
    joint = metrics.probabilities(joint, (len(wanted), 5), 'specialist joint probabilities', tolerance=1e-12)
    return indices, joint


def pooled(audit, role, indices, joint):
    cases = np.asarray(audit['roles'][role], dtype=np.int64)
    owners = np.asarray([audit['fields'][i]['case_index'] for i in indices], dtype=np.int64)
    magnifications = np.asarray([int(audit['fields'][i]['magnification']) for i in indices], dtype=np.int64)
    views = metrics.field_views(joint, owners, magnifications, cases)
    probabilities, predictions, scores = metrics.pool_subsets(views)
    truth = np.asarray([audit['cases'][i]['truth5'] for i in cases], dtype=np.int64)
    return probabilities, predictions, scores, metrics.case_subset_summary(truth, predictions)


def fixed_binary(audit, role, indices, actual, reference, tolerance=1e-12):
    """Every field marginal and every case/mask route must match the parent."""
    require(actual.shape == reference.shape, 'Parent binary field coverage differs')
    a = np.column_stack((actual[:, 0], actual[:, 1:].sum(1)))
    b = np.column_stack((reference[:, 0], reference[:, 1:].sum(1)))
    field_error = float(np.max(np.abs(a-b)))
    require(field_error <= tolerance, 'Fixed binary field marginal changed')
    _, _, scores, summary = pooled(audit, role, indices, actual)
    _, _, parent_scores, _ = pooled(audit, role, indices, reference)
    case_error = float(np.max(np.abs(scores-parent_scores)))
    require(case_error <= tolerance, 'Fixed binary case marginal changed')
    require(np.array_equal(scores >= .5, parent_scores >= .5), 'Fixed binary hard routing changed')
    mean_binary = float(np.mean([v['binary']['accuracy'] for v in summary['subsets'].values()]))
    return {'maximum_field_abs': field_error, 'maximum_case_abs': case_error,
            'hard_route_disagreements': 0, 'mean_subset_binary_accuracy': mean_binary}


def identity_check(audit, indices, actual, parent, atol=1e-10):
    error = float(np.max(np.abs(actual-parent)))
    require(error <= atol, 'Parent P identity field probabilities changed')
    a, pa, _, sa = pooled(audit, 'validation', indices, actual)
    b, pb, _, sb = pooled(audit, 'validation', indices, parent)
    require(np.max(np.abs(a-b)) <= atol and np.array_equal(pa, pb), 'Parent P identity case prediction changed')
    for key in ('mean_subset_macro_f1', 'mean_subset_accuracy'):
        require_equal(sa[key], sb[key], 'parent identity/'+key, 1e-12)
    for name in sa['subsets']:
        require_equal(sa['subsets'][name]['five_class']['confusion_matrix'],
                      sb['subsets'][name]['five_class']['confusion_matrix'], 'parent identity/confusion', 0)
    return {'maximum_field_abs': error, 'maximum_case_abs': float(np.max(np.abs(a-b))),
            'case_decisions_identical': True, 'confusions_identical': True}


def geometry(audit, p_features, uni_features):
    """Independent FP64 case centers, normalized supports and density weights."""
    cases, fields = audit['cases'], audit['fields']
    train = np.asarray(audit['roles']['train'], dtype=np.int64)
    labels = np.asarray([c['truth5'] for c in cases], dtype=np.int64)
    require(len(set(train.tolist())) == len(train) and np.all(np.bincount(labels[train], minlength=5) > 0),
            'Fitting support lacks a final class')
    malignant = train[labels[train] > 0]
    result = {'train': train, 'malignant': malignant, 'labels': labels, 'centers': {h: [] for h in ('direct', 'binary', 'p', 'uni')},
              'support': {h: [] for h in ('binary', 'p', 'uni')}, 'weights': {h: [] for h in ('binary', 'p', 'uni')}}
    for view in range(4):
        case_fields = {i: np.asarray([f['index'] for f in fields if f['case_index'] == i and f['view'] == view], dtype=int)
                       for i in train}
        require(all(len(ix) for ix in case_fields.values()), 'Fitting view is empty')
        means_p = np.asarray([p_features[case_fields[i]].astype(np.float64).mean(0) for i in train])
        means_u = np.asarray([uni_features[case_fields[i]].astype(np.float64).mean(0) for i in train])
        classes = np.asarray([means_p[labels[train] == c].mean(0) for c in range(5)])
        result['centers']['direct'].append(classes.mean(0))
        result['centers']['binary'].append(np.stack((means_p[labels[train] == 0].mean(0), means_p[labels[train] > 0].mean(0))).mean(0))
        result['centers']['p'].append(classes[1:].mean(0))
        result['centers']['uni'].append(means_u.mean(0))
        for head, features, eligible, head_labels in (
                ('binary', p_features, train, (labels > 0).astype(int)),
                ('p', p_features, malignant, labels-1), ('uni', uni_features, malignant, labels-1)):
            count = 2 if head == 'binary' else 4
            case_counts = np.bincount(head_labels[eligible], minlength=count)
            values, weights = [], []
            for i in eligible:
                selected = case_fields[i]
                values.append(normalize(features[selected], result['centers'][head][view]))
                weight = np.zeros((len(selected), count), dtype=np.float64)
                weight[:, head_labels[i]] = 1./(case_counts[head_labels[i]]*len(selected))
                weights.append(weight)
            result['support'][head].append(np.concatenate(values))
            result['weights'][head].append(np.concatenate(weights))
            require(np.allclose(result['weights'][head][-1].sum(0), 1, rtol=0, atol=1e-12), 'Support class mass differs')
    return result


def normalize(features, center):
    values = np.asarray(features, dtype=np.float64)-np.asarray(center, dtype=np.float64)
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def density_logs(cosine, weights, temperature):
    scaled = cosine/temperature
    maximum = scaled.max(1, keepdims=True)
    mass = np.exp(scaled-maximum) @ weights
    require(np.isfinite(mass).all() and np.all(mass > 0), 'Independent KDE mass is invalid')
    logs = np.log(mass)+maximum
    return logs-logsumexp(logs, axis=1, keepdims=True)


def replay_densities(audit, p_features, uni_features, prepared, role, settings):
    """Only cache arithmetic; no producer specialist helper or encoder is used."""
    indices = np.asarray([f['index'] for f in audit['fields'] if f['case_index'] in set(audit['roles'][role])], dtype=np.int64)
    positions = {int(i): j for j, i in enumerate(indices)}
    outputs = {(arm, float(t)): np.full((len(indices), 5), np.nan) for arm, temps in settings.items() for t in temps}
    for view in range(4):
        selected = np.asarray([i for i in indices if audit['fields'][i]['view'] == view], dtype=int)
        where = [positions[int(i)] for i in selected]
        cosine = {}
        for head, features in (('binary', p_features), ('p', p_features), ('uni', uni_features)):
            query = normalize(features[selected], prepared['centers'][head][view])
            raw_cosine = query @ prepared['support'][head][view].T
            require(np.isfinite(raw_cosine).all(), 'Independent cosine is nonfinite before clipping')
            cosine[head] = np.clip(raw_cosine, -1., 1.)
        binary = density_logs(cosine['binary'], prepared['weights']['binary'][view], .1)
        for arm, temps in settings.items():
            head = 'p' if arm == 'p_control' else 'uni'
            for temperature in temps:
                malignant = density_logs(cosine[head], prepared['weights'][head][view], temperature)
                logs = np.column_stack((binary[:, 0], binary[:, 1, None]+malignant))
                outputs[arm, float(temperature)][where] = np.exp(logs-logsumexp(logs, axis=1, keepdims=True))
    require(all(np.isfinite(v).all() for v in outputs.values()), 'Independent density coverage is incomplete')
    return indices, outputs


def verify_support(summary, prepared, audit, p_dimension, uni_dimension):
    require(set(summary) == set(ARMS), 'Support summary arms differ')
    train, malignant, labels = prepared['train'], prepared['malignant'], prepared['labels']
    counts = np.bincount(labels[train], minlength=5).tolist()
    p = summary['p_control']
    wanted_p = {'support_indices': train.tolist(), 'class_support': counts, 'dimension': p_dimension,
                'head_case_counts': {'direct': counts, 'binary': [counts[0], sum(counts[1:])], 'malignant': counts[1:]},
                'field_counts_by_view': [sum(f['case_index'] in set(train) and f['view'] == v for f in audit['fields']) for v in range(4)],
                'centers_sha256': {'direct': array_digest(prepared['centers']['direct']),
                                  'binary': array_digest(prepared['centers']['binary']),
                                  'malignant': array_digest(prepared['centers']['p'])}}
    for key, value in wanted_p.items():
        require_equal(p.get(key), value, 'independent P support/'+key, 0)
    u = summary['uni2_specialist']
    wanted_u = {'support_case_indices': train.tolist(), 'fitting_class_counts': counts,
                'malignant_support_case_indices': malignant.tolist(), 'dimension': uni_dimension,
                'centers_sha256': array_digest(prepared['centers']['uni']),
                'normalized_support_sha256': array_digest(prepared['support']['uni']),
                'weights_sha256': array_digest(prepared['weights']['uni']),
                'malignant_fields_by_view': [len(v) for v in prepared['support']['uni']]}
    for key, value in wanted_u.items():
        require_equal(u.get(key), value, 'independent UNI support/'+key, 0)
    require_equal(u.get('class_weight_sums_by_view'), [v.sum(0).tolist() for v in prepared['weights']['uni']],
                  'independent UNI class mass', 1e-12)
    return {'fitting_cases': len(train), 'malignant_cases': len(malignant), 'class_cases': counts,
            'p_centers_sha256': wanted_p['centers_sha256'], 'uni_centers_sha256': wanted_u['centers_sha256']}


def source_context(args):
    repo = Path(__file__).resolve().parents[1]
    require(re.fullmatch('[a-f0-9]{40}', args.code_sha or '') is not None, 'Full immutable source SHA required')
    require((repo/'.source-sha').read_text().strip() == args.code_sha, 'Immutable source receipt differs')
    archive = (repo/'.archive-sha256').read_text().strip()
    require(re.fullmatch('[a-f0-9]{64}', archive) is not None, 'Source archive receipt missing')
    require(file_hash(args.config) == file_hash(repo/'configs/breakhis-specialist-reader-v1.json'), 'Canonical specialist config differs')
    cfg = read_json(args.config)
    require(cfg['name'] == 'breakhis-specialist-reader-v1' and cfg['arms'] == list(ARMS)
            and cfg['primary_arm'] == 'uni2_specialist' and cfg['density']['malignant_temperatures'] == list(TEMPERATURES)
            and cfg['router']['temperature'] == .1 and cfg['router']['threshold'] == .5,
            'Registered specialist scope differs')
    for name, version in cfg['runtime'].items():
        require(importlib.metadata.version(name) == version, 'Pinned collector runtime differs/'+name)
    references = cfg['references']
    pins = {}
    for key in ('p_feature_config', 'p_readout_config', 'uni_feature_config', 'uni_extraction_config'):
        digest = file_hash(repo/references[key])
        require(digest == references[key+'_sha256'], 'Referenced config bytes differ/'+key)
        pins[references[key]] = digest
    require(file_hash(repo/references['uni_previous_result']) == references['uni_previous_result_sha256'], 'Prior UNI result bytes differ')
    context = read_json(Path(args.root)/'context.json')
    expected = {'code_sha': args.code_sha, 'source_archive_sha256': archive, 'config_sha256': file_hash(args.config),
                'parent_result_sha256': cfg['matched_reference']['sha256'], 'input_audit_sha256': cfg['input_audit']['private_sha256']}
    for key, value in expected.items():
        require_equal(context.get(key), value, 'family context/'+key, 0)
    for key, value in {'manifest_sha256': cfg['manifest_sha256'],
                       'parent_config_sha256': cfg['matched_reference']['config_sha256'],
                       'expected_counts': cfg['counts'], 'runtime_versions': cfg['runtime']}.items():
        require_equal(context.get(key), value, 'family context/'+key, 0)
    require_equal(context.get('reference_config_sha256'), pins, 'referenced config map', 0)
    implementation = context.get('implementation_sha256')
    require(isinstance(implementation, dict) and {'magset/specialist_reader.py', 'scripts/run_specialist_reader.py'} <= set(implementation),
            'Scientific implementation receipts missing')
    for path, digest in implementation.items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts and file_hash(repo/path) == digest,
                'Family implementation differs/'+path)
    return cfg, context, expected, repo


def parent_inputs(args, cfg, repo):
    root = Path(args.parent)
    parent_path = root/'checked/result.json'
    require(file_hash(parent_path) == cfg['matched_reference']['sha256'], 'Accepted parent result changed')
    parent = read_json(parent_path)
    require(parent['status'] == 'complete accepted matched three-way comparison', 'Parent comparison not accepted')
    require(parent['code_sha'] == cfg['matched_reference']['producer_code_sha'], 'Parent scientific source differs')
    parent_cfg_path = repo/cfg['matched_reference']['config']
    require(file_hash(parent_cfg_path) == cfg['matched_reference']['config_sha256'] == parent['config_sha256'], 'Parent configuration differs')
    audit_path = root/'input-audit/private.json'
    require(file_hash(audit_path) == cfg['input_audit']['private_sha256'] == parent['input_audit_sha256'], 'Parent audit bytes changed')
    parent_context = {k: parent[k] for k in ('code_sha', 'source_archive_sha256', 'config_sha256', 'implementation_sha256')}
    audit, counts = metrics.accepted_audit(SimpleNamespace(audit=root/'input-audit'), parent_context, read_json(parent_cfg_path))
    require(file_hash(root/'input-audit/indices.npz') == cfg['input_audit']['indices_sha256'], 'Parent role indices differ')
    for role, wanted in cfg['expected_roles'].items():
        for key in ('cases', 'groups', 'class_cases'):
            require_equal(counts[role][key], wanted[key], 'parent role/'+role+'/'+key, 0)
        require(counts[role]['fields'] == wanted['images'], 'Parent field count differs')
    folder = root/'models/midnight_p'
    selection_path = folder/'selection.json'
    require(file_hash(selection_path) == parent['models']['midnight_p']['selection_sha256'], 'Parent P selection differs')
    selection = read_json(selection_path)
    require(selection['selected_setting'] == .1 and selection['selected_index'] == 2, 'Parent P temperature differs')
    validation = folder/'candidates/candidate2/validation-fields.npz'
    receipt = selection['candidate_artifact_sha256'][2]
    require(file_hash(validation) == receipt['validation-fields.npz'], 'Parent P validation archive changed')
    test_receipt = read_json(folder/'test/test.json')
    require(test_receipt['status'] == 'complete selected test inference' and test_receipt['model'] == 'midnight_p'
            and test_receipt['code_sha'] == parent['code_sha'] and test_receipt['selection_sha256'] == file_hash(selection_path)
            and test_receipt['gate_sha256'] == parent['test_gate_sha256'], 'Parent P test receipt differs')
    test = folder/'test/fields.npz'
    require(file_hash(test) == test_receipt['artifact_sha256']['fields.npz'], 'Parent P follow-up archive changed')
    references = {}
    for role, path in (('validation', validation), ('test', test)):
        indices, values = load_field_archive(path, audit, role)
        references[role] = (indices, values)
        _, _, _, summary = pooled(audit, role, indices, values)
        accepted = parent['models']['midnight_p']['selected_validation' if role == 'validation' else 'test']
        require_equal(summary, accepted, 'canonical parent P/'+role, 2e-12)
    return audit, counts, references


def load_features(path, reference, audit, dimension):
    fields = audit['fields']
    lookup = {f['relative_path']: f['index'] for f in fields}
    require(len(lookup) == len(fields), 'Canonical field paths duplicate')
    output = np.empty((len(fields), dimension), dtype=np.float32)
    seen = set()
    for mag in metrics.MAGNIFICATIONS:
        folder = Path(path); target = folder/(reference['encoder_name']+'-%d.npz' % mag)
        pin = reference['cache_provenance'][str(mag)]
        require(file_hash(target) == pin['npz_sha256'] and file_hash(target.with_suffix('.json')) == pin['json_sha256'], 'Pinned cache bytes differ')
        if 'components_sha256' in pin:
            require(file_hash(target.with_name(target.stem+'-components.npz')) == pin['components_sha256'], 'Pinned crop components differ')
        metadata = read_json(target.with_suffix('.json'))
        require(metadata['manifest_sha256'] == audit['manifest_sha256'] and metadata['magnification'] == mag
                and metadata['feature_sha256'] == pin['npz_sha256'], 'Cache metadata lineage differs')
        with np.load(target, allow_pickle=False) as saved:
            require(set(saved.files) == {'paths', 'features'}, 'Cache array schema differs')
            paths, values = saved['paths'].tolist(), saved['features']
            wanted = {f['relative_path'] for f in fields if int(f['magnification']) == mag}
            require(len(paths) == len(set(paths)) == len(wanted) and set(paths) == wanted
                    and values.shape == (len(paths), dimension) and values.dtype == np.dtype('float32')
                    and np.isfinite(values).all(), 'Cache path/dimension/value coverage differs')
            require(not seen & set(paths), 'Cache fields repeat across shards')
            output[[lookup[p] for p in paths]] = values
            seen.update(paths)
    require(seen == set(lookup), 'Cache omits canonical fields')
    return output


def verify_dataset_context(context, audit, cfg, repo):
    train = audit['roles']['train']
    malignant = [i for i in train if audit['cases'][i]['truth5'] > 0]
    require_equal(context.get('support_case_indices'), train, 'fitting support indices', 0)
    require_equal(context.get('malignant_support_case_indices'), malignant, 'malignant fitting support indices', 0)
    require_equal(context.get('field_alignment_sha256'), alignment_digest(audit), 'canonical field alignment digest', 0)
    references = cfg['references']
    p_ref, u_ref = read_json(repo/references['p_feature_config']), read_json(repo/references['uni_feature_config'])
    require_equal(context.get('cache_provenance'), {'p': p_ref['cache_provenance'], 'uni': u_ref['cache_provenance']}, 'Cache provenance map', 0)
    return p_ref, u_ref


def verify_selection(selection, scores, arm, context, context_sha, support_sha):
    require(selection.get('status') == 'selected validation candidate' and selection.get('arm') == arm
            and selection.get('test_predictions') == 0, 'Validation selection status/arm differs')
    for key in CONTEXT_KEYS:
        require_equal(selection.get(key), context[key], 'selection/context/'+key, 0)
    require_equal(selection.get('context_sha256'), context_sha, 'selection/context bytes', 0)
    require_equal(selection.get('support_summary_sha256'), support_sha, 'selection/support bytes', 0)
    require_equal(selection.get('candidate_validation_scores'), scores, 'independent four validation scores', 1e-12)
    selected = selection.get('selected_index')
    require(type(selected) is int and selected == int(np.argmax(scores)), 'Selection is not first maximum')
    require(selection.get('selected_setting') == TEMPERATURES[selected], 'Selected malignant temperature differs')
    require(len(selection.get('candidate_artifact_sha256', [])) == 4, 'Four candidate receipts required')
    return selected


def verify_gate(gate, selections, context_sha, audit_sha, support_sha):
    require(gate.get('status') == 'sealed both specialist selections' and gate.get('test_queries') == 0,
            'Both-arm selection gate did not pass')
    require_equal(gate.get('selections'), selections, 'sealed selection hashes', 0)
    require_equal(gate.get('context_sha256'), context_sha, 'gate/context bytes', 0)
    require_equal(gate.get('input_audit_sha256'), audit_sha, 'gate/input audit', 0)
    require_equal(gate.get('support_summary_sha256'), support_sha, 'gate/support bytes', 0)


def check_prediction_set(folder, prefix, audit, role, reference, expected_density):
    indices, values = load_field_archive(Path(folder)/(prefix+'fields.npz'), audit, role)
    replay_records, summary, _ = metrics.replay_field_archive(Path(folder)/(prefix+'fields.npz'), audit, role)
    require_equal(metrics.read_records(Path(folder)/(prefix+'records.jsonl')), replay_records, 'independent case/subset records', 1e-10)
    reported = read_json(Path(folder)/'metrics.json')
    reported = reported[role] if role in reported else reported
    metrics.verify_reported_summary(reported, summary)
    # The older reviewed helper permits 2e-12; this registration requires 1e-12.
    for key in ('mean_subset_macro_f1', 'mean_subset_accuracy', 'worst_subset_macro_f1'):
        require_equal(reported[key], summary[key], 'registered metric tolerance/'+key, 1e-12)
    for name, subset in summary['subsets'].items():
        for key in ('records', 'accuracy', 'macro_f1', 'confusion_matrix', 'class_support', 'class_recall'):
            require_equal(reported['subsets'][name][key], subset['five_class'][key], 'registered subset metric/'+name+'/'+key, 1e-12)
        require_equal(reported['binary_summary']['subsets'][name]['accuracy'], subset['binary']['accuracy'],
                      'registered binary metric/'+name, 1e-12)
    density_error = float(np.max(np.abs(values-expected_density)))
    require(density_error <= 1e-10, 'Independent cached-feature density replay differs')
    _, actual_labels, _, _ = pooled(audit, role, indices, values)
    _, expected_labels, _, _ = pooled(audit, role, indices, expected_density)
    require(np.array_equal(actual_labels, expected_labels), 'Independent density replay changed a routed decision')
    binary = fixed_binary(audit, role, indices, values, reference)
    return values, summary, {'density_maximum_abs': density_error, 'fixed_binary': binary}


def collect(args):
    output = Path(args.output)
    require(not output.exists(), 'Refusing existing specialist collector output')
    result = {'status': 'failed or incomplete specialist-reader family', 'collector_new_fits': 0,
              'collector_new_encoder_forwards': 0, 'collector_new_image_decodes': 0,
              'interpretation': 'Prespecified fixed-P-router specialist comparison on reused development roles. '
              'The follow-up test has already been viewed; it is retrospective, not fresh clinical confirmation. '
              'Binary routing is unchanged and this family cannot improve the >97% binary objective.'}
    private = {'errors': {}}
    try:
        cfg, context, public_context, repo = source_context(args)
        result.update(public_context)
        audit, counts, parent = parent_inputs(args, cfg, repo)
        p_ref, u_ref = verify_dataset_context(context, audit, cfg, repo)
        p_features = load_features(context.get('p_cache', audit['coverage_cache']), p_ref, audit, 3072)
        uni_features = load_features(context['uni_cache'], u_ref, audit, 1536)
        prepared = geometry(audit, p_features, uni_features)
        root = Path(args.root); context_sha = file_hash(root/'context.json'); support_sha = file_hash(root/'support-summary.json')
        support = verify_support(read_json(root/'support-summary.json'), prepared, audit, 3072, 1536)
        validation_indices, expected = replay_densities(audit, p_features, uni_features, prepared, 'validation', {arm: TEMPERATURES for arm in ARMS})
        require(np.array_equal(validation_indices, parent['validation'][0]), 'Parent validation field alignment differs')
        selections, selection_hashes, checked = {}, {}, {}
        identity = None
        identity_path = root/'parent-identity/validation.json'
        identity_sha = file_hash(identity_path)
        for arm in ARMS:
            folder = root/'arms'/arm
            require({p.name for p in (folder/'candidates').iterdir()} == {'candidate%d' % i for i in range(4)}, 'Unexpected validation candidate family')
            selection = read_json(folder/'selection.json')
            require_equal(selection.get('validation_identity_sha256'), identity_sha, 'selection/validation identity bytes', 0)
            candidates, scores = [], []
            receipts = selection.get('candidate_artifact_sha256', [])
            require(len(receipts) == 4, 'Incomplete candidate hash receipts')
            for i, temperature in enumerate(TEMPERATURES):
                candidate = folder/'candidates'/('candidate%d' % i)
                require(set(receipts[i]) == CANDIDATE_FILES, 'Candidate artifact schema differs')
                metrics.checked_hashes(candidate, receipts[i], CANDIDATE_FILES)
                require(read_json(candidate/'metrics.json').get('setting') == temperature, 'Candidate temperature differs')
                values, summary, proof = check_prediction_set(candidate, 'validation-', audit, 'validation', parent['validation'][1], expected[arm, temperature])
                if arm == 'p_control' and i == 2:
                    identity = identity_check(audit, validation_indices, values, parent['validation'][1])
                    anchor = root/'parent-identity/validation-fields.npz'
                    anchor_indices, anchor_values = load_field_archive(anchor, audit, 'validation')
                    require(np.array_equal(anchor_indices, validation_indices), 'Identity field indices differ')
                    require(np.array_equal(anchor_values, values), 'Rederived parent identity does not match control candidate')
                    require_equal(read_json(identity_path), {
                        'status': 'accepted rederived parent P validation identity',
                        'parent_archive_sha256': file_hash(Path(args.parent)/'models/midnight_p/candidates/candidate2/validation-fields.npz'),
                        'maximum_field_probability_abs': identity['maximum_field_abs'],
                        'maximum_case_probability_abs': identity['maximum_case_abs'],
                        'decision_disagreements': 0, 'test_queries': 0}, 'validation identity receipt', 1e-12)
                candidates.append({'index': i, 'temperature': temperature, 'validation': summary, 'checks': proof})
                scores.append(summary['mean_subset_macro_f1'])
            selected = verify_selection(selection, scores, arm, context, context_sha, support_sha)
            selections[arm] = selection
            selection_hashes[arm] = file_hash(folder/'selection.json')
            checked[arm] = {'selected_index': selected, 'selected_temperature': TEMPERATURES[selected],
                            'candidates': candidates, 'selected_validation': candidates[selected]['validation']}
        gate_path = root/'family-gate.json'
        verify_gate(read_json(gate_path), selection_hashes, context_sha, cfg['input_audit']['private_sha256'], support_sha)
        settings = {arm: (TEMPERATURES[selections[arm]['selected_index']],) for arm in ARMS}
        test_indices, expected_test = replay_densities(audit, p_features, uni_features, prepared, 'test', settings)
        require(np.array_equal(test_indices, parent['test'][0]), 'Parent follow-up field alignment differs')
        optional_parent = root/'parent-identity/test-fields.npz'
        if optional_parent.exists():
            indices, copied = load_field_archive(optional_parent, audit, 'test')
            require(np.array_equal(indices, test_indices) and np.array_equal(copied, parent['test'][1]), 'Canonical parent follow-up reference copy differs')
        for arm in ARMS:
            selected = selections[arm]; folder = root/'arms'/arm/'test'; receipt = read_json(folder/'test.json')
            require(receipt.get('status') == 'complete selected retrospective inference' and receipt.get('arm') == arm,
                    'Selected retrospective inference incomplete')
            for key, value in {'selected_index': selected['selected_index'], 'selected_setting': selected['selected_setting'],
                               'selection_sha256': selection_hashes[arm], 'gate_sha256': file_hash(gate_path),
                               'context_sha256': context_sha, 'support_summary_sha256': support_sha}.items():
                require_equal(receipt.get(key), value, 'selected follow-up/'+key, 0)
            metrics.checked_hashes(folder, receipt.get('artifact_sha256'), {'fields.npz', 'records.jsonl', 'metrics.json'})
            _, summary, proof = check_prediction_set(folder, '', audit, 'test', parent['test'][1], expected_test[arm, selected['selected_setting']])
            checked[arm].update(retrospective_follow_up=summary, follow_up_checks=proof)
        control, candidate = (checked[arm]['selected_validation'] for arm in ARMS)
        delta_f1 = candidate['mean_subset_macro_f1']-control['mean_subset_macro_f1']
        delta_accuracy = candidate['mean_subset_accuracy']-control['mean_subset_accuracy']
        screen = {'validation_macro_f1_delta': delta_f1, 'validation_accuracy_delta': delta_accuracy,
                  'passed': delta_f1 >= cfg['progression_screen']['mean15_macro_f1_delta_at_least'] and delta_accuracy >= 0,
                  'scope': 'development validation screen only; every selected follow-up remains reported'}
        result.update(status='complete accepted specialist-reader family', roles=counts, support=support,
                      context_sha256=context_sha, support_summary_sha256=support_sha, parent_identity=identity,
                      family_gate_sha256=file_hash(gate_path), arms=checked, progression_screen=screen,
                      validation_case_subset_records=2040, retrospective_case_subset_records=480)
    except Exception as error:
        private['errors']['review'] = {'message': str(error), 'traceback': traceback.format_exc()}
        result['review'] = {'status': 'invalid_or_incomplete', 'error_type': type(error).__name__}
    assert_public_safe(result)
    output.mkdir(parents=True)
    write_json(output/'private_review.json', private)
    result['private_review_sha256'] = file_hash(output/'private_review.json')
    write_json(output/'result.json', result)
    lines = ['# Frozen specialist-reader check', '', 'Status: '+result['status']+'.', '']
    if 'arms' in result:
        for arm, value in result['arms'].items():
            score = value['retrospective_follow_up']
            lines.append(f"{arm}: selected malignant temperature {value['selected_temperature']}; retrospective mean15 "
                         f"accuracy {100*score['mean_subset_accuracy']:.2f}%, macro F1 {score['mean_subset_macro_f1']:.6f}.")
    else:
        lines.append('No model performance is accepted.')
    lines.extend(['', result['interpretation'], '', 'The collector performed no fitting, image decoding or encoder inference.', ''])
    (output/'summary.md').write_text('\n'.join(lines))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True)
    return p


if __name__ == '__main__':
    result = collect(parser().parse_args())
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'complete accepted specialist-reader family' else 1)

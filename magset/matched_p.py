"""Fit-only Midnight P supports and sealed validation/test actions for matched roles."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import matched_data as md
from . import prior_centered_model as p
from .bags import load_bags
from .data import MAGNIFICATIONS, file_hash
from .five_metrics import MAP8_TO5

MODEL = 'midnight_p'
TEMPERATURES = (0.01, 0.03, 0.1, 0.3)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text())


def _cache_bindings(cache, config):
    own = config['midnight_p']
    path = REPO_ROOT / own['feature_config']
    if file_hash(path) != own['feature_config_sha256']:
        raise ValueError('Pinned coverage feature configuration differs')
    coverage = _read(path)
    readout = _read(REPO_ROOT / own['readout_config'])
    if (coverage['encoder_name'] != 'midnight-coverage-v1' or coverage['dimension'] != 3072
            or readout['base_input_config_sha256'] != file_hash(path)
            or own['mode'] != 'head_specific' or 'head_specific' not in readout['modes']
            or tuple(own['temperatures']) != TEMPERATURES
            or tuple(coverage['temperatures']) != TEMPERATURES
            or tuple(readout['temperatures']) != TEMPERATURES):
        raise ValueError('Matched P feature/readout registration differs')
    for mag in MAGNIFICATIONS:
        shard = Path(cache) / ('midnight-coverage-v1-%d.npz' % mag)
        paths = {'npz_sha256': shard, 'json_sha256': shard.with_suffix('.json'),
                 'components_sha256': shard.with_name(shard.stem + '-components.npz')}
        pin = coverage['cache_provenance'][str(mag)]
        if any(file_hash(path) != pin[key] for key, path in paths.items()):
            raise ValueError('Pinned Midnight coverage cache bytes differ')
    return coverage


def _load_inputs(args):
    config = md.config(args.config)
    audit = md.load_audit(args.audit, args.config)
    context = md.context(args.code_sha, args.config)
    _cache_bindings(args.cache, config)
    fields = audit['fields']
    rows = sorted(fields, key=lambda row: row['relative_path'])
    if [row['index'] for row in rows] != list(range(len(fields))):
        raise ValueError('Audit field order is not canonical sorted manifest order')
    cases, bags, _ = load_bags(rows, args.cache, 'midnight-coverage-v1', config['manifest_sha256'])
    if [case['case_id'] for case in cases] != [case['case_id'] for case in audit['cases']]:
        raise ValueError('Cached feature case order differs from matched audit')
    for i, (case, audited) in enumerate(zip(cases, audit['cases'])):
        if MAP8_TO5[case['label']] != audited['truth5'] or case['stem_group'] != audited['group']:
            raise ValueError('Cached case labels/groups differ from matched audit')
        for view in range(4):
            wanted = [f for f in fields if f['case_index'] == i and f['view'] == view]
            if bags[i][view].shape != (len(wanted), 3072):
                raise ValueError('Cached bag dimensions/field membership differ')
    labels = np.asarray([case['truth5'] for case in audit['cases']], dtype=np.int64)
    return config, audit, context, bags, labels


def _field_probabilities(audit, bags, labels, support, queries, temperatures):
    """Prepare each query once, with fitting supports only, and share each T across heads."""
    support = list(map(int, support)); queries = list(map(int, queries))
    if (not queries or len(set(queries)) != len(queries)
            or len(set(support)) != len(support) or set(support) & set(queries)):
        raise ValueError('Unique disjoint support/query roles required')
    if any(float(t) not in TEMPERATURES for t in temperatures):
        raise ValueError('Only registered shared temperatures are permitted')
    fields = audit['fields']
    indices = sorted(f['index'] for f in fields if f['case_index'] in set(queries))
    positions = {value: i for i, value in enumerate(indices)}
    outputs = [np.full((len(indices), 5), np.nan, dtype=np.float64) for _ in temperatures]
    for query in sorted(queries):
        prepared = p.prepare_case(bags, labels, support, query, leave_query_case_out=False)
        for candidate, temperature in enumerate(temperatures):
            density = p.density_cache(prepared, temperature)
            logs = p.field_log_probabilities(prepared, temperature, 'dual', 'head_specific', cache=density)
            for view, values in enumerate(logs):
                keys = sorted((f for f in fields if f['case_index'] == query and f['view'] == view),
                              key=lambda f: f['relative_path'])
                values = np.asarray(values, dtype=np.float64)
                if values.shape != (len(keys), 5) or not np.isfinite(values).all():
                    raise ValueError('P query fields or finite log probabilities differ')
                joint = np.exp(values - logsumexp(values, axis=1, keepdims=True))
                outputs[candidate][[positions[f['index']] for f in keys]] = joint
    if any(not np.isfinite(values).all() for values in outputs):
        raise ValueError('P field predictions have incomplete role coverage')
    return np.asarray(indices, dtype=np.int64), outputs


def _artifacts(directory):
    return {str(path.relative_to(directory)): file_hash(path) for path in sorted(Path(directory).rglob('*'))
            if path.is_file() and path != Path(directory) / 'result.json'}


def _write_prediction_set(directory, prefix, audit, role, indices, joint):
    records = md.case_records(audit['fields'], audit['cases'], audit['roles'][role], indices, joint, role)
    metrics = md.summary(records)
    md.save_predictions(Path(directory) / (prefix + 'fields.npz'), indices, joint)
    md.save_records(Path(directory) / (prefix + 'records.jsonl'), records)
    return metrics


def select(args):
    """Evaluate exactly four validation candidates and never query the test role."""
    config, audit, context, bags, labels = _load_inputs(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists() or (output / 'result.json').exists():
        raise ValueError('Refusing to overwrite a completed P selection')
    support = list(map(int, audit['roles']['train']))
    indices, predictions = _field_probabilities(audit, bags, labels, support,
                                                audit['roles']['validation'], TEMPERATURES)
    scores = []; hashes = []
    for candidate, (temperature, joint) in enumerate(zip(TEMPERATURES, predictions)):
        target = output / 'candidates' / ('candidate%d' % candidate)
        target.mkdir(parents=True)
        metrics = _write_prediction_set(target, 'validation-', audit, 'validation', indices, joint)
        md.write_json(target / 'metrics.json', {'setting': temperature, 'validation': metrics})
        scores.append(float(metrics['mean_subset_macro_f1']))
        hashes.append({name: file_hash(target / name) for name in
                       ('validation-fields.npz', 'validation-records.jsonl', 'metrics.json')})
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite validation selection score')
    selected = int(np.argmax(scores))  # Earliest candidate on an exact tie.
    audit_sha = file_hash(Path(args.audit) / 'private.json')
    selection = dict(status='selected validation candidate', model=MODEL, **context,
                     input_audit_sha256=audit_sha, candidate_validation_scores=scores,
                     selected_index=selected, selected_setting=TEMPERATURES[selected],
                     selected_checkpoint_sha256=None, candidate_artifact_sha256=hashes,
                     test_predictions=0, support_case_indices=support)
    md.write_json(output / 'selection.json', selection)
    result = dict(status='complete training', model=MODEL, **context,
                  input_audit_sha256=audit_sha, artifact_sha256=_artifacts(output))
    md.write_json(output / 'result.json', result)
    return selection


def _selected_candidate(path, context, audit_sha, support):
    selection = _read(path)
    if (selection.get('status') != 'selected validation candidate' or selection.get('model') != MODEL
            or any(selection.get(key) != value for key, value in context.items())
            or selection.get('input_audit_sha256') != audit_sha
            or selection.get('support_case_indices') != list(map(int, support))
            or selection.get('selected_checkpoint_sha256') is not None
            or selection.get('test_predictions') != 0):
        raise ValueError('Sealed P selection context/support differs')
    scores = np.asarray(selection.get('candidate_validation_scores'), dtype=np.float64)
    index = selection.get('selected_index')
    if (scores.shape != (4,) or not np.isfinite(scores).all() or type(index) is not int
            or index != int(np.argmax(scores)) or selection.get('selected_setting') != TEMPERATURES[index]):
        raise ValueError('P selection does not follow earliest maximum validation score')
    receipts = selection.get('candidate_artifact_sha256')
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise ValueError('P candidate artifact receipts incomplete')
    for candidate, receipt in enumerate(receipts):
        if set(receipt) != {'validation-fields.npz', 'validation-records.jsonl', 'metrics.json'}:
            raise ValueError('P candidate artifact schema differs')
        folder = Path(path).parent / 'candidates' / ('candidate%d' % candidate)
        if any(file_hash(folder / name) != digest for name, digest in receipt.items()):
            raise ValueError('P candidate artifact changed after selection')
        metrics = _read(folder / 'metrics.json')
        if (metrics.get('setting') != TEMPERATURES[candidate]
                or metrics['validation']['mean_subset_macro_f1'] != scores[candidate]):
            raise ValueError('P recorded candidate score/setting differs')
    return selection


def test(args):
    """Query test cases only after the common all-three-model selection gate."""
    verifier = getattr(md, 'verify_gate', None)
    if verifier is None:
        raise RuntimeError('Common matched test-gate verifier is required')
    verifier(args.root, args.audit, args.config, args.code_sha)
    config, audit, context, bags, labels = _load_inputs(args)
    output = Path(args.output)
    selection_path = output.parent / 'selection.json'
    audit_sha = file_hash(Path(args.audit) / 'private.json')
    support = audit['roles']['train']
    selection = _selected_candidate(selection_path, context, audit_sha, support)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('fields.npz', 'records.jsonl', 'metrics.json', 'test.json')):
        raise ValueError('Refusing to overwrite selected P test predictions')
    indices, predictions = _field_probabilities(audit, bags, labels, support, audit['roles']['test'],
                                                (selection['selected_setting'],))
    metrics = _write_prediction_set(output, '', audit, 'test', indices, predictions[0])
    md.write_json(output / 'metrics.json', metrics)
    result = dict(status='complete selected test inference', model=MODEL, **context,
                  input_audit_sha256=audit_sha, selection_sha256=file_hash(selection_path),
                  gate_sha256=file_hash(Path(args.root) / 'test-gate.json'),
                  artifact_sha256={name: file_hash(output / name) for name in
                                   ('fields.npz', 'records.jsonl', 'metrics.json')})
    md.write_json(output / 'test.json', result)
    return result

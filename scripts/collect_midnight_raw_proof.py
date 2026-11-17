"""Check all-field raw Midnight features and independently replay its fixed KDE."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np

PRODUCER = 'a6e6a54e4fac090051837df17b0dea14efca3bb3'
RESULT_SHA = '3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62'
MAGS = (40, 100, 200, 400)
SUBSETS = tuple(s for size in range(1, 5) for s in combinations(range(4), size))
TEMPERATURES = (.01, .03, .1, .3)
FEATURE_ATOL = FEATURE_RTOL = 1e-4
PROBABILITY_ATOL = 1e-4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalize(fields, center):
    values = np.asarray(fields, dtype=np.float64)-center
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def support_geometry(features, audit):
    """Independent head-specific centers and case/class/field density weights."""
    owners = np.asarray([f['case_index'] for f in audit['fields']])
    views = np.asarray([f['view'] for f in audit['fields']])
    labels = np.asarray([c['truth5'] for c in audit['cases']])
    fitting = np.asarray(audit['roles']['train'], dtype=int)
    fitting_labels = labels[fitting]
    require(np.all(np.bincount(fitting_labels, minlength=5) > 0), 'Incomplete fitting classes')
    result = []
    for view in range(4):
        case_indices = [np.flatnonzero((owners == case) & (views == view)) for case in fitting]
        require(all(len(indices) for indices in case_indices), 'Fitting case has empty view')
        means = np.asarray([np.asarray(features[ix], dtype=np.float64).mean(0) for ix in case_indices])
        class_means = np.asarray([means[fitting_labels == label].mean(0) for label in range(5)])
        centers = {'binary': (means[fitting_labels == 0].mean(0)+means[fitting_labels > 0].mean(0))/2,
                   'malignant': class_means[1:].mean(0)}
        field_indices = np.concatenate(case_indices)
        field_labels = labels[owners[field_indices]]
        weights = {'binary': np.zeros((len(field_indices), 2)), 'malignant': np.zeros((len(field_indices), 4))}
        offset = 0
        for label, indices in zip(fitting_labels, case_indices):
            count = len(indices)
            binary_label = int(label > 0)
            weights['binary'][offset:offset+count, binary_label] = 1/(int(np.sum((fitting_labels > 0) == binary_label))*count)
            if label > 0:
                weights['malignant'][offset:offset+count, label-1] = 1/(int(np.sum(fitting_labels == label))*count)
            offset += count
        require(all(np.allclose(w.sum(0), 1, rtol=0, atol=1e-12) for w in weights.values()), 'Density class weights differ')
        result.append({'indices': field_indices, 'centers': centers, 'weights': weights,
                       'normalized': {head: normalize(features[field_indices], center) for head, center in centers.items()}})
    return result


def fixed_field_probabilities(features, audit, selected_temperature):
    """Four registered validation settings and only the fixed selected test setting."""
    require(selected_temperature in TEMPERATURES, 'Only the sealed registered test temperature is permitted')
    geometry = support_geometry(features, audit)
    owners = np.asarray([f['case_index'] for f in audit['fields']])
    views = np.asarray([f['view'] for f in audit['fields']])
    output = {}
    for role, temperatures in (('validation', TEMPERATURES), ('test', (selected_temperature,))):
        indices = np.flatnonzero(np.isin(owners, audit['roles'][role])).astype(np.int64)
        positions = {int(index): row for row, index in enumerate(indices)}
        probabilities = [np.full((len(indices), 5), np.nan) for _ in temperatures]
        for view, support in enumerate(geometry):
            query = indices[views[indices] == view]
            cosine = {}
            for head, center in support['centers'].items():
                product = np.einsum('nd,md->nm', normalize(features[query], center), support['normalized'][head], optimize=True)
                require(np.isfinite(product).all(), 'Nonfinite normalized cosine product')
                cosine[head] = np.clip(product, -1., 1.)
            rows = [positions[int(index)] for index in query]
            for candidate, temperature in enumerate(temperatures):
                heads = {}
                for head in ('binary', 'malignant'):
                    scaled = cosine[head]/temperature
                    density = np.exp(scaled-scaled.max(1, keepdims=True)) @ support['weights'][head]
                    require(np.isfinite(density).all() and np.all(density > 0), 'Invalid positive KDE density')
                    heads[head] = density/density.sum(1, keepdims=True)
                joint = np.column_stack((heads['binary'][:, 0], heads['binary'][:, 1, None]*heads['malignant']))
                joint /= joint.sum(1, keepdims=True)
                probabilities[candidate][rows] = joint
        require(all(np.isfinite(p).all() for p in probabilities), 'Incomplete raw-feature query probabilities')
        output[role] = {'indices': indices, 'settings': list(temperatures), 'probabilities': probabilities}
    return output, geometry


def case_scores(audit, role, indices, joint):
    require(len(indices) == len(set(map(int, indices))) and joint.shape == (len(indices), 5)
            and np.isfinite(joint).all() and np.all(joint >= 0)
            and np.allclose(joint.sum(1), 1, rtol=0, atol=1e-10), 'Invalid field probabilities')
    fields = audit['fields']
    role_cases = audit['roles'][role]
    wanted = np.asarray([f['index'] for f in fields if f['case_index'] in set(role_cases)], dtype=np.int64)
    require(np.array_equal(indices, wanted), 'Role field coverage/order differs')
    lookup = {int(index): values for index, values in zip(indices, joint)}
    view_probabilities = []
    for case in role_cases:
        view_probabilities.append([np.mean([lookup[f['index']] for f in fields if f['case_index'] == case and f['view'] == view], axis=0)
                                   for view in range(4)])
    views = np.asarray(view_probabilities)
    pooled = np.stack([views[:, subset].mean(1) for subset in SUBSETS], axis=1)
    pooled /= pooled.sum(-1, keepdims=True)
    predictions = np.where(pooled[:, :, 1:].sum(-1) >= .5, 1+pooled[:, :, 1:].argmax(-1), 0)
    truth = np.asarray([audit['cases'][case]['truth5'] for case in role_cases])
    details = []
    for mask in range(15):
        cm = np.zeros((5, 5), dtype=int)
        np.add.at(cm, (truth, predictions[:, mask]), 1)
        denominator = cm.sum(0)+cm.sum(1)
        f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(5), where=denominator > 0)
        details.append({'accuracy': float(np.trace(cm)/len(truth)), 'macro_f1': float(f1.mean()),
                        'confusion_matrix': cm.tolist(),
                        'binary_accuracy': float(np.mean((truth > 0) == (predictions[:, mask] > 0)))})
    summary = {'cases': len(truth), 'events': len(truth)*15,
               'mean_subset_accuracy': float(np.mean([s['accuracy'] for s in details])),
               'mean_subset_macro_f1': float(np.mean([s['macro_f1'] for s in details])),
               'mean_binary_accuracy': float(np.mean([s['binary_accuracy'] for s in details])),
               'all_four_accuracy': details[-1]['accuracy'], 'all_four_macro_f1': details[-1]['macro_f1']}
    return predictions, summary, details


def execute(args):
    require((args.producer_source/'.source-sha').read_text().strip() == PRODUCER, 'Frozen producer differs')
    require(digest(args.study/'checked/result.json') == RESULT_SHA, 'Accepted comparison bytes differ')
    baseline = read(args.study/'checked/result.json')
    for name, expected in baseline['implementation_sha256'].items():
        require(digest(args.producer_source/name) == expected, 'Frozen scientific source changed')
    config_path = args.producer_source/'configs/breakhis-matched-three-way-v1.json'
    require(digest(config_path) == baseline['config_sha256'], 'Frozen matched configuration differs')
    config = read(config_path)
    audit_path = args.study/'input-audit/private.json'
    require(digest(audit_path) == baseline['input_audit_sha256'], 'Frozen input audit differs')
    audit = read(audit_path)
    fields = audit['fields']
    require(len(fields) == 7909 and len(audit['cases']) == 82, 'Expected all82cases/all7909fields')
    coverage_path = args.producer_source/config['midnight_p']['feature_config']
    require(digest(coverage_path) == config['midnight_p']['feature_config_sha256'], 'Coverage cache registration differs')
    coverage = read(coverage_path)
    features = np.full((7909, 3072), np.nan, dtype=np.float32)
    cached = np.full_like(features, np.nan)
    receipts = {}; seen = set(); maximum_feature_abs = 0.; maximum_scaled_error = 0.
    replayer_sha = digest(Path(__file__).with_name('replay_midnight_raw_features.py'))
    for mag in MAGS:
        folder = args.features_root/str(mag)
        receipt_path = folder/'result.json';receipt = read(receipt_path)
        require(receipt.get('verifier_sha256') == replayer_sha, 'Raw feature verifier source differs')
        require(receipt.get('status') == 'accepted all-field frozen raw Midnight feature shard'
                and receipt.get('magnification') == mag and receipt.get('new_fits') == 0 and receipt.get('new_selections') == 0,
                'Raw Midnight shard did not pass')
        for key, wanted in {'producer_code_sha': PRODUCER, 'original_comparison_sha256': RESULT_SHA,
                            'input_audit_sha256': baseline['input_audit_sha256'], 'crop_forward_batch': 8,
                            'producer_archive_sha256': baseline['source_archive_sha256'],
                            'encoder_revision': 'adc6b15679c981cce6f9b018bbad09d16eeeda9f',
                            'encoder_weights_sha256': '52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1'}.items():
            require(receipt.get(key) == wanted, 'Raw shard provenance differs: '+key)
        wanted = np.asarray([f['index'] for f in fields if int(f['magnification']) == mag], dtype=np.int64)
        require(receipt.get('images') == len(wanted) and receipt.get('raw_crop_forwards') == 3*len(wanted)
                and receipt.get('pixel_cache_equal_fields') == len(wanted) and receipt.get('original_png_hashes_checked') == len(wanted),
                'Raw shard field/crop verification budget differs')
        path = folder/'raw_features.npz'
        require(digest(path) == receipt['artifact_sha256']['raw_features.npz'], 'Raw feature bytes differ')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {'field_indices', 'features', 'crop_features'}, 'Raw feature archive schema differs')
            require(archive['field_indices'].dtype == np.int64 and np.array_equal(archive['field_indices'], wanted), 'Raw feature field coverage/order differs')
            raw = archive['features'].copy();parts = archive['crop_features'].copy()
        require(raw.shape == (len(wanted), 3072) and parts.shape == (len(wanted), 3, 3072)
                and raw.dtype == parts.dtype == np.float32 and np.isfinite(raw).all() and np.isfinite(parts).all(), 'Invalid FP32 raw features')
        fused = ((parts[:, 0].astype(np.float64)+parts[:, 1].astype(np.float64)+parts[:, 2].astype(np.float64))/3).astype(np.float32)
        require(np.array_equal(raw, fused), 'Independent three-crop descriptor fusion differs')
        require(not seen.intersection(map(int, wanted)), 'Duplicated raw field across views')
        seen.update(map(int, wanted));features[wanted] = raw
        old = Path(audit['coverage_cache'])/('midnight-coverage-v1-%d.npz' % mag)
        for p, key in ((old, 'npz_sha256'), (old.with_suffix('.json'), 'json_sha256'),
                       (old.with_name(old.stem+'-components.npz'), 'components_sha256')):
            require(digest(p) == coverage['cache_provenance'][str(mag)][key] == audit['coverage_cache_sha256'][str(mag)][key],
                    'Original cache bytes differ')
        with np.load(old, allow_pickle=False) as archive:
            keys = archive['paths'].tolist();values = archive['features']
            require(len(keys) == len(set(keys)) == len(wanted) and values.shape == raw.shape
                    and values.dtype == np.float32 and np.isfinite(values).all(), 'Original cache field coverage/dtype/value differs')
            lookup = {key: i for i, key in enumerate(keys)}
            expected = np.asarray([values[lookup[fields[index]['relative_path']]] for index in wanted], dtype=np.float32)
        cached[wanted] = expected
        difference = np.abs(raw.astype(np.float64)-expected.astype(np.float64))
        maximum_feature_abs = max(maximum_feature_abs, float(difference.max()))
        maximum_scaled_error = max(maximum_scaled_error, float(np.max(difference/(FEATURE_ATOL+FEATURE_RTOL*np.abs(expected)))))
        require(np.allclose(raw, expected, atol=FEATURE_ATOL, rtol=FEATURE_RTOL), 'Raw features differ above registered cache parity')
        receipts[str(mag)] = {'receipt_sha256': digest(receipt_path), 'features_sha256': digest(path), 'images': len(wanted)}
    require(seen == set(range(7909)) and np.isfinite(features).all() and np.isfinite(cached).all(), 'Incomplete all-field raw feature proof')
    require(len({read(args.features_root/str(mag)/'result.json')['verifier_sha256'] for mag in MAGS}) == 1,
            'Mixed raw feature verifier versions')
    folder = args.study/'models/midnight_p'
    selection_path = folder/'selection.json';selection = read(selection_path)
    require(digest(selection_path) == baseline['models']['midnight_p']['selection_sha256'], 'Original P selection changed')
    require(selection['support_case_indices'] == audit['roles']['train'] and selection['selected_setting'] in TEMPERATURES,
            'Fitting support/fixed temperature differs')
    require(digest(args.study/'test-gate.json') == baseline['test_gate_sha256'], 'Original common gate changed')
    require(tuple(config['midnight_p']['temperatures']) == TEMPERATURES, 'Original four validation temperatures differ')
    output, geometry = fixed_field_probabilities(features, audit, selection['selected_setting'])
    old_geometry = support_geometry(cached, audit)
    center_error = {head: max(float(np.max(np.abs(a['centers'][head]-b['centers'][head])))
                              for a, b in zip(geometry, old_geometry)) for head in ('binary', 'malignant')}
    comparisons = [];scores = [];artifact_hashes = {}
    test_receipt = read(folder/'test/test.json')
    require(test_receipt['selection_sha256'] == digest(selection_path) and test_receipt['gate_sha256'] == baseline['test_gate_sha256'],
            'Original selected-test provenance differs')
    for role in ('validation', 'test'):
        values = output[role]
        for candidate, (setting, joint) in enumerate(zip(values['settings'], values['probabilities'])):
            original = (folder/'candidates'/('candidate%d' % candidate)/'validation-fields.npz') if role == 'validation' else folder/'test/fields.npz'
            expected_sha = selection['candidate_artifact_sha256'][candidate]['validation-fields.npz'] if role == 'validation' else test_receipt['artifact_sha256']['fields.npz']
            require(digest(original) == expected_sha, 'Original query probabilities changed')
            with np.load(original, allow_pickle=False) as archive:
                require(np.array_equal(archive['field_indices'], values['indices']), 'Original role field order differs')
                expected = archive['joint_probabilities5'].copy()
            probability_error = float(np.max(np.abs(joint-expected)))
            require(probability_error <= PROBABILITY_ATOL, 'Raw-feature KDE probability parity exceeds1e-4')
            predictions, summary, details = case_scores(audit, role, values['indices'], joint)
            expected_predictions, expected_summary, _ = case_scores(audit, role, values['indices'], expected)
            require(np.array_equal(predictions, expected_predictions), 'Raw-feature case/subset decision changed')
            require(all(abs(summary[key]-expected_summary[key]) <= 1e-12 for key in summary), 'Raw-feature aggregate score changed')
            if role == 'validation':
                metric = read(original.parent/'metrics.json')['validation']
                scores.append(summary['mean_subset_macro_f1'])
            else:
                metric = baseline['models']['midnight_p']['test']
            for key in ('mean_subset_accuracy', 'mean_subset_macro_f1'):
                require(abs(summary[key]-metric[key]) <= 1e-12, 'Accepted score differs from raw replay')
            name = 'validation-candidate%d.npz' % candidate if role == 'validation' else 'test-selected.npz'
            np.savez_compressed(args.output/name, field_indices=values['indices'], joint_probabilities5=joint, case_subset_predictions=predictions)
            artifact_hashes[name] = digest(args.output/name)
            comparisons.append({'role': role, 'setting': setting, 'original_fields_sha256': expected_sha,
                                'field_probability_max_abs': probability_error,
                                'case_subset_disagreements': 0, 'summary': summary})
    require(np.allclose(scores, selection['candidate_validation_scores'], rtol=0, atol=1e-12)
            and int(np.argmax(scores)) == selection['selected_index'], 'Existing first-maximum selection failed raw replay')
    return {'status': 'accepted complete all-field raw Midnight proof', 'original_comparison_sha256': RESULT_SHA,
            'producer_code_sha': PRODUCER, 'input_audit_sha256': baseline['input_audit_sha256'],
            'original_selection_sha256': digest(selection_path),
            'original_test_receipt_sha256': digest(folder/'test/test.json'), 'raw_images': 7909, 'raw_crop_forwards': 23727,
            'raw_support_cases': 49, 'raw_validation_cases': 17, 'raw_test_cases': 16,
            'shards': receipts, 'feature_parity_atol': FEATURE_ATOL, 'feature_parity_rtol': FEATURE_RTOL,
            'feature_maximum_abs': maximum_feature_abs, 'feature_maximum_scaled_error': maximum_scaled_error,
            'head_center_maximum_abs': center_error, 'field_probability_atol': PROBABILITY_ATOL,
            'raw_field_probability_max_abs': max(c['field_probability_max_abs'] for c in comparisons),
            'case_subset_disagreements': 0, 'comparisons': comparisons, 'artifact_sha256': artifact_hashes,
            'method': 'independent float64 head-specific centering, L2 cosine densities, class/case/field weights and equal-view joint pooling',
            'new_fits': 0, 'new_selections': 0, 'environment': {'numpy': np.__version__, 'python': platform.python_version()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-source', 'study', 'features-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing existing proof output')
    args.output.mkdir(parents=True);started = time.monotonic()
    result = {'status': 'raw Midnight proof failed', 'collector_sha256': digest(__file__), 'new_fits': 0, 'new_selections': 0}
    try:
        result.update(execute(args), elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        import traceback
        result['error_type'] = type(error).__name__
        (args.output/'private-traceback.txt').write_text(traceback.format_exc())
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'].startswith('accepted ') else 1


if __name__ == '__main__':
    raise SystemExit(main())

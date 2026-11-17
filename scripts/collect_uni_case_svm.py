"""Independent frozen-feature and dense libsvm replay of the registered case SVCs.

No fitting, image decoding, encoder calls or new producer-math import occurs.
Public sklearn SVC arrays are sufficient to reconstruct binary/OVO margins and
the OVR vote/confidence transform; joblib APIs are an additional export check.
"""
import argparse
import hashlib
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import traceback

import joblib
import numpy as np
from scipy.special import expit, softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.five_class import write_json
from magset.five_collect import assert_public_safe, read_json, require_equal
from scripts import collect_matched_three_way as metrics
from scripts import collect_specialist_reader as prior_checker

ARMS = ('linear', 'rbf')
CS = (.1, 1., 10., 100.)
SUBSETS = tuple(s for n in range(1, 5) for s in combinations(range(4), n))
MASKS = np.asarray([[int(v in subset) for v in range(4)] for subset in SUBSETS], dtype=np.int64)
CONTEXT_KEYS = ('code_sha', 'source_archive_sha256', 'config_sha256', 'parent_result_sha256',
                'input_audit_sha256', 'specialist_result_sha256')
CANDIDATE_FILES = {'model.joblib', 'state.npz', 'validation-probabilities.npz',
                   'validation-records.jsonl', 'metrics.json', 'reload.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes(order='C')).hexdigest()


def finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    require(np.isfinite(value).all(), name+' contains nonfinite values')
    return value


def close(actual, expected, name, tolerance=1e-10):
    actual, expected = finite(actual, name), finite(expected, name)
    require(actual.shape == expected.shape, name+' shape differs')
    error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
    require(error <= tolerance, name+' differs')
    return error


def case_means(audit, features):
    """Equal original fields within each canonical case/view, in FP64."""
    values = np.empty((len(audit['cases']), 4, features.shape[1]), dtype=np.float64)
    for case in range(len(values)):
        for view in range(4):
            selected = [f['index'] for f in audit['fields'] if f['case_index'] == case and f['view'] == view]
            require(selected, 'Empty canonical case/view')
            values[case, view] = np.asarray(features[selected], dtype=np.float64).mean(0)
    return finite(values, 'case-view means')


def normalization(means, train):
    selected = means[np.asarray(train, dtype=np.int64)]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0, ddof=0)
    return mean, std, np.where(std < 1e-6, 1., std)


def descriptors(means, mean, scale, cases):
    transformed = (means[np.asarray(cases, dtype=np.int64)]-mean)/scale
    samples = []
    for subset, mask in zip(SUBSETS, MASKS):
        pooled = transformed[:, subset, :].mean(axis=1)
        embedding = pooled/np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        samples.append(np.concatenate((embedding, np.broadcast_to(.5*mask, (len(cases), 4))), axis=1))
    result = np.stack(samples, axis=1).reshape(len(cases)*15, means.shape[-1]+4)
    return np.ascontiguousarray(finite(result, 'case-mask descriptors'))


def verify_preprocessing(path, audit, features):
    expected_means = case_means(audit, features)
    train = np.asarray(audit['roles']['train'], dtype=np.int64)
    mean, std, scale = normalization(expected_means, train)
    expected = {'case_view_means': expected_means, 'mean': mean, 'std': std, 'scale': scale,
                'train_case_indices': train, 'case_indices': np.arange(len(audit['cases']), dtype=np.int64), 'masks': MASKS}
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), 'Preprocessing archive schema differs')
        for key, wanted in expected.items():
            actual = saved[key]
            require(actual.dtype == wanted.dtype and np.array_equal(actual, wanted), 'Independent preprocessing differs/'+key)
    x = {role: descriptors(expected_means, mean, scale, cases) for role, cases in audit['roles'].items()}
    labels = np.asarray([c['truth5'] for c in audit['cases']], dtype=np.int64)
    y = {role: np.repeat(labels[cases], 15) for role, cases in audit['roles'].items()}
    return x, y, expected


def kernel_matrix(x, support, kernel, gamma):
    dot = finite(x @ support.T, 'raw kernel dot product')
    if kernel == 'linear':
        return dot
    require(kernel == 'rbf' and np.isfinite(gamma) and gamma > 0, 'Invalid registered kernel/gamma')
    distance = finite(np.sum(x*x, axis=1)[:, None]+np.sum(support*support, axis=1)[None, :]-2*dot,
                      'raw squared distances')
    return finite(np.exp(-gamma*np.maximum(distance, 0.)), 'RBF kernel')


def raw_margins(state, x, kernel):
    """Public coefficients already have sklearn's binary sign convention."""
    count = len(state['classes_'])
    values = kernel_matrix(x, state['support_vectors_'], kernel, float(state['actual_gamma']))
    dual, intercept = state['dual_coef_'], state['intercept_']
    if count == 2:
        return finite(values @ dual[0]+intercept[0], 'binary margin')
    offsets = np.r_[0, np.cumsum(state['n_support_'])]
    outputs = []
    for pair, (i, j) in enumerate(combinations(range(count), 2)):
        a, b = slice(offsets[i], offsets[i+1]), slice(offsets[j], offsets[j+1])
        outputs.append(values[:, a] @ dual[j-1, a]+values[:, b] @ dual[i, b]+intercept[pair])
    return finite(np.column_stack(outputs), 'OVO margins')


def ovr_from_ovo(pairwise, count=4):
    """Independent equivalent of pinned sklearn's public OVR convention."""
    votes = np.zeros((len(pairwise), count), dtype=np.float64)
    confidence = np.zeros_like(votes)
    for column, (i, j) in enumerate(combinations(range(count), 2)):
        margin = pairwise[:, column]
        votes[:, i] += margin >= 0
        votes[:, j] += margin < 0
        confidence[:, i] += margin
        confidence[:, j] -= margin
    return votes+confidence/(3*(np.abs(confidence)+1))


def scores_and_labels(binary, malignant):
    binary, malignant = finite(binary, 'binary margins'), finite(malignant, 'malignant OVR scores')
    require(binary.ndim == 1 and malignant.shape == (len(binary), 4), 'Head margin dimensions differ')
    b, q = expit(binary), softmax(malignant, axis=1)
    joint = np.column_stack((1-b, b[:, None]*q))
    metrics.probabilities(joint, (len(binary), 5), 'uncalibrated normalized scores', tolerance=1e-12)
    prediction = np.where(binary >= 0, 1+malignant.argmax(1), 0).astype(np.int64)
    require(np.array_equal(binary >= 0, b >= .5), 'Finite sigmoid route differs from authoritative margin route')
    return joint, prediction


def state_head(saved, prefix):
    return {key[len(prefix)+1:]: saved[key].copy() for key in saved.files if key.startswith(prefix+'_')}


def validate_head(state, model, x, y, train_cases, kernel, c):
    """Validate exports, fitting-support membership and fixed solver semantics."""
    from sklearn.svm import SVC
    require(type(model) is SVC, 'Saved head is not the registered SVC estimator')
    parameters = {'kernel': kernel, 'C': c, 'gamma': 'scale', 'class_weight': 'balanced', 'probability': False,
                  'decision_function_shape': 'ovr', 'break_ties': True, 'random_state': 812, 'tol': 1e-4,
                  'shrinking': True, 'max_iter': 100000, 'cache_size': 256}
    for key, expected in parameters.items():
        require_equal(model.get_params().get(key), expected, 'saved SVC parameter/'+key, 0)
    classes = np.unique(y)
    require(np.array_equal(classes, np.arange(len(classes))) and len(classes) in (2, 4), 'Fitting class order differs')
    integer_keys = ('n_support_', 'support_', 'classes_', 'n_iter_', 'training_case_indices')
    for key in integer_keys:
        require(state[key].dtype == np.dtype('int64'), 'Dense head integer dtype differs/'+key)
    for key in ('support_vectors_', 'dual_coef_', 'intercept_', 'class_weight_'):
        require(state[key].dtype == np.dtype('float64'), 'Dense head float dtype differs/'+key)
        finite(state[key], key)
    require(np.array_equal(state['classes_'], classes), 'Dense head class order differs')
    require(np.array_equal(state['training_case_indices'], train_cases), 'Dense head fitting cases differ')
    support = state['support_']
    require(support.ndim == 1 and len(set(support.tolist())) == len(support) and np.all((support >= 0) & (support < len(x))),
            'Support indices are not unique fitting-row indices')
    require(state['support_vectors_'].shape == (len(support), x.shape[1])
            and np.array_equal(state['support_vectors_'], x[support]), 'Support vectors differ from fitting inputs')
    n_support = state['n_support_']
    require(n_support.shape == classes.shape and np.all(n_support > 0) and n_support.sum() == len(support), 'Support class counts differ')
    require(np.array_equal(y[support], np.repeat(classes, n_support)), 'Support vectors not grouped by fitting class')
    count = len(classes)
    require(state['dual_coef_'].shape == (count-1, len(support))
            and state['intercept_'].shape == (count*(count-1)//2,), 'Dense coefficient dimensions differ')
    require(np.asarray(state['actual_gamma']).shape == () and np.asarray(state['fit_status_']).shape == (), 'Head scalar shapes differ')
    variance = x.var()
    gamma = 1/(x.shape[1]*variance) if variance != 0 else 1.
    close(state['actual_gamma'], gamma, 'resolved training-only gamma', 1e-12)
    require(int(state['fit_status_']) == model.fit_status_ == 0, 'SVC did not converge')
    iterations = state['n_iter_']
    require(iterations.shape == (count*(count-1)//2,) and np.all((iterations >= 0) & (iterations <= 100000)), 'SVC iteration receipt differs')
    weights = np.full(len(y), 1/15, dtype=np.float64)
    require(str(state['training_features_sha256'].item()) == array_hash(x), 'Fitting descriptor hash differs')
    require(str(state['sample_weight_sha256'].item()) == array_hash(weights), 'Fitting sample weights differ')
    require(str(state['training_targets_sha256'].item()) == hashlib.sha256(np.asarray(y, dtype='<i8').tobytes()).hexdigest(),
            'Fitting target hash differs')
    close(state['sample_weight_sum'], len(train_cases), 'one normalized mask mass per case', 1e-12)
    class_weight = len(y)/(count*np.bincount(y, minlength=count))
    close(state['class_weight_'], class_weight, 'balanced class weights', 1e-12)
    effective = weights*class_weight[y]
    require(str(state['effective_weight_sha256'].item()) == array_hash(effective), 'Effective class/sample weight hash differs')
    close(state['effective_weight_sum'], len(train_cases), 'effective class/sample weight mass', 1e-12)
    require(np.array_equal(state['support_case_indices'], np.repeat(train_cases, 15)[support]), 'Support case ownership differs')
    for key in ('support_vectors_', 'dual_coef_', 'intercept_', 'n_support_', 'support_', 'classes_', 'class_weight_', 'n_iter_'):
        require(np.array_equal(state[key], getattr(model, key)), 'Joblib/public-state export differs/'+key)
    close(state['actual_gamma'], model._gamma, 'joblib gamma', 0)
    require(model.shape_fit_ == x.shape and model.n_features_in_ == x.shape[1], 'Joblib fitting shape differs')
    require(model.probA_.size == model.probB_.size == 0, 'Unexpected probability calibration parameters')
    offsets = np.r_[0, np.cumsum(n_support)]
    for i, j in combinations(range(count), 2):
        if count == 2:
            a, b = state['dual_coef_'][0, :n_support[0]], state['dual_coef_'][0, n_support[0]:]
            signed = np.r_[-a, b]
        else:
            a = state['dual_coef_'][j-1, offsets[i]:offsets[i+1]]
            b = state['dual_coef_'][i, offsets[j]:offsets[j+1]]
            signed = np.r_[a, -b]
        bound = np.r_[np.full(len(a), c*class_weight[i]/15), np.full(len(b), c*class_weight[j]/15)]
        require(np.all(signed >= -1e-12) and np.all(signed <= bound+1e-12), 'SVC coefficients exceed registered sample/class C bounds')
        require(abs(float(a.sum()+b.sum())) <= 1e-8, 'SVC pairwise dual equality differs')
    return {'training_cases': len(train_cases), 'training_rows': len(x), 'sample_weight_sum': float(weights.sum()),
            'class_weight': class_weight.tolist(), 'actual_gamma': float(state['actual_gamma']),
            'support_vectors': len(support), 'fit_status': 0, 'iterations': iterations.tolist()}


def independent_predictions(binary, malignant, x, kernel):
    margin = raw_margins(binary, x, kernel)
    type_scores = ovr_from_ovo(raw_margins(malignant, x, kernel))
    joint, prediction = scores_and_labels(margin, type_scores)
    return margin, type_scores, joint, prediction


def classification(truth, predictions, classes):
    base = metrics.classification(truth, predictions, classes)
    return {key: value for key, value in base.items() if key != 'weight_sum'}


def auc(truth, margin):
    """Pairwise Mann-Whitney ROC AUC, with half credit for score ties."""
    truth = np.asarray(truth, dtype=bool)
    if truth.all() or not truth.any():
        return None
    difference = margin[truth, None]-margin[None, ~truth]
    return float(np.mean((difference > 0)+.5*(difference == 0)))


def summary(audit, role, margins, predictions):
    cases = audit['roles'][role]
    truth = np.asarray([audit['cases'][i]['truth5'] for i in cases], dtype=np.int64)
    require(margins.shape == predictions.shape == (len(cases), 15), 'Case/subset summary shape differs')
    subsets = {}
    for column, subset in enumerate(SUBSETS):
        name = '+'.join(str(metrics.MAGNIFICATIONS[v]) for v in subset)
        result = metrics.single_subset_summary(truth, predictions[:, column])
        result['five_class'] = classification(truth, predictions[:, column], 5)
        result['binary'] = classification((truth > 0).astype(np.int64), (predictions[:, column] > 0).astype(np.int64), 2)
        result['binary']['roc_auc'] = auc(truth > 0, margins[:, column])
        subsets[name] = result
    return {'cases': len(cases), 'case_subset_events': len(cases)*15,
            'mean_subset_macro_f1': float(np.mean([s['five_class']['macro_f1'] for s in subsets.values()])),
            'mean_subset_accuracy': float(np.mean([s['five_class']['accuracy'] for s in subsets.values()])),
            'mean_subset_binary_accuracy': float(np.mean([s['binary']['accuracy'] for s in subsets.values()])),
            'worst_subset_macro_f1': float(min(s['five_class']['macro_f1'] for s in subsets.values())),
            'worst_subset_accuracy': float(min(s['five_class']['accuracy'] for s in subsets.values())),
            'all_four': subsets['40+100+200+400'], 'subsets': subsets}


def verify_predictions(path, record_path, audit, role, expected):
    role_cases = np.asarray(audit['roles'][role], dtype=np.int64)
    n = len(role_cases)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'case_indices', 'masks', 'binary_margin', 'malignant_scores', 'joint_probabilities5', 'predictions'},
                'Case probability archive schema differs')
        indices, masks = saved['case_indices'], saved['masks']
        margin, malignant, joint, prediction = (saved[key].copy() for key in
            ('binary_margin', 'malignant_scores', 'joint_probabilities5', 'predictions'))
    require(indices.dtype == masks.dtype == np.dtype('int64') and np.array_equal(indices, role_cases)
            and np.array_equal(masks, MASKS), 'Case probability archive case/mask order differs')
    require(margin.dtype == malignant.dtype == joint.dtype == np.dtype('float64') and prediction.dtype == np.dtype('int64'),
            'Case probability archive dtypes differ')
    require(margin.shape == (n, 15) and malignant.shape == (n, 15, 4) and joint.shape == (n, 15, 5)
            and prediction.shape == (n, 15), 'Case probability archive dimensions differ')
    own_joint, own_prediction = scores_and_labels(margin.reshape(-1), malignant.reshape(-1, 4))
    close(joint, own_joint.reshape(n, 15, 5), 'Saved normalized score transform', 1e-12)
    require(np.array_equal(prediction, own_prediction.reshape(n, 15)), 'Saved authoritative margin routing differs')
    e_margin, e_malignant, e_joint, e_prediction = expected
    maximum = max(close(margin.reshape(-1), e_margin, 'Independent binary margin'),
                  close(malignant.reshape(-1, 4), e_malignant, 'Independent malignant OVR scores'),
                  close(joint.reshape(-1, 5), e_joint, 'Independent normalized scores'))
    require(np.array_equal(prediction.reshape(-1), e_prediction), 'Independent replay changes a hard decision')
    records = []
    for row, case in enumerate(role_cases):
        original = audit['cases'][int(case)]
        for column, subset in enumerate(SUBSETS):
            records.append({'role': role, 'case_id': original['case_id'], 'truth': original['truth8'], 'truth5': original['truth5'],
                'magnifications': [metrics.MAGNIFICATIONS[v] for v in subset], 'binary_margin': float(margin[row, column]),
                'malignant_scores4': malignant[row, column].tolist(), 'binary_score': float(expit(margin[row, column])),
                'probabilities5': joint[row, column].tolist(), 'prediction': int(prediction[row, column]),
                'score_interpretation': 'uncalibrated normalized scores; margin-based routing'})
    require_equal(metrics.read_records(record_path), records, 'Independent case/mask records', 1e-12)
    return summary(audit, role, margin, prediction), maximum


def source_context(args):
    repo = Path(__file__).resolve().parents[1]
    require(re.fullmatch('[a-f0-9]{40}', args.code_sha or '') is not None
            and (repo/'.source-sha').read_text().strip() == args.code_sha, 'Immutable case-SVC source required')
    archive = (repo/'.archive-sha256').read_text().strip()
    require(re.fullmatch('[a-f0-9]{64}', archive) is not None, 'Immutable archive digest required')
    require(file_hash(args.config) == file_hash(repo/'configs/breakhis-uni-case-svm-v1.json'), 'Canonical case-SVC config differs')
    cfg = read_json(args.config)
    require(cfg['name'] == 'breakhis-uni-case-svm-v1' and cfg['arms'] == list(ARMS)
            and cfg['primary_arm'] == 'rbf' and cfg['svc']['C_values'] == list(CS)
            and cfg['fusion']['final_dimension'] == 1540, 'Registered case-SVC scope differs')
    for name, version in cfg['runtime'].items():
        require(importlib.metadata.version(name) == version, 'Pinned runtime differs/'+name)
    context = read_json(Path(args.root)/'context.json')
    expected = {'code_sha': args.code_sha, 'source_archive_sha256': archive, 'config_sha256': file_hash(args.config),
                'parent_result_sha256': cfg['matched_reference']['sha256'],
                'input_audit_sha256': cfg['input_audit']['private_sha256'],
                'specialist_result_sha256': cfg['contextual_specialist_reference']['sha256']}
    for key, value in expected.items():
        require_equal(context.get(key), value, 'case-SVC context/'+key, 0)
    for key, value in {'manifest_sha256': cfg['manifest_sha256'], 'runtime_versions': cfg['runtime'],
                       'feature_dimension': 1536, 'input_dimension': 1540, 'case_view_shape': [82, 4, 1536]}.items():
        require_equal(context.get(key), value, 'case-SVC context/'+key, 0)
    refs = cfg['references']; pins = {}
    for key in ('uni_feature_config', 'uni_extraction_config'):
        require(file_hash(repo/refs[key]) == refs[key+'_sha256'], 'Referenced UNI config differs/'+key)
        pins[refs[key]] = refs[key+'_sha256']
    require_equal(context.get('reference_config_sha256'), pins, 'Referenced UNI config map', 0)
    h_path = repo/cfg['contextual_specialist_reference']['file']
    require(file_hash(h_path) == cfg['contextual_specialist_reference']['sha256']
            and read_json(h_path)['status'] == 'complete accepted specialist-reader family', 'Contextual specialist evidence differs')
    implementation = context.get('implementation_sha256')
    require(isinstance(implementation, dict) and {'magset/uni_case_svm.py', 'scripts/run_uni_case_svm.py'} <= set(implementation),
            'Case-SVC implementation receipts missing')
    for path, digest in implementation.items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts and file_hash(repo/path) == digest,
                'Case-SVC implementation differs/'+path)
    return cfg, context, expected, repo


def dataset_inputs(args, cfg, context, repo):
    audit, counts, _ = prior_checker.parent_inputs(args, cfg, repo)
    train = audit['roles']['train']
    malignant = [i for i in train if audit['cases'][i]['truth5'] > 0]
    require_equal(context.get('train_case_indices'), train, 'Case-SVC fitting case order', 0)
    require_equal(context.get('malignant_train_case_indices'), malignant, 'Case-SVC malignant fitting order', 0)
    require_equal(context.get('field_alignment_sha256'), prior_checker.alignment_digest(audit), 'Field path/label/view alignment', 0)
    reference = read_json(repo/cfg['references']['uni_feature_config'])
    require_equal(context.get('cache_provenance'), {'uni': reference['cache_provenance']}, 'Case-SVC cache provenance', 0)
    features = prior_checker.load_features(context['uni_cache'], reference, audit, 1536)
    return audit, counts, features


def verify_reload(receipt, model_sha, state_sha, validation_cases):
    require(receipt.get('status') == 'accepted validation model reload' and receipt.get('decisions_identical') is True
            and receipt.get('atol') == 1e-10, 'Validation reload guard did not pass')
    for key in ('binary_margin_max_abs', 'malignant_score_max_abs', 'joint_probability_max_abs'):
        value = receipt.get(key)
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1e-10, 'Validation reload error differs/'+key)
    for key, expected in {'model_sha256': model_sha, 'state_sha256': state_sha,
                          'validation_cases': validation_cases, 'masks': 15, 'test_queries': 0}.items():
        require_equal(receipt.get(key), expected, 'Validation reload binding/'+key, 0)


def verify_selection(selection, arm, scores, context, context_sha, preprocessing_sha):
    require(selection.get('status') == 'selected validation candidate' and selection.get('arm') == arm
            and selection.get('test_predictions') == 0, 'Case-SVC validation selection status differs')
    for key in CONTEXT_KEYS:
        require_equal(selection.get(key), context[key], 'Case-SVC selection context/'+key, 0)
    require_equal(selection.get('context_sha256'), context_sha, 'Case-SVC selection context bytes', 0)
    require_equal(selection.get('preprocessing_sha256'), preprocessing_sha, 'Case-SVC selection preprocessing', 0)
    require_equal(selection.get('candidate_validation_scores'), scores, 'Case-SVC independent candidate scores', 1e-12)
    index = selection.get('selected_index')
    require(type(index) is int and index == int(np.argmax(scores)) and selection.get('selected_setting') == CS[index],
            'Case-SVC selection is not first maximum/shared C')
    return index


def verify_gate(gate, hashes, context_sha, preprocessing_sha, audit_sha):
    require(gate.get('status') == 'sealed both case-SVC selections' and gate.get('test_queries') == 0,
            'Both case-SVC selections were not sealed')
    for key, value in {'selections': hashes, 'context_sha256': context_sha, 'preprocessing_sha256': preprocessing_sha,
                       'input_audit_sha256': audit_sha}.items():
        require_equal(gate.get(key), value, 'Case-SVC gate/'+key, 0)


def load_candidate(folder, preprocessing, audit, x, y, kernel, c):
    bundle = joblib.load(Path(folder)/'model.joblib')
    require(isinstance(bundle, dict) and bundle.get('kernel') == kernel and bundle.get('C') == c, 'Saved head bundle identity differs')
    for key in ('mean', 'std', 'scale', 'train_case_indices'):
        require(np.array_equal(bundle[key], preprocessing[key]), 'Joblib normalizer persistence differs/'+key)
    train = np.asarray(audit['roles']['train'], dtype=np.int64)
    labels = np.asarray([case['truth5'] for case in audit['cases']], dtype=np.int64)
    malignant_cases = train[labels[train] > 0]
    require(np.array_equal(bundle['malignant_train_case_indices'], malignant_cases), 'Joblib malignant fitting cases differ')
    targets = {'binary': (y['train'] > 0).astype(np.int64), 'malignant': y['train'][y['train'] > 0]-1}
    inputs = {'binary': x['train'], 'malignant': x['train'][y['train'] > 0]}
    states, proof = {}, {}
    with np.load(Path(folder)/'state.npz', allow_pickle=False) as saved:
        for head in ('binary', 'malignant'):
            states[head] = state_head(saved, head)
            selected = train if head == 'binary' else malignant_cases
            proof[head] = validate_head(states[head], bundle[head], inputs[head], targets[head], selected, kernel, c)
            receipt = bundle['training_receipts'][head]
            expected = {'case_indices': selected.tolist(), 'rows': len(inputs[head]),
                'training_features_sha256': states[head]['training_features_sha256'].item(),
                'training_targets_sha256': states[head]['training_targets_sha256'].item(),
                'sample_weight_sha256': states[head]['sample_weight_sha256'].item(),
                'effective_weight_sha256': states[head]['effective_weight_sha256'].item(),
                'sample_weight_sum': float(states[head]['sample_weight_sum']),
                'effective_weight_sum': float(states[head]['effective_weight_sum']),
                'actual_gamma': float(states[head]['actual_gamma']), 'class_weight': states[head]['class_weight_'].tolist(),
                'fit_status': 0, 'n_iter': states[head]['n_iter_'].tolist(), 'n_support': states[head]['n_support_'].tolist()}
            require_equal(receipt, expected, 'Joblib independent training receipt/'+head, 1e-12)
    own = independent_predictions(states['binary'], states['malignant'], x['validation'], kernel)
    close(own[0], bundle['binary'].decision_function(x['validation']), 'Independent/joblib binary validation margins')
    close(own[1], bundle['malignant'].decision_function(x['validation']), 'Independent/joblib malignant validation scores')
    return states, bundle, proof, own


def collect(args):
    output = Path(args.output)
    require(not output.exists(), 'Refusing existing case-SVC collector output')
    result = {'status': 'failed or incomplete UNI case-SVC family', 'collector_new_fits': 0,
              'collector_new_encoder_forwards': 0, 'collector_new_image_decodes': 0,
              'margin_replay': 'independent dense support-vector/dual/intercept kernel and OVO-to-OVR arithmetic; '
                               'joblib decision API used additionally to verify exports',
              'interpretation': 'Prespecified standard case-unit linear versus RBF SVC development comparison. '
              'The 16-case follow-up was already viewed; this is retrospective reused-development evidence, '
              'not fresh clinical confirmation. >90% final-five and >97% binary are unpromised objectives.'}
    private = {'errors': {}}
    try:
        cfg, context, public, repo = source_context(args)
        result.update(public)
        audit, counts, features = dataset_inputs(args, cfg, context, repo)
        root = Path(args.root); context_sha = file_hash(root/'context.json'); preprocessing_sha = file_hash(root/'preprocessing.npz')
        require_equal(context.get('preprocessing_sha256'), preprocessing_sha, 'Context/preprocessing binding', 0)
        x, y, preprocessing = verify_preprocessing(root/'preprocessing.npz', audit, features)
        require(x['train'].shape == (735, 1540) and np.sum(y['train'] > 0) == 525, 'Registered fitting input shapes differ')
        checked, selections, selection_hashes, selected_states = {}, {}, {}, {}
        for arm in ARMS:
            folder = root/'candidates'/arm
            require({p.name for p in folder.iterdir()} == {'candidate%d' % i for i in range(4)}, 'Unexpected case-SVC candidate family')
            selection_path = root/'arms'/arm/'selection.json'; selection = read_json(selection_path)
            hashes = selection.get('candidate_artifact_sha256', [])
            require(len(hashes) == 4, 'Four complete case-SVC candidate receipts required')
            candidates, scores, states_by_index = [], [], []
            for index, c in enumerate(CS):
                candidate = folder/('candidate%d' % index)
                require(set(hashes[index]) == CANDIDATE_FILES, 'Case-SVC candidate artifact schema differs')
                metrics.checked_hashes(candidate, hashes[index], CANDIDATE_FILES)
                verify_reload(read_json(candidate/'reload.json'), file_hash(candidate/'model.joblib'),
                              file_hash(candidate/'state.npz'), len(audit['roles']['validation']))
                states, bundle, training, own = load_candidate(candidate, preprocessing, audit, x, y, arm, c)
                independent, error = verify_predictions(candidate/'validation-probabilities.npz', candidate/'validation-records.jsonl',
                                                        audit, 'validation', own)
                reported = read_json(candidate/'metrics.json')
                require(reported.get('setting') == c, 'Case-SVC candidate C differs')
                require_equal(reported.get('validation'), independent, 'Independent case-SVC validation metrics', 1e-12)
                candidates.append({'index': index, 'C': c, 'validation': independent,
                                   'training': training, 'maximum_margin_or_score_abs': error})
                scores.append(independent['mean_subset_macro_f1']); states_by_index.append((states, bundle))
            index = verify_selection(selection, arm, scores, context, context_sha, preprocessing_sha)
            selections[arm] = selection; selection_hashes[arm] = file_hash(selection_path)
            selected_states[arm] = states_by_index[index]
            checked[arm] = {'selected_index': index, 'selected_C': CS[index], 'candidates': candidates,
                            'selected_validation': candidates[index]['validation']}
        gate_path = root/'family-gate.json'
        verify_gate(read_json(gate_path), selection_hashes, context_sha, preprocessing_sha, cfg['input_audit']['private_sha256'])
        for arm in ARMS:
            folder = root/'arms'/arm/'test'; receipt = read_json(folder/'test.json'); index = selections[arm]['selected_index']
            candidate = root/'candidates'/arm/('candidate%d' % index)
            require(receipt.get('status') == 'complete selected retrospective inference' and receipt.get('arm') == arm,
                    'Case-SVC selected retrospective inference incomplete')
            for key, value in {'selected_index': index, 'selected_setting': CS[index], 'context_sha256': context_sha,
                               'preprocessing_sha256': preprocessing_sha, 'selection_sha256': selection_hashes[arm],
                               'gate_sha256': file_hash(gate_path), 'selected_model_sha256': file_hash(candidate/'model.joblib'),
                               'selected_state_sha256': file_hash(candidate/'state.npz')}.items():
                require_equal(receipt.get(key), value, 'Selected case-SVC retrospective receipt/'+key, 0)
            require(set(receipt.get('artifact_sha256', {})) == {'probabilities.npz', 'records.jsonl', 'metrics.json'},
                    'Selected case-SVC retrospective artifact schema differs')
            metrics.checked_hashes(folder, receipt['artifact_sha256'])
            states, bundle = selected_states[arm]
            own = independent_predictions(states['binary'], states['malignant'], x['test'], arm)
            close(own[0], bundle['binary'].decision_function(x['test']), 'Independent/joblib binary selected follow-up')
            close(own[1], bundle['malignant'].decision_function(x['test']), 'Independent/joblib malignant selected follow-up')
            independent, error = verify_predictions(folder/'probabilities.npz', folder/'records.jsonl', audit, 'test', own)
            require_equal(read_json(folder/'metrics.json'), independent, 'Independent case-SVC retrospective metrics', 1e-12)
            checked[arm].update(retrospective_follow_up=independent, follow_up_maximum_margin_or_score_abs=error)
        linear, rbf = (checked[arm]['selected_validation'] for arm in ARMS)
        delta = rbf['mean_subset_macro_f1']-linear['mean_subset_macro_f1']
        accuracy_delta = rbf['mean_subset_accuracy']-linear['mean_subset_accuracy']
        screen = {'validation_macro_f1_delta': delta, 'validation_accuracy_delta': accuracy_delta,
                  'passed': delta >= cfg['progression_screen']['mean15_macro_f1_delta_at_least'] and accuracy_delta >= 0,
                  'scope': 'validation progression only; retrospective follow-up cannot override this decision'}
        result.update(status='complete accepted UNI case-SVC family', roles=counts, context_sha256=context_sha,
                      preprocessing_sha256=preprocessing_sha, family_gate_sha256=file_hash(gate_path), arms=checked,
                      progression_screen=screen, head_estimator_fits=16, binary_optimization_subproblems=56,
                      validation_case_subset_records=2040, retrospective_case_subset_records=480)
    except Exception as error:
        private['errors']['review'] = {'message': str(error), 'traceback': traceback.format_exc()}
        result['review'] = {'status': 'invalid_or_incomplete', 'error_type': type(error).__name__}
    assert_public_safe(result)
    output.mkdir(parents=True)
    write_json(output/'private_review.json', private)
    result['private_review_sha256'] = file_hash(output/'private_review.json')
    write_json(output/'result.json', result)
    lines = ['# UNI2-h case-SVC independent check', '', 'Status: '+result['status']+'.', '']
    if 'arms' in result:
        for arm, value in result['arms'].items():
            score = value['retrospective_follow_up']
            lines.append(f"{arm}: selected C {value['selected_C']}; retrospective mean15 accuracy "
                         f"{100*score['mean_subset_accuracy']:.2f}%, macro F1 {score['mean_subset_macro_f1']:.6f}, "
                         f"binary accuracy {100*score['mean_subset_binary_accuracy']:.2f}%.")
    else:
        lines.append('No model performance is accepted.')
    lines.extend(['', result['interpretation'], '', 'The collector fitted no model and decoded or encoded no image.', ''])
    (output/'summary.md').write_text('\n'.join(lines))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'config', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--code-sha', required=True)
    return p


if __name__ == '__main__':
    result = collect(parser().parse_args())
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'complete accepted UNI case-SVC family' else 1)

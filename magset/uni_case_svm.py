"""Registered case/mask UNI2-h SVC math; no cache I/O or split selection."""
import hashlib
from itertools import combinations

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

ARMS = ('linear', 'rbf')
CS = (.1, 1., 10., 100.)
MAGNIFICATIONS = (40, 100, 200, 400)
SUBSETS = tuple(s for size in range(1, 5) for s in combinations(range(4), size))
MASKS = np.asarray([[int(view in subset) for view in range(4)] for subset in SUBSETS], dtype=np.int64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(values, dtype='<f8'):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def checked_views(values):
    values = np.asarray(values)
    require(values.ndim == 3 and values.shape[1] == 4 and values.shape[2] > 0
            and values.dtype.kind == 'f' and np.isfinite(values).all(), 'Finite case/four-view features required')
    return values.astype(np.float64, copy=False)


def checked_indices(indices, count):
    values = np.asarray(indices)
    require(values.ndim == 1 and len(values) > 0 and values.dtype.kind in 'iu'
            and len(set(values.tolist())) == len(values) and np.all((values >= 0) & (values < count)),
            'Unique nonempty canonical case indices required')
    return values.astype(np.int64)


def case_view_means(bags):
    require(len(bags) > 0, 'Nonempty frozen field bags required')
    means = []
    for bag in bags:
        require(len(bag) == 4, 'Every case requires four views')
        views = []
        for fields in bag:
            values = np.asarray(fields)
            require(values.ndim == 2 and len(values) > 0 and values.dtype.kind == 'f'
                    and np.isfinite(values).all(), 'Finite nonempty field matrix required')
            views.append(values.astype(np.float64).mean(axis=0))
        means.append(views)
    return checked_views(np.asarray(means, dtype=np.float64))


def fit_preprocessing(values, train_case_indices):
    values = checked_views(values); train = checked_indices(train_case_indices, len(values))
    mean = values[train].mean(axis=0)
    std = values[train].std(axis=0, ddof=0)
    scale = np.where(std < 1e-6, 1., std)
    return {'mean': mean, 'std': std, 'scale': scale, 'train_case_indices': train.copy()}


def vectorize(values, preprocessing, case_indices):
    """Case-major 15-mask rows: active mean, L2, then half-scale flags."""
    values = checked_views(values); indices = checked_indices(case_indices, len(values))
    mean, scale = np.asarray(preprocessing['mean']), np.asarray(preprocessing['scale'])
    require(mean.shape == scale.shape == values.shape[1:] and np.isfinite(mean).all()
            and np.isfinite(scale).all() and np.all(scale > 0), 'Frozen per-view normalizer differs')
    transformed = (values[indices] - mean) / scale
    rows = []
    for subset, mask in zip(SUBSETS, MASKS):
        pooled = transformed[:, subset, :].mean(axis=1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        rows.append(np.concatenate((pooled, np.broadcast_to(.5 * mask, (len(indices), 4))), axis=1))
    result = np.stack(rows, axis=1)
    require(np.isfinite(result).all(), 'Nonfinite normalized case/mask features')
    return result


def fit_bundle(values, labels, preprocessing, kernel, C):
    values = checked_views(values); labels = np.asarray(labels)
    require(labels.shape == (len(values),) and labels.dtype.kind in 'iu'
            and np.all((labels >= 0) & (labels < 5)), 'Canonical final-five case labels required')
    require(kernel in ARMS and not isinstance(C, (bool, np.bool_)) and C in CS, 'Registered kernel/shared C required')
    train = checked_indices(preprocessing['train_case_indices'], len(values))
    require(np.all(np.bincount(labels[train], minlength=5) > 0), 'All five fitting classes required')
    malignant = train[labels[train] > 0]
    bundle = {'mean': np.asarray(preprocessing['mean']).copy(), 'std': np.asarray(preprocessing['std']).copy(),
              'scale': np.asarray(preprocessing['scale']).copy(), 'train_case_indices': train.copy(),
              'malignant_train_case_indices': malignant.copy(), 'kernel': kernel, 'C': float(C),
              'training_receipts': {}}
    for name, indices, y in (('binary', train, (labels[train] > 0).astype(np.int64)),
                              ('malignant', malignant, (labels[malignant] - 1).astype(np.int64))):
        X = vectorize(values, preprocessing, indices).reshape(len(indices) * 15, values.shape[-1] + 4)
        targets = np.repeat(y, 15)
        weights = np.full(len(targets), 1. / 15., dtype=np.float64)
        model = SVC(kernel=kernel, C=float(C), gamma='scale', class_weight='balanced', probability=False,
                    decision_function_shape='ovr', break_ties=True, random_state=812, tol=1e-4,
                    shrinking=True, max_iter=100000, cache_size=256)
        model.fit(X, targets, sample_weight=weights)
        classes = np.arange(2 if name == 'binary' else 4, dtype=np.int64)
        require(model.fit_status_ == 0 and np.array_equal(model.classes_, classes), 'SVC convergence or canonical class order differs')
        require(np.array_equal(model.support_vectors_, X[model.support_]), 'SVC support rows differ from fitting inputs')
        effective = weights * model.class_weight_[targets]
        require(np.allclose(np.bincount(targets, weights=effective, minlength=len(classes)),
                            len(indices) / len(classes), rtol=0, atol=1e-12), 'Effective fitting class mass differs')
        bundle[name] = model
        bundle['training_receipts'][name] = {'case_indices': indices.tolist(), 'rows': len(X),
            'training_features_sha256': array_hash(X), 'training_targets_sha256': array_hash(targets, '<i8'),
            'sample_weight_sha256': array_hash(weights), 'effective_weight_sha256': array_hash(effective),
            'sample_weight_sum': float(weights.sum()), 'effective_weight_sum': float(effective.sum()),
            'actual_gamma': float(model._gamma), 'class_weight': model.class_weight_.tolist(),
            'fit_status': int(model.fit_status_), 'n_iter': model.n_iter_.tolist(), 'n_support': model.n_support_.tolist()}
    return bundle


def score_arrays(binary_margin, malignant_scores):
    binary_margin = np.asarray(binary_margin, dtype=np.float64)
    malignant_scores = np.asarray(malignant_scores, dtype=np.float64)
    require(binary_margin.ndim == 2 and binary_margin.shape[1] == 15
            and malignant_scores.shape == (*binary_margin.shape, 4)
            and np.isfinite(binary_margin).all() and np.isfinite(malignant_scores).all(), 'Finite case/mask decision scores required')
    p = expit(binary_margin)
    require(np.array_equal(p >= .5, binary_margin >= 0.), 'Transformed score route disagrees with authoritative binary margin')
    q = softmax(malignant_scores, axis=-1)
    joint = np.concatenate(((1. - p)[..., None], p[..., None] * q), axis=-1)
    predictions = np.where(binary_margin >= 0., 1 + malignant_scores.argmax(axis=-1), 0).astype(np.int64)
    require(np.isfinite(joint).all() and np.allclose(joint.sum(-1), 1., rtol=0, atol=1e-12), 'Invalid normalized score distribution')
    return joint, predictions


def predict(bundle, values, case_indices):
    values = checked_views(values); indices = checked_indices(case_indices, len(values))
    X = vectorize(values, bundle, indices).reshape(len(indices) * 15, values.shape[-1] + 4)
    binary = np.asarray(bundle['binary'].decision_function(X), dtype=np.float64).reshape(len(indices), 15)
    malignant = np.asarray(bundle['malignant'].decision_function(X), dtype=np.float64).reshape(len(indices), 15, 4)
    joint, predictions = score_arrays(binary, malignant)
    return {'case_indices': indices.copy(), 'masks': MASKS.copy(), 'binary_margin': binary,
            'malignant_scores': malignant, 'joint_probabilities5': joint, 'predictions': predictions}


def state_arrays(bundle):
    result = {}
    for name in ('binary', 'malignant'):
        model = bundle[name]; receipt = bundle['training_receipts'][name]
        for attr in ('support_vectors_', 'dual_coef_', 'intercept_', 'class_weight_'):
            result[name + '_' + attr] = np.asarray(getattr(model, attr), dtype=np.float64)
        for attr in ('n_support_', 'support_', 'classes_', 'n_iter_'):
            result[name + '_' + attr] = np.asarray(getattr(model, attr), dtype=np.int64)
        result[name + '_actual_gamma'] = np.array(model._gamma, np.float64)
        result[name + '_fit_status_'] = np.array(model.fit_status_, np.int64)
        training = np.asarray(receipt['case_indices'], np.int64)
        result[name + '_training_case_indices'] = training
        result[name + '_support_case_indices'] = np.repeat(training, 15)[model.support_]
        for key in ('training_features_sha256', 'training_targets_sha256', 'sample_weight_sha256', 'effective_weight_sha256'):
            result[name + '_' + key] = np.array(receipt[key])
        for key in ('sample_weight_sum', 'effective_weight_sum'):
            result[name + '_' + key] = np.array(receipt[key], np.float64)
    return result


def case_records(cases, archive, role):
    require(role in ('train', 'validation', 'test'), 'Canonical role required')
    joint, predictions = score_arrays(archive['binary_margin'], archive['malignant_scores'])
    require(np.array_equal(joint, archive['joint_probabilities5']) and np.array_equal(archive['masks'], MASKS)
            and np.array_equal(predictions, archive['predictions']), 'Saved scores/masks/labels differ from declared transforms')
    records = []
    for row, index in enumerate(archive['case_indices']):
        case = cases[int(index)]
        for mask, subset in enumerate(SUBSETS):
            records.append({'role': role, 'case_id': case['case_id'], 'truth': int(case['truth8']), 'truth5': int(case['truth5']),
                'magnifications': [MAGNIFICATIONS[v] for v in subset],
                'binary_margin': float(archive['binary_margin'][row, mask]),
                'malignant_scores4': archive['malignant_scores'][row, mask].tolist(),
                'binary_score': float(expit(archive['binary_margin'][row, mask])),
                'probabilities5': joint[row, mask].tolist(), 'prediction': int(predictions[row, mask]),
                'score_interpretation': 'uncalibrated normalized scores; margin-based routing'})
    return records


def classification(truth, prediction, classes):
    truth = np.asarray(truth, dtype=np.int64); prediction = np.asarray(prediction, dtype=np.int64)
    matrix = np.zeros((classes, classes), dtype=np.int64); np.add.at(matrix, (truth, prediction), 1)
    support = matrix.sum(1); called = matrix.sum(0); correct = matrix.diagonal()
    f1 = np.divide(2. * correct, support + called, out=np.zeros(classes), where=support + called > 0)
    recall = np.divide(correct, support, out=np.zeros(classes), where=support > 0)
    return {'records': len(truth), 'correct': int(correct.sum()), 'accuracy': float(correct.sum() / len(truth)),
            'macro_f1': float(f1.mean()), 'macro_recall': float(recall.mean()), 'confusion_matrix': matrix.tolist(),
            'class_support': support.tolist(), 'class_correct': correct.tolist(), 'class_recall': recall.tolist()}


def summary(cases, archive):
    require(np.array_equal(archive['masks'], MASKS), 'Canonical mask order required')
    indices = np.asarray(archive['case_indices'], dtype=np.int64)
    truth = np.array([cases[int(i)]['truth5'] for i in indices], np.int64)
    _, predictions = score_arrays(archive['binary_margin'], archive['malignant_scores'])
    subsets = {}
    for mask, subset in enumerate(SUBSETS):
        pred = predictions[:, mask]; malignant = truth > 0; conditional = malignant & (pred > 0)
        correct = int(np.sum(malignant & (pred == truth))); denominator = int(malignant.sum()); routed = int(conditional.sum())
        binary = classification(malignant.astype(int), (pred > 0).astype(int), 2)
        binary['roc_auc'] = float(roc_auc_score(malignant, archive['binary_margin'][:, mask])) if len(set(malignant.tolist())) == 2 else None
        subsets['+'.join(str(MAGNIFICATIONS[v]) for v in subset)] = {
            'five_class': classification(truth, pred, 5), 'binary': binary,
            'end_to_end_malignant': {'correct': correct, 'denominator': denominator, 'recall': correct / denominator if denominator else None},
            'conditional_malignant': {'correct': correct, 'denominator': routed, 'accuracy': correct / routed if routed else 0.}}
    return {'cases': len(indices), 'case_subset_events': len(indices) * 15,
            'mean_subset_macro_f1': float(np.mean([s['five_class']['macro_f1'] for s in subsets.values()])),
            'mean_subset_accuracy': float(np.mean([s['five_class']['accuracy'] for s in subsets.values()])),
            'mean_subset_binary_accuracy': float(np.mean([s['binary']['accuracy'] for s in subsets.values()])),
            'worst_subset_macro_f1': float(min(s['five_class']['macro_f1'] for s in subsets.values())),
            'worst_subset_accuracy': float(min(s['five_class']['accuracy'] for s in subsets.values())),
            'all_four': subsets['40+100+200+400'], 'subsets': subsets}

"""Five-category metrics with explicit direct or hard-routed predictions."""
import numpy as np
from scipy.special import logsumexp

from .capstone_binary import run_summary as binary_run_summary
from .collect import SUBSETS


FINAL5 = ('benign', 'ductal_carcinoma', 'lobular_carcinoma',
          'mucinous_carcinoma', 'papillary_carcinoma')
MAP8_TO5 = (0, 1, 0, 2, 3, 4, 0, 0)
MEAN_METRICS = ('macro_f1', 'accuracy', 'balanced_accuracy', 'cross_entropy', 'brier')


def five_labels(y8):
    """Pool the four canonical benign labels after existing roles are assigned."""
    labels = np.asarray(y8)
    if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= len(MAP8_TO5)))):
        raise ValueError('Expected one-dimensional canonical eight-class integer labels')
    return np.asarray(MAP8_TO5, dtype=int)[labels]


def _checked_probabilities(probabilities, log_probabilities, shape):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    log_probabilities = np.asarray(log_probabilities, dtype=np.float64)
    if (probabilities.shape != shape or log_probabilities.shape != shape
            or not np.isfinite(probabilities).all()
            or not np.isfinite(log_probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-12)
            or not np.allclose(np.exp(log_probabilities), probabilities, rtol=0, atol=1e-12)
            or not np.allclose(logsumexp(log_probabilities, axis=-1), 0, rtol=0, atol=1e-12)):
        raise ValueError('Invalid normalized probabilities or stable log probabilities')
    return probabilities, log_probabilities


def _classification_metrics(y, pred, probabilities, log_probabilities, classes, weights=None):
    labels = np.asarray(y)
    predictions = np.asarray(pred)
    if (labels.ndim != 1 or not len(labels)
            or not np.issubdtype(labels.dtype, np.integer)
            or predictions.shape != labels.shape
            or not np.issubdtype(predictions.dtype, np.integer)
            or np.any((labels < 0) | (labels >= classes))
            or np.any((predictions < 0) | (predictions >= classes))):
        raise ValueError('Invalid integer truth or explicit prediction labels')
    probabilities, log_probabilities = _checked_probabilities(
        probabilities, log_probabilities, (len(labels), classes))
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    if (weights.shape != labels.shape or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isfinite(weights.sum())):
        raise ValueError('Expected positive finite metric weights')
    matrix = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(matrix, (labels, predictions), weights)
    support = matrix.sum(axis=1)
    called = matrix.sum(axis=0)
    correct = matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * correct, support + called, out=np.zeros(classes),
                   where=(support + called) > 0)
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    return {
        'accuracy': float(correct.sum() / weights.sum()),
        'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall[support > 0].mean()),
        'cross_entropy': float(np.average(-log_probabilities[np.arange(len(labels)), labels],
                                          weights=weights)),
        'brier': float(np.average(((probabilities - np.eye(classes)[labels]) ** 2).sum(axis=1),
                                 weights=weights)),
        'confusion_matrix': matrix.tolist(),
        'class_recall': recall.tolist(),
        'class_support': support.tolist(),
        'records': len(labels),
        'weight_sum': float(weights.sum()),
        'zero_true_probability_records': int(np.sum(true_probabilities == 0)),
    }


def classification_metrics(y, pred, probabilities, log_probabilities, weights=None):
    """Use explicit labels for classification and stable probabilities for scores.

    Macro F1 includes all five labels. Balanced accuracy averages recall over
    represented truth classes. Complete role summaries require all five classes.
    """
    return _classification_metrics(y, pred, probabilities, log_probabilities, len(FINAL5), weights)


def _summary(subsets, pooled, support):
    result = {
        'subsets': subsets,
        'pooled': pooled,
        'all_view': subsets['40+100+200+400'],
        'distinct_cases': int(np.sum(support)),
        'distinct_class_support': support.tolist(),
        'worst_subset_macro_f1': float(min(value['macro_f1'] for value in subsets.values())),
    }
    for key in MEAN_METRICS:
        result['mean_subset_' + key] = float(np.mean([value[key] for value in subsets.values()]))
    return result


def _record_metrics(records, malignant=False):
    if malignant:
        records = [row for row in records if row['truth5'] > 0]
        probabilities = np.asarray([row['malignant_probabilities4'] for row in records])
        return _classification_metrics(
            np.asarray([row['truth5'] - 1 for row in records]), probabilities.argmax(axis=1),
            probabilities, [row['malignant_log_probabilities'] for row in records], 4)
    return classification_metrics(
        np.asarray([row['truth5'] for row in records]),
        np.asarray([row['prediction'] for row in records]),
        [row['probabilities5'] for row in records],
        [row['log_probabilities5'] for row in records])


def summarize_records(records, decision_rule):
    """Validate one complete fitting/validation role and summarize every subset.

    Records retain original truth, pooled truth, an explicit prediction, stable
    five-way probabilities, a malignant score and conditional malignant scores.
    ``decision_rule`` is ``direct`` or ``dual``. Only dual predictions follow the
    fixed binary threshold. Each case must occur once in all fifteen subsets.
    """
    if decision_rule not in ('direct', 'dual'):
        raise ValueError('Expected direct or dual decision rule')
    records = list(records)
    if not records:
        raise ValueError('Expected complete nonempty case/subset records')
    required = {'role', 'case_id', 'truth', 'truth5', 'prediction', 'magnifications',
                'probabilities5', 'log_probabilities5', 'binary_score',
                'binary_log_probabilities', 'malignant_probabilities4',
                'malignant_log_probabilities'}
    identities = {}
    roles = set()
    seen = set()
    selected = {subset: [] for subset in SUBSETS}
    binary_events = {subset: [] for subset in SUBSETS}
    for row in records:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError('Case/subset record fields are incomplete')
        if (row['role'] not in ('train', 'validation')
                or not isinstance(row['case_id'], str) or not row['case_id']
                or type(row['truth']) is not int or row['truth'] not in range(8)
                or type(row['truth5']) is not int or row['truth5'] != MAP8_TO5[row['truth']]
                or type(row['prediction']) is not int or row['prediction'] not in range(5)):
            raise ValueError('Invalid case role, identity, truth or prediction')
        magnifications = row['magnifications']
        if (not isinstance(magnifications, (list, tuple))
                or any(type(value) is not int for value in magnifications)):
            raise ValueError('Expected canonical integer magnification subset')
        subset = tuple(magnifications)
        identity = row['case_id']
        if subset not in selected or (identity, subset) in seen:
            raise ValueError('Unexpected or duplicate case/subset record')
        if identity in identities and identities[identity] != row['truth']:
            raise ValueError('Original case truth changed across subsets')
        score = row['binary_score']
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid malignant routing score')
        probabilities, log_probabilities = _checked_probabilities(
            row['probabilities5'], row['log_probabilities5'], (5,))
        malignant, malignant_log = _checked_probabilities(
            row['malignant_probabilities4'], row['malignant_log_probabilities'], (4,))
        _, binary_log = _checked_probabilities(
            [1 - score, score], row['binary_log_probabilities'], (2,))
        composed_log = np.r_[binary_log[0], binary_log[1] + malignant_log]
        if not np.allclose(log_probabilities, composed_log, rtol=0, atol=1e-10):
            raise ValueError('Five-way and conditional probability decomposition differ')
        expected = int(probabilities.argmax())
        if decision_rule == 'dual':
            expected = 0 if score < .5 else 1 + int(malignant.argmax())
        if row['prediction'] != expected:
            raise ValueError('Explicit prediction differs from declared decision rule')
        identities[identity] = row['truth']
        roles.add(row['role'])
        seen.add((identity, subset))
        selected[subset].append(row)
        binary_events[subset].append((int(row['truth5'] > 0), float(score)))
    if (len(roles) != 1 or len(records) != len(identities) * len(SUBSETS)
            or any(len(rows) != len(identities) for rows in selected.values())):
        raise ValueError('Expected one role and complete fifteen-subset case coverage')
    support = np.bincount(five_labels(np.asarray(list(identities.values()))), minlength=5)
    if np.any(support == 0):
        raise ValueError('Complete role summary requires all five final classes')
    subsets = {}
    malignant_subsets = {}
    binary = binary_run_summary(binary_events)
    for subset, rows in selected.items():
        name = '+'.join(map(str, subset))
        subsets[name] = _record_metrics(rows)
        malignant_subsets[name] = _record_metrics(rows, malignant=True)
        binary['subsets'][name]['cross_entropy'] = float(np.mean([
            -row['binary_log_probabilities'][int(row['truth5'] > 0)] for row in rows]))
    binary_ce = [value['cross_entropy'] for value in binary['subsets'].values()]
    binary['mean_all_subsets']['cross_entropy'] = float(np.mean(binary_ce))
    binary['worst_subset_by_metric']['cross_entropy'] = float(max(binary_ce))
    binary['pooled_case_subset_events']['cross_entropy'] = float(np.mean(binary_ce))
    result = _summary(subsets, _record_metrics(records), support)
    result.update(class_names=list(FINAL5), role=next(iter(roles)), decision_rule=decision_rule,
                  binary_summary=binary,
                  malignant_summary=_summary(malignant_subsets, _record_metrics(records, malignant=True),
                                             support[1:]))
    result['malignant_summary']['class_names'] = list(FINAL5[1:])
    return result

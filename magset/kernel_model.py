"""Case-balanced centered-cosine density heads on immutable frozen field bags.

No optimizer, encoder execution, query-statistic adaptation or cross-view field
matching occurs here. Prepared same-view similarities are reused for all four T.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import logsumexp

EPS = 1e-12
TEMPERATURES = (.01, .03, .1, .3)
NORMALIZATION = {'centering': 'equal support-case raw mean per view', 'dtype': 'float64',
                 'l2_zero_epsilon': EPS, 'zero_policy': 'divide by max(norm, epsilon); exact zero stays zero'}
HEAD_SLICES = {'direct': slice(0, 5), 'binary': slice(5, 7), 'malignant': slice(7, 11)}


@dataclass(frozen=True)
class PreparedView:
    cosine: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PreparedCase:
    query_index: int
    support_indices: tuple
    centers: tuple
    views: tuple


def _indices(indices, count):
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or len(set(values.tolist())) != len(values)
            or np.any((values < 0) | (values >= count))):
        raise ValueError('Expected unique nonempty support case indices')
    return values.astype(int)


def _validate(bags, y5):
    labels = np.asarray(y5)
    if (labels.shape != (len(bags),) or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels > 4))):
        raise ValueError('Expected final-five integer case labels')
    dimension = None
    for case in bags:
        if len(case) != 4:
            raise ValueError('Every case requires four raw field bags')
        for fields in case:
            fields = np.asarray(fields)
            if (fields.ndim != 2 or not len(fields) or not fields.shape[1]
                    or fields.dtype.kind != 'f' or not np.isfinite(fields).all()):
                raise ValueError('Expected nonempty finite floating-point fields')
            if dimension is None:
                dimension = fields.shape[1]
            if fields.shape[1] != dimension:
                raise ValueError('Field dimensions differ')
    return labels, dimension


def normalize_centered(fields, center):
    x = np.asarray(fields, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or center.shape != (x.shape[1],)
            or not np.isfinite(x).all() or not np.isfinite(center).all()):
        raise ValueError('Invalid raw fields or support-only center')
    x = x - center
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError('Nonfinite centered feature norms')
    return x / np.maximum(norms, EPS)


def class_weights(labels, field_counts, head='direct'):
    """One density column per head class, each with unit total field mass."""
    labels = np.asarray(labels)
    counts = np.asarray(field_counts)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels > 4)) or counts.shape != labels.shape
            or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0)
            or head not in HEAD_SLICES):
        raise ValueError('Invalid support labels, field counts or head')
    head_labels = (labels > 0).astype(int) if head == 'binary' else labels
    classes = np.arange(2) if head == 'binary' else np.arange(1, 5) if head == 'malignant' else np.arange(5)
    case_weights = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for column, label in enumerate(classes):
        selected = head_labels == label
        number = int(selected.sum())
        if not number:
            raise ValueError('Every head class needs at least one eligible support case')
        case_weights[selected, column] = 1. / (number * counts[selected])
    return np.repeat(case_weights, counts, axis=0)


def _centers(bags, indices):
    return tuple(np.mean([np.asarray(bags[i][view], dtype=np.float64).mean(0)
                          for i in indices], axis=0) for view in range(4))


def support_summary(bags, y5, indices):
    labels, dimension = _validate(bags, y5)
    indices = _indices(indices, len(bags))
    counts = np.bincount(labels[indices], minlength=5)
    if np.any(counts == 0):
        raise ValueError('Support must cover all five final labels')
    centers = _centers(bags, indices)
    return {'support_indices': indices.tolist(), 'class_support': counts.tolist(),
            'head_case_counts': {'direct': counts.tolist(),
                                'binary': [int(counts[0]), int(counts[1:].sum())],
                                'malignant': counts[1:].tolist()},
            'field_counts_by_view': [sum(len(bags[i][v]) for i in indices) for v in range(4)],
            'centers_sha256': hashlib.sha256(np.stack(centers).tobytes()).hexdigest(),
            'dimension': int(dimension), 'normalization': dict(NORMALIZATION)}


def prepare_case(bags, y5, support_indices, query_index, *, leave_query_case_out=False):
    """Use one common support-only center per view for all three density heads."""
    labels, _ = _validate(bags, y5)
    indices = _indices(support_indices, len(bags))
    if (isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer))
            or query_index < 0 or query_index >= len(bags)):
        raise ValueError('Invalid query case index')
    in_support = query_index in indices
    if bool(leave_query_case_out) != bool(in_support):
        raise ValueError('Fitting queries require whole-case exclusion; validation queries must not be support')
    if leave_query_case_out:
        indices = indices[indices != query_index]
    if not len(indices) or np.any(np.bincount(labels[indices], minlength=5) == 0):
        raise ValueError('All five classes must remain after whole-case exclusion')
    centers = _centers(bags, indices)
    views = []
    for view in range(4):
        counts = np.asarray([len(bags[i][view]) for i in indices], dtype=int)
        support = np.concatenate([normalize_centered(bags[i][view], centers[view]) for i in indices])
        query = normalize_centered(bags[query_index][view], centers[view])
        cosine = np.clip(query @ support.T, -1., 1.)
        weights = np.concatenate([class_weights(labels[indices], counts, head) for head in HEAD_SLICES], axis=1)
        cosine.setflags(write=False)
        weights.setflags(write=False)
        views.append(PreparedView(cosine, weights))
    for center in centers:
        center.setflags(write=False)
    return PreparedCase(int(query_index), tuple(int(i) for i in indices), centers, tuple(views))


def density_logs(prepared, temperature):
    """Return normalized direct5, binary2 and malignant4 logs in canonical orders."""
    if temperature not in TEMPERATURES or isinstance(temperature, (bool, np.bool_)):
        raise ValueError('Use one of the four registered shared temperatures')
    if not isinstance(prepared, PreparedCase) or len(prepared.views) != 4:
        raise ValueError('Expected four prepared same-view density caches')
    outputs = {head: [] for head in HEAD_SLICES}
    for view in prepared.views:
        cosine, weights = view.cosine, view.weights
        if (cosine.ndim != 2 or not cosine.size or weights.shape != (cosine.shape[1], 11)
                or not np.isfinite(cosine).all() or np.any(np.abs(cosine) > 1)
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.allclose(weights.sum(0), 1., rtol=0, atol=1e-12)):
            raise ValueError('Prepared cosine or class density weights differ')
        scaled = cosine / float(temperature)
        maximum = scaled.max(axis=1, keepdims=True)
        # Range is at most 200 for T>=.01; exp is safely positive in float64.
        mass = np.exp(scaled - maximum) @ weights
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('Invalid stabilized kernel density')
        log_density = np.log(mass) + maximum
        for head, columns in HEAD_SLICES.items():
            value = log_density[:, columns]
            value = value - logsumexp(value, axis=1, keepdims=True)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite normalized kernel probabilities')
            outputs[head].append(value)
    return {head: tuple(values) for head, values in outputs.items()}


def field_log_probabilities(prepared, temperature, arm):
    logs = density_logs(prepared, temperature)
    if arm == 'direct':
        return logs['direct']
    if arm != 'dual':
        raise ValueError('Unknown final-five arm')
    return tuple(np.column_stack((binary[:, 0], binary[:, 1, None] + malignant))
                 for binary, malignant in zip(logs['binary'], logs['malignant']))

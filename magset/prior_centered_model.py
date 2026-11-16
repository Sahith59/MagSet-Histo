"""Two fixed supervised centering geometries with unchanged reviewed KDE math.

Uniform density priors do not require these centers. This is a fit-only geometry
ablation; no query truth or optimized coefficient enters a center or density.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from . import kernel_model as old

MODES = ('class_shared', 'head_specific')
HEADS = ('direct', 'binary', 'malignant')
TEMPERATURES = old.TEMPERATURES
EPS = old.EPS
MODE_CENTER_MAP = {'class_shared': dict.fromkeys(HEADS, 'direct'),
                   'head_specific': {head: head for head in HEADS}}
NORMALIZATION = {'dtype': 'float64', 'l2_zero_epsilon': EPS, 'zero_policy': old.NORMALIZATION['zero_policy'],
                 'canonical_center': 'per view: field mean per support case, then case means within each head class, then equal mean over head classes',
                 'class_shared': {'direct': 'direct-five center', 'binary': 'direct-five center', 'malignant': 'direct-five center'},
                 'head_specific': {'direct': 'direct-five center', 'binary': 'equal benign-case and malignant-case group centers',
                                   'malignant': 'equal four malignant-type centers; benign cases excluded'}}


@dataclass(frozen=True)
class PreparedBundle:
    query_index: int
    support_indices: tuple
    heads: dict


@dataclass(frozen=True)
class DensityCache:
    bundle: PreparedBundle
    temperature: float
    logs: dict


def _support(bags, y5, indices):
    labels, dimension = old._validate(bags, y5)
    indices = old._indices(indices, len(bags))
    if np.any(np.bincount(labels[indices], minlength=5) == 0):
        raise ValueError('All five support classes must remain after whole-case exclusion')
    return labels, dimension, indices


def centers(bags, labels, indices):
    """Every case first contributes its raw field mean, independent of field count."""
    result = {head: [] for head in HEADS}
    selected = labels[indices]
    for view in range(4):
        case_means = np.asarray([np.asarray(bags[i][view], dtype=np.float64).mean(0) for i in indices])
        classes = np.asarray([case_means[selected == label].mean(0) for label in range(5)])
        result['direct'].append(classes.mean(0))
        result['binary'].append(np.asarray([case_means[selected == 0].mean(0), case_means[selected > 0].mean(0)]).mean(0))
        result['malignant'].append(classes[1:].mean(0))
    for head in HEADS:
        result[head] = tuple(result[head])
        for value in result[head]:
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite support-only head center')
            value.setflags(write=False)
    return result


def support_summary(bags, y5, indices, mode=None):
    if mode is not None and mode not in MODES:
        raise ValueError('Unknown fixed centering mode')
    labels, dimension, indices = _support(bags, y5, indices)
    counts = np.bincount(labels[indices], minlength=5)
    mu = centers(bags, labels, indices)
    return {'support_indices': indices.tolist(), 'class_support': counts.tolist(),
            'head_case_counts': {'direct': counts.tolist(), 'binary': [int(counts[0]), int(counts[1:].sum())],
                                 'malignant': counts[1:].tolist()},
            'field_counts_by_view': [sum(len(bags[i][v]) for i in indices) for v in range(4)],
            'centers_sha256': {head: hashlib.sha256(np.stack(mu[head]).tobytes()).hexdigest() for head in HEADS},
            'dimension': int(dimension), 'normalization': dict(NORMALIZATION),
            'mode_used_center_mapping': {m: dict(MODE_CENTER_MAP[m]) for m in MODES} if mode is None else dict(MODE_CENTER_MAP[mode])}


def prepare_case(bags, y5, support_indices, query_index, *, leave_query_case_out=False):
    """Remove a fitting query before all case/class means, counts and field weights."""
    labels, _ = old._validate(bags, y5)
    indices = old._indices(support_indices, len(bags))
    if (isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer))
            or not 0 <= query_index < len(bags) or not isinstance(leave_query_case_out, (bool, np.bool_))):
        raise ValueError('Valid query index and whole-case exclusion flag required')
    if bool(leave_query_case_out) != bool(query_index in indices):
        raise ValueError('Fitting queries require exclusion; validation queries cannot be support')
    if leave_query_case_out:
        indices = indices[indices != query_index]
    if not len(indices) or np.any(np.bincount(labels[indices], minlength=5) == 0):
        raise ValueError('All five support classes must remain after whole-case exclusion')
    mu = centers(bags, labels, indices)
    views = {head: [] for head in HEADS}
    for view in range(4):
        counts = np.asarray([len(bags[i][view]) for i in indices], dtype=int)
        weights = np.concatenate([old.class_weights(labels[indices], counts, head) for head in HEADS], axis=1)
        weights.setflags(write=False)
        for head in HEADS:
            support = np.concatenate([old.normalize_centered(bags[i][view], mu[head][view]) for i in indices])
            query = old.normalize_centered(bags[query_index][view], mu[head][view])
            cosine = np.clip(query @ support.T, -1., 1.)
            cosine.setflags(write=False)
            views[head].append(old.PreparedView(cosine, weights))
    eligible = tuple(int(i) for i in indices)
    prepared = {head: old.PreparedCase(int(query_index), eligible, mu[head], tuple(views[head])) for head in HEADS}
    return PreparedBundle(int(query_index), eligible, prepared)


def density_cache(bundle, temperature):
    """Three old density evaluations once per query/T, shared across modes and arms."""
    if not isinstance(bundle, PreparedBundle) or set(bundle.heads) != set(HEADS):
        raise ValueError('Expected all three prepared head-center cases')
    for value in bundle.heads.values():
        if value.query_index != bundle.query_index or value.support_indices != bundle.support_indices:
            raise ValueError('Head centers do not share the exact eligible support/query')
    values = {head: old.density_logs(bundle.heads[head], temperature) for head in HEADS}
    for result in values.values():
        for arrays in result.values():
            for array in arrays:
                array.setflags(write=False)
    return DensityCache(bundle, float(temperature), values)


def field_log_probabilities(bundle, temperature, arm, mode, cache=None):
    if mode not in MODES or arm not in ('direct', 'dual'):
        raise ValueError('Declared fixed centering mode and arm required')
    if cache is None:
        cache = density_cache(bundle, temperature)
    if (not isinstance(cache, DensityCache) or cache.bundle is not bundle
            or cache.temperature != temperature or isinstance(temperature, (bool, np.bool_))):
        raise ValueError('Density cache belongs to a different query or temperature')
    mapping = MODE_CENTER_MAP[mode]
    if arm == 'direct':
        return cache.logs['direct']['direct']
    binary = cache.logs[mapping['binary']]['binary']
    malignant = cache.logs[mapping['malignant']]['malignant']
    return tuple(np.column_stack((r[:, 0], r[:, 1, None]+q)) for r, q in zip(binary, malignant))

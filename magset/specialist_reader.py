"""Frozen specialist-reader math with a fixed parent-P binary router.

No files, split selection, model inference, optimization or test gate lives here.
The runner must bind cache bytes and align their original fields before calling
these functions. All statistics use fitting cases only; UNI centering includes
all fitting cases, while its density support contains malignant cases only.
"""
from dataclasses import dataclass
import hashlib
import re

import numpy as np
from scipy.special import logsumexp

from . import kernel_model as kernel
from . import prior_centered_model as parent
from .data import MAGNIFICATIONS, case_id

ARMS = ('p_control', 'uni2_c')
TEMPERATURES = (.01, .03, .1, .3)
ROUTER_TEMPERATURE = .1
EPS = 1e-12


@dataclass(frozen=True)
class UniSupport:
    support_indices: tuple
    fitting_labels: tuple
    malignant_indices: tuple
    centers: tuple
    features: tuple
    weights: tuple
    dimension: int


@dataclass(frozen=True)
class UniQueryView:
    cosine: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PreparedQuery:
    query_index: int
    support_indices: tuple
    parent_bundle: parent.PreparedBundle
    uni_views: tuple
    reference_joint_logs: tuple
    fixed_binary_logs: tuple


def _readonly(value):
    value = np.asarray(value, dtype=np.float64)
    value.setflags(write=False)
    return value


def _temperature(value):
    if isinstance(value, (bool, np.bool_)) or value not in TEMPERATURES:
        raise ValueError('Use one of four registered malignant temperatures')
    return float(value)


def assert_field_alignment(parent_rows, uni_rows):
    """Reject reordered/missing/duplicated paths or changed bytes, provider or view.

    Rows must have relative_path/sha256 and either a qualified case_id/provider_key
    or the canonical filename/case_id_raw pair, plus magnification or view index.
    The runner obtains these rows through the hash-bound source manifest/cache
    mapping; equal array lengths alone are not an alignment proof.
    """
    def signature(row):
        path, digest = row['relative_path'], row['sha256']
        provider = row.get('provider_key', row.get('case_id'))
        if provider is None:
            provider = case_id(row)
        if 'magnification' in row:
            mag = int(row['magnification'])
        else:
            view = row['view']
            if isinstance(view, (bool, np.bool_)) or not isinstance(view, (int, np.integer)) or not 0 <= view < 4:
                raise ValueError('Canonical view index required')
            mag = MAGNIFICATIONS[view]
        if ('view' in row and row['view'] != MAGNIFICATIONS.index(mag)) or mag not in MAGNIFICATIONS:
            raise ValueError('Field view/magnification differs')
        if (not isinstance(path, str) or not path or not isinstance(provider, str) or not provider
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('Field identity/hash is incomplete')
        return path, digest, provider, mag
    left = [signature(row) for row in parent_rows]
    right = [signature(row) for row in uni_rows]
    if not left or len({row[0] for row in left}) != len(left) or left != right:
        raise ValueError('Parent and specialist fields are not exactly aligned')
    return len(left)


def prepare_support(uni_bags, y5, support_indices):
    """All-fitting-case mean centers; class/case/field-balanced Mal4 support."""
    labels, dimension = kernel._validate(uni_bags, y5)
    indices = np.sort(kernel._indices(support_indices, len(uni_bags)))
    if np.any(np.bincount(labels[indices], minlength=5) == 0):
        raise ValueError('Fitting cases must contain all five classes')
    malignant = indices[labels[indices] > 0]
    centers = []; features = []; weights = []
    for view in range(4):
        center = np.mean([np.asarray(uni_bags[i][view], dtype=np.float64).mean(0) for i in indices], axis=0)
        counts = np.asarray([len(uni_bags[i][view]) for i in malignant], dtype=int)
        support = np.concatenate([kernel.normalize_centered(uni_bags[i][view], center) for i in malignant])
        mass = kernel.class_weights(labels[malignant], counts, 'malignant')
        if not np.allclose(mass.sum(0), 1., rtol=0, atol=1e-12):
            raise ValueError('Each malignant support class must have unit field mass')
        centers.append(_readonly(center)); features.append(_readonly(support)); weights.append(_readonly(mass))
    return UniSupport(tuple(map(int, indices)), tuple(map(int, labels[indices])), tuple(map(int, malignant)),
                      tuple(centers), tuple(features), tuple(weights), int(dimension))


def support_summary(support):
    if not isinstance(support, UniSupport):
        raise ValueError('Prepared fitting-only UNI support required')
    digest = lambda arrays: hashlib.sha256(b''.join(np.asarray(a, dtype='<f8').tobytes(order='C') for a in arrays)).hexdigest()
    return {'support_case_indices': list(support.support_indices),
            'fitting_class_counts': np.bincount(support.fitting_labels, minlength=5).tolist(),
            'malignant_support_case_indices': list(support.malignant_indices),
            'dimension': support.dimension, 'centering': 'equal mean of all fitting case field means per view',
            'normalization': 'float64 centered L2; divide by max(norm,1e-12)',
            'centers_sha256': digest(support.centers), 'normalized_support_sha256': digest(support.features),
            'weights_sha256': digest(support.weights),
            'malignant_fields_by_view': [len(values) for values in support.features],
            'class_weight_sums_by_view': [values.sum(0).tolist() for values in support.weights]}


def decompose_joint_logs(joint_logs):
    """Recover binary2 and conditional malignant4 logs without a second readout."""
    binary = []; malignant = []
    for values in joint_logs:
        values = np.asarray(values, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all()
                or not np.allclose(logsumexp(values, axis=1), 0., rtol=0, atol=1e-12)):
            raise ValueError('Finite normalized joint-five logs required')
        log_malignant = logsumexp(values[:, 1:], axis=1, keepdims=True)
        binary.append(_readonly(np.column_stack((values[:, 0], log_malignant[:, 0]))))
        malignant.append(_readonly(values[:, 1:] - log_malignant))
    if len(binary) != 4:
        raise ValueError('Exactly four nonempty views required')
    return tuple(binary), tuple(malignant)


def prepare_query(parent_bags, uni_bags, y5, support_indices, query_index, support=None,
                  parent_reference_joint_logs=None):
    """Prepare one query, optionally using a runner-verified parent reference.

    The optional reference must come from the hash-bound canonical parent role
    archive. It avoids evaluating an unselected P specialist merely to obtain
    the fixed router during retrospective follow-up. Validation identity uses
    the default path and independently rederives the complete parent P output.
    """
    labels, _ = kernel._validate(parent_bags, y5)
    uni_labels, dimension = kernel._validate(uni_bags, y5)
    indices = tuple(map(int, np.sort(kernel._indices(support_indices, len(parent_bags)))))
    if (isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer))
            or not 0 <= query_index < len(parent_bags) or query_index in indices):
        raise ValueError('Query must be outside the fitting support')
    if len(parent_bags) != len(uni_bags) or not np.array_equal(labels, uni_labels):
        raise ValueError('Parent/specialist cases or labels differ')
    if any(len(a[v]) != len(b[v]) for a, b in zip(parent_bags, uni_bags) for v in range(4)):
        raise ValueError('Parent/specialist per-case view field counts differ')
    if support is None:
        support = prepare_support(uni_bags, y5, indices)
    if (not isinstance(support, UniSupport) or support.support_indices != indices
            or support.fitting_labels != tuple(map(int, labels[list(indices)])) or support.dimension != dimension):
        raise ValueError('UNI support belongs to different fitting cases/labels/dimensions')
    bundle = parent.prepare_case(parent_bags, y5, indices, int(query_index), leave_query_case_out=False)
    if parent_reference_joint_logs is None:
        cache = parent.density_cache(bundle, ROUTER_TEMPERATURE)
        reference = parent.field_log_probabilities(bundle, ROUTER_TEMPERATURE, 'dual', 'head_specific', cache=cache)
    else:
        reference = tuple(np.asarray(values, dtype=np.float64) for values in parent_reference_joint_logs)
        if len(reference) != 4 or any(value.shape != (len(parent_bags[query_index][view]), 5)
                                      for view, value in enumerate(reference)):
            raise ValueError('Canonical parent reference field/view coverage differs')
    reference = tuple(_readonly(values) for values in reference)
    binary, _ = decompose_joint_logs(reference)
    views = []
    for view in range(4):
        query = kernel.normalize_centered(uni_bags[query_index][view], support.centers[view])
        cosine = np.clip(query @ support.features[view].T, -1., 1.)
        views.append(UniQueryView(_readonly(cosine), support.weights[view]))
    return PreparedQuery(int(query_index), indices, bundle, tuple(views), reference, binary)


def _uni_malignant_logs(prepared, temperature):
    outputs = []
    for view in prepared.uni_views:
        scaled = view.cosine / temperature
        maximum = scaled.max(axis=1, keepdims=True)
        mass = np.exp(scaled - maximum) @ view.weights
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('Nonpositive/nonfinite stabilized specialist density')
        value = np.log(mass) + maximum
        outputs.append(value - logsumexp(value, axis=1, keepdims=True))
    return tuple(outputs)


def field_log_probabilities(prepared, temperature, arm, fixed_binary_logs=None):
    """Keep parent binary T=.1 fixed; vary only the chosen malignant reader/T."""
    if not isinstance(prepared, PreparedQuery) or arm not in ARMS:
        raise ValueError('Prepared query and registered specialist arm required')
    temperature = _temperature(temperature)
    binary = (prepared.fixed_binary_logs if fixed_binary_logs is None else
              tuple(np.asarray(values, dtype=np.float64) for values in fixed_binary_logs))
    if len(binary) != 4:
        raise ValueError('Four fixed binary view arrays required')
    for actual, expected in zip(binary, prepared.fixed_binary_logs):
        actual = np.asarray(actual, dtype=np.float64)
        if (actual.shape != expected.shape or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=0, atol=1e-12)):
            raise ValueError('Fixed binary router differs from parent P at temperature .1')
    if arm == 'p_control' and temperature == ROUTER_TEMPERATURE:
        # Preserve original parent arithmetic exactly for the identity control.
        return prepared.reference_joint_logs
    if arm == 'p_control':
        cache = parent.density_cache(prepared.parent_bundle, temperature)
        joint = parent.field_log_probabilities(prepared.parent_bundle, temperature, 'dual', 'head_specific', cache=cache)
        _, malignant = decompose_joint_logs(joint)
    else:
        malignant = _uni_malignant_logs(prepared, temperature)
    return tuple(np.column_stack((b[:, 0], b[:, 1, None] + m)) for b, m in zip(binary, malignant))


def field_probabilities(prepared, temperature, arm, fixed_binary_logs=None):
    """Use the same stable exponentiation as the accepted parent P producer."""
    logs = field_log_probabilities(prepared, temperature, arm, fixed_binary_logs)
    return tuple(np.exp(values - logsumexp(values, axis=1, keepdims=True)) for values in logs)

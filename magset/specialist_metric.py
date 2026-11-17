"""Fixed-budget head-specific diagonal metrics on immutable frozen descriptors.

Binary and malignant metrics are fitted independently with an identical recipe.
Support-case omission does not constitute leave-case-out model fitting.
Torch is imported lazily so saved states can be checked using NumPy alone.
"""
import hashlib
import os

import numpy as np
from scipy.special import logsumexp

EPS = 1e-12
STEPS = 200
LEARNING_RATE = .01
REGULARIZATION = 10.
LOG_FACTOR_BOUND = .5
SEED = 812
TEMPERATURES = (.01, .03, .1, .3)
FITTING_DIAGNOSTIC = 'in-sample learned metric with whole query omitted from density support'
TRAINING_SPEC = {
    'steps': STEPS, 'optimizer': 'Adam', 'learning_rate': LEARNING_RATE,
    'betas': [.9, .999], 'adam_epsilon': 1e-8, 'weight_decay': 0.,
    'regularization': REGULARIZATION, 'log_factor_bound': LOG_FACTOR_BOUND,
    'seed': SEED, 'dtype': 'float32', 'batch': 'all fitting head-eligible cases and four views',
    'checkpoint': 'final fixed step only', 'tf32': False, 'epsilon': EPS,
    'objective': 'class-balanced same-view cross-case case-centroid KDE cross-entropy',
}


def _inputs(case_centroids, y5, temperature, head='malignant'):
    x = np.asarray(case_centroids)
    labels = np.asarray(y5)
    if (x.ndim != 3 or x.shape[1] != 4 or not x.shape[0] or not x.shape[2]
            or x.dtype.kind != 'f' or not np.isfinite(x).all()
            or labels.shape != (x.shape[0],) or labels.dtype.kind not in 'iu'
            or np.any((labels < 0) | (labels > 4))):
        raise ValueError('Expected finite floating cases x four views x dimensions and final-five labels')
    if isinstance(temperature, (bool, np.bool_)) or temperature not in TEMPERATURES:
        raise ValueError('Use the inherited registered UNI2-h temperature')
    if head not in ('malignant', 'binary'):
        raise ValueError('Only binary and malignant metric heads are registered')
    counts = np.bincount(labels, minlength=5)
    if head == 'malignant' and (counts[0] < 1 or np.any(counts[1:] < 2)):
        raise ValueError('Need benign centering support and two distinct cases per malignant class')
    if head == 'binary' and (counts[0] < 2 or counts[1:].sum() < 2):
        raise ValueError('Need two distinct fitting cases per binary class')
    center = np.asarray(x, dtype=np.float64).mean(axis=0)
    selected = np.flatnonzero(labels > 0) if head == 'malignant' else np.arange(len(labels))
    head_labels = labels[selected] - 1 if head == 'malignant' else (labels > 0).astype(np.int64)
    centered = np.asarray(x[selected], dtype=np.float64) - center[None]
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), EPS)
    return normalized, head_labels.astype(np.int64), center, selected, counts


def factors_from_theta(theta):
    """Float64 mathematical map for independent approximate state checks.

    Fitting saves the actual FP32 Torch factors, cast to float64. This NumPy
    calculation need only match them to the declared FP32 rounding tolerance.
    """
    theta = np.asarray(theta)
    if theta.ndim != 1 or not theta.size or theta.dtype.kind != 'f' or not np.isfinite(theta).all():
        raise ValueError('Expected a finite floating theta vector')
    log_factors = LOG_FACTOR_BOUND * np.tanh(theta.astype(np.float64))
    return log_factors, np.exp(log_factors)


def transform_vectors(z, factors):
    """Apply a positive diagonal metric to normalized vectors, then re-normalize.

    The identity branch preserves the original normalized field bytes. Inputs
    are already support-centered/L2 normalized; no new centering occurs here.
    """
    z = np.asarray(z)
    factors = np.asarray(factors)
    if (z.ndim < 1 or not z.shape[-1] or z.dtype.kind != 'f' or not np.isfinite(z).all()
            or factors.shape != (z.shape[-1],) or factors.dtype.kind != 'f'
            or not np.isfinite(factors).all() or np.any(factors <= 0)
            or np.any(factors < np.exp(-LOG_FACTOR_BOUND) - 2e-7)
            or np.any(factors > np.exp(LOG_FACTOR_BOUND) + 2e-7)):
        raise ValueError('Expected finite normalized vectors and bounded positive metric factors')
    value = np.asarray(z, dtype=np.float64)
    if np.array_equal(factors, np.ones(factors.shape)):
        return value.copy()
    value = value * factors.astype(np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), EPS)


normalize_scaled = transform_vectors


def numpy_objective(case_centroids, y5, temperature, *, head='malignant', theta=None, factors=None):
    """Independent float64 loss with support self-exclusion, not model exclusion.

    Returns eligible query rows in their original fitting-case order. A caller
    supplies theta OR factors; the omitted state means exact identity.
    """
    x, labels, center, selected, counts = _inputs(case_centroids, y5, temperature, head)
    classes = range(4 if head == 'malignant' else 2)
    if theta is not None and factors is not None:
        raise ValueError('Provide theta or factors, not both')
    if theta is not None:
        log_factors, factors = factors_from_theta(theta)
    elif factors is not None:
        factors = np.asarray(factors, dtype=np.float64)
        if np.any(factors <= 0):
            raise ValueError('Metric factors must be positive')
        log_factors = np.log(factors)
    else:
        factors = np.ones(x.shape[-1], dtype=np.float64)
        log_factors = np.zeros(x.shape[-1], dtype=np.float64)
    if factors.shape != (x.shape[-1],):
        raise ValueError('Metric dimension differs from descriptors')
    z = transform_vectors(x, factors)
    cosine = np.clip(np.einsum('imk,jmk->mij', z, z), -1., 1.)
    scaled = cosine / float(temperature)
    np.fill_diagonal(scaled[0], -np.inf)
    for view in range(1, 4):
        np.fill_diagonal(scaled[view], -np.inf)
    density = []
    for label in classes:
        mask = labels == label
        denominator = int(mask.sum()) - (labels == label).astype(int)
        density.append(logsumexp(scaled[:, :, mask], axis=-1) - np.log(denominator)[None])
    density = np.stack(density, axis=-1)
    logs = density - logsumexp(density, axis=-1, keepdims=True)
    losses = -logs[:, np.arange(len(labels)), labels].mean(axis=0)
    ce = float(np.mean([losses[labels == label].mean() for label in classes]))
    penalty = float(REGULARIZATION * np.mean(log_factors ** 2))
    result = {'loss': ce + penalty, 'ce': ce, 'penalty': penalty,
              'head_log_probabilities': logs.transpose(1, 0, 2),
              'head_indices': selected, 'centers': center, 'fitting_counts': counts}
    if head == 'malignant':
        result.update(malignant_log_probabilities=result['head_log_probabilities'], malignant_indices=selected)
    return result


def torch_objective(normalized_malignant, malignant_labels, temperature, theta, *, head='malignant'):
    """Differentiable FP32 objective on fitting-only pre-normalized centroids."""
    import torch
    if head not in ('malignant', 'binary'):
        raise ValueError('Only binary and malignant metric heads are registered')
    classes = range(4 if head == 'malignant' else 2)
    if (normalized_malignant.ndim != 3 or normalized_malignant.shape[1] != 4
            or theta.shape != (normalized_malignant.shape[-1],)
            or malignant_labels.shape != (normalized_malignant.shape[0],)):
        raise ValueError('Invalid malignant descriptor/label/theta shapes')
    a = LOG_FACTOR_BOUND * torch.tanh(theta)
    scaled_vectors = normalized_malignant * torch.exp(a)
    z = scaled_vectors / torch.linalg.vector_norm(scaled_vectors, dim=-1, keepdim=True).clamp_min(EPS)
    cosine = torch.einsum('imk,jmk->mij', z, z).clamp(-1., 1.)
    scores = cosine / float(temperature)
    scores = scores.masked_fill(torch.eye(len(malignant_labels), device=scores.device, dtype=torch.bool)[None], -torch.inf)
    columns = []
    for label in classes:
        selected = malignant_labels == label
        denominator = selected.sum() - selected.to(dtype=scores.dtype)
        columns.append(torch.logsumexp(scores[:, :, selected], dim=-1) - torch.log(denominator)[None])
    logs = torch.log_softmax(torch.stack(columns, dim=-1), dim=-1)
    losses = -logs[:, torch.arange(len(malignant_labels), device=logs.device), malignant_labels].mean(0)
    ce = torch.stack([losses[malignant_labels == label].mean() for label in classes]).mean()
    penalty = REGULARIZATION * a.square().mean()
    result = {'loss': ce + penalty, 'ce': ce, 'penalty': penalty,
              'head_log_probabilities': logs.permute(1, 0, 2)}
    if head == 'malignant':
        result['malignant_log_probabilities'] = result['head_log_probabilities']
    return result


def _scalars(result):
    return {key: float(result[key].detach() if hasattr(result[key], 'detach') else result[key])
            for key in ('loss', 'ce', 'penalty')}


def fit_metric(case_centroids, y5, temperature, *, head='malignant', device='cpu'):
    """Fit exactly one fixed 200-update diagonal metric, returning private state.

    The production runner enforces the registered 46 x 4 x 1536 support shape;
    the mathematical implementation also accepts smaller synthetic fixtures.
    """
    import torch
    x, labels, center, selected, counts = _inputs(case_centroids, y5, temperature, head)
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Registered FP32 fitting supports CPU or CUDA only')
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous = {'deterministic': torch.are_deterministic_algorithms_enabled(),
                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark}
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(SEED)
            vectors = torch.as_tensor(x, dtype=torch.float32, device=device)
            truth = torch.as_tensor(labels, dtype=torch.long, device=device)
            theta = torch.nn.Parameter(torch.zeros(x.shape[-1], dtype=torch.float32, device=device))
            optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE, betas=(.9, .999), eps=1e-8, weight_decay=0.)
            initial_torch = _scalars(torch_objective(vectors, truth, temperature, theta, head=head))
            history = []
            for step in range(1, STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                objective = torch_objective(vectors, truth, temperature, theta, head=head)
                if not all(torch.isfinite(objective[key]).all() for key in ('loss', 'ce', 'penalty')):
                    raise ValueError('Nonfinite supervised metric objective')
                objective['loss'].backward()
                if theta.grad is None or not torch.isfinite(theta.grad).all():
                    raise ValueError('Missing or nonfinite metric gradient')
                row = {'step': step, 'loss_before_update': float(objective['loss'].detach()),
                       'ce_before_update': float(objective['ce'].detach()),
                       'penalty_before_update': float(objective['penalty'].detach()),
                       'gradient_l2': float(torch.linalg.vector_norm(theta.grad).detach())}
                optimizer.step()
                if not torch.isfinite(theta).all():
                    raise ValueError('Nonfinite metric parameter state')
                history.append(row)
            with torch.no_grad():
                final_torch = _scalars(torch_objective(vectors, truth, temperature, theta, head=head))
                saved_log_factors = (LOG_FACTOR_BOUND * torch.tanh(theta)).cpu().numpy().astype(np.float64)
                saved_factors = torch.exp(LOG_FACTOR_BOUND * torch.tanh(theta)).cpu().numpy().astype(np.float64)
            saved_theta = theta.detach().cpu().numpy().copy()
    finally:
        torch.use_deterministic_algorithms(previous['deterministic'], warn_only=previous['warn_only'])
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = previous['cudnn_tf32']
        torch.backends.cudnn.deterministic = previous['cudnn_deterministic']
        torch.backends.cudnn.benchmark = previous['cudnn_benchmark']
    log_factors, factors = saved_log_factors, saved_factors
    initial_numpy = _scalars(numpy_objective(case_centroids, y5, temperature, head=head))
    final_numpy = _scalars(numpy_objective(case_centroids, y5, temperature, head=head, factors=factors))
    receipt = {'recipe': dict(TRAINING_SPEC), 'device': str(device), 'torch_version': str(torch.__version__),
               'cases': len(case_centroids), 'head': head, 'head_cases': len(selected), 'views': 4,
               'dimension': x.shape[-1], 'fitting_counts': counts.tolist(), 'temperature': float(temperature),
               'head_counts': np.bincount(labels).tolist(),
               'centers_sha256': hashlib.sha256(center.tobytes()).hexdigest(),
               'training_vectors_sha256': hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest(),
               'theta_sha256': hashlib.sha256(saved_theta.tobytes()).hexdigest(),
               'initial_numpy_objective': initial_numpy, 'final_numpy_objective': final_numpy,
               'initial_torch_objective': initial_torch, 'final_torch_objective': final_torch,
               'factor_min': float(factors.min()), 'factor_max': float(factors.max()),
               'fitting_diagnostic': FITTING_DIAGNOSTIC, 'validation_selection_opportunities': 0,
               'deterministic_algorithms': True, 'tf32': False,
               'factor_storage': 'actual Torch FP32 log-factors and factors cast to float64'}
    return {'theta': saved_theta, 'log_factors': log_factors, 'factors': factors,
            'centers': center, 'history': history, 'receipt': receipt}

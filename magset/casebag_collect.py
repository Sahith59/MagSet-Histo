"""Fail-closed CPU collection of the registered 20-fit case-bag comparison.

This checker never decodes an image or executes a CNN forward.  It verifies
saved model/optimizer states and replays the declared linear readout from saved
sufficient statistics.  Successful collection is not a raw-image reproduction.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import tarfile
import time

import numpy as np
from scipy.special import logsumexp
import torch

from .baseline import SUBSETS as INDEX_SUBSETS, stratified_inner_split
from .capstone_binary import trusted_binary_labels
from .collect import SUBSETS, load_trusted_context
from .collect_adaptation import digest_state
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .five_collect import aggregate_summaries, assert_public_safe, read_json, require_equal
from .five_metrics import FINAL5, MAP8_TO5, five_labels, summarize_records


PRODUCER_SHA = 'b5f0df18a47c1607be492bdaab7ab6cd2171226b'
ARCHIVE_SHA256 = 'd4655c0290997adc5eaf19aef69c2c5f3e6ab460df66e443e98ff0532613e9ab'
PRODUCER_SHA256 = 'e34aba8a2112e67b90960cafbeb5aaca1e4c7c8b0568c6f78f4d7af75178663d'
CONFIG_SHA256 = '3f633d0e43f66f51c4e27b3324a17180cda2bf7f2a91c9a757d7b1187da7d7b9'
ARMS = ('direct', 'dual')
CONTEXT_COMMON = ('code_sha', 'source_archive_sha256', 'implementation_sha256',
                  'config', 'config_sha256', 'manifest_sha256', 'protocol_sha256',
                  'weights_sha256', 'membership_fingerprint_sha256')
HASH = re.compile(r'[0-9a-f]{64}')


def _integer(value, expected, name):
    if type(value) is not int or value != expected:
        raise ValueError(name + ': integer count differs')


def _finite(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(name + ': finite nonnegative value required')
    return value


def _hash(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(name + ': SHA256 required')


def verify_archive(archive, source, expected_sha, expected_archive_sha256):
    """Check actual git-archive bytes and every extracted tracked file, read-only."""
    source = Path(source).resolve()
    if file_hash(archive) != expected_archive_sha256:
        raise ValueError('Training source archive bytes differ')
    paths = set()
    with tarfile.open(archive, 'r:') as stream:
        if stream.pax_headers.get('comment') != expected_sha:
            raise ValueError('Training git archive commit identity differs')
        for member in stream.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or member.name in paths
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Unexpected or duplicate source archive member')
            paths.add(member.name)
            path = source.joinpath(*name.parts)
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError('Training source contains a symlink/escaping member')
            if member.isfile():
                if not path.is_file() or path.read_bytes() != stream.extractfile(member).read():
                    raise ValueError('Extracted training source differs from archive: ' + member.name)
            elif not path.is_dir():
                raise ValueError('Extracted training source directory missing')
    return paths


def load_producer(source, archive, expected_sha, config_path):
    source = Path(source).resolve()
    if expected_sha != PRODUCER_SHA:
        raise ValueError('This collector is registered for the frozen case-bag producer only')
    if ((source / '.source-sha').read_text().strip() != expected_sha
            or (source / '.archive-sha256').read_text().strip() != ARCHIVE_SHA256):
        raise ValueError('Training source identity receipts differ')
    paths = verify_archive(archive, source, expected_sha, ARCHIVE_SHA256)
    producer_path = source / 'magset/five_casebag.py'
    canonical = source / 'configs/breakhis-casebag-v1.json'
    if (file_hash(producer_path) != PRODUCER_SHA256 or file_hash(canonical) != CONFIG_SHA256
            or Path(config_path).read_bytes() != canonical.read_bytes()):
        raise ValueError('Frozen producer or canonical config bytes differ')
    # Relative imports in the separately loaded producer use this package.  Check
    # all its archived Python modules, including transitive imports, beforehand.
    local = Path(__file__).resolve().parent
    for name in paths:
        path = PurePosixPath(name)
        if path.parts[0] == 'magset' and path.suffix == '.py':
            current = local.joinpath(*path.parts[1:])
            if not current.is_file() or current.read_bytes() != (source / name).read_bytes():
                raise ValueError('Collector/producer shared Python dependency differs: ' + name)
    spec = importlib.util.spec_from_file_location('magset._producer_casebag', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    config = read_json(canonical)
    producer.validate_config(config)
    return producer, config


def checker_environment(config):
    """CPU execution intentionally retains the exact CUDA-compiled libraries."""
    versions = {name: importlib.metadata.version(name) for name in config['dependencies']}
    if versions != config['dependencies']:
        raise ValueError('CPU checker must use the pinned CUDA-build dependency environment')
    return {'dependencies': versions, 'python': platform.python_version(),
            'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu_family': config['environment_policy']['gpu_family'], 'device_type': 'cuda',
            'CUBLAS_WORKSPACE_CONFIG': config['optimization']['CUBLAS_WORKSPACE_CONFIG']}


def trusted_inputs(manifest, protocol, config, source, weights, producer):
    """Rebuild original-eight memberships before pooling, without image access."""
    trusted = load_trusted_context(manifest, protocol)
    for name in ('manifest_sha256', 'protocol_sha256'):
        require_equal(trusted[name], config[name], 'trusted/' + name, 0)
    rows = read_manifest(manifest)
    broad = trusted_binary_labels(rows)
    cases = trusted['cases']
    parts = {}
    fingerprint_rows = []
    for seed in config['seeds']:
        for trial in config['trials']:
            indices = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
            role_cases = [sorted([cases[i] for i in ix], key=lambda value: value['case_id']) for ix in indices]
            roles = {name: [case['case_id'] for case in values]
                     for name, values in zip(('train', 'validation', 'excluded'), role_cases)}
            if ([len(values) for values in role_cases] != config['expected_role_counts']
                    or len(set.union(*(set(value) for value in roles.values()))) != len(cases)
                    or sum(map(len, roles.values())) != len(cases)):
                raise ValueError('Trusted role coverage or disjointness differs')
            for name, values in zip(('fitting', 'validation', 'excluded'), role_cases):
                support = np.bincount(five_labels(np.asarray([c['label'] for c in values])), minlength=5)
                require_equal(support.tolist(), config[name + '_five_support'], 'trusted/' + name, 0)
            for case in cases:
                if broad[case['case_id']] != int(MAP8_TO5[case['label']] > 0):
                    raise ValueError('Trusted broad label mapping differs')
            for values in role_cases[:2]:
                allowed = set(case['case_id'] for case in values)
                producer.case_rows([row for row in rows if case_id(row) in allowed], values)
            parts[seed, trial] = (roles, *role_cases)
            fingerprint_rows.append({'seed': seed, 'trial': trial,
                                     'roles': {name: sorted(roles[old]) for name, old in
                                               (('train', 'train'), ('validation', 'validation'), ('test', 'excluded'))}})
    membership = hashlib.sha256(json.dumps(fingerprint_rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if membership != config['membership_fingerprint_sha256']:
        raise ValueError('Original-eight complete role fingerprint differs')
    if (file_hash(weights) != config['model']['weight_sha256']
            or Path(weights).stat().st_size != config['model']['weight_size_bytes']):
        raise ValueError('Pinned pretrained weight bytes differ')
    common = {'code_sha': PRODUCER_SHA, 'source_archive_sha256': ARCHIVE_SHA256,
              'implementation_sha256': {name: file_hash(Path(source) / 'magset' / name)
                                        for name in producer.SOURCE_FILES},
              'config': config, 'config_sha256': CONFIG_SHA256,
              'manifest_sha256': trusted['manifest_sha256'], 'protocol_sha256': trusted['protocol_sha256'],
              'weights_sha256': file_hash(weights), 'membership_fingerprint_sha256': membership,
              'environment': checker_environment(config)}
    return common, parts


def run_context(common, roles, arm, seed, trial, smoke=False, acceptance_sha256=None):
    value = dict(common, arm=arm, seed=seed, trial=trial, smoke=smoke, case_ids=roles,
                 role_sha256=hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest())
    if not smoke:
        value['smoke_acceptance_sha256'] = acceptance_sha256
    return value


def _load_state(path, context, model, producer):
    saved = producer.bound_checkpoint(path, context)
    template = model.state_dict()
    state = saved.get('model', {})
    if set(state) != set(template):
        raise ValueError('Checkpoint model keys differ from pinned architecture')
    for name, expected in template.items():
        value = state[name]
        if (not torch.is_tensor(value) or value.shape != expected.shape or value.dtype != expected.dtype
                or not torch.isfinite(value).all()):
            raise ValueError('Checkpoint model shape, dtype or finite values differ')
    model.load_state_dict(state, strict=True)
    return saved


def _optimizer(model, config, producer):
    selected = producer.adaptation_names(model)
    groups = [{'params': [parameter for name, parameter in model.named_parameters() if name in selected]},
              {'params': list(model.heads.parameters())}]
    opt = config['optimization']
    return torch.optim.AdamW(groups, lr=opt['head_lr'], weight_decay=opt['weight_decay'],
                             betas=tuple(opt['adamw_betas']), eps=opt['adamw_eps'])


def _check_rng(rng):
    if not isinstance(rng, dict) or set(rng) != {'torch', 'cuda', 'python', 'numpy'}:
        raise ValueError('Checkpoint RNG schema differs')
    values = rng['cuda']
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError('Checkpoint must retain its one-device CUDA RNG state')
    for tensor in [rng['torch'], *values]:
        if not torch.is_tensor(tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1 or not tensor.numel():
            raise ValueError('Checkpoint RNG tensor schema differs')
    if rng['torch'].shape != torch.get_rng_state().shape:
        raise ValueError('Checkpoint CPU RNG state length differs')
    try:
        random.Random().setstate(rng['python'])
        state = rng['numpy']
        if (len(state) != 5 or not torch.is_tensor(state[1]) or state[1].dtype != torch.int64
                or state[1].shape != (624,) or torch.any((state[1] < 0) | (state[1] > 2 ** 32 - 1))):
            raise ValueError('Invalid NumPy MT19937 state')
        np.random.RandomState().set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    except (TypeError, IndexError, OverflowError) as error:
        raise ValueError('Checkpoint RNG state is invalid') from error


def verify_lifecycle(saved, report, model, initial, fitting, validation, config, context, folder, producer, smoke=False):
    """Use pinned resume checks plus stricter complete-output and phase proofs."""
    epochs = config['budget']['technical_smoke_epochs'] if smoke else config['optimization']['epochs']
    warmup = 1 if smoke else config['optimization']['warmup_epochs']
    _integer(saved.get('epoch'), epochs, 'completed epochs')
    proof = report['proof']
    require_equal(proof['initial'], initial, 'pretrained initialization', 0)
    require_equal(saved['initial'], initial, 'checkpoint initialization', 0)
    producer.verify_training_checkpoint(saved, model, _optimizer(model, config, producer), fitting,
                                        validation, config, context, folder, smoke)
    for key in ('initial', 'history', 'candidates', 'best', 'smoke_reloads'):
        require_equal(proof[key], saved[key], 'proof/checkpoint/' + key, 0)
    require_equal(proof['model_receipt'], producer.model_receipt(model), 'model receipt', 0)
    require_equal(proof['final'], producer.state_hashes(model, saved['model']), 'final fingerprints', 0)
    if (proof['final']['bn'] != initial['bn'] or proof['final']['frozen'] != initial['frozen']
            or proof['final']['adaptive'] == initial['adaptive'] or proof['final']['heads'] == initial['heads']):
        raise ValueError('Frozen BN/early and changed adaptive/head proof failed')
    heads = {name for name, _ in model.named_parameters() if name.startswith('heads.')}
    selected = producer.adaptation_names(model)
    previous = initial
    for row in saved['history']:
        active = heads | (selected if row['epoch'] > warmup else set())
        if (row['gradient_parameter_names'] != sorted(active)
                or set(row['maximum_group_gradient_norms']) != {'heads', 'adaptive'}):
            raise ValueError('Epoch gradient whitelist differs')
        for key, value in row['fingerprints'].items():
            _hash(value, 'epoch fingerprint/' + key)
        if set(row['fingerprints']) != {'bn', 'frozen', 'adaptive', 'backbone', 'heads'}:
            raise ValueError('Epoch fingerprint groups differ')
        if row['epoch'] <= warmup:
            if (row['fingerprints']['backbone'] != initial['backbone']
                    or row['fingerprints']['adaptive'] != initial['adaptive']
                    or row['maximum_group_gradient_norms']['adaptive'] != 0):
                raise ValueError('Warmup backbone or adaptive gradients changed')
        elif row['fingerprints']['adaptive'] == previous['adaptive']:
            raise ValueError('Adaptive parameters did not change during an adaptation epoch')
        if row['fingerprints']['heads'] == previous['heads']:
            raise ValueError('Head parameters did not change during an epoch')
        previous = row['fingerprints']
        components = row['loss_components']
        if set(components) != ({'direct'} if model.arm == 'direct' else {'binary', 'malignant'}):
            raise ValueError('Epoch loss component schema differs')
        for value in components.values():
            _finite(value, 'epoch component loss')
        expected_loss = components['direct'] if model.arm == 'direct' else .5 * sum(components.values())
        require_equal(row['loss'], expected_loss, 'epoch weighted objective', 2e-6)
        _finite(row['elapsed_seconds'], 'epoch elapsed seconds')
    completed = sum(row['cases'] * 8 for row in saved['history'])
    _integer(proof['completed_epoch_training_field_forwards'], completed, 'completed field forwards')
    # A report is committed only after whole epochs.  A resumed invocation can
    # cover a suffix of the fixed epoch sequence, never a fractional epoch or
    # an arbitrary subset of candidate evaluations.
    epoch_fields = saved['history'][-1]['cases'] * 8
    invocation = proof['invocation_training_field_forwards']
    if type(invocation) is not int or not 0 <= invocation <= completed or invocation % epoch_fields:
        raise ValueError('Invocation training field forwards do not cover whole epochs')
    invocation_epochs = invocation // epoch_fields
    candidate_epochs = [] if smoke else [e for e in config['selection']['epochs'] if e > epochs - invocation_epochs]
    expected_candidate_fields = len(candidate_epochs) * sum(len(c['indices'][v]) for c in validation for v in MAGNIFICATIONS)
    _integer(proof['invocation_candidate_validation_field_forwards'], expected_candidate_fields,
             'invocation candidate-validation field forwards')
    if proof['last_checkpoint_sha256'] != file_hash(Path(folder) / 'last.pt'):
        raise ValueError('Authoritative last checkpoint bytes differ')
    if set(proof['reload']) != {'feature_max_abs', 'probability_max_abs'}:
        raise ValueError('Reload proof schema differs')
    for value in proof['reload'].values():
        if _finite(value, 'same-A40 reload error') > config['inference']['selected_reload_tolerance']:
            raise ValueError('Same-A40 reload error exceeds tolerance')
    _check_rng(saved['rng'])


def replay_scores(arrays, descriptors, arm):
    """Independent stable affine/probability replay, including hard route ties."""
    x = np.asarray(descriptors, dtype=np.float64)
    if arm == 'direct':
        logits = x @ arrays['direct_coef'].astype(np.float64).T + arrays['direct_intercept'].astype(np.float64)
        log5 = logits - logsumexp(logits, axis=1, keepdims=True)
        log_r = logsumexp(log5[:, 1:], axis=1)
        binary_log = np.column_stack((log5[:, 0], log_r))
        malignant_log = log5[:, 1:] - log_r[:, None]
        prediction = np.argmax(log5, axis=1)
    elif arm == 'dual':
        a = (x @ arrays['binary_coef'].astype(np.float64).T + arrays['binary_intercept'].astype(np.float64))[:, 0]
        binary_log = np.column_stack((-np.logaddexp(0., a), -np.logaddexp(0., -a)))
        z = x @ arrays['malignant_coef'].astype(np.float64).T + arrays['malignant_intercept'].astype(np.float64)
        malignant_log = z - logsumexp(z, axis=1, keepdims=True)
        log5 = np.column_stack((binary_log[:, 0], binary_log[:, 1, None] + malignant_log))
        prediction = np.where(np.exp(binary_log[:, 1]) >= .5, np.argmax(malignant_log, axis=1) + 1, 0)
    else:
        raise ValueError('Expected direct or dual replay arm')
    return {'prediction': prediction, 'probabilities5': np.exp(log5), 'log_probabilities5': log5,
            'binary_score': np.clip(np.exp(binary_log[:, 1]), 0, 1), 'binary_log_probabilities': binary_log,
            'malignant_probabilities4': np.exp(malignant_log), 'malignant_log_probabilities': malignant_log}


def verify_scores(folder, prefix, cases, role, arm, state, dimension, hashes):
    """Validate exact saved descriptors/head bytes and replay every saved score."""
    folder = Path(folder)
    feature_path = folder / (prefix + '-features.npz')
    score_path = folder / (prefix + '-predictions.jsonl')
    for path, key in ((feature_path, 'features_sha256'), (score_path, 'predictions_sha256')):
        if file_hash(path) != hashes[key]:
            raise ValueError('Saved feature/prediction artifact bytes differ')
    with np.load(feature_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
        if len(archive.files) != len(data):
            raise ValueError('Duplicate NPZ array name')
    heads = {'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}
    expected_keys = {'case_ids', 'truth8', 'truth5', 'view_features', 'field_counts', 'descriptors',
                     'descriptor_subset_masks', 'descriptor_magnifications'}
    expected_keys.update(name + suffix for name in heads for suffix in ('_coef', '_intercept', '_classes'))
    if set(data) != expected_keys:
        raise ValueError('Saved sufficient-statistic array schema differs')
    if (data['case_ids'].dtype.kind != 'U' or data['case_ids'].tolist() != [c['case_id'] for c in cases]
            or data['truth8'].dtype != np.int64 or data['truth5'].dtype != np.int64
            or not np.array_equal(data['truth8'], [c['label'] for c in cases])
            or not np.array_equal(data['truth5'], [MAP8_TO5[c['label']] for c in cases])):
        raise ValueError('Saved case identities or original/final truth differ from trusted role')
    counts = [[len(c['indices'][view]) for view in MAGNIFICATIONS] for c in cases]
    if (data['field_counts'].dtype != np.int64 or not np.array_equal(data['field_counts'], counts)
            or np.any(data['field_counts'] <= 0)):
        raise ValueError('Saved field counts differ from trusted all-field bags')
    views = data['view_features']
    if views.dtype != np.float64 or views.shape != (len(cases), 4, dimension) or not np.isfinite(views).all():
        raise ValueError('Saved full-field view means shape/dtype/finite values differ')
    masks = [sum(1 << i for i in subset) for subset in INDEX_SUBSETS]
    if (data['descriptor_subset_masks'].dtype != np.int64 or not np.array_equal(data['descriptor_subset_masks'], masks)
            or data['descriptor_magnifications'].dtype.kind != 'U'
            or data['descriptor_magnifications'].tolist() != ['+'.join(map(str, v)) for v in SUBSETS]):
        raise ValueError('Saved subset axis differs')
    descriptors = []
    for subset in INDEX_SUBSETS:
        flags = np.zeros((len(cases), 4), dtype=np.float64)
        flags[:, list(subset)] = 1
        descriptors.append(np.concatenate((np.mean(views[:, list(subset)], axis=1, dtype=np.float64), flags), axis=1).astype(np.float32))
    descriptors = np.stack(descriptors, axis=1)
    if (data['descriptors'].dtype != np.float32 or not np.array_equal(data['descriptors'], descriptors)):
        raise ValueError('Saved descriptors differ bitwise from full-field view means')
    arrays = {}
    for name, count in heads.items():
        for suffix, key, shape in (('_coef', 'weight', (count, dimension + 4)), ('_intercept', 'bias', (count,))):
            value = data[name + suffix]
            tensor = state['heads.' + name + '.' + key]
            if (value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all()
                    or tensor.dtype != torch.float32 or not np.array_equal(value, tensor.numpy())):
                raise ValueError('Saved linear head differs from bound checkpoint')
            arrays[name + suffix] = value
        classes = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
        if data[name + '_classes'].dtype != np.int64 or not np.array_equal(data[name + '_classes'], classes):
            raise ValueError('Saved head class order differs')
    records = []
    for mask, magnifications in enumerate(SUBSETS):
        scores = replay_scores(arrays, descriptors[:, mask], arm)
        for index, case in enumerate(cases):
            row = {'role': role, 'case_id': case['case_id'], 'truth': int(case['label']),
                   'truth5': MAP8_TO5[case['label']], 'magnifications': list(magnifications)}
            row.update({name: value[index].tolist() for name, value in scores.items()})
            records.append(row)
    # Strict parser rejects duplicate keys and NaN/Infinity at every JSON level.
    from .five_collect import _unique_object, _invalid_constant
    actual = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
              for line in score_path.read_text().splitlines()]
    require_equal(actual, records, 'independent saved-head replay', 1e-10)
    summary = summarize_records(actual, arm)
    return summary, data, actual


def expected_files(smoke=False):
    names = {'started.json', 'last.pt', 'metrics.json'}
    if not smoke:
        names.add('selected.pt')
        for prefix in ['candidate-epoch' + str(e) for e in (10, 20, 35, 50)]:
            names.update({prefix + '.pt', prefix + '-features.npz', prefix + '-predictions.jsonl'})
        for role in ('train', 'validation'):
            names.update({'selected-' + role + '-features.npz', 'selected-' + role + '-predictions.jsonl'})
    return names


def check_files(folder, smoke=False):
    folder = Path(folder)
    actual = list(folder.rglob('*'))
    if (any(path.is_symlink() or not path.is_file() for path in actual)
            or {path.relative_to(folder).as_posix() for path in actual} != expected_files(smoke)):
        raise ValueError('Complete run artifact set differs, including missing/extra/forbidden roles')


def verify_resources(report, proof, fitting, validation, config, smoke=False):
    _integer(report.get('outer_predictions'), 0, 'outer predictions')
    if smoke:
        _integer(report.get('validation_predictions'), 0, 'smoke validation predictions')
    _finite(report.get('elapsed_seconds'), 'reported final-invocation elapsed seconds')
    if not isinstance(report.get('slurm_job_id'), str) or not re.fullmatch(r'\d+', report['slurm_job_id']):
        raise ValueError('Scheduler execution receipt required')
    for name in ('peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
        value = report.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError('Actual allocated/reserved GPU memory receipt required')
    if report['peak_gpu_reserved_bytes'] < report['peak_gpu_allocated_bytes']:
        raise ValueError('Reserved GPU memory is lower than allocated memory')
    counts = {'training': proof['invocation_training_field_forwards'],
              'candidate_validation': proof['invocation_candidate_validation_field_forwards'],
              'selected_fitting': 0 if smoke else sum(len(c['indices'][v]) for c in fitting for v in MAGNIFICATIONS),
              'selected_validation': 0 if smoke else sum(len(c['indices'][v]) for c in validation for v in MAGNIFICATIONS),
              'reload_probes': 8}
    require_equal(report['invocation_field_forwards'], counts, 'actual final-invocation field budget', 0)
    return {'final_invocation_elapsed_seconds': report['elapsed_seconds'],
            'committed_training_epoch_seconds': sum(row['elapsed_seconds'] for row in proof['history']),
            'peak_gpu_allocated_bytes': report['peak_gpu_allocated_bytes'],
            'peak_gpu_reserved_bytes': report['peak_gpu_reserved_bytes'],
            'completed_training_field_forwards': proof['completed_epoch_training_field_forwards'],
            'final_invocation_field_forwards': counts}


def verify_smokes(path, common, parts, config, models, initial, producer):
    receipt = read_json(path)
    entries = receipt.get('reports', [])
    if (receipt.get('status') != 'accepted case-bag technical smokes' or len(entries) != 2
            or {entry['arm'] for entry in entries} != set(ARMS)):
        raise ValueError('Both accepted fitting-only technical smokes required')
    results = []
    roles, fitting, validation, _ = parts[812, 1]
    for entry in sorted(entries, key=lambda value: value['arm']):
        arm = entry['arm']
        report_path = Path(entry['path'])
        check_files(report_path.parent, smoke=True)
        if report_path.name != 'metrics.json' or file_hash(report_path) != entry['sha256']:
            raise ValueError('Accepted smoke report bytes differ')
        report = read_json(report_path)
        context = run_context(common, roles, arm, 812, 1, smoke=True)
        require_equal(report['context'], context, 'accepted smoke context', 0)
        require_equal(read_json(report_path.parent / 'started.json'), context, 'smoke start context', 0)
        if report.get('status') != 'fitting-only case-bag smoke':
            raise ValueError('Accepted smoke status differs')
        saved = _load_state(report_path.parent / 'last.pt', context, models[arm], producer)
        verify_lifecycle(saved, report, models[arm], initial[arm], fitting, validation, config,
                         context, report_path.parent, producer, smoke=True)
        resources = verify_resources(report, report['proof'], fitting, validation, config, smoke=True)
        # Also exercise the frozen producer's admission function.  Both smoke
        # checkpoints were fully revalidated above, beyond its report checks.
        results.append({'arm': arm, 'report_sha256': file_hash(report_path),
                        'last_checkpoint_sha256': file_hash(report_path.parent / 'last.pt'),
                        'reload': report['proof']['reload'], 'resources': resources})
    producer.accepted_smokes(path, run_context(common, roles, 'direct', 812, 1))
    return results


def verify_run(folder, arm, seed, trial, config, context, fitting, validation, model, initial, producer):
    folder = Path(folder)
    check_files(folder)
    report = read_json(folder / 'metrics.json')
    require_equal(report['context'], context, 'production context', 0)
    require_equal(read_json(folder / 'started.json'), context, 'production start context', 0)
    if report.get('status') != 'complete case-bag validation run':
        raise ValueError('Incomplete production report')
    saved = _load_state(folder / 'last.pt', context, model, producer)
    verify_lifecycle(saved, report, model, initial, fitting, validation, config, context, folder, producer)
    proof = report['proof']
    candidates = []
    for candidate in proof['candidates']:
        epoch = candidate['epoch']
        prefix = 'candidate-epoch' + str(epoch)
        checkpoint = _load_state(folder / (prefix + '.pt'), context, model, producer)
        fingerprints = producer.state_hashes(model, checkpoint['model'])
        require_equal(fingerprints, saved['history'][epoch - 1]['fingerprints'], 'candidate/epoch fingerprints', 0)
        metric, _, _ = verify_scores(folder, prefix, validation, 'validation', arm,
                                     checkpoint['model'], model.dimension, candidate)
        require_equal(candidate['validation'], metric, 'candidate all15 validation metrics')
        candidates.append({'epoch': epoch, 'validation': metric,
                           **{name: candidate[name] for name in ('state_sha256', 'checkpoint_sha256',
                                                                  'features_sha256', 'predictions_sha256')}})
    if [value['epoch'] for value in candidates] != config['selection']['epochs']:
        raise ValueError('All four candidate checkpoints required before ranking')
    best = max(candidates, key=lambda value: value['validation']['mean_subset_macro_f1'])
    require_equal(proof['best'], best, 'earliest maximal validation selection', 0)
    selected = _load_state(folder / 'selected.pt', context, model, producer)
    if selected['epoch'] != best['epoch'] or digest_state(selected['model']) != best['state_sha256']:
        raise ValueError('Selected checkpoint does not match earliest maximal candidate')
    if digest_state(saved['best_model']) != best['state_sha256']:
        raise ValueError('Authoritative selected state snapshot differs')
    selected_summaries = {}
    replay_data = {}
    replay_records = {}
    for role, cases in (('train', fitting), ('validation', validation)):
        metric, data, records = verify_scores(folder, 'selected-' + role, cases, role, arm,
                                               selected['model'], model.dimension, report[role + '_artifacts'])
        require_equal(report[role], metric, 'selected/' + role + ' all15 metrics')
        selected_summaries[role] = metric
        replay_data[role], replay_records[role] = data, records
    with np.load(folder / ('candidate-epoch%d-features.npz' % best['epoch']), allow_pickle=False) as archive:
        difference = float(np.max(np.abs(archive['view_features'] - replay_data['validation']['view_features'])))
    candidate_records = [json.loads(line) for line in
                         (folder / ('candidate-epoch%d-predictions.jsonl' % best['epoch'])).read_text().splitlines()]
    selected_records = replay_records['validation']
    probability_error = float(np.max(np.abs(np.asarray([r['probabilities5'] for r in candidate_records]) -
                                             np.asarray([r['probabilities5'] for r in selected_records]))))
    if ([row['prediction'] for row in candidate_records] != [row['prediction'] for row in selected_records]
            or difference > config['inference']['selected_reload_tolerance']
            or probability_error > config['inference']['selected_reload_tolerance']):
        raise ValueError('Selected validation replay decisions/features/probabilities differ')
    require_equal(report['selected_feature_replay_max_abs'], difference, 'selected feature replay', 1e-12)
    require_equal(report['selected_probability_replay_max_abs'], probability_error, 'selected probability replay', 1e-12)
    require_equal(selected_summaries['validation']['mean_subset_macro_f1'], best['validation']['mean_subset_macro_f1'],
                  'selected validation F1', 1e-12)
    resources = verify_resources(report, proof, fitting, validation, config)
    return {'arm': arm, 'seed': seed, 'trial': trial, 'selected_epoch': best['epoch'],
            'candidates': candidates, 'fitting': selected_summaries['train'],
            'validation': selected_summaries['validation'], 'resources': resources,
            'metrics_sha256': file_hash(folder / 'metrics.json'),
            'last_checkpoint_sha256': file_hash(folder / 'last.pt'),
            'selected_checkpoint_sha256': file_hash(folder / 'selected.pt'),
            'selected_state_sha256': best['state_sha256'], 'initial_state_groups': initial,
            'final_state_groups': proof['final'], 'model_receipt': proof['model_receipt'],
            'selected_artifacts': {role: report[role + '_artifacts'] for role in ('train', 'validation')},
            'reload': proof['reload'], 'selected_feature_replay_max_abs': difference,
            'selected_probability_replay_max_abs': probability_error,
            'committed_epochs': len(proof['history']),
            'committed_optimizer_updates': sum(row['steps'] for row in proof['history'])}


def verify_matrix_paths(root, config):
    root = Path(root)
    expected = {(arm, seed, trial): root / arm / ('split%d' % seed) / ('trial%d' % trial)
                for arm in ARMS for seed in config['seeds'] for trial in config['trials']}
    files = {folder / name for folder in expected.values() for name in expected_files()}
    actual = list(root.rglob('*'))
    if (any(path.is_symlink() for path in actual)
            or {path for path in actual if path.is_file()} != files):
        raise ValueError('Full registered 20-fit matrix has missing or extra artifacts')
    return expected


def reference_comparison(path, config):
    expected = config['comparison_gate']['reference_comparison_sha256']
    if expected != config['accepted_frozen_comparison_sha256'] or file_hash(path) != expected:
        raise ValueError('Accepted frozen direct-five reference comparison bytes differ')
    value = read_json(path)
    if (value.get('status') != 'complete exploratory five-class development comparison'
            or value.get('verified_runs') != 10 or value.get('class_names') != list(FINAL5)
            or value.get('outer_predictions') != 0):
        raise ValueError('Accepted reference comparison scope differs')
    result = {str(seed): value['summary'][str(seed)]['direct']['validation'] for seed in config['seeds']}
    assert_public_safe(result)
    return result


def aggregate(runs, reference, config):
    summaries, matched, gate = {}, {}, {}
    for seed in config['seeds']:
        summary = {}
        for arm in ARMS:
            group = [run for run in runs if (run['arm'], run['seed']) == (arm, seed)]
            if sorted(run['trial'] for run in group) != config['trials']:
                raise ValueError('Complete five-trial group required before aggregate/ranking')
            summary[arm] = {role: aggregate_summaries([run[role] for run in group])
                            for role in ('fitting', 'validation')}
            summary[arm]['fitting_minus_validation'] = {
                key: summary[arm]['fitting'][key] - summary[arm]['validation'][key]
                for key in summary[arm]['validation'] if key.startswith('mean_subset_')}
        direct, dual = (summary[arm]['validation'] for arm in ARMS)
        matched[str(seed)] = {'delta_mean_subset_macro_f1': dual['mean_subset_macro_f1'] - direct['mean_subset_macro_f1'],
                              'delta_mean_subset_accuracy': dual['mean_subset_accuracy'] - direct['mean_subset_accuracy']}
        old = reference[str(seed)]
        delta_f1 = dual['mean_subset_macro_f1'] - old['mean_subset_macro_f1']
        delta_accuracy = dual['mean_subset_accuracy'] - old['mean_subset_accuracy']
        positive = all(value > 0 for value in dual['pooled_class_recall'])
        gate[str(seed)] = {'delta_mean_subset_macro_f1': delta_f1, 'delta_mean_subset_accuracy': delta_accuracy,
                           'all_five_recall_positive': positive,
                           'gate_passed': bool(delta_f1 >= config['comparison_gate']['margin_mean15_macroF1_each_arrangement']
                                               and delta_accuracy >= 0 and positive)}
        summaries[str(seed)] = summary
    return summaries, matched, gate


def collect(args):
    started = time.monotonic()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing comparison output directory')
    collector_root = Path(__file__).resolve().parents[1]
    collector_sha = (collector_root / '.source-sha').read_text().strip()
    collector_archive_sha = (collector_root / '.archive-sha256').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{40}', collector_sha) or not HASH.fullmatch(collector_archive_sha):
        raise ValueError('Collector requires its own immutable source identity receipts')
    producer, config = load_producer(args.training_source_root, args.training_source_archive,
                                     args.expected_code_sha, args.config)
    common, parts = trusted_inputs(args.manifest, args.protocol, config, args.training_source_root,
                                   args.weights, producer)
    paths = verify_matrix_paths(args.runs, config)
    reference = reference_comparison(args.reference_comparison, config)
    models, initial = {}, {}
    # Actual model construction is CPU-only and strictly loads pinned local
    # pretrained bytes.  No call below invokes model.forward or opens an image.
    for arm in ARMS:
        models[arm] = producer.build_model(config, args.weights, torch.device('cpu'), arm)
        initial[arm] = producer.state_hashes(models[arm])
    smokes = verify_smokes(args.smoke_acceptance, common, parts, config, models, initial, producer)
    acceptance_sha = file_hash(args.smoke_acceptance)
    runs = []
    for (arm, seed, trial), folder in sorted(paths.items()):
        roles, fitting, validation, _ = parts[seed, trial]
        context = run_context(common, roles, arm, seed, trial, acceptance_sha256=acceptance_sha)
        runs.append(verify_run(folder, arm, seed, trial, config, context, fitting, validation,
                               models[arm], initial[arm], producer))
    summary, matched, gate = aggregate(runs, reference, config)
    totals = {name: sum(run['resources']['final_invocation_field_forwards'][name] for run in runs)
              for name in ('training', 'candidate_validation', 'selected_fitting', 'selected_validation', 'reload_probes')}
    completed = sum(run['resources']['completed_training_field_forwards'] for run in runs)
    expected_logical = {'training': completed,
                        'candidate_validation': config['budget']['production_candidate_validation_field_forwards'],
                        'selected_fitting': totals['selected_fitting'], 'selected_validation': totals['selected_validation'],
                        'reload_probes': totals['reload_probes']}
    for key, config_key in (('training', 'production_field_forwards_excluding_evaluation'),
                             ('selected_fitting', 'production_selected_fitting_field_forwards'),
                             ('selected_validation', 'production_selected_validation_replay_field_forwards'),
                             ('reload_probes', 'production_reload_probe_field_forwards')):
        _integer(expected_logical[key], config['budget'][config_key], 'complete matrix logical forwards/' + key)
    _integer(sum(expected_logical.values()), config['budget']['production_total_field_forwards_with_fixed_probes'],
             'complete matrix total logical forwards')
    result = {key: common[key] for key in CONTEXT_COMMON if key != 'config'}
    result.update(status='complete exploratory case-bag five-class development comparison',
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES),
                  training_environment=common['environment'], verified_fits=len(runs),
                  candidate_checkpoints=sum(len(run['candidates']) for run in runs),
                  production_head_fits=30, selection='earliest maximum end-to-end hard-label mean15-subset macro F1',
                  routing_threshold=.5, routing_tie='malignant', subset_masks=config['subset_masks'],
                  smoke_acceptance_sha256=acceptance_sha, technical_smokes=smokes,
                  accepted_frozen_comparison_sha256=config['accepted_frozen_comparison_sha256'],
                  reference_direct_validation=reference, summary=summary, matched_direct_vs_dual=matched,
                  comparison_against_frozen_direct=gate, gate_passed=all(value['gate_passed'] for value in gate.values()),
                  gate_scope='descriptive screen on reused provider-case validation; not a fresh test or clinical criterion',
                  runs=runs, distinct_validation_cases=len({case for roles, *_ in parts.values() for case in roles['validation']}),
                  collector_new_fits=0, collector_cnn_forwards=0, collector_image_decodes=0, outer_predictions=0,
                  resources={'logical_completed_field_forwards': expected_logical,
                             'reported_final_invocation_field_forwards': totals,
                             'reported_final_invocation_elapsed_seconds': sum(run['resources']['final_invocation_elapsed_seconds'] for run in runs),
                             'committed_training_epoch_seconds': sum(run['resources']['committed_training_epoch_seconds'] for run in runs),
                             'peak_gpu_allocated_bytes': max(run['resources']['peak_gpu_allocated_bytes'] for run in runs),
                             'peak_gpu_reserved_bytes': max(run['resources']['peak_gpu_reserved_bytes'] for run in runs),
                             'requested_production_gpu_hours': config['budget']['production_requested_gpu_hours'],
                             'timing_scope': 'Elapsed times are reported final invocations; committed epoch times include fitting only. Earlier failed/restarted invocation overhead is not recoverable from this producer schema.'},
                  limitations=['Saved-feature/head replay and checkpoint checks do not independently reproduce raw CNN image outputs.',
                               'Same-A40 raw-image reload errors are checked as recorded producer receipts; the CPU collector does not re-encode images.',
                               'Provider-case validation is reused and overlapping across trials; no globally fresh confirmation cohort or 94–95 percent result is implied.'])
    assert_public_safe(result)
    receipt = {'status': 'complete case-bag output verification accepted', 'schema_version': 1,
               'producer_code_sha': PRODUCER_SHA, 'producer_archive_sha256': ARCHIVE_SHA256,
               'producer_implementation_sha256': PRODUCER_SHA256, 'config_sha256': CONFIG_SHA256,
               'collector_implementation_sha256': file_hash(__file__),
               'collector_code_sha': collector_sha, 'collector_archive_sha256': collector_archive_sha,
               'checker_execution_device': 'cpu', 'checked_training_environment': common['environment'],
               'verified_fits': len(runs), 'verified_candidate_checkpoints': 80,
               'verified_selected_role_records': sum((len(parts[run['seed'], run['trial']][1]) +
                                                      len(parts[run['seed'], run['trial']][2])) * 15 for run in runs),
               'verified_candidate_validation_records': sum(len(parts[run['seed'], run['trial']][2]) * 60 for run in runs),
               'new_fits': 0, 'cnn_forwards': 0, 'image_decodes': 0, 'outer_predictions': 0,
               'elapsed_seconds': time.monotonic() - started,
               'checks': ['actual git archive identity and extracted tracked bytes', 'all shared Python dependency bytes',
                          'canonical config, trusted original-eight roles, pooled labels and public pretrained bytes',
                          'exact CUDA-build core environment with recorded A40 training identity',
                          'both accepted actual fitting-only smoke checkpoints and optimizer lifecycles',
                          'full 20-fit and 80-candidate artifact matrix before aggregate publication',
                          '50 epochs, 600 batch updates, per-parameter AdamW steps/moments and specialist skip rules',
                          'sampling and phase receipts, warmup backbone, frozen BN/early and adapted/head state fingerprints',
                          'all candidate and selected fitting/validation descriptors, heads, explicit hard decisions and all15 metrics',
                          'earliest maximal validation selection and same-A40 recorded reload tolerance',
                          'registered frozen reference gate and identifier-free whitelist output'],
               'scope': 'CPU checkpoint/saved-statistic verification; no independent raw-image CNN replay'}
    assert_public_safe(receipt)
    output.mkdir(parents=True)
    producer.write_json(output / 'comparison.json', result)
    receipt['comparison_sha256'] = file_hash(output / 'comparison.json')
    producer.write_json(output / 'verification.json', receipt)
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'protocol', 'weights', 'config', 'training-source-root',
                 'training-source-archive', 'smoke-acceptance', 'reference-comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-code-sha', required=True)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    _, receipt = collect(args)
    print(json.dumps({'status': receipt['status'], 'verified_fits': receipt['verified_fits'],
                      'comparison_sha256': receipt['comparison_sha256']}))


if __name__ == '__main__':
    main()

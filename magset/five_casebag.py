"""Bounded five-category case-bag training and deterministic full-field evaluation."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
from PIL import Image
from scipy.special import logsumexp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .adapt_breakhis import cpu_tree
from .baseline import SUBSETS as INDEX_SUBSETS, stratified_inner_split
from .breakhis_predict import load_checked_image
from .capstone_binary import trusted_binary_labels
from .collect import SUBSETS, load_trusted_context
from .collect_adaptation import digest_state
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .features import worker_init
from .five_metrics import FINAL5, MAP8_TO5, five_labels, summarize_records
from .supervised_breakhis import augment, pixels


ARMS = ('direct', 'dual')
SOURCE_FILES = ('five_casebag.py', 'five_metrics.py', 'baseline.py', 'data.py',
                'collect.py', 'capstone_binary.py', 'breakhis_predict.py',
                'supervised_breakhis.py', 'adapt_breakhis.py', 'collect_adaptation.py', 'features.py')


def stable_seed(*values):
    payload = json.dumps(values, separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def case_order(count, seed, epoch):
    if count < 1 or epoch < 1:
        raise ValueError('Expected a positive case count and fitting epoch')
    return np.random.default_rng(stable_seed('case-order', seed, epoch)).permutation(count).tolist()


def sampled_paths(case, seed, epoch, fields_per_view=2):
    """Use independent case/view keys, shared between both model arms."""
    if fields_per_view != 2 or epoch < 1:
        raise ValueError('Expected two fields per view and a positive fitting epoch')
    result = []
    for magnification in MAGNIFICATIONS:
        paths = sorted(case['indices'].get(magnification, []))
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Expected nonempty unique paths in every case/view bag')
        generator = random.Random(stable_seed('field-selection', seed, epoch, case['case_id'], magnification))
        chosen = generator.sample(paths, 2) if len(paths) >= 2 else paths * 2
        result.append(chosen)
    return result


def case_rows(rows, cases):
    """Check an exact role allowlist before any image file can be opened."""
    identities = {case['case_id']: case for case in cases}
    if not cases or len(identities) != len(cases):
        raise ValueError('Expected unique nonempty allowed-role cases')
    paths = [path for case in cases for magnification in MAGNIFICATIONS
             for path in case['indices'].get(magnification, [])]
    expected = set(paths)
    if len(expected) != len(paths):
        raise ValueError('A field occurs more than once in the allowed-role case bags')
    if any(not case['indices'].get(magnification) for case in cases for magnification in MAGNIFICATIONS):
        raise ValueError('Expected all four magnifications in each allowed-role case')
    lookup = {}
    for row in rows:
        identity = case_id(row)
        path = row['relative_path']
        if (identity not in identities or path in lookup or path not in expected
                or CLASS_NAMES.index(row['subtype']) != identities[identity]['label']
                or path not in identities[identity]['indices'][int(row['magnification'])]):
            raise ValueError('Rows differ from the exact allowed-role case bags')
        lookup[path] = row
    if set(lookup) != expected:
        raise ValueError('Allowed-role field coverage differs')
    return lookup


class CaseBags(Dataset):
    def __init__(self, rows, root, cases, config, epoch):
        self.lookup = case_rows(rows, cases)
        self.root = root
        self.cases = cases
        self.config = config
        self.epoch = epoch

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        seed = self.config['training_seed']
        paths = sampled_paths(case, seed, self.epoch, self.config['sampling']['fields_per_view'])
        views = []
        for magnification, selected in zip(MAGNIFICATIONS, paths):
            fields = []
            for slot, path in enumerate(selected):
                image = load_checked_image(self.root, self.lookup[path])
                draw = stable_seed('augmentation', case['case_id'], magnification, slot)
                size = self.config['preprocessing']['resize'][0]
                image = image.resize((size, size), Image.Resampling.BICUBIC)
                image = augment(image, path, self.epoch, draw, seed)
                fields.append(pixels(image, size))
            views.append(torch.stack(fields))
        return index, MAP8_TO5[case['label']], torch.stack(views), paths


def collate_cases(batch):
    indices, labels, fields, paths = zip(*batch)
    return torch.tensor(indices), torch.tensor(labels), torch.stack(fields), list(paths)


def stream_cases(rows, root, cases, config, epoch, workers=0, max_cases=None):
    order = case_order(len(cases), config['training_seed'], epoch)
    if max_cases is not None:
        order = order[:max_cases]
    dataset = CaseBags(rows, root, cases, config, epoch)
    return DataLoader(dataset, batch_size=config['optimization']['case_batch_size'],
                      sampler=order, shuffle=False, drop_last=False, num_workers=workers,
                      collate_fn=collate_cases,
                      worker_init_fn=worker_init,
                      multiprocessing_context='spawn' if workers else None,
                      generator=torch.Generator().manual_seed(stable_seed('loader', config['training_seed'], epoch)))


def fuse_views(view_features, subset):
    """Mean observed views and append flags, ignoring all unobserved entries."""
    if (view_features.ndim != 3 or view_features.shape[1] != 4 or not subset
            or len(set(subset)) != len(subset) or any(index not in range(4) for index in subset)):
        raise ValueError('Expected four view features and a unique nonempty subset')
    mask = view_features.new_zeros((len(view_features), 4))
    mask[:, list(subset)] = 1
    return torch.cat((view_features[:, list(subset)].mean(dim=1), mask), dim=1)


class CaseBagModel(nn.Module):
    def __init__(self, backbone, dimension, arm):
        super().__init__()
        if arm not in ARMS or dimension < 1:
            raise ValueError('Expected direct or dual case-bag model')
        self.backbone = backbone
        self.dimension = dimension
        self.arm = arm
        sizes = {'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}
        self.heads = nn.ModuleDict({name: nn.Linear(dimension + 4, count) for name, count in sizes.items()})

    def head_logits(self, view_features):
        descriptors = torch.stack([fuse_views(view_features, subset) for subset in INDEX_SUBSETS], dim=1)
        return {name: head(descriptors) for name, head in self.heads.items()}

    def forward(self, fields):
        if fields.ndim != 6 or fields.shape[1:3] != (4, 2):
            raise ValueError('Expected case x four views x two fields x channels x height x width')
        batch = len(fields)
        features = self.backbone(fields.flatten(0, 2))
        if features.shape != (batch * 8, self.dimension):
            raise ValueError('Backbone pooled feature shape differs')
        views = features.reshape(batch, 4, 2, self.dimension).mean(dim=2)
        return self.head_logits(views)


def adaptation_names(model):
    if not hasattr(model.backbone, 'blocks') or len(model.backbone.blocks) != 6:
        raise ValueError('Expected the pinned six-group EfficientNetV2B0 layout')
    prefixes = ('backbone.blocks.4.', 'backbone.blocks.5.', 'backbone.conv_head.', 'backbone.bn2.')
    names = {name for name, _ in model.named_parameters() if name.startswith(prefixes)}
    if any(not any(name.startswith(prefix) for name in names) for prefix in prefixes):
        raise ValueError('Expected final two blocks, embedding convolution and BN affine parameters')
    return names


def set_phase(model, adapting):
    selected = adaptation_names(model)
    model.train()
    model.backbone.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('heads.') or (adapting and name in selected))
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return selected


def state_hashes(model, state=None):
    selected = adaptation_names(model)
    state = model.state_dict() if state is None else state
    groups = {
        'bn': {name: value for name, value in state.items()
               if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))},
        'frozen': {name: value for name, value in state.items()
                   if not name.startswith('heads.') and name not in selected},
        'adaptive': {name: value for name, value in state.items() if name in selected},
        'backbone': {name: value for name, value in state.items() if name.startswith('backbone.')},
        'heads': {name: value for name, value in state.items() if name.startswith('heads.')},
    }
    if any(not values for values in groups.values()):
        raise ValueError('Expected nonempty model fingerprint groups')
    return {name: digest_state(values) for name, values in groups.items()}


def case_weights(y5):
    labels = np.asarray(y5)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or set(labels.tolist()) != set(range(5))):
        raise ValueError('Fitting weight calculation requires every final class')
    support = np.bincount(labels, minlength=5)
    groups = np.asarray([support[0], support[1:].sum()])
    malignant = np.r_[0., len(labels) / (4 * support[1:])]
    return {'direct': len(labels) / (5 * support), 'binary': len(labels) / (2 * groups),
            'malignant': malignant}


def case_loss(outputs, y5, weights, configured_batch_size=4):
    """Keep the denominator fixed across full and partial minibatches."""
    if (y5.ndim != 1 or not len(y5) or y5.dtype != torch.long
            or len(y5) > configured_batch_size or torch.any((y5 < 0) | (y5 > 4))):
        raise ValueError('Invalid final-five minibatch targets')
    denominator = configured_batch_size * len(INDEX_SUBSETS)
    tensor_weights = {name: torch.as_tensor(value, dtype=torch.float32, device=y5.device)
                      for name, value in weights.items()}
    if set(outputs) == {'direct'}:
        logits = outputs['direct']
        if logits.shape != (len(y5), 15, 5):
            raise ValueError('Expected fifteen direct-five outputs per case')
        targets = y5[:, None].expand(-1, 15).reshape(-1)
        losses = nn.functional.cross_entropy(logits.reshape(-1, 5), targets, reduction='none').reshape(-1, 15)
        direct = (losses * tensor_weights['direct'][y5, None]).sum() / denominator
        return direct, {'direct': direct}
    if set(outputs) != {'binary', 'malignant'}:
        raise ValueError('Expected direct or binary-plus-malignant outputs')
    binary_logits = outputs['binary']
    malignant_logits = outputs['malignant']
    if binary_logits.shape != (len(y5), 15, 1) or malignant_logits.shape != (len(y5), 15, 4):
        raise ValueError('Expected fifteen binary and malignant-four outputs per case')
    groups = (y5 > 0).long()
    target = groups[:, None].expand(-1, 15).float()
    binary_losses = nn.functional.binary_cross_entropy_with_logits(binary_logits[..., 0], target, reduction='none')
    binary = (binary_losses * tensor_weights['binary'][groups, None]).sum() / denominator
    valid = groups.bool()
    malignant = binary.new_zeros(())
    if valid.any():
        target = (y5[valid] - 1)[:, None].expand(-1, 15).reshape(-1)
        losses = nn.functional.cross_entropy(malignant_logits[valid].reshape(-1, 4), target,
                                             reduction='none').reshape(-1, 15)
        malignant = (losses * tensor_weights['malignant'][y5[valid], None]).sum() / denominator
    return .5 * (binary + malignant), {'binary': binary, 'malignant': malignant}


def learning_rates(config, epoch, smoke=False):
    optimization = config['optimization']
    warmup = 1 if smoke else optimization['warmup_epochs']
    epochs = config['budget']['technical_smoke_epochs'] if smoke else optimization['epochs']
    first = warmup + 1
    progress = 0. if epoch <= first or epochs == first else (epoch - first) / (epochs - first)
    if epoch not in range(1, epochs + 1):
        raise ValueError('Epoch lies outside the fixed budget')
    factor = optimization['cosine_min_factor'] + (1 - optimization['cosine_min_factor']) * .5 * (1 + math.cos(math.pi * progress))
    return optimization['backbone_lr'] * factor, optimization['head_lr'] * factor


def build_model(config, weights, device, arm):
    """Import timm only for an explicitly requested real model construction."""
    import timm
    from safetensors.torch import load_file
    settings = config['model']
    if importlib.metadata.version('timm') != config['dependencies']['timm']:
        raise ValueError('Pinned timm environment differs')
    if file_hash(weights) != settings['weight_sha256'] or Path(weights).stat().st_size != settings['weight_size_bytes']:
        raise ValueError('Pinned pretrained weight bytes differ')
    torch.manual_seed(config['training_seed'])
    backbone = timm.create_model(settings['timm_name'], pretrained=False, num_classes=1000,
                                 drop_rate=0., drop_path_rate=0., global_pool='avg')
    backbone.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    backbone.reset_classifier(0)
    if backbone.num_features != settings['num_features'] or len(backbone.blocks) != 6:
        raise ValueError('Pinned pretrained architecture differs')
    torch.manual_seed(config['training_seed'])
    model = CaseBagModel(backbone, settings['num_features'], arm)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Expected float32 model parameters')
    set_phase(model, False)
    return model.to(device)


def head_arrays(model):
    result = {name + suffix: value.detach().cpu().numpy().copy()
              for name, head in model.heads.items()
              for suffix, value in (('_coef', head.weight), ('_intercept', head.bias))}
    for name in model.heads:
        result[name + '_classes'] = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
    return result


def descriptor_array(view_features, subset):
    features = np.asarray(view_features)
    if (features.ndim != 3 or features.shape[1] != 4 or not subset
            or len(set(subset)) != len(subset) or any(index not in range(4) for index in subset)):
        raise ValueError('Expected a valid observed-view subset')
    mask = np.zeros((len(features), 4), dtype=np.float64)
    mask[:, list(subset)] = 1
    return np.concatenate((features[:, list(subset)].mean(axis=1, dtype=np.float64), mask), axis=1).astype(np.float32)


def scores_from_heads(arrays, descriptor, arm):
    x = np.asarray(descriptor, dtype=np.float64)
    if arm == 'direct':
        logits = x @ arrays['direct_coef'].astype(np.float64).T + arrays['direct_intercept']
        log5 = logits - logsumexp(logits, axis=1, keepdims=True)
        log_malignant = logsumexp(log5[:, 1:], axis=1)
        binary_log = np.column_stack((log5[:, 0], log_malignant))
        malignant_log = log5[:, 1:] - log_malignant[:, None]
        prediction = log5.argmax(axis=1)
    elif arm == 'dual':
        binary_logits = (x @ arrays['binary_coef'].astype(np.float64).T + arrays['binary_intercept'])[:, 0]
        binary_log = np.column_stack((-np.logaddexp(0, binary_logits), -np.logaddexp(0, -binary_logits)))
        logits = x @ arrays['malignant_coef'].astype(np.float64).T + arrays['malignant_intercept']
        malignant_log = logits - logsumexp(logits, axis=1, keepdims=True)
        log5 = np.column_stack((binary_log[:, 0], binary_log[:, 1, None] + malignant_log))
        routing_score = np.clip(np.exp(binary_log[:, 1]), 0, 1)
        prediction = np.where(routing_score >= .5, 1 + malignant_log.argmax(axis=1), 0)
    else:
        raise ValueError('Expected direct or dual scoring arm')
    if not np.isfinite(log5).all() or not np.isfinite(binary_log).all() or not np.isfinite(malignant_log).all():
        raise ValueError('Nonfinite saved-head logits or stable log probabilities')
    return {'prediction': prediction, 'probabilities5': np.exp(log5), 'log_probabilities5': log5,
            'binary_score': np.clip(np.exp(binary_log[:, 1]), 0, 1), 'binary_log_probabilities': binary_log,
            'malignant_probabilities4': np.exp(malignant_log), 'malignant_log_probabilities': malignant_log}


def records_from_features(model, features, cases, role, arrays=None):
    if role not in ('train', 'validation'):
        raise ValueError('Only fitting and validation cases may be scored')
    if features.shape != (len(cases), 4, model.dimension) or not np.isfinite(features).all():
        raise ValueError('Invalid full-field case/view means')
    arrays = head_arrays(model) if arrays is None else arrays
    records = []
    for subset, magnifications in zip(INDEX_SUBSETS, SUBSETS):
        values = scores_from_heads(arrays, descriptor_array(features, subset), model.arm)
        for index, case in enumerate(cases):
            row = {'role': role, 'case_id': case['case_id'], 'truth': int(case['label']),
                   'truth5': MAP8_TO5[case['label']], 'magnifications': list(magnifications)}
            row.update({name: value[index].tolist() for name, value in values.items()})
            records.append(row)
    return records, arrays


def score_cases(model, rows, root, cases, config, device, role):
    if role not in ('train', 'validation'):
        raise ValueError('Only fitting and validation cases may be encoded')
    lookup = case_rows(rows, cases)
    expected = [path for case in cases for magnification in MAGNIFICATIONS
                for path in sorted(case['indices'][magnification])]
    positions = {path: (index, MAGNIFICATIONS.index(magnification))
                 for index, case in enumerate(cases) for magnification in MAGNIFICATIONS
                 for path in case['indices'][magnification]}
    sums = np.zeros((len(cases), 4, model.dimension), dtype=np.float64)
    counts = np.zeros((len(cases), 4), dtype=np.int64)
    batch_size = config['optimization']['inference_image_batch_size']
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(expected), batch_size):
            paths = expected[begin:begin + batch_size]
            size = config['preprocessing']['resize'][0]
            batch = torch.stack([pixels(load_checked_image(root, lookup[path]), size) for path in paths])
            values = model.backbone(batch.to(device)).float().cpu().numpy()
            if values.shape != (len(paths), model.dimension) or not np.isfinite(values).all():
                raise ValueError('Invalid canonical field embeddings')
            for path, value in zip(paths, values):
                index, view = positions[path]
                sums[index, view] += value.astype(np.float64)
                counts[index, view] += 1
    if np.any(counts == 0) or counts.sum() != len(lookup):
        raise ValueError('Incomplete full-field case/view accumulation')
    features = sums / counts[..., None]
    records, arrays = records_from_features(model, features, cases, role)
    archive = {'case_ids': np.asarray([case['case_id'] for case in cases]),
               'truth8': np.asarray([case['label'] for case in cases], dtype=np.int64),
               'truth5': five_labels(np.asarray([case['label'] for case in cases], dtype=np.int64)),
               'view_features': features, 'field_counts': counts,
               'descriptors': np.stack([descriptor_array(features, subset) for subset in INDEX_SUBSETS], axis=1),
               'descriptor_subset_masks': np.asarray([sum(1 << index for index in subset) for subset in INDEX_SUBSETS]),
               'descriptor_magnifications': np.asarray(['+'.join(map(str, subset)) for subset in SUBSETS]),
               **arrays}
    return archive, records, summarize_records(records, model.arm)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_scores(output, prefix, archive, records):
    output = Path(output)
    feature_path = output / (prefix + '-features.npz')
    temporary = feature_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **archive)
    os.replace(temporary, feature_path)
    score_path = output / (prefix + '-predictions.jsonl')
    temporary = score_path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records))
    os.replace(temporary, score_path)
    return {'features_sha256': file_hash(feature_path), 'predictions_sha256': file_hash(score_path)}


def bound_checkpoint(path, context):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('context') != context:
        raise ValueError('Checkpoint source, configuration, roles or runtime context differs')
    if 'model' in saved:
        if (type(saved.get('epoch')) is not int or saved['epoch'] < 1
                or not isinstance(saved['model'], dict) or not saved['model']
                or any(not torch.is_tensor(value) or not torch.isfinite(value).all()
                       for value in saved['model'].values())):
            raise ValueError('Invalid checkpoint epoch or finite model state')
    return saved


def _sampling_receipts(fitting, order, seed, epoch):
    return [{'case_id': fitting[index]['case_id'], 'paths': sampled_paths(fitting[index], seed, epoch),
             'augmentation_draws': [[stable_seed('augmentation', fitting[index]['case_id'], magnification, slot)
                                    for slot in range(2)] for magnification in MAGNIFICATIONS]}
            for index in order]


def _verify_optimizer(saved, model, optimizer, fitting, config, smoke):
    groups = saved['optimizer'].get('param_groups', [])
    states = saved['optimizer'].get('state', {})
    selected = adaptation_names(model)
    parameter_groups = [[(name, parameter) for name, parameter in model.named_parameters() if name in selected],
                        [(name, parameter) for name, parameter in model.named_parameters() if name.startswith('heads.')]]
    expected_ids = []
    rates = learning_rates(config, saved['epoch'], smoke)
    current = optimizer.state_dict()['param_groups']
    optimization = config['optimization']
    warmup = 1 if smoke else optimization['warmup_epochs']
    if len(groups) != len(parameter_groups):
        raise ValueError('Checkpoint optimizer group count differs')
    for index, (group, parameters) in enumerate(zip(groups, parameter_groups)):
        if (set(group) != set(current[index])
                or any(group[key] != current[index][key] for key in group if key not in ('lr', 'params'))
                or group.get('params') != current[index]['params'] or len(group['params']) != len(parameters)
                or group.get('lr') != rates[index] or tuple(group.get('betas', ())) != tuple(optimization['adamw_betas'])
                or group.get('eps') != optimization['adamw_eps'] or group.get('weight_decay') != optimization['weight_decay']
                or any(group.get(key) is not False for key in ('amsgrad', 'maximize', 'capturable', 'differentiable'))):
            raise ValueError('Checkpoint AdamW settings, learning rates or parameter ordering differ')
        for parameter_id, (name, parameter) in zip(group['params'], parameters):
            steps = 0
            for epoch in saved['history']:
                if name in selected:
                    steps += epoch['steps'] if epoch['epoch'] > warmup else 0
                elif name.startswith('heads.malignant.'):
                    order = epoch['case_order']
                    batch_size = optimization['case_batch_size']
                    steps += sum(any(MAP8_TO5[fitting[position]['label']] > 0 for position in order[begin:begin + batch_size])
                                 for begin in range(0, len(order), batch_size))
                else:
                    steps += epoch['steps']
            if steps == 0:
                if parameter_id in states:
                    raise ValueError('Optimizer state exists before a parameter receives supervised updates')
                continue
            expected_ids.append(parameter_id)
            state = states.get(parameter_id, {})
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Checkpoint optimizer moments are incomplete')
            step = state['step']
            if not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all() or step.item() != steps:
                raise ValueError('Checkpoint optimizer per-parameter update count differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = state[key]
                if (not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype
                        or not torch.isfinite(value).all() or (key == 'exp_avg_sq' and torch.any(value < 0))):
                    raise ValueError('Checkpoint optimizer moment shape, dtype or finite values differ')
    if set(states) != set(expected_ids):
        raise ValueError('Checkpoint optimizer contains unexpected parameter states')


def _equal_values(actual, expected, tolerance=1e-10):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError('Saved result keys differ from replay')
        for key in actual:
            _equal_values(actual[key], expected[key], tolerance)
    elif isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or len(actual) != len(expected):
            raise ValueError('Saved result sequence differs from replay')
        for left, right in zip(actual, expected):
            _equal_values(left, right, tolerance)
    elif isinstance(actual, (str, bool)) or actual is None:
        if actual != expected:
            raise ValueError('Saved result identity differs from replay')
    else:
        left, right = np.asarray(actual), np.asarray(expected)
        if left.shape != right.shape:
            raise ValueError('Saved result shape differs from replay')
        if left.dtype.kind in 'USO' or right.dtype.kind in 'USO':
            if not np.array_equal(left, right):
                raise ValueError('Saved result labels differ from replay')
        elif (not np.isfinite(left).all() or not np.isfinite(right).all()
                or not np.allclose(left, right, rtol=0, atol=tolerance)):
            raise ValueError('Saved numeric result differs from replay')


def _verify_candidates(saved, model, validation, config, context, output, smoke):
    expected_epochs = [] if smoke else [epoch for epoch in config['selection']['epochs'] if epoch <= saved['epoch']]
    if [candidate['epoch'] for candidate in saved['candidates']] != expected_epochs:
        raise ValueError('Committed checkpoint candidate coverage differs')
    for candidate in saved['candidates']:
        prefix = 'candidate-epoch' + str(candidate['epoch'])
        checkpoint_path = Path(output) / (prefix + '.pt')
        feature_path = Path(output) / (prefix + '-features.npz')
        score_path = Path(output) / (prefix + '-predictions.jsonl')
        if any(file_hash(path) != candidate[key] for path, key in
               ((checkpoint_path, 'checkpoint_sha256'), (feature_path, 'features_sha256'),
                (score_path, 'predictions_sha256'))):
            raise ValueError('Committed candidate artifact hashes differ')
        checkpoint = bound_checkpoint(checkpoint_path, context)
        if checkpoint['epoch'] != candidate['epoch'] or digest_state(checkpoint['model']) != candidate['state_sha256']:
            raise ValueError('Committed candidate model state differs')
        with np.load(feature_path, allow_pickle=False) as source:
            archive = {name: source[name] for name in source.files}
        if (archive['case_ids'].tolist() != [case['case_id'] for case in validation]
                or not np.array_equal(archive['truth8'], [case['label'] for case in validation])
                or not np.array_equal(archive['truth5'], [MAP8_TO5[case['label']] for case in validation])
                or archive['truth8'].dtype != np.int64 or archive['truth5'].dtype != np.int64
                or archive['view_features'].shape != (len(validation), 4, model.dimension)
                or archive['view_features'].dtype != np.float64 or not np.isfinite(archive['view_features']).all()
                or archive['field_counts'].shape != (len(validation), 4)
                or archive['field_counts'].dtype != np.int64
                or not np.array_equal(archive['field_counts'], [[len(case['indices'][view]) for view in MAGNIFICATIONS] for case in validation])
                or not np.array_equal(archive['descriptor_subset_masks'], [sum(1 << index for index in subset) for subset in INDEX_SUBSETS])
                or archive['descriptor_magnifications'].tolist() != ['+'.join(map(str, subset)) for subset in SUBSETS]):
            raise ValueError('Committed candidate sufficient statistics or subset axis differ')
        descriptors = np.stack([descriptor_array(archive['view_features'], subset) for subset in INDEX_SUBSETS], axis=1)
        if archive['descriptors'].dtype != np.float32 or not np.array_equal(descriptors, archive['descriptors']):
            raise ValueError('Committed candidate descriptors differ from saved view means')
        arrays = {}
        for name in model.heads:
            for suffix, field in (('_coef', 'weight'), ('_intercept', 'bias')):
                value = checkpoint['model']['heads.' + name + '.' + field].numpy()
                if (value.dtype != np.float32 or archive[name + suffix].dtype != np.float32
                        or not np.array_equal(archive[name + suffix], value)):
                    raise ValueError('Candidate head arrays differ from the bound model checkpoint')
                arrays[name + suffix] = value
            classes = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
            if not np.issubdtype(archive[name + '_classes'].dtype, np.integer) or not np.array_equal(archive[name + '_classes'], classes):
                raise ValueError('Committed candidate head class order differs')
        records = [json.loads(line) for line in score_path.read_text().splitlines()]
        replay, _ = records_from_features(model, archive['view_features'], validation, 'validation', arrays)
        _equal_values(replay, records)
        _equal_values(summarize_records(records, model.arm), candidate['validation'])
    if expected_epochs:
        best = max(saved['candidates'], key=lambda value: value['validation']['mean_subset_macro_f1'])
        if saved['best'] != best or digest_state(saved['best_model']) != best['state_sha256']:
            raise ValueError('Committed earliest-best checkpoint snapshot differs')
    elif saved['best'] is not None or saved['best_model'] is not None:
        raise ValueError('A best checkpoint exists before any selection opportunity')


def verify_training_checkpoint(saved, model, optimizer, fitting, validation, config, context, output, smoke):
    required = {'epoch', 'model', 'optimizer', 'initial', 'history', 'candidates', 'best', 'best_model', 'rng', 'phase'}
    if not required.issubset(saved):
        raise ValueError('Authoritative checkpoint schema is incomplete')
    epochs = config['budget']['technical_smoke_epochs'] if smoke else config['optimization']['epochs']
    warmup = 1 if smoke else config['optimization']['warmup_epochs']
    if saved['epoch'] not in range(1, epochs + 1) or [row['epoch'] for row in saved['history']] != list(range(1, saved['epoch'] + 1)):
        raise ValueError('Checkpoint completed epoch history differs')
    batch_size = config['optimization']['case_batch_size']
    for row in saved['history']:
        order = case_order(len(fitting), config['training_seed'], row['epoch'])
        if smoke:
            order = order[:batch_size * config['budget']['technical_smoke_batches_per_epoch']]
        receipts = _sampling_receipts(fitting, order, config['training_seed'], row['epoch'])
        receipt_hash = hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (row['case_order'] != order or row['cases'] != len(order) or row['steps'] != math.ceil(len(order) / batch_size)
                or row['sampling_receipts'] != receipts or row['field_draw_sha256'] != receipt_hash
                or row['backbone_adapting'] != (row['epoch'] > warmup)
                or tuple(row['learning_rates']) != learning_rates(config, row['epoch'], smoke)
                or not math.isfinite(row['loss']) or any(not math.isfinite(value) for value in row['loss_components'].values())
                or any(not math.isfinite(value) for value in row['maximum_group_gradient_norms'].values())
                or row['maximum_group_gradient_norms']['heads'] <= 0
                or (row['epoch'] > warmup and row['maximum_group_gradient_norms']['adaptive'] <= 0)
                or row['fingerprints']['bn'] != saved['initial']['bn']
                or row['fingerprints']['frozen'] != saved['initial']['frozen']):
            raise ValueError('Checkpoint epoch sampling, phase, loss or frozen-state receipt differs')
    phase = 'adaptation' if saved['epoch'] > warmup else 'head_warmup'
    if saved.get('phase') != phase:
        raise ValueError('Checkpoint phase differs from the completed epoch')
    reloads = saved.get('smoke_reloads', [])
    if ((not smoke and reloads) or (smoke and saved['epoch'] == 1 and reloads not in ([], [1]))
            or (smoke and saved['epoch'] > 1 and reloads != [1])):
        raise ValueError('Checkpoint technical reload receipt differs')
    if state_hashes(model, saved['model']) != saved['history'][-1]['fingerprints']:
        raise ValueError('Checkpoint final model differs from its epoch fingerprints')
    _verify_optimizer(saved, model, optimizer, fitting, config, smoke)
    _verify_candidates(saved, model, validation, config, context, output, smoke)


def model_receipt(model):
    selected = adaptation_names(model)
    groups = {
        'adaptive': [(name, parameter) for name, parameter in model.named_parameters() if name in selected],
        'heads': [(name, parameter) for name, parameter in model.named_parameters() if name.startswith('heads.')],
    }
    return {'total_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'backbone_parameters': sum(parameter.numel() for parameter in model.backbone.parameters()),
            'optimizer_groups': {name: {'parameter_names': [key for key, _ in values],
                                        'parameters': sum(parameter.numel() for _, parameter in values)}
                                 for name, values in groups.items()},
            'head_schema': {name: {'weight_shape': list(head.weight.shape),
                                   'bias_shape': list(head.bias.shape), 'dtype': str(head.weight.dtype)}
                            for name, head in model.heads.items()}}


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'python': random.getstate(), 'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                                                  *numpy_state[2:])}


def _restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))


def _candidate(output, epoch, model, val_rows, validation, root, config, context, device):
    archive, records, metric = score_cases(model, val_rows, root, validation, config, device, 'validation')
    prefix = 'candidate-epoch' + str(epoch)
    hashes = save_scores(output, prefix, archive, records)
    state = cpu_tree(model.state_dict())
    path = Path(output) / (prefix + '.pt')
    atomic_save({'context': context, 'epoch': epoch, 'model': state}, path)
    candidate = {'epoch': epoch, 'validation': metric, 'state_sha256': digest_state(state),
                 'checkpoint_sha256': file_hash(path), **hashes}
    return candidate, state


def train(model, fit_rows, val_rows, fitting, validation, root, config, context,
          output, device, workers=0, smoke=False, resume=False):
    output = Path(output)
    optimization = config['optimization']
    epochs = config['budget']['technical_smoke_epochs'] if smoke else optimization['epochs']
    warmup = 1 if smoke else optimization['warmup_epochs']
    initial = state_hashes(model)
    selected_names = adaptation_names(model)
    groups = [{'params': [parameter for name, parameter in model.named_parameters() if name in selected_names]},
              {'params': list(model.heads.parameters())}]
    optimizer = torch.optim.AdamW(groups, lr=optimization['head_lr'], weight_decay=optimization['weight_decay'],
                                  betas=tuple(optimization['adamw_betas']), eps=optimization['adamw_eps'])
    weights = case_weights(five_labels(np.asarray([case['label'] for case in fitting])))
    history = []
    candidates = []
    best = None
    best_state = None
    first = 1
    smoke_reloads = []
    invocation_training_fields = 0
    invocation_validation_fields = 0
    last_path = output / 'last.pt'
    if resume:
        if not last_path.exists():
            raise ValueError('Resume requires an authoritative epoch checkpoint')
        saved = bound_checkpoint(last_path, context)
        if saved['initial'] != initial or saved['epoch'] not in range(1, epochs + 1):
            raise ValueError('Resume initialization or fixed epoch budget differs')
        verify_training_checkpoint(saved, model, optimizer, fitting, validation, config, context, output, smoke)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']
        candidates = saved['candidates']
        best = saved['best']
        best_state = saved['best_model']
        smoke_reloads = saved.get('smoke_reloads', [])
        first = saved['epoch'] + 1
        if [row['epoch'] for row in history] != list(range(1, first)):
            raise ValueError('Resume epoch history is incomplete')
        if best is not None and digest_state(best_state) != best['state_sha256']:
            raise ValueError('Resume best-model snapshot differs')
        _restore_rng(saved['rng'], device)
        if smoke and saved['epoch'] == 1 and not smoke_reloads:
            # This resume performed the epoch-one reload even if the earlier
            # process stopped between the authoritative commit and its receipt.
            smoke_reloads = [1]
            saved['smoke_reloads'] = smoke_reloads
            atomic_save(saved, last_path)
    for epoch in range(first, epochs + 1):
        started = time.monotonic()
        adapting = epoch > warmup
        set_phase(model, adapting)
        rates = learning_rates(config, epoch, smoke)
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
        batch_size = optimization['case_batch_size']
        limit = batch_size * config['budget']['technical_smoke_batches_per_epoch'] if smoke else None
        loader = stream_cases(fit_rows, root, fitting, config, epoch, workers, limit)
        loss_sum = 0.
        component_sums = Counter()
        indices_seen = []
        paths_seen = []
        gradient_names = set()
        gradient_norms = {'heads': 0., 'adaptive': 0.}
        steps = 0
        for indices, labels, fields, paths in loader:
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss, components = case_loss(model(fields.to(device)), labels, weights, batch_size)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite case-supervised fitting loss')
            loss.backward()
            gradient_names.update(name for name, parameter in model.named_parameters() if parameter.grad is not None)
            for group, predicate in (('heads', lambda name: name.startswith('heads.')),
                                     ('adaptive', lambda name: name in selected_names)):
                gradients = [parameter.grad.detach().norm() for name, parameter in model.named_parameters()
                             if predicate(name) and parameter.grad is not None]
                if gradients:
                    gradient_norms[group] = max(gradient_norms[group], float(torch.stack(gradients).norm()))
            nn.utils.clip_grad_norm_(model.parameters(), optimization['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            invocation_training_fields += len(labels) * 8
            loss_sum += float(loss.detach()) * batch_size
            for name, value in components.items():
                component_sums[name] += float(value.detach()) * batch_size
            indices_seen.extend(indices.tolist())
            paths_seen.extend(paths)
            steps += 1
        expected_order = case_order(len(fitting), config['training_seed'], epoch)
        if limit is not None:
            expected_order = expected_order[:limit]
        if indices_seen != expected_order or steps != math.ceil(len(expected_order) / batch_size):
            raise ValueError('Registered once-per-case epoch or minibatch budget differs')
        receipts = _sampling_receipts(fitting, expected_order, config['training_seed'], epoch)
        if paths_seen != [receipt['paths'] for receipt in receipts]:
            raise ValueError('Actual sampled fields differ from the stateless paired receipt')
        fingerprints = state_hashes(model)
        if fingerprints['bn'] != initial['bn'] or fingerprints['frozen'] != initial['frozen']:
            raise ValueError('Frozen backbone or BatchNorm running state changed')
        if not adapting and fingerprints['backbone'] != initial['backbone']:
            raise ValueError('Backbone changed during head warmup')
        allowed = selected_names | {name for name, _ in model.named_parameters() if name.startswith('heads.')}
        if gradient_names - allowed or (not adapting and gradient_names & selected_names):
            raise ValueError('Gradients crossed the declared phase boundary')
        if adapting and not gradient_names.intersection(selected_names):
            raise ValueError('Adaptive backbone received no fitting gradient')
        if gradient_norms['heads'] <= 0 or (adapting and gradient_norms['adaptive'] <= 0):
            raise ValueError('Expected nonzero fitting gradients in the active parameter groups')
        row = {'epoch': epoch, 'loss': loss_sum / len(expected_order),
               'loss_components': {name: total / len(expected_order) for name, total in component_sums.items()},
               'cases': len(expected_order), 'steps': steps, 'case_order': indices_seen,
               'field_draw_sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'sampling_receipts': receipts, 'backbone_adapting': adapting,
               'learning_rates': rates, 'gradient_parameter_names': sorted(gradient_names),
               'maximum_group_gradient_norms': gradient_norms,
               'fingerprints': fingerprints, 'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        if not smoke and epoch in config['selection']['epochs']:
            candidate, state = _candidate(output, epoch, model, val_rows, validation, root, config, context, device)
            invocation_validation_fields += len(val_rows)
            candidates.append(candidate)
            if best is None or candidate['validation']['mean_subset_macro_f1'] > best['validation']['mean_subset_macro_f1']:
                best = candidate
                best_state = state
        record = {'context': context, 'epoch': epoch, 'model': cpu_tree(model.state_dict()),
                  'phase': 'adaptation' if adapting else 'head_warmup',
                  'optimizer': cpu_tree(optimizer.state_dict()), 'initial': initial,
                  'history': history, 'candidates': candidates, 'best': best, 'best_model': best_state,
                  'smoke_reloads': smoke_reloads, 'rng': _rng_state(device)}
        atomic_save(record, last_path)
        if smoke and epoch == 1:
            reloaded = bound_checkpoint(last_path, context)
            model.load_state_dict(reloaded['model'], strict=True)
            optimizer.load_state_dict(reloaded['optimizer'])
            _restore_rng(reloaded['rng'], device)
            smoke_reloads.append(epoch)
            record['smoke_reloads'] = smoke_reloads
            atomic_save(record, last_path)
        if best is not None:
            atomic_save({'context': context, 'epoch': best['epoch'], 'model': best_state}, output / 'selected.pt')
        print(json.dumps({'epoch': epoch, 'cases': row['cases'], 'loss': row['loss'],
                          'backbone_adapting': adapting}), flush=True)
    final = bound_checkpoint(last_path, context)
    if final['epoch'] != epochs or [row['epoch'] for row in history] != list(range(1, epochs + 1)):
        raise ValueError('Fixed training budget is incomplete')
    verify_training_checkpoint(final, model, optimizer, fitting, validation, config, context, output, smoke)
    hashes = state_hashes(model)
    if (hashes['bn'] != initial['bn'] or hashes['frozen'] != initial['frozen']
            or hashes['adaptive'] == initial['adaptive'] or hashes['heads'] == initial['heads']):
        raise ValueError('Final frozen/adapted/head parameter proof failed')
    if not smoke:
        if [candidate['epoch'] for candidate in candidates] != config['selection']['epochs']:
            raise ValueError('Fixed checkpoint selection coverage differs')
        if best != max(candidates, key=lambda candidate: candidate['validation']['mean_subset_macro_f1']):
            raise ValueError('Earliest-maximum checkpoint selection differs')
        atomic_save({'context': context, 'epoch': best['epoch'], 'model': best_state}, output / 'selected.pt')
    return {'initial': initial, 'final': hashes, 'history': history, 'candidates': candidates,
            'best': best, 'smoke_reloads': smoke_reloads, 'model_receipt': model_receipt(model),
            'completed_epoch_training_field_forwards': sum(row['cases'] * 8 for row in history),
            'invocation_training_field_forwards': invocation_training_fields,
            'invocation_candidate_validation_field_forwards': invocation_validation_fields,
            'last_checkpoint_sha256': file_hash(last_path)}


def runtime_environment(config, device):
    versions = {name: importlib.metadata.version(name) for name in config['dependencies']}
    if versions != config['dependencies']:
        raise ValueError('Pinned case-bag numerical environment differs')
    gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    matched = re.search(r'(H100|H200|A100|A40|L40S?|V100)', gpu, re.IGNORECASE)
    return {'dependencies': versions, 'python': platform.python_version(),
            'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu_family': matched.group(1).upper() if matched else gpu,
            'device_type': device.type,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def validate_config(config):
    """Reject scientific settings that this bounded implementation does not use."""
    expected = {
        'model': {'num_features': 1280, 'descriptor_dimension': 1284, 'head_dropout': 0.,
                  'encoder_drop_rate': 0., 'encoder_drop_path_rate': 0., 'global_pool': 'avg',
                  'timm_name': 'tf_efficientnetv2_b0.in1k',
                  'hf_repository': 'timm/tf_efficientnetv2_b0.in1k',
                  'hf_revision': '30b1df5883752d711e047538d9b01db797782818',
                  'weight_sha256': 'e134f5fc07f28cff44a0601a13d79d2eff6304fa7ed7d64332984e227c2785b3',
                  'weight_size_bytes': 28840352},
        'sampling': {'fields_per_view': 2, 'case_visits_per_epoch': 1,
                     'without_replacement_when_possible': True},
        'optimization': {'epochs': 50, 'warmup_epochs': 5, 'case_batch_size': 4,
                         'inference_image_batch_size': 64, 'head_lr': .001, 'backbone_lr': .00001,
                         'weight_decay': .0001, 'gradient_clip_norm': 1., 'adamw_betas': [.9, .999],
                         'adamw_eps': 1e-8, 'adamw_amsgrad': False, 'cosine_min_factor': .1,
                         'cosine_start_epoch': 6, 'cosine_last_epoch': 50,
                         'loss_denominator_cases': 4, 'drop_last': False,
                         'zero_grad_set_to_none': True, 'mixed_precision': False, 'tf32': False},
        'selective_adaptation': {'expected_blocks_groups': 6, 'blocks': [4, 5],
                                 'modules': ['blocks.4', 'blocks.5', 'conv_head', 'bn2'],
                                 'train_selected_bn_affine': True, 'freeze_early_bn_affine': True,
                                 'freeze_all_bn_running_buffers': True},
        'selection': {'epochs': [10, 20, 35, 50], 'early_stopping': False,
                      'additional_threshold_fit': False},
        'loss': {'dual_binary_coefficient': .5, 'dual_expert_coefficient': .5,
                 'expert_benign_weight': 0., 'label_smoothing': 0.,
                 'field_auxiliary_loss': False, 'class_resampling': False},
        'preprocessing': {'color_mode': 'RGB', 'resize': [224, 224],
                          'interpolation': 'Pillow.Resampling.BICUBIC', 'crop': 'none',
                          'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                          'augmentation_order': ['resize', 'horizontal_flip', 'vertical_flip', 'quarter_turn',
                                                 'brightness', 'contrast', 'saturation', 'tensor_and_normalize']},
        'inference': {'routing_threshold': .5, 'routing_tie': 'malignant',
                      'selected_reload_tolerance': 1e-6},
        'budget': {'production_fits': 20, 'technical_smoke_fits': 2,
                   'technical_smoke_epochs': 2, 'technical_smoke_batches_per_epoch': 2,
                   'technical_smoke_warmup_epochs': 1,
                   'technical_smoke_trial': 1, 'technical_smoke_seed': 812},
        'environment_policy': {'gpu_family': 'A40'},
    }
    for section, settings in expected.items():
        if any(config.get(section, {}).get(name) != value for name, value in settings.items()):
            raise ValueError('Case-bag configuration differs from implemented ' + section + ' settings')
    if (tuple(config.get('final_labels', ())) != FINAL5 or tuple(config.get('map8_to5', ())) != MAP8_TO5
            or config.get('training_seed') != 812 or config.get('seeds') != [812, 813]
            or config.get('trials') != list(range(1, 6)) or config.get('arms') != list(ARMS)
            or config.get('magnifications') != list(MAGNIFICATIONS)
            or sorted(config.get('subset_masks', [])) != list(range(1, 16))):
        raise ValueError('Case-bag labels, roles, views or mask coverage differ')
    weights = case_weights(np.repeat(np.arange(5), [11, 25, 2, 5, 3]))
    for name, values in (('direct_weights', weights['direct']), ('binary_weights', weights['binary']),
                         ('expert_weights', weights['malignant'][1:])):
        if not np.allclose(config['loss'].get(name, []), values, rtol=0, atol=1e-14):
            raise ValueError('Configured global head weights differ')


def execution_gate(action, config, device):
    if action == 'run' and config.get('production_ready') is not True:
        raise ValueError('Production requires a frozen production-ready registration')
    if action in ('run', 'smoke') and device.type != 'cuda':
        raise ValueError('Real case-bag fitting requires a scheduler-allocated CUDA device')
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != config['optimization']['CUBLAS_WORKSPACE_CONFIG']:
        raise ValueError('CUBLAS workspace differs from the deterministic runtime contract')


def accepted_smokes(path, context):
    receipt = json.loads(Path(path).read_text())
    entries = receipt.get('reports', [])
    if (receipt.get('status') != 'accepted case-bag technical smokes'
            or len(entries) != 2 or {entry['arm'] for entry in entries} != set(ARMS)):
        raise ValueError('Production requires both accepted technical smoke reports')
    for entry in entries:
        report_path = Path(entry['path'])
        if file_hash(report_path) != entry['sha256']:
            raise ValueError('Accepted technical smoke report bytes differ')
        report = json.loads(report_path.read_text())
        smoke_context = report['context']
        if (report['status'] != 'fitting-only case-bag smoke' or smoke_context['arm'] != entry['arm']
                or smoke_context.get('smoke') is not True
                or smoke_context.get('seed') != 812 or smoke_context.get('trial') != 1
                or any(smoke_context[key] != context[key] for key in
                       ('code_sha', 'config_sha256', 'weights_sha256', 'manifest_sha256',
                        'protocol_sha256', 'source_archive_sha256', 'membership_fingerprint_sha256',
                        'environment', 'implementation_sha256'))
                or report.get('validation_predictions') != 0 or report.get('outer_predictions') != 0):
            raise ValueError('Technical smoke source, runtime or fitting-only scope differs')
        proof = report['proof']
        if (len(proof['history']) != 2 or [row['epoch'] for row in proof['history']] != [1, 2]
                or [row['steps'] for row in proof['history']] != [2, 2]
                or [row['cases'] for row in proof['history']] != [8, 8]
                or [row['backbone_adapting'] for row in proof['history']] != [False, True]
                or proof['candidates'] or proof['smoke_reloads'] != [1]
                or proof['initial']['bn'] != proof['final']['bn']
                or proof['initial']['frozen'] != proof['final']['frozen']
                or proof['initial']['adaptive'] == proof['final']['adaptive']
                or proof['initial']['heads'] == proof['final']['heads']
                or file_hash(report_path.parent / 'last.pt') != proof['last_checkpoint_sha256']):
            raise ValueError('Accepted smoke lifecycle or parameter proof differs')
        checkpoint = bound_checkpoint(report_path.parent / 'last.pt', smoke_context)
        _equal_values(checkpoint['history'], proof['history'])
        _equal_values(checkpoint['initial'], proof['initial'])
        if checkpoint['epoch'] != 2 or checkpoint['smoke_reloads'] != [1] or set(proof['reload']) != {'feature_max_abs', 'probability_max_abs'}:
            raise ValueError('Accepted smoke checkpoint or reload schema differs')
        for value in proof['reload'].values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1e-6:
                raise ValueError('Accepted smoke reload exceeds the fixed finite tolerance')
    return file_hash(path)


def prepare_context(manifest, protocol, config_path, weights, code_sha, trial, seed, arm, device, smoke=False):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    source = Path(__file__).resolve().parents[1]
    if (not re.fullmatch(r'[0-9a-f]{40}', code_sha)
            or (source / '.source-sha').read_text().strip() != code_sha
            or Path(config_path).read_bytes() != (source / 'configs' / Path(config_path).name).read_bytes()):
        raise ValueError('Use an immutable source snapshot with matching configuration bytes')
    archive_hash = (source / '.archive-sha256').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', archive_hash):
        raise ValueError('Immutable source archive receipt missing')
    if arm not in ARMS or seed not in config['seeds'] or trial not in config['trials']:
        raise ValueError('Unexpected registered arm, arrangement or trial')
    if smoke and (seed, trial) != (config['budget']['technical_smoke_seed'], config['budget']['technical_smoke_trial']):
        raise ValueError('Technical smokes require their fixed arrangement and provider trial')
    if tuple(config['map8_to5']) != MAP8_TO5:
        raise ValueError('Canonical final-label mapping differs')
    trusted = load_trusted_context(manifest, protocol)
    if any(trusted[name] != config[name] for name in ('manifest_sha256', 'protocol_sha256')):
        raise ValueError('Registered manifest or protocol differs')
    rows = read_manifest(manifest)
    broad = trusted_binary_labels(rows)
    cases = trusted['cases']
    memberships = []
    for arrangement in config['seeds']:
        for provider_trial in config['trials']:
            parts = stratified_inner_split(cases, trusted['trials'][str(provider_trial)]['case_roles'], arrangement)
            membership = {'seed': arrangement, 'trial': provider_trial,
                          'roles': {name: sorted(cases[index]['case_id'] for index in part)
                                    for name, part in zip(('train', 'validation', 'test'), parts)}}
            memberships.append(membership)
    fingerprint = hashlib.sha256(json.dumps(memberships, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != config['membership_fingerprint_sha256']:
        raise ValueError('Accepted original-eight membership fingerprint differs')
    indices = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
    fitting, validation, excluded = [sorted([cases[index] for index in part], key=lambda case: case['case_id'])
                                     for part in indices]
    if [len(part) for part in indices] != [46, 8, 28]:
        raise ValueError('Exact original-eight role counts differ')
    supports = [np.bincount(five_labels(np.asarray([case['label'] for case in part])), minlength=5).tolist()
                for part in (fitting, validation, excluded)]
    if supports[:2] != [[11, 25, 2, 5, 3], [4, 1, 1, 1, 1]]:
        raise ValueError('Exact five-category fitting or validation support differs')
    if any(broad[case['case_id']] != int(MAP8_TO5[case['label']] > 0) for case in cases):
        raise ValueError('Broad labels differ from final-five mapping')
    roles = {name: [case['case_id'] for case in part]
             for name, part in (('train', fitting), ('validation', validation), ('excluded', excluded))}
    if len(set.union(*(set(values) for values in roles.values()))) != 82:
        raise ValueError('Role identities overlap or coverage differs')
    fit_rows = [row for row in rows if case_id(row) in set(roles['train'])]
    val_rows = [row for row in rows if case_id(row) in set(roles['validation'])]
    case_rows(fit_rows, fitting)
    case_rows(val_rows, validation)
    if file_hash(weights) != config['model']['weight_sha256']:
        raise ValueError('Pinned pretrained weights differ')
    environment = runtime_environment(config, device)
    if device.type == 'cuda' and environment['gpu_family'] != config['environment_policy']['gpu_family']:
        raise ValueError('Allocated GPU family differs from the fixed A40 contract')
    context = {'code_sha': code_sha, 'source_archive_sha256': archive_hash,
               'implementation_sha256': {name: file_hash(source / 'magset' / name) for name in SOURCE_FILES},
               'config': config, 'config_sha256': file_hash(config_path),
               'manifest_sha256': trusted['manifest_sha256'], 'protocol_sha256': trusted['protocol_sha256'],
               'weights_sha256': file_hash(weights), 'environment': environment,
               'membership_fingerprint_sha256': fingerprint,
               'arm': arm, 'seed': seed, 'trial': trial, 'smoke': smoke, 'case_ids': roles,
               'role_sha256': hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest()}
    return config, context, fitting, validation, fit_rows, val_rows


def maximum_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not left.size or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid finite replay arrays or shapes')
    return float(np.max(np.abs(left - right)))


def probe_reload(model, restored, rows, root, config, device):
    probes = [next(row for row in rows if int(row['magnification']) == magnification) for magnification in MAGNIFICATIONS]
    size = config['preprocessing']['resize'][0]
    batch = torch.stack([pixels(load_checked_image(root, row), size) for row in probes]).to(device)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        before = model.backbone(batch).float().cpu().numpy()
        after = restored.backbone(batch).float().cpu().numpy()
    feature_error = maximum_difference(before, after)
    left, _ = records_from_features(model, before[None].astype(np.float64),
                                     [{'case_id': 'fitting-probe', 'label': 0}], 'train')
    right, _ = records_from_features(restored, after[None].astype(np.float64),
                                      [{'case_id': 'fitting-probe', 'label': 0}], 'train')
    probability_error = maximum_difference([row['probabilities5'] for row in left], [row['probabilities5'] for row in right])
    if (max(feature_error, probability_error) > 1e-6
            or [row['prediction'] for row in left] != [row['prediction'] for row in right]):
        raise ValueError('Same-runtime selected model reload exceeds the fixed tolerance')
    return {'feature_max_abs': feature_error, 'probability_max_abs': probability_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'run', 'smoke'))
    for name in ('manifest', 'protocol', 'images-root', 'weights', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-sha', required=True)
    parser.add_argument('--trial', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--seed', type=int, choices=(812, 813), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-acceptance', type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    declared = json.loads(args.config.read_text())
    validate_config(declared)
    execution_gate(args.action, declared, device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Allocated CUDA runtime unavailable')
    config, context, fitting, validation, fit_rows, val_rows = prepare_context(
        args.manifest, args.protocol, args.config, args.weights, args.code_sha,
        args.trial, args.seed, args.arm, device, args.action == 'smoke')
    if args.action == 'run':
        if args.smoke_acceptance is None:
            raise ValueError('Production requires the accepted two-arm smoke receipt')
        context['smoke_acceptance_sha256'] = accepted_smokes(args.smoke_acceptance, context)
    if args.output.exists():
        if (not args.resume or (args.output / 'metrics.json').exists()
                or json.loads((args.output / 'started.json').read_text()) != context):
            raise ValueError('Output exists or resume context differs')
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / 'started.json', context)
    if args.action == 'audit':
        write_json(args.output / 'audit.json', {'status': 'zero-fit case-bag input audit', 'context': context,
                                               'new_fits': 0, 'new_forwards': 0, 'outer_predictions': 0})
        return
    torch.manual_seed(config['training_seed'])
    np.random.seed(config['training_seed'])
    random.seed(config['training_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, args.weights, device, args.arm)
    proof = train(model, fit_rows, val_rows, fitting, validation, args.images_root, config, context,
                  args.output, device, args.workers, args.action == 'smoke', args.resume)
    chosen = args.output / ('last.pt' if args.action == 'smoke' else 'selected.pt')
    checkpoint = bound_checkpoint(chosen, context)
    model.load_state_dict(checkpoint['model'], strict=True)
    restored = build_model(config, args.weights, device, args.arm)
    restored.load_state_dict(checkpoint['model'], strict=True)
    proof['reload'] = probe_reload(model, restored, fit_rows, args.images_root, config, device)
    report = {'status': 'fitting-only case-bag smoke' if args.action == 'smoke' else 'complete case-bag validation run',
              'context': context, 'proof': proof, 'outer_predictions': 0,
              'elapsed_seconds': time.monotonic() - started, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    if args.action != 'smoke':
        for role, rows, cases in (('train', fit_rows, fitting), ('validation', val_rows, validation)):
            archive, records, metric = score_cases(restored, rows, args.images_root, cases, config, device, role)
            report[role] = metric
            report[role + '_artifacts'] = save_scores(args.output, 'selected-' + role, archive, records)
        best = proof['best']
        with np.load(args.output / ('candidate-epoch%d-features.npz' % best['epoch']), allow_pickle=False) as saved:
            with np.load(args.output / 'selected-validation-features.npz', allow_pickle=False) as replay:
                difference = maximum_difference(saved['view_features'], replay['view_features'])
        if difference > 1e-6 or abs(report['validation']['mean_subset_macro_f1'] - best['validation']['mean_subset_macro_f1']) > 1e-12:
            raise ValueError('Selected full-field validation replay differs')
        candidate_records = [json.loads(line) for line in
                             (args.output / ('candidate-epoch%d-predictions.jsonl' % best['epoch'])).read_text().splitlines()]
        selected_records = [json.loads(line) for line in
                            (args.output / 'selected-validation-predictions.jsonl').read_text().splitlines()]
        if [(row['case_id'], row['magnifications'], row['prediction']) for row in candidate_records] != [
                (row['case_id'], row['magnifications'], row['prediction']) for row in selected_records]:
            raise ValueError('Selected full-field validation decisions differ')
        probability_error = maximum_difference([row['probabilities5'] for row in candidate_records],
                                               [row['probabilities5'] for row in selected_records])
        if probability_error > 1e-6:
            raise ValueError('Selected full-field validation probabilities differ')
        report['selected_feature_replay_max_abs'] = difference
        report['selected_probability_replay_max_abs'] = probability_error
    else:
        report['validation_predictions'] = 0
    report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    report['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    report['invocation_field_forwards'] = {
        'training': proof['invocation_training_field_forwards'],
        'candidate_validation': proof['invocation_candidate_validation_field_forwards'],
        'selected_fitting': len(fit_rows) if args.action == 'run' else 0,
        'selected_validation': len(val_rows) if args.action == 'run' else 0,
        'reload_probes': 8,
    }
    report['elapsed_seconds'] = time.monotonic() - started
    write_json(args.output / 'metrics.json', report)


if __name__ == '__main__':
    main()

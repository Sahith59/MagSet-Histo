"""One registered Midnight LoRA recipe, fixed-C1 initialization and four checkpoints."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from . import midnight_five as base
from . import lora_io as io
from .bags import fit_normalization
from .breakhis_predict import load_checked_image
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest
from .field_five import field_log_probabilities
from .five_class import load_model
from .five_collect import read_json, require_equal, verify_audit
from .five_metrics import FINAL5, summarize_records

ARMS = ('direct', 'dual')
CANDIDATES = [2, 6, 12, 20]


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]; canonical = root/'configs/breakhis-midnight-lora-v1.json'
    if file_hash(args.config) != file_hash(canonical): raise ValueError('LoRA config differs from immutable registration')
    own = read_json(args.config)
    fixed = {'name': 'breakhis-midnight-lora-v1', 'epochs': 20, 'warmup_epochs': 2, 'candidate_epochs': CANDIDATES,
             'fields_per_view': 2, 'case_batch_size': 4, 'microbatch_cases': 1, 'last_batch_denominator': 4,
             'initial_candidate_index': 2, 'initial_C': 1., 'initial_readout': 'field', 'production_runs': 20}
    if any(own.get(k) != v for k, v in fixed.items()): raise ValueError('Registered LoRA recipe differs')
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']: raise ValueError('Accepted Midnight input registration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path
    config, context, trusted, cases, bags, z, y5, roles = base.load_inputs(forwarded)
    context = dict(context, lora_config_sha256=file_hash(args.config))
    names = tuple(dict.fromkeys(base.DEPENDENCIES+('midnight_lora.py', 'lora_io.py', 'lora_five.py', 'lora_collect.py', 'breakhis_predict.py')))
    context['implementation_hashes'] = {name: file_hash(root/'magset'/name) for name in names}
    return {**config, **own}, context, trusted, cases, bags, z, y5, roles


def initial_model_path(initial_runs, seed, trial, arm):
    return Path(initial_runs)/'field'/('split%d'%seed)/('trial%d'%trial)/arm/'candidate2/model.npz'


def initial_bindings(config, cases, bags, roles, initial_runs):
    bindings = []
    for (seed, trial), (train, _, _) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        for arm in ARMS:
            path = initial_model_path(initial_runs, seed, trial, arm); digest = file_hash(path)
            if digest != config['initial_model_sha256']['%d/%d/%s'%(seed, trial, arm)]: raise ValueError('Fixed-C1 initial checkpoint hash differs')
            arrays = load_model(path, arm, dimension=config['encoder_dimension'])
            if not np.array_equal(arrays['mean'], mean) or not np.array_equal(arrays['std'], std): raise ValueError('Initial normalizer is not exact fitting-only moments')
            bindings.append({'seed': seed, 'trial': trial, 'arm': arm, 'model_sha256': digest,
                             'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    return bindings


def verify_input_audit(audit, config, context, cases, bags, y5, roles, initial_runs):
    verify_audit(audit, context, dict(config, production_runs=10), bags, y5, roles)
    require_equal(audit.get('initial_models'), initial_bindings(config, cases, bags, roles, initial_runs), 'initial fixed-C1 bindings', 0)


def audit_input(args):
    config, context, trusted, cases, bags, z, y5, roles = load_inputs(args)
    checks = []
    for (seed, trial), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(excluded)],
                       'fitting_support': np.bincount(y5[train], minlength=5).tolist(), 'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                       'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing audit output')
    bindings = initial_bindings(config, cases, bags, roles, args.initial_runs)
    output.mkdir(parents=True); base.write_json(output/'audit.json', {'status': 'zero-fit inputs accepted', 'new_fits': 0,
        'new_forwards': 0, 'outer_predictions': 0, 'checks': checks, 'initial_models': bindings, **context})


def encode_paths(model, paths, lookup, images_root, device, transform, batch_size):
    values = []; model.set_stage(True, training=False)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start+batch_size]
            pixels = torch.stack([transform(load_checked_image(images_root, lookup[p])) for p in batch]).to(device)
            value = model.encode(pixels).float().cpu().numpy()
            if value.shape != (len(batch), model.mean.shape[1]) or not np.isfinite(value).all(): raise ValueError('Invalid adapted field features')
            values.append(value)
    return np.concatenate(values).astype(np.float32)


def save_scores(folder, model, rows, cases, indices, images_root, config, device, transform, role):
    lookup = io.role_lookup(rows, cases, indices); paths = io.expected_paths(cases, indices)
    features = encode_paths(model, paths, lookup, images_root, device, transform, config['eval_batch_size'])
    arrays = {k: (v.astype(np.float64) if k.endswith(('_coef', '_intercept')) else v) for k, v in model.head_arrays().items()}
    np.savez_compressed(folder/'model.npz', **arrays)
    np.savez_compressed(folder/(role+'-features.npz'), paths=np.asarray(paths), features=features)
    records = io.records_from_features(paths, features, arrays, cases, indices, role, model.arm)
    path = folder/(role+'-predictions.jsonl'); path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    return summarize_records(records, model.arm), {'model_sha256': file_hash(folder/'model.npz'),
        role+'_features_sha256': file_hash(folder/(role+'-features.npz')), role+'_predictions_sha256': file_hash(path)}


def probe_paths(cases, train):
    return [(view, path) for view, mag in enumerate(MAGNIFICATIONS) for path in sorted(p for i in train for p in cases[i]['indices'][mag])[:2]]


def probe_outputs(model, probes, lookup, images_root, device, transform):
    paths = [p for _, p in probes]
    features = encode_paths(model, paths, lookup, images_root, device, transform, 8)
    with torch.inference_mode():
        logits = model.logits(torch.as_tensor(features, device=device), torch.tensor([v for v, _ in probes], device=device))
        probabilities = io.joint_probabilities(logits, model.arm).cpu().numpy()
    return features, probabilities


def reference_probe(arrays, probes, cases, bags, train, arm):
    by_path = {p: x for i in train for v, mag in enumerate(MAGNIFICATIONS) for p, x in zip(cases[i]['indices'][mag], bags[i][v])}
    features = np.asarray([by_path[p] for _, p in probes], np.float32); probabilities = []
    for (v, _), x in zip(probes, features):
        flags = np.zeros((1, 4), np.float32); flags[0, v] = 1
        descriptor = np.concatenate((((x-arrays['mean'][v])/arrays['std'][v])[None], flags), axis=1).astype(np.float64)
        probabilities.append(np.exp(field_log_probabilities(arrays, descriptor, arm))[0])
    return features, np.asarray(probabilities)


def parity_receipt(reference, actual, probes, config):
    return {'atol': config['atol'], 'rtol': config['rtol'], 'feature_max_abs': io.parity(reference[0], actual[0], config),
            'probability_max_abs': io.parity(reference[1], actual[1], config), 'passed': True, 'probe_paths': [p for _, p in probes]}


def build_adapter(args, config, arrays, device):
    from .midnight_features import load_model as backbone
    from .midnight_lora import AdapterClassifier
    c = config['lora']
    return AdapterClassifier(backbone(args.weights, device), arrays, args.arm, rank=c['rank'], alpha=c['alpha'], dropout=c['dropout']).to(device)


def runtime(config):
    from .midnight_features import read_config, runtime_versions
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('LoRA image work requires a CUDA SLURM allocation')
    runtime_versions(read_config(Path(__file__).resolve().parents[1]/'configs/midnight-12k-v1.json'))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    return torch.device('cuda')


def train_epochs(model, rows, cases, train, val, y5, args, config, context, device, deterministic, initial_parity, initial_digest, smoke=False):
    from .midnight_lora import weighted_case_loss
    output = Path(args.output); epochs = config['technical_smoke']['epochs'] if smoke else config['epochs']
    optimizer = torch.optim.AdamW(model.small_parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'],
                                  betas=tuple(config['optimizer']['betas']), eps=config['optimizer']['eps'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'], eta_min=config['scheduler']['eta_min'])
    seed = io.run_seed(args.seed, args.trial); lookup = io.role_lookup(rows, cases, train); augment = io.build_augmentation()
    history, candidates = [], []; first = 1; updates = 0
    initial_small = io.small_digests(model.small_state())
    if getattr(args, 'resume', False):
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        require_equal(saved['context'], context, 'resume context', 0)
        if saved['frozen_digest_initial'] != initial_digest: raise ValueError('Resume frozen initialization differs')
        model.load_small_state(saved['small_state']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        history, candidates = saved['history'], saved['candidates']; updates = saved['updates']; first = saved['epoch']+1
        initial_small = saved['initial_small_digests']; initial_parity = saved['initial_parity']; io.restore_rng(saved['rng'])
        if [r['epoch'] for r in history] != list(range(1, first)) or first > epochs+1: raise ValueError('Resume epoch history differs')
    counts = np.bincount(y5[train], minlength=5).tolist()
    for epoch in range(first, epochs+1):
        start = time.monotonic(); adapting = epoch > config['warmup_epochs']; model.set_stage(adapting, training=True)
        order = io.case_order(train, seed, epoch)
        if smoke: order = order[:config['technical_smoke']['batches_per_epoch']*config['case_batch_size']]
        optimizer.zero_grad(set_to_none=True); rates = []; receipts = []; gradient_steps = []; loss_sum = 0.; epoch_updates = 0
        for position, index in enumerate(order):
            draws = io.sampled_paths(cases[index], seed, epoch); receipts.append({'case_index': index, 'draws': draws})
            pixels = torch.stack([augment(load_checked_image(args.images_root, lookup[p]), io.stable_seed(seed, epoch, cases[index]['case_id'], view, sample, 'augmentation')) for view, sample, p in draws]).to(device)
            views = torch.tensor([v for v, _, _ in draws], device=device)
            _, logits = model(pixels, views)
            loss = weighted_case_loss(logits, torch.tensor([int(y5[index])], device=device), counts, args.arm, batch_denominator=config['last_batch_denominator'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite LoRA fitting loss')
            loss.backward(); loss_sum += float(loss.detach())
            if (position+1)%config['case_batch_size'] == 0 or position+1 == len(order):
                gradients = {'A': 0., 'B': 0., 'heads': 0.}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None: continue
                    group = 'A' if name.endswith('.lora_A') else 'B' if name.endswith('.lora_B') else 'heads' if name.startswith('heads.') else None
                    if group is None: raise ValueError('Gradient crossed the adapter/head whitelist')
                    maximum = float(parameter.grad.detach().abs().max())
                    if not math.isfinite(maximum): raise ValueError('Nonfinite small-parameter gradient')
                    gradients[group] = max(gradients[group], maximum)
                gradient_steps.append(gradients)
                torch.nn.utils.clip_grad_norm_(model.small_parameters(), config['gradient_clip'], error_if_nonfinite=True)
                rates.append(optimizer.param_groups[0]['lr']); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1; epoch_updates += 1
        current = io.small_digests(model.small_state())
        if current['normalization'] != initial_small['normalization'] or (not adapting and (current['A'] != initial_small['A'] or current['B'] != initial_small['B'])):
            raise ValueError('Fixed normalization or warmup adapter state changed')
        row = {'epoch': epoch, 'adapting': adapting, 'cases': len(order), 'optimizer_updates': epoch_updates,
               'microbatches': len(order), 'case_order': order, 'sampling_sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
               'learning_rates': rates, 'gradient_steps': gradient_steps, 'loss_sum': loss_sum, 'small_digests': current, 'elapsed_seconds': time.monotonic()-start}
        history.append(row)
        if not smoke and epoch in config['candidate_epochs']:
            folder = output/('epoch%d'%epoch); folder.mkdir(exist_ok=True)
            io.atomic_save({'context': context, 'epoch': epoch, 'small_state': model.small_state()}, folder/'state.pt')
            metric, hashes = save_scores(folder, model, rows, cases, val, args.images_root, config, device, deterministic, 'validation')
            candidate = {'epoch': epoch, 'validation': metric, 'state_sha256': file_hash(folder/'state.pt'), **hashes}
            base.write_json(folder/'metrics.json', candidate); candidates.append(candidate)
        best = max(candidates, key=lambda c: c['validation']['mean_subset_macro_f1'])['epoch'] if candidates else None
        io.atomic_save({'context': context, 'epoch': epoch, 'small_state': model.small_state(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng': io.rng_state(), 'history': history, 'candidates': candidates, 'best_epoch': best, 'updates': updates,
            'initial_parity': initial_parity, 'frozen_digest_initial': initial_digest, 'initial_small_digests': initial_small}, output/'last.pt')
        print(json.dumps({'epoch': epoch, 'cases': len(order), 'optimizer_updates': epoch_updates, 'adapting': adapting}), flush=True)
    final = io.small_digests(model.small_state())
    if final['A'] == initial_small['A'] or final['B'] == initial_small['B'] or final['heads'] == initial_small['heads']: raise ValueError('Active LoRA A/B or heads failed to update')
    return history, candidates, initial_parity


def execute(args, smoke=False):
    config, common, trusted, cases, bags, z, y5, roles = load_inputs(args)
    verify_input_audit(read_json(Path(args.audit)/'audit.json'), config, common, cases, bags, y5, roles, args.initial_runs)
    if smoke and (args.seed != config['technical_smoke']['seed'] or args.trial != config['technical_smoke']['trial']): raise ValueError('Technical smoke must use registered role')
    device = runtime(config); seed = io.run_seed(args.seed, args.trial)
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    output = Path(args.output)
    if output.exists() and not getattr(args, 'resume', False): raise ValueError('Refusing existing LoRA output')
    output.mkdir(parents=True, exist_ok=getattr(args, 'resume', False))
    train, val, excluded = roles[args.seed, args.trial]
    path = initial_model_path(args.initial_runs, args.seed, args.trial, args.arm); arrays = load_model(path, args.arm, dimension=config['encoder_dimension'])
    context = dict(common, arm=args.arm, seed=args.seed, trial=args.trial, initial_model_sha256=file_hash(path))
    from .midnight_features import build_transform
    deterministic = build_transform(); model = build_adapter(args, config, arrays, device)
    manifest = model.trainable_manifest(); frozen_initial = model.frozen_digest()
    if manifest['total_parameters'] != config['production_trainable_parameters_per_run']: raise ValueError('Trainable LoRA parameter budget differs')
    rows = read_manifest(args.manifest); lookup = io.role_lookup(rows, cases, train); probes = probe_paths(cases, train)
    initial_parity = parity_receipt(reference_probe(arrays, probes, cases, bags, train, args.arm), probe_outputs(model, probes, lookup, args.images_root, device, deterministic), probes, config['parity'])
    # Head constructors have different allocation shapes across arms; start dropout from a shared stream.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    history, candidates, initial_parity = train_epochs(model, rows, cases, train, val, y5, args, config, context, device, deterministic, initial_parity, frozen_initial, smoke)
    frozen_final = model.frozen_digest()
    if frozen_final != frozen_initial: raise ValueError('Frozen original Midnight parameters or buffers changed')
    selected_epoch = history[-1]['epoch'] if smoke else max(candidates, key=lambda c: c['validation']['mean_subset_macro_f1'])['epoch']
    if not smoke: model.load_small_state(torch.load(output/('epoch%d'%selected_epoch)/'state.pt', map_location='cpu', weights_only=True)['small_state'])
    selected_small = model.small_state(); reference = probe_outputs(model, probes, lookup, args.images_root, device, deterministic)
    del model; gc.collect(); torch.cuda.empty_cache()
    model = build_adapter(args, config, arrays, device); model.load_small_state(selected_small)
    reload_parity = parity_receipt(reference, probe_outputs(model, probes, lookup, args.images_root, device, deterministic), probes, config['parity'])
    if model.frozen_digest() != frozen_initial: raise ValueError('Reconstructed frozen Midnight digest differs')
    selected_fitting = None
    if not smoke:
        folder = output/'selected'; folder.mkdir(exist_ok=True)
        io.atomic_save({'context': context, 'epoch': selected_epoch, 'small_state': selected_small}, folder/'state.pt')
        fitting, hashes = save_scores(folder, model, rows, cases, train, args.images_root, config, device, deterministic, 'train')
        selected_fitting = {'epoch': selected_epoch, 'fitting': fitting, 'state_sha256': file_hash(folder/'state.pt'), **hashes}
        base.write_json(folder/'metrics.json', selected_fitting)
    report = {'status': 'accepted fitting-only LoRA technical smoke' if smoke else 'complete Midnight LoRA validation run', **context,
        'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'role_counts': [len(train), len(val), len(excluded)],
        'case_ids': {r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]},
        'audit_sha256': file_hash(Path(args.audit)/'audit.json'), 'history': history, 'candidates': candidates, 'selected_epoch': selected_epoch,
        'selected_validation': None if smoke else next(c['validation'] for c in candidates if c['epoch'] == selected_epoch),
        'selected_fitting': selected_fitting, 'initial_parity': initial_parity, 'reload_parity': reload_parity,
        'frozen_digest_initial': frozen_initial, 'frozen_digest_final': frozen_final, 'trainable_manifest': manifest,
        'last_checkpoint_sha256': file_hash(output/'last.pt'), 'outer_predictions': 0, 'smoke_validation_predictions': 0,
        'elapsed_seconds': time.monotonic()-started, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    base.write_json(output/'run.json', report)


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('audit', 'smoke', 'run', 'collect'))
    for name in ('manifest', 'protocol', 'cache', 'config', 'output', 'weights', 'initial-runs', 'images-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--arm', choices=ARMS); p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6)); p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action == 'collect':
        from .lora_collect import collect
        collect(args)
    else:
        if args.audit is None or args.arm is None or args.seed is None or args.trial is None: raise ValueError('Image jobs need audit, arm, seed and trial')
        execute(args, smoke=args.action == 'smoke')

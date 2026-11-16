"""Registered whole-case MIL on immutable F fields; no image encoder execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from . import coverage_five as source, midnight_five as base
from .bags import fit_normalization, normalize_bags, pad_bags
from .baseline import SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .five_collect import read_json, require_equal, verify_audit
from .five_metrics import FINAL5, MAP8_TO5, summarize_records

POOLS = ('mean', 'attention')
ARMS = ('direct', 'dual')
CANDIDATES = [10, 25, 50, 100]


def load_inputs(args):
    root = Path(__file__).resolve().parents[1]
    if file_hash(args.config) != file_hash(root/'configs/breakhis-case-mil-v1.json'):
        raise ValueError('Case-MIL config differs from immutable registration')
    own = read_json(args.config)
    fixed = dict(name='breakhis-case-mil-v1', schema_version=1, pool_modes=list(POOLS), arms=list(ARMS),
                 dimension=3072, hidden=128, attention_hidden=32, dropout=.25, epochs=100,
                 candidate_epochs=CANDIDATES, case_batch_size=4, training_seed=812, gradient_clip=1.,
                 production_runs=40, candidate_pipelines=160, checkpoint_archives=160,
                 prediction_archives=320, case_subset_records=129600, encoder_forwards=0,
                 outer_predictions=0, routing_threshold=.5, routing_tie='malignant',
                 normalization='fit_normalization case-balanced per-view moments')
    for key, value in fixed.items(): require_equal(own.get(key), value, 'registered MIL/'+key, 0)
    require_equal(own['optimizer'], dict(name='AdamW', learning_rate=.0003, weight_decay=.01,
                  betas=[.9, .999], eps=1e-8), 'registered MIL optimizer', 0)
    require_equal(own['scheduler'], dict(minimum_lr=.00003, scope='per epoch', formula='cosine epoch1..100'), 'registered MIL schedule', 0)
    path = root/own['base_input_config']
    if file_hash(path) != own['base_input_config_sha256']: raise ValueError('Accepted F input configuration differs')
    forwarded = SimpleNamespace(**vars(args)); forwarded.config = path
    inherited, context, trusted, cases, bags, z, y5, roles = source.load_inputs(forwarded)
    context = dict(context, config_sha256=file_hash(args.config), case_mil_config_sha256=file_hash(args.config),
                   case_mil_base_config_sha256=own['base_input_config_sha256'])
    context['implementation_hashes'] = dict(context['implementation_hashes'])
    for name in ('case_mil.py', 'case_mil_five.py', 'case_mil_collect.py'):
        context['implementation_hashes'][name] = file_hash(root/'magset'/name)
    return {**inherited, **own}, context, trusted, cases, bags, z, y5, roles


def verify_input_audit(audit, context, config, bags, y5, roles):
    verify_audit(audit, context, dict(config, production_runs=10), bags, y5, roles)


def audit_input(args):
    config, context, _, _, bags, _, y5, roles = load_inputs(args)
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing MIL audit output')
    checks = []
    for (s, t), (train, val, excluded) in sorted(roles.items()):
        mean, std = fit_normalization(bags, train)
        checks.append(dict(seed=s, trial=t, role_counts=[len(train), len(val), len(excluded)],
                           fitting_support=np.bincount(y5[train], minlength=5).tolist(),
                           validation_support=np.bincount(y5[val], minlength=5).tolist(),
                           normalization_sha256=hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()))
    record = dict(context, status='zero-fit inputs accepted', new_fits=0, new_forwards=0, outer_predictions=0, checks=checks)
    verify_input_audit(record, context, config, bags, y5, roles)
    output.mkdir(parents=True); base.write_json(output/'audit.json', record)


def run_seed(config, seed, trial): return config['training_seed']+10000*seed+trial


def case_order(indices, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(indices).astype(int).tolist()


def learning_rate(config, epoch):
    return config['scheduler']['minimum_lr'] + .5*(config['optimizer']['learning_rate']-config['scheduler']['minimum_lr'])*(1+math.cos(math.pi*(epoch-1)/(config['epochs']-1)))


def state_digest(state, shared=False):
    h = hashlib.sha256()
    for name in sorted(state):
        if shared and not name.startswith(('project.', 'heads.')): continue
        array = state[name].detach().cpu().numpy()
        h.update(name.encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def build_model(config, pool, arm):
    from .case_mil import CaseMIL
    return CaseMIL(config['dimension'], config['hidden'], config['attention_hidden'], config['dropout'], pool, arm)


def save_checkpoint(path, value):
    import torch
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    if path.exists() or temporary.exists(): raise ValueError('Refusing checkpoint overwrite')
    torch.save(value, temporary); temporary.replace(path)


def payload(model, context, epoch, mean, std):
    import torch
    return dict(context=context, epoch=epoch, model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                mean=torch.from_numpy(mean.copy()), std=torch.from_numpy(std.copy()))


def records_from_outputs(outputs, indices, cases, role, arm):
    if role not in ('train', 'validation') or arm not in ARMS: raise ValueError('Only fitting/validation outputs accepted')
    records = []
    for n, index in enumerate(indices):
        for s, subset in enumerate(SUBSETS):
            log5 = outputs['joint_log_probabilities'][s, n]
            if arm == 'dual':
                a = outputs['binary_logits'][s, n]; q = outputs['malignant_logits'][s, n]
                binary = -np.logaddexp(0, np.asarray([a, -a])); malignant = q-logsumexp(q)
            else:
                broad = logsumexp(log5[1:]); binary = np.asarray([log5[0], broad]); malignant = log5[1:]-broad
            score = float(np.exp(binary[1])); original = int(cases[index]['label'])
            prediction = int(log5.argmax()) if arm == 'direct' else 0 if score < .5 else 1+int(malignant.argmax())
            records.append(dict(role=role, case_id=cases[index]['case_id'], truth=original, truth5=int(MAP8_TO5[original]),
                magnifications=[MAGNIFICATIONS[v] for v in subset], probabilities5=np.exp(log5).tolist(),
                log_probabilities5=log5.tolist(), prediction=prediction, binary_score=score,
                binary_log_probabilities=binary.tolist(), malignant_probabilities4=np.exp(malignant).tolist(),
                malignant_log_probabilities=malignant.tolist()))
    return records


def infer_checkpoint(saved, config, bags, indices, pool, arm):
    """Canonical FP64 CPU inference using the frozen FP32-normalized field inputs."""
    import torch
    from .case_mil import all_subset_outputs
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, pool, arm).double().eval()
    model.load_state_dict({k: v.double() for k, v in saved['model_state'].items()}, strict=True)
    selected = [[(x-saved['mean'].numpy()[v])/saved['std'].numpy()[v] for v, x in enumerate(bags[i])] for i in indices]
    rows = []
    with torch.inference_mode():
        for start in range(0, len(indices), 4):
            x, mask = pad_bags(selected[start:start+4])
            rows.append({k: v.numpy() for k, v in all_subset_outputs(model, torch.from_numpy(x).double(), torch.from_numpy(mask)).items()})
    return {k: np.concatenate([r[k] for r in rows], axis=1) for k in rows[0]}


def parity(saved, config, bags, indices, pool, arm, device):
    import torch
    from .case_mil import all_subset_outputs
    expected = infer_checkpoint(saved, config, bags, indices, pool, arm)['joint_log_probabilities']
    with torch.random.fork_rng(devices=[]): model = build_model(config, pool, arm).to(device).eval()
    model.load_state_dict(saved['model_state'], strict=True)
    selected = [[(x-saved['mean'].numpy()[v])/saved['std'].numpy()[v] for v, x in enumerate(bags[i])] for i in indices]
    x, mask = pad_bags(selected)
    with torch.inference_mode(): actual = all_subset_outputs(model, torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))['joint_log_probabilities'].cpu().numpy()
    if not np.allclose(actual, expected, **config['parity']): raise ValueError('Fresh model/canonical probability parity failed')
    return dict(config['parity'], passed=True, maximum_log_abs=float(np.max(np.abs(actual-expected))),
                probe_indices=list(map(int, indices)))


def runtime():
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('MIL fitting requires a CUDA SLURM allocation')
    if str(torch.__version__) != '2.6.0+cu124' or 'A40' not in torch.cuda.get_device_name(): raise RuntimeError('Registered Torch/A40 runtime required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    return torch.device('cuda')


def execute(args, inputs, device, smoke=False, smoke_receipts=None):
    import torch
    from .case_mil import all_subset_outputs, fixed_batch_case_loss
    config, common, _, cases, bags, _, y5, roles = inputs; output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing MIL trajectory output')
    train, val, excluded = roles[args.seed, args.trial]
    counts = np.bincount(y5[train], minlength=5).tolist(); mean, std = fit_normalization(bags, train)
    chosen = [int(next(i for i in train if y5[i] == label)) for label in range(5)] if smoke else list(map(int, train))
    seed = run_seed(config, args.seed, args.trial); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = build_model(config, args.pool, args.arm).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    context = dict(common, pool=args.pool, arm=args.arm, seed=args.seed, trial=args.trial, smoke=smoke)
    normalized = normalize_bags([bags[i] for i in chosen], mean, std); x, mask = pad_bags(normalized)
    x, mask = torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device); local = {i: n for n, i in enumerate(chosen)}
    opt = config['optimizer']; optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'],
        weight_decay=opt['weight_decay'], betas=tuple(opt['betas']), eps=opt['eps'])
    history, candidates = [], []; updates = 0; output.mkdir(parents=True); start = time.monotonic()
    epochs = config['technical_smoke']['epochs'] if smoke else config['epochs']
    for epoch in range(1, epochs+1):
        begin = time.monotonic(); model.train(); order = case_order(chosen, seed, epoch); gradients = []; loss_sum = 0.
        rate = learning_rate(config, epoch)
        for group in optimizer.param_groups: group['lr'] = rate
        for offset in range(0, len(order), 4):
            batch = order[offset:offset+4]; ids = [local[i] for i in batch]; optimizer.zero_grad(set_to_none=True)
            loss = fixed_batch_case_loss(all_subset_outputs(model, x[ids], mask[ids]), torch.tensor(y5[batch], device=device), counts)
            if not torch.isfinite(loss): raise ValueError('Nonfinite MIL fitting loss')
            loss.backward(); receipt = dict(project=0., heads=0., attention=0., expert_supervised=bool(np.any(y5[batch] > 0)))
            for name, parameter in model.named_parameters():
                if parameter.grad is None: continue
                key = 'project' if name.startswith('project.') else 'heads' if name.startswith('heads.') else 'attention'
                maximum = float(parameter.grad.detach().abs().max())
                if not math.isfinite(maximum): raise ValueError('Nonfinite MIL gradient')
                receipt[key] = max(receipt[key], maximum)
                if name.startswith('heads.malignant.') and not receipt['expert_supervised']: raise ValueError('Benign batch updated expert')
            gradients.append(receipt); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); updates += 1; loss_sum += float(loss.detach())
        history.append(dict(epoch=epoch, case_order=order, cases=len(order), optimizer_updates=len(gradients),
                            learning_rate=rate, loss_sum=loss_sum, gradient_steps=gradients, elapsed_seconds=time.monotonic()-begin))
        if not smoke and epoch in config['candidate_epochs']:
            directory = output/('epoch%d'%epoch); directory.mkdir(); saved = payload(model, context, epoch, mean, std)
            save_checkpoint(directory/'state.pt', saved); restored = torch.load(directory/'state.pt', map_location='cpu', weights_only=True)
            if state_digest(restored['model_state']) != state_digest(saved['model_state']): raise ValueError('Checkpoint reload changed parameters')
            values = {}; hashes = {}
            for role, indices in [('train', train), ('validation', val)]:
                rows = records_from_outputs(infer_checkpoint(restored, config, bags, indices, args.pool, args.arm), indices, cases, role, args.arm)
                path = directory/(role+'-predictions.jsonl'); path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
                values[role] = summarize_records(rows, args.arm); hashes[role+'_predictions_sha256'] = file_hash(path)
            candidate = dict(epoch=epoch, fitting=values['train'], validation=values['validation'],
                state_sha256=file_hash(directory/'state.pt'), reload_parity=parity(restored, config, bags, chosen[:5], args.pool, args.arm, device), **hashes)
            base.write_json(directory/'metrics.json', candidate); candidates.append(candidate)
    last = dict(payload(model, context, epochs, mean, std), optimizer=optimizer.state_dict(), history=history,
                candidates=candidates, updates=updates, initial_model_state=initial, rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
    save_checkpoint(output/'last.pt', last)
    probe = parity(last, config, bags, chosen[:5], args.pool, args.arm, device)
    common_report = dict(context, audit_sha256=file_hash(Path(args.audit)/'audit.json'), history=history,
        initial_shared_sha256=state_digest(initial, shared=True), initial_model_sha256=state_digest(initial),
        final_model_sha256=state_digest(last['model_state']), last_checkpoint_sha256=file_hash(output/'last.pt'),
        parameter_count=sum(p.numel() for p in model.parameters()), reload_parity=probe, encoder_forwards=0,
        outer_predictions=0, elapsed_seconds=time.monotonic()-start,
        runtime=dict(torch=str(torch.__version__), device=device.type,
                     gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                     tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)),
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
    if smoke:
        base.write_json(output/'technical.json', dict(common_report, status='complete fitting-only case-MIL technical run',
            fitting_indices=chosen, validation_predictions=0, excluded_predictions=0, updates=updates))
    else:
        best = max(range(len(candidates)), key=lambda i: candidates[i]['validation']['mean_subset_macro_f1'])
        base.write_json(output/'run.json', dict(common_report, status='complete whole-case MIL validation run',
            class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), role_counts=[len(train), len(val), len(excluded)],
            case_ids={r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]},
            smoke_receipts=smoke_receipts, candidates=candidates, selected_epoch=candidates[best]['epoch'],
            fitting=candidates[best]['fitting'], validation=candidates[best]['validation']))


def run(args, smoke=False):
    inputs = load_inputs(args); config, context, _, cases, bags, _, y5, roles = inputs
    verify_input_audit(read_json(Path(args.audit)/'audit.json'), context, config, bags, y5, roles)
    if smoke and (args.seed, args.trial) != (config['technical_smoke']['seed'], config['technical_smoke']['trial']): raise ValueError('Smoke must use fixed fitting role')
    receipts = None
    if not smoke:
        from .case_mil_collect import verify_smokes
        receipts = verify_smokes(args.smokes, config, context, cases, bags, y5, roles, file_hash(Path(args.audit)/'audit.json'))
    execute(args, inputs, runtime(), smoke, receipts)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'smoke', 'run', 'collect', 'verify-smokes'))
    for name in ('manifest', 'protocol', 'cache', 'center-cache', 'feature-audit', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--code-sha', required=True); p.add_argument('--audit', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--seed', type=int, choices=(812, 813)); p.add_argument('--trial', type=int, choices=range(1, 6))
    p.add_argument('--pool', choices=POOLS); p.add_argument('--arm', choices=ARMS); p.add_argument('--smokes', type=Path)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'audit': audit_input(args)
    elif args.action in ('smoke', 'run'):
        if args.audit is None or args.pool is None or args.arm is None or args.seed is None or args.trial is None: raise ValueError('Fitting needs audit/pool/arm/seed/trial')
        run(args, args.action == 'smoke')
    else:
        from .case_mil_collect import collect, verify_smokes
        if args.action == 'collect': collect(args)
        else:
            c, ctx, _, cases, bags, _, y5, roles = load_inputs(args)
            verify_input_audit(read_json(Path(args.audit)/'audit.json'), ctx, c, bags, y5, roles)
            receipts = verify_smokes(args.smokes, c, ctx, cases, bags, y5, roles, file_hash(Path(args.audit)/'audit.json'))
            args.output.mkdir(parents=True, exist_ok=False); base.write_json(args.output/'smokes.json', receipts)

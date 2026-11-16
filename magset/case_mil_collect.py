"""Strict small-checkpoint validation and independent NumPy replay for case MIL."""
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from . import case_mil_five as runner
from .bags import fit_normalization
from .baseline import SUBSETS
from .data import CLASS_NAMES, file_hash
from .five_collect import aggregate_summaries, assert_public_safe, read_json, require_equal, verify_records
from .five_metrics import FINAL5
from .midnight_collect import delta, gate_comparison


def state_schema(config, pool, arm):
    d, h, a = (config[k] for k in ('dimension', 'hidden', 'attention_hidden'))
    result = {'project.0.weight': (h, d), 'project.0.bias': (h,)}
    for name, size in ([('direct', 5)] if arm == 'direct' else [('binary', 1), ('malignant', 4)]):
        result['heads.'+name+'.weight'] = (size, h+4); result['heads.'+name+'.bias'] = (size,)
    if pool == 'attention':
        for name in ('field_value', 'field_gate'):
            result[name+'.weight'] = (a, h); result[name+'.bias'] = (a,)
        result['field_score.weight'] = (1, a)
    return result


def check_model_state(state, config, pool, arm):
    import torch
    schema = state_schema(config, pool, arm)
    if not isinstance(state, dict) or set(state) != set(schema): raise ValueError('Model parameter whitelist differs')
    for name, shape in schema.items():
        v = state[name]
        if not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != torch.float32 or tuple(v.shape) != shape or not torch.isfinite(v).all():
            raise ValueError('Invalid model parameter: '+name)


def read_state(path, context, epoch, config, mean, std, last=False):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    keys = {'context', 'epoch', 'model_state', 'mean', 'std'}
    if last: keys.update(('optimizer', 'history', 'candidates', 'updates', 'initial_model_state', 'rng', 'cuda_rng'))
    if not isinstance(saved, dict) or set(saved) != keys: raise ValueError('Checkpoint schema differs')
    require_equal(saved['context'], context, 'checkpoint context', 0); require_equal(saved['epoch'], epoch, 'checkpoint epoch', 0)
    check_model_state(saved['model_state'], config, context['pool'], context['arm'])
    for name, wanted in [('mean', mean), ('std', std)]:
        value = saved[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or not np.array_equal(value.numpy(), wanted):
            raise ValueError('Original fitting-only normalizer differs')
    return saved


def numpy_outputs(saved, config, bags, indices, pool, arm):
    """Independent affine/ReLU/gated-attention/view/head implementation; no Torch model."""
    weights = {k: v.numpy().astype(np.float64) for k, v in saved['model_state'].items()}
    mean, std = saved['mean'].numpy(), saved['std'].numpy()
    views = np.empty((len(indices), 4, config['hidden']), dtype=np.float64)
    def affine(x, name): return x @ weights[name+'.weight'].T + weights[name+'.bias']
    for view in range(4):
        fields = [((bags[i][view]-mean[view])/std[view]).astype(np.float64) for i in indices]
        if any(x.ndim != 2 or not len(x) or x.shape[1] != config['dimension'] or not np.isfinite(x).all() for x in fields):
            raise ValueError('Invalid actual field inputs')
        projected = np.maximum(affine(np.concatenate(fields), 'project.0'), 0.)
        if pool == 'attention':
            score = (np.tanh(affine(projected, 'field_value'))*expit(affine(projected, 'field_gate'))) @ weights['field_score.weight'].T
        start = 0
        for n, field in enumerate(fields):
            end = start+len(field); value = projected[start:end]
            if pool == 'mean': views[n, view] = value.mean(0)
            else:
                logits = score[start:end, 0]; alpha = np.exp(logits-logsumexp(logits))
                views[n, view] = (alpha[:, None]*value).sum(0)
            start = end
    result = []
    for subset in SUBSETS:
        flags = np.zeros((len(indices), 4)); flags[:, list(subset)] = 1
        x = np.concatenate((views[:, list(subset)].sum(1)/len(subset), flags), axis=1)
        if arm == 'direct':
            logits = affine(x, 'heads.direct'); row = dict(direct_logits=logits, joint_log_probabilities=logits-logsumexp(logits, axis=1, keepdims=True))
        else:
            a = affine(x, 'heads.binary')[:, 0]; q = affine(x, 'heads.malignant'); qlog = q-logsumexp(q, axis=1, keepdims=True)
            row = dict(binary_logits=a, malignant_logits=q, joint_log_probabilities=np.column_stack((-np.logaddexp(0, a), -np.logaddexp(0, -a)[:, None]+qlog)))
        result.append(row)
    return {name: np.stack([r[name] for r in result]) for name in result[0]}


def _runtime(report):
    runtime = report.get('runtime', {})
    if runtime.get('torch') != '2.6.0+cu124' or 'A40' not in runtime.get('gpu', '') or runtime.get('device') != 'cuda' or runtime.get('tf32') is not False:
        raise ValueError('Actual registered CUDA/A40 runtime receipt differs')


def _parity(value, config, indices):
    for key, wanted in dict(config['parity'], passed=True, probe_indices=list(map(int, indices))).items():
        require_equal(value.get(key), wanted, 'reload parity/'+key, 0)
    v = value.get('maximum_log_abs')
    if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0: raise ValueError('Invalid parity diagnostic')


def _history(history, config, indices, y5, seed, trial, smoke=False):
    epochs = config['technical_smoke']['epochs'] if smoke else config['epochs']
    if not isinstance(history, list) or len(history) != epochs: raise ValueError('Incomplete training epoch history')
    for epoch, row in enumerate(history, 1):
        order = runner.case_order(indices, runner.run_seed(config, seed, trial), epoch)
        for key, value in dict(epoch=epoch, cases=len(indices), case_order=order, optimizer_updates=math.ceil(len(indices)/4), learning_rate=runner.learning_rate(config, epoch)).items():
            require_equal(row.get(key), value, 'history/'+key, 1e-15 if key == 'learning_rate' else 0)
        steps = row.get('gradient_steps')
        if not isinstance(steps, list) or len(steps) != math.ceil(len(indices)/4): raise ValueError('Incomplete gradient receipts')
        for n, step in enumerate(steps):
            if set(step) != {'project', 'heads', 'attention', 'expert_supervised'}: raise ValueError('Gradient receipt keys differ')
            require_equal(step['expert_supervised'], bool(np.any(y5[order[4*n:4*n+4]] > 0)), 'true malignant supervision', 0)
            if any(isinstance(step[k], bool) or not isinstance(step[k], (int, float)) or not math.isfinite(step[k]) or step[k] < 0 for k in ('project', 'heads', 'attention')):
                raise ValueError('Invalid finite gradient receipt')
        for name in ('loss_sum', 'elapsed_seconds'):
            if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]) or row[name] < 0: raise ValueError('Invalid epoch diagnostic')


def _optimizer(last, config, pool, arm):
    import torch
    schema = state_schema(config, pool, arm); optimizer = last['optimizer']; groups = optimizer.get('param_groups', [])
    if len(groups) != 1 or groups[0].get('params') != list(range(len(schema))): raise ValueError('Optimizer parameter ordering differs')
    for name in ('weight_decay', 'betas', 'eps'):
        require_equal(groups[0].get(name), config['optimizer'][name], 'optimizer/'+name, 0)
    require_equal(groups[0].get('lr'), last['history'][-1]['learning_rate'], 'last optimizer rate', 1e-15)
    updates = sum(r['optimizer_updates'] for r in last['history'])
    require_equal(last['updates'], updates, 'optimizer updates', 0)
    expert = sum(step['expert_supervised'] for row in last['history'] for step in row['gradient_steps'])
    states = optimizer.get('state', {})
    if set(states) != set(range(len(schema))): raise ValueError('Missing AdamW moments')
    for i, (name, shape) in enumerate(schema.items()):
        value = states[i]; step = value.get('step'); expected = expert if name.startswith('heads.malignant.') else updates
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != expected: raise ValueError('AdamW parameter update count differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            v = value.get(key)
            if not isinstance(v, torch.Tensor) or tuple(v.shape) != shape or v.dtype != torch.float32 or not torch.isfinite(v).all(): raise ValueError('Invalid AdamW moments')


def verify_smokes(root, config, common, cases, bags, y5, roles, audit_sha):
    if root is None: raise ValueError('Four accepted fitting-only smokes are required')
    root = Path(root); s, t = (config['technical_smoke'][k] for k in ('seed', 'trial')); train = roles[s, t][0]
    chosen = [int(next(i for i in train if y5[i] == label)) for label in range(5)]; mean, std = fit_normalization(bags, train)
    receipts, shared = {}, {}
    wanted = {root/p/a/'technical.json' for p in runner.POOLS for a in runner.ARMS}
    if set(root.rglob('technical.json')) != wanted or set(root.rglob('*.pt')) != {p.parent/'last.pt' for p in wanted} or list(root.rglob('*predictions*.jsonl')) or list(root.rglob('*.npz')):
        raise ValueError('Complete fitting-only four-smoke artifact matrix differs')
    for pool in runner.POOLS:
        for arm in runner.ARMS:
            folder = root/pool/arm; report = read_json(folder/'technical.json'); context = dict(common, pool=pool, arm=arm, seed=s, trial=t, smoke=True)
            for key, value in dict(context, status='complete fitting-only case-MIL technical run', fitting_indices=chosen,
                validation_predictions=0, excluded_predictions=0, encoder_forwards=0, outer_predictions=0, audit_sha256=audit_sha).items():
                require_equal(report.get(key), value, 'technical/'+key, 0)
            _runtime(report); _history(report['history'], config, chosen, y5, s, t, True); _parity(report['reload_parity'], config, chosen)
            if file_hash(folder/'last.pt') != report['last_checkpoint_sha256']: raise ValueError('Technical checkpoint hash differs')
            last = read_state(folder/'last.pt', context, config['technical_smoke']['epochs'], config, mean, std, True)
            require_equal(last['history'], report['history'], 'technical history', 0); require_equal(last['candidates'], [], 'technical candidates', 0)
            check_model_state(last['initial_model_state'], config, pool, arm); _optimizer(last, config, pool, arm)
            for key, wanted_hash in dict(initial_shared_sha256=runner.state_digest(last['initial_model_state'], True),
                initial_model_sha256=runner.state_digest(last['initial_model_state']), final_model_sha256=runner.state_digest(last['model_state'])).items():
                require_equal(report.get(key), wanted_hash, 'technical/'+key, 0)
            if report['final_model_sha256'] == report['initial_model_sha256']: raise ValueError('Technical model did not update')
            for group in ('project', 'heads') + (('attention',) if pool == 'attention' else ()):
                if max(step[group] for row in report['history'] for step in row['gradient_steps']) <= 0: raise ValueError('Technical trainable group received no gradients')
            shared[pool, arm] = report['initial_shared_sha256']
            receipts[pool+'/'+arm] = dict(technical_sha256=file_hash(folder/'technical.json'), last_checkpoint_sha256=file_hash(folder/'last.pt'))
    for arm in runner.ARMS: require_equal(shared['mean', arm], shared['attention', arm], 'pool-paired initial state', 0)
    return receipts


def verify_run(folder, pool, arm, seed, trial, config, common, cases, bags, y5, roles, audit_sha, smokes):
    folder = Path(folder); report = read_json(folder/'run.json'); context = dict(common, pool=pool, arm=arm, seed=seed, trial=trial, smoke=False)
    for key, value in dict(context, status='complete whole-case MIL validation run', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), role_counts=config['expected_role_counts'], encoder_forwards=0,
        outer_predictions=0, audit_sha256=audit_sha, smoke_receipts=smokes).items(): require_equal(report.get(key), value, 'run/'+key, 0)
    train, val, excluded = roles[seed, trial]; mean, std = fit_normalization(bags, train)
    require_equal(report.get('case_ids'), {r: [cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]}, 'original case memberships', 0)
    _runtime(report); _history(report.get('history'), config, train, y5, seed, trial)
    for name in ('elapsed_seconds', 'peak_gpu_bytes'):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid execution resource receipt')
    _parity(report['reload_parity'], config, list(train)[:5])
    candidates = report.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != 4: raise ValueError('Incomplete four-checkpoint matrix')
    public, states = [], {}
    for index, epoch in enumerate(config['candidate_epochs']):
        directory = folder/('epoch%d'%epoch); metrics = read_json(directory/'metrics.json')
        require_equal(candidates[index], metrics, 'embedded candidate', 0); require_equal(metrics['epoch'], epoch, 'candidate epoch', 0)
        if file_hash(directory/'state.pt') != metrics.get('state_sha256'): raise ValueError('Candidate checkpoint hash differs')
        saved = read_state(directory/'state.pt', context, epoch, config, mean, std); states[epoch] = saved['model_state']
        _parity(metrics['reload_parity'], config, list(train)[:5])
        for role, indices, name in [('train', train, 'fitting'), ('validation', val, 'validation')]:
            path = directory/(role+'-predictions.jsonl')
            if file_hash(path) != metrics.get(role+'_predictions_sha256'): raise ValueError('Candidate prediction hash differs')
            expected = runner.records_from_outputs(numpy_outputs(saved, config, bags, indices, pool, arm), indices, cases, role, arm)
            require_equal(metrics.get(name), verify_records(path, expected, cases, indices, role, arm), 'candidate/'+name)
        public.append(dict(metrics, metrics_sha256=file_hash(directory/'metrics.json')))
    index = max(range(4), key=lambda i: public[i]['validation']['mean_subset_macro_f1']); chosen = public[index]
    require_equal(report.get('selected_epoch'), chosen['epoch'], 'earliest best checkpoint', 0)
    for name in ('fitting', 'validation'): require_equal(report.get(name), chosen[name], 'selected/'+name)
    if file_hash(folder/'last.pt') != report.get('last_checkpoint_sha256'): raise ValueError('Last checkpoint hash differs')
    last = read_state(folder/'last.pt', context, config['epochs'], config, mean, std, True)
    require_equal(last['history'], report['history'], 'last history', 0); require_equal(last['candidates'], candidates, 'last candidates', 0)
    check_model_state(last['initial_model_state'], config, pool, arm); _optimizer(last, config, pool, arm)
    for name, value in last['model_state'].items():
        if not np.array_equal(value.numpy(), states[config['epochs']][name].numpy()): raise ValueError('Last model differs from final candidate')
    for key, value in dict(initial_shared_sha256=runner.state_digest(last['initial_model_state'], True), initial_model_sha256=runner.state_digest(last['initial_model_state']),
                            final_model_sha256=runner.state_digest(last['model_state']), parameter_count=sum(math.prod(v) for v in state_schema(config, pool, arm).values())).items():
        require_equal(report.get(key), value, 'run/'+key, 0)
    return dict(pool=pool, arm=arm, seed=seed, trial=trial, selected_epoch=chosen['epoch'], candidates=public,
                fitting=chosen['fitting'], validation=chosen['validation'], initial_shared_sha256=report['initial_shared_sha256'],
                elapsed_seconds=report['elapsed_seconds'], peak_gpu_bytes=report['peak_gpu_bytes'], runtime=report['runtime'],
                parameter_count=report['parameter_count'], run_sha256=file_hash(folder/'run.json'), last_checkpoint_sha256=file_hash(folder/'last.pt'))


def references(config, common, roles):
    from .prior_centered_collect import references as old_references
    result = old_references(config, common, roles); path = Path(__file__).resolve().parents[1]/config['context_reference_file']
    if file_hash(path) != config['context_reference_comparison_sha256']: raise ValueError('Accepted P reference hash differs')
    p = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance'):
        require_equal(p.get(key), common[key], 'P reference/'+key, 0)
    if p.get('verified_run_pairs') != 10 or {(r['seed'], r['trial']) for r in p['runs']} != set(roles): raise ValueError('P reference role matrix differs')
    result['head_specific_center'] = p
    return result


def collect(args):
    if args.runs is None or args.audit is None or args.smokes is None: raise ValueError('Collector needs runs/audit/four smokes')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing MIL comparison output')
    config, common, _, cases, bags, _, y5, roles = runner.load_inputs(args); audit = Path(args.audit)/'audit.json'
    runner.verify_input_audit(read_json(audit), common, config, bags, y5, roles)
    smokes = verify_smokes(args.smokes, config, common, cases, bags, y5, roles, file_hash(audit)); old = references(config, common, roles)
    root = Path(args.runs); paths = {root/p/a/('split%d'%s)/('trial%d'%t)/'run.json' for p in runner.POOLS for a in runner.ARMS for s, t in roles}
    directories = {p.parent/('epoch%d'%e) for p in paths for e in config['candidate_epochs']}
    if (set(root.rglob('run.json')) != paths or set(root.rglob('*.pt')) != {p.parent/'last.pt' for p in paths}|{p/'state.pt' for p in directories}
            or set(root.rglob('metrics.json')) != {p/'metrics.json' for p in directories} or list(root.rglob('*.npz'))
            or set(root.rglob('*predictions*.jsonl')) != {p/(r+'-predictions.jsonl') for p in directories for r in ('train', 'validation')}):
        raise ValueError('Complete forty-trajectory artifact matrix differs')
    runs = [verify_run(root/p/a/('split%d'%s)/('trial%d'%t), p, a, s, t, config, common, cases, bags, y5, roles, file_hash(audit), smokes)
            for p in runner.POOLS for a in runner.ARMS for s, t in sorted(roles)]
    for arm in runner.ARMS:
        for seed, trial in roles:
            pair = [r for r in runs if (r['arm'], r['seed'], r['trial']) == (arm, seed, trial)]
            require_equal(pair[0]['initial_shared_sha256'], pair[1]['initial_shared_sha256'], 'paired pool initialization', 0)
    records = sum(c[r]['pooled']['records'] for run in runs for c in run['candidates'] for r in ('fitting', 'validation'))
    if len(runs) != 40 or sum(len(r['candidates']) for r in runs) != 160 or records != 129600: raise ValueError('MIL fit/candidate/record budget differs')
    summary, paired, comparisons, passed = {}, {}, {}, {}
    for pool in runner.POOLS:
        summary[pool], paired[pool], comparisons[pool] = {}, {}, {}
        for seed in config['seeds']:
            values = {a: {role: aggregate_summaries([r[role] for r in runs if (r['pool'], r['arm'], r['seed']) == (pool, a, seed)]) for role in ('fitting', 'validation')} for a in runner.ARMS}
            summary[pool][str(seed)] = values; paired[pool][str(seed)] = delta(values['dual']['validation'], values['direct']['validation'])
            comparisons[pool][str(seed)] = {name: {a: gate_comparison(values[a]['validation'], (ref['summary']['head_specific'] if name == 'head_specific_center' else ref['summary'])[str(seed)]['dual']['validation'], config['improvement_screen']) for a in runner.ARMS} for name, ref in old.items()}
        passed[pool] = all(comparisons[pool][str(s)]['coverage']['dual']['gate_passed'] for s in config['seeds'])
    pooling_delta = {str(s): {a: delta(summary['attention'][str(s)][a]['validation'], summary['mean'][str(s)][a]['validation']) for a in runner.ARMS} for s in config['seeds']}
    result = dict(common, status='complete exploratory frozen-field whole-case MIL comparison', class_names=list(FINAL5),
        original_class_names=list(CLASS_NAMES), verified_runs=40, candidate_pipelines=160, verified_checkpoint_archives=160,
        verified_prediction_archives=320, verified_case_subset_records=records, trained_trajectories=40,
        collector_new_fits=0, encoder_forwards=0, outer_predictions=0, fitting_scope='in-sample neural diagnostic; not KDE leave-case-out',
        selection_opportunities_per_pool_arm=4, selection_opportunities_if_choosing_pool=8,
        audit_sha256=file_hash(audit), smoke_receipts=smokes, summary=summary, paired_direct_vs_dual=paired,
        attention_minus_mean=pooling_delta, comparisons_against_reference_dual=comparisons, gate_passed_by_pool=passed,
        reference_comparison_sha256=config['reference_comparison_sha256'], secondary_reference_comparison_sha256=config['secondary_reference_comparison_sha256'],
        context_reference_comparison_sha256=config['context_reference_comparison_sha256'],
        replay_scope='independent NumPy FP64 replay from every saved FP32 model and original accepted frozen-field caches; no retraining',
        distinct_validation_cases=len({cases[i]['case_id'] for _, val, _ in roles.values() for i in val}), runs=runs)
    assert_public_safe(result); output.mkdir(parents=True); runner.base.write_json(output/'comparison.json', result)
    return result

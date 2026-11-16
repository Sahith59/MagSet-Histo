"""CPU verification of the complete bounded Midnight LoRA output matrix."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .bags import fit_normalization
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from .field_five import score_records
from .five_class import load_model, write_json
from .five_collect import (aggregate_summaries, assert_public_safe, read_json,
                           require_equal, verify_records)
from .five_metrics import FINAL5
from .midnight_collect import gate_comparison, delta

ARMS = ('direct', 'dual')


def _state_schema(arm, config, dimension=3072):
    schema = {'mean': (4, dimension), 'std': (4, dimension)}
    rank = config['lora']['rank']
    for layer in config['lora']['layers']:
        for projection in config['lora']['projections']:
            prefix = 'backbone.encoder.layer.%d.attention.attention.%s.' % (layer, projection)
            schema[prefix+'lora_A'] = (rank, 1536)
            schema[prefix+'lora_B'] = (1536, rank)
    heads = {'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}
    for head, count in heads.items():
        schema['heads.'+head+'.weight'] = (count, dimension+4)
        schema['heads.'+head+'.bias'] = (count,)
    return schema


def _small_state(value, arrays, arm, config):
    schema = _state_schema(arm, config, arrays['mean'].shape[1])
    if not isinstance(value, dict) or set(value) != set(schema):
        raise ValueError('Small state parameter whitelist differs')
    for key, shape in schema.items():
        tensor = value[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
                or not torch.isfinite(tensor).all()):
            raise ValueError('Small state shape/dtype/finite values differ: '+key)
    for key in ('mean', 'std'):
        if not np.array_equal(value[key].numpy(), arrays[key]):
            raise ValueError('Small state normalizer differs from head archive')
    for head in ('direct',) if arm == 'direct' else ('binary', 'malignant'):
        for suffix, parameter in (('coef', 'weight'), ('intercept', 'bias')):
            tensor = value['heads.'+head+'.'+parameter].numpy().astype(np.float64)
            if not np.array_equal(tensor, arrays[head+'_'+suffix]):
                raise ValueError('Small state head differs from NPZ coefficients')
    return value


def verify_state(path, arrays, context, epoch, arm, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError('Expected a small-state checkpoint object')
    require_equal(saved.get('context'), context, 'state context', 0)
    require_equal(saved.get('epoch'), epoch, 'state epoch', 0)
    return _small_state(saved.get('small_state'), arrays, arm, config)


def _equal_states(actual, expected, label):
    if not isinstance(actual, dict) or set(actual) != set(expected) or any(not isinstance(actual[key], torch.Tensor) or not torch.equal(actual[key], expected[key]) for key in expected):
        raise ValueError(label+' small state differs')


def feature_bags(path, cases, indices, dimension=3072):
    """Require exact role path order, original coverage and unmodified FP32 fields."""
    expected = [key for index in indices for view in MAGNIFICATIONS
                for key in sorted(cases[index]['indices'][view])]
    if len(expected) != len(set(expected)):
        raise ValueError('Trusted role image paths overlap')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'paths', 'features'}:
            raise ValueError('Unexpected feature archive schema')
        paths = saved['paths'].tolist(); features = saved['features']
        if (paths != expected or features.shape != (len(expected), dimension)
                or features.dtype != np.float32 or not np.isfinite(features).all()):
            raise ValueError('Adapted feature paths/shape/dtype/role coverage differ')
        lookup = dict(zip(paths, features))
        bags = [[None]*4 for _ in cases]
        for index in indices:
            for view_index, view in enumerate(MAGNIFICATIONS):
                bags[index][view_index] = np.stack([lookup[key] for key in sorted(cases[index]['indices'][view])])
    return bags


def _heads(path, arm, mean, std):
    arrays = load_model(path, arm, dimension=3072)
    if not np.array_equal(arrays['mean'], mean) or not np.array_equal(arrays['std'], std):
        raise ValueError('Fixed original fitting normalizer changed')
    return arrays


def _score_archive(feature_path, prediction_path, arrays, cases, indices, role, arm):
    bags = feature_bags(feature_path, cases, indices)
    expected, _ = score_records(arrays, bags, indices, cases, role, arm)
    return verify_records(prediction_path, expected, cases, indices, role, arm)


def _parity(receipt, config, cases, train):
    if not isinstance(receipt, dict):
        raise ValueError('Missing producer parity receipt')
    for key in ('atol', 'rtol'):
        require_equal(receipt.get(key), config['parity'][key], 'parity/'+key, 0)
    require_equal(receipt.get('passed'), True, 'producer allclose acceptance', 0)
    for key in ('feature_max_abs', 'probability_max_abs'):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid producer parity error')
    allowed = {path for index in train for view in MAGNIFICATIONS for path in cases[index]['indices'][view]}
    paths = receipt.get('probe_paths')
    if not isinstance(paths, list) or not paths or any(path not in allowed for path in paths):
        raise ValueError('Producer parity probe contains a forbidden-role path')
    expected = [path for view in MAGNIFICATIONS for path in sorted(path for index in train for path in cases[index]['indices'][view])[:2]]
    require_equal(paths, expected, 'fixed fitting probe coverage', 0)
    return {key: receipt[key] for key in ('atol', 'rtol', 'feature_max_abs', 'probability_max_abs', 'passed')}


def _manifest(value, arm, config):
    schema = _state_schema(arm, config)
    schema = {key: shape for key, shape in schema.items() if key not in ('mean', 'std')}
    if not isinstance(value, dict) or not isinstance(value.get('parameters'), list):
        raise ValueError('Missing trainable parameter manifest')
    entries = value['parameters']
    if len(entries) != len(schema) or {item.get('name') for item in entries} != set(schema):
        raise ValueError('Trainable parameter whitelist differs')
    for item in entries:
        shape = schema[item['name']]
        require_equal(item.get('shape'), list(shape), 'trainable shape', 0)
        require_equal(item.get('count'), math.prod(shape), 'trainable count', 0)
        if type(item.get('requires_grad')) is not bool:
            raise ValueError('Trainable flag is invalid')
    expected = {'adapter_parameters': 98304, 'head_parameters': 15385, 'total_parameters': 113689}
    for key, count in expected.items():
        require_equal(value.get(key), count, 'trainable manifest/'+key, 0)
    return {key: value[key] for key in expected}


def _history(history, config, cases, train, seed, trial):
    from . import lora_io as io
    if not isinstance(history, list) or len(history) != config['epochs']:
        raise ValueError('Incomplete fixed training epoch history')
    normalization = None; gradient_totals = {'A': 0., 'B': 0., 'heads': 0.}
    for epoch, row in enumerate(history, 1):
        for key, expected in {'epoch': epoch, 'adapting': epoch > config['warmup_epochs'],
                              'cases': 46, 'optimizer_updates': 12, 'microbatches': 46}.items():
            require_equal(row.get(key), expected, 'history/'+key, 0)
        order = row.get('case_order')
        if not isinstance(order, list) or len(order) != 46 or set(order) != set(map(int, train)):
            raise ValueError('Epoch fitting case visitation differs')
        require_equal(order, io.case_order(train, io.run_seed(seed, trial), epoch), 'paired case order', 0)
        receipts = [{'case_index': index, 'draws': io.sampled_paths(cases[index], io.run_seed(seed, trial), epoch)} for index in order]
        expected_hash = hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()
        require_equal(row.get('sampling_sha256'), expected_hash, 'paired field sampling', 0)
        rates = row.get('learning_rates')
        if not isinstance(rates, list) or len(rates) != 12 or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError('Epoch learning-rate receipt is incomplete')
        expected_rates = [config['optimizer']['lr']*.5*(1+math.cos(math.pi*((epoch-1)*12+step)/config['scheduler']['T_max'])) for step in range(12)]
        require_equal(rates, expected_rates, 'registered cosine schedule', 1e-14)
        gradients = row.get('gradient_steps')
        if not isinstance(gradients, list) or len(gradients) != 12:
            raise ValueError('Missing optimizer gradient receipts')
        for step in gradients:
            if set(step) != set(gradient_totals) or any(not math.isfinite(value) or value < 0 for value in step.values()):
                raise ValueError('Invalid optimizer gradient receipt')
            if epoch <= config['warmup_epochs'] and (step['A'] != 0 or step['B'] != 0):
                raise ValueError('Adapter gradients crossed the head warmup boundary')
            for key in gradient_totals:
                gradient_totals[key] = max(gradient_totals[key], step[key])
        for key in ('loss_sum', 'elapsed_seconds'):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid training history value')
        digest = row.get('small_digests', {})
        if set(digest) != {'A', 'B', 'heads', 'normalization'}:
            raise ValueError('Missing small-state phase digests')
        if normalization is None:
            normalization = digest['normalization']
        if digest['normalization'] != normalization:
            raise ValueError('Original normalizer changed during adaptation')
    if any(value <= 0 for value in gradient_totals.values()):
        raise ValueError('An active adapter/head group never received a gradient')


def _optimizer(last, config, arm, train, y5):
    optimizer = last['optimizer']; groups = optimizer.get('param_groups', [])
    schema = _state_schema(arm, config)
    names = [key for key in schema if key not in ('mean', 'std')]
    if len(groups) != 1 or groups[0].get('params') != list(range(len(names))):
        raise ValueError('Optimizer parameter whitelist/order differs')
    for key in ('weight_decay', 'betas', 'eps'):
        require_equal(groups[0].get(key), config['optimizer'][key], 'optimizer/'+key, 0)
    require_equal(last['scheduler'].get('last_epoch'), 240, 'scheduler updates', 0)
    require_equal(last['scheduler'].get('T_max'), config['scheduler']['T_max'], 'scheduler period', 0)
    states = optimizer.get('state', {})
    if set(states) != set(range(len(names))):
        raise ValueError('Optimizer lacks small-parameter moments')
    malignant_updates = sum(any(y5[index] > 0 for index in row['case_order'][start:start+4]) for row in last['history'] for start in range(0, 46, 4))
    for index, name in enumerate(names):
        state = states[index]
        expected = 216 if name.startswith('backbone.') else malignant_updates if name.startswith('heads.malignant.') else 240
        step = state.get('step')
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != expected:
            raise ValueError('Optimizer update count differs: '+name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != schema[name] or not torch.isfinite(value).all():
                raise ValueError('Invalid optimizer moments: '+name)


def verify_run(folder, arm, seed, trial, config, common, cases, bags, y5, roles, audit_sha):
    folder = Path(folder); report = read_json(folder/'run.json')
    initial_hash = config['initial_model_sha256']['%d/%d/%s' % (seed, trial, arm)]
    context = dict(common, arm=arm, seed=seed, trial=trial, initial_model_sha256=initial_hash)
    for key, value in context.items():
        require_equal(report.get(key), value, 'run context/'+key, 0)
    for key, value in {'status': 'complete Midnight LoRA validation run', 'class_names': list(FINAL5),
                       'original_class_names': list(CLASS_NAMES), 'role_counts': [46, 8, 28],
                       'audit_sha256': audit_sha, 'outer_predictions': 0}.items():
        require_equal(report.get(key), value, 'run/'+key, 0)
    train, val, excluded = roles[seed, trial]
    identities = {role: [cases[index]['case_id'] for index in indices] for role, indices in
                  (('train', train), ('validation', val), ('excluded', excluded))}
    require_equal(report.get('case_ids'), identities, 'original role membership', 0)
    _history(report.get('history'), config, cases, train, seed, trial)
    manifest = _manifest(report.get('trainable_manifest'), arm, config)
    parity = {name: _parity(report.get(name), config, cases, train) for name in ('initial_parity', 'reload_parity')}
    frozen = report.get('frozen_digest_initial')
    if not isinstance(frozen, str) or len(frozen) != 64 or frozen != report.get('frozen_digest_final'):
        raise ValueError('Frozen backbone digest receipt changed')
    mean, std = fit_normalization(bags, train)
    candidates = report.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != len(config['candidate_epochs']):
        raise ValueError('Incomplete candidate checkpoint matrix')
    public = []; states = {}; expected_pt = {folder/'last.pt'}; expected_npz = set(); expected_scores = set()
    for index, epoch in enumerate(config['candidate_epochs']):
        directory = folder/('epoch%d' % epoch); metrics = read_json(directory/'metrics.json')
        require_equal(candidates[index], metrics, 'embedded candidate', 0)
        require_equal(metrics.get('epoch'), epoch, 'candidate epoch', 0)
        files = {'state_sha256': directory/'state.pt', 'model_sha256': directory/'model.npz',
                 'validation_features_sha256': directory/'validation-features.npz',
                 'validation_predictions_sha256': directory/'validation-predictions.jsonl'}
        if any(file_hash(path) != metrics.get(key) for key, path in files.items()):
            raise ValueError('Candidate artifact hash differs')
        arrays = _heads(files['model_sha256'], arm, mean, std)
        states[epoch] = verify_state(files['state_sha256'], arrays, context, epoch, arm, config)
        summary = _score_archive(files['validation_features_sha256'], files['validation_predictions_sha256'], arrays, cases, val, 'validation', arm)
        require_equal(metrics.get('validation'), summary, 'candidate validation')
        public.append(dict(metrics, metrics_sha256=file_hash(directory/'metrics.json')))
        expected_pt.add(files['state_sha256']); expected_npz.update((files['model_sha256'], files['validation_features_sha256']))
        expected_scores.add(files['validation_predictions_sha256'])
    best = max(range(len(public)), key=lambda index: public[index]['validation']['mean_subset_macro_f1'])
    epoch = config['candidate_epochs'][best]
    require_equal(report.get('selected_epoch'), epoch, 'earliest maximum epoch', 0)
    require_equal(report.get('selected_validation'), public[best]['validation'], 'selected validation')
    selected = folder/'selected'; metrics = read_json(selected/'metrics.json')
    require_equal(metrics.get('epoch'), epoch, 'selected fitting epoch', 0)
    selected_files = {'state_sha256': selected/'state.pt', 'model_sha256': selected/'model.npz',
                      'train_features_sha256': selected/'train-features.npz',
                      'train_predictions_sha256': selected/'train-predictions.jsonl'}
    if any(file_hash(path) != metrics.get(key) for key, path in selected_files.items()):
        raise ValueError('Selected fitting artifact hash differs')
    arrays = _heads(selected_files['model_sha256'], arm, mean, std)
    selected_state = verify_state(selected_files['state_sha256'], arrays, context, epoch, arm, config)
    _equal_states(selected_state, states[epoch], 'Selected candidate')
    fitting = _score_archive(selected_files['train_features_sha256'], selected_files['train_predictions_sha256'], arrays, cases, train, 'train', arm)
    require_equal(metrics.get('fitting'), fitting, 'selected fitting metrics')
    require_equal(report.get('selected_fitting'), metrics, 'run selected fitting record', 0)
    expected_pt.add(selected_files['state_sha256']); expected_npz.update((selected_files['model_sha256'], selected_files['train_features_sha256']))
    expected_scores.add(selected_files['train_predictions_sha256'])
    last_path = folder/'last.pt'
    if file_hash(last_path) != report.get('last_checkpoint_sha256'):
        raise ValueError('Last checkpoint hash differs')
    last = torch.load(last_path, map_location='cpu', weights_only=True)
    for key, expected in {'context': context, 'epoch': config['epochs'], 'history': report['history'],
                          'candidates': candidates, 'best_epoch': epoch, 'initial_parity': report['initial_parity'],
                          'frozen_digest_initial': frozen, 'updates': 240}.items():
        require_equal(last.get(key), expected, 'last checkpoint/'+key, 0)
    _equal_states(last.get('small_state', {}), states[config['epochs']], 'Last candidate')
    if any(key not in last for key in ('optimizer', 'scheduler', 'rng')):
        raise ValueError('Last checkpoint lacks resumable state')
    _optimizer(last, config, arm, train, y5)
    if set(folder.rglob('*.pt')) != expected_pt or set(folder.rglob('*.npz')) != expected_npz or set(folder.rglob('*predictions*.jsonl')) != expected_scores:
        raise ValueError('Unexpected run artifacts or forbidden-role scores')
    return {'seed': seed, 'trial': trial, 'arm': arm, 'selected_epoch': epoch, 'candidates': public,
            'fitting': fitting, 'validation': public[best]['validation'], 'trainable_manifest': manifest,
            'initial_parity': parity['initial_parity'], 'reload_parity': parity['reload_parity'],
            'frozen_digest_initial': frozen, 'frozen_digest_final': frozen,
            'run_sha256': file_hash(folder/'run.json'), 'last_checkpoint_sha256': file_hash(last_path),
            'selected_metrics_sha256': file_hash(selected/'metrics.json')}


def reference_comparison(config, common, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'docs/evidence/BREAKHIS_MIDNIGHT_RESULTS.json'
    if file_hash(path) != config['reference_comparison_sha256']:
        raise ValueError('Accepted Midnight reference hash differs')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance', 'config_sha256'):
        require_equal(reference.get(key), common[key], 'reference/'+key, 0)
    for key, value in {'readouts': ['case', 'field'], 'class_names': list(FINAL5),
                       'original_class_names': list(CLASS_NAMES), 'verified_runs': 20,
                       'head_fits': 240, 'outer_predictions': 0}.items():
        require_equal(reference.get(key), value, 'reference/'+key, 0)
    for run in reference['runs']:
        if run['readout'] == 'field':
            for arm in ARMS:
                candidate = run['arms'][arm]['candidates'][config['initial_candidate_index']]
                identity = '%d/%d/%s' % (run['seed'], run['trial'], arm)
                require_equal(candidate['C'], config['initial_C'], 'fixed initialization C', 0)
                require_equal(candidate['model_sha256'], config['initial_model_sha256'][identity], 'fixed initialization model', 0)
    return reference


def support_reference(config, common, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/config['support_reference_file']
    if file_hash(path) != config['support_reference_comparison_sha256']:
        raise ValueError('Accepted kernel reference hash differs')
    reference = read_json(path)
    for key in ('manifest_sha256', 'protocol_sha256', 'encoder', 'cache_provenance', 'config_sha256'):
        require_equal(reference.get(key), common[key], 'kernel reference/'+key, 0)
    for key, value in {'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                       'verified_runs': 10, 'candidate_pipelines': 80, 'outer_predictions': 0}.items():
        require_equal(reference.get(key), value, 'kernel reference/'+key, 0)
    expected = {(seed, trial) for seed in config['seeds'] for trial in config['trials']}
    runs = reference.get('runs', [])
    if len(runs) != 10 or {(run['seed'], run['trial']) for run in runs} != expected:
        raise ValueError('Kernel reference role matrix differs')
    for run in runs:
        for arm in ARMS:
            require_equal(run['arms'][arm]['validation']['distinct_class_support'], config['validation_five_support'], 'kernel validation support', 0)
    return reference


def aggregate(runs, config, common, cases, roles, reference, kernel, audit_sha):
    summary = {}; comparisons = {}; paired = {}; kernel_comparisons = {}
    for seed in config['seeds']:
        values = {}
        for arm in ARMS:
            selected = [run for run in runs if run['seed'] == seed and run['arm'] == arm]
            values[arm] = {role: aggregate_summaries([run[role] for run in selected]) for role in ('fitting', 'validation')}
        summary[str(seed)] = values
        paired[str(seed)] = delta(values['dual']['validation'], values['direct']['validation'])
        comparisons[str(seed)] = {readout: {arm: gate_comparison(values[arm]['validation'], reference['summary'][readout][str(seed)]['dual']['validation'], config['improvement_screen']) for arm in ARMS} for readout in ('case', 'field')}
        kernel_comparisons[str(seed)] = {arm: gate_comparison(values[arm]['validation'], kernel['summary'][str(seed)]['dual']['validation'], config['improvement_screen']) for arm in ARMS}
    result = dict(common, status='complete exploratory Midnight LoRA five-class development comparison',
                  class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=20,
                  candidate_pipelines=80, verified_candidate_validation_records=9600,
                  verified_selected_fitting_records=13800, trainable_parameters_per_run=113689,
                  collector_new_fits=0, collector_encoder_forwards=0, outer_predictions=0,
                  audit_sha256=audit_sha, reference_comparison_sha256=config['reference_comparison_sha256'],
                  reference_code_sha=reference['code_sha'], summary=summary, paired_direct_vs_dual=paired,
                  comparison_against_midnight_dual=comparisons,
                  support_reference_comparison_sha256=config['support_reference_comparison_sha256'],
                  comparison_against_kernel_dual=kernel_comparisons,
                  gate_passed_against_midnight_case=all(comparisons[str(seed)]['case']['dual']['gate_passed'] for seed in config['seeds']),
                  gate_passed=all(kernel_comparisons[str(seed)]['dual']['gate_passed'] for seed in config['seeds']),
                  gate_scope='descriptive reused provider-case validation against fixed kernel dual; all reference deltas retained',
                  replay_scope='CPU replays saved adapted field features and small states; producer records full-base reconstruction probes',
                  distinct_validation_cases=len({cases[index]['case_id'] for _, val, _ in roles.values() for index in val}), runs=runs)
    assert_public_safe(result)
    return result


def collect(args):
    from . import lora_five as runner
    if args.runs is None or args.audit is None:
        raise ValueError('Collector requires complete runs and the common input audit')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing existing comparison output')
    config, common, trusted, cases, bags, z, y5, roles = runner.load_inputs(args)
    audit_path = Path(args.audit)/'audit.json'
    runner.verify_input_audit(read_json(audit_path), config, common, cases, bags, y5, roles, args.initial_runs)
    reference = reference_comparison(config, common, getattr(args, 'reference', None))
    kernel = support_reference(config, common, getattr(args, 'support_reference', None))
    root = Path(args.runs)
    wanted = {root/arm/('split%d'%seed)/('trial%d'%trial)/'run.json' for arm in ARMS for seed, trial in roles}
    if set(root.rglob('run.json')) != wanted:
        raise ValueError('Complete twenty-run LoRA matrix differs')
    checkpoints = {path.parent/'last.pt' for path in wanted}
    npzs = set(); scores = set()
    for path in wanted:
        for epoch in config['candidate_epochs']:
            directory = path.parent/('epoch%d'%epoch)
            checkpoints.add(directory/'state.pt'); npzs.update((directory/'model.npz', directory/'validation-features.npz'))
            scores.add(directory/'validation-predictions.jsonl')
        selected = path.parent/'selected'
        checkpoints.add(selected/'state.pt'); npzs.update((selected/'model.npz', selected/'train-features.npz'))
        scores.add(selected/'train-predictions.jsonl')
    if set(root.rglob('*.pt')) != checkpoints or set(root.rglob('*.npz')) != npzs or set(root.rglob('*predictions*.jsonl')) != scores:
        raise ValueError('Complete LoRA artifact coverage differs, including extra-role files')
    runs = [verify_run(root/arm/('split%d'%seed)/('trial%d'%trial), arm, seed, trial, config, common,
                       cases, bags, y5, roles, file_hash(audit_path)) for arm in ARMS for seed, trial in sorted(roles)]
    if len(runs) != 20 or sum(len(run['candidates']) for run in runs) != 80:
        raise ValueError('Registered LoRA run/candidate budget differs')
    result = aggregate(runs, config, common, cases, roles, reference, kernel, file_hash(audit_path))
    output.mkdir(parents=True); write_json(output/'comparison.json', result)
    return result

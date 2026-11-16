#!/usr/bin/env python3
"""CPU-only independent artifact acceptance of both fitting-only LoRA smokes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import lora_five as runner, lora_io
from magset.data import MAGNIFICATIONS, file_hash
from magset.five_class import load_model
from magset.midnight_features import verify_files
from magset.midnight_lora import small_state_schema


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == 'cpu' and bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def check_parity(receipt, expected_paths, config):
    require(isinstance(receipt, dict) and receipt.get('passed') is True, 'Missing producer parity acceptance')
    require(receipt.get('probe_paths') == expected_paths and len(set(expected_paths)) == 8, 'Probe paths are not the exact eight deterministic fitting paths')
    for key in ('atol', 'rtol'):
        require(receipt.get(key) == config['parity'][key] == 1e-4, 'Parity tolerance differs')
    for key in ('feature_max_abs', 'probability_max_abs'):
        require(number(receipt.get(key)), 'Nonfinite/negative producer parity error')
    # For probabilities in [0,1], the registered allclose bound cannot exceed atol+rtol.
    require(receipt['probability_max_abs'] <= 2e-4, 'Probability error contradicts the recorded tolerance')
    return {key: receipt[key] for key in ('atol', 'rtol', 'feature_max_abs', 'probability_max_abs', 'passed')}


def check_manifest(manifest, schema):
    expected = {k: v for k, v in schema.items() if k not in ('mean', 'std')}
    entries = manifest.get('parameters', [])
    require(len(entries) == len(expected) and {x.get('name') for x in entries} == set(expected), 'Small-parameter manifest whitelist differs')
    for item in entries:
        shape = expected[item['name']]['shape']
        require(item.get('shape') == shape and item.get('count') == math.prod(shape), 'Manifest shape/count differs')
        require(item.get('requires_grad') is item['name'].startswith('heads.'), 'Initial warmup trainability differs')
    for key, value in {'adapter_parameters': 98304, 'head_parameters': 15385,
                       'total_parameters': 113689, 'active_parameters': 15385}.items():
        require(manifest.get(key) == value, 'Incorrect canonical parameter count: ' + key)


def check_state(state, arrays, schema):
    require(isinstance(state, dict) and set(state) == set(schema), 'Small state contains a missing or unregistered tensor')
    for key, expected in schema.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.float32 and list(value.shape) == expected['shape']
                and bool(torch.isfinite(value).all()), 'Invalid small-state tensor: ' + key)
    for key in ('mean', 'std'):
        require(np.array_equal(state[key].numpy(), arrays[key]), 'Original fixed normalizer changed')


def check_history(history, initial, config, cases, train, seed):
    require(isinstance(history, list) and len(history) == 3, 'Exactly three smoke epochs are required')
    require(set(initial) == {'A', 'B', 'heads', 'normalization'} and all(digest(v) for v in initial.values()), 'Initial small-state digests missing')
    comparison = []; gradients = []
    for epoch, row in enumerate(history, 1):
        order = lora_io.case_order(train, seed, epoch)[:8]
        receipts = [{'case_index': i, 'draws': lora_io.sampled_paths(cases[i], seed, epoch)} for i in order]
        for receipt in receipts:
            draws = receipt['draws']; case = cases[receipt['case_index']]
            require(len(draws) == 8, 'Smoke must draw exactly eight fields per case')
            for view, mag in enumerate(MAGNIFICATIONS):
                selected = [(sample, path) for v, sample, path in draws if v == view]
                require([sample for sample, _ in selected] == [0, 1]
                        and all(path in case['indices'][mag] for _, path in selected), 'Smoke field draw crossed its case/view')
        sampling = hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()
        expected_lr = [config['optimizer']['lr'] * (1 + math.cos(math.pi * step / 240)) / 2
                       for step in range((epoch - 1) * 2, epoch * 2)]
        require(row.get('epoch') == epoch and row.get('adapting') is (epoch == 3), 'Smoke stage transition differs')
        require(row.get('cases') == row.get('microbatches') == 8 and row.get('optimizer_updates') == 2, 'Smoke case/update budget differs')
        require(row.get('case_order') == order and len(set(order)) == 8 and set(order) <= set(train), 'Smoke case order contains a forbidden or repeated case')
        require(row.get('sampling_sha256') == sampling, 'Stateless fitting field draws differ')
        rates = row.get('learning_rates', [])
        require(len(rates) == 2 and np.allclose(rates, expected_lr, rtol=0, atol=1e-15), 'Six-step common cosine learning rates differ')
        require(number(row.get('elapsed_seconds'), True) and number(row.get('loss_sum'), True), 'Invalid epoch timing or loss')
        current = row.get('small_digests', {})
        require(set(current) == set(initial) and all(digest(v) for v in current.values()), 'Epoch small-state digests differ')
        require(current['normalization'] == initial['normalization'], 'Training normalization changed')
        if epoch < 3:
            require(current['A'] == initial['A'] and current['B'] == initial['B'], 'Adapter changed during head warmup')
        else:
            require(all(current[k] != initial[k] for k in ('A', 'B', 'heads')), 'Adaptation did not change required small groups')
        steps = row.get('gradient_steps', [])
        require(len(steps) == 2 and all(set(s) == {'A', 'B', 'heads'} and all(number(v) for v in s.values()) for s in steps), 'Actual finite gradient receipts missing')
        require(all(s['heads'] > 0 for s in steps), 'Head gradients are absent')
        if epoch < 3:
            require(all(s['A'] == s['B'] == 0 for s in steps), 'Adapter gradients crossed warmup')
        else:
            require(steps[0]['A'] == 0 and steps[0]['B'] > 0 and steps[1]['A'] > 0 and steps[1]['B'] > 0,
                    'Actual A/B gradients did not exercise zero-B initialization and the subsequent A update')
        gradients.extend(steps)
        comparison.append({'case_order': order, 'sampling_sha256': sampling, 'learning_rates': rates})
    return comparison, gradients


def review(args):
    require(not args.output.exists(), 'Refusing to overwrite a smoke review')
    config, common, _, cases, bags, _, y5, roles = runner.load_inputs(args)
    audit = json.loads((args.audit / 'audit.json').read_text())
    runner.verify_input_audit(audit, config, common, cases, bags, y5, roles, args.initial_runs)
    verify_files(args.weights)  # byte hashes only; no model construction or forward
    require(config['technical_smoke']['epochs'] == 3 and config['technical_smoke']['batches_per_epoch'] == 2, 'Smoke registration differs')
    train, val, excluded = roles[812, 1]
    probes = [p for mag in MAGNIFICATIONS for p in sorted(p for i in train for p in cases[i]['indices'][mag])[:2]]
    paths = list(args.smokes.rglob('run.json'))
    require(len(paths) == 2, 'Both and only both smoke run artifacts are required')
    results = {}; paired = {}; frozen = []; initial_a = []
    for path in paths:
        report = json.loads(path.read_text()); arm = report.get('arm')
        require(arm in ('direct', 'dual') and arm not in results, 'Missing, duplicate or unknown smoke arm')
        files = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()}
        require(files == {'run.json', 'last.pt'}, 'Smoke contains unexpected validation/excluded/candidate archives')
        checkpoint = path.parent / 'last.pt'
        require(file_hash(checkpoint) == report.get('last_checkpoint_sha256'), 'Last checkpoint byte hash differs')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        initial_path = runner.initial_model_path(args.initial_runs, 812, 1, arm)
        arrays = load_model(initial_path, arm, dimension=3072)
        context = dict(common, arm=arm, seed=812, trial=1, initial_model_sha256=file_hash(initial_path))
        require(all(report.get(k) == v for k, v in context.items()) and saved.get('context') == context, 'Smoke source/data/initialization context differs')
        require(report.get('status') == 'accepted fitting-only LoRA technical smoke' and report.get('role_counts') == [46, 8, 28], 'Smoke status/roles differ')
        require(report.get('case_ids') == {role: [cases[i]['case_id'] for i in indices] for role, indices in [('train', train), ('validation', val), ('excluded', excluded)]}, 'Smoke role identities differ')
        require(report.get('audit_sha256') == file_hash(args.audit / 'audit.json'), 'Input audit binding differs')
        require(report.get('candidates') == saved.get('candidates') == [] and report.get('selected_epoch') == saved.get('epoch') == 3
                and saved.get('best_epoch') is None and saved.get('updates') == 6, 'Smoke performed unexpected selection or updates')
        require(all(report.get(k) is None for k in ('selected_validation', 'selected_fitting'))
                and report.get('outer_predictions') == report.get('smoke_validation_predictions') == 0, 'Smoke contains validation or excluded predictions')
        require(report.get('history') == saved.get('history') and report.get('initial_parity') == saved.get('initial_parity'), 'Run/checkpoint history or probe receipts differ')
        schema = small_state_schema(arrays, arm); check_manifest(report.get('trainable_manifest', {}), schema)
        check_state(saved.get('small_state'), arrays, schema)
        paired[arm], gradients = check_history(report['history'], saved['initial_small_digests'], config, cases, train, lora_io.run_seed(812, 1))
        require(lora_io.small_digests(saved['small_state']) == report['history'][-1]['small_digests'], 'Actual final tensors do not match recorded state digests')
        require(isinstance(saved.get('optimizer'), dict) and isinstance(saved.get('scheduler'), dict)
                and finite_tree(saved['optimizer']) and finite_tree(saved['scheduler']), 'Invalid resumable optimizer/scheduler state')
        groups = saved['optimizer'].get('param_groups', [])
        parameters = [i for group in groups for i in group.get('params', [])]
        require(len(groups) == 1 and len(parameters) == len(schema) - 2
                and len(set(parameters)) == len(parameters), 'Optimizer parameter whitelist size differs')
        require(set(saved['optimizer'].get('state', {})) <= set(parameters), 'Optimizer has state for a forbidden parameter')
        require(saved['scheduler'].get('last_epoch') == 6 and saved['scheduler'].get('T_max') == 240, 'Saved scheduler does not reflect six updates')
        require(saved['frozen_digest_initial'] == report.get('frozen_digest_initial') == report.get('frozen_digest_final')
                and digest(report.get('frozen_digest_initial')), 'Frozen base digest changed or is missing')
        for key in ('elapsed_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
            require(number(report.get(key), True), 'Invalid smoke timing/memory: ' + key)
        require(report['peak_gpu_reserved_bytes'] >= report['peak_gpu_allocated_bytes'], 'GPU reserved/allocated peaks contradict')
        parity = {name: check_parity(report[name], probes, config) for name in ('initial_parity', 'reload_parity')}
        results[arm] = {'run_sha256': file_hash(path), 'last_checkpoint_sha256': file_hash(checkpoint),
                        'epochs': 3, 'optimizer_updates': 6, 'sampled_field_forwards': 192,
                        'elapsed_seconds': report['elapsed_seconds'], 'peak_gpu_allocated_bytes': report['peak_gpu_allocated_bytes'],
                        'peak_gpu_reserved_bytes': report['peak_gpu_reserved_bytes'], 'parity': parity, 'gradient_steps': gradients}
        frozen.append(report['frozen_digest_initial']); initial_a.append(saved['initial_small_digests']['A'])
    require(set(results) == {'direct', 'dual'} and paired['direct'] == paired['dual'], 'Arms do not share case order, field draws and learning rates')
    require(len(set(frozen)) == len(set(initial_a)) == 1, 'Arms do not share identical base tensors and initial A matrices')
    record = {'status': 'accepted both fitting-only LoRA technical smokes', 'code_sha': args.code_sha,
              'review_script_sha256': file_hash(__file__), 'config_sha256': file_hash(args.config),
              'audit_sha256': file_hash(args.audit / 'audit.json'), 'guards_passed': True, 'arms': results,
              'validation_predictions': 0, 'outer_predictions': 0, 'new_model_forwards': 0,
              'limits': 'CPU review verifies archived tensors/receipts; producer raw-image initial/reload probes are not independently re-executed.',
              'public_summary': 'Both fitting-only three-epoch smoke runs passed state, gradient, input and reload-receipt checks. Each used 192 sampled training fields. No validation accuracy was measured by this review.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'guards_passed': True, 'arms': sorted(results)}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('smokes', 'manifest', 'protocol', 'cache', 'config', 'initial-runs', 'audit', 'weights', 'images-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-sha', required=True)
    review(parser.parse_args())


if __name__ == '__main__':
    main()

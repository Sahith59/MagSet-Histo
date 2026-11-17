"""Run the registered case-unit linear/RBF SVC probe on frozen UNI2-h features."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset import matched_data as md
from magset import uni_case_svm as core
from magset.data import file_hash
from scripts.run_specialist_reader import canonical_hash, load_cache

REPO = Path(__file__).resolve().parents[1]
CANDIDATE_FILES = ('model.joblib', 'state.npz', 'validation-probabilities.npz',
                   'validation-records.jsonl', 'metrics.json', 'reload.json')
ARCHIVE_KEYS = ('case_indices', 'masks', 'binary_margin', 'malignant_scores', 'joint_probabilities5', 'predictions')


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def registered_config(path):
    fixed = REPO / 'configs/breakhis-uni-case-svm-v1.json'
    require(file_hash(path) == file_hash(fixed), 'Use exact registered case-SVC configuration')
    cfg = read_json(path)
    require(tuple(cfg['arms']) == core.ARMS and tuple(cfg['svc']['C_values']) == core.CS
            and cfg['primary_arm'] == 'rbf' and cfg['svc']['probability'] is False
            and cfg['svc']['shared_C_across_binary_and_malignant'] is True, 'Registered case-SVC scope differs')
    return cfg


def source_context(args):
    require(re.fullmatch('[0-9a-f]{40}', args.code_sha or '') is not None
            and (REPO / '.source-sha').read_text().strip() == args.code_sha, 'Immutable case-SVC source required')
    archive = (REPO / '.archive-sha256').read_text().strip()
    require(re.fullmatch('[0-9a-f]{64}', archive) is not None, 'Immutable source archive digest required')
    files = ('magset/uni_case_svm.py', 'scripts/run_uni_case_svm.py', 'scripts/collect_uni_case_svm.py', 'scripts/run_specialist_reader.py',
             'magset/matched_data.py', 'magset/bags.py', 'magset/data.py')
    return dict(code_sha=args.code_sha, source_archive_sha256=archive, config_sha256=file_hash(args.config),
                implementation_sha256={name: file_hash(REPO / name) for name in files})


def load_inputs(args):
    cfg = registered_config(args.config); context = source_context(args)
    versions = {name: importlib.metadata.version(name) for name in cfg['runtime']}
    require(versions == cfg['runtime'], 'Pinned case-SVC CPU runtime differs')
    parent = Path(args.parent); audit_dir = parent / 'input-audit'
    reference = cfg['matched_reference']; specialist = cfg['contextual_specialist_reference']
    require(file_hash(parent / 'checked/result.json') == reference['sha256']
            and file_hash(REPO / reference['file']) == reference['sha256'], 'Accepted matched parent reference differs')
    accepted = read_json(parent / 'checked/result.json')
    require(accepted['status'] == 'complete accepted matched three-way comparison'
            and accepted['code_sha'] == reference['producer_code_sha'], 'Parent result is not the accepted matched experiment')
    require(file_hash(REPO / specialist['file']) == specialist['sha256'], 'Fixed contextual specialist reference changed')
    require(file_hash(REPO / reference['config']) == reference['config_sha256']
            and file_hash(REPO / reference['validation_file']) == reference['validation_sha256'], 'Parent configuration/validation reference differs')
    pins = cfg['input_audit']
    require(file_hash(audit_dir / 'private.json') == pins['private_sha256']
            and file_hash(audit_dir / 'indices.npz') == pins['indices_sha256']
            and file_hash(audit_dir / 'public.json') == pins['public_sha256']
            and file_hash(REPO / pins['public_file']) == pins['public_sha256'], 'Parent audit binding changed')
    audit = md.load_audit(audit_dir, REPO / reference['config'])
    require(audit['manifest_sha256'] == cfg['manifest_sha256'], 'Canonical manifest differs')
    for role, counts in cfg['expected_roles'].items():
        require(all(audit['role_summary'][role][key] == value for key, value in counts.items()), 'Registered case/mask role differs')
    refs = cfg['references']; hashes = {}
    for key in ('uni_feature_config', 'uni_extraction_config'):
        require(file_hash(REPO / refs[key]) == refs[key + '_sha256'], 'Frozen UNI reference config differs')
        hashes[refs[key]] = refs[key + '_sha256']
    feature_cfg = read_json(REPO / refs['uni_feature_config']); extraction = read_json(REPO / refs['uni_extraction_config'])
    bags, fields = load_cache(args.uni_cache, feature_cfg, extraction, audit['fields'], audit['cases'], cfg['manifest_sha256'], 1536, False)
    values = core.case_view_means(bags)
    require(values.shape == tuple(cfg['features']['case_view_shape']) and values.dtype == np.float64, 'Case-view means differ')
    train = np.asarray(audit['roles']['train'], np.int64)
    labels = np.asarray([c['truth5'] for c in audit['cases']], np.int64)
    preprocessing = core.fit_preprocessing(values, train)
    alignment = [(f['relative_path'], f['sha256'], audit['cases'][f['case_index']]['case_id'],
                  audit['cases'][f['case_index']]['group'], int(f['magnification']),
                  audit['cases'][f['case_index']]['truth8']) for f in fields]
    context.update(parent_result_sha256=reference['sha256'], specialist_result_sha256=specialist['sha256'],
        input_audit_sha256=pins['private_sha256'], manifest_sha256=cfg['manifest_sha256'],
        reference_config_sha256=hashes, cache_provenance={'uni': feature_cfg['cache_provenance']},
        uni_cache=str(Path(args.uni_cache).resolve()), field_alignment_sha256=canonical_hash(alignment),
        train_case_indices=train.tolist(), malignant_train_case_indices=train[labels[train] > 0].tolist(),
        runtime_versions=versions, feature_dimension=1536, input_dimension=1540, case_view_shape=list(values.shape),
        score_interpretation='uncalibrated normalized scores; authoritative binary margin >= 0')
    return dict(config=cfg, context=context, audit=audit, values=values, labels=labels, preprocessing=preprocessing)


def save_preprocessing(path, state):
    arrays = {'case_view_means': state['values'], 'mean': state['preprocessing']['mean'],
              'std': state['preprocessing']['std'], 'scale': state['preprocessing']['scale'],
              'train_case_indices': state['preprocessing']['train_case_indices'],
              'case_indices': np.arange(len(state['values']), dtype=np.int64), 'masks': core.MASKS.copy()}
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == set(arrays) and all(np.array_equal(loaded[key], value) for key, value in arrays.items()),
                'Preprocessing save/reload differs')


def save_archive(path, archive):
    arrays = {key: archive[key] for key in ARCHIVE_KEYS}
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == set(arrays) and all(np.array_equal(loaded[key], value) for key, value in arrays.items()),
                'Case/mask archive save/reload differs')


def reload_check(bundle, path, state, case_indices):
    original = core.predict(bundle, state['values'], case_indices)
    restored = joblib.load(path)  # Only this run's newly saved, hash-recorded trusted bundle.
    for key in ('mean', 'std', 'scale', 'train_case_indices'):
        require(np.array_equal(bundle[key], restored[key]), 'Trusted model normalizer reload differs')
    before, after = core.state_arrays(bundle), core.state_arrays(restored)
    require(set(before) == set(after) and all(np.array_equal(before[key], after[key]) for key in before), 'SVC state reload differs')
    actual = core.predict(restored, state['values'], case_indices)
    db = float(np.max(np.abs(original['binary_margin'] - actual['binary_margin'])))
    dm = float(np.max(np.abs(original['malignant_scores'] - actual['malignant_scores'])))
    dq = float(np.max(np.abs(original['joint_probabilities5'] - actual['joint_probabilities5'])))
    require(max(db, dm, dq) <= 1e-10 and np.array_equal(original['predictions'], actual['predictions']), 'Validation model reload guard failed')
    return actual, dict(status='accepted validation model reload', binary_margin_max_abs=db,
        malignant_score_max_abs=dm, joint_probability_max_abs=dq, atol=1e-10, decisions_identical=True,
        validation_cases=len(case_indices), masks=15, test_queries=0, model_sha256=file_hash(path))


def verify_gate(output):
    output = Path(output); gate = read_json(output / 'family-gate.json'); context = read_json(output / 'context.json')
    require(gate['status'] == 'sealed both case-SVC selections' and gate['test_queries'] == 0
            and gate['context_sha256'] == file_hash(output / 'context.json')
            and gate['preprocessing_sha256'] == file_hash(output / 'preprocessing.npz')
            and gate['input_audit_sha256'] == context['input_audit_sha256'], 'Case-SVC gate context differs')
    require(set(gate['selections']) == set(core.ARMS), 'Both case-SVC selections must be sealed')
    for arm in core.ARMS:
        path = output / 'arms' / arm / 'selection.json'
        require(file_hash(path) == gate['selections'][arm], 'Case-SVC selection changed after seal')
        selection = read_json(path); scores = selection['candidate_validation_scores']; index = selection['selected_index']
        require(selection['status'] == 'selected validation candidate' and selection['arm'] == arm
                and selection['test_predictions'] == 0 and len(scores) == 4 and np.isfinite(scores).all()
                and type(index) is int and index == int(np.argmax(scores)) and selection['selected_setting'] == core.CS[index]
                and selection['context_sha256'] == gate['context_sha256']
                and selection['preprocessing_sha256'] == gate['preprocessing_sha256'], 'Case-SVC validation selection differs')
        require(len(selection['candidate_artifact_sha256']) == 4, 'Four case-SVC candidate receipts required')
        for candidate, hashes in enumerate(selection['candidate_artifact_sha256']):
            folder = output / 'candidates' / arm / ('candidate%d' % candidate)
            require(set(hashes) == set(CANDIDATE_FILES) and all(file_hash(folder / name) == digest for name, digest in hashes.items()),
                    'Case-SVC candidate artifacts changed after seal')
            metric = read_json(folder / 'metrics.json'); reload = read_json(folder / 'reload.json')
            require(metric['setting'] == core.CS[candidate] and metric['validation']['mean_subset_macro_f1'] == scores[candidate]
                    and reload['status'] == 'accepted validation model reload' and reload['decisions_identical'] is True
                    and max(reload['binary_margin_max_abs'], reload['malignant_score_max_abs'], reload['joint_probability_max_abs']) <= 1e-10,
                    'Case-SVC candidate selection/reload receipt differs')
    return gate


def run(args):
    output = Path(args.output)
    require(not output.exists(), 'Refusing to overwrite case-SVC study')
    state = load_inputs(args); output.mkdir(parents=True)
    save_preprocessing(output / 'preprocessing.npz', state)
    context = dict(state['context'], preprocessing_sha256=file_hash(output / 'preprocessing.npz'))
    md.write_json(output / 'context.json', context); context_sha = file_hash(output / 'context.json')
    validation = np.asarray(state['audit']['roles']['validation'], np.int64)
    selections = {}; validation_records = 0; fits = 0
    for arm in core.ARMS:
        scores = []; receipts = []
        for candidate, C in enumerate(core.CS):
            folder = output / 'candidates' / arm / ('candidate%d' % candidate); folder.mkdir(parents=True)
            bundle = core.fit_bundle(state['values'], state['labels'], state['preprocessing'], arm, C); fits += 2
            joblib.dump(bundle, folder / 'model.joblib', compress=3)
            np.savez_compressed(folder / 'state.npz', **core.state_arrays(bundle))
            archive, reload = reload_check(bundle, folder / 'model.joblib', state, validation)
            rows = core.case_records(state['audit']['cases'], archive, 'validation'); metric = core.summary(state['audit']['cases'], archive)
            save_archive(folder / 'validation-probabilities.npz', archive); md.save_records(folder / 'validation-records.jsonl', rows)
            md.write_json(folder / 'metrics.json', {'setting': C, 'validation': metric})
            reload['state_sha256'] = file_hash(folder / 'state.npz'); md.write_json(folder / 'reload.json', reload)
            scores.append(float(metric['mean_subset_macro_f1'])); validation_records += len(rows)
            receipts.append({name: file_hash(folder / name) for name in CANDIDATE_FILES})
            print(json.dumps({'arm': arm, 'C': C, 'validation_macro_f1': scores[-1],
                              'binary_gamma': bundle['binary']._gamma, 'malignant_gamma': bundle['malignant']._gamma}), flush=True)
        require(np.isfinite(scores).all(), 'Nonfinite case-SVC validation scores')
        index = int(np.argmax(scores)); selection_folder = output / 'arms' / arm; selection_folder.mkdir(parents=True)
        selected = dict(context, status='selected validation candidate', arm=arm, context_sha256=context_sha,
            preprocessing_sha256=file_hash(output / 'preprocessing.npz'), selected_index=index, selected_setting=core.CS[index],
            candidate_validation_scores=scores, candidate_artifact_sha256=receipts, test_predictions=0)
        md.write_json(selection_folder / 'selection.json', selected); selections[arm] = selected
    require(fits == 16 and validation_records == state['config']['counts']['validation_case_subset_records'], 'Case-SVC fitting/validation budget differs')
    gate = dict(status='sealed both case-SVC selections', selections={arm: file_hash(output / 'arms' / arm / 'selection.json') for arm in core.ARMS},
        context_sha256=context_sha, preprocessing_sha256=file_hash(output / 'preprocessing.npz'),
        input_audit_sha256=context['input_audit_sha256'], test_queries=0)
    md.write_json(output / 'family-gate.json', gate); verify_gate(output)
    test_indices = np.asarray(state['audit']['roles']['test'], np.int64); test_records = 0
    for arm in core.ARMS:
        verify_gate(output)
        selected = selections[arm]; candidate = output / 'candidates' / arm / ('candidate%d' % selected['selected_index'])
        require(file_hash(candidate / 'model.joblib') == selected['candidate_artifact_sha256'][selected['selected_index']]['model.joblib'], 'Selected trusted model changed')
        bundle = joblib.load(candidate / 'model.joblib')
        archive = core.predict(bundle, state['values'], test_indices)
        rows = core.case_records(state['audit']['cases'], archive, 'test'); metric = core.summary(state['audit']['cases'], archive)
        folder = output / 'arms' / arm / 'test'; folder.mkdir()
        save_archive(folder / 'probabilities.npz', archive); md.save_records(folder / 'records.jsonl', rows); md.write_json(folder / 'metrics.json', metric)
        value = dict(status='complete selected retrospective inference', arm=arm, selected_index=selected['selected_index'],
            selected_setting=selected['selected_setting'], context_sha256=context_sha, preprocessing_sha256=file_hash(output / 'preprocessing.npz'),
            selection_sha256=file_hash(output / 'arms' / arm / 'selection.json'), gate_sha256=file_hash(output / 'family-gate.json'),
            selected_model_sha256=file_hash(candidate / 'model.joblib'), selected_state_sha256=file_hash(candidate / 'state.npz'),
            artifact_sha256={name: file_hash(folder / name) for name in ('probabilities.npz', 'records.jsonl', 'metrics.json')})
        md.write_json(folder / 'test.json', value); test_records += len(rows)
    require(test_records == state['config']['counts']['retrospective_case_subset_records'], 'Case-SVC selected follow-up budget differs')
    result = dict(status='complete case-SVC family', context_sha256=context_sha, preprocessing_sha256=file_hash(output / 'preprocessing.npz'),
        gate_sha256=file_hash(output / 'family-gate.json'), head_estimator_fits=fits, binary_optimization_subproblems=56,
        validation_case_subset_records=validation_records, retrospective_case_subset_records=test_records,
        validation_prediction_archives=8, selected_retrospective_prediction_archives=2,
        new_encoder_forwards=0, new_encoder_updates=0, new_image_decodes=0,
        artifact_sha256={str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob('*'))
                         if path.is_file() and path.name != 'result.json'})
    md.write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'uni-cache', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-sha', required=True); args = parser.parse_args(); existed = args.output.exists()
    try:
        value = run(args)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output / 'result.json').exists():
            md.write_json(args.output / 'failure.json', {'status': 'failed case-SVC execution',
                'error_type': type(error).__name__, 'message': str(error)})
        raise
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()

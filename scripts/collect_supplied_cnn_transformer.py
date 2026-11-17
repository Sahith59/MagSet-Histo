"""Read-only CPU acceptance of both supplied-notebook variants; never imports TF."""
import argparse
import csv
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.five_collect import assert_public_safe, read_json, require_equal
from magset.five_class import write_json

NOTEBOOK_SHA = '2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709'
IMAGENET_SHA = 'fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285'
VARIANTS = ('faithful', 'input_fixed')
SUBTYPES = ('adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma',
            'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma')
MALIGNANT = frozenset(('ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer_array(value, shape):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'iu', 'Integer array shape/dtype differs')
    return value


def source_context(args):
    root = Path(__file__).resolve().parents[1]
    require(re.fullmatch('[0-9a-f]{40}', args.code_sha or '') is not None, 'Full source SHA required')
    require((root/'.source-sha').read_text().strip() == args.code_sha, 'Immutable source receipt differs')
    archive = (root/'.archive-sha256').read_text().strip()
    require(re.fullmatch('[0-9a-f]{64}', archive) is not None, 'Source archive digest missing')
    require(file_hash(args.config) == file_hash(root/'configs/supplied-cnn-transformer-v1.json'), 'Canonical configuration differs')
    require(file_hash(args.notebook) == NOTEBOOK_SHA, 'Exact original notebook differs')
    config = read_json(args.config)
    require(config.get('name') == 'supplied-cnn-transformer-v1' and config.get('variants') == list(VARIANTS)
            and config.get('notebook_sha256') == NOTEBOOK_SHA, 'Registered variant family differs')
    require(config['runtime'].get('imagenet_weights_sha256') == IMAGENET_SHA, 'Registered official ImageNet weight pin differs')
    original = read_json(args.notebook)
    require(len(original.get('cells', [])) == 15 and all(c['cell_type'] == 'code' and not c.get('outputs')
            for c in original['cells']), 'Original notebook is not the unexecuted fifteen-cell source')
    context = {'code_sha': args.code_sha, 'source_archive_sha256': archive,
               'adapter_sha256': file_hash(root/'scripts/run_supplied_cnn_transformer.py'),
               'collector_sha256': file_hash(__file__), 'config_sha256': file_hash(args.config),
               'notebook_sha256': NOTEBOOK_SHA}
    return context, config, original


def accepted_audit(args, context, config):
    root = Path(args.audit); private = read_json(root/'private.json'); public = read_json(root/'public.json')
    require_equal(public['private_audit_sha256'], file_hash(root/'private.json'), 'private shared audit bytes', 0)
    for key in ('code_sha', 'source_archive_sha256', 'adapter_sha256', 'config_sha256', 'notebook_sha256'):
        require_equal(private.get(key), context[key], 'shared audit/'+key, 0)
    require_equal(private.get('manifest_sha256'), config['dataset']['canonical_manifest_sha256'], 'canonical manifest', 0)
    require_equal(private.get('binary_classes'), ['benign', 'malignant'], 'binary class order', 0)
    require_equal(private.get('subtype_classes'), list(SUBTYPES), 'subtype class order', 0)
    for name, key in [('ordered.csv', 'ordered_csv_sha256'), ('indices.npz', 'indices_sha256')]:
        require_equal(file_hash(root/name), private[key], 'shared audit artifact/'+name, 0)
    rows = private['rows']; require(len(rows) == 7909, 'Expected exactly 7909 ordered images')
    with np.load(root/'indices.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'train_indices', 'test_indices', 'y_binary', 'y_subtype'}, 'Shared split archive schema differs')
        arrays = {key: saved[key].copy() for key in saved.files}
    train = integer_array(arrays['train_indices'], (6327,)); test = integer_array(arrays['test_indices'], (1582,))
    yb = integer_array(arrays['y_binary'], (7909,)); ys = integer_array(arrays['y_subtype'], (7909,))
    require(set(train) | set(test) == set(range(7909)) and not set(train) & set(test)
            and len(set(train)) == 6327 and len(set(test)) == 1582, 'Shared image roles overlap or lack coverage')
    for index, row in enumerate(rows):
        require(row['index'] == index and row['subclass_label'] in SUBTYPES, 'Ordered row identity/class differs')
        require(ys[index] == SUBTYPES.index(row['subclass_label']) and yb[index] == int(row['subclass_label'] in MALIGNANT)
                and row['class_label'] == ('malignant' if yb[index] else 'benign'), 'Ordered row truth differs')
        require(re.fullmatch('[0-9a-f]{64}', row['image_sha256']) is not None, 'Original image digest missing')
    expected_train, expected_test = train_test_split(np.arange(7909), test_size=.2, random_state=42, stratify=ys)
    require(np.array_equal(train, expected_train) and np.array_equal(test, expected_test), 'Original stratified image split differs')
    with (root/'ordered.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == len(rows), 'Ordered CSV row count differs')
    for actual, expected in zip(csv_rows, rows):
        require_equal(actual, {key: str(value) for key, value in expected.items()}, 'ordered CSV row', 0)
    overlap = {}
    for key in ('provider_key', 'conservative_group'):
        fitting = {rows[i][key] for i in train}; validation = {rows[i][key] for i in test}
        overlap[key] = {'fitting_keys': len(fitting), 'validation_keys': len(validation), 'overlap_keys': len(fitting & validation)}
    require_equal(private['provider_overlap'], overlap, 'provider overlap accounting', 0)
    return private, arrays


def classification(labels, predictions, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    truth = matrix.sum(1); predicted = matrix.sum(0); diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(diagonal, truth, out=np.zeros(classes), where=truth > 0)
    f1 = np.divide(2*diagonal, truth+predicted, out=np.zeros(classes), where=truth+predicted > 0)
    weights = (np.arange(classes)[:, None]-np.arange(classes)[None, :])**2
    expected = np.outer(truth, predicted)/len(labels)
    denominator = float(np.sum(weights*expected))
    return {'records': len(labels), 'correct': int(diagonal.sum()), 'accuracy': float(diagonal.sum()/len(labels)),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'quadratic_kappa': 1.-float(np.sum(weights*matrix))/denominator if denominator else None,
            'confusion_matrix': matrix.tolist(), 'class_support': truth.tolist(), 'class_recall': recall.tolist()}


def prediction_metrics(arrays, split):
    train = integer_array(arrays['train_indices'], (6327,)); test = integer_array(arrays['test_indices'], (1582,))
    require(np.array_equal(train, split['train_indices']) and np.array_equal(test, split['test_indices']), 'Variant changed shared image indices/order')
    yb = integer_array(arrays['truth_binary'], (1582,)); ys = integer_array(arrays['truth_subtype'], (1582,))
    require(np.array_equal(yb, split['y_binary'][test]) and np.array_equal(ys, split['y_subtype'][test]), 'Variant prediction truths/order differ')
    for alias, expected in [('yb_test', yb), ('ys_test', ys)]:
        require(np.array_equal(integer_array(arrays[alias], (1582,)), expected), 'Original truth aliases differ')
    scores = {}
    for name, classes in [('binary_scores', 2), ('subtype_scores', 8)]:
        value = np.asarray(arrays[name])
        require(value.shape == (1582, classes) and value.dtype.kind == 'f' and np.isfinite(value).all()
                and np.all(value >= 0) and np.all(value <= 1) and np.allclose(value.sum(1), 1, rtol=0, atol=1e-5), 'Invalid saved softmax scores')
        scores[name] = value
    bp = scores['binary_scores'].argmax(1); sp = scores['subtype_scores'].argmax(1)
    for name, expected in [('yb_pred', bp), ('ys_pred', sp)]:
        require(np.array_equal(integer_array(arrays[name], (1582,)), expected), 'Original argmax prediction differs')
    binary = classification(yb, bp, 2); eight = classification(ys, sp, 8)
    binary['roc_auc'] = float(roc_auc_score(yb, scores['binary_scores'][:, 1]))
    eight['macro_ovr_roc_auc'] = float(roc_auc_score(ys, scores['subtype_scores'], multi_class='ovr', average='macro'))
    malignant = yb == 1; denominator = int(malignant.sum()); ungated = int(np.sum(malignant & (sp == ys)))
    strict = int(np.sum(malignant & (bp == 1) & (sp == ys)))
    joint = int(np.sum((bp == yb) & (sp == ys)))
    scalar = {'binary_accuracy': binary['accuracy'], 'binary_malignant_recall': binary['class_recall'][1],
              'binary_benign_recall': binary['class_recall'][0], 'binary_auc': binary['roc_auc'],
              'eight_class_accuracy': eight['accuracy'], 'eight_class_macro_precision': eight['macro_precision'],
              'eight_class_macro_recall': eight['macro_recall'], 'eight_class_macro_f1': eight['macro_f1'],
              'eight_class_quadratic_kappa': eight['quadratic_kappa'], 'eight_class_macro_ovr_auc': eight['macro_ovr_roc_auc'],
              'ungated_true_malignant_subtype_accuracy': ungated/denominator,
              'strict_true_malignant_cascade_recall': strict/denominator,
              'strict_cascade_correct_malignant_images': strict, 'true_malignant_images': denominator}
    return scalar, {'binary': binary, 'eight_class': eight,
                    'ungated_true_malignant': {'correct': ungated, 'records': denominator, 'accuracy': ungated/denominator},
                    'strict_malignant_cascade': {'correct': strict, 'records': denominator, 'recall': strict/denominator},
                    'strict_joint_eight_head_correctness': {'correct': joint, 'records': 1582, 'accuracy': joint/1582}}


def review_variant(args, variant, context, config, original, audit, split):
    root = Path(args.runs)/variant
    result_path = root/'result.json'; metadata_path = root/'metadata.json'
    if not result_path.exists():
        return {'status': 'incomplete', 'reason': 'result_artifact_missing',
                'partial_notebook_preserved': (root/'executed.ipynb').exists()}
    value = read_json(result_path)
    require_equal(value, read_json(metadata_path), 'result/metadata aliases', 0)
    for key in ('code_sha', 'source_archive_sha256', 'adapter_sha256', 'config_sha256', 'notebook_sha256'):
        require_equal(value.get(key), context[key], 'variant source/'+key, 0)
    require(value.get('variant') == variant and value.get('action') == 'run', 'Variant/action identity differs')
    require_equal(value.get('audit_sha256'), file_hash(Path(args.audit)/'private.json'), 'shared audit binding', 0)
    if value.get('status') != 'complete notebook execution':
        require(value.get('status') in ('failed notebook execution', 'notebook execution started'), 'Unknown execution status')
        return {'status': 'failed' if value['status'] == 'failed notebook execution' else 'incomplete',
                'reason': 'notebook_not_completed', 'result_sha256': file_hash(result_path),
                'partial_notebook_preserved': (root/'executed.ipynb').exists(),
                'failure_receipt_preserved': (root/'execution_error.json').exists()}
    hashes = value['artifact_sha256']
    required = {'adapted.ipynb', 'executed.ipynb', 'history.json', 'metrics.json', 'predictions.npz', 'environment.json',
                'prefit.json', 'prefit_split.npz', 'pretrained_weights.json',
                'cnn_transformer_dualhd.keras', 'breakhis_dataset.csv', 'progress.csv'}
    require(required <= set(hashes), 'Completed run lacks required artifacts')
    for name, digest in hashes.items():
        require(Path(name).name == name and re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Invalid artifact receipt')
        require_equal(file_hash(root/name), digest, 'variant artifact/'+name, 0)
    require_equal(value['adapted_notebook_sha256'], hashes['adapted.ipynb'], 'preexecution notebook digest', 0)
    adapted = read_json(root/'adapted.ipynb'); executed = read_json(root/'executed.ipynb')
    sources = [''.join(cell['source']) for cell in adapted['cells']]
    require(len(sources) == 15 and all(c['cell_type'] == 'code' for c in adapted['cells']), 'Adapted real notebook cell count differs')
    require_equal([''.join(cell['source']) for cell in executed['cells']], sources, 'executed notebook sources', 0)
    require_equal([hashlib.sha256(source.encode()).hexdigest() for source in sources], value['executed_cell_source_sha256'], 'executed cell source digests', 0)
    require(all(cell.get('execution_count') is not None and not any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
                for cell in executed['cells']), 'Completed notebook has unexecuted/error cells')
    # Pure source construction only: the adapter is never executed as a notebook here.
    adapter_path = Path(__file__).with_name('run_supplied_cnn_transformer.py')
    spec = importlib.util.spec_from_file_location('_supplied_source_only', adapter_path)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    base_sources = adapter.patched_sources(original, variant, audit['dataset_root'],
                                           value.get('execution_root', str(root.resolve())))
    expected = list(base_sources)
    expected[6] += adapter.WEIGHT_GUARD
    expected[0] += adapter.runtime_preamble(True)
    expected[3] += adapter.prefit_guard(value.get('audit_root', str(Path(args.audit).resolve())))
    expected[7] += "\ncallbacks.append(tf.keras.callbacks.CSVLogger('progress.csv',append=False))\n"
    expected[8] += "\n_Path('history.json').write_text(_json.dumps({k:[float(x)for x in v]for k,v in hist.history.items()},indent=2))\n"
    expected[11] += adapter.EXPORT_METRICS
    patches = [{'original_cell': i,
                'original_sha256': hashlib.sha256(''.join(cell['source']).encode()).hexdigest(),
                'patched_sha256': hashlib.sha256(base_sources[i].encode()).hexdigest()}
               for i, cell in enumerate(original['cells'])]
    require_equal(expected, sources, 'frozen adapter source recipe', 0)
    require_equal(value['cell_patch_receipts'], patches, 'original/compatibility patch receipts', 0)
    with (root/'breakhis_dataset.csv').open(newline='') as handle:
        actual_rows = list(csv.DictReader(handle))
    require_equal(actual_rows, [{k: row[k] for k in ('image_path', 'class_label', 'subclass_label')} for row in audit['rows']], 'actual notebook dataframe order', 0)
    prefit = read_json(root/'prefit.json')
    require(prefit.get('status') == 'accepted actual notebook order, labels and split before fitting', 'Prefit guard did not pass')
    require(prefit.get('train_images') == 6327 and prefit.get('validation_images') == 1582, 'Prefit image counts differ')
    for key in ('ordered_csv_sha256', 'indices_sha256'):
        require_equal(prefit[key], audit[key], 'prefit shared data/'+key, 0)
    with np.load(root/'prefit_split.npz', allow_pickle=False) as saved:
        expected_arrays = {'train_indices': split['train_indices'], 'test_indices': split['test_indices'],
            'yb_train': split['y_binary'][split['train_indices']], 'yb_test': split['y_binary'][split['test_indices']],
            'ys_train': split['y_subtype'][split['train_indices']], 'ys_test': split['y_subtype'][split['test_indices']]}
        require(set(saved.files) == set(expected_arrays) and all(np.array_equal(saved[k], a) for k, a in expected_arrays.items()), 'Actual prefit indices/truths differ')
    environment = read_json(root/'environment.json')
    require(environment.get('tensorflow') == config['runtime']['tensorflow'] and environment.get('keras') == config['runtime']['keras']
            and type(environment.get('gpu_count')) is int and environment['gpu_count'] >= 1, 'Pinned GPU runtime differs')
    history = read_json(root/'history.json'); completed = len(history['loss'])
    require(1 <= completed <= 50 and all(isinstance(v, list) and len(v) == completed and np.isfinite(v).all() for v in history.values()), 'Incomplete/nonfinite epoch history')
    validation = np.asarray(history['val_subtype_output_accuracy']); best = int(validation.argmax())+1
    require(np.all((validation >= 0) & (validation <= 1)) and (completed == 50 or completed-best == 12), 'Original epoch/early-stop budget differs')
    metrics = read_json(root/'metrics.json')
    for obj in (value, metrics):
        require_equal(obj['selected_epoch_first_max_validation_eight_class_accuracy'], best, 'earliest best epoch', 0)
    require_equal(metrics['epochs_completed'], completed, 'completed epochs', 0)
    with np.load(root/'predictions.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'binary_scores', 'subtype_scores', 'yb_test', 'ys_test', 'truth_binary', 'truth_subtype',
                                   'yb_pred', 'ys_pred', 'test_indices', 'train_indices'}, 'Saved prediction schema differs')
        scalar, numbers = prediction_metrics({k: saved[k].copy() for k in saved.files}, split)
    for key, expected in scalar.items():
        require_equal(metrics.get(key), expected, 'independent prediction metric/'+key, 1e-10)
    require_equal(scalar['eight_class_accuracy'], float(validation[best-1]), 'selected checkpoint/full held-image accuracy', 1e-6)
    reload = metrics['checkpoint_full_model_reload']
    for key, expected in {'images': 32, 'batch_size': 32, 'atol': 1e-5, 'checkpoint_sha256': hashes['cnn_transformer_dualhd.keras']}.items():
        require_equal(reload[key], expected, 'full checkpoint reload/'+key, 0)
    require(all(isinstance(reload[k], (int, float)) and math.isfinite(reload[k]) and 0 <= reload[k] <= 1e-5
                for k in ('binary_max_abs', 'subtype_max_abs')), 'Full checkpoint reload parity failed')
    weights = value.get('keras_cached_weight_sha256')
    require(isinstance(weights, dict) and weights and all(Path(k).name == k and re.fullmatch('[0-9a-f]{64}', v) for k, v in weights.items()), 'Pretrained weight receipt missing')
    require('efficientnetv2-b0_notop.h5' in weights, 'Official EfficientNetV2B0 backbone weight receipt missing')
    require_equal(weights['efficientnetv2-b0_notop.h5'], IMAGENET_SHA, 'registered official ImageNet weight bytes', 0)
    before_fit_weights = read_json(root/'pretrained_weights.json')
    for key, expected in {'filename': 'efficientnetv2-b0_notop.h5', 'sha256': IMAGENET_SHA, 'bytes': 24274472}.items():
        require_equal(before_fit_weights.get(key), expected, 'pre-fitting official weight receipt/'+key, 0)
    return {'status': 'accepted', 'result_sha256': file_hash(result_path), 'artifacts': hashes,
            'epochs_completed': completed, 'selected_epoch': best, 'environment': environment,
            'keras_cached_weight_sha256': weights, 'checkpoint_reload': reload, 'metrics': numbers,
            'original_code_metrics': scalar}


def collect(args):
    output = Path(args.output); require(not output.exists(), 'Refusing existing collector output')
    private = {'errors': {}}; result = {'status': 'incomplete supplied-notebook two-variant comparison',
        'variants': {}, 'collector_new_fits': 0, 'collector_new_inference': 0,
        'interpretation': 'eight-subtype single-image reused image-random validation; original and strict cascade metrics are separate; not provider-case five-class performance or a fresh clinical test'}
    try:
        context, config, original = source_context(args); result.update(context)
        audit, split = accepted_audit(args, context, config)
        result['shared_audit'] = {'private_audit_sha256': file_hash(Path(args.audit)/'private.json'),
            'indices_sha256': audit['indices_sha256'], 'ordered_csv_sha256': audit['ordered_csv_sha256'],
            'manifest_sha256': audit['manifest_sha256'], 'images': 7909, 'fitting_images': 6327,
            'reused_validation_images': 1582, 'provider_overlap': audit['provider_overlap']}
        for variant in VARIANTS:
            try:
                result['variants'][variant] = review_variant(args, variant, context, config, original, audit, split)
            except Exception as error:
                result['variants'][variant] = {'status': 'invalid_output', 'error_type': type(error).__name__}
                private['errors'][variant] = {'message': str(error), 'traceback': traceback.format_exc()}
        if all(result['variants'][v]['status'] == 'accepted' for v in VARIANTS):
            faithful, fixed = (result['variants'][v] for v in VARIANTS)
            require_equal(faithful['keras_cached_weight_sha256'], fixed['keras_cached_weight_sha256'], 'same pretrained input weights', 0)
            for key in ('python', 'tensorflow', 'keras', 'numpy', 'scikit-learn'):
                require_equal(faithful['environment'][key], fixed['environment'][key], 'matched variant environment/'+key, 0)
            result['status'] = 'complete accepted supplied-notebook two-variant comparison'
            result['input_fixed_minus_faithful'] = {key: fixed['original_code_metrics'][key]-faithful['original_code_metrics'][key]
                for key in ('binary_accuracy', 'eight_class_accuracy', 'eight_class_macro_f1', 'strict_true_malignant_cascade_recall')}
    except Exception as error:
        private['errors']['shared_or_family'] = {'message': str(error), 'traceback': traceback.format_exc()}
        result['status'] = 'failed or incomplete supplied-notebook two-variant comparison'
        result['shared_or_family_review'] = {'status': 'invalid', 'error_type': type(error).__name__}
        for variant in VARIANTS:
            result['variants'].setdefault(variant, {'status': 'unverified', 'reason': 'shared_input_or_family_review_failed'})
    if result['status'].startswith('incomplete') and any(v['status'] in ('failed', 'invalid_output', 'unverified') for v in result['variants'].values()):
        result['status'] = 'failed or incomplete supplied-notebook two-variant comparison'
    assert_public_safe(result); output.mkdir(parents=True)
    write_json(output/'private_review.json', private)
    result['private_review_sha256'] = file_hash(output/'private_review.json')
    write_json(output/'comparison.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'runs', 'notebook', 'config', 'output'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--code-sha', required=True)
    return result


if __name__ == '__main__':
    print(collect(parser().parse_args())['status'])

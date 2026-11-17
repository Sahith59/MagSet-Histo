"""Independently replay saved supplied-Swin outputs on CPU; never fit or infer."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.five_collect import assert_public_safe, read_json, require_equal
from magset.five_class import write_json

NOTEBOOK_SHA = 'fd88d851fbd8cf049a0ebbd718ed8757ffe28f629e2472262efca30bbca029af'
MALIGNANT = ('DC', 'LC', 'MC', 'PC')
SUBTYPES = ('A', 'F', 'PT', 'TA', *MALIGNANT)
TEST_N = {'DC': 10, 'LC': 2, 'MC': 2, 'PC': 2, 'A': 1, 'F': 3, 'PT': 1, 'TA': 2}
VAL_N = {'DC': 3, 'LC': 0, 'MC': 1, 'PC': 1, 'A': 0, 'F': 1, 'PT': 0, 'TA': 1}
ROLES = ('train', 'val', 'test')
REQUIRED_ARTIFACTS = frozenset((
    'predictions.npz', 'metrics.json', 'history.json', 'prefit.json', 'environment.json',
    'pretrained_weights.json', 'checkpoint_reload.json', 'executed.ipynb', 'adapted.ipynb',
    'progress.csv', 'pretrained_load.json', 'window_reverse_check.json', 'swin_dual_split0.weights.h5'))
PREDICTION_KEYS = frozenset(('test_indices', 'y5_true', 'y5_pred', 'p_mal',
                            'binary_output', 'subtype_output'))
CONTEXT_KEYS = ('code_sha', 'source_archive_sha256', 'adapter_sha256', 'config_sha256', 'notebook_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer_array(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'iu', name+': integer shape/dtype differs')
    return value


def probabilities(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind == 'f' and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1)), name+': invalid probability shape/value')
    if value.ndim == 2:
        require(np.allclose(value.sum(1), 1, rtol=0, atol=1e-5), name+': probabilities are not normalized')
    return value


def reconstruct_split(table, test_n=TEST_N, val_n=VAL_N, seed=0):
    """Rebuild rarity assignment and the original ordered RandomState draws."""
    subtype_patients = {sub: set(table.loc[table.subtype == sub, 'patient']) for sub in set(table.subtype)}
    require(set(subtype_patients) == set(test_n) == set(val_n), 'Split subtype coverage differs')
    # dict preserves the subtype first-occurrence order used by pandas.unique;
    # this also preserves the notebook's min() tie behavior.
    memberships = {}
    for patient, subtype in zip(table.patient, table.subtype):
        memberships.setdefault(patient, {})[subtype] = None
    primary = {patient: min(subs, key=lambda s: len(subtype_patients[s]))
               for patient, subs in memberships.items()}
    rng = np.random.RandomState(seed)
    roles = {}
    for subtype in sorted(test_n):
        patients = sorted(p for p, sub in primary.items() if sub == subtype)
        rng.shuffle(patients)
        nt, nv = test_n[subtype], val_n[subtype]
        require(len(patients)-nt-nv >= 1, 'Insufficient patients for registered subtype split')
        for index, patient in enumerate(patients):
            roles[patient] = 'test' if index < nt else 'val' if index < nt+nv else 'train'
    labels = np.asarray([roles[p] for p in table.patient])
    return {role: np.flatnonzero(labels == role) for role in ROLES}, primary


def overlap_counts(table):
    public, private = {}, {}
    for key in ('patient', 'provider_key', 'conservative_group'):
        sets = {role: set(table.loc[table.split == role, key]) for role in ROLES}
        pairs = {(a+'_'+b): sorted(sets[a] & sets[b]) for a, b in
                 (('train', 'val'), ('train', 'test'), ('val', 'test'))}
        public[key] = {'role_keys': {role: len(sets[role]) for role in ROLES},
                       'overlap_keys': {pair: len(keys) for pair, keys in pairs.items()}}
        private[key] = pairs
    return public, private


def source_context(args):
    root = Path(__file__).resolve().parents[1]
    require(re.fullmatch('[0-9a-f]{40}', args.code_sha or '') is not None, 'Full source SHA required')
    require((root/'.source-sha').read_text().strip() == args.code_sha, 'Immutable source receipt differs')
    archive = (root/'.archive-sha256').read_text().strip()
    require(re.fullmatch('[0-9a-f]{64}', archive) is not None, 'Source archive digest missing')
    require(file_hash(args.config) == file_hash(root/'configs/supplied-swin-v1.json'), 'Canonical configuration differs')
    config = read_json(args.config)
    require(config.get('name') == 'supplied-swin-v1' and config.get('mode') == 'dual'
            and config.get('notebook_sha256') == NOTEBOOK_SHA, 'Registered notebook family differs')
    require_equal(config['split']['test_counts'], TEST_N, 'test quotas', 0)
    require_equal(config['split']['validation_counts'], VAL_N, 'validation quotas', 0)
    require(config['split']['seed'] == 0 and config['training']['maximum_epochs'] == 40
            and config['training']['early_stopping_patience'] == 10, 'Original bounded split/training settings differ')
    return {'code_sha': args.code_sha, 'source_archive_sha256': archive,
            'adapter_sha256': file_hash(root/'scripts/run_supplied_swin.py'),
            'collector_sha256': file_hash(__file__), 'config_sha256': file_hash(args.config),
            'notebook_sha256': NOTEBOOK_SHA}, config


def accepted_audit(args, context, config):
    root = Path(args.audit)
    private = read_json(root/'private.json')
    public = read_json(root/'public.json')
    require_equal(public.get('private_audit_sha256'), file_hash(root/'private.json'), 'private audit binding', 0)
    require(private.get('status') == 'accepted exact original Swin input and split audit', 'Input audit did not pass')
    for key in CONTEXT_KEYS:
        require_equal(private.get(key), context[key], 'audit/'+key, 0)
    require_equal(private.get('manifest_sha256'), config['dataset']['canonical_manifest_sha256'], 'manifest revision', 0)
    for name, key in [('ordered.csv', 'ordered_csv_sha256'), ('indices.npz', 'indices_sha256')]:
        require_equal(file_hash(root/name), private[key], 'audit artifact/'+name, 0)
    with (root/'ordered.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        required_columns = {'image_path', 'binary', 'subtype', 'patient', 'mag', 'split',
                            'provider_key', 'conservative_group', 'image_sha256'}
        require(required_columns <= set(reader.fieldnames or []), 'Ordered dataframe columns differ')
        rows = list(reader)
    require(len(rows) == config['dataset']['images'] == 7909, 'Expected exactly 7909 original images')
    if 'rows' in private:
        require_equal(rows, [{key: str(value) for key, value in row.items()} for row in private['rows']], 'audit/CSV ordered rows', 0)
    table = pd.DataFrame(rows)
    require(table.image_path.tolist() == sorted(table.image_path) and table.image_path.is_unique,
            'Original dataframe path order/uniqueness differs')
    for row in rows:
        match = re.fullmatch(r'SOB_([BM])_([A-Z]+)-([^-]+)-([^-]+)-(40|100|200|400)-([^-]+)\.png',
                             Path(row['image_path']).name)
        require(match is not None, 'Original image filename grammar differs')
        broad, subtype, first, second, mag, _ = match.groups()
        require(subtype in SUBTYPES and row['subtype'] == subtype
                and row['binary'] == ('malignant' if broad == 'M' else 'benign')
                and (broad == 'M') == (subtype in MALIGNANT)
                and row['patient'] == first+'-'+second and row['mag'] == mag,
                'Original parsed dataframe label/ID/magnification differs')
        require(row['provider_key'] and row['conservative_group']
                and re.fullmatch('[0-9a-f]{64}', row['image_sha256']) is not None,
                'Manifest-bound identity/digest missing')
    expected_split, primary = reconstruct_split(table)
    truth5 = np.asarray([1+MALIGNANT.index(s) if s in MALIGNANT else 0 for s in table.subtype])
    expected = {role+'_indices': expected_split[role] for role in ROLES}
    expected.update(y5=truth5, y_binary=(truth5 > 0).astype(int), y_malignant=truth5-1)
    with np.load(root/'indices.npz', allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), 'Split archive schema differs')
        arrays = {key: saved[key].copy() for key in saved.files}
    for key, wanted in expected.items():
        actual = integer_array(arrays[key], wanted.shape, key)
        require(np.array_equal(actual, wanted), 'Independent original split/truth differs: '+key)
    for role in ROLES:
        indices = expected_split[role]
        require(len(indices) == config['split']['expected_images'][role]
                and table.iloc[indices].patient.nunique() == config['split']['expected_bare_ids'][role],
                'Registered split counts differ')
        require(np.array_equal(np.flatnonzero(table.split.values == role), indices), 'Ordered CSV split differs')
    require(set(table.split) == set(ROLES) and set(table.mag) == {'40', '100', '200', '400'},
            'Original split/magnification coverage differs')
    overlap, overlap_private = overlap_counts(table)
    require_equal(private.get('overlap'), {key: item['overlap_keys'] for key, item in overlap.items()}, 'audited overlap accounting', 0)
    require_equal(private.get('split_images'), {s: len(expected_split[s]) for s in ROLES}, 'audited image counts', 0)
    require_equal(private.get('split_bare_ids'), {s: table.iloc[expected_split[s]].patient.nunique() for s in ROLES}, 'audited bare ID counts', 0)
    require(not any(overlap['patient']['overlap_keys'].values()), 'Bare filename IDs cross original splits')
    require(not np.any(truth5[expected_split['val']] == 2), 'Original absent-validation-LC property differs')
    summary = {'images': len(rows), 'split_images': {s: len(expected_split[s]) for s in ROLES},
               'split_five_class_support': {s: np.bincount(truth5[expected_split[s]], minlength=5).tolist() for s in ROLES},
               'overlap': overlap, 'validation_lobular_images': 0,
               'primary_subtype_role_counts': {sub: {role: sum(primary[p] == sub for p in primary
                                                               if table.loc[table.patient == p, 'split'].iloc[0] == role)
                                                   for role in ROLES} for sub in sorted(TEST_N)}}
    return private, arrays, table, summary, overlap_private


def classification(truth, prediction, classes):
    truth = integer_array(truth, (len(truth),), 'classification truth')
    prediction = integer_array(prediction, truth.shape, 'classification predictions')
    require(len(truth) > 0 and np.all((truth >= 0) & (truth < classes))
            and np.all((prediction >= 0) & (prediction < classes)), 'Classification label range differs')
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support, predicted, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros(classes), where=support+predicted > 0)
    accuracy = float(diagonal.sum()/len(truth))
    chance = float(np.dot(support.astype(float), predicted.astype(float))/len(truth)**2)
    return {'records': len(truth), 'correct': int(diagonal.sum()), 'accuracy': accuracy,
            'macro_f1': float(f1.mean()), 'macro_recall': float(recall.mean()),
            'unweighted_cohen_kappa': (accuracy-chance)/(1-chance) if chance < 1 else None,
            'confusion_matrix': matrix.tolist(), 'class_support': support.tolist(),
            'class_correct': diagonal.tolist(), 'class_recall': recall.tolist()}


def metric_replay(truth, prediction, p_mal=None):
    """Original nominal-label metrics in fractions, plus explicit denominators."""
    five = classification(np.asarray(truth), np.asarray(prediction), 5)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    binary = classification((truth > 0).astype(int), (prediction > 0).astype(int), 2)
    malignant = truth > 0
    benign = ~malignant
    conditional = malignant & (prediction > 0)
    correct_malignant = int(np.sum(malignant & (truth == prediction)))
    malignant_count, benign_count, conditional_count = int(malignant.sum()), int(benign.sum()), int(conditional.sum())
    present = np.asarray(five['class_support'][1:]) > 0
    macro_malignant = float(np.asarray(five['class_recall'][1:])[present].mean()) if present.any() else None
    scalar = {'binary_acc': binary['accuracy'],
              'malignant_recall': binary['class_correct'][1]/malignant_count if malignant_count else None,
              'benign_recall': binary['class_correct'][0]/benign_count if benign_count else None,
              'mal_subtype_recall': correct_malignant/malignant_count if malignant_count else None,
              'mal_subtype_macro_recall': macro_malignant,
              'subtype_acc_given_malignant': correct_malignant/conditional_count if conditional_count else 0.0,
              'five_class_acc': five['accuracy'], 'five_macro_f1': five['macro_f1'],
              'five_cohen_kappa': five['unweighted_cohen_kappa']}
    if p_mal is not None:
        scores = probabilities(p_mal, (len(truth),), 'malignancy scores')
        if malignant_count and benign_count:
            scalar['binary_auc'] = float(roc_auc_score(malignant.astype(int), scores))
            binary['roc_auc'] = scalar['binary_auc']
    details = {'five_class': five, 'binary': binary,
               'end_to_end_malignant': {'correct': correct_malignant, 'records': malignant_count,
                                       'recall': scalar['mal_subtype_recall']},
               'conditional_malignant_subtype': {'correct': correct_malignant, 'records': conditional_count,
                                                'accuracy': scalar['subtype_acc_given_malignant']},
               'malignant_macro_recall': {'classes_present': int(present.sum()), 'recall': macro_malignant}}
    return scalar, details


def prediction_metrics(arrays, split, table):
    require(set(arrays) == PREDICTION_KEYS, 'Saved prediction schema differs')
    expected_indices = split['test_indices']
    count = len(expected_indices)
    indices = integer_array(arrays['test_indices'], (count,), 'test indices')
    truth = integer_array(arrays['y5_true'], (count,), 'test truth')
    prediction = integer_array(arrays['y5_pred'], (count,), 'test prediction')
    require(np.array_equal(indices, expected_indices), 'Test indices/order differ')
    require(np.array_equal(truth, split['y5'][indices]), 'Test truths/order differ')
    binary = probabilities(arrays['binary_output'], (count, 2), 'binary head')
    subtype = probabilities(arrays['subtype_output'], (count, 4), 'malignant head')
    p_mal = probabilities(arrays['p_mal'], (count,), 'malignancy scores')
    require(np.array_equal(p_mal, binary[:, 1]), 'Saved p_mal differs from binary head')
    routed = np.where(p_mal >= .5, 1+subtype.argmax(1), 0)
    require(np.array_equal(prediction, routed), 'Original hard threshold/argmax routing differs')
    scalar, details = metric_replay(truth, prediction, p_mal)
    held = table.iloc[indices].copy()
    held['y5'], held['pred'] = truth, prediction
    per_mag = {}
    for mag, group in held.groupby('mag', sort=False):
        original, numbers = metric_replay(group.y5.values, group.pred.values)
        per_mag[str(mag)] = {'original_code_metrics': original, 'metrics': numbers}
    # value_counts preserves first observed order for equal counts in pandas;
    # reproducing this operation avoids replacing notebook ties with min(label).
    vote = held.groupby(['patient', 'y5']).pred.agg(lambda s: s.value_counts().idxmax()).reset_index()
    voted_scalar, voted = metric_replay(vote.y5.values, vote.pred.values)
    voting = {'unit': 'bare filename ID and true five-class label', 'groups': len(vote),
              'bare_ids': held.patient.nunique(), 'original_code_metrics': voted_scalar, 'metrics': voted}
    return scalar, details, per_mag, voting


def verify_artifacts(root, result):
    hashes = result.get('artifact_sha256')
    require(isinstance(hashes, dict) and REQUIRED_ARTIFACTS <= set(hashes), 'Completed run lacks required artifacts')
    for name, digest in hashes.items():
        require(isinstance(name, str) and Path(name).name == name
                and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                'Invalid artifact receipt')
        require_equal(file_hash(root/name), digest, 'run artifact/'+name, 0)
    return hashes


def verify_history(history, metrics, config, progress_path):
    require(isinstance(history, dict) and 'loss' in history and 'val_score' in history, 'Epoch history lacks loss/val_score')
    completed = len(history['loss'])
    maximum = config['training']['maximum_epochs']
    require(1 <= completed <= maximum and all(isinstance(v, list) and len(v) == completed
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
            and np.isfinite(v).all() for v in history.values()), 'Incomplete/nonfinite epoch history')
    validation = np.asarray(history['val_score'], dtype=float)
    require(np.all((validation >= 0) & (validation <= 100)), 'Validation score percentage range differs')
    for name in ('val_binary_acc', 'val_mal_subtype_recall'):
        require(name in history, 'Epoch history lacks original validation component')
    recomputed = .5*(np.asarray(history['val_binary_acc'])+np.asarray(history['val_mal_subtype_recall']))
    require(np.allclose(validation, recomputed, rtol=0, atol=1e-10), 'Validation score formula differs')
    best = int(validation.argmax())+1
    require(completed == maximum or completed-best == config['training']['early_stopping_patience'],
            'Original early-stop budget differs')
    require_equal(metrics.get('epochs_completed'), completed, 'completed epochs', 0)
    require_equal(metrics.get('selected_epoch_first_max_validation_score'), best, 'first maximum selected epoch', 0)
    require_equal(metrics.get('validation_max_score'), float(validation[best-1]), 'maximum validation score', 1e-10)
    with Path(progress_path).open(newline='') as handle:
        progress = list(csv.DictReader(handle))
    require(len(progress) == completed and all(int(row['epoch']) == i for i, row in enumerate(progress)),
            'CSV progress epoch coverage/order differs')
    for name, values in history.items():
        require(all(name in row and math.isclose(float(row[name]), float(values[i]), rel_tol=1e-7, abs_tol=1e-10)
                    for i, row in enumerate(progress)), 'CSV progress/history differs: '+name)
    return completed, best


def class_weights(labels, classes):
    counts = np.bincount(labels, minlength=classes).astype(np.float32)
    return (counts.sum()/(classes*np.maximum(counts, 1))).astype(np.float32).tolist()


def verify_pretrained_load(info):
    require(isinstance(info, dict), 'Pretrained loading diagnostics missing')
    # Native TF checkpoint omits these deterministic integer lookup buffers.
    # Their post-load values/nontrainability must be exhaustively verified in
    # the isolated producer probe; learned relative-position bias is never exempt.
    expected_indices = {
        f'swin/encoder/layers.{stage}/blocks.{block}/attention/self/relative_position_index:0'
        for stage, depth in enumerate((2, 2, 6, 2)) for block in range(depth)}
    missing = info.get('missing_keys')
    require(isinstance(missing, list) and all(isinstance(key, str) for key in missing)
            and len(missing) == len(set(missing))
            and (not missing or set(missing) == expected_indices),
            'Pretrained backbone loading failed: missing learned or unexpected keys')
    for key in ('mismatched_keys', 'error_msgs'):
        require(info.get(key) == [], 'Pretrained backbone loading failed: '+key)
    proof = {'checked_buffers': 12, 'window_size': [7, 7], 'shape': [49, 49],
             'maximum_abs_error': 0, 'all_nontrainable': True}
    require_equal(info.get('deterministic_relative_position_index_check'), proof,
                  'deterministic relative position index proof', 0)
    index_digest = '146d178a603c3a1c98a8100a70fe396369b281782f599312dde58016be6f7db2'
    require_equal(info.get('deterministic_relative_position_index_sha256'),
                  {key: index_digest for key in expected_indices},
                  'deterministic relative position index buffer hashes', 0)
    unexpected = info.get('unexpected_keys')
    require(isinstance(unexpected, list) and all(isinstance(key, str)
            and re.search(r'(^|/)classifier(?:/|$)', key) is not None for key in unexpected),
            'Unexpected non-classifier pretrained keys')
    return {'missing_keys': len(missing), 'unresolved_missing_keys': 0,
            'mismatched_keys': 0, 'error_msgs': 0, 'unused_classifier_keys': len(unexpected),
            'deterministic_relative_position_index_check': proof,
            'deterministic_relative_position_index_sha256': index_digest}


def verify_window_reverse(info):
    require(isinstance(info, dict), 'Window reversal diagnostic missing')
    expected_cases = [{'height': height, 'width': height, 'batch': batch, 'channels': 3}
                      for height in (7, 14, 28, 56) for batch in (1, 2, 8, 32)]
    require(type(info.get('original_graph_reproduced_failure')) is bool,
            'Window reversal original graph diagnostic is not boolean')
    expected = {'status': 'accepted exact integer window reversal', 'cases': expected_cases,
                'maximum_reconstruction_abs': 0.0, 'maximum_eager_graph_abs': 0.0,
                'maximum_gradient_abs': 0.0,
                'original_graph_reproduced_failure': info['original_graph_reproduced_failure'],
                'training_math_change': False}
    require_equal(info, expected, 'exact integer window reversal proof', 0)
    return info


def verify_runtime(root, hashes, config, audit, split, audit_sha):
    prefit = read_json(root/'prefit.json')
    require(prefit.get('status') == 'accepted exact original Swin split and labels', 'Prefit guard did not pass')
    for key, wanted in [('audit_sha256', audit_sha), ('indices_sha256', audit['indices_sha256']),
                        ('split_images', config['split']['expected_images'])]:
        require_equal(prefit.get(key), wanted, 'prefit/'+key, 0)
    train = split['y5'][split['train_indices']]
    weights = {'binary': class_weights((train > 0).astype(int), 2),
               'malignant': class_weights(train[train > 0]-1, 4), 'five': class_weights(train, 5)}
    require_equal(prefit.get('class_weights'), weights, 'training-only balanced class weights', 1e-7)
    environment = read_json(root/'environment.json')
    for name, version in config['runtime'].items():
        require_equal(environment.get(name), version, 'pinned runtime/'+name, 0)
    require(type(environment.get('gpu_count')) is int and environment['gpu_count'] >= 1, 'Pinned GPU execution not demonstrated')
    pretrained = read_json(root/'pretrained_weights.json')
    require_equal(pretrained.get('model_revision'), config['model']['revision'], 'pretrained model revision', 0)
    require_equal(pretrained.get('pooled_shape'), [1, 768], 'pretrained pooled dimensions', 0)
    for name, version in config['runtime'].items():
        require_equal(pretrained.get(name), version, 'pretrained runtime/'+name, 0)
    require(str(pretrained.get('status', '')).startswith('prepared'), 'Pretrained preparation did not pass')
    files = pretrained.get('files')
    required_files = {'config.json', 'tf_model.h5', 'preprocessor_config.json'}
    require(isinstance(files, dict) and required_files == set(files)
            and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None for v in files.values()),
            'Official native TensorFlow pretrained file receipts missing')
    snapshot = Path(pretrained['snapshot_path'])
    for name, digest in files.items():
        require_equal(file_hash(snapshot/name), digest, 'cached pretrained file/'+name, 0)
    loading = verify_pretrained_load(read_json(root/'pretrained_load.json'))
    window_reverse = verify_window_reverse(read_json(root/'window_reverse_check.json'))
    reload = read_json(root/'checkpoint_reload.json')
    for key, wanted in {'images': 32, 'atol': 1e-5, 'checkpoint_sha256': hashes['swin_dual_split0.weights.h5']}.items():
        require_equal(reload.get(key), wanted, 'checkpoint reload/'+key, 0)
    require(all(isinstance(reload.get(k), (float, int)) and not isinstance(reload[k], bool)
                and math.isfinite(reload[k]) and 0 <= reload[k] <= 1e-5
                for k in ('binary_max_abs', 'subtype_max_abs')), 'Checkpoint reload parity failed')
    return {key: environment[key] for key in ('python', *config['runtime'], 'gpu_count')}, files, loading, window_reverse, reload


def verify_notebooks(root, result, audit, args):
    adapted, executed = read_json(root/'adapted.ipynb'), read_json(root/'executed.ipynb')
    require(len(adapted.get('cells', [])) == len(executed.get('cells', [])) == 15,
            'Adapted/executed real notebook cell count differs')
    sources = [''.join(cell['source']) for cell in adapted['cells']]
    require_equal([''.join(cell['source']) for cell in executed['cells']], sources, 'executed notebook sources', 0)
    require_equal(result.get('adapted_notebook_sha256'), file_hash(root/'adapted.ipynb'), 'adapted source digest', 0)
    require_equal(result.get('executed_cell_source_sha256'),
                  [hashlib.sha256(s.encode()).hexdigest() for s in sources], 'executed source receipts', 0)
    require(adapted['cells'][0]['cell_type'] == 'markdown'
            and all(cell['cell_type'] == 'code' for cell in adapted['cells'][1:]), 'Original notebook cell types differ')
    require(all(cell.get('execution_count') is not None
                and not any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
                for cell in executed['cells'][2:]), 'Completed notebook has unexecuted/error experiment cells')
    # The setup cell runs in the prepared environment before a fresh experiment
    # kernel. Its IPython pip command is intentionally not re-executed here.
    require(executed['cells'][1].get('execution_count') is None, 'Package bootstrap unexpectedly executed in experiment kernel')
    original_path = Path(audit['notebook_path'])
    require(file_hash(original_path) == NOTEBOOK_SHA, 'Exact preserved original notebook differs')
    original = read_json(original_path)
    spec = importlib.util.spec_from_file_location('_swin_source_only', Path(__file__).with_name('run_supplied_swin.py'))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Source reconstruction only. Split/labels/metrics above are independently
    # implemented here and never delegated to the producer.
    require_equal(result.get('original_cell_source_sha256'),
                  [hashlib.sha256(''.join(c['source']).encode()).hexdigest() for c in original['cells']],
                  'original source receipts', 0)
    require_equal(result.get('dataset_root'), audit['dataset_root'], 'source dataset path binding', 0)
    prepared = read_json(root/'pretrained_weights.json')
    require_equal(prepared, read_json(result['prepared_path']), 'prepared backbone receipt', 0)
    require_equal(prepared.get('snapshot_path'), result['snapshot_path'], 'backbone snapshot path binding', 0)
    source_args = SimpleNamespace(action='run', dataset_root=audit['dataset_root'],
                                  output=result['execution_root'], audit=result['audit_root'],
                                  prepared=result['prepared_path'])
    expected = adapter.adapted_notebook(original, source_args, prepared)
    require(adapted['cells'][1].get('metadata', {}).get('tags') == ['skip-execution']
            and executed['cells'][1].get('metadata', {}).get('tags') == ['skip-execution'],
            'Separate setup/bootstrap skip declaration differs')
    require_equal([''.join(c['source']) for c in expected['cells']], sources, 'frozen adapter source recipe', 0)


def review_run(args, context, config, audit, split, table):
    root = Path(args.run)
    value = read_json(root/'result.json')
    if (root/'metadata.json').exists():
        require_equal(value, read_json(root/'metadata.json'), 'result/metadata aliases', 0)
    for key in CONTEXT_KEYS:
        require_equal(value.get(key), context[key], 'run/'+key, 0)
    require_equal(value.get('audit_sha256'), file_hash(Path(args.audit)/'private.json'), 'run audit binding', 0)
    require(value.get('action') == 'run', 'Run action differs')
    if value.get('status') != 'complete notebook execution':
        require(value.get('status') in ('failed notebook execution', 'notebook execution started'), 'Unknown execution status')
        return {'status': 'failed' if value['status'].startswith('failed') else 'incomplete',
                'reason': 'notebook_not_completed', 'result_sha256': file_hash(root/'result.json')}
    hashes = verify_artifacts(root, value)
    verify_notebooks(root, value, audit, args)
    metrics = read_json(root/'metrics.json')
    completed, best = verify_history(read_json(root/'history.json'), metrics, config, root/'progress.csv')
    parameters = metrics.get('model_parameters', {})
    require(set(parameters) == {'all', 'trainable'} and all(type(v) is int and v > 0 for v in parameters.values())
            and parameters['trainable'] < parameters['all'], 'Frozen/total model parameter receipt differs')
    environment, pretrained, loading, window_reverse, reload = verify_runtime(root, hashes, config, audit, split,
                                                    file_hash(Path(args.audit)/'private.json'))
    with np.load(root/'predictions.npz', allow_pickle=False) as saved:
        scalar, numbers, per_mag, voting = prediction_metrics({k: saved[k].copy() for k in saved.files}, split, table)
    require(all(value is not None for value in scalar.values()), 'Full test metrics are undefined')
    require_equal(metrics.get('original'), {key: 100*val for key, val in scalar.items()}, 'independent original metrics (%)', 1e-8)
    return {'status': 'accepted', 'result_sha256': file_hash(root/'result.json'),
            'artifacts': {name: hashes[name] for name in sorted(REQUIRED_ARTIFACTS)},
            'epochs_completed': completed, 'selected_epoch': best,
            'validation_max_score_percent': metrics['validation_max_score'],
            'model_parameters': parameters, 'environment': environment, 'pretrained_files': pretrained,
            'pretrained_loading': loading, 'window_reverse_check': window_reverse,
            'checkpoint_reload': reload, 'original_code_metrics': scalar, 'metrics': numbers,
            'per_magnification': per_mag, 'majority_vote': voting}


def summary_text(result):
    lines = ['# Supplied Swin notebook check', '', 'Status: '+result['status']+'.', '']
    if result.get('run', {}).get('status') == 'accepted':
        run = result['run']
        original = run['original_code_metrics']
        lines += [f"Five-class accuracy: {100*original['five_class_acc']:.2f}%; macro F1: {original['five_macro_f1']:.6f}.",
                  f"Binary accuracy: {100*original['binary_acc']:.2f}%; malignant/benign recall: "
                  f"{100*original['malignant_recall']:.2f}% / {100*original['benign_recall']:.2f}%.",
                  f"End-to-end malignant-four recall: {100*original['mal_subtype_recall']:.2f}%; "
                  f"conditional subtype accuracy: {100*original['subtype_acc_given_malignant']:.2f}%.",
                  f"Selected epoch {run['selected_epoch']} of {run['epochs_completed']} completed epochs.", '']
    else:
        lines += ['No accuracy or other performance result is accepted.', '']
    lines += [result['interpretation'], '', 'The checker performed no new fitting or inference.', '']
    return '\n'.join(lines)


def collect(args):
    output = Path(args.output)
    require(not output.exists(), 'Refusing existing collector output')
    result = {'status': 'failed or incomplete supplied-Swin execution', 'collector_new_fits': 0,
              'collector_new_inference': 0,
              'interpretation': 'Separate all-magnification single-image default dual notebook execution. '
              'Bare filename IDs define the supplied split; conservative stem overlap and absent validation lobular '
              'carcinoma are preserved. These are not main provider-case or fifteen-subset benchmark results, '
              'a globally fresh clinical validation, or a verified reproduction of published paper scores.'}
    private = {'errors': {}}
    try:
        context, config = source_context(args)
        result.update(context)
        audit, split, table, summary, overlaps = accepted_audit(args, context, config)
        private['overlap_keys'] = overlaps
        result['audit'] = {**summary, 'private_audit_sha256': file_hash(Path(args.audit)/'private.json'),
                           'manifest_sha256': audit['manifest_sha256'], 'indices_sha256': audit['indices_sha256'],
                           'ordered_csv_sha256': audit['ordered_csv_sha256']}
        checked = review_run(args, context, config, audit, split, table)
        result['run'] = checked
        if checked['status'] == 'accepted':
            result['status'] = 'complete accepted supplied-Swin execution'
    except Exception as error:
        private['errors']['review'] = {'message': str(error), 'traceback': traceback.format_exc()}
        result['run'] = {'status': 'invalid_or_incomplete', 'error_type': type(error).__name__}
    # Only safe aggregate data reach the checked result/summary. Exception messages,
    # original IDs, paths, and overlap identities remain in the hash-bound review.
    assert_public_safe(result)
    output.mkdir(parents=True)
    write_json(output/'private_review.json', private)
    result['private_review_sha256'] = file_hash(output/'private_review.json')
    write_json(output/'result.json', result)
    (output/'summary.md').write_text(summary_text(result))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'audit', 'config', 'output'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--code-sha', required=True)
    return result


if __name__ == '__main__':
    print(collect(parser().parse_args())['status'])

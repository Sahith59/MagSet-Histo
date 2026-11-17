"""Execute the exact supplied CNN/Transformer notebook with registered adapters."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import case_id, file_hash, read_manifest
from magset.five_collect import assert_public_safe, read_json, require_equal

NOTEBOOK_SHA = '2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709'
VARIANTS = ('faithful', 'input_fixed')
MAG_DIRS = ('40X', '100X', '200X', '400X')
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
WEIGHT_NAME = 'efficientnetv2-b0_notop.h5'
WEIGHT_SHA = 'fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def original_notebook(path):
    if file_hash(path) != NOTEBOOK_SHA: raise ValueError('Exact supplied notebook SHA differs')
    notebook = read_json(path)
    if len(notebook.get('cells', [])) != 15 or any(c['cell_type'] != 'code' for c in notebook['cells']):
        raise ValueError('Expected exactly fifteen original code cells')
    if any(c.get('outputs') for c in notebook['cells']): raise ValueError('Original notebook unexpectedly has saved outputs')
    return notebook


def enumerate_images(root):
    """Literal cell1 order: never sort os.listdir, os.walk directories or files."""
    root = os.fspath(root)
    classes = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    result = []
    for cls in classes:
        class_path = os.path.join(root, cls, 'SOB')
        if os.path.exists(class_path):
            subclasses = [name for name in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, name))]
            for subclass in subclasses:
                for current, dirs, files in os.walk(os.path.join(class_path, subclass)):
                    if os.path.basename(current) in MAG_DIRS:
                        for name in files:
                            if name.lower().endswith(EXTENSIONS):
                                result.append({'image_path': os.path.join(current, name),
                                               'class_label': cls, 'subclass_label': subclass})
    return result


def split_labels(rows):
    binary = LabelEncoder(); subtype = LabelEncoder()
    yb = binary.fit_transform([r['class_label'] for r in rows]).astype(int)
    ys = subtype.fit_transform([r['subclass_label'] for r in rows]).astype(int)
    train, test = train_test_split(np.arange(len(rows)), test_size=.2, random_state=42, stratify=ys)
    return yb, ys, train, test, binary.classes_.tolist(), subtype.classes_.tolist()


def bound_rows(manifest, raw_root, dataset_root, expected_count=7909):
    manifest_rows = read_manifest(manifest)
    canonical = {(Path(raw_root)/r['relative_path']).resolve(): r for r in manifest_rows}
    if len(canonical) != len(manifest_rows): raise ValueError('Canonical manifest resolves to duplicate files')
    rows = enumerate_images(Path(dataset_root).resolve())
    if len(rows) != expected_count or len(manifest_rows) != expected_count:
        raise ValueError('Notebook enumeration must cover exactly7909 canonical images')
    seen = set(); result = []
    for index, row in enumerate(rows):
        resolved = Path(row['image_path']).resolve()
        if resolved not in canonical or resolved in seen: raise ValueError('Notebook paths differ from canonical image membership')
        reference = canonical[resolved]; seen.add(resolved)
        if row['class_label'] != reference['broad_class'] or row['subclass_label'] != reference['subtype']:
            raise ValueError('Notebook directory labels disagree with canonical manifest')
        if file_hash(resolved) != reference['sha256']:
            raise ValueError('Notebook image bytes disagree with canonical manifest')
        result.append(dict(row, index=index, relative_path=reference['relative_path'],
            image_sha256=reference['sha256'], provider_key=case_id(reference),
            conservative_group=reference['patient_group_conservative'], magnification=int(reference['magnification'])))
    if seen != set(canonical): raise ValueError('Notebook enumeration omitted canonical fields')
    return result


def source_receipt(code_sha):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch('[0-9a-f]{40}', code_sha or '') or (root/'.source-sha').read_text().strip() != code_sha:
        raise ValueError('Notebook adapter requires an immutable source snapshot')
    archive = (root/'.archive-sha256').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}', archive): raise ValueError('Source archive receipt missing')
    return {'code_sha': code_sha, 'source_archive_sha256': archive, 'adapter_sha256': file_hash(__file__)}


def registered_config(path):
    canonical = Path(__file__).resolve().parents[1]/'configs/supplied-cnn-transformer-v1.json'
    if path is None or file_hash(path) != file_hash(canonical):
        raise ValueError('Notebook execution requires the exact immutable registered configuration')
    value = read_json(path)
    checks = [('name', value.get('name'), 'supplied-cnn-transformer-v1'),
        ('notebook', value.get('notebook_sha256'), NOTEBOOK_SHA), ('variants', value.get('variants'), list(VARIANTS)),
        ('images', value['dataset']['images'], 7909), ('split seed', value['split']['random_state'], 42),
        ('split sizes', [value['split']['fitting_images'], value['split']['reused_validation_images']], [6327,1582]),
        ('training seed', value['training']['seed'], 812), ('epochs', value['training']['maximum_epochs'], 50),
        ('backbone layers', value['model']['fine_tune_last_layers'],60), ('transformer blocks',value['model']['transformer_blocks'],4),
        ('d_model',value['model']['d_model'],512), ('loss weights',value['training']['loss_weights'],{'binary_output':.3,'subtype_output':.7}),
        ('runtime', [value['runtime']['tensorflow'],value['runtime']['keras']], ['2.15.1','2.15.0'])]
    for name, actual, expected in checks: require_equal(actual,expected,'registered notebook '+name,0)
    return value


def audit(args):
    original_notebook(args.notebook); output = Path(args.output)
    config = registered_config(args.config)
    if file_hash(args.manifest) != config['dataset']['canonical_manifest_sha256']:
        raise ValueError('Registered notebook manifest bytes differ')
    if output.exists(): raise ValueError('Refusing existing notebook audit')
    rows = bound_rows(args.manifest, args.raw_root, args.dataset_root)
    yb, ys, train, test, binary_classes, subtype_classes = split_labels(rows)
    if [len(train), len(test)] != [6327, 1582] or binary_classes != ['benign', 'malignant'] or len(subtype_classes) != 8:
        raise ValueError('Supplied image-random split/label counts differ')
    overlap = {}
    for key in ('provider_key', 'conservative_group'):
        fit = {rows[i][key] for i in train}; held = {rows[i][key] for i in test}
        overlap[key] = {'fitting_keys': len(fit), 'validation_keys': len(held), 'overlap_keys': len(fit & held)}
    output.mkdir(parents=True)
    columns = list(rows[0])
    with (output/'ordered.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/'indices.npz', train_indices=train, test_indices=test, y_binary=yb, y_subtype=ys)
    private = dict(source_receipt(args.code_sha), status='accepted exact supplied-notebook current enumeration and image split',
        notebook_sha256=NOTEBOOK_SHA, config_sha256=file_hash(args.config), manifest_sha256=file_hash(args.manifest), images=7909,
        ordered_csv_sha256=file_hash(output/'ordered.csv'), indices_sha256=file_hash(output/'indices.npz'),
        binary_classes=binary_classes, subtype_classes=subtype_classes, train_images=6327, validation_images=1582,
        split_seed=42, training_seed=812, provider_overlap=overlap,
        dataset_root=str(Path(args.dataset_root).resolve()), raw_root=str(Path(args.raw_root).resolve()),
        enumeration_policy='literal unsorted original cell1 current filesystem order; historical author ordering is unknown',
        rows=rows)
    write_json(output/'private.json', private)
    public = {key: value for key, value in private.items() if key not in ('rows', 'dataset_root', 'raw_root')}
    public['private_audit_sha256'] = file_hash(output/'private.json')
    public['endpoint'] = 'eight-class single-image reused image-random validation; related provider images overlap fitting and validation'
    assert_public_safe(public); write_json(output/'public.json', public)
    return public


def accepted_audit(args):
    config = registered_config(args.config)
    root = Path(args.audit); value = read_json(root/'private.json')
    public = read_json(root/'public.json')
    if (public['private_audit_sha256'] != file_hash(root/'private.json')
            or value['notebook_sha256'] != NOTEBOOK_SHA or value['manifest_sha256'] != file_hash(args.manifest)
            or value.get('config_sha256') != file_hash(args.config) or public.get('config_sha256') != file_hash(args.config)
            or value['ordered_csv_sha256'] != file_hash(root/'ordered.csv') or value['indices_sha256'] != file_hash(root/'indices.npz')):
        raise ValueError('Accepted notebook audit bytes differ')
    rows = bound_rows(args.manifest, args.raw_root, args.dataset_root)
    require_equal(rows, value['rows'], 'exact original unsorted dataframe order', 0)
    yb, ys, train, test, binary, subtype = split_labels(rows)
    with np.load(root/'indices.npz', allow_pickle=False) as saved:
        for name, expected in [('train_indices', train), ('test_indices', test), ('y_binary', yb), ('y_subtype', ys)]:
            if not np.array_equal(saved[name], expected): raise ValueError('Accepted image split/encoding changed')
    require_equal(binary, value['binary_classes'], 'binary LabelEncoder order', 0)
    require_equal(subtype, value['subtype_classes'], 'subtype LabelEncoder order', 0)
    return value


def replace_once(source, old, new):
    if source.count(old) != 1: raise ValueError('Exact notebook patch anchor differs: '+old[:50])
    return source.replace(old, new, 1)


def patched_sources(notebook, variant, dataset_root, output):
    if variant not in VARIANTS: raise ValueError('Only the two registered notebook variants are allowed')
    cells = [''.join(cell['source']) for cell in notebook['cells']]
    cells[1] = replace_once(cells[1], "root_dir = '/kaggle/input/breakhisv1/histology_slides/breast'", 'root_dir = '+repr(str(Path(dataset_root).resolve())))
    cells[2] = replace_once(cells[2], "'/kaggle/working/breakhis_dataset.csv'", repr(str(Path(output).resolve()/'breakhis_dataset.csv')))
    cells[7] = replace_once(cells[7], "'/kaggle/working/cnn_transformer_dualhd.keras'", repr(str(Path(output).resolve()/'cnn_transformer_dualhd.keras')))
    cells[5] = replace_once(cells[5], '        super().__init__(**kwargs)\n        self.attention',
        '        super().__init__(**kwargs)\n        self.d_model = d_model\n        self.num_heads = num_heads\n        self.dff = dff\n        self.dropout_rate = dropout_rate\n        self.attention')
    cells[5] = replace_once(cells[5], "            'd_model': self.attention.key_dim * self.attention._num_heads,\n            'num_heads': self.attention._num_heads,",
        "            'd_model': self.d_model,\n            'num_heads': self.num_heads,\n            'dff': self.dff,\n            'dropout_rate': self.dropout_rate,")
    if variant == 'input_fixed':
        cells[4] = replace_once(cells[4], 'datagen.random_transform(img / 255.)', 'datagen.random_transform(img)')
        cells[10] = replace_once(cells[10], 'X_test_norm = X_test / 255.', 'X_test_norm = X_test')
    return cells


def runtime_preamble(gpu_required):
    return '''
import json as _json, hashlib as _hashlib, platform as _platform
from pathlib import Path as _Path
import importlib.metadata as _metadata
assert tf.__version__ == '2.15.1', tf.__version__
assert _metadata.version('keras') == '2.15.0', _metadata.version('keras')
import keras.src.utils.data_utils as _keras_data_utils
_keras_cache_root = os.environ['KERAS_HOME']
assert _Path(_keras_cache_root).is_dir(), 'Project KERAS_HOME must exist before model construction'
_original_keras_get_file = _keras_data_utils.get_file
def _project_keras_get_file(*args, **kwargs):
    kwargs['cache_dir'] = _keras_cache_root
    return _original_keras_get_file(*args, **kwargs)
_keras_data_utils.get_file = _project_keras_get_file
_gpus = tf.config.list_physical_devices('GPU')
''' + ("assert _gpus, 'Production notebook run requires an allocated GPU'\n" if gpu_required else '') + '''
_runtime = {'python': _platform.python_version(), 'tensorflow': tf.__version__, 'keras': _metadata.version('keras'),
 'numpy': np.__version__, 'scikit-learn': _metadata.version('scikit-learn'), 'gpu_count': len(_gpus),
 'keras_cache_policy':'common I/O wrapper forces get_file cache_dir to existing project KERAS_HOME; original URL and hash validation preserved'}
_Path('environment.json').write_text(_json.dumps(_runtime, indent=2))
'''


WEIGHT_GUARD = '''
# Common I/O receipt before any fitting: preserve official weights and hash validation.
_official_weight_path = _Path(_keras_cache_root)/'models'/'efficientnetv2-b0_notop.h5'
assert _official_weight_path.is_file(), 'Official pretrained weight cache receipt missing'
_official_weight_sha = _hashlib.sha256(_official_weight_path.read_bytes()).hexdigest()
assert _official_weight_sha == 'fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285'
_Path('pretrained_weights.json').write_text(_json.dumps({'filename':_official_weight_path.name,
 'sha256':_official_weight_sha,'bytes':_official_weight_path.stat().st_size,
 'cache_policy':'explicit project cache_dir override; official get_file URL/hash behavior unchanged'},indent=2))
'''


def prefit_guard(audit_root):
    return '''
# Adapter guard: prove actual original cells1--3 used the accepted order/split before fitting.
_audit_root = _Path('''+repr(str(Path(audit_root).resolve()))+''')
_audit = _json.loads((_audit_root/'private.json').read_text())
_order = [[r['image_path'], r['class_label'], r['subclass_label']] for r in _audit['rows']]
assert df[['image_path','class_label','subclass_label']].values.tolist() == _order
assert data_csv[['image_path','class_label','subclass_label']].values.tolist() == _order
with np.load(_audit_root/'indices.npz', allow_pickle=False) as _saved:
    _train_idx = _saved['train_indices'].copy(); _test_idx = _saved['test_indices'].copy()
    assert np.array_equal(y_binary, _saved['y_binary']) and np.array_equal(y_subtype, _saved['y_subtype'])
_actual_train, _actual_test = train_test_split(np.arange(len(images)), test_size=.2, random_state=42, stratify=y_subtype)
assert np.array_equal(_actual_train, _train_idx) and np.array_equal(_actual_test, _test_idx)
assert np.array_equal(yb_train, y_binary[_train_idx]) and np.array_equal(yb_test, y_binary[_test_idx])
assert np.array_equal(ys_train, y_subtype[_train_idx]) and np.array_equal(ys_test, y_subtype[_test_idx])
assert list(binary_encoder.classes_) == _audit['binary_classes'] and list(subtype_encoder.classes_) == _audit['subtype_classes']
assert len(images) == 7909 and len(X_train) == 6327 and len(X_test) == 1582
for _array, _indices in ((X_train,_train_idx),(X_test,_test_idx)):
    for _start in range(0,len(_indices),32):
        assert np.array_equal(_array[_start:_start+32], images[_indices[_start:_start+32]].astype('float32'))
np.savez_compressed('prefit_split.npz', train_indices=_train_idx, test_indices=_test_idx, yb_train=yb_train, yb_test=yb_test, ys_train=ys_train, ys_test=ys_test)
_Path('prefit.json').write_text(_json.dumps({'status':'accepted actual notebook order, labels and split before fitting',
 'ordered_csv_sha256':_audit['ordered_csv_sha256'], 'indices_sha256':_audit['indices_sha256'],
 'train_images':len(X_train),'validation_images':len(X_test)},indent=2))
'''


EXPORT_METRICS = '''
# Adapter exports authored metrics distinctly from corrected cascade diagnostics.
_history = {k:[float(x) for x in v] for k,v in hist.history.items()}
_Path('history.json').write_text(_json.dumps(_history,indent=2))
_best_epoch = int(np.argmax(_history['val_subtype_output_accuracy']))+1
_mal = (yb_test == malignant_label)
_strict_correct = int(np.sum(_mal & (yb_pred == malignant_label) & (ys_pred == ys_test)))
_metrics = {'binary_accuracy':float(overall_binary/100), 'binary_malignant_recall':float(malignant_recall/100),
 'binary_benign_recall':float(benign_recall/100), 'binary_auc':float(binary_auc/100),
 'eight_class_accuracy':float(acc),'eight_class_macro_precision':float(mpre),'eight_class_macro_recall':float(mrecal),
 'eight_class_macro_f1':float(mf1),'eight_class_quadratic_kappa':float(kappa),'eight_class_macro_ovr_auc':float(auc),
 'ungated_true_malignant_subtype_accuracy':float(correct_mal/total_mal),
 'strict_true_malignant_cascade_recall':float(_strict_correct/total_mal),
 'strict_cascade_correct_malignant_images':_strict_correct,'true_malignant_images':int(total_mal),
 'selected_epoch_first_max_validation_eight_class_accuracy':_best_epoch,
 'epochs_completed':len(_history['loss']),
 'interpretation':'single-image eight-class image-random reused validation; original malignant metric is ungated; not the provider-case five-class endpoint'}
_checkpoint_model = keras.models.load_model(filepath,custom_objects={
 'TransformerEncoderBlock':TransformerEncoderBlock,'CLSTokenAndPositionalEncoding':CLSTokenAndPositionalEncoding},compile=False)
_reload_count = 32
_reloaded_scores = _checkpoint_model.predict(X_test_norm[:_reload_count],batch_size=32,verbose=0)
_binary_reload_error = float(np.max(np.abs(_reloaded_scores[0]-binary_scores[:_reload_count])))
_subtype_reload_error = float(np.max(np.abs(_reloaded_scores[1]-subtype_scores[:_reload_count])))
assert max(_binary_reload_error,_subtype_reload_error) <= 1e-5
_metrics['checkpoint_full_model_reload'] = {'images':_reload_count,'batch_size':32,'atol':1e-5,
 'binary_max_abs':_binary_reload_error,'subtype_max_abs':_subtype_reload_error,
 'checkpoint_sha256':_hashlib.sha256(_Path(filepath).read_bytes()).hexdigest(),
 'input_policy':'reuse exact existing X_test_norm from original cell10; no additional normalization'}
_Path('metrics.json').write_text(_json.dumps(_metrics,indent=2,allow_nan=False))
np.savez_compressed('predictions.npz',binary_scores=binary_scores,subtype_scores=subtype_scores,
 yb_test=yb_test,ys_test=ys_test,truth_binary=yb_test,truth_subtype=ys_test,
 yb_pred=yb_pred,ys_pred=ys_pred,test_indices=_test_idx,train_indices=_train_idx)
'''


SMOKE_CODE = '''
get_ipython().run_line_magic('matplotlib','inline')
_figure, _axis = plt.subplots()
_axis.plot([0,1],[0,1]); _figure.savefig('smoke-plot.png'); plt.close(_figure)
_rng = np.random.default_rng(812)
_synthetic = _rng.integers(0,256,size=(8,224,224,3),dtype=np.uint8).astype('float32')
_input = _synthetic / 255. if _variant == 'faithful' else _synthetic
_targets = {'binary_output':np.asarray([0,1,0,1,1,1,0,0]),'subtype_output':np.arange(8)}
_batch = model.train_on_batch(_input,_targets,return_dict=True)
assert all(np.isfinite(float(v)) for v in _batch.values())
_before = [v.numpy() for v in model(_input,training=False)]
model.save_weights('smoke.weights.h5'); model.save('smoke.keras')
_reloaded_weights = build_cnn_transformer(num_classes=8)
_reloaded_weights.load_weights('smoke.weights.h5')
_after_weights = [v.numpy() for v in _reloaded_weights(_input,training=False)]
_reloaded_keras_weights = build_cnn_transformer(num_classes=8)
_reloaded_keras_weights.load_weights('smoke.keras')
_after_keras_weights = [v.numpy() for v in _reloaded_keras_weights(_input,training=False)]
_reloaded_full = keras.models.load_model('smoke.keras',custom_objects={
 'TransformerEncoderBlock':TransformerEncoderBlock,'CLSTokenAndPositionalEncoding':CLSTokenAndPositionalEncoding})
_after_full = [v.numpy() for v in _reloaded_full(_input,training=False)]
_errors = [float(np.max(np.abs(a-b))) for a,b in zip(_before,_after_weights)]+[float(np.max(np.abs(a-b))) for a,b in zip(_before,_after_keras_weights)]+[float(np.max(np.abs(a-b))) for a,b in zip(_before,_after_full)]
assert max(_errors) <= 1e-5, _errors
_Path('smoke.json').write_text(_json.dumps({'status':'accepted synthetic eight-image train and serialization smoke',
 'synthetic_images':8,'train_batches':1,'real_validation_predictions':0,'weight_reload_max_abs':max(_errors[:2]),
 'keras_weights_reload_max_abs':max(_errors[2:4]),'full_model_reload_max_abs':max(_errors[4:]),
 'batch_metrics':{k:float(v)for k,v in _batch.items()}},indent=2))
'''


def adapted_notebook(original, args):
    import nbformat
    sources = patched_sources(original, args.variant, args.dataset_root, args.output)
    sources[6] += WEIGHT_GUARD
    if args.action == 'smoke':
        sources = [sources[i] for i in (0, 5, 6, 7)]
        sources[0] += runtime_preamble(False)+"\nnum_classes = 8\n_variant = "+repr(args.variant)+'\n'
        sources.append(SMOKE_CODE)
    else:
        sources[0] += runtime_preamble(True)
        sources[3] += prefit_guard(args.audit)
        sources[7] += "\ncallbacks.append(tf.keras.callbacks.CSVLogger('progress.csv',append=False))\n"
        sources[8] += "\n_Path('history.json').write_text(_json.dumps({k:[float(x)for x in v]for k,v in hist.history.items()},indent=2))\n"
        sources[11] += EXPORT_METRICS
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources])
    return notebook, [{'original_cell': i, 'original_sha256': hashlib.sha256(''.join(cell['source']).encode()).hexdigest(),
        'patched_sha256': hashlib.sha256(patched_sources(original,args.variant,args.dataset_root,args.output)[i].encode()).hexdigest()}
        for i, cell in enumerate(original['cells'])]


def execute_notebook(notebook, output):
    import nbformat
    error = None; manager = None
    def checkpoint(**unused):
        temporary = Path(output)/'executed.ipynb.partial'
        nbformat.write(notebook,temporary)
        temporary.replace(Path(output)/'executed.ipynb')
    try:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=None, allow_errors=False,
                                resources={'metadata': {'path': str(Path(output).resolve())}},
                                on_cell_executed=checkpoint)
        client.execute(cwd=str(Path(output).resolve()))
    except BaseException as caught:
        error = caught
        write_json(Path(output)/'execution_error.json', {'type':type(caught).__name__,'message':str(caught),'traceback':traceback.format_exc()})
    finally:
        checkpoint()
        if manager is not None and getattr(manager, 'has_kernel', False):
            manager.shutdown_kernel(now=True)
    if error is not None: raise error


def run(args):
    original = original_notebook(args.notebook); output = Path(args.output)
    registered_config(args.config)
    if output.exists(): raise ValueError('Refusing existing notebook execution output')
    if args.action == 'run':
        if args.audit is None: raise ValueError('Real execution requires the common accepted audit')
        accepted_audit(args)
    output.mkdir(parents=True)
    receipt = dict(source_receipt(args.code_sha), notebook_sha256=NOTEBOOK_SHA, variant=args.variant,
        action=args.action, status='notebook execution started',
        config_sha256=file_hash(args.config),
        audit_sha256=file_hash(Path(args.audit)/'private.json') if args.action == 'run' else None,
        execution_root=str(output.resolve()), audit_root=str(Path(args.audit).resolve()) if args.action == 'run' else None,
        keras_cache_policy='common I/O get_file wrapper forces cache_dir to existing project KERAS_HOME, preserving original URL/hash validation',
        image_input_policy='faithful divides generator/evaluation images by255; input_fixed removes only those divisions; cell13 raw visualization inputs unchanged',
        visualization_caution='faithful cell13 uses raw0..255 unlike its divided training/evaluation inputs; original token-norm visualization is not attention rollout',
        comparison_caution='original notebook paper-beating prose/tables are preserved author text, not an endorsed matched comparison or novelty claim')
    notebook, patches = adapted_notebook(original, args)
    receipt['cell_patch_receipts'] = patches
    receipt['executed_cell_source_sha256'] = [hashlib.sha256(cell.source.encode()).hexdigest() for cell in notebook.cells]
    import nbformat
    nbformat.write(notebook, output/'adapted.ipynb')
    receipt['adapted_notebook_sha256'] = file_hash(output/'adapted.ipynb')
    write_json(output/'metadata.json', receipt)
    try:
        execute_notebook(notebook, output)
        required = 'smoke.json' if args.action == 'smoke' else 'metrics.json'
        if not (output/required).exists(): raise ValueError('Notebook did not produce its required execution artifact')
        weights = read_json(output/'pretrained_weights.json')
        if weights.get('filename') != WEIGHT_NAME or weights.get('sha256') != WEIGHT_SHA:
            raise ValueError('Actual official pretrained weight receipt differs')
        receipt['status'] = 'complete notebook execution'
        if args.action == 'run':
            metrics = read_json(output/'metrics.json')
            receipt['selected_epoch_first_max_validation_eight_class_accuracy'] = metrics['selected_epoch_first_max_validation_eight_class_accuracy']
    except BaseException:
        receipt['status'] = 'failed notebook execution'
        raise
    finally:
        receipt['artifact_sha256'] = {path.name:file_hash(path) for path in sorted(output.iterdir())
                                     if path.is_file() and path.name not in ('metadata.json','result.json')}
        cache = Path(os.environ.get('KERAS_HOME', str(Path.home()/'.keras')))/'models'
        receipt['keras_cached_weight_sha256'] = {path.name:file_hash(path) for path in sorted(cache.glob('*efficientnetv2*')) if path.is_file()}
        write_json(output/'metadata.json', receipt)
        write_json(output/'result.json', receipt)
    return receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('audit','smoke','run'),required=True)
    for key in ('notebook','manifest','raw-root','dataset-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--variant',choices=VARIANTS,default='faithful')
    p.add_argument('--audit',type=Path);p.add_argument('--config',type=Path,required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args(); (audit if args.action=='audit' else run)(args)

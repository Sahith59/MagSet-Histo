"""Run the preserved default Swin notebook; append provenance, not a new recipe."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import case_id, file_hash, read_manifest
from magset.five_collect import assert_public_safe, require_equal
from scripts.run_supplied_cnn_transformer import execute_notebook, replace_once, source_receipt as cnn_source_receipt

NOTEBOOK_SHA = 'fd88d851fbd8cf049a0ebbd718ed8757ffe28f629e2472262efca30bbca029af'
REVISION = 'd00d478bfaf1f417d34a1186673ad4b4be264c51'
SUBTYPES = {'A':'adenosis','F':'fibroadenoma','PT':'phyllodes_tumor','TA':'tubular_adenoma',
            'DC':'ductal_carcinoma','LC':'lobular_carcinoma','MC':'mucinous_carcinoma','PC':'papillary_carcinoma'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_receipt(sha):
    return dict(cnn_source_receipt(sha), adapter_sha256=file_hash(__file__))


def load_original(path):
    if file_hash(path) != NOTEBOOK_SHA:
        raise ValueError('Exact supplied Swin notebook hash differs')
    value = json.loads(Path(path).read_text())
    if len(value['cells']) != 15 or value['cells'][0]['cell_type'] != 'markdown':
        raise ValueError('Original Swin cell structure differs')
    if any(c.get('outputs') for c in value['cells']):
        raise ValueError('Supplied Swin unexpectedly has saved outputs')
    return value


def config_receipt(path):
    fixed = Path(__file__).resolve().parents[1]/'configs/supplied-swin-v1.json'
    if file_hash(path) != file_hash(fixed):
        raise ValueError('Use exact immutable Swin configuration')
    config = json.loads(Path(path).read_text())
    if config['notebook_sha256'] != NOTEBOOK_SHA or config['model']['revision'] != REVISION or config['mode'] != 'dual':
        raise ValueError('Swin configuration contract differs')
    return config


def patched_sources(original, dataset_root, output, snapshot):
    cells = [''.join(c['source']) for c in original['cells']]
    cells[2] = replace_once(cells[2], "OUT_DIR     = '/kaggle/working'", 'OUT_DIR     = '+repr(str(Path(output).resolve())))
    cells[3] = replace_once(cells[3], "root_dir = '/kaggle/input/breakhisv1/histology_slides/breast'", 'root_dir = '+repr(str(Path(dataset_root).resolve())))
    cells[8] = replace_once(cells[8], 'TFSwinModel.from_pretrained(SWIN_CKPT)',
        'TFSwinModel.from_pretrained('+repr(str(Path(snapshot).resolve()))+', local_files_only=True, use_safetensors=False)')
    return cells


def reconstructed(original, dataset_root):
    sources = patched_sources(original, dataset_root, '.', '.')
    space = {'os':os, 'pd':pd, 'np':np, 'SPLIT_SEED':0}
    # Literal original parsing/order/quota code, no TensorFlow or image predictions.
    exec(compile(sources[3], 'supplied-swin-cell3', 'exec'), space)
    exec(compile(sources[4], 'supplied-swin-cell4', 'exec'), space)
    frame = space['df']
    mal = ['DC','LC','MC','PC']
    y5 = np.array([0 if r.binary=='benign' else 1+mal.index(r.subtype) for r in frame.itertuples()], dtype=int)
    arrays = {role+'_indices':np.flatnonzero(frame.split.values==role) for role in ('train','val','test')}
    arrays.update(y5=y5, y_binary=(y5>0).astype(int), y_malignant=y5-1)
    return frame, arrays


def audit(args):
    config = config_receipt(args.config); original = load_original(args.notebook)
    if file_hash(args.manifest) != config['dataset']['canonical_manifest_sha256']:
        raise ValueError('Canonical manifest changed')
    output = Path(args.output)
    if output.exists(): raise ValueError('Refusing existing audit output')
    frame, arrays = reconstructed(original, args.dataset_root)
    manifest = read_manifest(args.manifest)
    canonical = {(Path(args.raw_root)/r['relative_path']).resolve():r for r in manifest}
    if len(frame)!=7909 or len(canonical)!=7909: raise ValueError('Expected exact7909 canonical fields')
    seen=set(); rows=[]
    for item in frame.to_dict('records'):
        path=Path(item['image_path']).resolve()
        if path not in canonical or path in seen: raise ValueError('Swin enumeration membership differs')
        ref=canonical[path]; seen.add(path)
        if (SUBTYPES[item['subtype']]!=ref['subtype'] or item['binary']!=ref['broad_class']
                or int(item['mag'])!=int(ref['magnification']) or file_hash(path)!=ref['sha256']):
            raise ValueError('Swin image bytes/labels differ from canonical manifest')
        rows.append(dict(item, provider_key=case_id(ref), conservative_group=ref['patient_group_conservative'],image_sha256=ref['sha256']))
    if seen != set(canonical): raise ValueError('Incomplete original enumeration')
    role_images={k:len(arrays[k+'_indices']) for k in ('train','val','test')}
    role_ids={k:int(frame[frame.split==k].patient.nunique()) for k in role_images}
    require_equal(role_images,config['split']['expected_images'],'Swin image counts',0)
    require_equal(role_ids,config['split']['expected_bare_ids'],'Swin bare IDs',0)
    overlap={}
    for key in ('patient','provider_key','conservative_group'):
        sets={role:{rows[i][key] for i in arrays[role+'_indices']} for role in role_images}
        overlap[key]={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlap['patient'].values()): raise ValueError('Original bare IDs overlap roles')
    output.mkdir(parents=True)
    with (output/'ordered.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'indices.npz',**arrays)
    private=dict(source_receipt(args.code_sha),status='accepted exact original Swin input and split audit',
        notebook_sha256=NOTEBOOK_SHA,config_sha256=file_hash(args.config),manifest_sha256=file_hash(args.manifest),
        ordered_csv_sha256=file_hash(output/'ordered.csv'),indices_sha256=file_hash(output/'indices.npz'),
        images=7909,split_images=role_images,split_bare_ids=role_ids,overlap=overlap,
        test_five_support=np.bincount(arrays['y5'][arrays['test_indices']],minlength=5).tolist(),
        validation_five_support=np.bincount(arrays['y5'][arrays['val_indices']],minlength=5).tolist(),
        secondary_test_id_target_groups=int(frame.iloc[arrays['test_indices']].assign(y5=arrays['y5'][arrays['test_indices']]).groupby(['patient','y5']).ngroups),
        rows=rows, notebook_path=str(Path(args.notebook).resolve()), dataset_root=str(Path(args.dataset_root).resolve()))
    write_json(output/'private.json',private)
    public={k:v for k,v in private.items() if k not in ('rows','notebook_path','dataset_root')}
    public['private_audit_sha256']=file_hash(output/'private.json')
    public['interpretation']='filename-ID image holdout with conservative-stem overlaps; not verified clinical identity or globally fresh test'
    assert_public_safe(public);write_json(output/'public.json',public)
    return public


def accepted_audit(args):
    folder=Path(args.audit);value=json.loads((folder/'private.json').read_text());public=json.loads((folder/'public.json').read_text())
    for key,path in [('ordered_csv_sha256',folder/'ordered.csv'),('indices_sha256',folder/'indices.npz'),('config_sha256',args.config),('manifest_sha256',args.manifest)]:
        if value[key]!=file_hash(path): raise ValueError('Audit artifact differs: '+key)
    if public['private_audit_sha256']!=file_hash(folder/'private.json'): raise ValueError('Audit binding differs')
    for row in value['rows']:
        if file_hash(row['image_path'])!=row['image_sha256']:
            raise ValueError('Original image bytes changed after audit')
    frame,arrays=reconstructed(load_original(args.notebook),args.dataset_root)
    require_equal(frame.to_dict('records'),[{k:r[k] for k in frame.columns} for r in value['rows']],'literal original dataframe',0)
    with np.load(folder/'indices.npz',allow_pickle=False) as saved:
        for key,x in arrays.items():
            if not np.array_equal(saved[key],x): raise ValueError('Audit index/label mismatch '+key)
    return value


def runtime(prepared, gpu):
    return '''
from pathlib import Path as _Path
import hashlib as _hashlib, platform as _platform, importlib.metadata as _metadata
assert tf.__version__=='2.15.1' and keras.__version__=='2.15.1' and transformers.__version__=='4.46.3'
_prepared=json.loads(_Path('''+repr(str(Path(prepared).resolve()))+''').read_text())
_snapshot=_Path(_prepared['snapshot_path'])
assert _prepared['model_revision']=='''+repr(REVISION)+'''
assert set(_prepared['files'])=={'config.json','tf_model.h5','preprocessor_config.json'}
for _name,_sha in _prepared['files'].items():
    assert _hashlib.sha256((_snapshot/_name).read_bytes()).hexdigest()==_sha, 'Official Swin weights changed'
_Path('pretrained_weights.json').write_text(json.dumps(_prepared,indent=2))
_load_info=json.loads(_Path('pretrained_load.json').read_text())
_index_keys={'swin/encoder/layers.%d/blocks.%d/attention/self/relative_position_index:0'%(s,b)
 for s,d in enumerate([2,2,6,2])for b in range(d)}
assert set(_load_info['missing_keys']) in (set(),_index_keys), _load_info
assert not _load_info['mismatched_keys'] and not _load_info['error_msgs'], _load_info
assert _load_info['deterministic_relative_position_index_check']=={'checked_buffers':12,'window_size':[7,7],
 'shape':[49,49],'maximum_abs_error':0,'all_nontrainable':True}
assert _load_info['deterministic_relative_position_index_sha256']=={k:'146d178a603c3a1c98a8100a70fe396369b281782f599312dde58016be6f7db2'for k in _index_keys}
_gpu_count=len(tf.config.list_physical_devices('GPU'))
'''+("assert _gpu_count>=1, 'Production run requires allocated GPU'\n" if gpu else '')+'''
_Path('environment.json').write_text(json.dumps({'python':_platform.python_version(),'tensorflow':tf.__version__,
'tf_keras':keras.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'gpu_count':_gpu_count},indent=2))
'''


def prefit(audit_root):
    return '''
_audit_root=_Path('''+repr(str(Path(audit_root).resolve()))+''')
_audit=json.loads((_audit_root/'private.json').read_text())
_saved=np.load(_audit_root/'indices.npz',allow_pickle=False)
assert df[['image_path','binary','subtype','patient','mag','split']].to_dict('records')==[
 {k:r[k] for k in ['image_path','binary','subtype','patient','mag','split']} for r in _audit['rows']]
for _role in ('train','val','test'):
    assert np.array_equal(idx[_role],_saved[_role+'_indices'])
    for _k,_full in [('y5','y5'),('ybin','y_binary'),('ymal','y_malignant')]:
        assert np.array_equal(Y[_role][_k],_saved[_full][idx[_role]])
_Path('prefit.json').write_text(json.dumps({'status':'accepted exact original Swin split and labels',
'audit_sha256':_hashlib.sha256((_audit_root/'private.json').read_bytes()).hexdigest(),
'indices_sha256':_hashlib.sha256((_audit_root/'indices.npz').read_bytes()).hexdigest(),
'split_images':{k:len(idx[k])for k in idx},'class_weights':{'binary':W_BIN.tolist(),'malignant':W_MAL.tolist(),'five':W_5.tolist()}},indent=2))
'''


EXPORT = '''
# Additional receipts; original final-test predictions above are unchanged.
np.savez_compressed('predictions.npz',test_indices=idx['test'],y5_true=y5_true,y5_pred=y5_pred,p_mal=p_mal,
 binary_output=test_scores['binary_output'],subtype_output=test_scores['subtype_output'])
_history={k:[float(x)for x in v]for k,v in hist.history.items()}
_selected=int(np.argmax(_history['val_score']))+1
_Path('metrics.json').write_text(json.dumps({'original':{k:float(v)for k,v in res.items()},
 'selected_epoch_first_max_validation_score':_selected,'epochs_completed':len(_history['loss']),
 'validation_max_score':float(max(_history['val_score'])),'model_parameters':{'all':n_all,'trainable':n_tr}},indent=2))
_reloaded=SwinBreakHis(mode=MODEL_TYPE)
_reloaded(np.zeros((1,224,224,3),np.float32),training=False)
_reloaded.load_weights(ckpt_path)
_rp=predict_scores(_reloaded,X_test[:32],bs=32)
_reload={'images':32,'atol':1e-5,'checkpoint_sha256':_hashlib.sha256(_Path(ckpt_path).read_bytes()).hexdigest()}
for _short,_key in [('binary','binary_output'),('subtype','subtype_output')]:
    assert np.isfinite(_rp[_key]).all()
    _reload[_short+'_max_abs']=float(np.max(np.abs(_rp[_key]-test_scores[_key][:32])))
    assert _reload[_short+'_max_abs']<=1e-5
_Path('checkpoint_reload.json').write_text(json.dumps(_reload,indent=2))
'''

SMOKE = '''
_rng=np.random.RandomState(812)
_x=_rng.uniform(-1,1,(8,224,224,3)).astype(np.float32)
_yb=np.array([0,1,1,1,1,0,1,1],np.int32)
_ym=np.array([-1,0,1,2,3,-1,0,1],np.int32)
_masked=float(weighted_sparse_ce(np.ones(4,np.float32))(tf.constant([-1,-1]),tf.constant([[.1,.2,.3,.4],[.4,.3,.2,.1]])))
assert _masked==0.0
model.train_on_batch(_x,{'binary_output':_yb,'subtype_output':_ym})
_first=model(_x[:2],training=False)
model.save_weights('smoke.weights.h5')
_other=SwinBreakHis(mode='dual');_other(np.zeros((1,224,224,3),np.float32),training=False)
_other.load_weights('smoke.weights.h5')
_second=_other(_x[:2],training=False)
_err=max(float(np.max(np.abs(_first[k].numpy()-_second[k].numpy())))for k in _first)
assert _err<=1e-5
assert not model.swin.swin.embeddings.trainable
assert [x.trainable for x in model.swin.swin.encoder.layers]==[False,False,True,True]
_Path('smoke.json').write_text(json.dumps({'status':'accepted synthetic Swin train and weight reload',
'synthetic_images':8,'train_batches':1,'real_test_predictions':0,'masked_benign_loss':_masked,
'weight_reload_max_abs':_err,'all_parameters':n_all,'trainable_parameters':n_tr,
'encoder_stage_trainable':[x.trainable for x in model.swin.swin.encoder.layers]},indent=2))
'''


def adapted_notebook(original, args, prepared):
    output=Path(args.output)
    sources=patched_sources(original,args.dataset_root,output,prepared['snapshot_path'])
    sources[2]+=runtime(args.prepared,args.action=='run')
    if args.action=='smoke':
        sources[2]+="\nW_BIN=np.ones(2,np.float32);W_MAL=np.ones(4,np.float32);W_5=np.ones(5,np.float32)\n"
        selected=[sources[i]for i in (2,8,9)]+[SMOKE]
    else:
        sources[6]+=prefit(args.audit)
        sources[9]+="\ncallbacks.append(keras.callbacks.CSVLogger('progress.csv',append=False))\n"
        sources[10]+="\n_Path('history.json').write_text(json.dumps({k:[float(x)for x in v]for k,v in hist.history.items()},indent=2))\n"
        sources[12]+=EXPORT
        selected=sources[2:]
    import nbformat
    if args.action=='run':
        notebook=nbformat.reads(json.dumps(original),as_version=4)
        notebook.cells[1].metadata['tags']=['skip-execution']
        for i,s in enumerate(selected,start=2):
            notebook.cells[i].source=s
    else:
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s)for s in selected])
    return notebook


def run(args):
    config_receipt(args.config);original=load_original(args.notebook);output=Path(args.output)
    if output.exists(): raise ValueError('Refusing existing Swin execution output')
    if args.action=='run': accepted_audit(args)
    prepared=json.loads(Path(args.prepared).read_text())
    if prepared['model_revision']!=REVISION: raise ValueError('Pinned Swin revision differs')
    notebook=adapted_notebook(original,args,prepared)
    import nbformat
    output.mkdir(parents=True);nbformat.write(notebook,output/'adapted.ipynb')
    # Separate process: never advance the original notebook kernel's seeded RNG.
    probe='''import json,sys,re,hashlib
from pathlib import Path
import numpy as np, tensorflow as tf
from transformers import TFSwinModel
model,info=TFSwinModel.from_pretrained(sys.argv[1],local_files_only=True,use_safetensors=False,output_loading_info=True)
info.setdefault('error_msgs',[])
allowed={'swin/encoder/layers.%d/blocks.%d/attention/self/relative_position_index:0'%(s,b)for s,d in enumerate([2,2,6,2])for b in range(d)}
assert set(info['missing_keys']) in (set(),allowed),info
assert not info['mismatched_keys'] and not info['error_msgs'],info
coords=np.stack(np.meshgrid(np.arange(7),np.arange(7),indexing='ij')).reshape(2,-1)
delta=coords[:,:,None]-coords[:,None,:]
expected=((delta[0]+6)*13+(delta[1]+6)).astype(np.int32)
found=set(); hashes={}
for weight in model.weights:
    match=re.search(r'(swin/encoder/layers\\.\\d+/blocks\\.\\d+/attention/self/relative_position_index:0)$',weight.name)
    if match:
        assert not weight.trainable and weight.dtype==tf.int32
        assert np.array_equal(weight.numpy(),expected)
        assert match.group(1) not in found
        found.add(match.group(1))
        hashes[match.group(1)]=hashlib.sha256(weight.numpy().astype('<i4').tobytes(order='C')).hexdigest()
assert found==allowed,(found,allowed)
info['deterministic_relative_position_index_check']={'checked_buffers':12,'window_size':[7,7],'shape':[49,49],'maximum_abs_error':0,'all_nontrainable':True}
info['deterministic_relative_position_index_sha256']=hashes
Path(sys.argv[2]).write_text(json.dumps(info,indent=2))
'''
    receipt=dict(source_receipt(args.code_sha),notebook_sha256=NOTEBOOK_SHA,config_sha256=file_hash(args.config),
        action=args.action,status='notebook execution started',
        audit_sha256=file_hash(Path(args.audit)/'private.json')if args.action=='run' else None,
        original_cell_source_sha256=[hashlib.sha256(''.join(c['source']).encode()).hexdigest()for c in original['cells']],
        executed_cell_source_sha256=[hashlib.sha256(c.source.encode()).hexdigest()for c in notebook.cells],
        adapted_notebook_sha256=file_hash(output/'adapted.ipynb'),execution_root=str(output.resolve()),
        audit_root=str(Path(args.audit).resolve())if args.action=='run'else None,
        dataset_root=str(Path(args.dataset_root).resolve()),prepared_path=str(Path(args.prepared).resolve()),
        snapshot_path=prepared['snapshot_path'],
        setup_policy='cell1 installed in separate pinned environment; fresh kernel executes original cells2 onward',
        comparison_caution='authored paper Beat printouts are not endorsed matched comparisons; token-norm map is not attention rollout')
    write_json(output/'metadata.json',receipt)
    try:
        subprocess.run([sys.executable,'-c',probe,prepared['snapshot_path'],str((output/'pretrained_load.json').resolve())],check=True)
        execute_notebook(notebook,output)
        expected='smoke.json'if args.action=='smoke' else'checkpoint_reload.json'
        if not(output/expected).exists(): raise ValueError('Missing required Swin execution output')
        receipt['status']='complete notebook execution'
    except BaseException:
        receipt['status']='failed notebook execution'
        raise
    finally:
        receipt['artifact_sha256']={p.name:file_hash(p)for p in sorted(output.iterdir())if p.is_file()and p.name not in('metadata.json','result.json')}
        write_json(output/'metadata.json',receipt);write_json(output/'result.json',receipt)
    return receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('audit','smoke','run'),required=True)
    for key in ('notebook','manifest','raw-root','dataset-root','output','config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-sha',required=True)
    p.add_argument('--audit',type=Path);p.add_argument('--prepared',type=Path)
    return p


if __name__=='__main__':
    args=parser().parse_args();(audit if args.action=='audit'else run)(args)

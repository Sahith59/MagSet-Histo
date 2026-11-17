"""Train matched CNN/Swin adaptations and seal validation before test inference."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from magset import matched_data as md
from magset.data import file_hash


def seed_number(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')


def learning_rate(config,model,epoch):
    if not 0<=epoch<40:raise ValueError('Zero-based registered epoch required')
    initial=config['training']['learning_rates'][model]
    return initial*(.01+.99*(1+math.cos(math.pi*epoch/39))/2)


def augment(pixels,indices,epoch,seed=812):
    import cv2
    out=np.asarray(pixels,dtype=np.float32).copy()
    for pos,index in enumerate(indices):
        rng=np.random.RandomState(seed_number('augmentation',seed,epoch,int(index)))
        angle=float(rng.uniform(-90,90));horizontal=rng.rand()<.5;vertical=rng.rand()<.5
        matrix=cv2.getRotationMatrix2D((111.5,111.5),angle,1.)
        for crop in range(3):
            value=cv2.warpAffine(out[pos,crop],matrix,(224,224),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            if horizontal:value=value[:,::-1]
            if vertical:value=value[::-1]
            out[pos,crop]=value
    return out


def runtime(config,require_gpu):
    os.environ['TF_USE_LEGACY_KERAS']='1'
    import tensorflow as tf
    import tf_keras as keras
    versions={name:importlib.metadata.version(name)for name in config['runtime']}
    if versions!=config['runtime']:raise ValueError('Pinned matched runtime differs')
    seed=config['training']['seed'];random.seed(seed);np.random.seed(seed);tf.random.set_seed(seed);keras.utils.set_random_seed(seed)
    gpus=tf.config.list_physical_devices('GPU')
    if require_gpu and not gpus:raise ValueError('An allocated GPU is required')
    return tf,keras,dict(versions,gpu_count=len(gpus),python=sys.version.split()[0],
        gpu_name=tf.config.experimental.get_device_details(gpus[0]).get('device_name','unknown')if gpus else None)


def compile_model(model,config):
    import tf_keras as keras
    training=config['training']
    opt=keras.optimizers.Adam(learning_rate=training['learning_rates'][model.kind],beta_1=.9,beta_2=.999,epsilon=1e-7,amsgrad=False)
    losses={name:keras.losses.SparseCategoricalCrossentropy(reduction='sum')for name in ('binary_output','subtype_output')}
    model.compile(optimizer=opt,loss=losses,loss_weights={'binary_output':.5,'subtype_output':.5})


def model_paths(args):
    if file_hash(args.cnn_weights)!=md.config(args.config)['weights']['cnn_sha256']:
        raise ValueError('Original ImageNet CNN weights changed')
    prepared=json.loads(Path(args.swin_prepared).read_text())
    if prepared['model_revision']!=md.config(args.config)['weights']['swin_revision']:
        raise ValueError('Pinned original Swin revision differs')
    snapshot=Path(prepared['snapshot_path'])
    if file_hash(snapshot/'tf_model.h5')!=md.config(args.config)['weights']['swin_tf_sha256']:
        raise ValueError('Original native Swin weights changed')
    return dict(cnn_weights=args.cnn_weights,swin_snapshot=snapshot)


def predict_heads(model,pixels,indices,batch=16):
    import tensorflow as tf
    @tf.function(input_signature=[tf.TensorSpec([None,3,224,224,3],tf.float32)])
    def forward(x):return model(x,training=False)
    binary=[];malignant=[]
    for start in range(0,len(indices),batch):
        values=forward(np.asarray(pixels[np.asarray(indices[start:start+batch],dtype=int)],dtype=np.float32))
        binary.append(np.asarray(values['binary_output'],dtype=np.float64))
        malignant.append(np.asarray(values['subtype_output'],dtype=np.float64))
    b=np.concatenate(binary);m=np.concatenate(malignant)
    md.joint_probabilities(b,m)
    return b,m


def checked_pixels(args,audit):
    folder=Path(args.pixels);receipt=json.loads((folder/'receipt.json').read_text())
    if (receipt['status']!='accepted common three-crop pixels'or receipt['config_sha256']!=file_hash(args.config)
            or receipt['input_audit_sha256']!=file_hash(Path(args.audit)/'private.json')
            or receipt['pixels_sha256']!=file_hash(folder/'pixels.npy')or receipt['shape']!=[7909,3,224,224,3]):
        raise ValueError('Shared pixel cache differs')
    pixels=np.load(folder/'pixels.npy',mmap_mode='r',allow_pickle=False)
    if pixels.dtype!=np.uint8 or list(pixels.shape)!=receipt['shape']:raise ValueError('Shared pixels shape/dtype differs')
    return pixels,dict(input_pixels_receipt_sha256=file_hash(folder/'receipt.json'),pixels_sha256=receipt['pixels_sha256'])


def artifacts(root):
    root=Path(root)
    return {str(p.relative_to(root)):file_hash(p)for p in sorted(root.rglob('*'))
            if p.is_file()and p.name!='result.json'and'resume'not in p.relative_to(root).parts}


def smoke(args):
    cfg=md.config(args.config);tf,keras,env=runtime(cfg,True)
    from magset.matched_models import build_model,architecture_receipt
    from magset.swin_compat import check_window_reverse
    # Technical checks use a separate process/job, not the real fitting kernel.
    out=Path(args.output)
    if out.exists():raise ValueError('Refusing existing matched model smoke')
    out.mkdir(parents=True);window=check_window_reverse();md.write_json(out/'window_reverse_check.json',window)
    probe_prob=np.array([[.2,.8],[.7,.3],[.4,.6]],np.float32);probe_y=np.array([1,0,1],np.int32)
    probe_w=np.array([.3,0.,2.1],np.float32)
    loss_probe=keras.losses.SparseCategoricalCrossentropy(reduction='sum')
    actual=float(loss_probe(probe_y,probe_prob,sample_weight=probe_w/16))
    expected=float(np.sum(-np.log(probe_prob[np.arange(3),probe_y])*probe_w)/16)
    masked=float(loss_probe(probe_y,probe_prob,sample_weight=np.zeros(3,np.float32)))
    if abs(actual-expected)>1e-6 or masked!=0:raise ValueError('Fixed denominator/masked objective check failed')
    keras.utils.set_random_seed(cfg['training']['seed'])
    model=build_model(args.model,**model_paths(args));compile_model(model,cfg)
    receipt=architecture_receipt(model);md.write_json(out/'architecture.json',receipt)
    rng=np.random.RandomState(812);x=rng.uniform(0,255,(16,3,224,224,3)).astype(np.float32)
    y5=np.arange(16)%5;y={'binary_output':(y5>0).astype(int),'subtype_output':np.maximum(y5-1,0)}
    wb=np.ones(16)/16;wm=(y5>0).astype(float)/16
    start=time.monotonic();loss=model.train_on_batch(x,y,sample_weight={'binary_output':wb,'subtype_output':wm},return_dict=True)
    if not np.isfinite(list(loss.values())).all():raise ValueError('Synthetic matched loss is nonfinite')
    # A partial batch retains the fixed denominator16.
    model.train_on_batch(x[:5],{k:v[:5]for k,v in y.items()},sample_weight={'binary_output':wb[:5],'subtype_output':wm[:5]},return_dict=True)
    before=model(x[:2],training=False);model.save_weights(out/'smoke.weights.h5')
    restored=build_model(args.model,**model_paths(args));restored.load_weights(out/'smoke.weights.h5')
    after=restored(x[:2],training=False)
    delta=max(float(np.max(np.abs(np.asarray(before[k])-np.asarray(after[k]))))for k in before)
    if delta>1e-5:raise ValueError('Synthetic matched checkpoint replay differs')
    # Benign targets have exactly zero malignant sample weight; regression tests
    # separately establish global head/class/case/view mass and denominator.
    result=dict(md.context(args.code_sha,args.config),status='accepted synthetic matched model smoke',model=args.model,
        synthetic_fields=16,full_batch_fields=16,partial_batch_fields=5,synthetic_updates=2,real_validation_predictions=0,
        real_test_predictions=0,weight_reload_max_abs=delta,environment=env,architecture=receipt,
        window_reverse_check=window,weighted_objective_check={'partial_batch':3,'denominator':16,
            'maximum_abs_error':abs(actual-expected),'zero_weight_loss':masked},elapsed_seconds=time.monotonic()-start)
    md.write_json(out/'smoke.json',result);return result


def train(args):
    cfg=md.config(args.config);tf,keras,env=runtime(cfg,True);audit=md.load_audit(args.audit,args.config)
    from magset.matched_models import build_model,architecture_receipt
    smoke_folder=Path(args.root)/'technical'/args.model
    passed=json.loads((smoke_folder/'smoke.json').read_text())
    if (passed['status']!='accepted synthetic matched model smoke'or passed['code_sha']!=args.code_sha
            or passed['config_sha256']!=file_hash(args.config)or passed['weight_reload_max_abs']>1e-5
            or passed['full_batch_fields']!=16 or passed['partial_batch_fields']!=5):
        raise ValueError('Same-source actual full/partial-batch GPU smoke required')
    pixels,pixel_context=checked_pixels(args,audit);out=Path(args.output)
    if out.exists():raise ValueError('Refusing existing matched neural fit')
    out.mkdir(parents=True);model=build_model(args.model,**model_paths(args));compile_model(model,cfg)
    provenance=dict(md.context(args.code_sha,args.config),model=args.model,
        input_audit_sha256=file_hash(Path(args.audit)/'private.json'),**pixel_context)
    md.write_json(out/'environment.json',env);md.write_json(out/'architecture.json',architecture_receipt(model))
    md.write_json(out/'started.json',dict(provenance,status='training started',test_predictions=0))
    fields=audit['fields'];cases=audit['cases'];train_cases=set(audit['roles']['train']);val_cases=set(audit['roles']['validation'])
    fitting=np.array([f['index']for f in fields if f['case_index']in train_cases],np.int64)
    validation=np.array([f['index']for f in fields if f['case_index']in val_cases],np.int64)
    y5=np.array([cases[f['case_index']]['truth5']for f in fields],np.int64);wb,wm=md.field_weights(cases,fields,audit['roles']['train'])
    with np.load(Path(args.audit)/'training-weights.npz',allow_pickle=False)as weights:
        if not np.array_equal(wb,weights['binary'])or not np.array_equal(wm,weights['malignant']):raise ValueError('Registered fitting objective changed')
    history=[];candidate_scores=[];candidate_hashes=[];batch=16
    checkpoint=tf.train.Checkpoint(model=model,optimizer=model.optimizer)
    manager=tf.train.CheckpointManager(checkpoint,str(out/'resume'),max_to_keep=1)
    with (out/'progress.csv').open('w',newline='')as progress:
        writer=csv.DictWriter(progress,fieldnames=['epoch','learning_rate','loss','binary_loss','malignant_loss','optimizer_updates']);writer.writeheader();progress.flush()
        for epoch in range(40):
            lr=learning_rate(cfg,args.model,epoch);model.optimizer.learning_rate.assign(lr)
            order=np.random.RandomState(seed_number('order',812,epoch)).permutation(fitting)
            logs=[]
            for start in range(0,len(order),batch):
                sel=order[start:start+batch];x=augment(pixels[sel],sel,epoch)
                value=model.train_on_batch(x,{'binary_output':(y5[sel]>0).astype(int),'subtype_output':np.maximum(y5[sel]-1,0)},
                    sample_weight={'binary_output':wb[sel]/batch,'subtype_output':wm[sel]/batch},return_dict=True)
                if not np.isfinite(list(value.values())).all():raise ValueError('Matched training produced nonfinite loss')
                logs.append(value)
            entry=dict(epoch=epoch+1,learning_rate=lr,loss=float(np.mean([v['loss']for v in logs])),
                binary_loss=float(np.mean([v['binary_output_loss']for v in logs])),
                malignant_loss=float(np.mean([v['subtype_output_loss']for v in logs])),
                optimizer_updates=int(model.optimizer.iterations.numpy()))
            history.append(entry);writer.writerow(entry);progress.flush()
            md.write_json(out/'history.json',{'epochs':history,'candidate_epochs':[10,20,30,40],
                'field_batches_per_epoch':int(math.ceil(len(fitting)/batch))})
            manager.save(checkpoint_number=epoch+1)
            md.write_json(out/'resume'/'continuation.json',dict(provenance,completed_epoch=epoch+1,
                exact_stochastic_resume_supported=False,note='optimizer/model state preserved; no automatic continuation or reseeded dropout claim'))
            if epoch+1 in (10,20,30,40):
                index=len(candidate_scores);target=out/'candidates'/('candidate%d'%index);target.mkdir(parents=True)
                model.save_weights(target/'weights.weights.h5')
                b,m=predict_heads(model,pixels,validation);joint=md.joint_probabilities(b,m)
                md.save_predictions(target/'validation-fields.npz',validation,joint)
                np.savez_compressed(target/'validation-heads.npz',field_indices=validation,binary_probabilities2=b,malignant_probabilities4=m)
                records=md.case_records(fields,cases,audit['roles']['validation'],validation,joint,'validation')
                metric=md.summary(records);md.save_records(target/'validation-records.jsonl',records)
                md.write_json(target/'metrics.json',{'setting':epoch+1,'validation':metric})
                candidate_scores.append(metric['mean_subset_macro_f1'])
                candidate_hashes.append({p.name:file_hash(p)for p in sorted(target.iterdir())if p.is_file()})
            print(json.dumps(dict(model=args.model,**entry)),flush=True)
    selected=int(np.argmax(candidate_scores));setting=cfg['training']['candidate_epochs'][selected]
    selection=dict(provenance,status='selected validation candidate',candidate_validation_scores=candidate_scores,
        selected_index=selected,selected_setting=setting,selected_checkpoint_sha256=candidate_hashes[selected]['weights.weights.h5'],
        candidate_artifact_sha256=candidate_hashes,test_predictions=0)
    md.write_json(out/'selection.json',selection)
    md.write_json(out/'result.json',dict(provenance,status='complete training',artifact_sha256=artifacts(out),test_predictions=0))
    return selection


def test(args):
    md.verify_gate(args.root,args.audit,args.config,args.code_sha)
    cfg=md.config(args.config);tf,keras,env=runtime(cfg,True);audit=md.load_audit(args.audit,args.config)
    pixels,pixel_context=checked_pixels(args,audit)
    from magset.matched_models import build_model
    folder=Path(args.root)/'models'/args.model;selection=json.loads((folder/'selection.json').read_text());candidate=folder/'candidates'/('candidate%d'%selection['selected_index'])
    if file_hash(candidate/'weights.weights.h5')!=selection['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    out=Path(args.output)
    if out.exists():raise ValueError('Refusing existing selected test run')
    out.mkdir(parents=True);model=build_model(args.model,**model_paths(args));model.load_weights(candidate/'weights.weights.h5')
    with np.load(candidate/'validation-heads.npz',allow_pickle=False)as heads:
        indices=heads['field_indices'][:16];b,m=predict_heads(model,pixels,indices)
        db=float(np.max(np.abs(b-heads['binary_probabilities2'][:16])));dm=float(np.max(np.abs(m-heads['malignant_probabilities4'][:16])))
    if max(db,dm)>1e-5:raise ValueError('Selected real checkpoint validation replay failed')
    md.write_json(out/'checkpoint_reload.json',dict(field_indices=indices.tolist(),images=len(indices),
        maximum_binary_abs=db,maximum_subtype_abs=dm,atol=1e-5,checkpoint_sha256=selection['selected_checkpoint_sha256'],role='validation'))
    wanted=set(audit['roles']['test']);indices=np.array([f['index']for f in audit['fields']if f['case_index']in wanted],np.int64)
    b,m=predict_heads(model,pixels,indices);joint=md.joint_probabilities(b,m)
    records=md.case_records(audit['fields'],audit['cases'],audit['roles']['test'],indices,joint,'test')
    md.save_predictions(out/'fields.npz',indices,joint);md.save_records(out/'records.jsonl',records);md.write_json(out/'metrics.json',md.summary(records))
    md.write_json(out/'environment.json',env)
    value=dict(md.context(args.code_sha,args.config),**pixel_context,status='complete selected test inference',model=args.model,
        input_audit_sha256=file_hash(Path(args.audit)/'private.json'),selection_sha256=file_hash(folder/'selection.json'),
        gate_sha256=file_hash(Path(args.root)/'test-gate.json'),artifact_sha256=artifacts(out))
    md.write_json(out/'test.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('smoke','train','test','gate','p-select','p-test'))
    for key in ('root','audit','pixels','cache','config','output','cnn-weights','swin-prepared'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--model',choices=('cnn','swin'),default='cnn');a=p.parse_args()
    if a.action=='gate':value=md.seal_gate(a.root,a.audit,a.config,a.code_sha)
    elif a.action.startswith('p-'):
        from magset import matched_p
        value=(matched_p.select if a.action=='p-select'else matched_p.test)(a)
    else:value={'smoke':smoke,'train':train,'test':test}[a.action](a)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()

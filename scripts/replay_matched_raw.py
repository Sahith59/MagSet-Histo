"""Recompute the frozen matched results from PNG bytes, without fitting or selection."""
import argparse
import hashlib
import importlib.metadata
from io import BytesIO
from itertools import combinations
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

PRODUCER='a6e6a54e4fac090051837df17b0dea14efca3bb3'
RESULT_SHA='3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62'
SUBSETS=tuple(s for n in range(1,5)for s in combinations(range(4),n))
CODES={'A':0,'F':0,'PT':0,'TA':0,'DC':1,'LC':2,'MC':3,'PC':4}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def crop_png(data,row):
    require(hashlib.sha256(data).hexdigest()==row['sha256'],'Original PNG hash differs')
    with Image.open(BytesIO(data))as im:
        im.load();require(im.mode=='RGB' and im.format=='PNG','Unexpected image mode or format')
        require(im.size==(int(row['width']),int(row['height'])),'Original geometry differs')
        require(im.size in ((700,460),(700,456)),'Unaudited raw geometry')
        w=int(im.width*224/im.height);im=im.resize((w,224),resample=Image.Resampling.BILINEAR)
        lefts=(0,round((w-224)/2),w-224)
        return np.stack([np.asarray(im.crop((x,0,x+224,224)),np.uint8)for x in lefts])


def checked_label(row,case):
    match=re.fullmatch(r'SOB_([BM])_(A|F|PT|TA|DC|LC|MC|PC)-.+-(40|100|200|400)-\d+\.png',Path(row['relative_path']).name)
    require(match is not None,'Unrecognized original filename')
    broad,code,mag=match.groups();label=CODES[code]
    require((broad=='M')==(label>0),'Filename binary/type contradiction')
    require(int(mag)==int(row['magnification'])==(40,100,200,400)[row['view']],'Filename/manifest view contradiction')
    require(code==row['subtype_code'] and label==case['truth5'],'Filename/manifest/case target contradiction')
    require(('/malignant/'if label else'/benign/')in('/'+row['relative_path']),'Directory/filename contradiction')
    return label


def compose(binary,malignant):
    b=np.asarray(binary,dtype=np.float64);m=np.asarray(malignant,dtype=np.float64)
    require(b.shape==(len(b),2) and m.shape==(len(b),4),'Head dimensions differ')
    require(all(np.isfinite(a).all()and(a>=0).all()and(a<=1).all()and np.allclose(a.sum(1),1,atol=1e-5,rtol=0)for a in (b,m)),'Invalid head probabilities')
    q=np.column_stack((b[:,0],b[:,1,None]*m));return q/q.sum(1,keepdims=True)


def score_fields(audit,indices,probabilities):
    cases=audit['roles']['test'];lookup={int(i):p for i,p in zip(indices,probabilities)}
    views=np.empty((len(cases),4,5))
    for r,c in enumerate(cases):
        for v in range(4):
            keys=[f['index']for f in audit['fields']if f['case_index']==c and f['view']==v]
            views[r,v]=np.mean([lookup[k]for k in keys],axis=0)
    p=np.stack([views[:,s].mean(1)for s in SUBSETS],axis=1);p/=p.sum(-1,keepdims=True)
    pred=np.where(p[:,:,1:].sum(-1)>=.5,1+p[:,:,1:].argmax(-1),0)
    truth=np.asarray([audit['cases'][c]['truth5']for c in cases]);summaries=[]
    for col in range(15):
        cm=np.zeros((5,5),int);np.add.at(cm,(truth,pred[:,col]),1)
        den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(5),where=den>0)
        summaries.append({'accuracy':float(np.trace(cm)/len(truth)),'macro_f1':float(f1.mean()),'binary_correct':int(np.sum((truth>0)==(pred[:,col]>0)))})
    return pred,{'cases':len(cases),'events':len(cases)*15,'mean_subset_accuracy':float(np.mean([s['accuracy']for s in summaries])),'mean_subset_macro_f1':float(np.mean([s['macro_f1']for s in summaries])),'mean_binary_accuracy':float(np.mean([s['binary_correct']/len(cases)for s in summaries])),'all_four_accuracy':summaries[-1]['accuracy']}


def raw_batches(audit,indices,pixels,batch):
    base=Path(audit['raw_root']).resolve()
    for start in range(0,len(indices),batch):
        selected=indices[start:start+batch];arrays=[]
        for index in selected:
            row=audit['fields'][int(index)];checked_label(row,audit['cases'][row['case_index']])
            path=(base/row['relative_path']).resolve();require(path.is_relative_to(base),'Image path escaped raw root')
            crop=crop_png(path.read_bytes(),row)
            require(np.array_equal(crop,pixels[index]),'Raw PNG crops differ from original shared cache')
            arrays.append(crop)
        yield selected,np.stack(arrays)


def neural(args,audit,indices,pixels,selection):
    os.environ['TF_USE_LEGACY_KERAS']='1'
    import tensorflow as tf
    import tf_keras as keras
    from magset.matched_models import build_model
    require(tf.__version__=='2.15.1' and keras.__version__=='2.15.1','Pinned neural runtime differs')
    keras.utils.set_random_seed(812);gpus=tf.config.list_physical_devices('GPU');require(len(gpus)==1,'One allocated GPU required')
    require(tf.config.experimental.get_device_details(gpus[0]).get('device_name')=='NVIDIA A40','Use original A40 hardware')
    prepared=json.loads((args.cluster_root/'envs/swin-tf215-v1/prepared.json').read_text())
    model=build_model(args.model,cnn_weights=args.cluster_root/'cache/keras-notebook-tf215/models/efficientnetv2-b0_notop.h5',swin_snapshot=Path(prepared['snapshot_path']))
    weight=args.study/'models'/args.model/'candidates'/('candidate%d'%selection['selected_index'])/'weights.weights.h5'
    require(digest(weight)==selection['selected_checkpoint_sha256'],'Selected weights changed');model.load_weights(weight)
    @tf.function(input_signature=[tf.TensorSpec((None,3,224,224,3),tf.float32)])
    def forward(x):return model(x,training=False)
    values=[]
    for selected,crops in raw_batches(audit,indices,pixels,16):
        out=forward(crops.astype(np.float32));values.append(compose(out['binary_output'].numpy(),out['subtype_output'].numpy()))
        print(json.dumps({'model':args.model,'raw_fields':int(selected[-1]),'completed':sum(map(len,values))}),flush=True)
    return np.concatenate(values),{'gpu':'NVIDIA A40','tensorflow':tf.__version__,'tf_keras':keras.__version__,'checkpoint_sha256':digest(weight)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('cnn','swin'),required=True)
    for key in ('producer-source','study','cluster-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();require(not a.output.exists(),'Refusing existing proof output');a.output.mkdir(parents=True)
    result={'status':'raw replay failed','model':a.model,'new_fits':0,'new_selections':0,'producer_code_sha':PRODUCER,'verifier_sha256':digest(__file__)}
    try:
        proof_path=Path(__file__).resolve().parents[1]/'configs/breakhis-matched-raw-replay-v1.json';proof=json.loads(proof_path.read_text())
        versions={k:importlib.metadata.version(k)for k in proof['runtime']}
        require(versions==proof['runtime'],'Pinned proof runtime differs')
        result.update(proof_config_sha256=digest(proof_path),runtime=versions)
        require(digest(a.producer_source/'configs/breakhis-matched-three-way-v1.json')==proof['original_config_sha256'],'Frozen model configuration changed')
        require((a.producer_source/'.source-sha').read_text().strip()==PRODUCER,'Frozen producer identity differs')
        checked=a.study/'checked/result.json';require(digest(checked)==RESULT_SHA,'Accepted comparison bytes differ');baseline=json.loads(checked.read_text())
        for name,expected in baseline['implementation_sha256'].items():require(digest(a.producer_source/name)==expected,'Frozen scientific source changed')
        sys.path.insert(0,str(a.producer_source.resolve()))
        audit_path=a.study/'input-audit/private.json';require(digest(audit_path)==baseline['input_audit_sha256'],'Original audit changed');audit=json.loads(audit_path.read_text())
        require(digest(a.study/'test-gate.json')==baseline['test_gate_sha256'],'Original gate changed')
        folder=a.study/'models'/a.model;sel_path=folder/'selection.json';require(digest(sel_path)==baseline['models'][a.model]['selection_sha256'],'Selected candidate changed');sel=json.loads(sel_path.read_text())
        test_path=folder/'test/test.json';require(digest(test_path)==proof['test_artifacts'][a.model]['test_receipt_sha256'],'Original selected-test receipt changed');test=json.loads(test_path.read_text())
        require(test['status']=='complete selected test inference'and test['model']==a.model and test['code_sha']==PRODUCER,'Original test provenance differs')
        for key,wanted in [('config_sha256',proof['original_config_sha256']),('input_audit_sha256',baseline['input_audit_sha256']),('selection_sha256',digest(sel_path)),('gate_sha256',baseline['test_gate_sha256'])]:require(test[key]==wanted,'Original test binding differs/'+key)
        field_path=folder/'test/fields.npz';require(digest(field_path)==test['artifact_sha256']['fields.npz']==proof['test_artifacts'][a.model]['field_npz_sha256'],'Original field probabilities changed')
        result.update(original_test_receipt_sha256=digest(test_path),original_field_probability_sha256=digest(field_path))
        indices=np.asarray([f['index']for f in audit['fields']if f['case_index']in set(audit['roles']['test'])],dtype=np.int64)
        require(len(indices)==1704,'Expected every original test image')
        pixels=np.load(a.study/'pixels/pixels.npy',mmap_mode='r',allow_pickle=False)
        require(pixels.shape==(7909,3,224,224,3)and pixels.dtype==np.uint8,'Original pixel cache schema differs')
        with np.load(folder/'test/fields.npz',allow_pickle=False)as saved:
            require(np.array_equal(saved['field_indices'],indices),'Original test field membership/order differs');expected=saved['joint_probabilities5'].copy()
        raw,environment=neural(a,audit,indices,pixels,sel)
        require(raw.shape==expected.shape and np.isfinite(raw).all(),'Invalid raw replay probabilities')
        maximum=float(np.max(abs(raw-expected)));require(maximum<=1e-5,'Raw replay field probabilities differ above original neural tolerance')
        pred,summary=score_fields(audit,indices,raw);expected_pred,_=score_fields(audit,indices,expected)
        require(np.array_equal(pred,expected_pred),'Raw replay changed a case/subset decision')
        for key in ('mean_subset_accuracy','mean_subset_macro_f1'):require(abs(summary[key]-baseline['models'][a.model]['test'][key])<=1e-12,'Raw replay headline score differs')
        np.savez_compressed(a.output/'raw-replay.npz',field_indices=indices,joint_probabilities5=raw,case_subset_predictions=pred)
        result['replay_npz_sha256']=digest(a.output/'raw-replay.npz')
        result.update(status='accepted independent raw-image selected-model replay',original_comparison_sha256=RESULT_SHA,input_audit_sha256=digest(audit_path),selection_sha256=digest(sel_path),raw_images=1704,raw_byte_hashes_checked=1704,filename_directory_labels_checked=1704,pixel_cache_equal_fields=1704,field_probability_max_abs=maximum,case_subset_disagreements=0,summary=summary,environment=environment,elapsed_seconds=time.monotonic()-start)
    except Exception as error:
        import traceback
        result.update(error_type=type(error).__name__,error=str(error));(a.output/'private-traceback.txt').write_text(traceback.format_exc())
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['status'].startswith('accepted ')else 1


if __name__=='__main__':raise SystemExit(main())

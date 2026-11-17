"""Shared grouping, pixels, weighting and case metrics for the three-way study."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold

from .baseline import SUBSETS
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .five_metrics import FINAL5, MAP8_TO5, summarize_records

MODELS=('midnight_p','cnn','swin')
ROLES=('train','validation','test')
GEOMETRY={(700,460):((340,224),(0,58,116)),(700,456):((343,224),(0,60,119))}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def config(path):
    fixed=Path(__file__).resolve().parents[1]/'configs/breakhis-matched-three-way-v1.json'
    if file_hash(path)!=file_hash(fixed):raise ValueError('Use immutable matched configuration')
    value=json.loads(Path(path).read_text())
    if (value['models']!=list(MODELS) or value['labels']!=list(FINAL5)
            or value['training']['candidate_epochs']!=[10,20,30,40]
            or value['selection']['opportunities_per_model']!=4 or value['split']['shuffle'] is not False):
        raise ValueError('Matched recipe contract differs')
    return value


def context(code_sha,config_path):
    root=Path(__file__).resolve().parents[1]
    if not re.fullmatch('[a-f0-9]{40}',code_sha or '')or(root/'.source-sha').read_text().strip()!=code_sha:
        raise ValueError('Immutable matched source required')
    return dict(code_sha=code_sha,source_archive_sha256=(root/'.archive-sha256').read_text().strip(),
                config_sha256=file_hash(config_path),
                implementation_sha256={n:file_hash(root/n)for n in ['magset/matched_data.py','magset/matched_models.py',
                    'magset/matched_p.py','magset/swin_compat.py','scripts/prepare_matched_three_way.py',
                    'scripts/run_matched_three_way.py','scripts/collect_matched_three_way.py','magset/prior_centered_model.py',
                    'magset/kernel_model.py','magset/field_five.py','magset/five_metrics.py']})


def indexed_rows(rows):
    lookup={};cases=[];fields=[]
    for row in sorted(rows,key=lambda r:r['relative_path']):
        key=case_id(row);label=MAP8_TO5[CLASS_NAMES.index(row['subtype'])]
        if key not in lookup:
            lookup[key]=dict(case_id=key,group=row['patient_group_conservative'],truth5=label,truth8=CLASS_NAMES.index(row['subtype']))
        if lookup[key]!=dict(case_id=key,group=row['patient_group_conservative'],truth5=label,truth8=CLASS_NAMES.index(row['subtype'])):
            raise ValueError('Provider case has contradictory identity/label')
    cases=sorted(lookup.values(),key=lambda c:c['case_id'])
    index={c['case_id']:i for i,c in enumerate(cases)}
    for row in sorted(rows,key=lambda r:r['relative_path']):
        fields.append(dict(row,index=len(fields),case_index=index[case_id(row)],view=MAGNIFICATIONS.index(int(row['magnification']))))
    for i in range(len(cases)):
        if {r['view']for r in fields if r['case_index']==i}!=set(range(4)):
            raise ValueError('All cases must have four audited views')
    return cases,fields


def split_cases(cases,seed=812,require_version=True):
    import sklearn
    if require_version and sklearn.__version__!='1.6.1':raise ValueError('Pinned sklearn1.6.1 required for registered split')
    labels=np.array([c['truth5']for c in cases],dtype=np.int64)
    groups=np.array([c['group']for c in cases])
    unique=sorted(set(groups));permutation=np.random.RandomState(seed).permutation(len(unique))
    mapping=dict(zip(unique,permutation));codes=np.array([mapping[g]for g in groups])
    fold=np.full(len(cases),-1,dtype=np.int64)
    for number,(_,held)in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=False).split(np.zeros((len(cases),1)),labels,codes)):
        fold[held]=number
    if np.any(fold<0):raise ValueError('Incomplete group folds')
    for test in range(5):
        for val in range(5):
            if val==test:continue
            roles={'train':np.flatnonzero((fold!=test)&(fold!=val)),
                   'validation':np.flatnonzero(fold==val),'test':np.flatnonzero(fold==test)}
            if any(np.any(np.bincount(labels[ix],minlength=5)==0)for ix in roles.values()):continue
            if any(len(set(groups[roles['train']][labels[roles['train']]==label]))<2 for label in range(5)):continue
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                if set(groups[roles[a]])&set(groups[roles[b]]):raise ValueError('Conservative groups overlap')
            return roles,fold,{'test_fold':test,'validation_fold':val,'group_code_permutation':permutation.tolist()}
    raise ValueError('Fixed seed has no class-complete eligible role pair; no outcome-driven fallback allowed')


def split_summary(cases,fields,roles):
    return {role:{'cases':len(ix),'groups':len({cases[i]['group']for i in ix}),
        'class_cases':np.bincount([cases[i]['truth5']for i in ix],minlength=5).tolist(),
        'class_groups':[len({cases[i]['group']for i in ix if cases[i]['truth5']==label})for label in range(5)],
        'images':sum(f['case_index']in set(ix)for f in fields),
        'class_images':np.bincount([cases[f['case_index']]['truth5']for f in fields if f['case_index']in set(ix)],minlength=5).tolist()}
        for role,ix in roles.items()}


def crop_pixels(image):
    if image.mode!='RGB'or image.size not in GEOMETRY:raise ValueError('Unexpected raw image geometry/mode')
    shape,lefts=GEOMETRY[image.size]
    resized=image.resize(shape,resample=Image.Resampling.BILINEAR)
    return np.stack([np.asarray(resized.crop((left,0,left+224,224)),dtype=np.uint8)for left in lefts])


def field_weights(cases,fields,train):
    train=set(map(int,train));selected=[f for f in fields if f['case_index']in train]
    counts={}
    for f in selected:
        key=(f['case_index'],f['view']);counts[key]=counts.get(key,0)+1
    case5=np.array([cases[i]['truth5']for i in sorted(train)])
    binary=np.bincount((case5>0).astype(int),minlength=2)
    malignant=np.bincount(case5[case5>0]-1,minlength=4)
    if np.any(binary==0)or np.any(malignant==0):raise ValueError('All fitting head classes required')
    weight_b=np.zeros(len(fields),np.float64);weight_m=np.zeros(len(fields),np.float64)
    for f in selected:
        y=cases[f['case_index']]['truth5'];mass=1/(4*counts[f['case_index'],f['view']])
        weight_b[f['index']]=mass/(2*binary[int(y>0)])
        if y>0:weight_m[f['index']]=mass/(4*malignant[y-1])
    if not np.isclose(weight_b.sum(),1)or not np.isclose(weight_m.sum(),1):raise ValueError('Head masses must sum to one')
    # Uniform fields with SUM(weight*CE)/configured batch size estimates this objective.
    return weight_b*len(selected),weight_m*len(selected)


def joint_probabilities(binary,malignant):
    b=np.asarray(binary,dtype=np.float64);m=np.asarray(malignant,dtype=np.float64)
    for p,width in [(b,2),(m,4)]:
        if p.ndim!=2 or p.shape[1]!=width or not np.isfinite(p).all()or np.any(p<0)or np.any(p>1)or not np.allclose(p.sum(1),1,atol=1e-5,rtol=0):
            raise ValueError('Invalid normalized head probabilities')
    if len(b)!=len(m):raise ValueError('Head rows differ')
    out=np.column_stack((b[:,0],b[:,1,None]*m));out/=out.sum(1,keepdims=True)
    return out


def case_records(fields,cases,indices,field_indices,probabilities,role):
    ix=np.asarray(field_indices,dtype=int);p=np.asarray(probabilities,dtype=np.float64)
    if ix.ndim!=1 or len(set(ix))!=len(ix)or p.shape!=(len(ix),5)or not np.isfinite(p).all()or np.any(p<0)or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0):
        raise ValueError('Invalid field prediction archive')
    wanted={f['index']for f in fields if f['case_index']in set(map(int,indices))}
    if set(ix)!=wanted:raise ValueError('Predicted fields do not exactly cover role cases')
    by={(int(i)):q for i,q in zip(ix,p)};records=[]
    for i in sorted(map(int,indices)):
        views=[]
        for view in range(4):
            keys=[f['index']for f in fields if f['case_index']==i and f['view']==view]
            views.append(np.mean([by[k]for k in keys],axis=0))
        for subset in SUBSETS:
            q=np.mean([views[v]for v in subset],axis=0);q/=q.sum()
            # Stable reporting floor only; routing uses original pooled values.
            score=float(q[1:].sum());pred=0 if score<.5 else 1+int(q[1:].argmax())
            stable=np.maximum(q,np.finfo(np.float64).tiny);stable/=stable.sum()
            log5=np.log(stable);logmal=logsumexp(log5[1:]);mal=log5[1:]-logmal
            records.append(dict(role=role,case_id=cases[i]['case_id'],truth=cases[i]['truth8'],
                truth5=cases[i]['truth5'],magnifications=[MAGNIFICATIONS[v]for v in subset],
                probabilities5=stable.tolist(),log_probabilities5=log5.tolist(),prediction=pred,binary_score=score,
                binary_log_probabilities=[float(log5[0]),float(logmal)],
                malignant_probabilities4=np.exp(mal).tolist(),malignant_log_probabilities=mal.tolist()))
    return records


def summary(records):
    return summarize_records(records,'dual',allowed_roles=ROLES)


def save_predictions(path,field_indices,joint):
    np.savez_compressed(path,field_indices=np.asarray(field_indices,np.int64),joint_probabilities5=np.asarray(joint,np.float64))


def save_records(path,records):
    Path(path).write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in records))


def load_audit(folder,config_path):
    folder=Path(folder);value=json.loads((folder/'private.json').read_text());cfg=config(config_path)
    public=json.loads((folder/'public.json').read_text())
    if (public['private_audit_sha256']!=file_hash(folder/'private.json')
            or value['config_sha256']!=file_hash(config_path)or value['manifest_sha256']!=cfg['manifest_sha256']
            or value['indices_sha256']!=file_hash(folder/'indices.npz')):
        raise ValueError('Matched input audit binding differs')
    cases,fields=value['cases'],value['fields']
    roles,fold,choice=split_cases(cases)
    if ({k:v.tolist()for k,v in roles.items()}!=value['roles']or fold.tolist()!=value['fold_assignment']or choice!=value['split_choice']):
        raise ValueError('Registered grouped split changed')
    with np.load(folder/'indices.npz',allow_pickle=False)as saved:
        expected={'field_case_index':np.array([f['case_index']for f in fields]),'field_view':np.array([f['view']for f in fields]),
                  'case_y5':np.array([c['truth5']for c in cases]),'case_y8':np.array([c['truth8']for c in cases]),'case_fold':fold,
                  **{k+'_cases':v for k,v in roles.items()}}
        if set(saved.files)!=set(expected)or any(not np.array_equal(saved[k],v)for k,v in expected.items()):
            raise ValueError('Audit index arrays changed')
    return value


def verified_selections(root,audit,config_path,code_sha):
    root=Path(root);config(config_path);current=context(code_sha,config_path);accepted=load_audit(audit,config_path)
    values={};hashes={}
    for model in MODELS:
        folder=root/'models'/model;path=folder/'selection.json';value=json.loads(path.read_text())
        if value['status']!='selected validation candidate'or value['model']!=model or value['test_predictions']!=0:
            raise ValueError('All three validation selections must be complete before test access')
        for key in ('code_sha','source_archive_sha256','config_sha256','implementation_sha256'):
            if value[key]!=current[key]:raise ValueError('Selected model source/configuration differs')
        completed=json.loads((folder/'result.json').read_text())
        if completed.get('status')!='complete training'or completed.get('model')!=model:
            raise ValueError('All three training result receipts must be complete')
        if value['input_audit_sha256']!=file_hash(Path(audit)/'private.json'):raise ValueError('Selected input audit differs')
        scores=value['candidate_validation_scores']
        if len(scores)!=4 or not np.isfinite(scores).all()or value['selected_index']!=int(np.argmax(scores)):
            raise ValueError('First maximum among four validation candidates required')
        settings=config(config_path)['midnight_p']['temperatures']if model=='midnight_p'else config(config_path)['training']['candidate_epochs']
        if value['selected_setting']!=settings[value['selected_index']]:raise ValueError('Selected setting differs')
        if len(value['candidate_artifact_sha256'])!=4:raise ValueError('All candidate artifacts required')
        for i,receipt in enumerate(value['candidate_artifact_sha256']):
            cfolder=folder/'candidates'/('candidate%d'%i)
            for name,digest in receipt.items():
                if Path(name).name!=name or file_hash(cfolder/name)!=digest:raise ValueError('Candidate artifact changed')
            fields=np.load(cfolder/'validation-fields.npz',allow_pickle=False)
            records=case_records(accepted['fields'],accepted['cases'],accepted['roles']['validation'],fields['field_indices'],fields['joint_probabilities5'],'validation')
            metric=summary(records)
            if abs(metric['mean_subset_macro_f1']-scores[i])>1e-12:raise ValueError('Candidate validation replay differs')
        values[model]=value;hashes[model]=file_hash(path)
    return values,hashes,current


def seal_gate(root,audit,config_path,code_sha):
    _,hashes,current=verified_selections(root,audit,config_path,code_sha)
    path=Path(root)/'test-gate.json'
    value=dict(current,status='sealed all three validation selections',input_audit_sha256=file_hash(Path(audit)/'private.json'),
               selections=hashes,test_predictions=0)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Existing test gate differs')
    else:write_json(path,value)
    return value


def verify_gate(root,audit,config_path,code_sha):
    path=Path(root)/'test-gate.json'
    if not path.exists():raise ValueError('Sealed common test gate required')
    value=json.loads(path.read_text());_,hashes,current=verified_selections(root,audit,config_path,code_sha)
    expected=dict(current,status='sealed all three validation selections',input_audit_sha256=file_hash(Path(audit)/'private.json'),
                  selections=hashes,test_predictions=0)
    if value!=expected:raise ValueError('Common test gate changed')
    return value

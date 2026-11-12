"""Validation-only frozen-feature baselines using provider-defined case identities."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from .data import CLASS_NAMES, MAGNIFICATIONS, cases_from_rows, file_hash, read_manifest

SUBSETS = tuple(tuple(comb) for count in range(1,5) for comb in combinations(range(4),count))


def stratified_inner_split(cases, outer_roles, seed):
    by_class = defaultdict(list)
    for i, c in enumerate(cases):
        if outer_roles[c['case_id']] == 'train':
            by_class[c['label']].append(i)
    rng = random.Random(seed)
    validation = []
    for label in range(len(CLASS_NAMES)):
        eligible = sorted(by_class[label], key=lambda i:cases[i]['case_id'])
        if len(eligible) < 2:
            raise ValueError('Need at least two outer-training cases in every class')
        validation.append(rng.choice(eligible))
    validation = np.asarray(sorted(validation),dtype=int)
    held = set(validation.tolist())
    train = np.asarray([i for indices in by_class.values() for i in indices if i not in held],dtype=int)
    test = np.asarray([i for i,c in enumerate(cases) if outer_roles[c['case_id']]=='test'],dtype=int)
    return train, validation, test


def load_case_features(rows, cache, encoder):
    pooled = {k:[] for k in MAGNIFICATIONS}
    cases = sorted(cases_from_rows(rows).values(),key=lambda c:c['case_id'])
    metadata=[]
    manifest_paths={r['relative_path'] for r in rows}
    all_paths=set()
    for m in MAGNIFICATIONS:
        path=Path(cache)/('%s-%d.npz'%(encoder,m))
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['feature_sha256'] != file_hash(path):
            raise ValueError('Feature file checksum differs from metadata')
        metadata.append(meta)
        with np.load(path,allow_pickle=False) as values:
            keys=values['paths'].tolist(); z=values['features']
            if len(set(keys))!=len(keys) or not np.isfinite(z).all():
                raise ValueError('Feature keys duplicate or nonfinite values')
            lookup={k:z[i] for i,k in enumerate(keys)}
            if all_paths.intersection(keys):raise ValueError('Image occurs in multiple shards')
            all_paths.update(keys)
            for c in cases:
                paths=c['indices'].get(m,[])
                if not paths:raise ValueError('This initial benchmark requires four-view cases')
                pooled[m].append(np.mean([lookup[p] for p in paths],axis=0))
    if all_paths != manifest_paths:raise ValueError('Feature files do not exactly cover the manifest')
    if len({m['manifest_sha256'] for m in metadata})!=1 or len({json.dumps(m['encoder'],sort_keys=True) for m in metadata})!=1:
        raise ValueError('Feature shard provenance is inconsistent')
    return cases,np.stack([np.stack(pooled[m]) for m in MAGNIFICATIONS],axis=1),metadata


def vectorize(z, subset, kind):
    if z.ndim != 3 or z.shape[1] != 4 or not subset or len(set(subset)) != len(subset) or any(s not in range(4) for s in subset):
        raise ValueError('Expected four scale features and a nonempty unique subset')
    mask=np.zeros(4,dtype=np.float32); mask[list(subset)]=1
    if kind=='mean':
        result=z[:,subset,:].mean(axis=1)
    elif kind=='concat':
        result=np.where(mask[None,:,None].astype(bool),z,0).reshape(len(z),-1)
    else:raise ValueError('Unsupported fusion kind')
    return np.concatenate([result,np.broadcast_to(mask,(len(z),4))],axis=1)


def fit_models(z,y,kind,C):
    if kind=='late':
        models=[LogisticRegression(C=C,class_weight='balanced',max_iter=1000).fit(z[:,m,:],y) for m in range(4)]
    else:
        x=np.concatenate([vectorize(z,s,kind) for s in SUBSETS])
        models=[LogisticRegression(C=C,class_weight='balanced',max_iter=1000).fit(x,np.tile(y,len(SUBSETS)),sample_weight=np.full(len(x),1.0/len(SUBSETS)))]
    return models


def predict_models(models,z,kind,subset):
    if kind=='late':return np.mean([models[m].predict_proba(z[:,m,:]) for m in subset],axis=0)
    return models[0].predict_proba(vectorize(z,subset,kind))


def evaluate(models,z,y,kind):
    result={}
    for subset in SUBSETS:
        prob=predict_models(models,z,kind,subset);pred=prob.argmax(axis=1)
        name='+'.join(str(MAGNIFICATIONS[k]) for k in subset)
        result[name]={'macro_f1':float(f1_score(y,pred,labels=np.arange(8),average='macro',zero_division=0)),
                      'accuracy':float(accuracy_score(y,pred)),
                      'balanced_accuracy':float(balanced_accuracy_score(y,pred))}
    scores=[v['macro_f1'] for v in result.values()]
    return {'mean_subset_macro_f1':float(np.mean(scores)), 'worst_subset_macro_f1':float(min(scores)),
            'subsets':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--encoder',choices=['phikon-v2','resnet50'],required=True)
    p.add_argument('--method',choices=['mean','concat','late'],required=True)
    p.add_argument('--trial',type=int,choices=range(1,6),required=True);p.add_argument('--seed',type=int,default=812)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--code-sha',required=True)
    p.add_argument('--evaluate-test',action='store_true',help='Explicitly enable outer evaluation after protocol freeze')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Run output already exists; refusing to overwrite')
    rows=read_manifest(a.manifest);protocol=json.loads(a.protocol.read_text());mh=file_hash(a.manifest)
    if protocol['manifest_sha256']!=mh:raise ValueError('Protocol and manifest differ')
    cases,z,metadata=load_case_features(rows,a.features,a.encoder)
    if metadata[0]['manifest_sha256']!=mh:raise ValueError('Features and current manifest differ')
    y=np.asarray([c['label'] for c in cases]);roles=protocol['trials'][str(a.trial)]['case_roles']
    train,val,test=stratified_inner_split(cases,roles,a.seed)
    # Only outer-training/inner-training cases fit normalization; then all absent
    # standardized views are zeroed by vectorize rather than contributing pixels.
    mean=z[train].mean(axis=0);std=z[train].std(axis=0);std=np.where(std>1e-6,std,1.0)
    z=(z-mean)/std
    candidates=[];best=None
    for C in (0.01,0.1,1.0,10.0):
        models=fit_models(z[train],y[train],a.method,C)
        metric=evaluate(models,z[val],y[val],a.method)
        candidates.append({'C':C,**metric})
        if best is None or metric['mean_subset_macro_f1']>best[0]:best=(metric['mean_subset_macro_f1'],C,models,metric)
    _,C,models,metric=best
    a.output.mkdir(parents=True)
    arrays={'mean':mean,'std':std}
    for i,model in enumerate(models):
        arrays['coef_%d'%i]=model.coef_;arrays['intercept_%d'%i]=model.intercept_;arrays['classes_%d'%i]=model.classes_
    np.savez_compressed(a.output/'linear-model.npz',**arrays)
    report={'status':'validation-only pilot' if not a.evaluate_test else 'outer evaluation',
            'unit':protocol['unit'],'encoder':a.encoder,'method':a.method,'trial':a.trial,'seed':a.seed,
            'code_sha':a.code_sha,'manifest_sha256':mh,'protocol_sha256':file_hash(a.protocol),
            'class_names':CLASS_NAMES,'selected_C':C,'validation':metric,'candidates':candidates,
            'case_ids':{role:[cases[i]['case_id'] for i in indices] for role,indices in [('train',train),('validation',val),('test',test)]},
            'field_policy':'mean over all fields within each magnification; identical across fusion baselines',
            'model_selection':'fixed four-value C grid on inner validation only; outer cases excluded',
            'python':platform.python_version(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'feature_provenance':[{k:v for k,v in m.items() if k in ['encoder','manifest_sha256','feature_sha256','precision']} for m in metadata]}
    if a.evaluate_test:
        report['test']=evaluate(models,z[test],y[test],a.method)
    # Save raw score records without pickle; validation by default, test only when requested.
    with (a.output/'predictions.jsonl').open('w') as stream:
        for role,indices in [('validation',val)]+([('test',test)] if a.evaluate_test else []):
            for subset in SUBSETS:
                probabilities=predict_models(models,z[indices],a.method,subset)
                for i,prob in zip(indices,probabilities):
                    stream.write(json.dumps({'role':role,'case_id':cases[i]['case_id'],'truth':int(y[i]),
                        'magnifications':[MAGNIFICATIONS[s] for s in subset],'probabilities':prob.tolist()})+'\n')
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','encoder','method','trial','seed','selected_C','validation']},indent=2),flush=True)


if __name__=='__main__':main()

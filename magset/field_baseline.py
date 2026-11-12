"""Classify frozen individual fields, then average probabilities within case and scale."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from .baseline import SUBSETS, stratified_inner_split
from .bags import load_bags, fit_normalization, normalize_bags
from .collect import metrics_from_scores
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest


def field_training_data(bags, y, indices, scale, early_pool=False):
    counts = np.bincount(y[indices],minlength=8)
    if np.any(counts == 0):
        raise ValueError('All classes must be present in fitting cases')
    x = []; labels = []; weights = []
    for i in indices:
        fields = bags[i][scale]
        if early_pool:
            fields = fields.mean(0,keepdims=True)
        n = len(fields)
        x.append(fields); labels.extend([y[i]]*n)
        # Sum of weights equals number of cases; classes have equal total mass.
        weights.extend([len(indices)/(8*counts[y[i]]*n)]*n)
    return np.concatenate(x),np.asarray(labels),np.asarray(weights)


def case_probabilities(models, bags, indices, early_pool=False):
    return np.asarray([[models[m].predict_proba(bags[i][m].mean(0,keepdims=True) if early_pool else bags[i][m]).mean(0)
                        for m in range(4)] for i in indices])


def probability_metrics(probabilities, y):
    """probabilities: (subsets, cases, 8), ordered as baseline.SUBSETS."""
    if np.asarray(probabilities).shape != (len(SUBSETS),len(y),8):
        raise ValueError('Expected probabilities for all 15 subsets and every case')
    result = {}
    for subset,prob in zip(SUBSETS, probabilities):
        name = '+'.join(str(MAGNIFICATIONS[m]) for m in subset)
        result[name] = metrics_from_scores([{'truth':int(label),'probabilities':p.tolist()} for label,p in zip(y,prob)])
    values = [r['macro_f1'] for r in result.values()]
    return {'mean_subset_macro_f1':sum(values)/len(values),
            'worst_subset_macro_f1':min(values),'subsets':result}


def score_records(cases, indices, y, probabilities):
    for subset,prob in zip(SUBSETS,probabilities):
        for i,p in zip(indices,prob):
            yield {'role':'validation','case_id':cases[i]['case_id'],'truth':int(y[i]),
                   'magnifications':[MAGNIFICATIONS[m] for m in subset],'probabilities':p.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','features','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--encoder',choices=['phikon-v2','resnet50'],required=True)
    p.add_argument('--trial',type=int,choices=range(1,6),required=True)
    p.add_argument('--seed',type=int,default=812); p.add_argument('--code-sha',required=True)
    p.add_argument('--pool-before-classifier',action='store_true',help='Normalization-matched early-mean control')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Output exists; use a new versioned directory')
    mh = file_hash(a.manifest); protocol = json.loads(a.protocol.read_text())
    if protocol['manifest_sha256'] != mh:
        raise ValueError('Protocol and manifest differ')
    cases,bags,metadata = load_bags(read_manifest(a.manifest),a.features,a.encoder,mh)
    y = np.asarray([c['label'] for c in cases])
    train,val,test = stratified_inner_split(cases,protocol['trials'][str(a.trial)]['case_roles'],a.seed)
    mean,std = fit_normalization(bags,train); bags = normalize_bags(bags,mean,std)
    start = time.monotonic(); candidates = []; best = None
    for C in (.01,.1,1.,10.):
        models = []
        for m in range(4):
            x,labels,weights = field_training_data(bags,y,train,m,a.pool_before_classifier)
            model = LogisticRegression(C=C,max_iter=1000).fit(x,labels,sample_weight=weights)
            models.append(model)
        per_scale = case_probabilities(models,bags,val,a.pool_before_classifier)
        prob = np.asarray([per_scale[:,s,:].mean(1) for s in SUBSETS])
        metric = probability_metrics(prob,y[val]); candidates.append({'C':C,**metric})
        if best is None or metric['mean_subset_macro_f1'] > best[0]:
            best = metric['mean_subset_macro_f1'],C,models,metric,prob
    _,C,models,metric,prob = best
    a.output.mkdir(parents=True)
    arrays = {'mean':mean,'std':std}
    for m,model in enumerate(models):
        arrays.update({'coef_%d'%m:model.coef_,'intercept_%d'%m:model.intercept_,'classes_%d'%m:model.classes_})
    np.savez_compressed(a.output/'linear-model.npz',**arrays)
    # Check the portable coefficient artifact against sklearn's probabilities.
    with np.load(a.output/'linear-model.npz',allow_pickle=False) as saved:
        for m,model in enumerate(models):
            for i in val:
                fields = bags[i][m].mean(0,keepdims=True) if a.pool_before_classifier else bags[i][m]
                scores = fields@saved['coef_%d'%m].T+saved['intercept_%d'%m]
                scores = np.exp(scores-scores.max(1,keepdims=True)); scores /= scores.sum(1,keepdims=True)
                if not np.array_equal(saved['classes_%d'%m],np.arange(8)) or not np.allclose(scores,model.predict_proba(fields),atol=1e-10):
                    raise ValueError('Reloaded linear scores differ')
    report = {'status':'validation-only pilot','unit':protocol['unit'],'encoder':a.encoder,
              'method':'field-early-late' if a.pool_before_classifier else 'field-late',
              'trial':a.trial,'seed':a.seed,'code_sha':a.code_sha,'manifest_sha256':mh,
              'protocol_sha256':file_hash(a.protocol),'class_names':CLASS_NAMES,
              'selected_C':C,'validation':metric,'candidates':candidates,
              'case_ids':{role:[cases[i]['case_id'] for i in idx] for role,idx in [('train',train),('validation',val),('test',test)]},
              'normalization':'training-case-balanced field first/second moments per scale; no held-out cases',
              'field_policy':('mean field features then per-scale classification; mean available scale probabilities' if a.pool_before_classifier else
                              'all fields; case/class-balanced field CE; mean field probabilities then mean available views'),
              'model_selection':'four fixed C settings; validation only; no test scoring supported',
              'elapsed_seconds':time.monotonic()-start,'python':platform.python_version(),
              'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
              'reload_verified':True,'model_sha256':file_hash(a.output/'linear-model.npz'),
              'feature_provenance':[{k:v for k,v in m.items() if k in ['encoder','manifest_sha256','feature_sha256','precision']} for m in metadata]}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in score_records(cases,val,y,prob)))
    print(json.dumps({'encoder':a.encoder,'trial':a.trial,'selected_C':C,'validation':metric},indent=2),flush=True)


if __name__ == '__main__':
    main()

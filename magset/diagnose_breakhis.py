"""Fixed no-refit diagnosis of saved BreakHis fitting/validation artifacts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax, logsumexp
import torch
from torch import nn

from .adapt_breakhis import prepare_context, bound_checkpoint, sampling_weights, checked_difference
from .breakhis_predict import cached_validation_fields, load_head, predict_fields
from .collect import SUBSETS
from .collect_adaptation import verify_one
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .baseline import vectorize

READOUTS=('frozen_case_head','adapted_case_head','aux_mean_logits','aux_mean_probabilities')


def metrics(probabilities,labels,weights=None,log_probabilities=None):
    p=np.asarray(probabilities,dtype=np.float64);y=np.asarray(labels)
    if (p.shape!=(len(y),8) or not len(y) or y.ndim!=1 or not np.issubdtype(y.dtype,np.integer)
            or np.any((y<0)|(y>7)) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0)):
        raise ValueError('Invalid diagnostic labels/probabilities')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    if w.shape!=y.shape or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid diagnostic weights')
    if log_probabilities is None:
        if np.any(p[np.arange(len(y)),y]==0):raise ValueError('Stable log probabilities required for underflowed true probability')
        lp=np.log(np.maximum(p,np.finfo(np.float64).tiny))
    else:
        lp=np.asarray(log_probabilities,dtype=np.float64)
        if lp.shape!=p.shape or not np.isfinite(lp).all() or not np.allclose(np.exp(lp),p,atol=1e-12,rtol=1e-10):
            raise ValueError('Log probabilities differ')
    pred=p.argmax(1);matrix=np.zeros((8,8));np.add.at(matrix,(y,pred),w)
    actual=matrix.sum(1);called=matrix.sum(0);tp=matrix.diagonal()
    recall=np.divide(tp,actual,out=np.zeros(8),where=actual>0)
    f1=np.divide(2*tp,actual+called,out=np.zeros(8),where=(actual+called)>0)
    return {'accuracy':float(tp.sum()/w.sum()),'balanced_accuracy':float(recall[actual>0].mean()),
            'macro_f1':float(f1.mean()),'cross_entropy':float(np.average(-lp[np.arange(len(y)),y],weights=w)),
            'brier':float(np.average(np.square(p-np.eye(8)[y]).sum(1),weights=w)),
            'class_recall':recall.tolist(),'confusion_matrix':matrix.tolist(),
            'zero_true_probability_records':int(np.sum(p[np.arange(len(y)),y]==0)),'records':len(y)}


def auxiliary_scores(features,state):
    x=np.asarray(features,dtype=np.float64);w=state['1.weight'].detach().cpu().numpy().astype(np.float64)
    b=state['1.bias'].detach().cpu().numpy().astype(np.float64)
    if x.ndim!=2 or w.shape!=(8,x.shape[1]) or b.shape!=(8,) or not all(np.isfinite(a).all() for a in (x,w,b)):
        raise ValueError('Auxiliary raw feature/head schema differs')
    logits=x@w.T+b;lp=logits-logsumexp(logits,axis=1,keepdims=True)
    return logits,np.exp(lp),lp


def auxiliary_case(fields,views,state):
    views=tuple(views)
    if not views or len(set(views))!=len(views) or not set(views).issubset(MAGNIFICATIONS):
        raise ValueError('Invalid auxiliary views')
    means=[];mean_logits=[];mean_probabilities=[];log_mean_probabilities=[]
    for m in views:
        x=np.asarray(fields[m],dtype=np.float64)
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid auxiliary field bag')
        logits,prob,lp=auxiliary_scores(x,state)
        means.append(x.mean(0));mean_logits.append(logits.mean(0));mean_probabilities.append(prob.mean(0))
        log_mean_probabilities.append(logsumexp(lp,axis=0)-np.log(len(x)))
    pooled_logits=auxiliary_scores(np.mean(means,axis=0)[None,:],state)[0][0]
    error=checked_difference(np.mean(mean_logits,axis=0),pooled_logits)
    logit_lp=pooled_logits-logsumexp(pooled_logits)
    mean_lp=logsumexp(log_mean_probabilities,axis=0)-np.log(len(views));mean_prob=np.mean(mean_probabilities,axis=0)
    if checked_difference(mean_prob,np.exp(mean_lp))>1e-12:raise ValueError('Stable probability average differs')
    return softmax(pooled_logits),mean_prob,error,logit_lp,mean_lp


def case_head_scores(fields,views,head):
    """Retain stable log probabilities while replaying the original scorer exactly."""
    p=predict_fields(fields,views,head);z=np.zeros((1,4,head['mean'].shape[1]),dtype=np.float32)
    for m in views:z[0,MAGNIFICATIONS.index(m)]=np.asarray(fields[m],dtype=np.float32).mean(0)
    x=vectorize((z-head['mean'])/head['std'],tuple(MAGNIFICATIONS.index(m) for m in sorted(views)),'mean')
    logits=(x@head['coef_0'].T+head['intercept_0'])[0];lp=logits-logsumexp(logits)
    if checked_difference(p,np.exp(lp))>1e-12:raise ValueError('Case-head stable log replay differs')
    return p,lp


def weighted_drift(left,right,weights):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or w.shape!=(len(a),) or not all(np.isfinite(v).all() for v in (a,b,w)) or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid paired feature geometry')
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);cos_ok=(na>1e-12)&(nb>1e-12);rel_ok=na>1e-12
    def average(values,ok):return float(np.average(values,weights=w[ok])) if w[ok].sum()>0 else None
    cos=np.clip(np.sum(a[cos_ok]*b[cos_ok],axis=1)/(na[cos_ok]*nb[cos_ok]),-1,1)
    rel=np.linalg.norm((b-a)[rel_ok],axis=1)/na[rel_ok]
    return {'cosine':average(cos,cos_ok),'relative_l2':average(rel,rel_ok),
            'cosine_valid_mass':float(w[cos_ok].sum()/w.sum()),'relative_l2_valid_mass':float(w[rel_ok].sum()/w.sum()),
            'near_zero_reference_rows':int(np.sum(~rel_ok)),'near_zero_adapted_rows':int(np.sum(nb<=1e-12))}


def case_summary(records):
    subsets={};identities={};seen=set()
    for r in records:
        key=(r['case_id'],tuple(r['magnifications']))
        if key in seen or key[1] not in SUBSETS:raise ValueError('Invalid/duplicate diagnostic subset')
        seen.add(key)
        if r['case_id'] in identities and identities[r['case_id']]!=r['truth']:raise ValueError('Diagnostic truth changed')
        identities[r['case_id']]=r['truth']
    if len(records)!=len(identities)*15:raise ValueError('Incomplete diagnostic case/subset coverage')
    for views in SUBSETS:
        rows=[r for r in records if tuple(r['magnifications'])==views]
        if not rows or len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Missing/duplicate diagnostic cases')
        y=np.array([r['truth'] for r in rows]);p=np.array([r['probabilities'] for r in rows]);lp=np.array([r['log_probabilities'] for r in rows]);counts=Counter(y)
        value=metrics(p,y,log_probabilities=lp);value['class_balanced_cross_entropy']=metrics(p,y,[1/(8*counts[k]) for k in y],lp)['cross_entropy']
        subsets['+'.join(map(str,views))]=value
    pooled=metrics([r['probabilities'] for r in records],[r['truth'] for r in records],log_probabilities=[r['log_probabilities'] for r in records]);full=subsets['40+100+200+400']
    return {'mean_subset_macro_f1':float(np.mean([r['macro_f1'] for r in subsets.values()])),
            'mean_subset_accuracy':float(np.mean([r['accuracy'] for r in subsets.values()])),
            'mean_subset_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in subsets.values()])),
            'mean_subset_class_balanced_cross_entropy':float(np.mean([r['class_balanced_cross_entropy'] for r in subsets.values()])),
            'full_view_accuracy':full['accuracy'],'full_view_balanced_accuracy':full['balanced_accuracy'],
            'worst_subset_macro_f1':min(r['macro_f1'] for r in subsets.values()),
            'class_recall':pooled['class_recall'],'confusion_matrix':pooled['confusion_matrix'],'subsets':subsets}


def case_records(cases,frozen,adapted,heads,auxiliary,role,config):
    if role not in ('train','validation'):raise ValueError('Diagnostic scoring refuses outer roles')
    result=[];error=0.
    for c in cases:
        cid=c['case_id']
        for views in SUBSETS:
            logit_prob,mean_prob,delta,logit_lp,mean_lp=auxiliary_case(adapted[cid],views,auxiliary);error=max(error,delta)
            values=(case_head_scores(frozen[cid],views,heads[0]),case_head_scores(adapted[cid],views,heads[1]),(logit_prob,logit_lp),(mean_prob,mean_lp))
            for name,(p,lp) in zip(READOUTS,values):
                metrics(p[None,:],np.array([c['label']]),log_probabilities=lp[None,:])
                result.append({'role':role,'case_id':cid,'truth':c['label'],'magnifications':views,'readout':name,
                               'probabilities':p.tolist(),'log_probabilities':lp.tolist()})
    if error>config['affine_pooling_tolerance']:raise ValueError('Affine head/pooling identity differs')
    return result,error


def replay_archive(cases,fields,head,archive):
    expected={(c['case_id'],s) for c in cases for s in SUBSETS}
    if set(archive)!=expected:raise ValueError('Validation archive coverage differs')
    error=max(checked_difference(predict_fields(fields[c['case_id']],views,head),archive[c['case_id'],views])
              for c in cases for views in SUBSETS)
    if error>1e-8:raise ValueError('Archived case probability replay differs')
    return error


def diagnose_run(manifest,protocol,reference_root,runs,cache,training_source,seed,trial,comparison,config,output):
    folder=Path(runs)/('split%d'%seed)/('trial%d'%trial)
    report,hashes=verify_one(folder,manifest,protocol,reference_root,training_source,trial,seed)
    bound=next(r for r in comparison['artifacts'] if (r['seed'],r['trial'])==(seed,trial))
    for key in ('metrics_sha256','predictions_sha256','checkpoint_sha256','feature_sha256','head_sha256','frozen_reference'):
        if hashes[key]!=bound[key]:raise ValueError('Run differs from accepted complete comparison')
    trusted,fit,val,reference,context=prepare_context(manifest,protocol,trial,seed,reference_root,report['config'],comparison['source_sha'])
    source=Path(training_source);context['implementation_sha256']=file_hash(source/'magset/adapt_breakhis.py')
    encoder=json.loads((source/'configs/encoders.json').read_text())['phikon-v2'];context.update(base_encoder=encoder,smoke=False)
    saved=bound_checkpoint(folder/'last-checkpoint.pt',context);aux=saved['auxiliary_head']
    rows=read_manifest(manifest);cases=fit+val;ids={c['case_id'] for c in cases}
    allowed_rows=[r for r in rows if case_id(r) in ids];labels={c['case_id']:c['label'] for c in cases}
    if len(fit)!=46 or len(val)!=8 or len(ids)!=54:raise ValueError('Diagnostic role membership differs')
    frozen=cached_validation_fields(rows,cases,cache,reference,encoder)
    with np.load(folder/'adapted-features.npz',allow_pickle=False) as data:
        paths=data['paths'].tolist();features=data['features'].copy()
    if paths!=[r['relative_path'] for r in allowed_rows]:raise ValueError('Diagnostic feature order differs')
    lookup={p:i for i,p in enumerate(paths)}
    adapted={c['case_id']:{m:features[[lookup[p] for p in c['indices'][m]]] for m in MAGNIFICATIONS} for c in cases}
    old_folder=Path(reference_root)/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed))
    frozen_head_sha=file_hash(old_folder/'linear-model.npz')
    heads=(load_head(old_folder/'linear-model.npz',frozen_head_sha,encoder['dimension']),
           load_head(folder/'linear-model.npz',report['head_sha256'],encoder['dimension']))
    archived={}
    for name,directory in (('frozen_case_head',old_folder),('adapted_case_head',folder)):
        archived[name]={(r['case_id'],tuple(r['magnifications'])):np.asarray(r['probabilities'])
                       for r in map(json.loads,(directory/'predictions.jsonl').read_text().splitlines())}
    replay_error=max(replay_archive(val,frozen,heads[0],archived['frozen_case_head']),
                     replay_archive(val,adapted,heads[1],archived['adapted_case_head']))
    logits,probabilities,log_probabilities=auxiliary_scores(features,aux)
    actual=nn.Sequential(nn.Dropout(report['config']['head_dropout']),nn.Linear(encoder['dimension'],8));actual.load_state_dict(aux);actual.eval()
    with torch.inference_mode():probe=torch.softmax(actual(torch.from_numpy(features[:64])),dim=1).numpy()
    torch_error=checked_difference(probe,probabilities[:64])
    if torch_error>config['aux_torch_probability_tolerance']:raise ValueError('Auxiliary Torch/NumPy probability replay differs')
    output=Path(output);output.mkdir();records=[];field_weights=np.zeros(len(paths));roles=np.full(len(paths),'',dtype='<U10')
    case_result={};field_result={};drift={};affine_error=0.
    for role,role_cases in (('train',fit),('validation',val)):
        role_ids={c['case_id'] for c in role_cases};indices=[i for i,r in enumerate(allowed_rows) if case_id(r) in role_ids]
        role_rows=[allowed_rows[i] for i in indices];weights=sampling_weights(role_rows,role_cases).numpy()
        if not np.isclose(weights.sum(),1,atol=1e-12):raise ValueError('Role weights do not sum to one')
        field_weights[indices]=weights;roles[indices]=role;truth=np.array([labels[case_id(r)] for r in role_rows])
        field_result[role]={'raw':metrics(probabilities[indices],truth,log_probabilities=log_probabilities[indices]),
                            'class_case_view_balanced':metrics(probabilities[indices],truth,weights,log_probabilities[indices])}
        scores,error=case_records(role_cases,frozen,adapted,heads,aux,role,config);records.extend(scores);affine_error=max(affine_error,error)
        case_result[role]={name:case_summary([r for r in scores if r['readout']==name]) for name in READOUTS}
        if role=='validation':
            for r in scores:
                if r['readout'] in archived:
                    replay_error=max(replay_error,checked_difference(np.array(r['probabilities']),archived[r['readout']][r['case_id'],tuple(r['magnifications'])]))
        raw_old=[];raw_new=[]
        for row in role_rows:
            c=next(c for c in role_cases if c['case_id']==case_id(row));m=int(row['magnification']);j=c['indices'][m].index(row['relative_path'])
            raw_old.append(frozen[c['case_id']][m][j]);raw_new.append(adapted[c['case_id']][m][j])
        first=[];second=[];case_weights=[];counts=Counter(c['label'] for c in role_cases)
        for c in role_cases:
            for j,m in enumerate(MAGNIFICATIONS):
                first.append((frozen[c['case_id']][m].mean(0)-heads[0]['mean'][j])/heads[0]['std'][j])
                second.append((adapted[c['case_id']][m].mean(0)-heads[1]['mean'][j])/heads[1]['std'][j])
                case_weights.append(1/(8*counts[c['label']]*4))
        drift[role]={'raw_fields':weighted_drift(raw_old,raw_new,weights),
                     'own_normalized_view_means':weighted_drift(first,second,case_weights)}
    if replay_error>1e-8:raise ValueError('Archived case probability replay differs')
    (output/'case-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    np.savez_compressed(output/'field-scores.npz',paths=np.asarray(paths),case_ids=np.array([case_id(r) for r in allowed_rows]),
        magnifications=np.array([int(r['magnification']) for r in allowed_rows]),truths=np.array([labels[case_id(r)] for r in allowed_rows]),
        roles=roles,probabilities=probabilities,log_probabilities=log_probabilities,weights=field_weights)
    result={'seed':seed,'trial':trial,'status':'saved-artifact diagnosis','case_ids':context['case_ids'],
        'case_scores':case_result,'auxiliary_field_scores':field_result,'feature_drift':drift,
        'replay':{'archived_case_probability_error':replay_error,'auxiliary_torch_probability_error':torch_error,'affine_pooling_logit_error':affine_error},
        'artifacts':{**hashes,'frozen_head_sha256':frozen_head_sha,'case_scores_sha256':file_hash(output/'case-scores.jsonl'),
                     'field_scores_sha256':file_hash(output/'field-scores.npz')},'fitting_fields':report['fitting_fields'],'encoded_fields':len(paths),
        'new_fits':0,'new_encoder_forwards':0,'outer_predictions':0}
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def summarize_runs(runs):
    if {(r['seed'],r['trial']) for r in runs}!={(s,t) for s in (812,813) for t in range(1,6)} or len(runs)!=10:
        raise ValueError('Diagnosis requires all ten unique runs')
    result=[]
    for seed in (812,813):
        group=[r for r in runs if r['seed']==seed];row={'seed':seed,'cases':{},'auxiliary_fields':{},'feature_drift':{}}
        for role in ('train','validation'):
            row['cases'][role]={}
            for name in READOUTS:
                values=[r['case_scores'][role][name] for r in group]
                row['cases'][role][name]={k:float(np.mean([r[k] for r in values])) for k in
                    ('mean_subset_macro_f1','mean_subset_accuracy','mean_subset_balanced_accuracy','mean_subset_class_balanced_cross_entropy',
                     'full_view_accuracy','full_view_balanced_accuracy','worst_subset_macro_f1')}
                row['cases'][role][name]['class_recall']=np.mean([r['class_recall'] for r in values],axis=0).tolist()
            row['auxiliary_fields'][role]={}
            for kind in ('raw','class_case_view_balanced'):
                values=[r['auxiliary_field_scores'][role][kind] for r in group]
                row['auxiliary_fields'][role][kind]={k:float(np.mean([r[k] for r in values])) for k in ('accuracy','balanced_accuracy','macro_f1','cross_entropy','brier')}
                row['auxiliary_fields'][role][kind]['class_recall']=np.mean([r['class_recall'] for r in values],axis=0).tolist()
            row['feature_drift'][role]={}
            for kind in ('raw_fields','own_normalized_view_means'):
                values=[r['feature_drift'][role][kind] for r in group]
                row['feature_drift'][role][kind]={k:float(np.mean([v[k] for v in values if v[k] is not None]))
                    if any(v[k] is not None for v in values) else None for k in ('cosine','relative_l2','cosine_valid_mass','relative_l2_valid_mass')}
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','reference-root','runs','frozen-cache','training-source','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
    source=Path(__file__).resolve().parents[1]
    if (source/'.source-sha').read_text().strip()!=a.code_sha:raise ValueError('Use bound immutable diagnostic source')
    config=json.loads(a.config.read_text())
    if config!=json.loads((Path(__file__).resolve().parents[1]/'configs/breakhis-diagnosis-v1.json').read_text()):raise ValueError('Use exact diagnostic probes')
    comparison_path=a.runs/'paired-comparison.json'
    if file_hash(comparison_path)!=config['comparison_sha256']:raise ValueError('Accepted complete comparison hash differs')
    comparison=json.loads(comparison_path.read_text())
    if comparison['source_sha']!=config['training_source_sha'] or comparison['verified_runs']!=10:
        raise ValueError('Unexpected input experiment')
    if (a.training_source/'.source-sha').read_text().strip()!=config['training_source_sha']:raise ValueError('Training snapshot differs')
    if a.output.exists():raise ValueError('Diagnosis output exists; preserve original diagnosis')
    a.output.mkdir(parents=True);started=time.monotonic();runs=[]
    for seed in (812,813):
        for trial in range(1,6):
            runs.append(diagnose_run(a.manifest,a.protocol,a.reference_root,a.runs,a.frozen_cache,a.training_source,
                seed,trial,comparison,config,a.output/('split%d-trial%d'%(seed,trial))))
            print(json.dumps({'diagnosed_seed':seed,'trial':trial}),flush=True)
    result={'status':'complete saved-artifact diagnosis','dataset':'BreakHis','class_names':CLASS_NAMES,'code_sha':a.code_sha,
            'implementation_sha256':file_hash(__file__),'config_sha256':file_hash(a.config),'comparison_sha256':file_hash(comparison_path),
            'manifest_sha256':file_hash(a.manifest),'protocol_sha256':file_hash(a.protocol),'diagnosed_runs':10,
            'summary':summarize_runs(runs),'runs':runs,'new_fits':0,'new_encoder_forwards':0,'outer_predictions':0,
            'elapsed_seconds':time.monotonic()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'interpretation':'Posthoc fixed descriptive probes on reused fitting/validation cases; no model selection, new primary result, clinical confirmation or causal proof'}
    (a.output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'diagnosed_runs':10,'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()

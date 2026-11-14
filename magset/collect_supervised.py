"""Verify the complete fixed ResNet reference and its selected field/case artifacts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from .adapt_breakhis import prepare_context,bound_checkpoint,sampling_weights
from .collect import SUBSETS,metrics_from_scores,verify_run
from .collect_adaptation import digest_state,summarize
from .data import CLASS_NAMES,MAGNIFICATIONS,case_id,file_hash,read_manifest
from .diagnose_breakhis import metrics
from .supervised_breakhis import learning_rates


def equal_metric(actual,claimed):
    if isinstance(actual,dict):
        if not isinstance(claimed,dict) or set(actual)!=set(claimed):raise ValueError('Metric keys differ')
        for k,v in actual.items():equal_metric(v,claimed[k])
    elif isinstance(actual,list):
        if not isinstance(claimed,list) or len(actual)!=len(claimed):raise ValueError('Metric shape differs')
        for a,b in zip(actual,claimed):equal_metric(a,b)
    elif type(actual) is int:
        if type(claimed) is not int or actual!=claimed:raise ValueError('Metric count differs')
    elif type(claimed) not in (int,float) or not math.isfinite(actual) or not math.isfinite(claimed) or abs(actual-claimed)>1e-12:
        raise ValueError('Finite metric arithmetic differs')


def fingerprint(state,kind):
    if kind=='bn':values={k:v for k,v in state.items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    elif kind=='backbone':values={k:v for k,v in state.items() if not k.startswith('fc.')}
    elif kind=='early':values={k:v for k,v in state.items() if k.startswith('conv1.')}
    elif kind=='late':values={k:v for k,v in state.items() if k.startswith('layer4.') and not k.endswith(('running_mean','running_var','num_batches_tracked'))}
    else:raise ValueError('Unknown component')
    if not values:raise ValueError('Missing model component')
    return digest_state(values)


def check_state(state,original,config):
    schema={k:(tuple(v.shape),v.dtype) for k,v in original.items() if not k.startswith('fc.')}
    schema.update({'fc.1.weight':((8,config['dimension']),torch.float32),'fc.1.bias':((8,),torch.float32)})
    if list(state)!=list(schema) or any(tuple(state[k].shape)!=shape or state[k].dtype!=dtype or not torch.isfinite(state[k]).all()
                                     for k,(shape,dtype) in schema.items()):raise ValueError('Full ResNet schema/dtype/finite state differs')
    if fingerprint(state,'bn')!=fingerprint(original,'bn'):raise ValueError('Pretrained BN buffers changed')


def check_optimizer(saved,config,n_fields,smoke):
    epochs=config['smoke_epochs'] if smoke else config['epochs'];b=min(math.ceil(n_fields/config['batch_size']),config['smoke_batches_per_epoch']) if smoke else math.ceil(n_fields/config['batch_size'])
    state=saved['model'];backbone=[k for k in state if not k.startswith('fc.') and not k.endswith(('running_mean','running_var','num_batches_tracked'))]
    heads=['fc.1.weight','fc.1.bias'];names=backbone+heads;n=len(backbone);opt=saved['optimizer'];groups=opt['param_groups']
    defaults={'weight_decay':config['weight_decay'],'betas':(.9,.999),'eps':1e-8,'amsgrad':False,'maximize':False,
              'foreach':None,'capturable':False,'differentiable':False,'fused':None}
    if len(groups)!=2 or set(opt['state'])!=set(range(n+2)):raise ValueError('Optimizer coverage differs')
    for group,lr,ids in zip(groups,learning_rates(config,epochs),(list(range(n)),[n,n+1])):
        if group!={**defaults,'lr':lr,'params':ids}:raise ValueError('AdamW settings/order differ')
    for i,name in enumerate(names):
        item=opt['state'][i];updates=b*(epochs-config['head_only_epochs'] if i<n else epochs)
        if (set(item)!={'step','exp_avg','exp_avg_sq'} or item['step'].shape!=torch.Size([]) or item['step'].dtype!=torch.float32
                or not torch.isfinite(item['step']) or item['step'].item()!=updates
                or any(item[k].shape!=state[name].shape or item[k].dtype!=torch.float32 or not torch.isfinite(item[k]).all() for k in ('exp_avg','exp_avg_sq'))
                or (item['exp_avg_sq']<0).any()):raise ValueError('Optimizer step/moment budget differs')


def check_budget(report,saved,original,config,fit_rows,fit,smoke):
    proof=report['proof'];epochs=config['smoke_epochs'] if smoke else config['epochs']
    if saved['epoch']!=epochs or json.loads(json.dumps(saved['history']))!=proof['history'] or saved['candidates']!=proof['candidates'] or saved['best']!=proof['best']:
        raise ValueError('Checkpoint/proof histories differ')
    initial={k:fingerprint(original,k) for k in ('bn','backbone','early','late')}
    if proof['initial']!=initial or saved['initial']!=initial:raise ValueError('Original initialization fingerprints differ')
    if [h['epoch'] for h in proof['history']]!=list(range(1,epochs+1)):raise ValueError('Epoch budget differs')
    ids={c['case_id']:c['label'] for c in fit};draws=min(len(fit_rows),config['batch_size']*config['smoke_batches_per_epoch']) if smoke else len(fit_rows)
    for row in proof['history']:
        epoch=row['epoch'];observed=Counter()
        for cid,count in row['case_draws'].items():
            if cid not in ids or type(count) is not int or count<0:raise ValueError('Forbidden fitting case/count')
            observed[ids[cid]]+=count
        if (row['draws']!=draws or sum(row['case_draws'].values())!=draws or row['class_draws']!=[observed[i] for i in range(8)]
                or not set(row['view_draws']).issubset({str(m) for m in MAGNIFICATIONS})
                or any(type(v) is not int or v<0 for v in row['view_draws'].values()) or sum(row['view_draws'].values())!=draws
                or row['backbone_adapting']!=(epoch>config['head_only_epochs']) or row['learning_rates']!=list(learning_rates(config,epoch))
                or not math.isfinite(row['loss']) or row['loss']<0 or row['bn_sha256']!=initial['bn']):raise ValueError('Fitting budget/phase/LR trace differs')
        if epoch<=config['head_only_epochs'] and (row['early_sha256']!=initial['early'] or row['late_sha256']!=initial['late']):
            raise ValueError('Warmup backbone changed')
    check_state(saved['model'],original,config);check_optimizer(saved,config,len(fit_rows),smoke)
    for kind in ('bn','early','late'):
        if proof['final_%s_sha256'%kind]!=fingerprint(saved['model'],kind):raise ValueError('Final state proof differs')
    if fingerprint(saved['model'],'early')==initial['early'] or fingerprint(saved['model'],'late')==initial['late']:raise ValueError('Backbone did not adapt')
    error=proof['reload_logit_max_difference']
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1e-6:raise ValueError('Invalid selected reload error')


def load_fields(path,rows):
    with np.load(path,allow_pickle=False) as data:values={k:data[k] for k in ('paths','logits','probabilities','log_probabilities')}
    if values['paths'].tolist()!=[r['relative_path'] for r in rows] or len(set(values['paths'].tolist()))!=len(rows):raise ValueError('Scored field paths differ')
    for k in ('logits','probabilities','log_probabilities'):
        if values[k].shape!=(len(rows),8) or values[k].dtype!=np.float64 or not np.isfinite(values[k]).all():raise ValueError('Invalid field array')
    lp=values['logits']-logsumexp(values['logits'],axis=1,keepdims=True)
    if not np.allclose(lp,values['log_probabilities'],rtol=0,atol=1e-12) or not np.allclose(np.exp(lp),values['probabilities'],rtol=0,atol=1e-12):
        raise ValueError('Field log/probability arithmetic differs')
    return values


def independent_records(cases,fields,role):
    lookup={p:i for i,p in enumerate(fields['paths'].tolist())};prob=fields['probabilities'];result=[]
    for c in cases:
        by_view={m:prob[[lookup[p] for p in c['indices'][m]]].mean(0) for m in MAGNIFICATIONS}
        log_view={m:logsumexp(fields['log_probabilities'][[lookup[p] for p in c['indices'][m]]],axis=0)-np.log(len(c['indices'][m])) for m in MAGNIFICATIONS}
        for views in SUBSETS:
            p=np.mean([by_view[m] for m in views],axis=0)
            lp=logsumexp([log_view[m] for m in views],axis=0)-np.log(len(views))
            result.append({'case_id':c['case_id'],'role':role,'truth':c['label'],'magnifications':views,'probabilities':p.tolist(),'log_probabilities':lp.tolist()})
    return result


def verify_metrics(records,claimed):
    scores={}
    for views in SUBSETS:
        name='+'.join(map(str,views));score=metrics_from_scores([r for r in records if tuple(r['magnifications'])==views]);scores[name]=score
        for k,v in score.items():equal_metric(v,claimed['subsets'][name][k])
    detailed={}
    for views in SUBSETS:
        key='+'.join(map(str,views));rows=[r for r in records if tuple(r['magnifications'])==views]
        y=np.array([r['truth'] for r in rows]);p=np.array([r['probabilities'] for r in rows]);lp=np.array([r['log_probabilities'] for r in rows]);counts=Counter(y)
        value=metrics(p,y,log_probabilities=lp);value['class_balanced_cross_entropy']=metrics(p,y,[1/(8*counts[k]) for k in y],lp)['cross_entropy'];detailed[key]=value
    pooled=metrics([r['probabilities'] for r in records],np.array([r['truth'] for r in records]),log_probabilities=[r['log_probabilities'] for r in records]);full=detailed['40+100+200+400']
    expected={'mean_subset_macro_f1':sum(v['macro_f1'] for v in scores.values())/15,
        'mean_subset_accuracy':sum(v['accuracy'] for v in scores.values())/15,
        'mean_subset_balanced_accuracy':sum(v['balanced_accuracy'] for v in scores.values())/15,
        'mean_subset_class_balanced_cross_entropy':sum(v['class_balanced_cross_entropy'] for v in detailed.values())/15,
        'full_view_accuracy':full['accuracy'],'full_view_balanced_accuracy':full['balanced_accuracy'],
        'worst_subset_macro_f1':min(v['macro_f1'] for v in scores.values()),'class_recall':pooled['class_recall'],
        'confusion_matrix':pooled['confusion_matrix'],'subsets':detailed}
    equal_metric(expected,claimed)


def verify_one(folder,manifest,protocol,reference_root,source,weights,trial,seed,smoke=False,original=None):
    folder=Path(folder);source=Path(source);config=json.loads((source/'configs/breakhis-supervised-v1.json').read_text());code=(source/'.source-sha').read_text().strip()
    if config!=json.loads((Path(__file__).resolve().parents[1]/'configs/breakhis-supervised-v1.json').read_text()) or file_hash(weights)!=config['weights_sha256']:
        raise ValueError('Registered config/external initialization differs')
    if original is None:original=torch.load(weights,map_location='cpu',weights_only=True)
    trusted,fit,val,ref,context=prepare_context(manifest,protocol,trial,seed,reference_root,config,code)
    context.update(implementation_sha256=file_hash(source/'magset/supervised_breakhis.py'),weights_sha256=config['weights_sha256'],smoke=smoke)
    if json.loads((folder/'started.json').read_text())!=context:raise ValueError('Started source/role context differs')
    report=json.loads((folder/('smoke.json' if smoke else 'metrics.json')).read_text());proof=report['proof']
    if file_hash(folder/'last.pt')!=proof['last_sha256']:raise ValueError('Last checkpoint checksum differs')
    saved=bound_checkpoint(folder/'last.pt',context);rows=read_manifest(manifest)
    fit_ids={c['case_id'] for c in fit};val_ids={c['case_id'] for c in val};fit_rows=[r for r in rows if case_id(r) in fit_ids];val_rows=[r for r in rows if case_id(r) in val_ids]
    check_budget(report,saved,original,config,fit_rows,fit,smoke)
    if report['outer_predictions_computed']!=0:raise ValueError('Outer predictions forbidden')
    if smoke:
        if (report['context']!=context or report['status']!='fitting-only technical smoke' or report['validation_predictions_computed']!=0
                or proof['candidates'] or proof['best'] is not None or saved['best_model'] is not None
                or any((folder/p).exists() for p in ('selected.pt','metrics.json','predictions.jsonl','train-fields.npz','validation-fields.npz'))):
            raise ValueError('Technical smoke contains selection/scoring')
        return report,{'smoke_sha256':file_hash(folder/'smoke.json'),'last_sha256':proof['last_sha256']}
    verified,hashes=verify_run(folder/'metrics.json',trusted)
    if (verified['encoder']!='resnet50-supervised-v1' or verified['method']!='mean_probability' or verified['trial']!=trial or verified['seed']!=seed
            or verified['code_sha']!=code or verified['config']!=config or verified['config_sha256']!=file_hash(source/'configs/breakhis-supervised-v1.json')
            or verified['training_seed']!=config['training_seed'] or [c['epoch'] for c in proof['candidates']]!=config['selection_epochs']):raise ValueError('Pipeline settings differ')
    for candidate in proof['candidates']:
        path=folder/('selection-epoch%d.npz'%candidate['epoch'])
        if file_hash(path)!=candidate['fields_sha256']:raise ValueError('Selection field checksum differs')
        fields=load_fields(path,val_rows);verify_metrics(independent_records(val,fields,'validation'),candidate['validation'])
    best=max(proof['candidates'],key=lambda c:c['validation']['mean_subset_macro_f1'])
    if proof['best']!=best or verified['selected_epoch']!=best['epoch']:raise ValueError('Earliest-best selection differs')
    if file_hash(folder/'selected.pt')!=verified['selected_model_sha256']:raise ValueError('Selected checkpoint checksum differs')
    selected=bound_checkpoint(folder/'selected.pt',context);check_state(selected['model'],original,config);check_state(saved['best_model'],original,config)
    if selected['epoch']!=best['epoch'] or digest_state(selected['model'])!=best['state_sha256'] or digest_state(saved['best_model'])!=best['state_sha256']:
        raise ValueError('Selected model/last best snapshot differs')
    for role,cases,role_rows,claim in (('train',fit,fit_rows,verified['fitting']),('validation',val,val_rows,verified['validation'])):
        path=folder/('%s-fields.npz'%role)
        if file_hash(path)!=verified['field_artifact_sha256'][role]:raise ValueError('Final field hash differs')
        fields=load_fields(path,role_rows);records=independent_records(cases,fields,role);verify_metrics(records,claim)
        predictions=folder/('fitting-predictions.jsonl' if role=='train' else 'predictions.jsonl')
        archive=[json.loads(line) for line in predictions.read_text().splitlines()]
        lookup={(r['case_id'],tuple(r['magnifications'])):r for r in archive}
        if len(archive)!=len(records) or len(lookup)!=len(records):raise ValueError('Duplicate/missing saved case scores')
        for r in records:
            other=lookup.get((r['case_id'],tuple(r['magnifications'])))
            if other is None or other['role']!=role or other['truth']!=r['truth']:
                raise ValueError('Saved case probability replay differs')
            metrics_from_scores([other]);equal_metric(r['probabilities'],other['probabilities']);equal_metric(r['log_probabilities'],other['log_probabilities'])
        if role=='validation':
            with np.load(folder/('selection-epoch%d.npz'%best['epoch']),allow_pickle=False) as expected:
                if np.max(np.abs(fields['probabilities']-expected['probabilities']))>1e-6:raise ValueError('Selected field re-encoding differs')
        label={c['case_id']:c['label'] for c in cases};y=np.array([label[case_id(r)] for r in role_rows])
        for name,w in (('raw',None),('balanced',sampling_weights(role_rows,cases).numpy())):
            actual=metrics(fields['probabilities'],y,w,fields['log_probabilities']);claimed=verified['field_metrics'][role][name]
            equal_metric(actual,claimed)
    return verified,{**hashes,'last_sha256':proof['last_sha256'],'selected_model_sha256':verified['selected_model_sha256'],
                     'field_artifact_sha256':verified['field_artifact_sha256'],'frozen_reference':context['frozen_reference']}


def collect(root,manifest,protocol,reference_root,source,weights):
    root=Path(root);config=json.loads((Path(source)/'configs/breakhis-supervised-v1.json').read_text())
    original=torch.load(weights,map_location='cpu',weights_only=True)
    _,smoke_hashes=verify_one(root/'technical-smoke-retry1',manifest,protocol,reference_root,source,weights,1,812,True,original)
    expected={root/('split%d'%s)/('trial%d'%t)/'metrics.json' for s in (812,813) for t in range(1,6)}
    if set(root.glob('split*/trial*/metrics.json'))!=expected:raise ValueError('Complete exact ten-run matrix required')
    pairs=[];proof=[]
    for seed in (812,813):
        for trial in range(1,6):
            folder=root/('split%d'%seed)/('trial%d'%trial);r,h=verify_one(folder,manifest,protocol,reference_root,source,weights,trial,seed,False,original)
            baseline=Path(reference_root)/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed));frozen,_=verify_run(baseline/'metrics.json')
            pair={'seed':seed,'trial':trial,'selected_epoch':r['selected_epoch'],'fitting':r['fitting'],'field_metrics':r['field_metrics']}
            for name,directory,report in (('frozen',baseline,frozen),('supervised',folder,r)):
                records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()];pair[name]=summarize(report,records)
            pair['delta_mean_subset_macro_f1']=pair['supervised']['mean_subset_macro_f1']-pair['frozen']['mean_subset_macro_f1']
            pairs.append(pair);proof.append({'seed':seed,'trial':trial,**h})
    summary=[]
    for seed in (812,813):
        group=[p for p in pairs if p['seed']==seed];row={'seed':seed}
        for name in ('frozen','supervised'):
            row[name]={k:sum(p[name][k] for p in group)/5 for k in ('mean_subset_macro_f1','mean_subset_accuracy','full_view_macro_f1','full_view_accuracy','worst_subset_macro_f1','brier')}
            row[name]['class_recall']=np.mean([p[name]['class_recall'] for p in group],axis=0).tolist()
        row['delta_mean_subset_macro_f1']=row['supervised']['mean_subset_macro_f1']-row['frozen']['mean_subset_macro_f1'];summary.append(row)
    promising=all(r['delta_mean_subset_macro_f1']>=.02 and r['supervised']['class_recall'][6]>0 for r in summary)
    return {'status':'complete verified supervised reference','dataset':'BreakHis','verified_runs':10,'class_names':CLASS_NAMES,
            'training_source_sha':(Path(source)/'.source-sha').read_text().strip(),'config_sha256':file_hash(Path(source)/'configs/breakhis-supervised-v1.json'),
            'manifest_sha256':file_hash(manifest),'protocol_sha256':file_hash(protocol),'smoke':smoke_hashes,
            'summary':summary,'pairs':pairs,'artifacts':proof,'promising_descriptive_gate':promising,'collector_sha256':file_hash(__file__),
            'interpretation':'Exploratory paired provider-case pipeline comparison; no clinical confirmation, independent-fold significance, exact author reproduction or novelty claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','manifest','protocol','reference-root','training-source','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.smoke_only:
        r,h=verify_one(a.runs/'technical-smoke-retry1',a.manifest,a.protocol,a.reference_root,a.training_source,a.weights,1,812,True)
        value={'status':'supervised technical smoke verified','proof':r['proof'],**h}
    else:value=collect(a.runs,a.manifest,a.protocol,a.reference_root,a.training_source,a.weights)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:value[k] for k in ('status','verified_runs','summary') if k in value},indent=2))


if __name__=='__main__':main()

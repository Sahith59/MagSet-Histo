"""Verify the fixed BreakHis adaptation budget, artifacts and complete paired matrix."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from .adapt_breakhis import prepare_context, bound_checkpoint, checked_difference
from .bags import fit_normalization
from .breakhis_predict import load_head, predict_fields
from .collect import SUBSETS, verify_run, load_trusted_context
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest


def digest_state(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update((name+str(x.dtype)+str(tuple(x.shape))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def fragment_schema(config):
    """Audited pinned Dinov2 block layout, independent of the reported key list."""
    d=config['dimension'];p='encoder.layer.%d.'%(config['encoder_layers']-1);result={}
    def linear(name,out,inside):
        result[p+name+'.weight']=(out,inside);result[p+name+'.bias']=(out,)
    def norm(name):
        result[p+name+'.weight']=(d,);result[p+name+'.bias']=(d,)
    norm('norm1')
    for name in ('query','key','value'):linear('attention.attention.'+name,d,d)
    linear('attention.output.dense',d,d);result[p+'layer_scale1.lambda1']=(d,)
    norm('norm2');linear('mlp.fc1',4*d,d);linear('mlp.fc2',d,4*d);result[p+'layer_scale2.lambda1']=(d,)
    result['layernorm.weight']=(d,);result['layernorm.bias']=(d,)
    return result


def verify_optimizer(checkpoint,config,steps_per_epoch,epochs):
    schema=fragment_schema(config);head_schema={'1.weight':(8,config['dimension']),'1.bias':(8,)}
    state=checkpoint['encoder_fragment'];head=checkpoint['auxiliary_head']
    if (list(state)!=list(schema) or any(tuple(state[k].shape)!=shape for k,shape in schema.items())
            or list(head)!=list(head_schema) or any(tuple(head[k].shape)!=shape or head[k].dtype!=torch.float32
                 or not torch.isfinite(head[k]).all() for k,shape in head_schema.items())):
        raise ValueError('Canonical encoder/auxiliary-head schema differs')
    opt=checkpoint['optimizer'];groups=opt['param_groups'];n=len(schema)
    expected={'weight_decay':config['weight_decay'],'betas':(.9,.999),'eps':1e-8,'amsgrad':False,
              'maximize':False,'foreach':None,'capturable':False,'differentiable':False,'fused':None}
    if len(groups)!=2:raise ValueError('AdamW group count differs')
    for group,lr,ids in zip(groups,(config['encoder_learning_rate'],config['head_learning_rate']),(list(range(n)),[n,n+1])):
        if group!={**expected,'lr':lr,'params':ids}:raise ValueError('AdamW parameter groups/settings differ')
    if set(opt['state'])!=set(range(n+2)):raise ValueError('AdamW state coverage differs')
    values=list(state.values())+list(head.values())
    for i,value in enumerate(values):
        saved=opt['state'][i];updates=steps_per_epoch*(epochs-config['head_only_epochs'] if i<n else epochs)
        if (set(saved)!={'step','exp_avg','exp_avg_sq'} or saved['step'].shape!=torch.Size([])
                or saved['step'].dtype!=torch.float32 or not torch.isfinite(saved['step']) or saved['step'].item()!=updates
                or any(saved[k].shape!=value.shape or saved[k].dtype!=torch.float32 or not torch.isfinite(saved[k]).all()
                       for k in ('exp_avg','exp_avg_sq')) or (saved['exp_avg_sq']<0).any()):
            raise ValueError('AdamW update budget/state differs')


def verify_budget(proof,checkpoint,config,fit_rows,fit_cases,smoke=False):
    epochs=config['smoke_epochs'] if smoke else config['epochs']
    n=len(fit_rows);draws=min(n,config['batch_size']*config['smoke_batches_per_epoch']) if smoke else n
    history=proof['history'];fit_ids={c['case_id'] for c in fit_cases}
    # JSON normalizes integer magnification keys; Torch preserves them.
    if (checkpoint['epoch']!=epochs or len(history)!=epochs
            or json.loads(json.dumps(checkpoint['history']))!=history
            or checkpoint['frozen_prefix_sha256']!=proof['frozen_prefix_sha256']
            or checkpoint['base_allowed_sha256']!=proof['allowed_before_sha256']):
        raise ValueError('Checkpoint/proof epoch or base binding differs')
    for name in ('frozen_prefix_sha256','allowed_before_sha256','allowed_after_sha256',
                 'base_last_block_sha256','base_final_norm_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',proof[name]):raise ValueError('Invalid state fingerprint')
    if proof['allowed_before_sha256']==proof['allowed_after_sha256']:raise ValueError('Encoder did not adapt')
    labels={c['case_id']:c['label'] for c in fit_cases}
    for i,row in enumerate(history,1):
        if (row['epoch']!=i or row['draws']!=draws or row['encoder_adapting']!=(i>config['head_only_epochs'])
                or type(row['loss']) not in (int,float) or not math.isfinite(row['loss']) or row['loss']<0
                or len(row['class_draws'])!=8 or any(type(x) is not int or x<0 for x in row['class_draws'])
                or sum(row['class_draws'])!=draws
                or not set(row['case_draws']).issubset(fit_ids)
                or any(type(x) is not int or x<0 for x in row['case_draws'].values())
                or sum(row['case_draws'].values())!=draws
                or not set(row['view_draws']).issubset({str(m) for m in MAGNIFICATIONS})
                or any(type(x) is not int or x<0 for x in row['view_draws'].values())
                or sum(row['view_draws'].values())!=draws):
            raise ValueError('Registered fitting budget/history differs')
        observed=Counter()
        for cid,count in row['case_draws'].items():observed[labels[cid]]+=count
        if [observed[k] for k in range(8)]!=row['class_draws']:raise ValueError('Class/case draw counts disagree')
        if i<=config['head_only_epochs'] and (row['last_block_sha256']!=proof['base_last_block_sha256']
                                             or row['final_norm_sha256']!=proof['base_final_norm_sha256']):
            raise ValueError('Encoder changed during warmup')
    state=checkpoint['encoder_fragment'];prefix='encoder.layer.%d.'%(config['encoder_layers']-1)
    if (set(state)!=set(proof['encoder_trainable_names']) or not any(n.startswith(prefix) for n in state)
            or not any(n.startswith('layernorm.') for n in state)
            or any(not n.startswith((prefix,'layernorm.')) for n in state)
            or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values())
            or sum(v.numel() for v in state.values())!=proof['encoder_trainable_parameters']
            or digest_state(state)!=proof['allowed_after_sha256']):
        raise ValueError('Final encoder fragment differs')
    verify_optimizer(checkpoint,config,math.ceil(draws/config['batch_size']),epochs)
    block=digest_state({k:v for k,v in state.items() if k.startswith(prefix)})
    norm=digest_state({k:v for k,v in state.items() if k.startswith('layernorm.')})
    if (block!=history[-1]['last_block_sha256'] or norm!=history[-1]['final_norm_sha256']
            or block==proof['base_last_block_sha256'] or norm==proof['base_final_norm_sha256']):
        raise ValueError('Final last-block/normalization proof differs')
    for key,tol in (('initial_base_feature_max_difference',1e-4),('encoder_reload_max_difference',1e-6)):
        value=proof[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=tol:
            raise ValueError('Invalid encoder probe/reload error')


def check_features(fields,rows,dimension):
    paths=fields['paths'].tolist();z=fields['features']
    if (paths!=[r['relative_path'] for r in rows] or len(paths)!=len(set(paths))
            or z.shape!=(len(rows),dimension) or z.dtype!=np.float32 or not np.isfinite(z).all()):
        raise ValueError('Adapted cache coverage/shape/values differ')


def replay_head(folder,cases,fields,records,expected_hash):
    lookup={p:i for i,p in enumerate(fields['paths'].tolist())};z=fields['features']
    bags=[[z[[lookup[p] for p in c['indices'][m]]] for m in MAGNIFICATIONS] for c in cases]
    mean,std=fit_normalization(bags,range(46));head=load_head(folder/'linear-model.npz',expected_hash,z.shape[1])
    if checked_difference(mean,head['mean'])>1e-12 or checked_difference(std,head['std'])>1e-12:
        raise ValueError('Head normalization differs from fitting-only field moments')
    refs={(r['case_id'],tuple(r['magnifications'])):r['probabilities'] for r in records};error=0.
    for j,c in enumerate(cases[46:],46):
        for views in SUBSETS:
            prob=predict_fields(dict(zip(MAGNIFICATIONS,bags[j])),views,head)
            error=max(error,checked_difference(prob,np.asarray(refs[c['case_id'],views])))
    if error>1e-8:raise ValueError('Saved adapted head/cache probabilities differ')
    return error


def verify_one(folder,manifest,protocol,reference_root,source,trial,seed,smoke=False):
    folder=Path(folder);source=Path(source);code=(source/'.source-sha').read_text().strip()
    if not re.fullmatch('[0-9a-f]{40}',code):raise ValueError('Invalid immutable source revision')
    config_path=source/'configs/breakhis-adaptation-v1.json';config=json.loads(config_path.read_text())
    if config!=json.loads((Path(__file__).resolve().parents[1]/'configs/breakhis-adaptation-v1.json').read_text()):
        raise ValueError('Collector and registered training configuration differ')
    trusted,fit,val,reference,context=prepare_context(manifest,protocol,trial,seed,reference_root,config,code)
    context['implementation_sha256']=file_hash(source/'magset/adapt_breakhis.py')
    context.update(base_encoder=json.loads((source/'configs/encoders.json').read_text())['phikon-v2'],smoke=smoke)
    if json.loads((folder/'started.json').read_text())!=context:raise ValueError('Adaptation start/source/reference binding differs')
    report=json.loads((folder/('smoke.json' if smoke else 'metrics.json')).read_text())
    rows=read_manifest(manifest);fit_ids={c['case_id'] for c in fit};allowed=fit_ids|{c['case_id'] for c in val}
    fit_rows=[r for r in rows if case_id(r) in fit_ids];encode_rows=[r for r in rows if case_id(r) in allowed]
    proof=report['proof'] if smoke else report['adaptation_proof']
    if file_hash(folder/'last-checkpoint.pt')!=proof['checkpoint_sha256']:raise ValueError('Checkpoint checksum differs')
    checkpoint=bound_checkpoint(folder/'last-checkpoint.pt',context)
    verify_budget(proof,checkpoint,config,fit_rows,fit,smoke)
    if report['outer_predictions_computed']!=0:raise ValueError('Outer predictions are forbidden')
    if smoke:
        if report['context']!=context or report['status']!='fitting-only technical smoke' or report['validation_predictions_computed']!=0:
            raise ValueError('Technical smoke contains scoring or differs from context')
        if any((folder/n).exists() for n in ('metrics.json','predictions.jsonl','adapted-features.npz')):
            raise ValueError('Technical smoke contains production artifacts')
        return report,{'smoke_sha256':file_hash(folder/'smoke.json'),'checkpoint_sha256':proof['checkpoint_sha256']}
    verified,hashes=verify_run(folder/'metrics.json',trusted=trusted)
    if (verified['encoder']!='phikon-v2-lastblock-v1' or verified['method']!='mean' or verified['trial']!=trial or verified['seed']!=seed
            or verified['code_sha']!=code or verified['config']!=config or verified['config_sha256']!=file_hash(config_path)
            or verified['training_seed']!=812 or verified['normalization_policy']!='field-moments-v2'
            or verified['base_encoder']!=context['base_encoder'] or verified['encoded_cases']!=54
            or verified['encoded_fields']!=len(encode_rows) or verified['fitting_fields']!=len(fit_rows)
            or verified['fitting_class_cases']!=[sum(c['label']==i for c in fit) for i in range(8)]
            or [c['C'] for c in verified['candidates']]!=config['linear_C_grid']):
        raise ValueError('Adaptation settings/coverage differ from registration')
    selected=max(verified['candidates'],key=lambda c:c['mean_subset_macro_f1'])
    if (selected['C']!=verified['selected_C'] or {k:v for k,v in selected.items() if k!='C'}!=verified['validation']):
        raise ValueError('Case-head selection differs')
    head_error=verified['head_reload_max_difference']
    if type(head_error) not in (int,float) or not math.isfinite(head_error) or not 0<=head_error<=1e-8:
        raise ValueError('Invalid head reload error')
    path=folder/'adapted-features.npz'
    if file_hash(path)!=verified['feature_sha256']:raise ValueError('Adapted feature checksum differs')
    records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    with np.load(path,allow_pickle=False) as fields:
        check_features(fields,encode_rows,config['dimension'])
        error=replay_head(folder,fit+val,fields,records,verified['head_sha256'])
    return verified,{**hashes,'checkpoint_sha256':proof['checkpoint_sha256'],'feature_sha256':verified['feature_sha256'],
                     'head_sha256':verified['head_sha256'],'independent_head_replay_error':error,
                     'frozen_reference':context['frozen_reference']}


def summarize(report,records):
    scores=report['validation'];full=scores['subsets']['40+100+200+400']
    correct=Counter();total=Counter();class_cases={i:set() for i in range(8)};matrix=np.zeros((8,8),dtype=int);brier=0.
    for r in records:
        y=r['truth'];p=np.asarray(r['probabilities']);pred=int(p.argmax());correct[y]+=int(pred==y);total[y]+=1
        class_cases[y].add(r['case_id']);matrix[y,pred]+=1;brier+=float(np.square(p-np.eye(8)[y]).sum())
    return {'mean_subset_macro_f1':scores['mean_subset_macro_f1'],
            'mean_subset_accuracy':sum(v['accuracy'] for v in scores['subsets'].values())/15,
            'mean_subset_balanced_accuracy':sum(v['balanced_accuracy'] for v in scores['subsets'].values())/15,
            'full_view_macro_f1':full['macro_f1'],'full_view_accuracy':full['accuracy'],
            'worst_subset_macro_f1':scores['worst_subset_macro_f1'],'brier':brier/len(records),
            'class_recall':[correct[i]/total[i] for i in range(8)],
            'class_distinct_cases':[len(class_cases[i]) for i in range(8)],'confusion_matrix':matrix.tolist(),
            'subsets':scores['subsets']}


def collect(root,manifest,protocol,reference_root,source):
    root=Path(root);pairs=[];artifacts=[];unique={s:set() for s in (812,813)}
    class_unique={s:{i:set() for i in range(8)} for s in (812,813)}
    _,smoke_hashes=verify_one(root/'technical-smoke',manifest,protocol,reference_root,source,1,812,True)
    missing=[(s,t) for s in (812,813) for t in range(1,6) if not(root/('split%d'%s)/('trial%d'%t)/'metrics.json').exists()]
    if missing:raise ValueError('Complete ten-run matrix required; missing '+str(missing))
    expected={root/('split%d'%s)/('trial%d'%t)/'metrics.json' for s in (812,813) for t in range(1,6)}
    if set(root.glob('split*/trial*/metrics.json'))!=expected:raise ValueError('Unexpected adaptation configuration')
    for seed in (812,813):
        for trial in range(1,6):
            folder=root/('split%d'%seed)/('trial%d'%trial)
            adapted,hashes=verify_one(folder,manifest,protocol,reference_root,source,trial,seed)
            unique[seed].update(adapted['case_ids']['validation'])
            ref=Path(reference_root)/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed))
            baseline,_=verify_run(ref/'metrics.json',trusted=load_trusted_context(manifest,protocol))
            pair={'seed':seed,'trial':trial}
            for name,directory,report in (('frozen',ref,baseline),('adapted',folder,adapted)):
                records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
                pair[name]=summarize(report,records)
                if name=='adapted':
                    for r in records:class_unique[seed][r['truth']].add(r['case_id'])
            pair['delta_mean_subset_macro_f1']=pair['adapted']['mean_subset_macro_f1']-pair['frozen']['mean_subset_macro_f1']
            pairs.append(pair);artifacts.append({'seed':seed,'trial':trial,**hashes})
    numeric=('mean_subset_macro_f1','mean_subset_accuracy','mean_subset_balanced_accuracy','full_view_macro_f1','full_view_accuracy','worst_subset_macro_f1','brier')
    summary=[]
    for seed in (812,813):
        group=[p for p in pairs if p['seed']==seed];row={'seed':seed,'distinct_validation_cases':len(unique[seed]),
            'class_distinct_validation_cases':[len(class_unique[seed][i]) for i in range(8)],'validation_records':600}
        for name in ('frozen','adapted'):
            row[name]={k:sum(p[name][k] for p in group)/5 for k in numeric}
            row[name]['class_recall']=[sum(p[name]['class_recall'][i] for p in group)/5 for i in range(8)]
        row['delta_mean_subset_macro_f1']=row['adapted']['mean_subset_macro_f1']-row['frozen']['mean_subset_macro_f1'];summary.append(row)
    return {'status':'complete verified exploratory adaptation comparison','dataset':'BreakHis','verified_runs':10,
            'class_names':CLASS_NAMES,'source_sha':(Path(source)/'.source-sha').read_text().strip(),
            'manifest_sha256':file_hash(manifest),'protocol_sha256':file_hash(protocol),
            'config_sha256':file_hash(Path(source)/'configs/breakhis-adaptation-v1.json'),
            'distinct_validation_cases_union':len(unique[812]|unique[813]),'technical_smoke':smoke_hashes,
            'collector_sha256':file_hash(__file__),'summary':summary,'pairs':pairs,'artifacts':artifacts,
            'interpretation':'Descriptive paired provider-case validation, overlapping development holdouts; no pristine confirmation, clinical performance, significance or novelty claim.'}


def main():
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','manifest','protocol','reference-root','training-source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    if a.smoke_only:
        report,hashes=verify_one(a.runs/'technical-smoke',a.manifest,a.protocol,a.reference_root,a.training_source,1,812,True)
        value={'status':'technical smoke independently verified','proof':report['proof'],**hashes}
    else:value=collect(a.runs,a.manifest,a.protocol,a.reference_root,a.training_source)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:value[k] for k in ('status','verified_runs','summary') if k in value},indent=2))


if __name__=='__main__':main()

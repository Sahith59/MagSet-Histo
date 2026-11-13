"""Fixed fitting-only last-block adaptation followed by the existing mean case head."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.exceptions import ConvergenceWarning

from .baseline import SUBSETS, stratified_inner_split, fit_models, evaluate, predict_models
from .bags import fit_normalization
from .breakhis_predict import load_checked_image, load_head, predict_fields
from .collect import load_trusted_context, verify_run
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .features import load_encoder, identity_collate, worker_init


def trainable_names(model):
    depth=len(model.encoder.layer)
    prefixes=('encoder.layer.%d.'%(depth-1),'layernorm.')
    names={name for name,_ in model.named_parameters() if name.startswith(prefixes)}
    if not names or not any(n.startswith(prefixes[0]) for n in names) or not any(n.startswith('layernorm.') for n in names):
        raise ValueError('Expected last encoder block and final LayerNorm')
    return names


def set_phase(model, adapting):
    names=trainable_names(model);model.eval()
    for name,param in model.named_parameters():param.requires_grad_(adapting and name in names)
    model.encoder.layer[-1].train(adapting);model.layernorm.train(adapting)
    return names


def fingerprint(model, names, include):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if (name in names)==include:
            x=value.detach().cpu().contiguous()
            h.update((name+str(x.dtype)+str(tuple(x.shape))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def fragment(model):
    names=trainable_names(model)
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in names}


def restore_fragment(model, state):
    expected=trainable_names(model)
    parameters=dict(model.named_parameters())
    if set(state)!=expected or any(not torch.is_tensor(state[n]) or tuple(state[n].shape)!=tuple(parameters[n].shape)
                                  or state[n].dtype!=parameters[n].dtype for n in state):
        raise ValueError('Checkpoint contains unexpected encoder parameters/shapes/dtypes')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite encoder fragment')
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected or set(missing)!=set(model.state_dict())-expected:
        raise ValueError('Encoder fragment reload identity differs')


def bound_checkpoint(path, context):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['context']!=context:
        raise ValueError('Checkpoint role/config/base/source binding differs')
    return checkpoint


def sampling_weights(rows, fit_cases):
    ids={c['case_id'] for c in fit_cases};classes={c['case_id']:c['label'] for c in fit_cases}
    if len(ids)!=len(fit_cases) or not rows or any(case_id(r) not in ids for r in rows):
        raise ValueError('Sampler accepts unique fitting cases only')
    counts=Counter(classes.values());fields=Counter((case_id(r),int(r['magnification'])) for r in rows)
    if set(counts)!=set(range(8)) or set(fields)!={(c,m) for c in ids for m in MAGNIFICATIONS}:
        raise ValueError('Fitting class/view coverage differs')
    if len({r['relative_path'] for r in rows})!=len(rows):raise ValueError('Duplicate fitting field')
    return torch.tensor([1/(8*counts[classes[case_id(r)]]*4*fields[case_id(r),int(r['magnification'])]) for r in rows],dtype=torch.float64)


def flipped(image, path, epoch, seed):
    from PIL import ImageOps
    bits=hashlib.sha256(('breakhis-adapt-v1|%d|%d|%s'%(seed,epoch,path)).encode()).digest()[0]
    if bits&1:image=ImageOps.mirror(image)
    if bits&2:image=ImageOps.flip(image)
    return image


class CheckedFields(Dataset):
    def __init__(self,rows,images_root,labels,epoch=0,seed=812):
        self.rows=rows;self.root=images_root;self.labels=labels;self.epoch=epoch;self.seed=seed
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];image=load_checked_image(self.root,row)
        if self.epoch:image=flipped(image,row['relative_path'],self.epoch,self.seed)
        return row['relative_path'],self.labels[case_id(row)],image


def loader(rows, root, labels, batch, workers, epoch=0, weights=None, seed=812):
    sampler=None
    if weights is not None:
        sampler=WeightedRandomSampler(weights,len(rows),replacement=True,
                   generator=torch.Generator().manual_seed(seed+epoch*1000003))
    return DataLoader(CheckedFields(rows,root,labels,epoch,seed),batch_size=batch,sampler=sampler,
        shuffle=False,num_workers=workers,drop_last=False,collate_fn=identity_collate,worker_init_fn=worker_init,
        multiprocessing_context='spawn' if workers else None,persistent_workers=False)


def prepare_context(manifest,protocol,trial,seed,reference_root,config,code_sha):
    trusted=load_trusted_context(manifest,protocol);cases=trusted['cases']
    train,val,test=stratified_inner_split(cases,trusted['trials'][str(trial)]['case_roles'],seed)
    fit=[cases[i] for i in train];validation=[cases[i] for i in val]
    path=Path(reference_root)/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed))/'metrics.json'
    reference,hashes=verify_run(path,trusted=trusted)
    roles={'train':[c['case_id'] for c in fit],'validation':[c['case_id'] for c in validation],
           'test':[cases[i]['case_id'] for i in test]}
    if (reference['encoder']!='phikon-v2' or reference['method']!='mean' or reference['normalization_policy']!='field-moments-v2'
            or reference['trial']!=trial or reference['seed']!=seed
            or any(set(roles[k])!=set(reference['case_ids'][k]) for k in roles)):
        raise ValueError('Frozen comparison roles/settings differ')
    context={'manifest_sha256':trusted['manifest_sha256'],'protocol_sha256':trusted['protocol_sha256'],
             'case_ids':roles,'trial':trial,'seed':seed,'config':config,'code_sha':code_sha,
             'implementation_sha256':file_hash(__file__),'frozen_reference':hashes}
    return trusted,fit,validation,reference,context


def frozen_probe(rows, cache, reference):
    probes=[];expected=[]
    for i,m in enumerate(MAGNIFICATIONS):
        row=next(r for r in rows if int(r['magnification'])==m);probes.append(row)
        path=Path(cache)/('phikon-v2-%d.npz'%m)
        if file_hash(path)!=reference['feature_provenance'][i]['feature_sha256']:
            raise ValueError('Base feature reference checksum differs')
        with np.load(path,allow_pickle=False) as values:
            keys=values['paths'].tolist();expected.append(values['features'][keys.index(row['relative_path'])])
    return probes,np.asarray(expected,dtype=np.float32)


def step(model,head,optimizer,pixels,labels,clip):
    optimizer.zero_grad(set_to_none=True)
    z=model(pixel_values=pixels).last_hidden_state[:,0,:]
    loss=nn.functional.cross_entropy(head(z),labels)
    if not torch.isfinite(loss):raise ValueError('Nonfinite fitting loss')
    loss.backward();nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],clip,error_if_nonfinite=True)
    optimizer.step();return float(loss.detach())


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def checked_difference(left,right):
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid feature parity arrays')
    return float(np.max(np.abs(left-right)))


def train(model,transform,fit_rows,fit_cases,images_root,config,context,output,device,workers,smoke,resume):
    names=trainable_names(model);frozen_before=fingerprint(model,names,False);allowed_before=fingerprint(model,names,True)
    block_names={n for n in names if n.startswith('encoder.')};norm_names=names-block_names
    block_before=fingerprint(model,block_names,True);norm_before=fingerprint(model,norm_names,True)
    torch.manual_seed(config['training_seed'])
    head=nn.Sequential(nn.Dropout(config['head_dropout']),nn.Linear(config['dimension'],8)).to(device)
    params=[p for n,p in model.named_parameters() if n in names]
    optimizer=torch.optim.AdamW([{'params':params,'lr':config['encoder_learning_rate']},
        {'params':list(head.parameters()),'lr':config['head_learning_rate']}],weight_decay=config['weight_decay'])
    weights=sampling_weights(fit_rows,fit_cases);labels={c['case_id']:c['label'] for c in fit_cases}
    row_lookup={r['relative_path']:r for r in fit_rows}
    epochs=config['smoke_epochs'] if smoke else config['epochs'];history=[];start_epoch=1
    checkpoint_path=Path(output)/'last-checkpoint.pt'
    if resume and checkpoint_path.exists():
        saved=bound_checkpoint(checkpoint_path,context)
        if saved['frozen_prefix_sha256']!=frozen_before or saved['base_allowed_sha256']!=allowed_before:
            raise ValueError('Checkpoint base-state fingerprint differs')
        restore_fragment(model,saved['encoder_fragment'])
        head.load_state_dict(saved['auxiliary_head']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];start_epoch=saved['epoch']+1
        if start_epoch>epochs:print('Reusing fixed final checkpoint; no extra fitting',flush=True)
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    for epoch in range(start_epoch,epochs+1):
        adapting=epoch>config['head_only_epochs'];set_phase(model,adapting);head.train()
        total=draws=0;class_draws=[0]*8;case_draws=Counter();view_draws=Counter();epoch_start=time.monotonic()
        stream=loader(fit_rows,images_root,labels,config['batch_size'],workers,epoch,weights,config['training_seed'])
        for batch_index,batch in enumerate(stream):
            if smoke and batch_index>=config['smoke_batches_per_epoch']:break
            keys,ys,images=zip(*batch);target=torch.tensor(ys,device=device)
            loss=step(model,head,optimizer,transform(list(images)).to(device),target,config['gradient_clip'])
            total+=loss*len(batch);draws+=len(batch)
            for label in ys:class_draws[label]+=1
            for key in keys:
                row=row_lookup[key];case_draws[case_id(row)]+=1;view_draws[int(row['magnification'])]+=1
            if (batch_index+1)%64==0:
                print(json.dumps({'epoch':epoch,'batches':batch_index+1,'draws':draws,'fitting_fields':len(fit_rows),
                                  'elapsed_seconds':time.monotonic()-epoch_start}),flush=True)
        if not draws or (not smoke and draws!=len(fit_rows)):raise ValueError('Fitting draw budget differs')
        block_hash=fingerprint(model,block_names,True);norm_hash=fingerprint(model,norm_names,True)
        if not adapting and (block_hash!=block_before or norm_hash!=norm_before):
            raise ValueError('Encoder changed during head-only warmup')
        history.append({'epoch':epoch,'loss':total/draws,'draws':draws,'class_draws':class_draws,'encoder_adapting':adapting,
                        'last_block_sha256':block_hash,'final_norm_sha256':norm_hash,
                        'case_draws':dict(case_draws),'view_draws':dict(view_draws),'elapsed_seconds':time.monotonic()-epoch_start})
        record={'context':context,'encoder_fragment':fragment(model),'auxiliary_head':cpu_tree(head.state_dict()),
            'optimizer':cpu_tree(optimizer.state_dict()),'epoch':epoch,'history':history,
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'frozen_prefix_sha256':frozen_before,'base_allowed_sha256':allowed_before}
        temp=checkpoint_path.with_suffix('.tmp');torch.save(record,temp);os.replace(temp,checkpoint_path)
        print(json.dumps({'trial':context['trial'],'seed':context['seed'],**history[-1]}),flush=True)
    saved=bound_checkpoint(checkpoint_path,context)
    if saved['epoch']!=epochs:raise ValueError('Fixed epoch budget not reached')
    frozen_after=fingerprint(model,names,False);allowed_after=fingerprint(model,names,True)
    if frozen_before!=frozen_after or allowed_before==allowed_after:
        raise ValueError('Frozen prefix changed or permitted encoder failed to adapt')
    if fingerprint(model,block_names,True)==block_before or fingerprint(model,norm_names,True)==norm_before:
        raise ValueError('Last block or final LayerNorm failed to adapt')
    return {'frozen_prefix_sha256':frozen_after,'allowed_before_sha256':allowed_before,
        'base_last_block_sha256':block_before,'base_final_norm_sha256':norm_before,
        'allowed_after_sha256':allowed_after,'encoder_trainable_parameters':sum(p.numel() for p in params),
        'encoder_trainable_names':sorted(names),'history':history,'checkpoint_sha256':file_hash(checkpoint_path)}


def fit_case_head(cases,fields,output):
    lookup={key:i for i,key in enumerate(fields['paths'].tolist())};features=fields['features']
    bags=[[features[[lookup[p] for p in c['indices'][m]]] for m in MAGNIFICATIONS] for c in cases]
    mean,std=fit_normalization(bags,range(46));z=np.asarray([[x.mean(0) for x in c] for c in bags])
    z=(z-mean)/std;y=np.asarray([c['label'] for c in cases]);choices=[];best=None
    for C in (.01,.1,1.,10.):
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            models=fit_models(z[:46],y[:46],'mean',C)
        value=evaluate(models,z[46:],y[46:],'mean');choices.append({'C':C,**value})
        if best is None or value['mean_subset_macro_f1']>best[0]:best=(value['mean_subset_macro_f1'],C,models,value)
    _,C,models,value=best;model=models[0]
    path=Path(output)/'linear-model.npz'
    np.savez_compressed(path,mean=mean,std=std,coef_0=model.coef_,intercept_0=model.intercept_,classes_0=model.classes_)
    head=load_head(path,file_hash(path),z.shape[-1]);records=[];error=0.
    for subset in SUBSETS:
        reference=predict_models(models,z[46:],'mean',subset)
        views=tuple(MAGNIFICATIONS[m] for m in subset)
        for j,c in enumerate(cases[46:]):
            prob=predict_fields({m:bags[46+j][i] for i,m in enumerate(MAGNIFICATIONS)},views,head)
            error=max(error,checked_difference(prob,reference[j]))
            records.append({'role':'validation','case_id':c['case_id'],'truth':int(y[46+j]),'magnifications':views,'probabilities':prob.tolist()})
    if error>1e-8:raise ValueError('Adapted case-head reload differs')
    (Path(output)/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return C,value,choices,error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','images-root','frozen-cache','reference-root','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--trial',type=int,choices=range(1,6),required=True);p.add_argument('--seed',type=int,choices=(812,813),required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--expected-protocol-sha',required=True);p.add_argument('--expected-config-sha',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if file_hash(a.protocol)!=a.expected_protocol_sha or file_hash(a.config)!=a.expected_config_sha:
        raise ValueError('Registered protocol/config hash differs')
    config=json.loads(a.config.read_text());encoder=json.loads((Path(__file__).resolve().parents[1]/'configs/encoders.json').read_text())
    if config!=json.loads((Path(__file__).resolve().parents[1]/'configs/breakhis-adaptation-v1.json').read_text()):
        raise ValueError('Use the exact registered adaptation recipe')
    trusted,fit,val,reference,context=prepare_context(a.manifest,a.protocol,a.trial,a.seed,a.reference_root,config,a.code_sha)
    context.update(base_encoder=encoder['phikon-v2'],smoke=a.smoke)
    if a.output.exists():
        if not a.resume or (a.output/'metrics.json').exists() or json.loads((a.output/'started.json').read_text())!=context:
            raise ValueError('Output exists or resume binding differs')
    else:a.output.mkdir(parents=True);(a.output/'started.json').write_text(json.dumps(context,indent=2)+'\n')
    rows=read_manifest(a.manifest);fit_ids={c['case_id'] for c in fit};allowed=fit_ids|{c['case_id'] for c in val}
    fit_rows=[r for r in rows if case_id(r) in fit_ids]
    encode_rows=[r for r in rows if case_id(r) in allowed]
    if len(fit)!=46 or len(val)!=8 or len(allowed)!=54:raise ValueError('Adaptation role counts differ')
    sampling_weights(fit_rows,fit)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Allocated CUDA unavailable')
    torch.set_num_threads(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))));torch.manual_seed(config['training_seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,transform,_,description=load_encoder('phikon-v2',encoder,device)
    if (model.config.model_type!=config['model_type'] or model.config.hidden_size!=config['dimension']
            or len(model.encoder.layer)!=config['encoder_layers']):raise ValueError('Pinned encoder architecture differs')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Encoder precision differs from registered float32')
    probes,base_features=frozen_probe(fit_rows,a.frozen_cache,reference)
    pixels=transform([load_checked_image(a.images_root,r) for r in probes]).to(device)
    with torch.inference_mode():initial=model(pixel_values=pixels).last_hidden_state[:,0,:].float().cpu().numpy()
    initial_error=checked_difference(initial,base_features)
    if initial_error>1e-4:raise ValueError('Initial base/cache feature parity differs: '+str(initial_error))
    started=time.monotonic();proof=train(model,transform,fit_rows,fit,a.images_root,config,context,a.output,device,a.workers,a.smoke,a.resume)
    model.eval()
    with torch.inference_mode():before=model(pixel_values=pixels).last_hidden_state[:,0,:].float().cpu().numpy()
    saved=bound_checkpoint(a.output/'last-checkpoint.pt',context)
    fresh,transform,_,_=load_encoder('phikon-v2',encoder,device);restore_fragment(fresh,saved['encoder_fragment']);fresh.eval()
    with torch.inference_mode():after=fresh(pixel_values=pixels).last_hidden_state[:,0,:].float().cpu().numpy()
    reload_error=checked_difference(before,after)
    if reload_error>1e-6:raise ValueError('Final encoder reload differs')
    del model;model=fresh
    proof.update(initial_base_feature_max_difference=initial_error,encoder_reload_max_difference=reload_error)
    if a.smoke:
        report={'status':'fitting-only technical smoke','context':context,'proof':proof,'validation_predictions_computed':0,
                'outer_predictions_computed':0,'elapsed_seconds':time.monotonic()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
        (a.output/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return
    labels={c['case_id']:c['label'] for c in fit+val};paths=[];values=[]
    with torch.inference_mode():
        for batch in loader(encode_rows,a.images_root,labels,config['batch_size'],a.workers):
            keys,_,images=zip(*batch);z=model(pixel_values=transform(list(images)).to(device)).last_hidden_state[:,0,:].float().cpu().numpy()
            if z.shape!=(len(batch),config['dimension']) or not np.isfinite(z).all():raise ValueError('Invalid adapted features')
            paths.extend(keys);values.append(z)
    if paths!=[r['relative_path'] for r in encode_rows] or len(set(paths))!=len(paths):raise ValueError('Adapted feature coverage differs')
    cache={'paths':np.asarray(paths),'features':np.concatenate(values)};np.savez_compressed(a.output/'adapted-features.npz',**cache)
    with np.load(a.output/'adapted-features.npz',allow_pickle=False) as restored:
        C,metric,candidates,head_error=fit_case_head(fit+val,restored,a.output)
    report={'status':'validation-only pilot','unit':trusted['unit'],'encoder':'phikon-v2-lastblock-v1','method':'mean',
        'trial':a.trial,'seed':a.seed,'training_seed':config['training_seed'],'code_sha':a.code_sha,
        'manifest_sha256':trusted['manifest_sha256'],'protocol_sha256':trusted['protocol_sha256'],'class_names':CLASS_NAMES,
        'case_ids':context['case_ids'],'selected_C':C,'validation':metric,'candidates':candidates,'config':config,'config_sha256':file_hash(a.config),
        'adaptation_proof':proof,'encoded_cases':54,'encoded_fields':len(paths),'fitting_fields':len(fit_rows),
        'fitting_class_cases':[sum(c['label']==i for c in fit) for i in range(8)],'normalization_policy':'field-moments-v2',
        'base_encoder':encoder['phikon-v2'],'transform':description,'feature_sha256':file_hash(a.output/'adapted-features.npz'),
        'head_sha256':file_hash(a.output/'linear-model.npz'),'head_reload_max_difference':head_error,
        'elapsed_seconds':time.monotonic()-started,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'outer_predictions_computed':0,'interpretation':'Fixed supervised adaptation package; exploratory provider-case validation, not fusion novelty or clean clinical confirmation'}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');verify_run(a.output/'metrics.json',trusted=trusted)
    print(json.dumps({'trial':a.trial,'seed':a.seed,'validation':metric,'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()

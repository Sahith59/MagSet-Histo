"""One registered supervised ResNet image-to-case BreakHis reference."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import PIL
from scipy.special import logsumexp
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler

from .adapt_breakhis import prepare_context, sampling_weights, bound_checkpoint, cpu_tree, checked_difference
from .breakhis_predict import load_checked_image
from .collect import SUBSETS, verify_run
from .collect_adaptation import digest_state
from .data import CLASS_NAMES, MAGNIFICATIONS, case_id, file_hash, read_manifest
from .diagnose_breakhis import metrics, case_summary
from .features import identity_collate, worker_init


def augment(image,path,epoch,draw,seed):
    bits=hashlib.sha256(('%d|%d|%d|%s'%(seed,epoch,draw,path)).encode()).digest()
    if bits[0]&1:image=ImageOps.mirror(image)
    if bits[0]&2:image=ImageOps.flip(image)
    rotation=bits[1]%4
    if rotation:image=image.transpose((Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270)[rotation-1])
    for j,operation in enumerate((ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color)):
        factor=.9+.2*int.from_bytes(bits[2+4*j:6+4*j],'big')/(2**32-1)
        image=operation(image).enhance(factor)
    return image


def pixels(image,size):
    x=np.asarray(image.resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/255.
    if x.shape!=(size,size,3):raise ValueError('Expected RGB canonical image')
    return (torch.from_numpy(x.copy()).permute(2,0,1)-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class DrawSampler(Sampler):
    def __init__(self,weights,seed,epoch):self.weights=weights;self.seed=seed;self.epoch=epoch
    def __len__(self):return len(self.weights)
    def __iter__(self):
        values=torch.multinomial(self.weights,len(self),replacement=True,generator=torch.Generator().manual_seed(self.seed+self.epoch*1000003))
        return iter((int(i),draw) for draw,i in enumerate(values))


class Images(Dataset):
    def __init__(self,rows,root,labels,config,epoch=0):self.rows=rows;self.root=root;self.labels=labels;self.config=config;self.epoch=epoch
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if isinstance(index,tuple):index,draw=index
        else:draw=0
        row=self.rows[index];image=load_checked_image(self.root,row)
        if self.epoch:image=augment(image,row['relative_path'],self.epoch,draw,self.config['training_seed'])
        return row['relative_path'],self.labels[case_id(row)],pixels(image,self.config['image_size'])


def stream(rows,root,cases,config,workers,epoch=0):
    ids={c['case_id'] for c in cases}
    if len(ids)!=len(cases) or not rows or len({r['relative_path'] for r in rows})!=len(rows) or any(case_id(r) not in ids for r in rows):
        raise ValueError('Image loader accepts unique allowed-role rows only')
    sampler=DrawSampler(sampling_weights(rows,cases),config['training_seed'],epoch) if epoch else None
    return DataLoader(Images(rows,root,{c['case_id']:c['label'] for c in cases},config,epoch),batch_size=config['batch_size'],
        sampler=sampler,shuffle=False,drop_last=False,num_workers=workers,collate_fn=identity_collate,worker_init_fn=worker_init,
        multiprocessing_context='spawn' if workers else None,persistent_workers=False,
        generator=torch.Generator().manual_seed(config['training_seed']+epoch))


def state_hash(model,kind):
    state=model.state_dict()
    if kind=='bn':selected={k:v for k,v in state.items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    elif kind=='backbone':selected={k:v for k,v in state.items() if not k.startswith('fc.')}
    elif kind=='early':selected={k:v for k,v in state.items() if k.startswith('conv1.')}
    elif kind=='late':selected={k:v for k,v in state.items() if k.startswith('layer4.') and not k.endswith(('running_mean','running_var','num_batches_tracked'))}
    else:raise ValueError('Unknown state fingerprint')
    if not selected:raise ValueError('Expected model component absent')
    return digest_state(selected)


def set_phase(model,adapting):
    model.train()
    for name,value in model.named_parameters():value.requires_grad_(adapting or name.startswith('fc.'))
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()


def learning_rates(config,epoch):
    first=config['head_only_epochs']+1
    if epoch<first:factor=1.
    else:
        progress=(epoch-first)/(config['epochs']-first)
        factor=config['minimum_learning_rate_fraction']+(1-config['minimum_learning_rate_fraction'])*.5*(1+math.cos(math.pi*progress))
    return config['backbone_learning_rate']*factor,config['head_learning_rate']*factor


def build_model(config,weights,device):
    import torchvision
    from torchvision.models import resnet50
    if torchvision.__version__.split('+')[0]!=config['torchvision_version']:raise ValueError('Pinned torchvision version differs')
    if file_hash(weights)!=config['weights_sha256']:raise ValueError('Pinned external weights checksum differs')
    model=resnet50(weights=None);state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
    before=state_hash(model,'backbone');torch.manual_seed(config['training_seed'])
    model.fc=nn.Sequential(nn.Dropout(config['head_dropout']),nn.Linear(config['dimension'],8))
    if state_hash(model,'backbone')!=before or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Initial backbone/dtype differs')
    return model.to(device)


def score_fields(model,rows,root,cases,config,device,workers):
    model.eval();paths=[];logits=[]
    with torch.inference_mode():
        for batch in stream(rows,root,cases,config,workers):
            keys,_,x=zip(*batch);value=model(torch.stack(x).to(device)).float().cpu().numpy()
            if value.shape!=(len(batch),8) or not np.isfinite(value).all():raise ValueError('Invalid image classifier outputs')
            paths.extend(keys);logits.append(value)
    if paths!=[r['relative_path'] for r in rows] or len(set(paths))!=len(paths):raise ValueError('Evaluation path coverage differs')
    z=np.concatenate(logits).astype(np.float64);lp=z-logsumexp(z,axis=1,keepdims=True)
    return {'paths':np.asarray(paths),'logits':z,'log_probabilities':lp,'probabilities':np.exp(lp)}


def aggregate(cases,fields,role):
    if role not in ('train','validation'):raise ValueError('Refusing outer scoring')
    paths=fields['paths'].tolist();lookup={p:i for i,p in enumerate(paths)};lp=fields['log_probabilities']
    expected={p for c in cases for values in c['indices'].values() for p in values}
    if set(paths)!=expected or len(paths)!=len(lookup) or lp.shape!=(len(paths),8) or not np.isfinite(lp).all():raise ValueError('Case/field coverage differs')
    records=[]
    for c in cases:
        views={m:logsumexp(lp[[lookup[p] for p in c['indices'][m]]],axis=0)-np.log(len(c['indices'][m])) for m in MAGNIFICATIONS}
        for subset in SUBSETS:
            value=logsumexp([views[m] for m in subset],axis=0)-np.log(len(subset))
            records.append({'role':role,'case_id':c['case_id'],'truth':c['label'],'magnifications':subset,
                            'probabilities':np.exp(value).tolist(),'log_probabilities':value.tolist()})
    return records,case_summary(records)


def atomic_save(record,path):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(record,temp);os.replace(temp,path)


def train(model,fit_rows,val_rows,fit,val,root,config,context,output,device,workers,smoke=False,resume=False):
    output=Path(output);initial={k:state_hash(model,k) for k in ('bn','backbone','early','late')}
    groups=[{'params':[p for n,p in model.named_parameters() if not n.startswith('fc.')],'lr':config['backbone_learning_rate']},
            {'params':[p for n,p in model.named_parameters() if n.startswith('fc.')],'lr':config['head_learning_rate']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay']);history=[];candidates=[];best=None;best_state=None;first=1
    last=output/'last.pt';selected=output/'selected.pt';epochs=config['smoke_epochs'] if smoke else config['epochs']
    torch.manual_seed(config['training_seed'])
    if resume and last.exists():
        saved=bound_checkpoint(last,context)
        if saved['initial']!=initial:raise ValueError('Resume pretrained state differs')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];candidates=saved['candidates'];best=saved['best'];best_state=saved['best_model'];first=saved['epoch']+1
        if best is not None:
            if digest_state(best_state)!=best['state_sha256']:raise ValueError('Resume selected snapshot differs')
            atomic_save({'context':context,'epoch':best['epoch'],'model':best_state},selected)
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    labels={r['relative_path']:r for r in fit_rows}
    for epoch in range(first,epochs+1):
        started=time.monotonic();adapting=epoch>config['head_only_epochs'];set_phase(model,adapting)
        rates=learning_rates(config,epoch)
        for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr
        loss_sum=draws=0;class_draws=[0]*8;case_draws=Counter();view_draws=Counter()
        for batch_index,batch in enumerate(stream(fit_rows,root,fit,config,workers,epoch)):
            if smoke and batch_index>=config['smoke_batches_per_epoch']:break
            paths,y,x=zip(*batch);target=torch.tensor(y,device=device);optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(torch.stack(x).to(device)),target,label_smoothing=config['label_smoothing'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite supervised fitting loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach())*len(batch);draws+=len(batch)
            for p,label in zip(paths,y):
                class_draws[label]+=1;row=labels[p];case_draws[case_id(row)]+=1;view_draws[int(row['magnification'])]+=1
            if (batch_index+1)%64==0:print(json.dumps({'epoch':epoch,'batches':batch_index+1,'draws':draws}),flush=True)
        if not draws or (not smoke and draws!=len(fit_rows)):raise ValueError('Registered draw budget differs')
        if state_hash(model,'bn')!=initial['bn']:raise ValueError('Pretrained BatchNorm statistics changed')
        if not adapting and state_hash(model,'backbone')!=initial['backbone']:raise ValueError('Backbone changed during head warmup')
        row={'epoch':epoch,'loss':loss_sum/draws,'draws':draws,'class_draws':class_draws,'case_draws':dict(case_draws),
             'view_draws':dict(view_draws),'backbone_adapting':adapting,'learning_rates':rates,
             'bn_sha256':state_hash(model,'bn'),'early_sha256':state_hash(model,'early'),'late_sha256':state_hash(model,'late'),
             'elapsed_seconds':time.monotonic()-started}
        history.append(row)
        if not smoke and epoch in config['selection_epochs']:
            values=score_fields(model,val_rows,root,val,config,device,workers);records,metric=aggregate(val,values,'validation')
            score_path=output/('selection-epoch%d.npz'%epoch);np.savez_compressed(score_path,**values)
            candidate_state=cpu_tree(model.state_dict())
            candidate={'epoch':epoch,'validation':metric,'fields_sha256':file_hash(score_path),'state_sha256':digest_state(candidate_state)};candidates.append(candidate)
            if best is None or metric['mean_subset_macro_f1']>best['validation']['mean_subset_macro_f1']:
                best=candidate;best_state=candidate_state
        record={'context':context,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'epoch':epoch,
                'history':history,'candidates':candidates,'best':best,'best_model':best_state,'initial':initial,
                'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        atomic_save(record,last)
        if best is not None:atomic_save({'context':context,'epoch':best['epoch'],'model':best_state},selected)
        print(json.dumps({'epoch':epoch,'loss':row['loss'],'draws':draws,'backbone_adapting':adapting}),flush=True)
    saved=bound_checkpoint(last,context)
    if saved['epoch']!=epochs:raise ValueError('Fixed epoch budget incomplete')
    if [h['epoch'] for h in history]!=list(range(1,epochs+1)) or (not smoke and [c['epoch'] for c in candidates]!=config['selection_epochs']):
        raise ValueError('History/selection epoch coverage differs')
    if not smoke and best!=max(candidates,key=lambda c:c['validation']['mean_subset_macro_f1']):raise ValueError('Selected checkpoint rule differs')
    if state_hash(model,'early')==initial['early'] or state_hash(model,'late')==initial['late']:raise ValueError('Early/late backbone failed to adapt')
    return {'initial':initial,'history':history,'candidates':candidates,'best':best,'last_sha256':file_hash(last),
            'final_bn_sha256':state_hash(model,'bn'),'final_early_sha256':state_hash(model,'early'),'final_late_sha256':state_hash(model,'late')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','reference-root','images-root','weights','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--trial',type=int,choices=range(1,6),required=True);p.add_argument('--seed',type=int,choices=(812,813),required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    if (source/'.source-sha').read_text().strip()!=a.code_sha:raise ValueError('Use immutable source')
    config=json.loads(a.config.read_text())
    if config!=json.loads((source/'configs/breakhis-supervised-v1.json').read_text()):raise ValueError('Registered recipe differs')
    trusted,fit,val,reference,context=prepare_context(a.manifest,a.protocol,a.trial,a.seed,a.reference_root,config,a.code_sha)
    context.update(implementation_sha256=file_hash(__file__),weights_sha256=config['weights_sha256'],smoke=a.smoke)
    if a.output.exists():
        if not a.resume or (a.output/'metrics.json').exists() or json.loads((a.output/'started.json').read_text())!=context:
            raise ValueError('Existing output/resume binding differs')
    else:a.output.mkdir(parents=True);(a.output/'started.json').write_text(json.dumps(context,indent=2)+'\n')
    rows=read_manifest(a.manifest);fit_ids={c['case_id'] for c in fit};val_ids={c['case_id'] for c in val}
    fit_rows=[r for r in rows if case_id(r) in fit_ids];val_rows=[r for r in rows if case_id(r) in val_ids]
    if len(fit)!=46 or len(val)!=8 or len(fit_ids|val_ids)!=54:raise ValueError('Role counts differ')
    sampling_weights(fit_rows,fit);sampling_weights(val_rows,val)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA allocation unavailable')
    torch.set_num_threads(2);torch.manual_seed(config['training_seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build_model(config,a.weights,device);probes=[next(r for r in fit_rows if int(r['magnification'])==m) for m in MAGNIFICATIONS]
    probe_pixels=torch.stack([pixels(load_checked_image(a.images_root,r),config['image_size']) for r in probes]).to(device)
    started=time.monotonic();proof=train(model,fit_rows,val_rows,fit,val,a.images_root,config,context,a.output,device,a.workers,a.smoke,a.resume)
    model.eval()
    if a.smoke:chosen=a.output/'last.pt'
    else:
        chosen=a.output/'selected.pt';saved=bound_checkpoint(chosen,context)
        if saved['epoch']!=proof['best']['epoch'] or digest_state(saved['model'])!=proof['best']['state_sha256']:
            raise ValueError('Selected state/epoch differs from recorded selection')
        model.load_state_dict(saved['model'],strict=True)
    with torch.inference_mode():before=model(probe_pixels).double().cpu().numpy()
    restored=build_model(config,a.weights,device);saved=bound_checkpoint(chosen,context);restored.load_state_dict(saved['model'],strict=True);restored.eval()
    with torch.inference_mode():after=restored(probe_pixels).double().cpu().numpy()
    error=checked_difference(before,after)
    if error>1e-6:raise ValueError('Selected model reload differs')
    proof['reload_logit_max_difference']=error;del model;model=restored
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':config['torchvision_version'],
                 'pillow':PIL.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    if a.smoke:
        report={'status':'fitting-only technical smoke','context':context,'proof':proof,'validation_predictions_computed':0,'outer_predictions_computed':0,
                'elapsed_seconds':time.monotonic()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'environment':environment}
        (a.output/'smoke.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Fitting-only supervised smoke passed',flush=True);return
    fit_fields=score_fields(model,fit_rows,a.images_root,fit,config,device,a.workers)
    val_fields=score_fields(model,val_rows,a.images_root,val,config,device,a.workers)
    fit_records,fit_metric=aggregate(fit,fit_fields,'train');val_records,val_metric=aggregate(val,val_fields,'validation')
    selected=proof['best'];expected=selected['validation']['mean_subset_macro_f1']
    if abs(val_metric['mean_subset_macro_f1']-expected)>1e-12:raise ValueError('Selected validation metric replay differs')
    with np.load(a.output/('selection-epoch%d.npz'%selected['epoch']),allow_pickle=False) as values:
        if values['paths'].tolist()!=val_fields['paths'].tolist() or checked_difference(values['probabilities'],val_fields['probabilities'])>1e-6:
            raise ValueError('Selected validation field replay differs')
    field_metrics={}
    for role,role_rows,cases,values,records in (('train',fit_rows,fit,fit_fields,fit_records),('validation',val_rows,val,val_fields,val_records)):
        np.savez_compressed(a.output/('%s-fields.npz'%role),**values)
        name='fitting-predictions.jsonl' if role=='train' else 'predictions.jsonl'
        (a.output/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
        labels={c['case_id']:c['label'] for c in cases};y=np.array([labels[case_id(r)] for r in role_rows])
        field_metrics[role]={'raw':metrics(values['probabilities'],y,log_probabilities=values['log_probabilities']),
            'balanced':metrics(values['probabilities'],y,sampling_weights(role_rows,cases).numpy(),values['log_probabilities'])}
    report={'status':'validation-only pilot','unit':trusted['unit'],'encoder':'resnet50-supervised-v1','method':'mean_probability',
        'trial':a.trial,'seed':a.seed,'training_seed':config['training_seed'],'code_sha':a.code_sha,'class_names':CLASS_NAMES,
        'manifest_sha256':trusted['manifest_sha256'],'protocol_sha256':trusted['protocol_sha256'],'case_ids':context['case_ids'],
        'config':config,'config_sha256':file_hash(a.config),'validation':val_metric,'fitting':fit_metric,'field_metrics':field_metrics,
        'selected_epoch':selected['epoch'],'proof':proof,'selected_model_sha256':file_hash(a.output/'selected.pt'),
        'field_artifact_sha256':{role:file_hash(a.output/('%s-fields.npz'%role)) for role in ('train','validation')},
        'elapsed_seconds':time.monotonic()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'torch':str(torch.__version__),
        'torchvision':config['torchvision_version'],'model_parameters':sum(p.numel() for p in model.parameters()),
        'environment':environment,
        'outer_predictions_computed':0,'interpretation':'One fixed supervised pipeline; exploratory provider-case validation, not novel mechanism or exact published-method reproduction'}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');verify_run(a.output/'metrics.json',trusted)
    print(json.dumps({'trial':a.trial,'seed':a.seed,'selected_epoch':selected['epoch'],'mean_f1':val_metric['mean_subset_macro_f1']}),flush=True)


if __name__=='__main__':main()

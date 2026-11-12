"""Validation-only, fixed-budget training of cached-field hierarchical fusion baselines."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .baseline import SUBSETS, stratified_inner_split
from .bags import load_bags, fit_normalization, normalize_bags, pad_bags
from .data import CLASS_NAMES, file_hash, read_manifest
from .field_baseline import probability_metrics, score_records
from .fusion import HierarchicalFusion


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def subset_mask(device):
    mask = torch.zeros((15,4),device=device,dtype=torch.bool)
    for i,subset in enumerate(SUBSETS):
        mask[i,list(subset)] = True
    return mask


def all_subset_logits(model, x, fields, masks):
    # Pooling is independent across views. Reuse it across the 15 exact subsets.
    pooled = model.pool_fields(x,fields)
    b,v,d = pooled.shape; s = len(masks)
    tokens = pooled.unsqueeze(0).expand(s,-1,-1,-1).reshape(s*b,v,d)
    observed = masks[:,None,:].expand(-1,b,-1).reshape(s*b,v)
    return model.fuse_views(tokens,observed).reshape(s,b,8)


def train_candidate(x, fields, y, train, val, config, choice, field_pool, view_fusion, seed):
    seed_everything(seed)
    model = HierarchicalFusion(x.shape[-1],config['hidden'],field_pool,view_fusion,config['dropout'],config['heads']).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=choice['learning_rate'],weight_decay=choice['weight_decay'])
    class_count = torch.bincount(y[train],minlength=8)
    if (class_count == 0).any():
        raise ValueError('All classes must occur in the fitting cases')
    weights = len(train)/(8*class_count.float())
    masks = subset_mask(x.device); losses = []
    generator = torch.Generator(device='cpu').manual_seed(seed)
    train_cpu = train.cpu()
    for epoch in range(1,config['epochs']+1):
        model.train(); total = 0.
        permutation = train_cpu[torch.randperm(len(train_cpu),generator=generator)]
        for ids in permutation.split(config['batch_size']):
            ids = ids.to(x.device); optimizer.zero_grad(set_to_none=True)
            logits = all_subset_logits(model,x[ids],fields[ids],masks)
            labels = y[ids].unsqueeze(0).expand(15,-1)
            loss = (F.cross_entropy(logits.reshape(-1,8),labels.reshape(-1),reduction='none')
                    * weights[labels.reshape(-1)]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total += loss.item()*len(ids)
        losses.append(total/len(train))
        if epoch == 1 or epoch % 25 == 0 or epoch == config['epochs']:
            print(json.dumps({'configuration':choice,'epoch':epoch,'training_loss':losses[-1]}),flush=True)
    # Exactly one validation check per candidate, at the prespecified final epoch.
    model.eval()
    with torch.inference_mode():
        probabilities = all_subset_logits(model,x[val],fields[val],masks).softmax(-1).cpu().numpy()
    metric = probability_metrics(probabilities,y[val].cpu().numpy())
    checkpoint = {'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  'optimizer':copy.deepcopy(optimizer.state_dict()),'epoch':config['epochs'],
                  'choice':choice,'training_seed':seed,'torch_rng_state':torch.get_rng_state(),
                  'cuda_rng_state':torch.cuda.get_rng_state_all() if x.device.type=='cuda' else [],
                  'shuffle_rng_state':generator.get_state(),
                  'python_rng_state':random.getstate(),
                  'numpy_rng_state':{'name':np.random.get_state()[0],
                                     'keys':torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                                     'position':np.random.get_state()[2],'has_gaussian':np.random.get_state()[3],
                                     'cached_gaussian':np.random.get_state()[4]}}
    return model,checkpoint,metric,probabilities,losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','features','output','config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--encoder',choices=['phikon-v2','resnet50'],required=True)
    p.add_argument('--field-pool',choices=['mean','attention'],required=True)
    p.add_argument('--view-fusion',choices=['mean','gated','attention'],required=True)
    p.add_argument('--trial',type=int,choices=range(1,6),required=True)
    p.add_argument('--seed',type=int,default=812,help='Inner split seed')
    p.add_argument('--training-seed',type=int,default=812)
    p.add_argument('--device',default='cuda'); p.add_argument('--code-sha',required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Output exists; use a new versioned directory')
    config = json.loads(a.config.read_text())
    if (config['epochs'] < 1 or config['batch_size'] < 1 or not config['grid'] or
            config['hidden'] % config['heads'] or not 0 <= config['dropout'] < 1):
        raise ValueError('Invalid training configuration')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation/framework unavailable')
    torch.set_num_threads(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))))
    mh = file_hash(a.manifest); protocol = json.loads(a.protocol.read_text())
    if protocol['manifest_sha256'] != mh:
        raise ValueError('Protocol and manifest differ')
    cases,bags,metadata = load_bags(read_manifest(a.manifest),a.features,a.encoder,mh)
    y_np = np.asarray([c['label'] for c in cases])
    train,val,test = stratified_inner_split(cases,protocol['trials'][str(a.trial)]['case_roles'],a.seed)
    mean,std = fit_normalization(bags,train)
    x_np,mask_np = pad_bags(normalize_bags(bags,mean,std))
    # Retain only fitting/validation features on device; no outer-test scoring code is provided.
    eligible = np.concatenate([train,val]); x = torch.from_numpy(x_np[eligible]).to(device)
    fields = torch.from_numpy(mask_np[eligible]).to(device); y = torch.from_numpy(y_np[eligible]).to(device)
    train_local = torch.arange(len(train),device=device); val_local = torch.arange(len(train),len(eligible),device=device)
    start = time.monotonic(); candidates = []; best = None
    for choice in config['grid']:
        model,checkpoint,metric,prob,losses = train_candidate(x,fields,y,train_local,val_local,config,choice,a.field_pool,a.view_fusion,a.training_seed)
        candidates.append({'configuration':choice,'validation':metric,'training_loss':losses})
        if best is None or metric['mean_subset_macro_f1'] > best[0]:
            best = metric['mean_subset_macro_f1'],checkpoint,metric,prob
    _,checkpoint,metric,prob = best
    a.output.mkdir(parents=True)
    checkpoint.update({'model_config':{'dimension':x.shape[-1],'hidden':config['hidden'],'field_pool':a.field_pool,
                                      'view_fusion':a.view_fusion,'dropout':config['dropout'],'heads':config['heads']},
                       'normalization_mean':torch.from_numpy(mean),'normalization_std':torch.from_numpy(std),
                       'code_sha':a.code_sha,'config_sha256':file_hash(a.config)})
    torch.save(checkpoint,a.output/'checkpoint.pt')
    # Verify reload prediction equality on real held-out validation inputs.
    restored = torch.load(a.output/'checkpoint.pt',map_location=device,weights_only=True)
    loaded = HierarchicalFusion(**restored['model_config']).to(device)
    loaded.load_state_dict(restored['model']); loaded.eval()
    with torch.inference_mode():
        reloaded = all_subset_logits(loaded,x[val_local],fields[val_local],subset_mask(device)).softmax(-1).cpu().numpy()
    if not np.allclose(reloaded,prob,atol=1e-7,rtol=1e-6):
        raise ValueError('Saved/reloaded predictions differ')
    report = {'status':'validation-only pilot','unit':protocol['unit'],'encoder':a.encoder,
              'method':'field-%s-view-%s'%(a.field_pool,a.view_fusion),'trial':a.trial,'seed':a.seed,
              'training_seed':a.training_seed,'code_sha':a.code_sha,'manifest_sha256':mh,
              'protocol_sha256':file_hash(a.protocol),'class_names':CLASS_NAMES,
              'config':config,'config_sha256':file_hash(a.config),'selected_configuration':checkpoint['choice'],
              'validation':metric,'candidates':candidates,
              'case_ids':{role:[cases[i]['case_id'] for i in ids] for role,ids in [('train',train),('validation',val),('test',test)]},
              'normalization':'training-case-balanced field moments per scale',
              'field_policy':'all fields; unpaired bags; same inputs for every hierarchical baseline',
              'model_selection':'fixed final epoch; four settings in production grid; validation only, ties choose earlier configuration',
              'training_objective':'case/class-balanced cross entropy averaged equally over all 15 nonempty subsets',
              'parameters':sum(v.numel() for v in loaded.parameters()),'elapsed_seconds':time.monotonic()-start,
              'precision':'float32; TF32 disabled; deterministic algorithms required','reload_verified':True,
              'checkpoint_sha256':file_hash(a.output/'checkpoint.pt'),'python':platform.python_version(),
              'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),
              'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
              'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
              'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
              'feature_provenance':[{k:v for k,v in m.items() if k in ['encoder','manifest_sha256','feature_sha256','precision']} for m in metadata]}
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in score_records(cases,val,y_np,prob)))
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['method','trial','selected_configuration','validation','elapsed_seconds','parameters']},indent=2),flush=True)


if __name__ == '__main__':
    main()

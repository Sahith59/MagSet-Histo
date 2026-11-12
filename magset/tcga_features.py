"""Label-blind frozen ImageNet ResNet encoding for the registered native TCGA task."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path, PurePosixPath
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import file_hash
from .features import Fields, identity_collate, worker_init, load_encoder
from .tcga_protocol import RESOLUTIONS


def native_rows(manifest,protocol,expected_protocol_sha=None):
    if expected_protocol_sha is not None and file_hash(protocol)!=expected_protocol_sha:
        raise ValueError('Native protocol differs from registered job hash')
    p=json.loads(Path(protocol).read_text())
    if p['manifest_sha256']!=file_hash(manifest):raise ValueError('Native manifest/protocol mismatch')
    with Path(manifest).open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=p['fields'] or len({r['relative_path'] for r in rows})!=len(rows):raise ValueError('Native coverage/paths differ')
    coverage=defaultdict(set);counts=Counter()
    for r in rows:
        path=PurePosixPath(r['relative_path'])
        if path.is_absolute() or '..' in path.parts or '\\' in str(path):raise ValueError('Unsafe native field path')
        if (r['resolution_mpp'] not in RESOLUTIONS or r['role']!=p['roles'].get(r['patient_id'])
                or int(r['label'])!=p['source_label_to_canonical'].get(r['label_id_source'])):
            raise ValueError('Native field identity/role/label differs')
        patient=p['patient_records'][r['patient_id']]
        if (path.parent.name!='mpp_'+r['resolution_mpp'] or path.stem.rsplit('_',2)[0]!=r['slide_uuid']
                or r['slide_uuid']!=patient['slide_uuid'] or r['label_id_source']!=patient['source_label_id']
                or int(r['label'])!=patient['label']):raise ValueError('Native path/patient identity differs')
        coverage[r['patient_id']].add(r['resolution_mpp']);counts[r['resolution_mpp']]+=1
    if set(coverage)!=set(p['roles']) or dict(counts)!=p['fields_by_resolution']:
        raise ValueError('Native patient/resolution coverage differs')
    if any(sorted(coverage[k])!=v['available_resolutions'] for k,v in p['patient_records'].items()):
        raise ValueError('Native available-view sets differ')
    return rows,p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','images-root','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resolution',choices=RESOLUTIONS,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--expected-protocol-sha',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();rows,protocol=native_rows(a.manifest,a.protocol,a.expected_protocol_sha)
    rows=[r for r in rows if r['resolution_mpp']==a.resolution]
    if a.limit<0:raise ValueError('Limit must be nonnegative')
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('Empty resolution shard')
    a.output_root.mkdir(parents=True,exist_ok=True)
    target=a.output_root/('resnet50-mpp%s.npz'%a.resolution);meta=target.with_suffix('.json')
    config=json.loads(Path('configs/encoders.json').read_text())
    mh=file_hash(a.manifest);ph=file_hash(a.protocol)
    if target.exists() or meta.exists():
        if not(target.exists() and meta.exists()):raise ValueError('Partial cache exists; inspect it')
        old=json.loads(meta.read_text())
        if (old['manifest_sha256']!=mh or old['protocol_sha256']!=ph or old['feature_sha256']!=file_hash(target)
                or old['images']!=len(rows) or old['encoder']!=config['resnet50'] or old['smoke_limit']!=a.limit):
            raise ValueError('Cache provenance differs')
        print('Verified existing native cache',flush=True);return
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Allocated CUDA unavailable')
    torch.set_num_threads(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))))
    torch.manual_seed(812);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,transform,encode,description=load_encoder('resnet50',config,device)
    # Fields reads only image paths/pixels. Labels/roles never enter the frozen encoder.
    loader=DataLoader(Fields(rows,a.images_root),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,
                      collate_fn=identity_collate,worker_init_fn=worker_init,
                      persistent_workers=a.workers>0,multiprocessing_context='spawn' if a.workers else None)
    start=time.monotonic();paths=[];features=[]
    with torch.inference_mode():
        for batch in loader:
            keys,images=zip(*batch);z=encode(transform(list(images)).to(device)).float().cpu().numpy()
            if z.shape!=(len(batch),2048) or not np.isfinite(z).all():raise ValueError('Invalid native embeddings')
            paths.extend(keys);features.append(z)
    temporary=target.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,paths=np.asarray(paths),features=np.concatenate(features))
    os.replace(temporary,target)
    record={'manifest_sha256':mh,'protocol_sha256':ph,'source_inventory_sha256':protocol['source_inventory_sha256'],
            'encoder':config['resnet50'],'resolution_mpp':a.resolution,'images':len(paths),'smoke_limit':a.limit,
            'code_sha':a.code_sha,'feature_sha256':file_hash(target),'precision':'float32; TF32 disabled',
            'elapsed_seconds':time.monotonic()-start,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'transform':repr(description),'torch':str(torch.__version__),'label_blind':True,
            'fit_status':'externally ImageNet-pretrained frozen encoder; no native fitting, tuning, or scoring',
            'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    temp_meta=meta.with_suffix('.json.tmp');temp_meta.write_text(json.dumps(record,indent=2)+'\n');os.replace(temp_meta,meta)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()

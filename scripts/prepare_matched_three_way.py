"""Zero-fit grouping, byte audit and shared three-crop pixels for matched models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from magset import matched_data as md
from magset.data import file_hash,read_manifest
from magset.five_collect import assert_public_safe


def audit(args):
    cfg=md.config(args.config);out=Path(args.output)
    if out.exists():raise ValueError('Refusing existing matched audit')
    if file_hash(args.manifest)!=cfg['manifest_sha256']:raise ValueError('Canonical manifest differs')
    rows=read_manifest(args.manifest);cases,fields=md.indexed_rows(rows)
    if len(fields)!=7909 or len(cases)!=82 or len({c['group']for c in cases})!=70:raise ValueError('Release identity counts differ')
    roles,fold,choice=md.split_cases(cases,seed=cfg['split']['seed'])
    counts=md.split_summary(cases,fields,roles)
    for role,wanted in cfg['expected_roles'].items():
        if any(counts[role][k]!=v for k,v in wanted.items()):raise ValueError('Pre-outcome registered role counts differ')
    for f in fields:
        path=Path(args.raw_root)/f['relative_path']
        if file_hash(path)!=f['sha256']:raise ValueError('Original image bytes differ')
    coverage_path=Path(__file__).resolve().parents[1]/cfg['midnight_p']['feature_config']
    if file_hash(coverage_path)!=cfg['midnight_p']['feature_config_sha256']:raise ValueError('Coverage recipe differs')
    coverage=json.loads(coverage_path.read_text())
    for mag,pin in coverage['cache_provenance'].items():
        path=Path(args.cache)/('midnight-coverage-v1-%s.npz'%mag)
        for file,key in [(path,'npz_sha256'),(path.with_suffix('.json'),'json_sha256'),(path.with_name(path.stem+'-components.npz'),'components_sha256')]:
            if file_hash(file)!=pin[key]:raise ValueError('Frozen label-independent feature cache changed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'indices.npz',field_case_index=np.array([f['case_index']for f in fields],np.int64),
        field_view=np.array([f['view']for f in fields],np.int64),case_y5=np.array([c['truth5']for c in cases],np.int64),
        case_y8=np.array([c['truth8']for c in cases],np.int64),case_fold=fold,**{k+'_cases':v for k,v in roles.items()})
    wb,wm=md.field_weights(cases,fields,roles['train'])
    np.savez_compressed(out/'training-weights.npz',binary=wb,malignant=wm)
    value=dict(md.context(args.code_sha,args.config),status='accepted matched grouping and input audit',
        manifest_sha256=file_hash(args.manifest),raw_root=str(Path(args.raw_root).resolve()),coverage_cache=str(Path(args.cache).resolve()),
        coverage_config_sha256=file_hash(coverage_path),coverage_cache_sha256=coverage['cache_provenance'],
        cases=cases,fields=fields,roles={k:v.tolist()for k,v in roles.items()},fold_assignment=fold.tolist(),split_choice=choice,
        role_summary=counts,indices_sha256=file_hash(out/'indices.npz'),training_weights_sha256=file_hash(out/'training-weights.npz'),
        new_fits=0,new_predictions=0)
    md.write_json(out/'private.json',value)
    public={k:value[k]for k in ['code_sha','source_archive_sha256','config_sha256','implementation_sha256','manifest_sha256',
        'status','role_summary','indices_sha256','training_weights_sha256','new_fits','new_predictions']}
    public.update(cases=82,conservative_groups=70,images=7909,test_fold=choice['test_fold'],validation_fold=choice['validation_fold'],
        group_overlap={'train_validation':0,'train_test':0,'validation_test':0},
        private_audit_sha256=file_hash(out/'private.json'),globally_fresh_clinical_cohort=False)
    assert_public_safe(public);md.write_json(out/'public.json',public)
    return public


def prepare(args):
    audit=md.load_audit(args.audit,args.config);out=Path(args.output)
    if out.exists():raise ValueError('Refusing existing crop cache')
    out.mkdir(parents=True);fields=audit['fields']
    pixels=np.lib.format.open_memmap(out/'pixels.npy',mode='w+',dtype=np.uint8,shape=(len(fields),3,224,224,3))
    def one(field):
        import hashlib
        data=(Path(args.raw_root)/field['relative_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=field['sha256']:raise ValueError('Field changed before shared crop extraction')
        with Image.open(BytesIO(data))as image:
            if image.size!=(int(field['width']),int(field['height'])):raise ValueError('Field geometry differs')
            return field['index'],md.crop_pixels(image)
    with ThreadPoolExecutor(max_workers=args.workers)as executor:
        for number,(index,value)in enumerate(executor.map(one,fields)):
            pixels[index]=value
            if number%500==0:print(json.dumps({'prepared_fields':number+1}),flush=True)
    pixels.flush();del pixels
    # Pixel-level parity against the exact operator used by the frozen P cache,
    # without running its encoder or any fitted model.
    from magset.coverage_features import crop_tensors
    parity=[]
    for width,height in ((700,460),(700,456)):
        field=next(f for f in fields if (int(f['width']),int(f['height']))==(width,height))
        with Image.open(Path(args.raw_root)/field['relative_path'])as image:
            tensors,_=crop_tensors(image)
            reference=np.rint((tensors.permute(0,2,3,1).numpy()+1.)*127.5).astype(np.uint8)
            actual=md.crop_pixels(image)
            difference=int(np.max(np.abs(reference.astype(int)-actual.astype(int))))
            if difference!=0:raise ValueError('Shared pixels differ from accepted P crop operators')
            parity.append({'raw_shape':[width,height],'maximum_uint8_abs':difference})
    value=dict(md.context(args.code_sha,args.config),status='accepted common three-crop pixels',images=len(fields),
        dtype='uint8',shape=[len(fields),3,224,224,3],pixels_sha256=file_hash(out/'pixels.npy'),
        input_audit_sha256=file_hash(Path(args.audit)/'private.json'),new_fits=0,new_predictions=0,crop_operator_parity=parity)
    md.write_json(out/'receipt.json',value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','prepare'))
    for key in ('config','manifest','raw-root','cache','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--audit',type=Path);p.add_argument('--code-sha',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(json.dumps((audit if a.action=='audit'else prepare)(a),indent=2))


if __name__=='__main__':main()

"""Register a new supervised TCGA patient quarantine before any classifier fitting."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from .data import file_hash

RESOLUTIONS=('0.25','0.5','1.0','2.0')
SALT='magset-tcga-quarantine-v1-20261112'
SOURCE_LABELS=('12','13','14','15','16','18','22','24','25','27','28','32','33','38','41','44')


def make_protocol(inventory, output, source_config=None):
    root=Path(inventory);output=Path(output)
    if output.exists():raise ValueError('Protocol directory exists; preserve the registered version')
    label_file=root/'metadata/tcga_ms/labels.csv'
    if source_config is None:source_config=json.loads(Path('configs/tcga-ms-source.json').read_text())
    if (file_hash(label_file)!=source_config['labels_sha256'] or
            file_hash(root/'members.csv')!=source_config['inventory_sha256']):
        raise ValueError('Native metadata differs from the pinned audited release')
    with label_file.open() as stream:labels=list(csv.DictReader(stream))
    by_slide={r['slide_uuid']:r for r in labels}
    if len(by_slide)!=len(labels) or len({r['case_uuid'] for r in labels})!=len(labels):
        raise ValueError('This release contract expects one labeled slide per patient')
    raw_ids=sorted({r['label_id'] for r in labels},key=int)
    if tuple(raw_ids)!=SOURCE_LABELS or len(labels)!=128 or set(Counter(r['label_id'] for r in labels).values())!={8}:
        raise ValueError('Unexpected native release label/patient count')
    mapping={label:i for i,label in enumerate(raw_ids)}
    fields=[];coverage=defaultdict(set)
    with (root/'members.csv').open() as stream:
        for row in csv.DictReader(stream):
            p=PurePosixPath(row['relative_path'])
            if p.is_absolute() or '..' in p.parts or '\\' in str(p):raise ValueError('Unsafe archive inventory path')
            if row['suffix']!='.png' or p.name.startswith('._'):continue
            resolution=p.parent.name.removeprefix('mpp_')
            if resolution not in RESOLUTIONS:continue
            slide,x,y=p.stem.rsplit('_',2)
            if slide not in by_slide or not x.isdigit() or not y.isdigit():raise ValueError('Unlabeled/invalid native field')
            label=by_slide[slide];patient=label['case_uuid'];coverage[patient].add(resolution)
            fields.append({'relative_path':str(p),'patient_id':patient,'slide_uuid':slide,
                           'resolution_mpp':resolution,'label_id_source':label['label_id'],
                           'label':mapping[label['label_id']],'bytes':int(row['bytes']),
                           'source_role_metadata':json.dumps({str(i):label['fold_%d'%i] for i in range(5)},sort_keys=True)})
    if len({r['relative_path'] for r in fields})!=len(fields):raise ValueError('Duplicate native image path')
    if set(coverage)!={r['case_uuid'] for r in labels}:raise ValueError('A labeled patient has no native images')
    roles={};incomplete=[]
    for raw_label in raw_ids:
        patients=sorted(r['case_uuid'] for r in labels if r['label_id']==raw_label)
        complete=[p for p in patients if coverage[p]==set(RESOLUTIONS)]
        complete.sort(key=lambda p:hashlib.sha256((SALT+'|'+p).encode()).hexdigest())
        if len(complete)<4:raise ValueError('Not enough complete patients for class-covered quarantine')
        for patient in patients:roles[patient]='fit'
        for patient in complete[:2]:roles[patient]='sealed_test'
        roles[complete[2]]='development'
        incomplete.extend(p for p in patients if p not in complete)
    for row in fields:row['role']=roles[row['patient_id']]
    counts=Counter(roles.values())
    if dict(counts)!={'fit':80,'development':16,'sealed_test':32}:raise ValueError('Registered role counts changed')
    if len(incomplete)!=6 or any(set(RESOLUTIONS)-coverage[p]!={'0.25'} for p in incomplete):
        raise ValueError('Native missing-resolution contract changed')
    if any(roles[p]!='fit' for p in incomplete):raise ValueError('Incomplete patients entered confirmation roles')
    if any(Counter(roles[r['case_uuid']] for r in labels if r['label_id']==raw)!={'fit':5,'development':1,'sealed_test':2} for raw in raw_ids):
        raise ValueError('Per-class role counts differ from registered protocol')
    output.mkdir(parents=True)
    manifest=output/'manifest.csv'
    fields.sort(key=lambda r:r['relative_path'])
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(fields[0]));writer.writeheader();writer.writerows(fields)
    report={'name':'TCGA-supervised-quarantine-v1','salt':SALT,'resolution_mpp':RESOLUTIONS,
            'source_label_order':raw_ids,'source_label_to_canonical':mapping,'patients':len(labels),
            'roles':roles,'role_counts':dict(counts),'incomplete_patients':sorted(incomplete),
            'patient_records':{r['case_uuid']:{'slide_uuid':r['slide_uuid'],'source_label_id':r['label_id'],
                              'label':mapping[r['label_id']],'available_resolutions':sorted(coverage[r['case_uuid']])}
                               for r in labels},
            'fields':len(fields),'fields_by_resolution':dict(Counter(r['resolution_mpp'] for r in fields)),
            'fields_by_role':dict(Counter(r['role'] for r in fields)),
            'class_patient_role_counts':{raw:dict(Counter(roles[r['case_uuid']] for r in labels if r['label_id']==raw)) for raw in raw_ids},
            'manifest_sha256':file_hash(manifest),'source_labels_sha256':file_hash(label_file),
            'source_inventory_sha256':file_hash(root/'members.csv'),
            'prediction_unit':'one native slide/patient set, four unpaired resolution bags',
            'selection_policy':'one prespecified fixed SHA256 ordering; no model scores used; salt must not be searched',
            'confirmation_policy':'sealed_test patients must never fit/tune any supervised model in this study; frozen label-blind encoding permitted',
            'incomplete_policy':'six naturally incomplete patients contribute fitting fields only; no natural-incomplete held-out performance claim',
            'provenance_policy':'secondary multi-organ16-class task; Phikon has known TCGA source pretraining, exact slide overlap unknown; ImageNet ResNet control first',
            'status':'registered metadata only; no classifier results'}
    (output/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=make_protocol(a.inventory,a.output)
    print(json.dumps({k:r[k] for k in ('name','patients','role_counts','fields','fields_by_resolution','manifest_sha256')},indent=2))


if __name__=='__main__':main()

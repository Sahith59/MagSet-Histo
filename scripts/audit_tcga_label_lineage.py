"""Audit pinned release labels and the read-only author loader without fitting."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_columns(source):
    candidates=[]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='label_map' for t in node.targets):continue
        value=node.value
        if not (isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='dict' and len(value.args)==1):continue
        pair=value.args[0]
        if not (isinstance(pair,ast.Call) and isinstance(pair.func,ast.Name) and pair.func.id=='zip' and len(pair.args)==2):continue
        if all(isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='label_df' and isinstance(x.slice,ast.Constant) for x in pair.args):
            candidates.append(tuple(x.slice.value for x in pair.args))
    if candidates!=[('slide_uuid','label_id')]:raise ValueError('Pinned loader target mapping differs or is ambiguous')
    return candidates[0]


def audit(labels,loader,provenance,source_config):
    config=json.loads(Path(source_config).read_text());code=json.loads(Path(provenance).read_text())
    if digest(labels)!=config['labels_sha256']:raise ValueError('Release label file differs from pinned source')
    records=[r for r in code['files'] if r['path']=='dataloaders.py']
    if len(records)!=1 or digest(loader)!=records[0]['sha256']:raise ValueError('Pinned author loader differs')
    columns=target_columns(Path(loader).read_text())
    with Path(labels).open() as stream:rows=list(csv.DictReader(stream))
    ids=sorted({r['label_id'] for r in rows},key=int)
    if (len(rows)!=128 or len(ids)!=16 or len({r['case_uuid'] for r in rows})!=128
            or len({r['slide_uuid'] for r in rows})!=128 or set(Counter(r['label_id'] for r in rows).values())!={8}):
        raise ValueError('Pinned patient/slide/class contract differs')
    classes=[]
    for canonical,label in enumerate(ids):
        selected=[r for r in rows if r['label_id']==label]
        descriptions=Counter(r['Histology/Behavior Description'] or '(blank)' for r in selected)
        original=sorted({r['label'] for r in selected})
        if len(original)!=1:raise ValueError('A released source target maps to multiple original label codes')
        classes.append({'canonical_id':canonical,'source_label_id':label,'original_label_code':original[0],
                        'cases':len(selected),'histology_descriptions':dict(sorted(descriptions.items())),
                        'histology_codes':dict(sorted(Counter(r['Histology/Behavior'] or '(blank)' for r in selected).items())),
                        'tissue_groups':dict(sorted(Counter(r['TCGA_tissue'] for r in selected).items()))})
    if len({r['original_label_code'] for r in classes})!=16:raise ValueError('Released numeric target columns are not bijective')
    folds={}
    for i in range(5):
        key='fold_%d'%i
        if set(r[key] for r in rows)!={'train','dev','test'}:raise ValueError('Unknown source role')
        folds[str(i)]={role:{'patients':sum(r[key]==role for r in rows),
                            'missing_source_classes':sorted(set(ids)-{r['label_id'] for r in rows if r[key]==role},key=int)}
                        for role in ('train','dev','test')}
    return {'schema':'magset.tcga-label-lineage.v1','source_labels_sha256':digest(labels),
            'author_code_revision':code['revision'],'author_loader_sha256':digest(loader),
            'author_loader_url':records[0]['url'],'confirmed_loader_columns':list(columns),
            'patients':128,'source_classes':16,'numeric_target_columns_are_bijective':True,
            'classes_with_multiple_descriptions':sum(len(c['histology_descriptions'])>1 for c in classes),
            'cases_with_blank_description':sum(not r['Histology/Behavior Description'] for r in rows),
            'classes':classes,'source_folds':folds,'classifier_fits_performed':0,'classifier_predictions_computed':0,
            'interpretation':'Numeric target lineage is confirmed. Text-description variation is not proof of label error; manual curation and exact source-ID/name crosswalk remain incompletely documented.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('labels','loader','provenance','source-config','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();result=audit(a.labels,a.loader,a.provenance,a.source_config)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('patients','source_classes','confirmed_loader_columns','classes_with_multiple_descriptions','cases_with_blank_description')}))

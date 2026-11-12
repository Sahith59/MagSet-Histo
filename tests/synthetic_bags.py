"""Tiny artificial features for software checks; never research-performance evidence."""
import csv
import json
from pathlib import Path
import numpy as np
from magset.data import CLASS_NAMES, MAGNIFICATIONS, file_hash


def write_fixture(root):
    root = Path(root); cache = root/'features'; cache.mkdir()
    rows = []; roles = {}; rng = np.random.default_rng(812)
    keys = {m:[] for m in MAGNIFICATIONS}; values = {m:[] for m in MAGNIFICATIONS}
    counts = (4,38,10,5,9,6,3,7); train_counts = (3,25,7,3,6,4,2,4)
    for label,subtype in enumerate(CLASS_NAMES):
        for k in range(counts[label]):
            raw = '14-%d'%(100*label+k); prefix = 'SOB_B_C%d'%label; case = prefix+'-'+raw
            roles[case] = 'train' if k < train_counts[label] else 'test'
            for scale in MAGNIFICATIONS:
                for field in range(2+k%2):
                    name = '%s-%s-%d-%03d.png'%(prefix,raw,scale,field)
                    path = '%s/%s/%d-%d.png'%(subtype,raw,scale,field)
                    rows.append({'filename':name,'relative_path':path,'case_id_raw':raw,
                                 'patient_group_conservative':raw,'subtype':subtype,'magnification':str(scale)})
                    keys[scale].append(path); values[scale].append(np.eye(8)[label]+rng.normal(0,.03,8))
    manifest = root/'manifest.csv'
    with manifest.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mh = file_hash(manifest)
    for scale in MAGNIFICATIONS:
        path = cache/('phikon-v2-%d.npz'%scale)
        np.savez_compressed(path,paths=np.asarray(keys[scale]),features=np.asarray(values[scale]))
        path.with_suffix('.json').write_text(json.dumps({'manifest_sha256':mh,'feature_sha256':file_hash(path),
            'magnification':scale,'images':len(keys[scale]),'smoke_limit':0,
            'encoder':{'name':'synthetic-fixture','dimension':8},'precision':'synthetic'}))
    protocol = root/'protocol.json'
    protocol.write_text(json.dumps({'manifest_sha256':mh,'unit':'synthetic cases',
                                    'trials':{'1':{'case_roles':roles}}}))
    return manifest,protocol,cache

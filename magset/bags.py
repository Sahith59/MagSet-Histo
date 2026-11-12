"""Validated frozen field bags and training-case-only, case-balanced normalization."""
import json
from pathlib import Path
import numpy as np
from .data import MAGNIFICATIONS, cases_from_rows, file_hash


def load_bags(rows, cache, encoder, manifest_sha256):
    cases = sorted(cases_from_rows(rows).values(), key=lambda c:c['case_id'])
    bags = [[None]*4 for _ in cases]
    metadata = []; all_paths = set(); dimensions = set()
    for m, scale in enumerate(MAGNIFICATIONS):
        path = Path(cache)/('%s-%d.npz'%(encoder,scale))
        meta = json.loads(path.with_suffix('.json').read_text())
        if (meta['feature_sha256'] != file_hash(path) or meta['manifest_sha256'] != manifest_sha256
                or meta['magnification'] != scale or meta.get('smoke_limit',0) != 0):
            raise ValueError('Feature checksum/manifest/scale/full-coverage provenance differs')
        with np.load(path, allow_pickle=False) as saved:
            keys = saved['paths'].tolist(); features = saved['features']
            if (features.ndim != 2 or len(keys) != len(features) or len(set(keys)) != len(keys)
                    or len(features) != meta['images'] or not np.isfinite(features).all()
                    or features.shape[1] != meta['encoder']['dimension']):
                raise ValueError('Invalid feature keys, dimensions, count, or values')
            expected = {r['relative_path'] for r in rows if int(r['magnification']) == scale}
            if set(keys) != expected or all_paths.intersection(keys):
                raise ValueError('Feature shard does not exactly cover its magnification')
            all_paths.update(keys); dimensions.add(features.shape[1])
            lookup = {key:i for i,key in enumerate(keys)}
            for i, case in enumerate(cases):
                paths = case['indices'].get(scale, [])
                if not paths:
                    raise ValueError('This pilot requires four-view cases')
                bags[i][m] = features[[lookup[p] for p in paths]].astype(np.float32)
        metadata.append(meta)
    if (len(dimensions) != 1 or len({json.dumps(m['encoder'],sort_keys=True) for m in metadata}) != 1
            or len(all_paths) != len(rows)):
        raise ValueError('Feature provenance or dimensions disagree')
    return cases, bags, metadata


def fit_normalization(bags, indices):
    """Every fitting case contributes equal mass at each scale, regardless of field count."""
    indices = list(indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Need unique nonempty fitting-case indices')
    mean = []; std = []
    for m in range(4):
        first = []; second = []
        for i in indices:
            x = np.asarray(bags[i][m],dtype=np.float64)
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError('Invalid fitting field bag')
            first.append(x.mean(0)); second.append(np.square(x).mean(0))
        mu = np.mean(first,axis=0)
        sigma = np.sqrt(np.maximum(np.mean(second,axis=0)-mu*mu,0))
        mean.append(mu); std.append(np.where(sigma>1e-6,sigma,1.))
    return np.asarray(mean,dtype=np.float32),np.asarray(std,dtype=np.float32)


def normalize_bags(bags, mean, std):
    return [[(x-mean[m])/std[m] for m,x in enumerate(case)] for case in bags]


def pad_bags(bags):
    if not bags:
        raise ValueError('No cases to pad')
    n = max(len(x) for case in bags for x in case); d = bags[0][0].shape[1]
    x = np.zeros((len(bags),4,n,d),dtype=np.float32)
    mask = np.zeros(x.shape[:3],dtype=bool)
    for i,case in enumerate(bags):
        for m,fields in enumerate(case):
            x[i,m,:len(fields)] = fields; mask[i,m,:len(fields)] = True
    return x,mask

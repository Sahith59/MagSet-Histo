"""Independently audit native frozen caches; never compute classifier predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import file_hash
from .tcga_features import native_rows
from .tcga_protocol import RESOLUTIONS


def verify(manifest,protocol,cache,protocol_sha):
    rows,p=native_rows(manifest,protocol,protocol_sha);results=[];sources=set()
    config=json.loads(Path('configs/encoders.json').read_text())['resnet50']
    for resolution in RESOLUTIONS:
        path=Path(cache)/('resnet50-mpp%s.npz'%resolution)
        meta=json.loads(path.with_suffix('.json').read_text())
        expected=[r['relative_path'] for r in rows if r['resolution_mpp']==resolution]
        if (meta['manifest_sha256']!=file_hash(manifest) or meta['protocol_sha256']!=protocol_sha
                or meta['feature_sha256']!=file_hash(path) or meta['images']!=len(expected)
                or meta['encoder']!=config or meta['smoke_limit']!=0 or meta['label_blind'] is not True):
            raise ValueError('Native cache provenance/count/label-blind contract differs')
        with np.load(path,allow_pickle=False) as data:
            z=data['features']
            if data['paths'].tolist()!=expected or z.shape!=(len(expected),2048) or z.dtype!=np.float32 or not np.isfinite(z).all():
                raise ValueError('Native feature keys/shape/dtype/values invalid')
        sources.add(meta['code_sha']);results.append({'resolution_mpp':resolution,'images':len(expected),
                                                    'feature_sha256':meta['feature_sha256']})
    if len(sources)!=1:raise ValueError('Native extraction source differs across shards')
    return {'status':'four frozen native caches verified','images':sum(r['images'] for r in results),
            'protocol_sha256':protocol_sha,'manifest_sha256':file_hash(manifest),
            'feature_source_sha':sources.pop(),'shards':results,'classifier_predictions_computed':0,
            'interpretation':'feature preparation only; no classifier fitting, tuning, or held-out result'}


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','cache','output'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--expected-protocol-sha',required=True);p=a.parse_args()
    r=verify(p.manifest,p.protocol,p.cache,p.expected_protocol_sha)
    p.output.parent.mkdir(parents=True,exist_ok=True);p.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

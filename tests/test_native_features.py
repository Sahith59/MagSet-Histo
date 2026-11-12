import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from magset.data import file_hash
from magset.tcga_protocol import make_protocol
from test_tcga_protocol import fixture
try:
    from magset.tcga_features import native_rows
    from magset.verify_native import verify
except ImportError:
    native_rows=None


@unittest.skipIf(native_rows is None,'Pinned Torch/Pillow needed for native feature adapter')
class NativeFeatureTests(unittest.TestCase):
    def test_all_cache_keys_values_and_provenance_are_verified_without_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');out=root/'registered'
            make_protocol(root/'inventory',out,source);manifest=out/'manifest.csv';protocol=out/'protocol.json'
            ph=file_hash(protocol);rows,p=native_rows(manifest,protocol,ph);cache=root/'cache';cache.mkdir()
            encoder=json.loads(Path('configs/encoders.json').read_text())['resnet50']
            for resolution in p['resolution_mpp']:
                keys=[r['relative_path'] for r in rows if r['resolution_mpp']==resolution]
                path=cache/('resnet50-mpp%s.npz'%resolution)
                np.savez_compressed(path,paths=np.asarray(keys),features=np.zeros((len(keys),2048),dtype=np.float32))
                path.with_suffix('.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest),
                    'protocol_sha256':ph,'feature_sha256':file_hash(path),'images':len(keys),'encoder':encoder,
                    'smoke_limit':0,'label_blind':True,'code_sha':'synthetic'}))
            report=verify(manifest,protocol,cache,ph)
            self.assertEqual(report['images'],p['fields']);self.assertEqual(report['classifier_predictions_computed'],0)
            path=cache/'resnet50-mpp0.25.npz'
            with np.load(path,allow_pickle=False) as saved:keys=saved['paths'];z=saved['features']
            z[0,0]=np.nan;np.savez_compressed(path,paths=keys,features=z)
            meta=json.loads(path.with_suffix('.json').read_text());meta['feature_sha256']=file_hash(path)
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'values invalid'):verify(manifest,protocol,cache,ph)

    def test_role_path_patient_and_registered_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');out=root/'registered'
            make_protocol(root/'inventory',out,source)
            manifest=out/'manifest.csv';protocol=out/'protocol.json';ph=file_hash(protocol)
            rows,p=native_rows(manifest,protocol,ph)
            self.assertEqual(len(rows),p['fields'])
            with self.assertRaisesRegex(ValueError,'registered job hash'):
                native_rows(manifest,protocol,'other')
            rows[0]['resolution_mpp']='2.0' if rows[0]['resolution_mpp']!='2.0' else '0.5'
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            p['manifest_sha256']=file_hash(manifest);protocol.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'path/patient'):
                native_rows(manifest,protocol,file_hash(protocol))


if __name__=='__main__':unittest.main()

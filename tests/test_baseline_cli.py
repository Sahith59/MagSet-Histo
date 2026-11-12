"""End-to-end smoke fixture: artifact checks and outer-score exclusion, not research accuracy."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from magset.data import CLASS_NAMES, MAGNIFICATIONS, file_hash


class BaselineCliTests(unittest.TestCase):
    def test_default_cli_keeps_outer_scores_out_of_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'features';cache.mkdir()
            rows=[];roles={};rng=np.random.default_rng(812)
            keys={m:[] for m in MAGNIFICATIONS};values={m:[] for m in MAGNIFICATIONS}
            for label,subtype in enumerate(CLASS_NAMES):
                # Synthetic filename prefixes are sufficient for this manifest-only test.
                for k in range(4):
                    raw='14-%d'%(label*100+k);prefix='SOB_B_C%d'%label;case=prefix+'-'+raw
                    roles[case]='train' if k<3 else 'test'
                    for m in MAGNIFICATIONS:
                        name='%s-%s-%d-001.png'%(prefix,raw,m)
                        path='%s/%s/%d.png'%(subtype,raw,m)
                        rows.append({'filename':name,'relative_path':path,'case_id_raw':raw,
                                     'patient_group_conservative':raw,'subtype':subtype,'magnification':str(m)})
                        keys[m].append(path)
                        z=np.eye(8)[label]+rng.normal(0,0.03,8)
                        values[m].append(z)
            manifest=root/'manifest.csv'
            with manifest.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            mh=file_hash(manifest)
            for m in MAGNIFICATIONS:
                p=cache/('phikon-v2-%d.npz'%m)
                np.savez_compressed(p,paths=np.array(keys[m]),features=np.array(values[m]))
                p.with_suffix('.json').write_text(json.dumps({'manifest_sha256':mh,'feature_sha256':file_hash(p),
                                                            'encoder':{'name':'synthetic-fixture'},'precision':'synthetic'}))
            protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'manifest_sha256':mh,'unit':'synthetic test cases',
                                            'trials':{'1':{'case_roles':roles}}}))
            output=root/'run'
            result=subprocess.run([sys.executable,'-m','magset.baseline','--manifest',str(manifest),
                '--protocol',str(protocol),'--features',str(cache),'--encoder','phikon-v2',
                '--method','mean','--trial','1','--output',str(output),'--code-sha','synthetic'],
                text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/'metrics.json').read_text())
            self.assertNotIn('test',report)
            records=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),8*15)
            self.assertEqual({r['role'] for r in records},{'validation'})
            self.assertTrue(all(roles[r['case_id']]=='train' for r in records))
            with np.load(output/'linear-model.npz',allow_pickle=False) as model:
                self.assertEqual(model['coef_0'].shape,(8,12))
                self.assertTrue(np.isfinite(model['coef_0']).all())


if __name__=='__main__':unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from magset.baseline import stratified_inner_split
from magset.collect import load_trusted_context, verify_trusted_report, verify_run
from synthetic_bags import write_fixture


class TrustedContextTests(unittest.TestCase):
    def fixture(self,root):
        manifest,protocol,_=write_fixture(root)
        trusted=load_trusted_context(manifest,protocol)
        cases=trusted['cases'];roles=trusted['trials']['1']['case_roles']
        train,val,test=stratified_inner_split(cases,roles,812)
        report={k:trusted[k] for k in ('manifest_sha256','protocol_sha256','unit')}
        report.update(trial=1,seed=812,case_ids={r:[cases[i]['case_id'] for i in idx]
                      for r,idx in [('train',train),('validation',val),('test',test)]})
        truth={cases[i]['case_id']:cases[i]['label'] for i in val}
        return trusted,report,truth

    def test_trusted_labels_roles_and_seed_are_independently_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            trusted,report,truth=self.fixture(tmp)
            verify_trusted_report(report,truth,trusted)
            wrong=dict(truth);key=next(iter(wrong));wrong[key]=(wrong[key]+1)%8
            with self.assertRaisesRegex(ValueError,'truth labels'):
                verify_trusted_report(report,wrong,trusted)
            changed=copy.deepcopy(report)
            changed['case_ids']['train'][0],changed['case_ids']['test'][0]=changed['case_ids']['test'][0],changed['case_ids']['train'][0]
            with self.assertRaisesRegex(ValueError,'case assignments'):
                verify_trusted_report(changed,truth,trusted)
            changed=copy.deepcopy(report);changed['seed']=813
            with self.assertRaisesRegex(ValueError,'case assignments'):
                verify_trusted_report(changed,truth,trusted)
            changed=copy.deepcopy(report);changed['manifest_sha256']='other'
            with self.assertRaisesRegex(ValueError,'trusted metadata'):
                verify_trusted_report(changed,truth,trusted)

    def test_one_trusted_path_without_the_other_is_rejected(self):
        with self.assertRaises(ValueError):load_trusted_context('manifest',None)

    def test_arithmetic_consistent_label_and_membership_tampering_fails_complete_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,protocol,cache=write_fixture(tmp);root=Path(tmp);out=root/'run'
            run=subprocess.run([sys.executable,'-m','magset.baseline','--manifest',str(manifest),
                '--protocol',str(protocol),'--features',str(cache),'--encoder','phikon-v2','--method','mean',
                '--trial','1','--output',str(out),'--code-sha','synthetic'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            trusted=load_trusted_context(manifest,protocol);path=out/'metrics.json';pred=out/'predictions.jsonl'
            report=json.loads(path.read_text());original=pred.read_text()
            verify_run(path,trusted=trusted)
            rows=[json.loads(line) for line in original.splitlines()]
            for row in rows:
                row['truth']=(row['truth']+1)%8
                row['probabilities']=[row['probabilities'][-1]]+row['probabilities'][:-1]
            pred.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            verify_run(path)  # A consistent permutation preserves all aggregate arithmetic.
            with self.assertRaisesRegex(ValueError,'truth labels'):verify_run(path,trusted=trusted)
            rows=[json.loads(line) for line in original.splitlines()]
            old=report['case_ids']['validation'][0];label=trusted['labels'][old]
            replacement=next(c for c in report['case_ids']['train'] if trusted['labels'][c]==label)
            report['case_ids']['validation'][0]=replacement
            report['case_ids']['train'][report['case_ids']['train'].index(replacement)]=old
            for row in rows:
                if row['case_id']==old:row['case_id']=replacement
            pred.write_text(''.join(json.dumps(r)+'\n' for r in rows));path.write_text(json.dumps(report))
            verify_run(path)
            with self.assertRaisesRegex(ValueError,'case assignments'):verify_run(path,trusted=trusted)


if __name__=='__main__':unittest.main()

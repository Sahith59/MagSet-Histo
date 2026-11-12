import json
from pathlib import Path
import tempfile
import unittest
from magset.collect import metrics_from_scores, collect, verify_run, SUBSETS
from magset.data import CLASS_NAMES


def fixture(root):
    folder=Path(root)/'phikon-v2-mean-trial1-seed812';folder.mkdir()
    records=[]
    for subset in SUBSETS:
        for label in range(8):
            prob=[0.]*8;prob[label]=1.
            records.append({'role':'validation','case_id':'val%d'%label,'truth':label,
                            'magnifications':subset,'probabilities':prob})
    (folder/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    metric={'macro_f1':1.,'accuracy':1.,'balanced_accuracy':1.}
    report={'status':'validation-only pilot','class_names':CLASS_NAMES,'encoder':'phikon-v2',
            'method':'mean','trial':1,'seed':812,'code_sha':'source','manifest_sha256':'manifest',
            'protocol_sha256':'protocol','unit':'synthetic cases','selected_C':0.1,
            'case_ids':{'train':['train%d'%i for i in range(46)],'validation':['val%d'%i for i in range(8)],'test':['test%d'%i for i in range(28)]},
            'validation':{'mean_subset_macro_f1':1.,'worst_subset_macro_f1':1.,
                          'subsets':{'+'.join(map(str,s)):metric.copy() for s in SUBSETS}}}
    path=folder/'metrics.json';path.write_text(json.dumps(report));return path,report


class CollectTests(unittest.TestCase):
    def test_score_metrics_with_known_errors(self):
        records=[]
        for truth,pred in [(0,0),(0,1),(1,1),(1,1)]:
            p=[0.]*8;p[pred]=1.;records.append({'truth':truth,'probabilities':p})
        result=metrics_from_scores(records)
        self.assertEqual(result['accuracy'],0.75)
        self.assertEqual(result['balanced_accuracy'],0.75)
        self.assertAlmostEqual(result['macro_f1'],((2/3)+(4/5))/8)

    def test_incomplete_matrix_has_no_ranking_and_fails_strict_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=fixture(tmp);verify_run(path)
            with self.assertRaises(ValueError):collect(tmp)
            report=collect(tmp,allow_incomplete=True)
            self.assertEqual(report['verified_runs'],1);self.assertEqual(len(report['missing']),29)
            self.assertEqual(report['summary'],[])

    def test_wrong_reported_metric_or_test_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(tmp)
            report['validation']['subsets']['40']['macro_f1']=0.9
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):verify_run(path)
        with tempfile.TemporaryDirectory() as tmp:
            path,_=fixture(tmp);p=path.parent/'predictions.jsonl';text=p.read_text().replace('"role": "validation"','"role": "test"',1);p.write_text(text)
            with self.assertRaises(ValueError):verify_run(path)

    def test_mixed_training_seeds_and_case_assignments_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(tmp)
            other=path.parent.with_name('other');other.mkdir()
            report.update(method='other',training_seed=813)
            (other/'metrics.json').write_text(json.dumps(report))
            (other/'predictions.jsonl').write_text((path.parent/'predictions.jsonl').read_text())
            with self.assertRaisesRegex(ValueError,'training_seed'):
                collect(tmp,methods=('mean','other'),encoders=('phikon-v2',),trials=(1,))
            report.pop('training_seed');report['case_ids']['train'][0]='different-fitting-case'
            (other/'metrics.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'case membership'):
                collect(tmp,methods=('mean','other'),encoders=('phikon-v2',),trials=(1,))


if __name__=='__main__':unittest.main()

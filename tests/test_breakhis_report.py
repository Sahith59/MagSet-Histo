import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from magset.collect import SUBSETS
from magset.data import CLASS_NAMES,MAGNIFICATIONS,file_hash
from scripts.report_breakhis_milestone import verify_demo


class BreakhisReportTests(unittest.TestCase):
    def fixture(self,root):
        p=Path(root)/'head.npz'
        np.savez_compressed(p,mean=np.zeros((4,1024),dtype=np.float32),std=np.ones((4,1024),dtype=np.float32),
            coef_0=np.zeros((8,1028)),intercept_0=np.zeros(8),classes_0=np.arange(8))
        report={'case_ids':{'validation':['z','a']},'feature_provenance':[{'encoder':{'name':'synthetic'}}]}
        trusted={'cases':[{'case_id':'a','label':0,'indices':{m:['example%d'%m] for m in MAGNIFICATIONS}}],
                 'manifest_sha256':'manifest','protocol_sha256':'protocol'}
        archive=[{'case_id':'a','magnifications':s,'probabilities':[.6,.4,0,0,0,0,0,0]} for s in SUBSETS]
        demo={'status':'raw-image validation replay verified','dataset':'BreakHis','role':'validation',
              'selection_rule':'lexicographically first validation case','case_id':'a','truth':CLASS_NAMES[0],
              'encoder':{'name':'synthetic'},'manifest_sha256':'manifest','protocol_sha256':'protocol','trial':1,'seed':812,
              'magnifications':list(MAGNIFICATIONS),'class_names':CLASS_NAMES,'cached_validation_records_replayed':120,
              'fields_by_view':{str(m):1 for m in MAGNIFICATIONS},'image_count':4,'model_sha256':file_hash(p),
              'new_classifier_fits':0,'outer_predictions_computed':0,'raw_replay_max_probability_difference':0.,
              'cached_replay_max_probability_difference':0.,'demo_code_sha':'a'*40,
              'demo_code_sha256':file_hash('magset/breakhis_predict.py'),
              'predictions':{'+'.join(map(str,s)):{'probabilities':[.6,.4,0,0,0,0,0,0],
                          'predicted_subtype':CLASS_NAMES[0],'archive_max_probability_difference':0.} for s in SUBSETS}}
        return p,report,trusted,archive,demo

    def test_demo_acceptance_is_tied_to_actual_head_source_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,r,t,a,d=self.fixture(tmp);verify_demo(d,r,t,a,p,'a'*40)
            with self.assertRaisesRegex(ValueError,'source'):verify_demo(d,r,t,a,p,'b'*40)
            with self.assertRaisesRegex(ValueError,'head'):verify_demo(d,r,t,a,None,'a'*40)

    def test_nan_wrong_labels_counts_vectors_and_archive_claims_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,r,t,a,d=self.fixture(tmp)
            changes=[('raw_replay_max_probability_difference',float('nan')),
                     ('cached_replay_max_probability_difference',float('nan')),
                     ('raw_replay_max_probability_difference',-1),('truth',CLASS_NAMES[1]),('case_id','z'),
                     ('image_count',-1),('fields_by_view',{'40':-1}),('model_sha256','unknown'),
                     ('encoder',{'name':'wrong'}),('role','test'),('demo_code_sha256','unknown')]
            for key,value in changes:
                with self.subTest(field=key):
                    bad=copy.deepcopy(d);bad[key]=value
                    with self.assertRaises(ValueError):verify_demo(bad,r,t,a,p,'a'*40)
            for field,value in [('probabilities',[float('nan')]*8),('probabilities',[.5,.5,0,0,0,0,0,0]),
                                ('predicted_subtype',CLASS_NAMES[1]),('archive_max_probability_difference',float('nan'))]:
                with self.subTest(payload=field):
                    bad=copy.deepcopy(d);bad['predictions']['40'][field]=value
                    with self.assertRaises(ValueError):verify_demo(bad,r,t,a,p,'a'*40)


if __name__=='__main__':unittest.main()

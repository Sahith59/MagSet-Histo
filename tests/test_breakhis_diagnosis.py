import copy
import unittest
from unittest.mock import patch
import numpy as np
try:
    import torch
    from magset.diagnose_breakhis import metrics,auxiliary_scores,auxiliary_case,weighted_drift,case_summary,case_records,replay_archive,summarize_runs,READOUTS
    from magset.collect import SUBSETS
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Pinned Torch needed for saved auxiliary head checks')
class DiagnosisTests(unittest.TestCase):
    def head(self,d=1):
        w=torch.zeros(8,d);w[0,0]=1;b=torch.full((8,),-1000.);b[:2]=0
        return {'1.weight':w,'1.bias':b}

    def test_weighted_errors_and_stable_log_loss(self):
        z=np.tile(np.array([1000.,0.,-1000,-1000,-1000,-1000,-1000,-1000]),(4,1))
        lp=z-np.logaddexp.reduce(z,axis=1,keepdims=True);p=np.exp(lp);y=np.array([0,0,0,1])
        raw=metrics(p,y,log_probabilities=lp);balanced=metrics(p,y,np.array([1/6]*3+[1/2]),lp)
        self.assertEqual(raw['accuracy'],.75);self.assertEqual(balanced['accuracy'],.5)
        self.assertEqual(balanced['balanced_accuracy'],.5);self.assertAlmostEqual(balanced['cross_entropy'],500.)
        with self.assertRaises(ValueError):metrics(p,y,np.array([1.,1.,-1.,1.]))
        with self.assertRaises(ValueError):metrics(p,y,np.array([1.,1.,1.,float('nan')]))

    def test_equal_view_weighting_and_affine_commutation(self):
        fields={40:np.ones((10,1)),100:np.array([[-2.]])};a,b,error,_,_=auxiliary_case(fields,(40,100),self.head())
        self.assertEqual(a.argmax(),1);self.assertEqual(b.argmax(),1);self.assertLess(error,1e-10)
        _,flat,_=auxiliary_scores(np.concatenate(list(fields.values())),self.head())
        self.assertEqual(flat.mean(0).argmax(),0)

    def test_probability_pooling_really_differs_from_logit_pooling(self):
        fields={40:np.array([[10.],[-2.],[-2.]])};logit,prob,error,_,_=auxiliary_case(fields,(40,),self.head())
        self.assertEqual(logit.argmax(),0);self.assertEqual(prob.argmax(),1);self.assertLess(error,1e-10)

    def test_excluded_views_and_field_order_do_not_change_auxiliary_readouts(self):
        fields={40:np.array([[1.],[3.],[-2.]]),100:np.ones((2,1)),400:np.array([[float('nan')]])}
        a=auxiliary_case(fields,(40,100),self.head())
        fields[40]=fields[40][::-1];b=auxiliary_case(fields,(100,40),self.head())
        for x,y in zip(a[:2],b[:2]):np.testing.assert_allclose(x,y,rtol=0,atol=1e-14)
        with self.assertRaises(ValueError):auxiliary_case(fields,(40,40),self.head())

    def test_zero_norm_policy_and_nonfinite_feature_rejection(self):
        a=np.array([[1.,0.],[0.,0.]]);r=weighted_drift(a,a,[.5,.5])
        self.assertEqual(r['cosine'],1.);self.assertEqual(r['relative_l2'],0.);self.assertEqual(r['cosine_valid_mass'],.5)
        self.assertIsNone(weighted_drift(a[1:],a[1:],[1.])['cosine'])
        with self.assertRaises(ValueError):weighted_drift(a,a*np.nan,[.5,.5])

    def test_case_loss_remains_exact_for_underflowed_probabilities(self):
        fields={'case1':{m:np.ones((2,1),np.float32) for m in (40,100,200,400)}}
        aux=self.head();aux['1.weight'][0,0]=1000.
        head={'mean':np.zeros((4,1),np.float32),'std':np.ones((4,1),np.float32),'coef_0':np.zeros((8,5)),
              'intercept_0':aux['1.bias'].numpy(),'classes_0':np.arange(8)};head['coef_0'][0,0]=1000.
        records,_=case_records([{'case_id':'case1','label':1}],fields,fields,(head,head),aux,'validation',{'affine_pooling_tolerance':1e-10})
        for name in READOUTS:
            s=case_summary([r for r in records if r['readout']==name])
            self.assertAlmostEqual(s['mean_subset_class_balanced_cross_entropy'],1000.)
        with self.assertRaisesRegex(ValueError,'Stable log'):
            metrics(np.eye(8)[[0]],np.array([1]))

    def test_saved_readout_pipeline_does_no_fitting_and_rejects_bad_coverage(self):
        rng=np.random.default_rng(8);cases=[{'case_id':'case%d'%i,'label':i} for i in range(8)]
        fields={c['case_id']:{m:rng.normal(size=(j+2,2)).astype(np.float32) for j,m in enumerate((40,100,200,400))} for c in cases}
        h={'mean':np.zeros((4,2),np.float32),'std':np.ones((4,2),np.float32),'coef_0':rng.normal(size=(8,6)),
           'intercept_0':rng.normal(size=8),'classes_0':np.arange(8)}
        with (patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('No fitting allowed')),
              patch('torch.optim.AdamW',side_effect=AssertionError('No optimizer allowed'))):
            records,error=case_records(cases,fields,fields,(h,h),self.head(2),'validation',{'affine_pooling_tolerance':1e-10})
        self.assertEqual(len(records),8*15*4);self.assertLess(error,1e-10)
        one=[r for r in records if r['readout']==READOUTS[0]];s=case_summary(one);self.assertEqual(len(s['subsets']),15)
        archive={(r['case_id'],tuple(r['magnifications'])):np.array(r['probabilities']) for r in one}
        self.assertLess(replay_archive(cases,fields,h,archive),1e-8)
        bad_archive=copy.deepcopy(archive);key=next(iter(bad_archive));bad_archive[key]=np.roll(bad_archive[key],1)
        with self.assertRaises(ValueError):replay_archive(cases,fields,h,bad_archive)
        with self.assertRaises(ValueError):case_records(cases,fields,fields,(h,h),self.head(2),'test',{'affine_pooling_tolerance':1e-10})
        for bad in (one[:-1],one+[one[0]]):
            with self.assertRaises(ValueError):case_summary(bad)
        changed=copy.deepcopy(one);changed[1]['truth']=(changed[1]['truth']+1)%8
        with self.assertRaises(ValueError):case_summary(changed)
        with self.assertRaises(ValueError):summarize_runs([])


if __name__=='__main__':unittest.main()

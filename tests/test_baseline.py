import unittest
import numpy as np
from scipy.special import softmax
from magset.baseline import SUBSETS, stratified_inner_split, vectorize, fit_models, predict_models, evaluate


class BaselineTests(unittest.TestCase):
    def test_exact_fifteen_nonempty_subsets(self):
        self.assertEqual(len(SUBSETS),15)
        self.assertEqual(len(set(SUBSETS)),15)
        self.assertEqual(SUBSETS[-1],(0,1,2,3))

    def test_absent_views_have_no_effect_even_with_nan_placeholders(self):
        z=np.arange(24,dtype=float).reshape(2,4,3)
        corrupted=z.copy();corrupted[:,1:,:]=np.nan
        for kind in ['mean','concat']:
            np.testing.assert_array_equal(vectorize(z,(0,),kind),vectorize(corrupted,(0,),kind))
            with self.assertRaises(ValueError):vectorize(z,(),kind)

    def test_inner_split_uses_only_outer_training_cases_and_covers_classes(self):
        cases=[];roles={}
        for label in range(8):
            for k in range(4):
                key='%d-%d'%(label,k);cases.append({'case_id':key,'label':label});roles[key]='train' if k<3 else 'test'
        train,val,test=stratified_inner_split(cases,roles,812)
        self.assertEqual(set(train)&set(val),set())
        self.assertEqual((set(train)|set(val))&set(test),set())
        self.assertEqual({cases[i]['label'] for i in val},set(range(8)))
        self.assertEqual({cases[i]['label'] for i in train},set(range(8)))
        again=stratified_inner_split(cases,roles,812)
        for x,y in zip((train,val,test),again):np.testing.assert_array_equal(x,y)

    def test_linear_weight_scores_reproduce_sklearn_predictions(self):
        rng=np.random.default_rng(7)
        y=np.repeat(np.arange(8),3)
        z=rng.normal(size=(24,4,10))
        z[:,:,0]+=y[:,None]
        for kind in ['mean','concat','late']:
            models=fit_models(z,y,kind,0.1)
            p=predict_models(models,z,kind,(0,2))
            self.assertEqual(p.shape,(24,8))
            np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-7)
            if kind!='late':
                x=vectorize(z,(0,2),kind);m=models[0]
                np.testing.assert_allclose(p,softmax(x@m.coef_.T+m.intercept_,axis=1),atol=1e-7)
            self.assertEqual(len(evaluate(models,z,y,kind)['subsets']),15)


if __name__=='__main__':unittest.main()

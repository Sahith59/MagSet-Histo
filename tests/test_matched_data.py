"""Shared split/pixel/objective and case-level evaluation invariants."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from magset import matched_data as md
from magset.five_metrics import summarize_records


def fixture():
    cases=[dict(case_id='synthetic%d'%i,group='synthetic-group%d'%i,truth5=i,truth8=[0,1,3,4,5][i])for i in range(5)]
    fields=[]
    for i in range(5):
        for view in range(4):
            for n in range(1+(i+view)%3):fields.append(dict(index=len(fields),case_index=i,view=view))
    return cases,fields


class MatchedDataTests(unittest.TestCase):
    def test_group_split_retains_mixed_label_groups_without_relabelling(self):
        cases=[]
        for label in range(5):
            for n in range(12):
                cases.append(dict(case_id='case-%d-%02d'%(label,n),group='group-%d-%02d'%(label,n),truth5=label,truth8=[0,1,3,4,5][label]))
        cases[12]['group']=cases[0]['group']
        roles,fold,choice=md.split_cases(cases,require_version=False)
        self.assertEqual(fold[0],fold[12]);self.assertNotEqual(cases[0]['truth5'],cases[12]['truth5'])
        self.assertEqual(sorted(np.concatenate(list(roles.values())).tolist()),list(range(len(cases))))
    def test_head_weights_balance_classes_cases_views_and_mask_benign(self):
        cases,fields=fixture();wb,wm=md.field_weights(cases,fields,np.arange(5));n=len(fields)
        self.assertAlmostEqual(wb.sum(),n);self.assertAlmostEqual(wm.sum(),n)
        for i in range(5):
            ix=[f['index']for f in fields if f['case_index']==i]
            self.assertAlmostEqual(wb[ix].sum(),n/2 if i==0 else n/8)
            self.assertAlmostEqual(wm[ix].sum(),0 if i==0 else n/4)
            for v in range(4):
                selected=[f['index']for f in fields if f['case_index']==i and f['view']==v]
                self.assertAlmostEqual(wb[selected].sum(),wb[ix].sum()/4)

    def test_truth8_and_explicit_test_optin_preserve_historical_guards(self):
        cases,fields=fixture();ix=np.arange(len(fields));p=np.array([np.eye(5)[cases[f['case_index']]['truth5']]for f in fields])
        records=md.case_records(fields,cases,np.arange(5),ix,p,'test')
        self.assertEqual(len(records),75);self.assertEqual(md.summary(records)['mean_subset_macro_f1'],1.)
        with self.assertRaises(ValueError):summarize_records(records,'dual')
        for i,c in enumerate(cases):self.assertTrue(all(r['truth']==c['truth8']for r in records if r['case_id']==c['case_id']))

    def test_field_permutation_and_zoom_subset_invariance(self):
        cases,fields=fixture();rng=np.random.RandomState(3);p=rng.dirichlet(np.ones(5),len(fields));ix=np.arange(len(fields))
        a=md.case_records(fields,cases,np.arange(5),ix,p,'validation')
        order=rng.permutation(len(fields));b=md.case_records(fields,cases,np.arange(5),ix[order],p[order],'validation')
        for x,y in zip(a,b):np.testing.assert_allclose(x['probabilities5'],y['probabilities5'],atol=1e-15)
        changed=p.copy();changed[[f['index']for f in fields if f['view']==3]]=np.eye(5)[0]
        c=md.case_records(fields,cases,np.arange(5),ix,changed,'validation')
        for x,y in zip(a,c):
            if 400 not in x['magnifications']:np.testing.assert_array_equal(x['probabilities5'],y['probabilities5'])

    def test_shared_crop_geometry_does_not_square_resize(self):
        for shape,(resized,lefts)in md.GEOMETRY.items():
            values=np.random.RandomState(shape[1]).randint(0,256,(shape[1],shape[0],3),dtype=np.uint8);image=Image.fromarray(values)
            actual=md.crop_pixels(image);scaled=image.resize(resized,Image.Resampling.BILINEAR)
            self.assertEqual(actual.shape,(3,224,224,3))
            for i,left in enumerate(lefts):np.testing.assert_array_equal(actual[i],np.asarray(scaled.crop((left,0,left+224,224))))
        with self.assertRaises(ValueError):md.crop_pixels(Image.new('RGB',(224,224)))

    def test_head_composition_and_half_threshold_tie(self):
        b=np.tile([.5,.5],(8,1));m=np.tile([0,1,0,0],(8,1));q=md.joint_probabilities(b,m)
        np.testing.assert_array_equal(q,np.tile([.5,0,.5,0,0],(8,1)))
        bad=b.copy();bad[0]=[0,0]
        with self.assertRaises(ValueError):md.joint_probabilities(bad,m)

    def test_registered_local_manifest_group_split_when_available(self):
        path=Path('.local/data-audit/manifest.csv')
        if not path.exists():self.skipTest('Real manifest is private')
        cases,fields=md.indexed_rows(md.read_manifest(path));roles,fold,choice=md.split_cases(cases,require_version=False)
        self.assertEqual((choice['test_fold'],choice['validation_fold']),(0,1))
        self.assertEqual([len(roles[k])for k in md.ROLES],[49,17,16])
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse({cases[i]['group']for i in roles[a]}&{cases[i]['group']for i in roles[b]})

    def test_cosine_schedule_and_augmentation_use_shared_stateless_field_seed(self):
        spec=importlib.util.spec_from_file_location('matched_driver',Path('scripts/run_matched_three_way.py'))
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        cfg=md.config('configs/breakhis-matched-three-way-v1.json')
        for model in ('cnn','swin'):
            self.assertAlmostEqual(driver.learning_rate(cfg,model,0),cfg['training']['learning_rates'][model])
            self.assertAlmostEqual(driver.learning_rate(cfg,model,39),cfg['training']['learning_rates'][model]*.01)
        try:import cv2
        except ImportError:self.skipTest('OpenCV technical augmentation check requires runtime')
        x=np.random.RandomState(7).uniform(0,255,(2,3,224,224,3)).astype(np.float32)
        first=driver.augment(x,[42,53],3);reverse=driver.augment(x[::-1],[53,42],3)
        np.testing.assert_array_equal(first,reverse[::-1]);self.assertFalse(np.array_equal(first,x))


if __name__=='__main__':unittest.main()

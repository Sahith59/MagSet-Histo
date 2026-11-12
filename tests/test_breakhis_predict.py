import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from magset.breakhis_predict import choose_case, load_head, predict_fields, load_checked_image
from magset.data import file_hash


class BreakhisPredictTests(unittest.TestCase):
    def head(self):
        return {'mean':np.zeros((4,4),dtype=np.float32),'std':np.ones((4,4),dtype=np.float32),
                'coef_0':np.zeros((8,8)),'intercept_0':np.zeros(8),'classes_0':np.arange(8)}

    def test_two_stage_pooling_normalization_mask_and_missing_content(self):
        h=self.head();h['coef_0'][0,0]=1;h['coef_0'][0,4]=1
        h['mean'][0]=1;h['std'][0]=2;h['mean'][1]=4;h['std'][1]=2
        fields={40:np.asarray([[1]*4,[3]*4]),100:np.asarray([[10]*4]),200:np.full((1,4),np.nan)}
        p=predict_fields(fields,(40,100),h)
        expected=np.ones(8);expected[0]=np.exp(2.75);expected/=expected.sum()
        np.testing.assert_allclose(p,expected,atol=1e-15)
        np.testing.assert_array_equal(p,predict_fields(fields,(100,40),h))
        changed=copy.deepcopy(fields);changed[200][:]=1e20
        np.testing.assert_array_equal(p,predict_fields(changed,(40,100),h))
        for views in ((),(40,40),(800,),(400,)):
            with self.assertRaises(ValueError):predict_fields(fields,views,h)

    def test_saved_head_class_order_hash_and_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.npz';h=self.head();np.savez_compressed(p,**h)
            loaded=load_head(p,file_hash(p),4)
            np.testing.assert_array_equal(loaded['coef_0'],h['coef_0'])
            with self.assertRaisesRegex(ValueError,'checksum'):load_head(p,'wrong',4)
            for key,value in [('classes_0',np.arange(8)[::-1]),('std',np.zeros((4,4))),('coef_0',np.full((8,8),np.nan))]:
                bad={**h,key:value};np.savez_compressed(p,**bad)
                with self.assertRaises(ValueError):load_head(p,file_hash(p),4)

    def test_validation_selection_does_not_choose_by_prediction(self):
        r={'case_ids':{'validation':['z','a'],'train':['fit'],'test':['outer']}}
        self.assertEqual(choose_case(r),'a')
        for case in ['fit','outer','unknown']:
            with self.assertRaisesRegex(ValueError,'refuses'):choose_case(r,case)

    def test_original_image_hash_size_and_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'image.png';Image.new('RGB',(7,5)).save(p)
            row={'relative_path':'image.png','sha256':file_hash(p),'width':'7','height':'5'}
            self.assertEqual(load_checked_image(root,row).size,(7,5))
            for bad in ({**row,'relative_path':'../image.png'},{**row,'sha256':'wrong'},{**row,'width':'8'}):
                with self.assertRaises(ValueError):load_checked_image(root,bad)


if __name__=='__main__':unittest.main()

import hashlib
from io import BytesIO
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from scripts import replay_matched_raw as raw
from magset import matched_data


class RawReplayTest(unittest.TestCase):
    def image(self,height=460):
        a=np.zeros((height,700,3),dtype=np.uint8);a[:,:,0]=np.arange(700,dtype=np.uint16)%256;a[:,:,1]=np.arange(height,dtype=np.uint16)[:,None]%256;a[:,:,2]=173
        im=Image.fromarray(a);b=BytesIO();im.save(b,format='PNG');data=b.getvalue()
        row={'sha256':hashlib.sha256(data).hexdigest(),'width':700,'height':height}
        return im,data,row

    def test_golden_both_geometries_against_registered_pixel_operator(self):
        for h in (460,456):
            im,data,row=self.image(h);actual=raw.crop_png(data,row)
            self.assertEqual(actual.shape,(3,224,224,3));self.assertEqual(actual.dtype,np.uint8)
            self.assertTrue(np.array_equal(actual,matched_data.crop_pixels(im)))

    def test_tampered_bytes_and_geometry_are_rejected(self):
        _,b,r=self.image()
        with self.assertRaises(ValueError):raw.crop_png(b+b'bad',r)
        r['height']=456
        with self.assertRaises(ValueError):raw.crop_png(b,r)

    def test_independent_filename_directory_label_checks(self):
        r={'relative_path':'x/malignant/SOB/SOB_M_PC-14-12345-200-001.png','magnification':200,'view':2,'subtype_code':'PC'}
        self.assertEqual(raw.checked_label(r,{'truth5':4}),4)
        with self.assertRaises(ValueError):raw.checked_label(r,{'truth5':1})
        r['relative_path']=r['relative_path'].replace('/malignant/','/benign/')
        with self.assertRaises(ValueError):raw.checked_label(r,{'truth5':4})

    def test_compose_coherent_joint_and_invalid_head_rejection(self):
        b=np.array([[.2,.8]]);m=np.array([[.1,.2,.3,.4]])
        np.testing.assert_allclose(raw.compose(b,m),[[.2,.08,.16,.24,.32]])
        with self.assertRaises(ValueError):raw.compose(b,np.ones((1,4)))

    def test_perfect_case_scores_and_field_order_invariance(self):
        audit={'roles':{'test':list(range(5))},'cases':[{'truth5':i}for i in range(5)],'fields':[]};probs=[]
        for c in range(5):
            for v in range(4):
                for _ in range(2):
                    audit['fields'].append({'index':len(probs),'case_index':c,'view':v});probs.append(np.eye(5)[c])
        probs=np.array(probs);idx=np.arange(len(probs));pred,score=raw.score_fields(audit,idx,probs)
        self.assertTrue(np.array_equal(pred,np.repeat(np.arange(5)[:,None],15,axis=1)))
        self.assertEqual(score['mean_subset_accuracy'],1);self.assertEqual(score['mean_subset_macro_f1'],1)
        perm=np.random.RandomState(3).permutation(idx);pred2,_=raw.score_fields(audit,idx[perm],probs[perm]);np.testing.assert_array_equal(pred,pred2)

    def test_raw_cache_disagreement_cannot_be_silently_accepted(self):
        _,data,row=self.image()
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'x/benign/SOB_B_A-14-12345-40-001.png';p.parent.mkdir(parents=True);p.write_bytes(data)
            row.update(relative_path=str(p.relative_to(tmp)),subtype_code='A',magnification=40,view=0,index=0,case_index=0)
            audit={'raw_root':tmp,'fields':[row],'cases':[{'truth5':0}]}
            cached=np.zeros((1,3,224,224,3),dtype=np.uint8)
            with self.assertRaises(ValueError):list(raw.raw_batches(audit,np.array([0]),cached,16))


if __name__=='__main__':unittest.main()

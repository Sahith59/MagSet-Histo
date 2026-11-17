import unittest
import numpy as np
from scripts import diagnose_router_groups as d

class RouterGroupTests(unittest.TestCase):
    def test_entire_related_group_is_excluded_not_only_query(self):
        c=[{'group':'a'},{'group':'a'},{'group':'b'},{'group':'c'}]
        self.assertEqual(d.group_support(c,[0,1,2,3],0),[2,3])
        self.assertEqual(d.group_support(c,[0,1,2,3],1),[2,3])
        with self.assertRaises(ValueError):d.group_support(c,[0,1,2],3)
    def test_perfect_binary_scores_with_probability_bins(self):
        y=np.array([0,0,1,1]);p=np.repeat(y[:,None],15,axis=1).astype(float)
        s=d.summarize(y,p);self.assertEqual(s['mean_accuracy'],1);self.assertEqual(s['mean_balanced_accuracy'],1);self.assertEqual(s['brier'],0);self.assertEqual(s['mean_roc_auc'],1)
        self.assertEqual(sum(b['events']for b in s['calibration_bins']),60)
    def test_pool_preserves_equal_zoom_not_field_count(self):
        audit={'fields':[{'index':i,'case_index':0,'view':v}for i,v in enumerate((0,0,0,1,2,3))]}
        q=np.array([[1,0,0,0,0]]*3+[[0,1,0,0,0]]*3,float)
        result=d.pool(audit,[0],np.arange(6),q)
        self.assertEqual(result[0,-1],.75)

if __name__=='__main__':unittest.main()

import copy
import unittest
import numpy as np
from magset.capacity_report import assert_matching_context


class CapacityContextTests(unittest.TestCase):
    def test_identical_aggregate_metrics_cannot_hide_different_inputs(self):
        report={'encoder':'phikon-v2','trial':1,'seed':812,'manifest_sha256':'m',
                'protocol_sha256':'p','class_names':['a','b'],'feature_provenance':[{'hash':'same'}],
                'case_ids':{'train':['fit'],'validation':['val'],'test':['outer']}}
        mean=np.zeros((4,2));std=np.ones((4,2))
        assert_matching_context(report,copy.deepcopy(report),mean,std,mean.copy(),std.copy())
        changed=copy.deepcopy(report);changed['feature_provenance'][0]['hash']='other'
        with self.assertRaisesRegex(ValueError,'inputs differ'):
            assert_matching_context(report,changed,mean,std,mean,std)
        changed=copy.deepcopy(report);changed['case_ids']['validation']=['different-case']
        with self.assertRaisesRegex(ValueError,'case assignments'):
            assert_matching_context(report,changed,mean,std,mean,std)
        with self.assertRaisesRegex(ValueError,'normalization arrays'):
            assert_matching_context(report,report,mean,std,mean,std*2)


if __name__=='__main__':
    unittest.main()

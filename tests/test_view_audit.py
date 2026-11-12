import copy
import unittest
import numpy as np
from magset.native_pilot import SUBSETS
from magset.view_audit import describe_arm, EDGES


class ViewAuditTests(unittest.TestCase):
    def fixture(self):
        records=[]
        for patient,truth in [('a',0),('b',1)]:
            for subset in SUBSETS:
                p=np.zeros(16);p[:2]=[.1,.9] if 3 in subset else [.8,.2]
                records.append({'patient_id':patient,'truth':truth,'subset':list(subset),'probabilities':p.tolist()})
        return records

    def test_known_help_harm_and_confidence_with_repeated_patients(self):
        self.assertEqual(len(EDGES),28)
        r=describe_arm(self.fixture());t=r['totals']
        self.assertEqual(r['patients'],2);self.assertEqual(r['patient_edge_comparisons'],56)
        self.assertEqual(t['helped'],7);self.assertEqual(t['harmed'],7)
        self.assertEqual(t['both_correct'],21);self.assertEqual(t['both_wrong'],21)
        self.assertEqual(t['changed_class'],14);self.assertEqual(t['harmed_with_increased_max_probability'],7)
        self.assertEqual(r['distinct_patients_harmed_at_least_once'],1)
        self.assertEqual(r['distinct_patients_helped_at_least_once'],1)
        self.assertTrue(all(abs(e['mean_true_class_probability_change'])<1e-14 for e in r['edges']))

    def test_duplicates_incomplete_coverage_changing_truth_and_nan_rejected(self):
        records=self.fixture()
        with self.assertRaises(ValueError):describe_arm([])
        with self.assertRaises(ValueError):describe_arm(records+[records[0]])
        with self.assertRaises(ValueError):describe_arm(records[:-1])
        bad=copy.deepcopy(records);bad[1]['truth']=2
        with self.assertRaisesRegex(ValueError,'truth'):describe_arm(bad)
        bad=copy.deepcopy(records);bad[1]['probabilities'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'probabilities'):describe_arm(bad)


if __name__=='__main__':unittest.main()

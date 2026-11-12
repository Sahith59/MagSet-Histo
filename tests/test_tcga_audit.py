import csv
from pathlib import Path
import tempfile
import unittest
from magset.tcga_audit import audit


def fixture(root, shared_patient=False):
    root=Path(root);metadata=root/'metadata/tcga_ms';metadata.mkdir(parents=True)
    labels=[]
    for slide,patient,role in [('a','patient-a','train'),('b','patient-a' if shared_patient else 'patient-b','test')]:
        labels.append({'slide_uuid':slide,'case_uuid':patient,'label_id':'10' if slide=='a' else '20',
                       **{'fold_%d'%i:role for i in range(5)}})
    with (metadata/'labels.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(labels[0]));writer.writeheader();writer.writerows(labels)
    with (root/'members.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['relative_path','bytes','suffix']);writer.writeheader()
        for name,size in [('a_1_2.png',5000),('b_3_4.png',5000),('._a_1_2.png',233)]:
            writer.writerow({'relative_path':'tcga_ms/mpp_0.5/'+name,'bytes':size,'suffix':'.png'})


class TcgaAuditTests(unittest.TestCase):
    def test_resource_sidecars_do_not_become_unlabeled_patients(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);report=audit(tmp)
            self.assertEqual(report['png_suffixed_entries'],3)
            self.assertEqual(report['excluded_sidecar_named_entries'],1)
            self.assertEqual(report['image_files'],2)
            self.assertEqual(report['image_slide_ids'],2)
            self.assertEqual(report['unlabeled_image_files'],0)
            self.assertEqual(report['unlabeled_slide_ids'],[])

    def test_shared_patient_crossing_roles_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp,shared_patient=True)
            with self.assertRaisesRegex(ValueError,'patient-role overlap'):
                audit(tmp)


if __name__=='__main__':
    unittest.main()

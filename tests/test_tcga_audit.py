import csv
import json
from pathlib import Path
import tempfile
import unittest
from magset.tcga_audit import audit, verify_decoding
from magset.data import file_hash


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
    def test_decode_evidence_requires_exact_counts_provenance_and_no_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);report=audit(tmp);root=Path(tmp)
            decoded=root/'decoded';decoded.mkdir()
            r={'resolution_mpp':'0.5','images':2,'modes':{'RGB':2},'dimensions':{'(224, 224)':2},
               'errors':[],'inventory_sha256':file_hash(root/'members.csv'),'archive_sha256':'expected','code_sha':'fixture'}
            path=decoded/'mpp-0.5.json';path.write_text(json.dumps(r))
            self.assertEqual(verify_decoding(report,root,decoded,'expected')['images'],2)
            r['errors']=[{'error':'failed image'}];path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):verify_decoding(report,root,decoded,'expected')
            r['errors']=[];r['images']=3;path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):verify_decoding(report,root,decoded,'expected')

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

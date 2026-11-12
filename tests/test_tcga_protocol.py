import csv
from pathlib import Path
import tempfile
import unittest
from magset.tcga_protocol import make_protocol, RESOLUTIONS, SOURCE_LABELS
from magset.data import file_hash


def fixture(root):
    root=Path(root);metadata=root/'metadata/tcga_ms';metadata.mkdir(parents=True)
    labels=[];fields=[]
    for label in range(16):
        for i in range(8):
            slide='slide%d-%d'%(label,i)
            labels.append({'slide_uuid':slide,'case_uuid':'patient%d-%d'%(label,i),'label_id':SOURCE_LABELS[label],
                           **{'fold_%d'%k:'train' for k in range(5)}})
            for resolution in RESOLUTIONS:
                if label<6 and i==7 and resolution=='0.25':continue
                fields.append({'relative_path':'tcga_ms/mpp_%s/%s_1_2.png'%(resolution,slide),'suffix':'.png','bytes':42})
    for path,rows in [(metadata/'labels.csv',labels),(root/'members.csv',fields)]:
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'labels_sha256':file_hash(metadata/'labels.csv'),'inventory_sha256':file_hash(root/'members.csv')}


class TcgaProtocolTests(unittest.TestCase):
    def test_patient_quarantine_is_deterministic_disjoint_class_covered_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory')
            a=make_protocol(root/'inventory',root/'a',source);b=make_protocol(root/'inventory',root/'b',source)
            self.assertEqual(a['roles'],b['roles']);self.assertEqual(a['manifest_sha256'],b['manifest_sha256'])
            self.assertEqual(a['role_counts'],{'fit':80,'sealed_test':32,'development':16})
            self.assertEqual(len(a['incomplete_patients']),6)
            self.assertTrue(all(a['roles'][p]=='fit' for p in a['incomplete_patients']))
            for counts in a['class_patient_role_counts'].values():
                self.assertEqual(counts,{'fit':5,'sealed_test':2,'development':1})
            with self.assertRaises(ValueError):make_protocol(root/'inventory',root/'a',source)

    def test_changed_hash_class_counts_and_missing_view_contract_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory')
            with self.assertRaisesRegex(ValueError,'pinned audited'):
                make_protocol(root/'inventory',root/'bad',{**source,'labels_sha256':'wrong'})
            labels=root/'inventory/metadata/tcga_ms/labels.csv'
            with labels.open() as f:rows=list(csv.DictReader(f))
            rows[0]['label_id']=SOURCE_LABELS[1]
            with labels.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'label/patient count'):
                make_protocol(root/'inventory',root/'bad2',{**source,'labels_sha256':file_hash(labels)})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');path=root/'inventory/members.csv'
            with path.open() as f:rows=list(csv.DictReader(f))
            rows=[r for r in rows if r['relative_path']!='tcga_ms/mpp_0.5/slide0-0_1_2.png']
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'missing-resolution contract|role counts'):
                make_protocol(root/'inventory',root/'bad',{**source,'inventory_sha256':file_hash(path)})


if __name__=='__main__':unittest.main()

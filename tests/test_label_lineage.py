import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_tcga_label_lineage import target_columns,audit


class LabelLineageTests(unittest.TestCase):
    def test_confirmed_target_is_not_confused_with_original_label_column(self):
        source='def loader(label_df):\n    label_map = dict(zip(label_df["slide_uuid"], label_df["label_id"]))\n'
        self.assertEqual(target_columns(source),('slide_uuid','label_id'))
        for invalid in (source.replace('label_id','label'),source+source,'label_map = {}'):
            with self.assertRaises(ValueError):target_columns(invalid)

    def test_pinned_csv_and_loader_are_bound_before_metadata_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=root/'labels.csv';loader=root/'loader.py';provenance=root/'provenance.json';config=root/'source.json'
            rows=[]
            for label in range(16):
                for case in range(8):
                    rows.append({'case_uuid':'case%d-%d'%(label,case),'slide_uuid':'slide%d-%d'%(label,case),
                        'label_id':str(label),'label':str(label+20),'Histology/Behavior Description':'example'+str(label),
                        'Histology/Behavior':'code'+str(label),'TCGA_tissue':'example tissue',
                        **{'fold_%d'%i:('train','dev','test')[case%3] for i in range(5)}})
            with labels.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            loader.write_text('label_map = dict(zip(label_df["slide_uuid"],label_df["label_id"]))')
            h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            config.write_text(json.dumps({'labels_sha256':h(labels)}))
            provenance.write_text(json.dumps({'revision':'synthetic','files':[{'path':'dataloaders.py','sha256':h(loader),'url':'synthetic'}]}))
            report=audit(labels,loader,provenance,config)
            self.assertEqual(report['source_classes'],16);self.assertTrue(report['numeric_target_columns_are_bijective'])
            self.assertEqual(report['classifier_fits_performed'],0)
            original=loader.read_text();loader.write_text(original+'\n# changed')
            with self.assertRaisesRegex(ValueError,'loader'):audit(labels,loader,provenance,config)
            loader.write_text(original);labels.write_text(labels.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'label file'):audit(labels,loader,provenance,config)


if __name__=='__main__':unittest.main()

import csv
import json
from pathlib import Path
import tempfile
import unittest
from magset.data import file_hash
from magset.tcga_protocol import make_protocol
from test_tcga_protocol import fixture
try:
    from magset.tcga_features import native_rows
except ImportError:
    native_rows=None


@unittest.skipIf(native_rows is None,'Pinned Torch/Pillow needed for native feature adapter')
class NativeFeatureTests(unittest.TestCase):
    def test_role_path_patient_and_registered_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'inventory');out=root/'registered'
            make_protocol(root/'inventory',out,source)
            manifest=out/'manifest.csv';protocol=out/'protocol.json';ph=file_hash(protocol)
            rows,p=native_rows(manifest,protocol,ph)
            self.assertEqual(len(rows),p['fields'])
            with self.assertRaisesRegex(ValueError,'registered job hash'):
                native_rows(manifest,protocol,'other')
            rows[0]['resolution_mpp']='2.0' if rows[0]['resolution_mpp']!='2.0' else '0.5'
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            p['manifest_sha256']=file_hash(manifest);protocol.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'path/patient'):
                native_rows(manifest,protocol,file_hash(protocol))


if __name__=='__main__':unittest.main()

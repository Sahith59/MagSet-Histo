import csv
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from PIL import Image
from magset.archive_audit import inventory
from magset.data import file_hash
from magset.prepare_tcga import extract, decode


class PrepareTcgaTests(unittest.TestCase):
    def test_safe_extract_decode_excludes_sidecar_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'fixture.tar.gz';pixels=io.BytesIO()
            Image.new('RGB',(16,12),(80,100,120)).save(pixels,format='PNG')
            with tarfile.open(archive,'w:gz') as bundle:
                for name,value in [('slide_1_2.png',pixels.getvalue()),('._slide_1_2.png',b'not an image')]:
                    item=tarfile.TarInfo('tcga_ms/mpp_0.25/'+name);item.size=len(value)
                    bundle.addfile(item,io.BytesIO(value))
            source={'filename':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive)}
            inventory(archive,root/'inventory-v1');extract(root,source)
            report=decode(root,0,'synthetic')
            self.assertEqual(report['images'],1);self.assertEqual(report['errors'],[])
            self.assertEqual(report['modes'],{'RGB':1});self.assertEqual(report['dimensions'],{'(16, 12)':1})
            (root/'raw/tcga_ms/mpp_0.25/slide_1_2.png').write_bytes(b'invalid')
            self.assertEqual(len(decode(root,0,'synthetic')['errors']),1)


if __name__=='__main__':
    unittest.main()

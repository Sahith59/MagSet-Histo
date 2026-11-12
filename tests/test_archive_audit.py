import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from magset.archive_audit import inventory


class ArchiveAuditTests(unittest.TestCase):
    def test_inventories_images_and_copies_only_text_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'fixture.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                for name,value in [('data/cases.csv',b'case,label\na,0\n'),('data/patch.png',b'not a decoded image')]:
                    item = tarfile.TarInfo(name); item.size = len(value); stream.addfile(item,io.BytesIO(value))
            report = inventory(archive,root/'audit')
            self.assertEqual(report['suffix_counts'],{'.csv':1,'.png':1})
            self.assertTrue((root/'audit/metadata/data/cases.csv').exists())
            self.assertFalse((root/'audit/metadata/data/patch.png').exists())
            self.assertEqual(len((root/'audit/members.csv').read_text().splitlines()),3)

    def test_unsafe_member_rejected_before_any_metadata_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'fixture.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                item = tarfile.TarInfo('safe.csv'); item.size = 1; stream.addfile(item,io.BytesIO(b'x'))
                item = tarfile.TarInfo('../escape.csv'); item.size = 1; stream.addfile(item,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                inventory(archive,root/'audit')
            self.assertFalse((root/'audit').exists())


if __name__ == '__main__':
    unittest.main()

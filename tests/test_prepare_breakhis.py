import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_breakhis.py'
spec = importlib.util.spec_from_file_location('prepare_breakhis', SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PrepareBreakHisTests(unittest.TestCase):
    def test_labels_raw_id_and_conservative_group_are_distinct(self):
        row = prep.parse_filename('SOB_B_F-14-21998AB-400-007.png')
        self.assertEqual(row['case_id_raw'], '14-21998AB')
        self.assertEqual(row['patient_group_conservative'], '14-21998')
        self.assertEqual(row['magnification'], 400)
        self.assertEqual(row['subtype'], 'fibroadenoma')
        self.assertIn('UNVERIFIED', row['group_rule'])

    def test_invalid_class_and_filename_fail(self):
        for name in ['SOB_B_DC-14-1234-40-001.png', 'SOB_M_DC-14-1234-20-001.png', 'other.png']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                prep.parse_filename(name)

    def test_index_exposes_merges_and_incomplete_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / 'images'
            images.mkdir()
            # Header fixtures, explicitly not complete decodable PNG images.
            header = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 700, 460) + bytes([8, 2, 0, 0, 0])
            for name in ['SOB_B_F-14-21998AB-40-001.png', 'SOB_B_F-14-21998CD-100-001.png']:
                (images / name).write_bytes(header)
            audit = prep.index_images(images, root / 'out')
            self.assertEqual(audit['images'], 2)
            self.assertEqual(audit['raw_case_count'], 2)
            self.assertEqual(audit['conservative_group_count'], 1)
            self.assertFalse(audit['standard_counts_match'])
            self.assertIn('14-21998', audit['merged_raw_ids_under_conservative_rule'])
            self.assertEqual(len(audit['identical_file_groups']), 1)
            self.assertEqual(json.loads((root / 'out/audit.json').read_text())['images'], 2)

    def test_archive_traversal_and_symlinks_are_rejected(self):
        for mode in ['traversal', 'symlink']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / 'bad.tar.gz'
                with tarfile.open(archive, 'w:gz') as bundle:
                    entry = tarfile.TarInfo('../escape' if mode == 'traversal' else 'link')
                    if mode == 'symlink':
                        entry.type = tarfile.SYMTYPE
                        entry.linkname = '../escape'
                        bundle.addfile(entry)
                    else:
                        entry.size = 1
                        bundle.addfile(entry, io.BytesIO(b'x'))
                with self.assertRaises(ValueError):
                    prep.extract_archive(archive, root / 'raw', prep.sha256(archive))
                self.assertFalse((root / 'escape').exists())

    def test_extraction_can_be_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'good.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                entry = tarfile.TarInfo('sample/a.txt')
                entry.size = 2
                bundle.addfile(entry, io.BytesIO(b'ok'))
            digest = prep.sha256(archive)
            prep.extract_archive(archive, root / 'raw', digest)
            prep.extract_archive(archive, root / 'raw', digest)
            self.assertEqual((root / 'raw/sample/a.txt').read_bytes(), b'ok')
            with self.assertRaises(ValueError):
                prep.extract_archive(archive, root / 'raw', 'different')


if __name__ == '__main__':
    unittest.main()

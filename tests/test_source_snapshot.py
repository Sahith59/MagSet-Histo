import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from scripts.create_source_snapshot import install


class SourceSnapshotTests(unittest.TestCase):
    def test_commit_binding_read_only_snapshot_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sha='1'*40;archive=root/'source.tar';target=root/sha
            with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT,pax_headers={'comment':sha}) as bundle:
                item=tarfile.TarInfo('module.py');item.size=4;bundle.addfile(item,io.BytesIO(b'pass'))
            with self.assertRaisesRegex(ValueError,'embedded commit'):
                install(archive,root/('2'*40),'2'*40)
            try:
                install(archive,target,sha);install(archive,target,sha)
                self.assertEqual((target/'module.py').read_text(),'pass')
                self.assertEqual((target/'.source-sha').read_text().strip(),sha)
                self.assertEqual((target/'module.py').stat().st_mode & 0o222,0)
            finally:
                if target.exists():
                    target.chmod(0o755)
                    for path in target.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o644)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sha='3'*40;archive=root/'bad.tar'
            with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT,pax_headers={'comment':sha}) as bundle:
                item=tarfile.TarInfo('../escape');item.size=1;bundle.addfile(item,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unsafe source'):
                install(archive,root/sha,sha)
            self.assertFalse((root/'escape').exists())


if __name__=='__main__':unittest.main()

"""Fully decode one magnification shard and record errors without skipping images."""
import argparse
from collections import Counter
import json
from pathlib import Path
from PIL import Image
from .data import MAGNIFICATIONS, file_hash, read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--images-root', type=Path, required=True)
    p.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [r for r in read_manifest(a.manifest) if int(r['magnification']) == a.magnification]
    errors, modes = [], Counter()
    for row in rows:
        path = a.images_root / row['relative_path']
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                if im.size != (int(row['width']), int(row['height'])):
                    raise ValueError('Decoded dimensions differ from manifest')
                modes[im.mode] += 1
        except Exception as exc:
            errors.append({'path': row['relative_path'], 'error': repr(exc)})
    report = {'manifest_sha256': file_hash(a.manifest), 'magnification': a.magnification,
              'images': len(rows), 'modes': dict(modes), 'errors': errors}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'errors'}, indent=2))
    if errors:
        raise SystemExit('Image decoding failed for %d images; inspect the audit' % len(errors))


if __name__ == '__main__':
    main()

"""Safely extract the pinned TCGA-MS archive and fully decode resolution shards."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path, PurePosixPath
from PIL import Image
from .data import file_hash
from scripts.prepare_breakhis import extract_archive, atomic_json

RESOLUTIONS=('0.25','0.375','0.5','0.75','1.0','1.5','2.0')


def extract(root, source):
    root=Path(root);archive=root/source['filename']
    if archive.stat().st_size != source['bytes'] or file_hash(archive) != source['sha256']:
        raise ValueError('Archive differs from the pinned release')
    extract_archive(archive,root/'raw',source['sha256'])


def decode(root, rank, code_sha):
    root=Path(root)
    if rank not in range(len(RESOLUTIONS)):
        raise ValueError('Unknown resolution shard')
    inventory=root/'inventory-v1/members.csv';scale='mpp_'+RESOLUTIONS[rank]
    with inventory.open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['suffix']=='.png'
              and PurePosixPath(r['relative_path']).parent.name==scale
              and not PurePosixPath(r['relative_path']).name.startswith('._')]
    if not rows:
        raise ValueError('No candidate images in shard')
    marker=root/'raw/.archive_sha256'
    if not marker.exists():
        raise ValueError('Safe extraction must complete before decoding')
    errors=[];modes=Counter();dimensions=Counter()
    for row in rows:
        relative=PurePosixPath(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['relative_path']:
            raise ValueError('Unsafe inventory image path')
        path=root/'raw'/Path(*relative.parts)
        try:
            if path.stat().st_size != int(row['bytes']):
                raise ValueError('Extracted size differs from archive inventory')
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load();modes[im.mode]+=1;dimensions[str(im.size)]+=1
        except Exception as exc:
            errors.append({'relative_path':row['relative_path'],'error':repr(exc)})
    return {'resolution_mpp':RESOLUTIONS[rank],'images':len(rows),'modes':dict(modes),
            'dimensions':dict(dimensions),'errors':errors,'code_sha':code_sha,
            'inventory_sha256':file_hash(inventory),'archive_sha256':marker.read_text().strip()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['extract','decode'],required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=range(7));p.add_argument('--code-sha',required=True)
    a=p.parse_args();source=json.loads(a.source.read_text())
    if a.stage=='extract':
        print('Verifying and safely extracting the pinned archive',flush=True)
        extract(a.data_root,source)
        print('Safe extraction complete',flush=True)
    else:
        if a.rank is None:
            raise ValueError('Decoding requires a rank')
        output=a.data_root/('decode-v1/mpp-'+RESOLUTIONS[a.rank]+'.json')
        if output.exists():
            raise ValueError('Decode report exists; inspect it before retrying')
        report=decode(a.data_root,a.rank,a.code_sha)
        if report['archive_sha256'] != source['sha256']:
            raise ValueError('Extraction marker differs from pinned release')
        atomic_json(output,report)
        print(json.dumps({k:v for k,v in report.items() if k!='errors'},indent=2),flush=True)
        if report['errors']:
            raise SystemExit('Image decoding failed; inspect all recorded errors')


if __name__=='__main__':
    main()

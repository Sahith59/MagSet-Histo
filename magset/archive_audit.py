"""Download a pinned public archive and inventory it without executing/extracting its contents."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from .data import file_hash


def inventory(archive, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Inventory output already exists; preserve it and use a new version')
    entries = []; counts = Counter(); total = 0; names = set(); metadata = []; metadata_bytes = 0
    with tarfile.open(archive,'r|gz') as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe archive member: '+member.name)
            if path.as_posix() in names:
                raise ValueError('Duplicate archive member: '+member.name)
            names.add(path.as_posix()); total += member.size
            if total > 40*1024**3 or len(names) > 150000:
                raise ValueError('Archive exceeds the prespecified inventory budget')
            if member.isdir():
                continue
            suffix = path.suffix.lower(); counts[suffix] += 1
            entries.append({'relative_path':member.name,'bytes':member.size,'suffix':suffix})
            if suffix in ('.csv','.json','.txt','.tsv','.yaml','.yml','.md') and member.size <= 5*1024**2:
                metadata_bytes += member.size
                if metadata_bytes > 64*1024**2:
                    raise ValueError('Text metadata exceeds the bounded in-memory inventory budget')
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError('Cannot read metadata member')
                with stream:
                    value = stream.read()
                metadata.append((path,value))
    # Only bounded text metadata is copied, after every member passes the audit.
    output.mkdir(parents=True)
    for path,value in metadata:
        target = output/'metadata'/Path(*path.parts)
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(value)
    with (output/'members.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['relative_path','bytes','suffix'])
        writer.writeheader(); writer.writerows(entries)
    report = {'files':len(entries),'expanded_bytes':total,'suffix_counts':dict(counts),
              'metadata_files':[p.as_posix() for p,_ in metadata],
              'status':'archive structure inventoried; image decoding, labels, groups, and split audit remain pending'}
    (output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--code-sha',required=True)
    a = p.parse_args(); source = json.loads(a.source.read_text())
    a.data_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.data_root).free < 15*1024**3:
        raise RuntimeError('Need at least 15 GiB free')
    archive = a.data_root/source['filename']
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix+'.part')
        subprocess.run(['curl','--fail','--location','--retry','2','--connect-timeout','30',
                        '--continue-at','-','--output',str(partial),'--url',source['url']],check=True)
        if partial.stat().st_size != source['bytes'] or file_hash(partial) != source['sha256']:
            raise ValueError('Download differs from pinned provider size/checksum')
        os.replace(partial,archive)
    if archive.stat().st_size != source['bytes'] or file_hash(archive) != source['sha256']:
        raise ValueError('Archive differs from pinned provider size/checksum')
    report = inventory(archive,a.data_root/'inventory-v1')
    (a.data_root/'archive-provenance.json').write_text(json.dumps({**source,'verified':True,
        'code_sha':a.code_sha,'source_config_sha256':file_hash(a.source),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Install an immutable tracked-source archive under a commit-specific directory."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


def install(archive,destination,sha):
    archive=Path(archive);destination=Path(destination)
    if not re.fullmatch('[0-9a-f]{40}',sha) or destination.name!=sha:raise ValueError('Use a full commit-specific snapshot path')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        if bundle.pax_headers.get('comment')!=sha:raise ValueError('Git archive embedded commit differs')
    if destination.exists():
        if (destination/'.source-sha').read_text().strip()!=sha or (destination/'.archive-sha256').read_text().strip()!=digest:
            raise ValueError('Existing immutable source differs')
        return
    stage=destination.with_name(sha+'.installing')
    if stage.exists():raise ValueError('Partial source installation exists; inspect it')
    stage.mkdir(parents=True)
    with tarfile.open(archive) as bundle:
        for item in bundle:
            p=PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or '\\' in str(p) or not(item.isdir() or item.isfile()):
                raise ValueError('Unsafe source archive member')
            target=stage.joinpath(*p.parts)
            if item.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(item) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
    (stage/'.source-sha').write_text(sha+'\n');(stage/'.archive-sha256').write_text(digest+'\n')
    os.replace(stage,destination)
    for path in destination.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--sha',required=True);a=p.parse_args();install(a.archive,a.destination,a.sha)

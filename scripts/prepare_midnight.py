#!/usr/bin/env python3
"""CPU-only anonymous, resumable download of the registered Midnight artifacts."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.midnight_features import (CARD_SHA256, MODEL_CONFIG_SHA256, REPOSITORY,
                                     REVISION, WEIGHTS_SHA256, WEIGHTS_SIZE, read_config)


def verify(path, size, sha256):
    return path.is_file() and path.stat().st_size == size and file_hash(path) == sha256


def download(url, target, size, sha256, opener=urllib.request.urlopen, attempts=3):
    """Resume only validated HTTP ranges; publish only a complete hash-matching file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not verify(target, size, sha256):
            raise ValueError('Existing published artifact differs: ' + target.name)
        return {'file': target.name, 'bytes': size, 'sha256': sha256, 'already_present': True}
    part = target.with_name(target.name + '.part')
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise ValueError('Partial artifact is larger than expected: ' + target.name)
        if offset == size:
            break
        headers = {'User-Agent': 'MagSet-Histo-pinned-artifact-preparation',
                   'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = 'bytes=%d-' % offset
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
                status = response.status
                if status == 206:
                    value = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
                    if (not match or int(match[1]) != offset or int(match[3]) != size
                            or int(match[2]) < offset or int(match[2]) >= size):
                        raise ValueError('Invalid partial-response range for ' + target.name)
                    response_size = int(match[2]) - offset + 1
                elif status == 200:
                    # Some servers ignore Range: restart the partial, never append a full body.
                    offset = 0
                    response_size = size
                else:
                    raise ValueError('Unexpected HTTP status for ' + target.name)
                content_length = response.headers.get('Content-Length')
                if content_length is not None and int(content_length) != response_size:
                    raise ValueError('HTTP response length differs for ' + target.name)
                received = 0
                with part.open('ab' if offset else 'wb') as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        if received + len(block) > response_size:
                            raise ValueError('HTTP response exceeds registered size: ' + target.name)
                        stream.write(block)
                        received += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != response_size:
                    raise OSError('Truncated artifact response: ' + target.name)
            if part.stat().st_size == size:
                break
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise RuntimeError('Bounded download attempts exhausted: ' + target.name) from None
            time.sleep(2)
    if not verify(part, size, sha256):
        raise ValueError('Complete artifact length/hash differs: ' + target.name)
    os.replace(part, target)
    return {'file': target.name, 'bytes': size, 'sha256': sha256, 'already_present': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--code-sha', required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Prepare large artifacts inside a scheduler CPU allocation')
    config = read_config(args.config)
    if not re.fullmatch('[0-9a-f]{40}', args.code_sha):
        raise ValueError('Full immutable source SHA required')
    artifacts = [('config.json', 879, MODEL_CONFIG_SHA256),
                 ('README.md', 8080, CARD_SHA256),
                 ('model.safetensors', WEIGHTS_SIZE, WEIGHTS_SHA256)]
    start = time.monotonic()
    records = []
    for name, size, digest in artifacts:
        print(json.dumps({'status': 'verifying/downloading pinned artifact', 'file': name,
                          'expected_bytes': size}), flush=True)
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (REPOSITORY, REVISION, name)
        records.append(download(url, args.cache_root / name, size, digest))
    record = {'status': 'accepted pinned artifacts; no model construction or forward',
              'repository': REPOSITORY, 'revision': REVISION, 'artifacts': records,
              'config_sha256': file_hash(args.config), 'code_sha': args.code_sha,
              'script_sha256': file_hash(__file__), 'elapsed_seconds': time.monotonic() - start,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'license': config['encoder'].get('license')}
    receipt = args.cache_root / 'preparation.json'
    temporary = receipt.with_suffix('.json.part')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary, receipt)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()

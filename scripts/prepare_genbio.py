#!/usr/bin/env python3
"""Download hash-pinned public GenBio artifacts without loading or executing them."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from scripts.prepare_midnight import download


def artifacts(config):
    encoder = config['encoder']
    if encoder['repository'] != 'genbio-ai/genbio-pathfm' or encoder['revision'] != '6eab2a4ea6fbaee16e5f193187952042ebe5d3ec':
        raise ValueError('Unregistered GenBio repository/revision')
    result = {}
    for prefix in ('weights', 'config', 'model_card', 'license'):
        name, size, digest = (encoder[prefix + suffix] for suffix in ('_file', '_size_bytes', '_sha256'))
        result[name] = {'size_bytes': size, 'sha256': digest}
    result.update(encoder['source_files'])
    if set(result) != {'model.safetensors', 'config.json', 'README.md', 'LICENSE.txt', 'configuration_genbio_pathfm.py', 'modeling_genbio_pathfm.py'}:
        raise ValueError('Unexpected artifact filename matrix')
    for name, record in result.items():
        if (Path(name).name != name or type(record['size_bytes']) is not int or record['size_bytes'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Invalid artifact name/length/hash')
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run artifact preparation in a scheduler CPU job')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if file_hash(args.config) != file_hash(root/'configs/genbio-pathfm-v1.json'):
        raise ValueError('Preparation config differs from frozen source')
    config = json.loads(args.config.read_text()); items = artifacts(config)
    encoder = config['encoder']; base = 'https://huggingface.co/%s/resolve/%s/' % (encoder['repository'], encoder['revision'])
    # Small source/license receipts precede the checkpoint. No downloaded Python is imported.
    ordered = sorted(items, key=lambda n: n == encoder['weights_file'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ordered:
        pin = items[name]
        record = download(base + quote(name), args.output/name, pin['size_bytes'], pin['sha256'])
        print(json.dumps(record), flush=True)
    receipt = {'status': 'all six pinned GenBio artifacts verified without model execution',
               'config_sha256': file_hash(args.config), 'encoder': encoder,
               'artifacts': {name: {'sha256': file_hash(args.output/name), 'bytes': (args.output/name).stat().st_size} for name in ordered},
               'model_executions': 0, 'encoder_forwards': 0}
    (args.output/'preparation-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': receipt['status']}), flush=True)


if __name__ == '__main__':
    main()

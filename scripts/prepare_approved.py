"""Download approved, revision-pinned encoder artifacts inside a CPU allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash


def prepare(config_path, output):
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text())
    canonical = root / 'configs' / (config['name'] + '.json')
    if (config['name'] not in ('uni2-h-v1', 'virchow2-v1', 'h-optimus-0-v1')
            or file_hash(config_path) != file_hash(canonical)):
        raise ValueError('Unregistered encoder preparation config')
    encoder = config['encoder']
    if not re.fullmatch('[0-9a-f]{40}', encoder['revision']):
        raise ValueError('Full model revision required')
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    names = sorted(config['artifacts'], key=lambda name: name == encoder['weights_file'])
    for name in names:
        pin = config['artifacts'][name]
        if Path(name).name != name:
            raise ValueError('Artifact must be a basename')
        print(json.dumps({'status': 'verifying approved artifact', 'model': config['name'],
                          'file': name, 'expected_bytes': pin['size_bytes']}), flush=True)
        path = Path(hf_hub_download(encoder['repository'], name, revision=encoder['revision'],
                                   local_dir=output, cache_dir=output.parent / 'hub-downloads'))
        if path.stat().st_size != pin['size_bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Pinned artifact size or SHA256 differs: ' + name)
        records[name] = dict(pin)
    receipt = {'status': 'approved pinned artifacts verified without model execution',
               'config_sha256': file_hash(config_path), 'encoder': encoder,
               'artifacts': records, 'encoder_forwards': 0, 'model_executions': 0,
               'elapsed_seconds': time.monotonic() - started,
               'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    target = output / 'preparation-receipt.json'
    temporary = target.with_suffix('.json.part')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(temporary, target)
    print(json.dumps({'status': receipt['status'], 'model': config['name'],
                      'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a scheduler CPU allocation for large downloads')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)

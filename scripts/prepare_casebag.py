"""Prepare one public, pinned EfficientNet checkpoint; no data or model fitting."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from magset.data import file_hash

REPO = 'timm/tf_efficientnetv2_b0.in1k'
REVISION = '30b1df5883752d711e047538d9b01db797782818'
WEIGHT_SHA = 'e134f5fc07f28cff44a0601a13d79d2eff6304fa7ed7d64332984e227c2785b3'
WEIGHT_BYTES = 28840352
WHEEL_SHA = '5a3dc460c24e322ecc7fd1f3e3eb112423ddee320cb059cc1956fbc9731748ef'
CORE = ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'Pillow',
        'safetensors', 'huggingface-hub', 'PyYAML')


def prepare(root, source_sha):
    root = Path(root); source = Path(__file__).resolve().parents[1]
    if (source/'.source-sha').read_text().strip() != source_sha:
        raise ValueError('Use the immutable preparation source')
    before = {name: importlib.metadata.version(name) for name in CORE}
    destination = root/'cache/casebag-effnetv2b0'; destination.mkdir(parents=True, exist_ok=True)
    try:
        installed = importlib.metadata.version('timm')
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed not in (None, '1.0.15'):
        raise ValueError('Existing timm version differs; do not overwrite implicitly')
    wheel_dir = root/'cache/package-wheels'; wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = wheel_dir/'timm-1.0.15-py3-none-any.whl'
    requirements = destination/'timm-requirement.txt'
    requirements.write_text('timm==1.0.15 --hash=sha256:'+WHEEL_SHA+'\n')
    with (destination/'package-install.log').open('a') as stream:
        if not wheel.exists():
            subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'download', '--no-deps',
                            '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                            '--require-hashes', '-r', str(requirements), '--dest', str(wheel_dir)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        if file_hash(wheel) != WHEEL_SHA:
            raise ValueError('Pinned timm wheel bytes differ')
        if installed is None:
            subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'install', '--no-deps',
                            '--no-index', str(wheel)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    after = {name: importlib.metadata.version(name) for name in CORE}
    if before != after or importlib.metadata.version('timm') != '1.0.15':
        raise ValueError('Preparation changed an existing numerical dependency')
    from huggingface_hub import hf_hub_download
    weights = destination/'model.safetensors'
    if not weights.exists():
        cached = hf_hub_download(REPO, 'model.safetensors', revision=REVISION,
                                cache_dir=root/'cache/huggingface-casebag')
        if file_hash(cached) != WEIGHT_SHA or Path(cached).stat().st_size != WEIGHT_BYTES:
            raise ValueError('Public pinned checkpoint bytes differ')
        shutil.copyfile(cached, weights)
    if file_hash(weights) != WEIGHT_SHA or weights.stat().st_size != WEIGHT_BYTES:
        raise ValueError('Prepared checkpoint differs')
    import timm
    from safetensors.torch import load_file
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    if len(model.blocks) != 6 or model.num_features != 1280:
        raise ValueError('Expected EfficientNetV2B0 topology differs')
    original_parameters = sum(p.numel() for p in model.parameters())
    model.reset_classifier(0)
    receipt = {'status': 'public pretrained checkpoint prepared; no fitting or forwards',
               'preparation_source_sha': source_sha, 'script_sha256': file_hash(__file__),
               'repository': REPO, 'revision': REVISION, 'filename': weights.name,
               'sha256': WEIGHT_SHA, 'bytes': WEIGHT_BYTES, 'license': 'apache-2.0',
               'weight_access': 'public ungated checkpoint; no gated access request',
               'dependencies': {**after, 'timm': importlib.metadata.version('timm')},
               'core_dependencies_unchanged': True, 'timm_wheel_sha256': WHEEL_SHA, 'feature_dimension': 1280,
               'blocks': 6, 'model_parameters_with_original_classifier': original_parameters,
               'backbone_parameters': sum(p.numel() for p in model.parameters()),
               'strict_checkpoint_schema': True, 'new_fits': 0, 'new_encoder_forwards': 0,
               'validation_predictions': 0, 'outer_predictions': 0}
    path = destination/'preparation.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('Existing receipt differs; preserve it for review')
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args(); prepare(args.root, args.source_sha)

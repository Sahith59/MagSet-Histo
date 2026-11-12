"""Extract immutable frozen image features, sharded by magnification."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .data import MAGNIFICATIONS, file_hash, read_manifest


class Fields(Dataset):
    def __init__(self, rows, root):
        self.rows, self.root = rows, Path(root)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['relative_path']) as im:
            image = im.convert('RGB')
        return row['relative_path'], image


def identity_collate(batch):
    return batch


def worker_init(_):
    torch.set_num_threads(1)


def load_encoder(name, config, device):
    if name == 'phikon-v2':
        from transformers import AutoImageProcessor, AutoModel
        c = config[name]
        processor = AutoImageProcessor.from_pretrained(c['repository'], revision=c['revision'], use_fast=False)
        model = AutoModel.from_pretrained(c['repository'], revision=c['revision'], use_safetensors=True)
        transform = lambda images: processor(images=images, return_tensors='pt')['pixel_values']
        encode = lambda x: model(pixel_values=x).last_hidden_state[:, 0, :]
        description = processor.to_dict()
    else:
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        operation = weights.transforms()
        transform = lambda images: torch.stack([operation(im) for im in images])
        encode = model
        description = repr(operation)
    model.eval().requires_grad_(False).to(device)
    return model, transform, encode, description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--images-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--encoder', choices=['phikon-v2', 'resnet50'], required=True)
    parser.add_argument('--encoders-config', type=Path, default=Path('configs/encoders.json'))
    parser.add_argument('--magnification', type=int, choices=MAGNIFICATIONS, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--code-sha', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Training-free smoke subset; use a separate output directory')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.encoders_config.read_text())
    rows = [r for r in read_manifest(args.manifest) if int(r['magnification']) == args.magnification]
    if args.limit < 0:
        raise ValueError('Limit must be nonnegative')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty magnification shard')
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / ('%s-%d.npz' % (args.encoder, args.magnification))
    meta_path = target.with_suffix('.json')
    manifest_hash = file_hash(args.manifest)
    if target.exists() or meta_path.exists():
        if not (target.exists() and meta_path.exists()):
            raise ValueError('Incomplete cache exists; inspect before replacing')
        old = json.loads(meta_path.read_text())
        if old['manifest_sha256'] != manifest_hash or old['encoder'] != config[args.encoder] or old['feature_sha256'] != file_hash(target) or old['images'] != len(rows):
            raise ValueError('Existing cache provenance differs; use a separate output directory')
        print('Verified existing cache: ' + str(target), flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation/framework not available')
    torch.manual_seed(812)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, transform, encode, transform_description = load_encoder(args.encoder, config, device)
    loader = DataLoader(Fields(rows, args.images_root), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=identity_collate,
                        worker_init_fn=worker_init, persistent_workers=args.workers > 0,
                        multiprocessing_context='spawn' if args.workers > 0 else None)
    paths, features = [], []
    start = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            keys, images = zip(*batch)
            pixels = transform(list(images)).to(device)
            z = encode(pixels).float().cpu().numpy()
            if z.shape != (len(batch), config[args.encoder]['dimension']) or not np.isfinite(z).all():
                raise ValueError('Invalid feature shape/values')
            paths.extend(keys)
            features.append(z)
            if len(paths) % 256 < len(batch):
                print(json.dumps({'encoded':len(paths),'total':len(rows)}), flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    temp = target.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, paths=np.asarray(paths), features=np.concatenate(features))
    os.replace(temp, target)
    record = {'manifest_sha256':manifest_hash, 'encoder':config[args.encoder], 'code_sha':args.code_sha,
              'magnification':args.magnification, 'images':len(paths), 'smoke_limit':args.limit, 'elapsed_seconds':elapsed,
              'images_per_second':len(paths)/elapsed, 'precision':'float32; TF32 disabled',
              'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
              'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
              'python':platform.python_version(), 'torch':torch.__version__,
              'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'transform':transform_description,
              'feature_sha256':file_hash(target),
              'fit_status':'frozen external pretrained encoder; no cohort fitting or labels used'}
    temp_meta = meta_path.with_suffix('.json.tmp')
    temp_meta.write_text(json.dumps(record, indent=2)+'\n')
    os.replace(temp_meta, meta_path)
    print(json.dumps({k:v for k,v in record.items() if k!='transform'},indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Small-state, stateless image sampling and cached-field scoring for LoRA runs."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .data import MAGNIFICATIONS
from .field_five import score_records


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).digest()[:8], 'big')%(2**63-1)


def run_seed(seed, trial):
    return 812+10000*int(seed)+int(trial)


def case_order(indices, seed, epoch):
    return np.random.default_rng(stable_seed(seed, epoch, 'order')).permutation(indices).tolist()


def sampled_paths(case, seed, epoch):
    result = []
    for view, mag in enumerate(MAGNIFICATIONS):
        paths = sorted(case['indices'][mag])
        if not paths: raise ValueError('Every fitting magnification requires fields')
        rng = np.random.default_rng(stable_seed(seed, epoch, case['case_id'], view, 'paths'))
        result.extend((view, sample, str(path)) for sample, path in enumerate(rng.choice(paths, 2, replace=len(paths)<2)))
    return result


def build_augmentation():
    from torchvision.transforms import v2, InterpolationMode
    before = v2.Compose([v2.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True), v2.CenterCrop(224)])
    after = v2.Compose([v2.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=0),
                        v2.ToTensor(), v2.Normalize((.5,)*3, (.5,)*3)])
    def transform(image, seed):
        from PIL import Image
        # Seed only the CPU generator; CUDA dropout RNG is neither seeded nor advanced here.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            image = before(image)
            rotation = int(torch.randint(4, ()).item())
            if rotation: image = image.transpose([None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270][rotation])
            if bool(torch.randint(2, ()).item()): image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return after(image)
    return transform


def expected_paths(cases, indices):
    return [path for i in indices for mag in MAGNIFICATIONS for path in sorted(cases[i]['indices'][mag])]


def role_lookup(rows, cases, indices):
    paths = expected_paths(cases, indices); by_path = {r['relative_path']: r for r in rows}
    if len(set(paths)) != len(paths) or not set(paths) <= set(by_path): raise ValueError('Allowed-role path coverage differs')
    return {p: by_path[p] for p in paths}


def records_from_features(paths, features, arrays, cases, indices, role, arm):
    if list(paths) != expected_paths(cases, indices) or features.shape != (len(paths), arrays['mean'].shape[1]) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('Exact allowed-role finite field feature coverage is required')
    lookup = {p: value for p, value in zip(paths, features)}; bags = [None]*len(cases)
    for i in indices:
        bags[i] = [np.asarray([lookup[p] for p in cases[i]['indices'][mag]], np.float32) for mag in MAGNIFICATIONS]
    return score_records(arrays, bags, indices, cases, role, arm)[0]


def joint_probabilities(logits, arm):
    if arm == 'direct': return torch.softmax(logits['direct'], dim=-1)
    r = torch.sigmoid(logits['binary'].reshape(-1)); q = torch.softmax(logits['malignant'], dim=-1)
    return torch.cat(((1-r)[:, None], r[:, None]*q), dim=-1)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.part')
    torch.save(cpu_tree(value), temporary); os.replace(temporary, path)


def rng_state():
    ns = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy': (ns[0], torch.from_numpy(ns[1].astype(np.int64)), *ns[2:])}


def restore_rng(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])
    n = state['numpy']; np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))


def digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def small_digests(state):
    groups = {'A': lambda n: n.endswith('.lora_A'), 'B': lambda n: n.endswith('.lora_B'),
              'heads': lambda n: n.startswith('heads.'), 'normalization': lambda n: n in ('mean', 'std')}
    return {g: digest({n: v for n, v in state.items() if pred(n)}) for g, pred in groups.items()}


def parity(reference, candidate, config):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all(): raise ValueError('Invalid parity shape/finite values')
    if not np.allclose(reference, candidate, atol=config['atol'], rtol=config['rtol']): raise ValueError('Actual LoRA feature/probability parity failed')
    return float(np.max(np.abs(reference-candidate)))

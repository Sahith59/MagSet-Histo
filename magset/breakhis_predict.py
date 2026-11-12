"""Replay a saved BreakHis validation case from checked original images; no fitting."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import time

import numpy as np
from scipy.special import softmax

from .baseline import vectorize
from .collect import SUBSETS, load_trusted_context, verify_run
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest


def choose_case(report, requested=None):
    ids = report['case_ids']['validation']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Need unique validation cases')
    case = sorted(ids)[0] if requested is None else requested
    if case not in ids:
        raise ValueError('Validation replay refuses fitting or outer-role cases')
    return case


def load_head(path, expected_sha, dimension):
    if file_hash(path) != expected_sha:
        raise ValueError('Saved head differs from submission-pinned checksum')
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ('mean', 'std', 'coef_0', 'intercept_0', 'classes_0')}
    if (arrays['mean'].shape != (4, dimension) or arrays['std'].shape != (4, dimension)
            or arrays['coef_0'].shape != (8, dimension+4) or arrays['intercept_0'].shape != (8,)
            or not np.array_equal(arrays['classes_0'], np.arange(8))
            or not all(np.isfinite(x).all() for x in arrays.values()) or np.any(arrays['std'] <= 0)):
        raise ValueError('Invalid saved preprocessing, class order or classifier shape')
    return arrays


def predict_fields(fields, views, head):
    views = tuple(views)
    if not views or len(set(views)) != len(views) or any(m not in MAGNIFICATIONS for m in views):
        raise ValueError('Need a unique nonempty subset of the four magnifications')
    dimension = head['mean'].shape[1]
    z = np.zeros((1, 4, dimension), dtype=np.float32)
    for m in views:
        if m not in fields:
            raise ValueError('A requested magnification has no images')
        x = np.asarray(fields[m], dtype=np.float32)
        if x.ndim != 2 or not len(x) or x.shape[1] != dimension or not np.isfinite(x).all():
            raise ValueError('Invalid observed image features')
        z[0, MAGNIFICATIONS.index(m)] = x.mean(axis=0)
    # This is the training code's affine normalization and mean+mask vectorizer.
    z = (z-head['mean'])/head['std']
    indices = tuple(MAGNIFICATIONS.index(m) for m in sorted(views))
    x = vectorize(z, indices, 'mean')
    p = softmax(x @ head['coef_0'].T + head['intercept_0'], axis=1)[0]
    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Invalid predicted probabilities')
    return p


def cached_validation_fields(rows, cases, cache, report, config):
    result = {c['case_id']: {} for c in cases}
    if len(report['feature_provenance']) != 4:
        raise ValueError('Expected four original feature provenance records')
    for j, m in enumerate(MAGNIFICATIONS):
        path = Path(cache)/('phikon-v2-%d.npz' % m)
        meta = json.loads(path.with_suffix('.json').read_text())
        expected = report['feature_provenance'][j]
        actual = {k: meta[k] for k in ('encoder', 'manifest_sha256', 'feature_sha256', 'precision')}
        if (actual != expected or meta['feature_sha256'] != file_hash(path)
                or meta['encoder'] != config or meta['magnification'] != m or meta['smoke_limit'] != 0):
            raise ValueError('Original feature cache provenance differs')
        with np.load(path, allow_pickle=False) as saved:
            keys = saved['paths'].tolist(); values = saved['features']
            paths = {r['relative_path'] for r in rows if int(r['magnification']) == m}
            if (len(set(keys)) != len(keys) or set(keys) != paths
                    or values.shape != (len(keys), config['dimension']) or values.dtype != np.float32
                    or not np.isfinite(values).all()):
                raise ValueError('Invalid original cache coverage/values')
            lookup = {key:i for i,key in enumerate(keys)}
            for c in cases:
                result[c['case_id']][m] = values[[lookup[p] for p in c['indices'][m]]]
    return result


def load_checked_image(root, row):
    from PIL import Image
    relative = PurePosixPath(row['relative_path']); root = Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('Unsafe image path')
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Image resolves outside the configured image root')
    if file_hash(path) != row['sha256']:
        raise ValueError('Original image checksum differs')
    with Image.open(path) as image:
        if image.size != (int(row['width']), int(row['height'])):
            raise ValueError('Original image size differs')
        return image.convert('RGB')


def run(manifest, protocol, run_dir, cache, images_root, model_sha, protocol_sha,
        output, code_sha, case_id=None, views=MAGNIFICATIONS, batch_size=16):
    if Path(output).exists():
        raise ValueError('Demo output exists; preserve the original replay artifact')
    if file_hash(protocol) != protocol_sha or batch_size < 1:
        raise ValueError('Registered protocol hash or batch size differs')
    trusted = load_trusted_context(manifest, protocol)
    report, hashes = verify_run(Path(run_dir)/'metrics.json', trusted=trusted)
    if (report['encoder'] != 'phikon-v2' or report['method'] != 'mean'
            or report.get('normalization_policy') != 'field-moments-v2'
            or report['trial'] != 1 or report['seed'] != 812):
        raise ValueError('Demo uses the fixed matched Phikon mean trial1/seed812 checkpoint')
    selected = choose_case(report, case_id)
    views = tuple(sorted(views))
    if not views or len(set(views)) != len(views) or any(m not in MAGNIFICATIONS for m in views):
        raise ValueError('Invalid observed magnifications')
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/encoders.json').read_text())
    head = load_head(Path(run_dir)/'linear-model.npz', model_sha, config['phikon-v2']['dimension'])
    rows = read_manifest(manifest)
    cases = [c for c in trusted['cases'] if c['case_id'] in report['case_ids']['validation']]
    cached = cached_validation_fields(rows, cases, cache, report, config['phikon-v2'])
    records = [json.loads(line) for line in (Path(run_dir)/'predictions.jsonl').read_text().splitlines()]
    reference = {(r['case_id'], tuple(r['magnifications'])): np.asarray(r['probabilities']) for r in records}
    cached_error = max(float(np.max(np.abs(predict_fields(cached[c['case_id']], s, head)-reference[c['case_id'],s])))
                       for c in cases for s in SUBSETS)
    if cached_error > 1e-8:
        raise ValueError('Saved head/cache replay differs from archived validation probabilities')
    # Nothing below fits an encoder, normalizer, head or hyperparameter.
    import torch
    from .features import load_encoder
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('This raw-image demo requires its allocated GPU')
    torch.manual_seed(812); torch.set_num_threads(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    model, transform, encode, _ = load_encoder('phikon-v2', config, device)
    selected_case = next(c for c in cases if c['case_id'] == selected)
    by_path = {r['relative_path']:r for r in rows}; fields = {}; images = 0
    with torch.inference_mode():
        for m in views:
            paths = selected_case['indices'][m]; values = []
            for offset in range(0, len(paths), batch_size):
                batch = [load_checked_image(images_root, by_path[p]) for p in paths[offset:offset+batch_size]]
                z = encode(transform(batch).to(device)).float().cpu().numpy()
                if z.shape != (len(batch), config['phikon-v2']['dimension']) or not np.isfinite(z).all():
                    raise ValueError('Invalid raw-image encoder output')
                values.append(z); images += len(batch)
            fields[m] = np.concatenate(values)
    predictions = {}; worst = 0.
    for s in SUBSETS:
        if not set(s).issubset(views):continue
        p = predict_fields(fields, s, head); error = float(np.max(np.abs(p-reference[selected,s])))
        worst = max(worst, error)
        predictions['+'.join(map(str,s))] = {'probabilities':p.tolist(),
            'predicted_subtype':CLASS_NAMES[int(p.argmax())], 'archive_max_probability_difference':error}
    if worst > 1e-4:
        raise ValueError('Raw-image replay exceeds registered probability tolerance: '+str(worst))
    result = {'status':'raw-image validation replay verified', 'dataset':'BreakHis', 'role':'validation',
              'selection_rule':'lexicographically first validation case' if case_id is None else 'explicit validation case',
              'case_id':selected, 'truth':CLASS_NAMES[selected_case['label']], 'trial':1, 'seed':812,
              'encoder':config['phikon-v2'], 'class_names':CLASS_NAMES, 'magnifications':views,
              'image_count':images, 'fields_by_view':{str(m):len(v) for m,v in fields.items()},
              'cached_validation_records_replayed':len(reference), 'cached_replay_max_probability_difference':cached_error,
              'raw_replay_max_probability_difference':worst, 'predictions':predictions,
              'model_sha256':model_sha, 'manifest_sha256':trusted['manifest_sha256'], 'protocol_sha256':protocol_sha,
              **hashes, 'demo_code_sha':code_sha, 'demo_code_sha256':file_hash(__file__),
              'elapsed_seconds':time.monotonic()-start, 'gpu':torch.cuda.get_device_name(device),
              'torch':str(torch.__version__), 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
              'new_classifier_fits':0, 'outer_predictions_computed':0,
              'interpretation':'End-to-end execution/parity demonstration on an already scored validation case, not new performance or independent confirmation'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','image_count','cached_validation_records_replayed',
        'cached_replay_max_probability_difference','raw_replay_max_probability_difference','new_classifier_fits','outer_predictions_computed')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','run-dir','cache','images-root','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-model-sha','expected-protocol-sha','code-sha'):p.add_argument('--'+name,required=True)
    p.add_argument('--case-id');p.add_argument('--magnifications',nargs='+',type=int,choices=MAGNIFICATIONS,default=MAGNIFICATIONS)
    p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
    run(a.manifest,a.protocol,a.run_dir,a.cache,a.images_root,a.expected_model_sha,a.expected_protocol_sha,
        a.output,a.code_sha,a.case_id,a.magnifications,a.batch_size)

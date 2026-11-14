"""Fixed-C matched thumbnail/multicrop development comparison (no outer scoring)."""
import argparse
import json
from pathlib import Path
import warnings
import re
import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from .bags import load_bags, fit_normalization
from .capstone_bridge import metrics as complete_metrics
from .baseline import SUBSETS, fit_models, vectorize, stratified_inner_split
from .collect import load_trusted_context, verify_run, metrics_from_scores
from .data import CLASS_NAMES, MAGNIFICATIONS, file_hash, read_manifest

ARMS = ('thumbnail', 'multicrop')
EXTRACTOR_SOURCE_FILES = ('magset/multicrop_features.py', 'magset/features.py', 'magset/data.py',
                          'magset/breakhis_predict.py', 'magset/baseline.py', 'magset/collect.py')


def validate_config(config):
    expected = {'thumbnail_weight': .5, 'quadrant_weight': .5, 'fixed_C': .01,
                'max_iter': 1000, 'normalization': 'field-moments-v2'}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed experiment settings differ')
    if config['encoder']['dimension'] != 1024:
        raise ValueError('Expected exact Phikon-v2 encoder dimension')


def load_inputs(rows, thumbnail, multicrop, manifest_hash, config, config_hash, expected_source=None):
    """Check original thumbnails, raw metadata, quadrants and fixed arithmetic."""
    cases, control, original_meta = load_bags(rows, thumbnail, 'phikon-v2', manifest_hash)
    other_cases, candidate, candidate_meta = load_bags(rows, multicrop, 'phikon-v2', manifest_hash)
    if [c['case_id'] for c in cases] != [c['case_id'] for c in other_cases]:
        raise ValueError('Case identities differ')
    if len({json.dumps({k: meta.get(k) for k in ('code_sha', 'archive_sha256', 'source_sha256')}, sort_keys=True) for meta in candidate_meta}) != 1:
        raise ValueError('Candidate shard source identities differ')
    manifest = {r['relative_path']: r for r in rows}
    for scale, old, new in zip(MAGNIFICATIONS, original_meta, candidate_meta):
        pins = {'npz_sha256': file_hash(Path(thumbnail)/('phikon-v2-%d.npz'%scale)),
                'json_sha256': file_hash(Path(thumbnail)/('phikon-v2-%d.json'%scale))}
        if expected_source is not None and any(new.get(k) != v for k, v in expected_source.items()):
            raise ValueError('Candidate source differs from immutable execution contract')
        if (new.get('config') != config or new.get('smoke') is not False
                or new.get('smoke_limit') != 0 or new.get('smoke_roles') is not None
                or new.get('quadrant_forwards') != 4*new['images']
                or new.get('technical_thumbnail_forwards') != 0
                or new.get('blend_replay_max_abs') != 0
                or new.get('thumbnail_parity_max_abs') is not None
                or new.get('blend') != 'float32(0.5*float64(g) + 0.5*mean(q1..q4,float64))'):
            raise ValueError('Full-cache budget/configuration proof differs')
        if pins != config['thumbnail_provenance'][str(scale)] or new.get('thumbnail_provenance') != pins:
            raise ValueError('Registered thumbnail provenance differs')
        if new.get('thumbnail_metadata') != old or new.get('transform') != old.get('transform'):
            raise ValueError('Thumbnail metadata/transform differs')
        if (not re.fullmatch(r'[0-9a-f]{40}', new.get('code_sha', ''))
                or not re.fullmatch(r'[0-9a-f]{64}', new.get('archive_sha256', ''))
                or not isinstance(new.get('source_sha256'), dict) or not new['source_sha256']
                or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in new['source_sha256'].values())):
            raise ValueError('Candidate source provenance differs')
        if old['encoder'] != config['encoder'] or new['encoder'] != config['encoder']:
            raise ValueError('Encoder identity differs')
        if (new.get('thumbnail_feature_sha256') != old['feature_sha256']
                or new.get('config_sha256') != config_hash or not new.get('code_sha')
                or not new.get('transform')):
            raise ValueError('Candidate source/transform/config/thumbnail provenance missing')
        with np.load(Path(thumbnail)/('phikon-v2-%d.npz' % scale), allow_pickle=False) as a:
            lookup = {p: a['features'][i].copy() for i, p in enumerate(a['paths'].tolist())}
        with np.load(Path(multicrop)/('phikon-v2-%d.npz' % scale), allow_pickle=False) as a:
            paths = a['paths'].tolist(); n = len(paths); d = config['encoder']['dimension']
            q = a['quadrant_features']; g = a['thumbnail_features']; f = a['features']
            if (q.shape != (n, 4, d) or g.shape != (n, d) or f.shape != (n, d)
                    or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (q, g, f))
                    or a['bounds'].shape != (n, 4, 4)
                    or a['widths'].shape != (n,) or a['heights'].shape != (n,)
                    or a['raw_sha256'].shape != (n,)):
                raise ValueError('Invalid candidate vector/metadata schema')
            if not np.array_equal(g, np.stack([lookup[p] for p in paths])):
                raise ValueError('Candidate thumbnail differs from original cache')
            expected = np.asarray(.5*g.astype(np.float64)+.5*q.mean(axis=1, dtype=np.float64), dtype=np.float32)
            if not np.array_equal(f, expected):
                raise ValueError('Fixed multicrop blend differs')
            for i, p in enumerate(paths):
                r = manifest[p]; w = int(r['width']); h = int(r['height'])
                bounds = [[0, 0, w//2, h//2], [w//2, 0, w, h//2],
                          [0, h//2, w//2, h], [w//2, h//2, w, h]]
                if (a['widths'][i] != w or a['heights'][i] != h
                        or a['raw_sha256'][i] != r['sha256']
                        or not np.array_equal(a['bounds'][i], bounds)):
                    raise ValueError('Raw identity/dimensions/quadrant bounds differ')
    return cases, {'thumbnail': control, 'multicrop': candidate}, original_meta, candidate_meta


def score_saved(path, z, indices, cases, role):
    """Independent NumPy replay without unpickling an estimator."""
    if role not in ('train', 'validation'):
        raise ValueError('Forbidden scoring role')
    with np.load(path, allow_pickle=False) as a:
        mean, std, coef, intercept, classes = [a[k] for k in ('mean', 'std', 'coef_0', 'intercept_0', 'classes_0')]
    if (mean.shape != z.shape[1:] or std.shape != mean.shape or np.any(std <= 0)
            or coef.shape != (8, z.shape[2]+4) or intercept.shape != (8,)
            or not np.array_equal(classes, np.arange(8))
            or not all(np.isfinite(x).all() for x in (mean, std, coef, intercept, z))):
        raise ValueError('Saved model schema/finite values differ')
    normalized = (z[indices]-mean)/std
    records = []
    for subset in SUBSETS:
        logits = vectorize(normalized, subset, 'mean')@coef.T+intercept
        log_probabilities = logits-logsumexp(logits, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        for index, probability, log_probability in zip(indices, probabilities, log_probabilities):
            records.append({'role': role, 'case_id': cases[index]['case_id'], 'truth': int(cases[index]['label']),
                            'magnifications': [MAGNIFICATIONS[m] for m in subset], 'probabilities': probability.tolist(),
                            'log_probabilities': log_probability.tolist()})
    return records


def summarize(records):
    subsets = {}
    identities = {}
    for row in records:
        cid, label = row['case_id'], row['truth']
        if cid in identities and identities[cid] != label:
            raise ValueError('Case truth changed')
        identities[cid] = label
    support = np.bincount(list(identities.values()), minlength=8)
    if np.any(support == 0):
        raise ValueError('Every class must have case support')
    def measured(selected, balanced=False):
        y = np.asarray([r['truth'] for r in selected], dtype=int)
        weights = np.asarray([len(identities)/(len(selected)*8*support[label]) for label in y]) if balanced else None
        return complete_metrics([r['probabilities'] for r in selected], y,
                                [r['log_probabilities'] for r in selected], weights)
    for subset in SUBSETS:
        views = [MAGNIFICATIONS[m] for m in subset]
        selected = [r for r in records if r['magnifications'] == views]
        if len(selected) != len(identities) or len({r['case_id'] for r in selected}) != len(identities):
            raise ValueError('Incomplete/duplicate case subset coverage')
        value = measured(selected)
        value['class_balanced'] = measured(selected, True)
        subsets['+'.join(map(str, views))] = value
    if len(records) != 15*len(identities):
        raise ValueError('Unexpected case/subset records')
    result = {'subsets': subsets, 'distinct_cases': len(identities), 'distinct_class_support': support.tolist(),
              'pooled': measured(records), 'pooled_class_balanced': measured(records, True),
              'all_view': subsets['40+100+200+400']}
    for key in ('macro_f1', 'accuracy', 'balanced_accuracy', 'cross_entropy', 'brier'):
        result['mean_subset_'+key] = float(np.mean([v[key] for v in subsets.values()]))
    result['mean_subset_class_balanced_cross_entropy'] = float(np.mean([v['class_balanced']['cross_entropy'] for v in subsets.values()]))
    result['worst_subset_macro_f1'] = float(min(v['macro_f1'] for v in subsets.values()))
    result['phyllodes_recall'] = float(result['pooled']['class_recall'][CLASS_NAMES.index('phyllodes_tumor')])
    return result


def run_one(folder, bags, cases, trusted, trial, seed, config, provenance, reference):
    train, val, test = stratified_inner_split(cases, trusted['trials'][str(trial)]['case_roles'], seed)
    if [len(x) for x in (train, val, test)] != [46, 8, 28]:
        raise ValueError('Exact retained role counts differ')
    z = np.asarray([[fields.mean(0) for fields in case] for case in bags], dtype=np.float32)
    mean, std = fit_normalization(bags, train)
    normalized = (z-mean)/std
    y = np.asarray([c['label'] for c in cases])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        models = fit_models(normalized[train], y[train], 'mean', config['fixed_C'])
    model = models[0]
    if len(models) != 1 or model.C != .01 or model.max_iter != 1000 or model.class_weight != 'balanced':
        raise ValueError('Unexpected estimator settings')
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/'model.npz', mean=mean, std=std, coef_0=model.coef_,
                        intercept_0=model.intercept_, classes_0=model.classes_)
    records = {}
    for role, indices in (('train', train), ('validation', val)):
        replay = score_saved(folder/'model.npz', z, indices, cases, role)
        for subset_index, subset in enumerate(SUBSETS):
            actual = model.predict_proba(vectorize(normalized[indices], subset, 'mean'))
            saved = np.asarray([r['probabilities'] for r in replay[subset_index*len(indices):(subset_index+1)*len(indices)]])
            if not np.allclose(actual, saved, rtol=0, atol=1e-12):
                raise ValueError('Independent saved-array probability replay differs')
        records[role] = replay
        name = 'predictions.jsonl' if role == 'validation' else 'fitting-predictions.jsonl'
        (folder/name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in replay))
    report = {'status': 'validation-only pilot', 'unit': trusted['unit'], 'encoder': 'phikon-v2',
              'method': 'mean', 'trial': trial, 'seed': seed, 'class_names': CLASS_NAMES,
              'manifest_sha256': trusted['manifest_sha256'], 'protocol_sha256': trusted['protocol_sha256'],
              'case_ids': {r: [cases[i]['case_id'] for i in ix] for r, ix in (('train', train), ('validation', val), ('test', test))},
              'fixed_C': config['fixed_C'], 'normalization_policy': config['normalization'],
              'model_selection': 'none: fixed C, no validation selection', 'converged': True,
              'n_iter': model.n_iter_.tolist(), 'validation': summarize(records['validation']),
              'fitting': summarize(records['train']), 'model_sha256': file_hash(folder/'model.npz'),
              'fitting_predictions_sha256': file_hash(folder/'fitting-predictions.jsonl'),
              'historical_reference': reference, **provenance}
    report['fitting_minus_validation'] = {key: report['fitting'][key]-report['validation'][key]
        for key in ('mean_subset_macro_f1', 'mean_subset_accuracy', 'mean_subset_balanced_accuracy',
                    'mean_subset_cross_entropy', 'mean_subset_class_balanced_cross_entropy', 'mean_subset_brier')}
    (folder/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    _, hashes = verify_run(folder/'metrics.json', trusted)
    return report, hashes


def compare(manifest, protocol, thumbnail, multicrop, reference_root, config_path, output, code_sha, expected_source=None):
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    if Path(output).exists():
        raise ValueError('Refusing existing comparison output')
    trusted = load_trusted_context(manifest, protocol); rows = read_manifest(manifest)
    cases, arms, old_meta, new_meta = load_inputs(rows, thumbnail, multicrop, file_hash(manifest), config, file_hash(config_path), expected_source)
    if config.get('manifest_sha256') != trusted['manifest_sha256'] or config.get('protocol_sha256') != trusted['protocol_sha256']:
        raise ValueError('Registered manifest/protocol differs')
    if cases != trusted['cases']:
        raise ValueError('Trusted case order differs')
    provenance = {'expected_extractor_source': expected_source, 'code_sha': code_sha, 'implementation_sha256': file_hash(__file__), 'config': config,
                  'config_sha256': file_hash(config_path), 'thumbnail_provenance': old_meta, 'multicrop_provenance': new_meta}
    # Verify every retained reference and cache binding before performing any fit.
    references = {}
    for seed in (812, 813):
        for trial in range(1, 6):
            path = Path(reference_root)/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed))/'metrics.json'
            ref, hashes = verify_run(path, trusted)
            if (ref['normalization_policy'] != 'field-moments-v2' or ref['method'] != 'mean'
                    or ref['encoder'] != 'phikon-v2' or ref['seed'] != seed or ref['trial'] != trial
                    or [m['feature_sha256'] for m in ref['feature_provenance']] != [m['feature_sha256'] for m in old_meta]):
                raise ValueError('Historical reference/cache binding differs')
            references[seed, trial] = {'hashes': hashes, 'validation': ref['validation'], 'selected_C': ref['selected_C']}
    reports = []
    for arm in ARMS:
        for seed in (812, 813):
            for trial in range(1, 6):
                report, hashes = run_one(Path(output)/arm/('split%d'%seed)/('trial%d'%trial), arms[arm], cases,
                    trusted, trial, seed, config, {'arm': arm, **provenance}, references[seed, trial])
                reports.append({'arm': arm, 'seed': seed, 'trial': trial, 'validation': report['validation'],
                                'fitting': report['fitting'], 'hashes': hashes, 'model_sha256': report['model_sha256']})
    if len(reports) != 20 or len({(r['arm'],r['seed'],r['trial']) for r in reports}) != 20:
        raise ValueError('Incomplete fixed 20-head matrix')
    summary = {}
    for seed in (812, 813):
        summary[str(seed)] = {}
        for arm in ARMS:
            selected = [r['validation'] for r in reports if r['arm'] == arm and r['seed'] == seed]
            summary[str(seed)][arm] = {k: float(np.mean([r[k] for r in selected])) for k in ('mean_subset_macro_f1', 'phyllodes_recall')}
        summary[str(seed)]['delta_f1'] = summary[str(seed)]['multicrop']['mean_subset_macro_f1']-summary[str(seed)]['thumbnail']['mean_subset_macro_f1']
    gate = all(r['delta_f1'] >= .02 and r['multicrop']['phyllodes_recall'] > 0 for r in summary.values())
    result = {'status': 'complete exploratory development comparison', 'fits': 20, 'gate_passed': gate,
              'gate_scope': 'descriptive only; reused development cases', 'summary': summary, 'runs': reports, **provenance}
    (Path(output)/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'thumbnail-cache', 'multicrop-cache', 'reference-root', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--code-sha', required=True); a = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (not re.fullmatch(r'[0-9a-f]{40}', a.code_sha)
            or not re.fullmatch(r'[0-9a-f]{64}', (source/'.archive-sha256').read_text().strip())
            or a.config.read_bytes() != (source/'configs/breakhis-multicrop-v1.json').read_bytes()
            or (source/'.source-sha').read_text().strip() != a.code_sha):
        raise ValueError('Use an immutable source snapshot with matching code SHA')
    expected_source = {'code_sha': a.code_sha, 'archive_sha256': (source/'.archive-sha256').read_text().strip(),
                       'source_sha256': {name: file_hash(source/name) for name in EXTRACTOR_SOURCE_FILES}}
    compare(a.manifest, a.protocol, a.thumbnail_cache, a.multicrop_cache, a.reference_root, a.config, a.output, a.code_sha, expected_source)


if __name__ == '__main__':
    main()

"""Audit the fixed native development artifacts and publish aggregate-only results."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.data import file_hash
from magset.native_pilot import ARMS, CONFIG, audit_predictions


def build(run, protocol, protocol_sha, markdown, ledger):
    run = Path(run)
    report = json.loads((run / 'pilot.json').read_text())
    trusted = json.loads(Path(protocol).read_text())
    if (report['schema'] != 'magset.native-development.v1' or report['config'] != CONFIG
            or report['fits_completed'] != 4 or report['sealed_predictions_computed'] != 0
            or report['fit_patients'] != 80 or report['development_patients'] != 16
            or report['training_rows'] != 1152 or not np.isclose(report['training_weight'], 80)
            or not all(np.isclose(v, 5) for v in report['class_training_weight'].values())
            or set(report['class_training_weight']) != set(map(str, range(16)))
            or report['protocol_sha256'] != protocol_sha or not report['preprocessing_reload_verified']):
        raise ValueError('Native report differs from the fixed pilot contract')
    if report['manifest_sha256'] != trusted['manifest_sha256']:
        raise ValueError('Native report manifest differs from trusted protocol')
    if set(report['arms']) != set(ARMS):
        raise ValueError('Native arm budget differs')
    expected_artifacts = {'started.json', 'preprocessing.npz', 'predictions.jsonl',
                          *(arm + '.npz' for arm in ARMS)}
    if set(report['artifacts']) != expected_artifacts:
        raise ValueError('Native artifact set is incomplete or unexpected')
    for name, digest in report['artifacts'].items():
        if Path(name).name != name or file_hash(run / name) != digest:
            raise ValueError('Saved artifact name/hash differs')
    started = json.loads((run / 'started.json').read_text())
    if started != {'config': CONFIG, 'code_sha': report['code_sha']}:
        raise ValueError('Start receipt differs from the recorded fixed settings')
    verified = audit_predictions(run / 'predictions.jsonl', protocol, protocol_sha)
    with np.load(run / 'preprocessing.npz', allow_pickle=False) as saved:
        if (saved['mean'].shape != (4, 2048) or saved['std'].shape != (4, 2048)
                or saved['projection'].shape != (2048, 32)
                or not all(np.isfinite(saved[k]).all() for k in ('mean', 'std', 'projection'))
                or np.any(saved['std'] <= 0)
                or not np.allclose(saved['projection'].T @ saved['projection'], np.eye(32), atol=1e-12, rtol=0)):
            raise ValueError('Invalid saved preprocessing')
    for arm in ARMS:
        if verified[arm] != report['arms'][arm]['development']:
            raise ValueError('Recomputed native metrics differ')
        expected_count = 32848 if arm == 'mean' else 33360
        if report['arms'][arm]['stored_coefficient_count'] != expected_count:
            raise ValueError('Unexpected coefficient count')
        with np.load(run / (arm + '.npz'), allow_pickle=False) as saved:
            dimension = 2052 if arm == 'mean' else 2084
            if (saved['coef'].shape != (16, dimension) or saved['intercept'].shape != (16,)
                    or not np.array_equal(saved['classes'], np.arange(16))
                    or not np.isfinite(saved['coef']).all() or not np.isfinite(saved['intercept']).all()):
                raise ValueError('Saved classifier shape/classes/values differ')
    raw = [json.loads(s) for s in (run / 'predictions.jsonl').read_text().splitlines()]
    lookup = {(r['arm'], r['patient_id'], tuple(r['subset'])): r for r in raw}
    base = [r for r in raw if r['arm'] == 'mean']
    changes = {}
    for arm in ARMS[1:]:
        changed = [r for r in base if np.argmax(r['probabilities']) !=
                   np.argmax(lookup[arm, r['patient_id'], tuple(r['subset'])]['probabilities'])]
        changes[arm] = {'changed_decisions_of240': len(changed),
                        'distinct_development_patients': len({r['patient_id'] for r in changed})}
    labels = {'mean': 'Mean + mask', 'duplicate': 'Mean + duplicate mean',
              'random': 'Mean + random contrast', 'affinity': 'Mean + affinity contrast'}
    scores = {a: verified[a]['mean_subset_macro_f1'] for a in ARMS}
    favorable = scores['affinity'] > max(scores[a] for a in ARMS if a != 'affinity')
    decision = ('The affinity arm exceeds all three controls on this development endpoint. This is a feasibility signal only; '
                'prespecified stronger controls are required before an agreement-specific claim.' if favorable else
                'The affinity arm does not exceed all three controls on the prespecified development endpoint. '
                'Stop the agreement-specific improvement claim for this pilot; do not expand a score-driven search or inspect sealed outcomes.')
    lines = ['# Native four-fit development results', '',
             'Measured November 12, 2026. Secondary 16-class multi-organ task; not eight-class breast validation, a published-method reproduction, or final test performance.', '',
             'Exactly four fits used the registered 80 fit /16 development /32 reserved patient protocol. '
             'The 32 reserved patients were excluded from classifier construction and prediction. '
             'Each arm has 240 development records (16 patients ×15 view subsets), representing 16 independent patients.', '',
             '| Arm | Mean subset macro F1 | Full-view F1 | Worst-subset F1 | Mean Brier ↓ | Coefficients | Fit seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in ARMS:
        m = verified[a]; info = report['arms'][a]
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f | %d | %.3f |' % (
            labels[a], m['mean_subset_macro_f1'], m['full_view_macro_f1'], m['worst_subset_macro_f1'],
            m['mean_subset_brier'], info['stored_coefficient_count'], info['fit_seconds']))
    lines += ['', '## Prespecified decision', '', decision, '',
              'Compared with mean-only predictions, duplicate/random/affinity change %d/%d/%d of240 decisions, '
              'covering %d/%d/%d distinct development patients respectively. These are descriptive counts, not independent replications.' % (
                  changes['duplicate']['changed_decisions_of240'], changes['random']['changed_decisions_of240'],
                  changes['affinity']['changed_decisions_of240'], changes['duplicate']['distinct_development_patients'],
                  changes['random']['distinct_development_patients'], changes['affinity']['distinct_development_patients']), '',
              'All models converged before development scoring. Fixed ImageNet ResNet features, fitting-only normalization, '
              '32D projection seed812, temperature0.2, logistic C0.1/L2/LBFGS, and inverse-subset weights were shared as specified. '
              'No setting was selected from these scores. Patient training weight totals80 and5 per class.', '',
              '## All availability subsets', '',
              'The same16 development patients appear in every row. Rows and class outcomes are dependent; '
              'no significance test or naive row-bootstrap confidence interval is claimed.', '',
              '| Available resolutions (microns/pixel) | Mean F1 | Duplicate F1 | Random F1 | Affinity F1 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for subset in verified['mean']['subsets']:
        lines.append('| %s | %s |' % (subset, ' | '.join('%.6f' % verified[a]['subsets'][subset]['macro_f1'] for a in ARMS)))
    lines += ['', '## Compute, artifacts, and limits', '',
              'All field features were cached before this pilot. Shared feature construction took %.3fs for fitting patients and %.3fs for development patients. '
              'Added-feature arms have the same classifier coefficient count, but equal dimensions do not establish identical effective capacity. '
              'Saved block RMS diagnostics are unweighted over expanded rows; they were not used to rescale or select models.' % (
                  report['feature_seconds']['fit_all_arms_shared'], report['feature_seconds']['development_all_arms_shared']), '',
              'The collector independently checked all960 prediction records against trusted protocol roles, labels, exact patient/subset coverage, '
              'canonical16-class probability columns, and recomputed metrics. Artifact hashes and reload checks passed. '
              'Full per-class recall/confusion matrices, Brier values, and per-patient predictions are retained privately. '
              'The public ledger contains aggregates and hashes, not patient identifiers.', '',
              'Only one development patient per class supports this screen. Six naturally incomplete patients are fit-only. '
              'Frozen ImageNet ResNet is the initial provenance control; Phikon has known TCGA-source pretraining. '
              'The wider project still needs compatible prior-work comparisons, independent breast evidence, and a defensible contribution before submission.', '',
              'Experimental source: `%s`. SLURM job: `%s`. Full private report SHA256: `%s`.' % (
                  report['code_sha'], report['environment']['slurm_job_id'], file_hash(run / 'pilot.json')), '',
              '[Registered specification](NATIVE_DEVELOPMENT_PILOT_SPEC.md) · '
              '[Plain-language project guide](PROJECT_EXPLAINED.md) · [Native protocol](TCGA_QUARANTINE_PROTOCOL.md)', '']
    Path(markdown).write_text('\n'.join(lines))
    public = {'schema': 'magset.native-pilot-public-evidence.v1', 'source_sha': report['code_sha'],
              'specification_sha256': report['specification_sha256'], 'protocol_sha256': protocol_sha,
              'manifest_sha256': report['manifest_sha256'], 'cache_audit_sha256': report['cache_audit_sha256'],
              'report_sha256': file_hash(run / 'pilot.json'), 'predictions_sha256': file_hash(run / 'predictions.jsonl'),
              'builder_sha256': file_hash(__file__), 'config': CONFIG, 'fits': 4, 'development_patients': 16,
              'verified_prediction_records': 960, 'sealed_predictions_computed': 0,
              'arms': report['arms'], 'changed_decisions': changes, 'decision': decision, 'environment': report['environment']}
    Path(ledger).write_text(json.dumps(public, indent=2) + '\n')
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'markdown', 'ledger'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    a = parser.parse_args()
    result = build(a.run, a.protocol, a.expected_protocol_sha, a.markdown, a.ledger)
    print(json.dumps({arm: result['arms'][arm]['development']['mean_subset_macro_f1'] for arm in ARMS}))

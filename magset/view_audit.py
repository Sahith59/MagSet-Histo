"""Describe view-addition effects using existing development predictions only."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import file_hash
from .native_pilot import ARMS, SUBSETS, audit_predictions
from .tcga_protocol import RESOLUTIONS

EDGES = tuple((s, tuple(sorted((*s, m)))) for s in SUBSETS for m in range(4) if m not in s)


def describe_arm(records):
    lookup = {(r['patient_id'], tuple(r['subset'])): r for r in records}
    patients = sorted({r['patient_id'] for r in records})
    if not patients or len(lookup) != len(records) or len(records) != len(patients) * 15:
        raise ValueError('Need exactly15 unique subsets per development patient')
    harmed, helped = set(), set(); edges = []
    for parent, child in EDGES:
        counts = {'helped': 0, 'harmed': 0, 'both_correct': 0, 'both_wrong': 0,
                  'changed_class': 0, 'harmed_with_increased_max_probability': 0}
        probability_changes, brier_changes = [], []
        for pid in patients:
            if (pid, parent) not in lookup or (pid, child) not in lookup:
                raise ValueError('Incomplete availability coverage')
            a, b = lookup[pid, parent], lookup[pid, child]
            if a['truth'] != b['truth']:
                raise ValueError('Patient truth changes across subsets')
            y = a['truth']; p = np.asarray(a['probabilities']); q = np.asarray(b['probabilities'])
            if (p.shape != (16,) or q.shape != (16,) or not np.isfinite([p, q]).all()
                    or np.any(p < 0) or np.any(q < 0)
                    or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0)
                    or not np.isclose(q.sum(), 1, atol=1e-10, rtol=0)):
                raise ValueError('Invalid class probabilities')
            before, after = p.argmax() == y, q.argmax() == y
            name = 'both_correct' if before and after else 'harmed' if before else 'helped' if after else 'both_wrong'
            counts[name] += 1; counts['changed_class'] += int(p.argmax() != q.argmax())
            if name == 'harmed':
                harmed.add(pid)
                counts['harmed_with_increased_max_probability'] += int(q.max() > p.max())
            if name == 'helped': helped.add(pid)
            target = np.eye(16)[y]
            probability_changes.append(float(q[y] - p[y]))
            brier_changes.append(float(np.square(q-target).sum() - np.square(p-target).sum()))
        edges.append({'from': list(parent), 'to': list(child), **counts,
                      'mean_true_class_probability_change': float(np.mean(probability_changes)),
                      'mean_brier_change': float(np.mean(brier_changes))})
    totals = {k: sum(e[k] for e in edges) for k in counts}
    return {'patients': len(patients), 'view_addition_edges': len(EDGES),
            'patient_edge_comparisons': len(patients)*len(EDGES),
            'distinct_patients_harmed_at_least_once': len(harmed),
            'distinct_patients_helped_at_least_once': len(helped), 'totals': totals, 'edges': edges}


def build(run, protocol, protocol_sha, report_sha, output, markdown):
    run = Path(run); report_path = run/'pilot.json'
    if file_hash(report_path) != report_sha:
        raise ValueError('Pilot report differs from the fixed audit input hash')
    report = json.loads(report_path.read_text())
    predictions = run/'predictions.jsonl'
    if (report['protocol_sha256'] != protocol_sha
            or report['artifacts']['predictions.jsonl'] != file_hash(predictions)):
        raise ValueError('Prediction provenance differs')
    metrics = audit_predictions(predictions, protocol, protocol_sha)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    arms = {a: describe_arm([r for r in rows if r['arm'] == a]) for a in ARMS}
    result = {'schema': 'magset.view-addition-audit.v1', 'protocol_sha256': protocol_sha,
              'pilot_report_sha256': report_sha, 'predictions_sha256': file_hash(predictions),
              'analysis_code_sha256': file_hash(__file__), 'classifier_fits_performed': 0,
              'sealed_predictions_computed': 0, 'independent_development_patients': 16,
              'arms': arms,
              'interpretation': 'Posthoc descriptive development analysis; repeated masks/edges do not increase independent sample size or establish a new mechanism'}
    if any(a['patients'] != 16 or a['patient_edge_comparisons'] != 448 for a in arms.values()):
        raise ValueError('Unexpected trusted development cohort')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# What happens when we add another view?', '',
             'Posthoc descriptive analysis of the existing fixed native development pilot. No new classifier was fitted and no reserved patient was scored.', '',
             'For each of the same 16 development patients, compare every nonempty availability subset with each subset obtained by adding exactly one view. '
             'There are 28 such edges and 448 patient/edge comparisons per arm. These remain 16 independent patients, not 448 observations from different patients.', '',
             '| Arm | Wrong → right | Right → wrong | Both right | Both wrong | Patients harmed at least once | Newly wrong with higher maximum probability |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, a in arms.items():
        t = a['totals']
        lines.append('| %s | %d | %d | %d | %d | %d | %d |' % (name,t['helped'],t['harmed'],t['both_correct'],t['both_wrong'],
                    a['distinct_patients_harmed_at_least_once'],t['harmed_with_increased_max_probability']))
    lines += ['', '“Harmed” means that this fixed classifier changes a correct label to a wrong label after another view is supplied. '
              'Higher maximum probability describes its numerical confidence; it is not evidence that the confidence is calibrated. '
              'A patient can be helped on one edge and harmed on another.', '',
              '## Availability scores already measured', '',
              '| Arm |0.25mpp only F1|0.5mpp only F1|1.0mpp only F1|2.0mpp only F1|All four F1|',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for a in ARMS:
        m = metrics[a]['subsets']
        lines.append('| %s | %s |' % (a,' | '.join('%.6f'%m[s]['macro_f1'] for s in (*RESOLUTIONS,'+'.join(RESOLUTIONS)))))
    lines += ['', '## Every one-view addition for the mean control', '',
              '| From mpp | To mpp | Helped | Harmed | Mean true-label probability change | Mean Brier change ↓ |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for e in arms['mean']['edges']:
        label = lambda values: '+'.join(RESOLUTIONS[m] for m in values)
        lines.append('| %s | %s | %d | %d | %.6f | %.6f |' % (label(e['from']),label(e['to']),e['helped'],e['harmed'],
                     e['mean_true_class_probability_change'],e['mean_brier_change']))
    lines += ['', '## What this permits us to say', '',
              'Additional input can change a fixed model from correct to wrong. This supports investigating failure patterns, but does not prove that '
              'a new gate, uncertainty method, transport mechanism, or larger encoder will fix them. Selective fusion has substantial prior work. '
              'The agreement-specific pilot still loses to the duplicate control. No new primary endpoint, best mask, model setting, or significance claim is selected here.', '',
              'Patient-level examples remain private; the public JSON retains aggregates for every edge and arm. '
              'All counts reuse the same small development cohort. The 32 reserved patients remain outside this analysis.', '',
              'Input report SHA256: `'+report_sha+'`. Analysis source SHA256: `'+result['analysis_code_sha256']+'`.', '',
              '[Original pilot](NATIVE_DEVELOPMENT_RESULTS.md) · [Next tasks](NEXT_STEPS.md)', '']
    Path(markdown).write_text('\n'.join(lines)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','protocol','output','markdown'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-protocol-sha','expected-report-sha'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=build(a.run,a.protocol,a.expected_protocol_sha,a.expected_report_sha,a.output,a.markdown)
    print(json.dumps({name:value['totals'] for name,value in r['arms'].items()},indent=2))

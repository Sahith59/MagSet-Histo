"""Generate the second-round public table only from complete, verified run matrices."""
import argparse
import json
from pathlib import Path
from .collect import collect
from .error_analysis import analyze

METHODS = tuple('field-%s-view-%s'%(f,v) for f in ('mean','attention') for v in ('mean','gated','attention'))


def render(initial, field, fusion, diagnostics):
    reference = next(r for r in initial['summary'] if r['encoder']=='phikon-v2' and r['method']=='mean')
    lines = ['# Field and learned-fusion validation results', '',
             'All 20 field-linear and 30 hierarchical neural configurations completed. Each saved probability '
             'record independently reproduces the reported metrics. All selected models passed reload checks. '
             'Outer-test outcomes were not scored.', '',
             'These are exploratory results from eight inner-validation cases per trial and five overlapping '
             'provider holdouts. Hyperparameters were selected on these validation cases. The tables do not '
             'establish independent generalization or statistical significance.', '',
             '## Field aggregation controls', '',
             'Both controls use identical case-balanced field normalization and the same four C values. '
             '`field-late` classifies individual fields, then averages probabilities. `field-early-late` '
             'classifies the mean field embedding. Both average available scale probabilities.', '',
             '| Encoder | Aggregation | Mean subset macro F1 | Full-view macro F1 |',
             '| --- | --- | ---: | ---: |']
    for r in field['summary']:
        lines.append('| %s | %s | %.4f | %.4f |'%(r['encoder'],r['method'],r['mean_validation_subset_macro_f1'],r['mean_validation_full_view_macro_f1']))
    lines += ['', '## Neural controls with Phikon-v2', '',
              'All methods use the same frozen fields, 100 fixed epochs, four optimizer settings, split, '
              'and training seed. Parameter counts differ; see the registered protocol.', '',
              '| Field pooling | View fusion | Mean subset macro F1 | Full-view macro F1 |',
              '| --- | --- | ---: | ---: |']
    for r in fusion['summary']:
        parts = r['method'].split('-')
        lines.append('| %s | %s | %.4f | %.4f |'%(parts[1],parts[3],r['mean_validation_subset_macro_f1'],r['mean_validation_full_view_macro_f1']))
    lines += ['', '## Interpretation and next decision', '',
              'The original Phikon mean-feature linear baseline remains higher on the primary descriptive '
              'validation average (%.4f). Neural capacity has not produced an improvement in this screen. '
              'Normalization and optimization differ from the original round, so this is not a pure capacity ablation.'%reference['mean_validation_subset_macro_f1'], '',
              'For Phikon, the matched field-classification and early-mean controls have very similar averages; '
              'the present results do not support a strong claim that early averaging is the dominant problem. '
              'The ResNet controls behave differently, which motivates encoder sensitivity rather than a universal assertion.', '',
              'The neural fitting losses approach zero while validation scores vary widely by trial. This is '
              'consistent with overfitting or limited case diversity; it does not by itself prove the cause. '
              'Check split/training-seed stability and lower-capacity controls before investing in a complex matching mechanism.', '',
              '### Rare-subtype recall in the neural controls', '',
              'Recall below counts repeated validation predictions across subsets/trials; distinct-case support '
              'is only two phyllodes cases and three papillary cases in the seed-812 pilot.', '',
              '| Field pooling / view fusion | Phyllodes recall | Papillary recall |',
              '| --- | ---: | ---: |']
    for g in diagnostics['groups']:
        recalls = {r['class']:r['recall'] for r in g['class_recall']}
        parts = g['method'].split('-')
        lines.append('| %s / %s | %.3f | %.3f |'%(parts[1],parts[3],recalls['phyllodes_tumor'],recalls['papillary_carcinoma']))
    lines += ['', 'Source revision for this round: `1b2a0dd917495c4d2e70e2bd3fc805f43141a087`. '
              'See [the protocol](FUSION_PILOT.md), [initial error analysis](ERROR_ANALYSIS.md), and '
              '[contribution gates](RESEARCH_DIRECTION.md).', '',
              'Reproduce with `python -m magset.round2_report --initial-runs /path/to/validation-v1 '
              '--field-runs /path/to/field-linear-v1 --fusion-runs /path/to/fusion-v1 --manifest /path/to/manifest.csv --protocol /path/to/protocol.json --output docs/FUSION_RESULTS.md`.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-runs','field-runs','fusion-runs','output','manifest','protocol'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    context={'manifest':a.manifest,'protocol':a.protocol}
    initial = collect(a.initial_runs,**context)
    field = collect(a.field_runs,methods=('field-late','field-early-late'),**context)
    fusion = collect(a.fusion_runs,encoders=('phikon-v2',),methods=METHODS,**context)
    diagnostics = analyze(a.fusion_runs,encoders=('phikon-v2',),methods=METHODS,**context)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(render(initial,field,fusion,diagnostics))
    print(json.dumps({'verified_initial':initial['verified_runs'],'verified_field':field['verified_runs'],
                      'verified_fusion':fusion['verified_runs']}))


if __name__=='__main__':
    main()

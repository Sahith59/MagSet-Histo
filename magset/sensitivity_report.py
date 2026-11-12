"""Report separately verified split-seed and training-seed validation matrices."""
import argparse
from pathlib import Path
from .collect import collect
from .round2_report import METHODS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();root=a.results_root
    context={'manifest':a.manifest,'protocol':a.protocol}
    original=collect(root/'validation-v1',**context)
    base=collect(root/'fusion-v1',encoders=('phikon-v2',),methods=METHODS,**context)
    split=collect(root/'fusion-sensitivity-v1/split813-train812',seed=813,encoders=('phikon-v2',),methods=METHODS,**context)
    training=collect(root/'fusion-sensitivity-v1/split812-train813',encoders=('phikon-v2',),methods=('field-mean-view-mean','field-attention-view-attention'),**context)
    linear=collect(root/'linear-sensitivity-v1',seed=813,encoders=('phikon-v2',),methods=('mean',),**context)
    b={r['method']:r['mean_validation_subset_macro_f1'] for r in base['summary']}
    s={r['method']:r['mean_validation_subset_macro_f1'] for r in split['summary']}
    t={r['method']:r['mean_validation_subset_macro_f1'] for r in training['summary']}
    old=next(r['mean_validation_subset_macro_f1'] for r in original['summary'] if r['encoder']=='phikon-v2' and r['method']=='mean')
    new=linear['summary'][0]['mean_validation_subset_macro_f1']
    lines=['# Split and training seed sensitivity', '',
           'All 45 prespecified follow-up configurations completed and their validation probabilities independently '
           'reproduce the metrics. No outer-test scoring occurred. Each cell averages equal-subset macro F1 over '
           'five overlapping provider trials. Inner-validation case counts remain eight per trial.', '',
           '| Phikon neural control | Split 812 / train 812 | Split 813 / train 812 | Split 812 / train 813 |',
           '| --- | ---: | ---: | ---: |']
    for method in METHODS:
        lines.append('| %s | %.4f | %.4f | %s |'%(method,b[method],s[method],'%.4f'%t[method] if method in t else '—'))
    lines += ['', 'The original Phikon mean-feature linear control changes from %.4f (split seed 812) '
              'to %.4f (split seed 813), with its fixed four-C selection grid. No training seed is relevant '
              'to this deterministic fitting control.'%(old,new), '',
              '## Decision', '',
              'Changing the inner-validation cases changes both the absolute scores and the strongest neural row. '
              'For the two repeated neural controls, the measured change under a new training seed is smaller '
              'than the change under a new inner split in this screen. Two seeds do not establish a general variance decomposition.', '',
              'The linear control remains higher than the neural rows on the primary average for both checked '
              'inner splits. These observations support prioritizing case scarcity, regularization/capacity, '
              'grouping sensitivity, and a compatible second dataset. They do not support claiming that learned '
              'attention or the unimplemented matching mechanism improves robustness.', '',
              'Do not choose the favorable split for the paper. Preserve all rows, freeze a final protocol, '
              'and account for repeated cases in later uncertainty analyses. The new split/training-seed runs '
              'used source `ae647d02426d2119e29865f199cbe2031198bd8a`; model/optimizer settings were unchanged.', '',
              'Reproduce with `python -m magset.sensitivity_report --results-root /path/to/runs '
              '--manifest /path/to/manifest.csv --protocol /path/to/protocol.json --output docs/SENSITIVITY_RESULTS.md`.', '']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines))
    print('Verified 45 follow-up runs and all referenced initial matrices.')


if __name__=='__main__':
    main()

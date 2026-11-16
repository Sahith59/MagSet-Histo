"""Render the complete, hash-bound five-category classifier comparison."""
import argparse
import json
from pathlib import Path
import shutil

from magset.data import file_hash
from magset.five_collect import assert_public_safe
from magset.five_metrics import FINAL5


def render(d, digest):
    lines = ['# First trained five-category comparison', '',
             'November15,2026. **All ten paired runs and120 classifier fits completed and passed independent numerical review. The dual-step improvement gate failed.94–95% has not been achieved.**', '',
             'This compares direct benign/DC/LC/MC/PC classification with the requested binary benign/malignant router followed by a malignant-four-type specialist. Both use the same accepted frozen Phikon-v2 features, case assignments and four shared-C selection opportunities. It is provider-case development validation, not an independently untouched final test.', '',
             '| Model | Arrangement | Five-class accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in ('812', '813'):
        for arm in ('direct', 'dual'):
            v = d['summary'][seed][arm]['validation']
            lines.append('|%s|%s|%.2f%%|%.4f|%.2f%%|%.2f%%|' %
                         (arm, seed, 100*v['mean_subset_accuracy'], v['mean_subset_macro_f1'],
                          100*v['mean_subset_balanced_accuracy'], 100*v['all_view']['accuracy']))
    lines += ['', 'Accuracy/F1 average equally over all15 nonempty zoom settings and five overlapping trials. The all-four-view column is a different endpoint. The two arrangements choose different validation cases; they are not different datasets.', '',
              'Dual minus direct F1 is−0.0186/+0.0290 and accuracy−0.83/+4.17percentage points. The mixed result fails the prospectively fixed both-arrangement screen. Preserve both pipelines and every candidate; do not promote the favorable arrangement alone or claim that hierarchy reliably improves the task.', '',
              '## Where the remaining errors occur', '',
              '| Model | Arrangement | Binary accuracy | Malignant sensitivity | Benign specificity | True-malignant four-type accuracy |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in ('812', '813'):
        for arm in ('direct', 'dual'):
            v = d['summary'][seed][arm]['validation']; b = v['binary_mean_metrics']; m = v['malignant_mean_metrics']
            lines.append('|%s|%s|%.2f%%|%.2f%%|%.2f%%|%.2f%%|' %
                         (arm, seed, 100*b['accuracy'], 100*b['sensitivity'], 100*b['specificity'],
                          100*m['mean_subset_accuracy']))
    lines += ['', 'For dual, binary values describe the actual router. For direct, they are a separate derived sum-of-malignant-probabilities diagnostic, not the broad group of its final argmax. True-malignant typing evaluates the specialist/conditional subtype scores on known malignant cases; it does not count routing failures and is not whole-system accuracy.', '',
              'The trained dual router is reasonably useful, but malignant typing remains around56%. On these saved outputs, perfect routing with unchanged typing would give about77.8%/78.2% overall. This is an oracle diagnostic bound, not a new trained result. The evidence supports improving representation/case learning, without proving a single biological or preprocessing cause.', '',
              '## Every trial', '',
              '| Arrangement | Trial | Model | Selected C | Mean-subset accuracy | Macro F1 |',
              '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in d['runs']:
        for arm in ('direct', 'dual'):
            a = r['arms'][arm]; v = a['validation']
            lines.append('|%d|%d|%s|%g|%.2f%%|%.4f|' %
                         (r['seed'], r['trial'], arm, a['selected_C'], 100*v['mean_subset_accuracy'], v['mean_subset_macro_f1']))
    lines += ['', 'All four candidates, all15 subsets, all class confusions/recalls, fitting gaps and probability scores are retained in the aggregate ledger and independent review. Minimum-over-trial-and-subset F1 and mean-of-trial-minimum F1 are named separately.', '',
              '## Inputs, training and split', '',
              '- BreakHis only:7,909 images at40×/100×/200×/400×. Original labels map to pooled benign plus ductal, lobular, mucinous and papillary carcinoma.',
              '- Each run keeps46 fitting/eight validation/28 excluded provider cases. Original eight-class roles are reconstructed before pooling labels. The router sees46 fitting cases; the subtype specialist sees35 truly malignant fitting cases.',
              '- Five-class fitting counts are11/25/2/5/3; validation counts4/1/1/1/1. Thus raw accuracy and balanced accuracy differ; always-benign gives50% raw accuracy and20% balanced accuracy.',
              '- Frozen1,024-dimensional image features are averaged within zoom, standardized with fitting-only field moments, averaged over available zooms and joined with four flags. Both pipelines have5,145 trainable linear decision parameters; the shared image reader remains frozen.',
              '- Each head uses balanced classes and sample weight1/15per view combination. Heads use L2/lbfgs and C=[.01,.1,1,10]. Both hierarchy heads share a single C. Selection maximizes end-to-end mean15-subset five-class macro F1, with smaller-C ties.',
              '- Explicit hard routing≥.5defines the dual label; five-way composed-probability argmax is not substituted. Stable composed probabilities supply CE/Brier separately. Head weighting and factorized losses differ, so this is a complete-pipeline comparison.', '',
              '## Execution and independent acceptance', '',
              'Actual immutable source: `%s`. Config SHA256: `%s`.' % (d['code_sha'], d['config_sha256']), '',
              'Zero-fit audit4979472 completed0 in4seconds; all ten worker tasks4979473 completed0 in13/17/15/13/13/11/13/15/15/13seconds; collector4979474 completed0 in64seconds. At most four two-CPU workers ran concurrently. No new image-encoder forwards or outer-role predictions were performed. These are measured scheduler times, not promised future CNN timings.', '',
              'Independent review checked all80 candidates/120 heads/64,800 dependent fitting/validation case-subset records:4,924,986 assertions. All ten fitting-only normalizers matched bitwise; maximum coefficient replay difference1.78e-14 and metric difference4.44e-16; public summaries/gates matched exactly. All120 heads also met the declared normalized-gradient tolerance under the independently reconstructed weighted objective.147 pinned unit tests passed before launch.', '',
              'Comparison SHA256: `%s`.' % digest, '',
              '[Complete aggregate ledger](evidence/BREAKHIS_FIVE_CLASS_RESULTS.json) · [Independent actual-output review](reviews/FIVE_CLASS_ACTUAL_RESULTS_REVIEW.md) · [Registered recipe](BREAKHIS_FIVE_CLASS_SPEC.md).', '',
              '## Limits and next work', '',
              'There are22/20 distinct validation cases and31 in their union. The ten trials and15 subsets repeat cases;64,800 records are not64,800 patients. Clinical identity remains unresolved and all82 provider keys entered development somewhere. The28 excluded cases are excluded per run, not a globally fresh cohort. No clinical validity, novelty or reliable95% generalization is established.', '',
              'The supplied93.73%/94.99% notebooks measured eight-type40× individual images under image-random reused validation. The earlier45.7%/38.3% case scores were eight-class. Neither is a direct same-task baseline for this five-category result. The reference paper\'s86.22% is binary image accuracy, not five-class accuracy.', '',
              'The next proposed experiment trains an EfficientNetV2B0 case-bag representation with matched direct-five and dual-step arms. Its separate recipe/source/technical smokes must pass before production; it is not included in the results above and does not guarantee94–95%.']
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True); p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args(); digest = file_hash(a.comparison)
    if digest != a.expected_sha256:
        raise ValueError('Accepted comparison checksum differs')
    d = json.loads(a.comparison.read_text()); assert_public_safe(d)
    if (d['verified_runs'] != 10 or d['production_head_fits'] != 120
            or d['outer_predictions'] != 0 or tuple(d['class_names']) != FINAL5
            or d['code_sha'] != '0471cd4ab8fb1da71eba8870f2ea0fa59ce0e8f3'
            or d['config_sha256'] != 'd3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b'):
        raise ValueError('Complete accepted scientific context differs')
    a.output.parent.mkdir(parents=True, exist_ok=True); a.evidence.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render(d, digest)); shutil.copyfile(a.comparison, a.evidence)
    print('Rendered complete reviewed five-category report and unchanged aggregate ledger')


if __name__ == '__main__':
    main()

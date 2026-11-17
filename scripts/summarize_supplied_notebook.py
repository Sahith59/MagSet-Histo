"""Render a brief report only from the independent notebook checker output."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def render(value, expected_producer):
    lines = ['# Supplied notebook results', '',
             'Updated '+datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y, %I:%M %p %Z')+'.', '']
    if value is None:
        return '\n'.join(lines+['The checker has not produced verified results. Review the job logs; no final accuracy is available.', ''])
    if value.get('code_sha') != expected_producer:
        raise ValueError('Independent result belongs to a different notebook execution source')
    complete = value.get('status') == 'complete accepted supplied-notebook two-variant comparison'
    if not complete:
        lines += ['The two-run comparison is incomplete. No final paired accuracy is reported.', '']
        for name in ('faithful', 'input_fixed'):
            lines.append('- '+name+': '+value.get('variants', {}).get(name, {}).get('status', 'unavailable'))
        return '\n'.join(lines+['', 'The original notebooks and failure details remain preserved for review.', ''])
    lines += ['Both runs and their saved prediction scores passed the independent checker.', '',
              '| Run | Benign/malignant accuracy | Eight-type accuracy | Eight-type macro F1 | Selected epoch |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in ('faithful', 'input_fixed'):
        result = value['variants'][name]
        if result['status'] != 'accepted':
            raise ValueError('Complete comparison contains an unaccepted variant')
        metrics = result['original_code_metrics']
        for key in ('binary_accuracy', 'eight_class_accuracy', 'eight_class_macro_f1'):
            if not isinstance(metrics[key], (float, int)) or not 0 <= metrics[key] <= 1:
                raise ValueError('Invalid checked classification score')
        lines.append('| {} | {:.2f}% | {:.2f}% | {:.4f} | {} |'.format(
            name, 100*metrics['binary_accuracy'], 100*metrics['eight_class_accuracy'],
            metrics['eight_class_macro_f1'], result['selected_epoch']))
    audit = value['shared_audit']
    overlap = audit['provider_overlap']['provider_key']
    lines += ['', 'The model uses EfficientNetV2B0 to describe image regions, four Transformer blocks to compare them, '
              'and two output heads for benign/malignant and eight tissue types.', '',
              'Both runs use the same {:,} fitting images and {:,} checking images from BreakHis. '
              '{} provider keys appear on both sides. These are reused image-validation results; '
              'they do not replace the separate five-category provider-case benchmark.'.format(
                  audit['fitting_images'], audit['reused_validation_images'], overlap['overlap_keys']), '',
              'Faithful preserves the notebook input scaling. Input_fixed changes only that scaling; '
              'both retain the declared compatibility repairs and training settings.', '',
              'The notebook’s original malignant subtype score is ungated. The independently checked strict '
              'cascade score additionally requires a correct malignant first decision:', '']
    for name in ('faithful', 'input_fixed'):
        metrics = value['variants'][name]['original_code_metrics']
        lines.append('- {}: {:.2f}% strict malignant cascade recall.'.format(
            name, 100*metrics['strict_true_malignant_cascade_recall']))
    lines += ['', 'The visualization is a token-norm proxy rather than attention rollout. '
              'No clinical validation, novel-method claim or matched published-paper superiority follows from these numbers.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-producer', required=True)
    args = parser.parse_args()
    value = json.loads(args.comparison.read_text()) if args.comparison.exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(value, args.expected_producer))


if __name__ == '__main__':
    main()

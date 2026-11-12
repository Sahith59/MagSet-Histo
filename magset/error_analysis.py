"""Describe verified validation errors without inspecting outer-test outcomes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .collect import collect, verify_run
from .data import CLASS_NAMES


def analyze(root, **matrix):
    verified = collect(root,**matrix)
    groups = defaultdict(list)
    for path in sorted(Path(root).glob('*/metrics.json')):
        report, _ = verify_run(path)
        rows = [json.loads(line) for line in (path.parent/'predictions.jsonl').read_text().splitlines()]
        groups[(report['encoder'], report['method'])].append((report, rows))
    result = []
    for (encoder, method), runs in sorted(groups.items()):
        matrix = [[0]*8 for _ in range(8)]
        unique = [set() for _ in range(8)]
        bad_confidence = []
        harmed = helped = comparisons = 0
        by_case = defaultdict(lambda: [0, 0])
        for report, rows in runs:
            lookup = {}
            for row in rows:
                pred = max(range(8), key=lambda k: row['probabilities'][k])
                truth = row['truth']; key = (row['case_id'], tuple(row['magnifications']))
                correct = pred == truth
                lookup[key] = correct
                matrix[truth][pred] += 1
                unique[truth].add(row['case_id'])
                by_case[row['case_id']][0] += int(correct)
                by_case[row['case_id']][1] += 1
                if not correct:
                    bad_confidence.append(max(row['probabilities']))
            for (case, subset), correct in lookup.items():
                for scale in (40, 100, 200, 400):
                    if scale in subset:
                        continue
                    enlarged = tuple(sorted((*subset, scale)))
                    other = lookup[(case, enlarged)]
                    comparisons += 1
                    harmed += int(correct and not other)
                    helped += int(not correct and other)
        per_class = []
        for label, name in enumerate(CLASS_NAMES):
            total = sum(matrix[label])
            per_class.append({'class': name, 'correct_records': matrix[label][label],
                              'total_records': total, 'recall': matrix[label][label]/total,
                              'distinct_cases': len(unique[label])})
        cardinality = {}
        for count in range(1, 5):
            scores = [v['macro_f1'] for report, _ in runs
                      for s, v in report['validation']['subsets'].items()
                      if len(s.split('+')) == count]
            cardinality[str(count)] = sum(scores)/len(scores)
        subset_scores = {s: sum(r['validation']['subsets'][s]['macro_f1'] for r, _ in runs)/len(runs)
                         for s in runs[0][0]['validation']['subsets']}
        confusion = Counter({(i,j):matrix[i][j] for i in range(8) for j in range(8) if i != j})
        result.append({'encoder': encoder, 'method': method, 'class_recall': per_class,
                       'mean_macro_f1_by_view_count': cardinality,
                       'mean_macro_f1_by_subset': subset_scores,
                       'top_confusions': [{'truth': CLASS_NAMES[i], 'predicted': CLASS_NAMES[j], 'records': n}
                                          for (i,j), n in confusion.most_common(5)],
                       'view_addition': {'paired_comparisons': comparisons, 'correct_to_wrong': harmed,
                                         'wrong_to_correct': helped},
                       'mean_max_probability_on_errors': sum(bad_confidence)/len(bad_confidence) if bad_confidence else None,
                       'private_case_error_counts': dict(sorted(by_case.items()))})
    return {'status': 'verified validation error analysis', 'verified_runs': verified['verified_runs'],
            'interpretation': 'Repeated validation predictions are dependent; counts are descriptive, not independent samples.',
            'groups': result}


def markdown(report):
    lines = ['# Validation error analysis', '',
             ('Computed from all %d verified validation runs. '%report['verified_runs'])+'Outer-test predictions were not used. '
             'Each trial has eight validation cases, and trials can reuse a case. '
             'Repeated records below are descriptive; they do not increase the independent sample size.', '',
             '## Available views', '',
             'Each cell is mean macro F1 across subsets of that size and the five trials.', '',
             '| Encoder / fusion | One view | Two views | Three views | Four views |',
             '| --- | ---: | ---: | ---: | ---: |']
    for group in report['groups']:
        scores = group['mean_macro_f1_by_view_count']
        lines.append('| %s / %s | %s |' % (group['encoder'],group['method'],
                     ' | '.join('%.4f'%scores[str(k)] for k in range(1,5))))
    lines += ['', '## Subtype failures', '',
              'Recall is correct records / all records for that subtype across trials and subsets. '
              'The distinct-case count shows how little independent evidence supports each row.', '',
              '| Encoder / fusion | Subtype | Recall | Distinct validation cases |',
              '| --- | --- | ---: | ---: |']
    for group in report['groups']:
        for row in group['class_recall']:
            lines.append('| %s / %s | %s | %.3f | %d |' %
                         (group['encoder'],group['method'],row['class'],row['recall'],row['distinct_cases']))
    lines += ['', '## Adding a view', '',
              'Compare each subset with the same case plus exactly one additional magnification. '
              'These comparisons overlap. More available views need not improve an individual prediction.', '',
              '| Encoder / fusion | Correct → wrong | Wrong → correct | Comparisons |',
              '| --- | ---: | ---: | ---: |']
    for group in report['groups']:
        v = group['view_addition']
        lines.append('| %s / %s | %d | %d | %d |' %
                     (group['encoder'],group['method'],v['correct_to_wrong'],v['wrong_to_correct'],v['paired_comparisons']))
    lines += ['', '## What to test next', '',
              '1. Compare classification of individual fields followed by case-balanced probability aggregation '
              'with classification after averaging field features. This tests whether early averaging loses useful distinctions.',
              '2. Compare learned mean and gated-attention field pooling, then mean/gated/set-attention view fusion '
              'using identical inputs and validation budgets. These are established baselines, not novelty claims.',
              '3. Repeat split selection and training seeds before drawing conclusions. Keep the outer test held out '
              'and resolve grouping sensitivity before final evaluation.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--markdown', type=Path, required=True)
    a = p.parse_args(); report = analyze(a.runs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    a.markdown.parent.mkdir(parents=True, exist_ok=True)
    a.markdown.write_text(markdown(report))
    print(json.dumps({'verified_runs': report['verified_runs'], 'groups': len(report['groups'])}))


if __name__ == '__main__':
    main()

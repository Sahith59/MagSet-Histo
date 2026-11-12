"""Publish the BreakHis-first baseline table from trusted saved validation scores."""
import argparse
from collections import defaultdict
import json
import math
import re
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from magset.collect import SUBSETS, load_trusted_context, verify_run
from magset.data import CLASS_NAMES, MAGNIFICATIONS, file_hash
from magset.breakhis_predict import load_head


def verify_demo(d, selected_report, trusted, archive, head_path, expected_code_sha):
    if head_path is None:
        raise ValueError('A demo requires the actual saved head artifact')
    selected=sorted(selected_report['case_ids']['validation'])[0]
    case=next(c for c in trusted['cases'] if c['case_id']==selected)
    counts={str(m):len(case['indices'][m]) for m in MAGNIFICATIONS}
    model_sha=file_hash(head_path);load_head(head_path,model_sha,1024)
    if (not isinstance(expected_code_sha,str) or not re.fullmatch('[0-9a-f]{40}',expected_code_sha)
            or d['demo_code_sha']!=expected_code_sha
            or d['demo_code_sha256']!=file_hash(Path(__file__).resolve().parents[1]/'magset/breakhis_predict.py')):
        raise ValueError('Demo source differs from the expected immutable implementation')
    errors=[d.get(k) for k in ('raw_replay_max_probability_difference','cached_replay_max_probability_difference')]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in errors):
        raise ValueError('Invalid demo replay error scalar')
    if (d['status']!='raw-image validation replay verified' or d['dataset']!='BreakHis' or d['role']!='validation'
            or d['selection_rule']!='lexicographically first validation case' or d['case_id']!=selected
            or d['truth']!=CLASS_NAMES[case['label']] or d['encoder']!=selected_report['feature_provenance'][0]['encoder']
            or d['manifest_sha256']!=trusted['manifest_sha256'] or d['protocol_sha256']!=trusted['protocol_sha256']
            or d['trial']!=1 or d['seed']!=812 or d['magnifications']!=list(MAGNIFICATIONS)
            or tuple(d['class_names'])!=CLASS_NAMES or d['cached_validation_records_replayed']!=120
            or d['fields_by_view']!=counts or type(d['image_count']) is not int or d['image_count']!=sum(counts.values())
            or d['model_sha256']!=model_sha or d['new_classifier_fits']!=0 or d['outer_predictions_computed']!=0
            or errors[0]>1e-4 or errors[1]>1e-8
            or set(d['predictions'])!={'+'.join(map(str,s)) for s in SUBSETS}):
        raise ValueError('Raw-image replay does not satisfy the registered milestone checks')
    reference={'+'.join(map(str,r['magnifications'])):r['probabilities'] for r in archive if r['case_id']==selected}
    differences=[]
    for name,payload in d['predictions'].items():
        values=payload['probabilities']
        if (len(values)!=8 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 or x>1 for x in values)
                or not math.isclose(sum(values),1,abs_tol=1e-12,rel_tol=0)):
            raise ValueError('Invalid demo probability vector')
        error=max(abs(a-b) for a,b in zip(values,reference[name]));differences.append(error)
        stated=payload['archive_max_probability_difference']
        if (type(stated) not in (int,float) or not math.isfinite(stated) or stated<0
                or not math.isclose(error,stated,abs_tol=1e-12,rel_tol=0) or error>1e-4
                or payload['predicted_subtype']!=CLASS_NAMES[max(range(8),key=lambda k:values[k])]):
            raise ValueError('Demo labels/archive probability comparison differ')
    if not math.isclose(max(differences),errors[0],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Demo maximum error differs from independently recomputed value')


def build(root, manifest, protocol, markdown, ledger, demo=None, head=None, expected_demo_code_sha=None):
    root=Path(root);trusted=load_trusted_context(manifest,protocol);summaries=[];trial_rows=[];proof=[];seen=set();selected_report=None
    for seed in (812,813):
        runs=[];unique=set();class_correct=defaultdict(int);class_total=defaultdict(int);class_cases=defaultdict(set)
        for trial in range(1,6):
            folder=root/'matched-mean-v1'/('split%d'%seed)/('phikon-v2-mean-trial%d-seed%d'%(trial,seed))
            r,h=verify_run(folder/'metrics.json',trusted=trusted)
            if (r['encoder']!='phikon-v2' or r['method']!='mean' or r['trial']!=trial or r['seed']!=seed
                    or r.get('normalization_policy')!='field-moments-v2' or len(r['candidates'])!=4):
                raise ValueError('BreakHis milestone baseline differs from its fixed context')
            if (seed,trial)==(812,1):selected_report=r
            unique.update(r['case_ids']['validation']);seen.update(r['case_ids']['validation'])
            scores=r['validation']['subsets'];full=scores['40+100+200+400']
            row={'seed':seed,'trial':trial,'C':r['selected_C'],
                 'mean_subset_macro_f1':r['validation']['mean_subset_macro_f1'],
                 'mean_subset_accuracy':sum(v['accuracy'] for v in scores.values())/15,
                 'full_view_macro_f1':full['macro_f1'],'full_view_accuracy':full['accuracy'],
                 'worst_subset_macro_f1':r['validation']['worst_subset_macro_f1']}
            records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            for p in records:
                y=p['truth'];class_total[y]+=1;class_correct[y]+=int(max(range(8),key=lambda k:p['probabilities'][k])==y)
                class_cases[y].add(p['case_id'])
            runs.append(row);trial_rows.append(row);proof.append({'seed':seed,'trial':trial,**h,'code_sha':r['code_sha']})
        summaries.append({'seed':seed,'trials':5,'unique_validation_cases':len(unique),'validation_records':600,
                          **{k:sum(r[k] for r in runs)/5 for k in ('mean_subset_macro_f1','mean_subset_accuracy','full_view_macro_f1','full_view_accuracy')},
                          'class_recall':{CLASS_NAMES[k]:{'recall':class_correct[k]/class_total[k],
                            'record_count':class_total[k],'distinct_validation_cases':len(class_cases[k])} for k in range(8)}})
    demo_summary={'status':'raw-image demonstration pending'}
    if demo is not None:
        d=json.loads(Path(demo).read_text())
        archive_path=root/'matched-mean-v1/split812/phikon-v2-mean-trial1-seed812/predictions.jsonl'
        archive=[json.loads(line) for line in archive_path.read_text().splitlines()]
        verify_demo(d,selected_report,trusted,archive,head,expected_demo_code_sha)
        for key in ('metrics_sha256','predictions_sha256'):
            if d[key]!=next(p[key] for p in proof if p['seed']==812 and p['trial']==1):
                raise ValueError('Demo and reported trial source differ')
        demo_summary={k:d[k] for k in ('status','image_count','fields_by_view','cached_validation_records_replayed',
            'cached_replay_max_probability_difference','raw_replay_max_probability_difference','model_sha256',
            'demo_code_sha','demo_code_sha256','slurm_job_id','elapsed_seconds','new_classifier_fits','outer_predictions_computed')}
        demo_summary['private_demo_sha256']=file_hash(demo)
    report={'schema':'magset.breakhis-milestone.v1','dataset':'BreakHis','priority':'quick baseline deliverable before novelty/paper; TCGA follow-up deferred',
            'manifest_sha256':trusted['manifest_sha256'],'protocol_sha256':trusted['protocol_sha256'],
            'verified_configurations':10,'verified_validation_records':1200,'distinct_validation_cases_union':len(seen),
            'summary':summaries,'trials':trial_rows,'source_proof':proof,'demo':demo_summary,
            'report_builder_sha256':file_hash(__file__),
            'interpretation':'Exploratory eight-class provider-case validation; no globally pristine BreakHis confirmation cohort or verified clinical-patient claim; no novel method claim'}
    Path(ledger).write_text(json.dumps(report,indent=2)+'\n')
    lines=['# BreakHis: first checked baseline deliverable','',
           'Updated November 12, 2026. **BreakHis is the main dataset. TCGA follow-up is deferred.** This milestone packages existing baseline work; it does not require a novel method or a completed paper.','',
           '## What the model does','',
           'Give it the images of one provider case at available microscope zooms. A fixed Phikon-v2 image reader creates numerical descriptions. '
           'The saved mean-pooling and logistic classifier produce eight subtype probabilities. No exact cross-zoom field pairing is assumed.','',
           '## First numbers: available now','',
           '| Validation split | Mean F1 over 15 view settings | Mean case accuracy over 15 settings | All-view F1 | All-view case accuracy | Distinct validation cases |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append('| %d | %.4f | %.1f%% | %.4f | %.1f%% | %d |'%(r['seed'],r['mean_subset_macro_f1'],100*r['mean_subset_accuracy'],
                    r['full_view_macro_f1'],100*r['full_view_accuracy'],r['unique_validation_cases']))
    lines+=['','These are modest exploratory results. F1 is not accuracy. Every number averages five repeated provider trials; each trial has 46 fitting, 8 validation and 28 excluded outer-role cases. '
            'Each view setting has 40 case evaluations across trials, with repeated cases. Across both splits there are 31 distinct validation cases, not 1,200 independent patients.','',
            'A second split is shown to expose sensitivity, not to select the better split. The full table below prevents one favorable trial or demonstration from standing in for overall performance.','',
            '## Every trial','',
            '| Split | Trial | Selected C | Mean-subset F1 | Mean-subset accuracy | All-view F1 | All-view accuracy |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in trial_rows:
        lines.append('| %d | %d | %g | %.4f | %.1f%% | %.4f | %.1f%% |'%(r['seed'],r['trial'],r['C'],r['mean_subset_macro_f1'],
                      100*r['mean_subset_accuracy'],r['full_view_macro_f1'],100*r['full_view_accuracy']))
    lines+=['','## Class weaknesses','',
            'Recall below uses all saved case/subset records within each split; cases recur. These are descriptive counts, not independent clinical estimates.','',
            '| Subtype | Split 812 recall | Distinct cases | Split 813 recall | Distinct cases |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASS_NAMES:
        a,b=[r['class_recall'][name] for r in summaries]
        lines.append('| %s | %.3f | %d | %.3f | %d |'%(name,a['recall'],a['distinct_validation_cases'],b['recall'],b['distinct_validation_cases']))
    lines+=['','## Working prediction demonstration','']
    if demo is None:
        lines.append('The saved raw-image prediction path is being checked. No execution-success claim is made yet.')
    else:
        lines.append('Verified the existing trial1/seed812 saved head against all 120 archived validation records, then re-encoded every original image of the first sorted validation case and replayed all 15 view settings. '
                     'The case was chosen by metadata order, not by correctness. This demonstrates execution and parity, not new accuracy or a successful diagnosis.')
        lines.append('')
        lines.append('Images checked: **%d**. Maximum archived-probability difference: **%.3g** from cached replay and **%.3g** from raw images. '
                     'SLURM job `%s`; source `%s`. No new fitting or outer-role scoring.'%(demo_summary['image_count'],demo_summary['cached_replay_max_probability_difference'],
                     demo_summary['raw_replay_max_probability_difference'],demo_summary['slurm_job_id'],demo_summary['demo_code_sha']))
    lines+=['','## What is complete and what is not','',
            '- Verified ten baseline run reports and all 1,200 saved validation records against trusted labels, splits and metric arithmetic.',
            '- Data comprise 7,909 images at 40×/100×/200×/400× and eight subtypes. The primary input is a whole provider-case bag, not one independent patient per image.',
            '- All 82 provider cases have entered development somewhere. Clinical identifier relationships remain unresolved; no globally fresh BreakHis confirmation cohort remains.',
            '- These numbers do not establish a solved task, clinical readiness, a novel method or a completed research paper. The historical notebook image accuracies are from a different split/selection procedure and are not comparable targets.',
            '- Next: package this baseline, then run one registered BreakHis improvement comparison. Further TCGA work is off the immediate path.','',
            '[Focused plan](BREAKHIS_MILESTONE_PLAN.md) · [Training strategy](TRAINING_STRATEGY.md) · [Registered evaluation](BENCHMARK_PROTOCOL.md)','']
    Path(markdown).write_text('\n'.join(lines));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','manifest','protocol','markdown','ledger'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--demo',type=Path);p.add_argument('--head',type=Path)
    p.add_argument('--professor-brief',type=Path)
    p.add_argument('--expected-demo-code-sha');a=p.parse_args()
    r=build(a.runs,a.manifest,a.protocol,a.markdown,a.ledger,a.demo,a.head,a.expected_demo_code_sha)
    if a.professor_brief:
        lines=['# BreakHis project: first baseline for Professor Dong','',
               '**Current priority: finish the BreakHis project deliverable first. TCGA follow-up and novel-paper work are deferred.**','',
               'The model reads the available microscope images of one provider case and predicts one of eight breast-tumor subtypes. '
               'It uses a frozen Phikon-v2 picture reader and a trained mean-pooling/logistic decision maker. This is an established baseline, not a novel mechanism.','',
               '| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |',
               '| --- | ---: | ---: | ---: |']
        for row in r['summary']:
            lines.append('| %d | %.4f | %.1f%% | %.1f%% |'%(row['seed'],row['mean_subset_macro_f1'],
                         100*row['mean_subset_accuracy'],100*row['full_view_accuracy']))
        lines+=['','Ten saved baseline runs and 1,200 validation records passed trusted label/split/metric checks. '
                'The same cases recur across five provider holdouts and 15 view settings; there are 31 distinct validation cases across both splits. '
                'Each trial has 46 fit/8 validation/28 excluded cases. These are modest exploratory scores, not final test performance.','']
        if a.demo:
            d=r['demo'];lines.append('The raw-image replay passed on %d original images, covering all 15 view settings of the first sorted validation case. '
                'Maximum archived-probability difference: %.3g. The demonstration verifies execution/parity rather than improved accuracy. '
                'No new model fitting or outer-role scoring occurred.'%(d['image_count'],d['raw_replay_max_probability_difference']))
        else:lines.append('Raw-image validation replay is pending its cluster check. No claim that the prediction demonstration has passed is made yet.')
        lines+=['','**Next:** package the baseline, then register one bounded standard BreakHis improvement comparison. '
                'Working estimate: the next one or two work sessions for the new experiment and initial results, subject to implementation/queue. '
                'No positive gain or completion date is guaranteed.','',
                '**Open limits:** clinical relationships between some provider IDs remain unverified; all 82 cases entered development somewhere, '
                'so there is no globally fresh BreakHis confirmation cohort. Reliable final evaluation, method novelty and a submission-ready paper remain unfinished.','',
                '[Complete first results](../BREAKHIS_FIRST_RESULTS.md) · [Focused plan](../BREAKHIS_MILESTONE_PLAN.md) · '
                '[Training explanation](../TRAINING_STRATEGY.md).','']
        a.professor_brief.write_text('\n'.join(lines))
    print(json.dumps({'summary':r['summary'],'demo':r['demo']},indent=2))

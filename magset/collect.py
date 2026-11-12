"""Collect a complete validation matrix and independently verify saved score metrics."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
import random
from pathlib import Path
from .data import CLASS_NAMES, MAGNIFICATIONS, cases_from_rows, file_hash, read_manifest

SUBSETS = tuple(tuple(c) for n in range(1,5) for c in combinations(MAGNIFICATIONS,n))
ENCODERS = ('phikon-v2','resnet50')
METHODS = ('mean','concat','late')


def load_trusted_context(manifest, protocol):
    if manifest is None or protocol is None:
        raise ValueError('Trusted verification requires both manifest and protocol')
    mh=file_hash(manifest);source=json.loads(Path(protocol).read_text())
    if source['manifest_sha256']!=mh:
        raise ValueError('Trusted protocol and manifest disagree')
    cases=sorted(cases_from_rows(read_manifest(manifest)).values(),key=lambda c:c['case_id'])
    ids={c['case_id'] for c in cases}
    for trial in source['trials'].values():
        roles=trial['case_roles']
        if set(roles)!=ids or not set(roles.values()).issubset({'train','test'}):
            raise ValueError('Trusted protocol does not assign every known case')
    return {'manifest_sha256':mh,'protocol_sha256':file_hash(protocol),'unit':source['unit'],
            'cases':cases,'trials':source['trials'],'labels':{c['case_id']:c['label'] for c in cases}}


def verify_trusted_report(report, truths, trusted):
    for field in ('manifest_sha256','protocol_sha256','unit'):
        if report[field]!=trusted[field]:
            raise ValueError('Report differs from trusted metadata: '+field)
    roles=trusted['trials'][str(report['trial'])]['case_roles']
    rng=random.Random(report['seed']);validation=set()
    for label in range(8):
        eligible=sorted(c['case_id'] for c in trusted['cases'] if c['label']==label and roles[c['case_id']]=='train')
        if len(eligible)<2:
            raise ValueError('Trusted trial cannot support the declared inner split')
        validation.add(rng.choice(eligible))
    expected={'validation':validation,'train':{c for c,r in roles.items() if r=='train'}-validation,
              'test':{c for c,r in roles.items() if r=='test'}}
    if set(report['case_ids'])!=set(expected) or any(set(report['case_ids'][r])!=ids for r,ids in expected.items()):
        raise ValueError('Reported case assignments differ from trusted protocol/seed')
    if set(truths)!=validation or any(truths[c]!=trusted['labels'][c] for c in truths):
        raise ValueError('Prediction truth labels differ from trusted manifest')


def metrics_from_scores(records):
    matrix = [[0]*8 for _ in range(8)]
    for row in records:
        truth = row['truth']; scores = row['probabilities']
        if type(truth) is not int or truth not in range(8) or len(scores) != 8:
            raise ValueError('Invalid class/score vector')
        if not all(isinstance(s,(int,float)) and math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError('Invalid probability value')
        if not math.isclose(sum(scores),1,abs_tol=1e-5):
            raise ValueError('Probabilities do not sum to one')
        prediction = max(range(8),key=lambda k:scores[k])
        matrix[truth][prediction] += 1
    count = sum(map(sum,matrix))
    if not count:raise ValueError('No predictions')
    f1,recall = [],[]
    for label in range(8):
        tp = matrix[label][label];actual = sum(matrix[label]);predicted = sum(r[label] for r in matrix)
        f1.append(2*tp/(actual+predicted) if actual+predicted else 0.)
        if actual:recall.append(tp/actual)
    return {'macro_f1':sum(f1)/8,'accuracy':sum(matrix[i][i] for i in range(8))/count,
            'balanced_accuracy':sum(recall)/len(recall)}


def verify_run(path, trusted=None):
    path = Path(path); report = json.loads(path.read_text())
    if report['status'] != 'validation-only pilot' or 'test' in report:
        raise ValueError('Collector accepts validation-only pilot runs')
    if tuple(report['class_names']) != CLASS_NAMES:
        raise ValueError('Class order differs from the benchmark')
    ids = report['case_ids']
    groups = [set(ids[role]) for role in ['train','validation','test']]
    if (any(len(ids[r])!=len(set(ids[r])) for r in ('train','validation','test'))
            or [len(g) for g in groups] != [46,8,28] or any(groups[i]&groups[j] for i in range(3) for j in range(i))):
        raise ValueError('Case partition counts/overlap differ from the pilot')
    buckets = defaultdict(list);seen = set();truths = {}
    score_path = path.parent/'predictions.jsonl'
    for line in score_path.read_text().splitlines():
        row = json.loads(line);subset = tuple(row['magnifications']);key=(row['case_id'],subset)
        if row['role'] != 'validation' or row['case_id'] not in groups[1] or subset not in SUBSETS or key in seen:
            raise ValueError('Unexpected role/case/subset or duplicate score record')
        seen.add(key)
        if row['case_id'] in truths and truths[row['case_id']] != row['truth']:
            raise ValueError('Case truth changed across subsets')
        truths[row['case_id']] = row['truth'];buckets[subset].append(row)
    if set(buckets)!=set(SUBSETS) or any(len(v)!=8 for v in buckets.values()):
        raise ValueError('Expected eight validation cases for every subset')
    if trusted is not None:
        verify_trusted_report(report,truths,trusted)
    recomputed={}
    for subset,records in buckets.items():
        name='+'.join(map(str,subset));value=metrics_from_scores(records)
        for metric,score in value.items():
            if not math.isclose(score,report['validation']['subsets'][name][metric],abs_tol=1e-9):
                raise ValueError('Saved metric disagrees with score records: '+name+'/'+metric)
        recomputed[name]=value
    values=[v['macro_f1'] for v in recomputed.values()]
    mean=sum(values)/len(values);worst=min(values)
    if not math.isclose(mean,report['validation']['mean_subset_macro_f1'],abs_tol=1e-9) or not math.isclose(worst,report['validation']['worst_subset_macro_f1'],abs_tol=1e-9):
        raise ValueError('Saved aggregate differs from recomputed metrics')
    return report, {'metrics_sha256':file_hash(path),'predictions_sha256':file_hash(score_path)}


def collect(root, seed=812, allow_incomplete=False, encoders=ENCODERS, methods=METHODS, trials=(1,2,3,4,5), manifest=None, protocol=None):
    trusted=load_trusted_context(manifest,protocol) if manifest is not None or protocol is not None else None
    if any(not values or len(set(values)) != len(values) for values in (encoders,methods,trials)):
        raise ValueError('Matrix dimensions must be nonempty and unique')
    expected={(e,m,t,seed) for e in encoders for m in methods for t in trials}
    records={};provenance=[]
    for path in sorted(Path(root).glob('*/metrics.json')):
        report,hashes=verify_run(path,trusted=trusted)
        key=(report['encoder'],report['method'],report['trial'],report['seed'])
        if key not in expected or key in records:
            raise ValueError('Unexpected or duplicate benchmark configuration')
        records[key]=report;provenance.append({'run':path.parent.name,**hashes})
    missing=sorted(expected-set(records))
    if missing and not allow_incomplete:
        raise ValueError('Validation matrix incomplete: %d of %d runs missing'%(len(missing),len(expected)))
    for field in ['code_sha','manifest_sha256','protocol_sha256','unit','training_seed','config_sha256']:
        if len({r.get(field) for r in records.values()})>1:
            raise ValueError('Run provenance differs: '+field)
    for trial in trials:
        assignments={json.dumps({role:sorted(ids) for role,ids in r['case_ids'].items()},sort_keys=True)
                     for r in records.values() if r['trial']==trial}
        if len(assignments)>1:
            raise ValueError('Methods disagree on case membership within trial')
    summaries=[]
    # Incomplete collections are status reports, never partial method rankings.
    if not missing:
        for encoder in encoders:
            for method in methods:
                runs=[records[(encoder,method,t,seed)] for t in trials]
                summaries.append({'encoder':encoder,'method':method,'trials':len(trials),
                    'mean_validation_subset_macro_f1':sum(r['validation']['mean_subset_macro_f1'] for r in runs)/len(runs),
                    'minimum_trial_worst_subset_macro_f1':min(r['validation']['worst_subset_macro_f1'] for r in runs),
                    'mean_validation_full_view_macro_f1':sum(r['validation']['subsets']['40+100+200+400']['macro_f1'] for r in runs)/len(runs),
                    'selected_configuration_by_trial':[r.get('selected_configuration',{'C':r.get('selected_C')}) for r in runs]})
    return {'status':'complete validation-only pilot' if not missing else 'incomplete',
            'verified_runs':len(records),'expected_runs':len(expected),'seed':seed,'missing':missing,
            'matrix':{'encoders':encoders,'methods':methods,'trials':trials},
            'collector_code_sha256':file_hash(__file__),
            'verification_level':'trusted labels and regenerated split plus score arithmetic' if trusted else 'reported membership and score arithmetic only',
            'trusted_metadata':{k:trusted[k] for k in ('manifest_sha256','protocol_sha256')} if trusted else None,
            'summary':summaries,'artifact_hashes':provenance,
            'interpretation':'Inner-validation results from repeated holdouts; not outer-test evidence, independent-fold confidence intervals, or a clinical performance claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=812);p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--encoders',nargs='+',choices=ENCODERS,default=ENCODERS)
    p.add_argument('--methods',nargs='+',default=METHODS)
    p.add_argument('--trials',nargs='+',type=int,choices=range(1,6),default=(1,2,3,4,5))
    p.add_argument('--manifest',type=Path,required=True,help='Trusted manifest for label/split verification')
    p.add_argument('--protocol',type=Path,required=True,help='Trusted provider protocol')
    a=p.parse_args();report=collect(a.runs,a.seed,a.allow_incomplete,a.encoders,a.methods,a.trials,a.manifest,a.protocol)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','verified_runs','expected_runs','summary']},indent=2))


if __name__=='__main__':main()

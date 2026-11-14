"""Render the hash-bound complete supervised comparison without case identifiers."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public(value):
    if isinstance(value,dict):
        if set(value)&{'case_id','case_ids','case_draws','paths','relative_path','indices'}:
            raise ValueError('Private identifier in public results')
        for item in value.values():public(item)
    elif isinstance(value,list):
        for item in value:public(item)
    elif isinstance(value,str) and any(x in value for x in ('SOB_','/Users/','/data/users','/home/users')):
        raise ValueError('Private path or identifier in public results')
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite public value')


def build(v):
    rows=v['summary'];pairs=v['pairs']
    lines=['# BreakHis supervised ResNet-50 reference: negative result','',
        'November14,2026. **All ten runs and the complete artifact checker passed. The registered improvement gate failed. Keep the frozen Phikon-v2 reference.**','',
        'In plain language: teaching this standard picture model did not improve our main score. It learns the fitting cases very well but struggles with different validation cases. This result concerns one fixed pipeline; it does not establish that all supervised models fail or identify one sole cause.','',
        '## Main comparison','',
        'Scores average all15 input combinations and five overlapping trials per arrangement. Macro F1 treats all eight classes equally and is not accuracy.','',
        '| Arrangement | Frozen F1 | ResNet F1 | F1 change | Frozen case accuracy | ResNet case accuracy |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        a,b=r['frozen'],r['supervised']
        lines.append('| %d | %.4f | %.4f | %+.4f | %.1f%% | %.1f%% |'%(r['seed'],a['mean_subset_macro_f1'],b['mean_subset_macro_f1'],r['delta_mean_subset_macro_f1'],100*a['mean_subset_accuracy'],100*b['mean_subset_accuracy']))
    lines+=['','F1 decreases on both arrangements. Accuracy increases on813 but decreases on812; neither this mixed accuracy result nor an individual trial overrides the predeclared F1 gate. Phyllodes validation recall remains0 on both.','',
        '## Every paired trial','',
        '| Arrangement | Trial | Frozen F1 | ResNet F1 | Frozen accuracy | ResNet accuracy | Selected epoch |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in pairs:
        a,b=p['frozen'],p['supervised']
        lines.append('| %d | %d | %.4f | %.4f | %.1f%% | %.1f%% | %d |'%(p['seed'],p['trial'],a['mean_subset_macro_f1'],b['mean_subset_macro_f1'],100*a['mean_subset_accuracy'],100*b['mean_subset_accuracy'],p['selected_epoch']))
    lines+=['','## Fitting versus validation','',
        '| Arrangement | Fitting balanced case accuracy | Validation balanced case accuracy | Fitting weighted field accuracy | Validation weighted field accuracy |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        group=[p for p in pairs if p['seed']==r['seed']]
        mean=lambda f:sum(f(p) for p in group)/5
        values=[mean(lambda p:p['fitting']['mean_subset_balanced_accuracy']),r['supervised']['mean_subset_accuracy'],
                mean(lambda p:p['field_metrics']['train']['balanced']['accuracy']),mean(lambda p:p['field_metrics']['validation']['balanced']['accuracy'])]
        lines.append('| %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'%((r['seed'],)+tuple(100*x for x in values)))
    lines+=['','Field weights give equal influence to class, case within class and zoom within case; fields within a zoom are uniform. Case and field endpoints differ. Canonical evaluation cross-entropy also differs from augmented label-smoothed training loss. The large fitting–validation gap persists.','',
        '## Full-view, worst-view and probability scores','',
        '| Arrangement | Pipeline | All-view F1 | All-view accuracy | Mean trial worst-subset F1 | Brier error |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        for name in ('frozen','supervised'):
            a=r[name];lines.append('| %d | %s | %.4f | %.1f%% | %.4f | %.4f |'%(r['seed'],name,a['full_view_macro_f1'],100*a['full_view_accuracy'],a['worst_subset_macro_f1'],a['brier']))
    lines+=['','Brier is the mean sum of eight squared probability errors; lower is better. All-view inputs also remain weak.','',
        '## Subtype recall and distinct validation support','',
        '| Subtype | Frozen812 | ResNet812 | Frozen813 | ResNet813 | Distinct cases812 /813 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(v['class_names']):
        a,b=rows;values=[a['frozen']['class_recall'][i],a['supervised']['class_recall'][i],b['frozen']['class_recall'][i],b['supervised']['class_recall'][i]]
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f | %d / %d |'%((name.replace('_',' '),)+tuple(values)+(a['class_distinct_cases'][i],b['class_distinct_cases'][i])))
    lines+=['','Each fitting set has one adenosis and one phyllodes provider case. Repeated fields and augmentation do not add independent cases. Two distinct phyllodes validation cases per arrangement are a very small support, and the same cases recur across settings.','',
        '## All15 view combinations','',
        '| Available views | Frozen812 F1 | ResNet812 F1 | Frozen813 F1 | ResNet813 F1 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for subset in pairs[0]['supervised']['subsets']:
        values=[sum(p[name]['subsets'][subset]['macro_f1'] for p in pairs if p['seed']==seed)/5 for seed in (812,813) for name in ('frozen','supervised')]
        lines.append('| %s | %.4f | %.4f | %.4f | %.4f |'%((subset,)+tuple(values)))
    lines+=['','## Recipe, execution and verification','',
        'One ordinary complete-pipeline reference: ImageNet-initialized ResNet-50, two head-only epochs then18 full-backbone epochs, fixed BatchNorm running buffers, class/case/view-balanced fitting draws, mild fitting augmentation, AdamW and label smoothing. Whole fields are resized to224square. Keep the trained image head; average probabilities within zoom, then equally across available zooms. Select the earliest best validation F1 at epochs5/10/15/20; complete all20. [Registered specification](BREAKHIS_SUPERVISED_SPEC.md).','',
        'The same46 fitting/eight validation/28 excluded provider roles and all15 view subsets are retained. This is a complete-pipeline comparison, not a same-encoder fusion ablation, identical optimization budget or exact published-method reproduction. No outer images are scored.','',
        'Eight original tasks completed in14m16s–17m29s. Two timed out after2h00m18s. Both A40 recovery attempts failed the unchanged selected-field probability replay safeguard; their states were retained. Original pre-A40 committed checkpoints were restored after hash verification. A100 continuations finished in1m59s/3m24s and passed replay. No tolerance or scientific setting was relaxed. The collector completed in58seconds with an empty error log. [Detailed execution record](BREAKHIS_SUPERVISED_STATUS.md).','',
        'For the recovered cases, total active scheduled segments are7,406s and7,549s, including original timeout, rejected A40 attempt and final A100 continuation. The119s/204s final invocation is not their total training cost. Repeated partial work is disclosed, not an extra selected model. The model has23,524,424 parameters;16,392 train during head-only warmup.','',
        'The full checker verifies exact source/configuration/roles, canonical model tensors and BatchNorm buffers, AdamW moments/update budgets, all four selection archives and earliest-maximum choice, selected/last snapshots, complete field coverage, stable probability arithmetic and every saved case/subset metric. Independent actual-output review is recorded in the [review ledger](reviews/BREAKHIS_SUPERVISED_REVIEW.md); its exact scope and limitations govern any acceptance claim.','',
        'Training source `'+v['training_source_sha']+'`; collector source `f4ba03ae1278ac7ecf1897328353abd43444b1f3`; private complete comparison SHA256 `'+v['comparison_sha256']+'`. [Public aggregate/hash ledger](evidence/BREAKHIS_SUPERVISED_RESULTS.json) retains confusion matrices, per-subset metrics, field losses and all artifact hashes without provider identifiers.','',
        '## Decision and remaining work','',
        'The predeclared screen requires at least+0.02 mean-subset F1 on both arrangements and positive phyllodes recall on both. It failed. Keep the frozen reference and retain this negative result. Do not launch an automatic backbone/rate/epoch sweep, resplit to improve scores or score outer cases as a rescue.','',
        'These are exploratory provider-case validations:22/20 distinct validation cases,31 in their union, overlapping trials and reused development data. All82 cases entered development somewhere; clinical identifier relationships remain unresolved and no globally untouched BreakHis confirmation cohort remains. Neither reliable clinical performance nor a novel mechanism is established.','',
        'Finish the BreakHis project report with the checked baseline, both failed improvement recipes, fitting–validation evidence and protocol limits. Agree on the intended endpoint and credible final-evaluation route with Professor Dong before registering another model experiment. TCGA stays deferred.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('docs'))
    a=p.parse_args()
    if digest(a.comparison)!=a.comparison_sha256:raise ValueError('Complete comparison checksum differs')
    v=json.loads(a.comparison.read_text())
    if v['status']!='complete verified supervised reference' or v['verified_runs']!=10:
        raise ValueError('Complete accepted matrix required')
    if {(r['seed'],r['trial']) for r in v['pairs']}!={(s,t) for s in (812,813) for t in range(1,6)} or len(v['pairs'])!=10:
        raise ValueError('Exact ten-run coverage required')
    execution=[]
    original=[7218,7218,956,955,1001,1041,1049,1008,926,856]
    for h in v['artifacts']:
        path=a.artifacts/f"split{h['seed']}"/f"trial{h['trial']}"/'metrics.json'
        if digest(path)!=h['metrics_sha256']:raise ValueError('Individual report differs from complete checker')
        m=json.loads(path.read_text());task=(0 if h['seed']==812 else 5)+h['trial']-1
        segments=[original[task]]+({0:[69,119],1:[127,204]}.get(task,[]))
        execution.append({'seed':h['seed'],'trial':h['trial'],'selected_epoch':m['selected_epoch'],'model_parameters':m['model_parameters'],
            'final_gpu':m['gpu'],'final_invocation_reported_seconds':m['elapsed_seconds'],
            'active_scheduled_segments_seconds':segments,'total_active_scheduled_seconds':sum(segments),
            'outer_predictions_computed':m['outer_predictions_computed']})
    v=copy.deepcopy(v);v['comparison_sha256']=a.comparison_sha256;v['run_execution']=execution
    public(v)
    if v['promising_descriptive_gate'] is not False:raise ValueError('Renderer text requires the actual negative gate')
    a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'evidence').mkdir(exist_ok=True)
    (a.output_dir/'BREAKHIS_SUPERVISED_RESULTS.md').write_text(build(v))
    (a.output_dir/'evidence/BREAKHIS_SUPERVISED_RESULTS.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

"""Build a plain-language evidence brief and offline presentation from trusted results."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from magset.collect import load_trusted_context, verify_run
from magset.data import file_hash


def audit_results(root,manifest,protocol):
    trusted=load_trusted_context(manifest,protocol);records=[];fit=set();val=set();outer=set()
    for path in sorted(Path(root).glob('**/metrics.json')):
        r,h=verify_run(path,trusted=trusted)
        fit.update(r['case_ids']['train']);val.update(r['case_ids']['validation']);outer.update(r['case_ids']['test'])
        records.append({'path':str(path.relative_to(root)),**h})
    if len(records)!=175:raise ValueError('Meeting snapshot expects exactly175 development configurations')
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return {'verified_runs':len(records),'unique_validation_provider_cases':len(val),
            'provider_cases_ever_used_for_fit_or_validation':len(fit|val),
            'validation_cases_also_outer_in_another_trial':len(val&outer),
            'manifest_sha256':trusted['manifest_sha256'],'protocol_sha256':trusted['protocol_sha256'],
            'result_ledger_sha256':digest,'interpretation':'trusted labels/declared protocol verified; clinical identity and globally independent confirmation are not established',
            'records':records}


def plot_results(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='magset-professor-evidence-v1'
    import numpy as np
    methods=[('Matched linear mean','matched'),('Original attention','original'),('Smaller mean model','small'),('Higher-dropout attention','dropout')]
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.5),sharey=True)
    colors=['#147d64','#466a9f','#8b79a9','#bd7b2b']
    for ax,seed in zip(axes,(812,813)):
        for i,(label,policy) in enumerate(methods):
            if policy=='matched':folder=root/'matched-mean-v1'/('split%d'%seed);method='mean'
            elif policy=='original':
                folder=root/'fusion-v1' if seed==812 else root/'fusion-sensitivity-v1/split813-train812'
                method='field-attention-view-attention'
            else:
                folder=root/'capacity-v1'/policy/('split%d'%seed)
                method='field-mean-view-mean' if policy=='small' else 'field-attention-view-attention'
            values=[]
            for trial in range(1,6):
                p=folder/('phikon-v2-%s-trial%d-seed%d'%(method,trial,seed))/'metrics.json'
                values.append(json.loads(p.read_text())['validation']['mean_subset_macro_f1'])
            mean=float(np.mean(values));ax.bar(i,mean,color=colors[i],alpha=.85,width=.7)
            ax.scatter(i+np.linspace(-.17,.17,5),values,color='#172a3a',s=21,zorder=3)
            ax.text(i,mean+.035,'%.3f'%mean,ha='center',va='bottom',fontsize=10)
        ax.set_title('Inner split seed %d'%seed,fontweight='bold',fontsize=12)
        ax.set_xticks(range(4));ax.set_xticklabels([m[0].replace(' ','\n',1) for m in methods],fontsize=9)
        ax.set_ylim(0,1);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean eight-class macro F1 across 15 view subsets')
    fig.suptitle('Controlled development results: simple averaging remains a strong baseline',fontsize=14,fontweight='bold')
    fig.text(.5,.015,'Dots: five overlapping provider holdouts; not independent patients. Eight validation cases per trial. No confidence intervals shown.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(output/'matched-development.svg',bbox_inches='tight',metadata={'Date':None})
    fig.savefig(output/'matched-development.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--native-protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('docs/evidence'))
    p.add_argument('--audit-output',type=Path,required=True)
    p.add_argument('--native-cache-audit',type=Path)
    a=p.parse_args();audit=audit_results(a.results_root,a.manifest,a.protocol)
    native=json.loads(a.native_protocol.read_text());a.output.mkdir(parents=True,exist_ok=True)
    addendum_path=a.output/'NATIVE_PILOT_LEDGER.json'
    addendum=json.loads(addendum_path.read_text()) if addendum_path.exists() else None
    if addendum and (addendum['protocol_sha256']!=file_hash(a.native_protocol)
                     or addendum['verified_prediction_records']!=960 or addendum['sealed_predictions_computed']!=0):
        raise ValueError('Native development addendum differs from the registered protocol')
    if native['role_counts']!={'fit':80,'development':16,'sealed_test':32}:raise ValueError('Native role counts differ')
    a.audit_output.parent.mkdir(parents=True,exist_ok=True);a.audit_output.write_text(json.dumps(audit,indent=2)+'\n')
    plot_results(a.results_root,a.output)
    review_names=['VALIDITY_CRITIQUE','NOVELTY_CRITIQUE','DATA_PROVENANCE_CRITIQUE','INTEGRITY_FOLLOWUP','PROTOCOL_FOLLOWUP','MEETING_CLAIMS_FOLLOWUP']
    if not all(Path('docs/reviews',name+'.md').exists() for name in review_names):raise ValueError('Expected six completed review artifacts')
    brief=f'''# MagSet-Histo: evidence for Professor Dong

Prepared November 12, 2026. Working preparation target: November 18; the meeting date is not confirmed. This is a progress and decision package, not a claim of a completed novel paper.

## The project in plain language

A tissue sample looks different when the microscope zoom changes. We want a model to combine the available views and classify the tissue, even if a view is missing. The first task uses all four BreakHis magnifications and eight tumor subtypes.

## From the beginning to now

1. Reviewed the six supplied files. The historical 40x notebooks reported high image accuracy, but their split/model-selection procedure did not establish generalization to unseen patients.
2. Obtained and checked7,909 BreakHis images at40x/100x/200x/400x. Built reproducible manifests, pinned encoders, frozen feature caches, scheduler jobs, and saved predictions.
3. Completed175 development configurations, including averaging, concatenation, late fusion, field aggregation, gates, attention, two validation splits, and capacity/dropout checks.
4. Found that matched linear averaging remains ahead of the checked neural controls on both inner splits. These are noisy validation findings, not final test performance.
5. Ran three critique agents in two rounds: six review artifacts. Added trusted label/split checks and verified all175 saved sets. The same data still contain only{audit['unique_validation_provider_cases']} distinct validation provider cases; the number of runs is not the sample size.
6. Downloaded and decoded65,312 TCGA-MS images. Registered a new secondary16-class task with80 fit,16 development, and32 reserved patients before classifier training. Frozen ImageNet features have now been created for37,024 selected images; no classifier fitting or reserved-patient scoring has occurred.

## Evidence we can show now

![Matched development results](matched-development.svg)

| Checked inner split | Matched linear mean F1 | Highest neural-control mean F1 |
| --- | ---: | ---: |
|812|0.3843|0.3636|
|813|0.3169|0.2744|

All values average15 availability subsets over five overlapping provider holdouts, with eight validation cases per trial. They are macro F1, not accuracy. Input cases, encoder caches, and actual saved normalization arrays were matched. Every saved probability set was checked against trusted labels, regenerated split membership, and its reported arithmetic.

## What the critiques changed

- **BreakHis identity:** provider cases are disjoint within a trial, but conservative numeric stems can cross roles. Their clinical meaning is unverified. A strict three-way class-covered stem split is infeasible with only two adenosis stems.
- **Confirmation wording:** all{audit['provider_cases_ever_used_for_fit_or_validation']} provider cases have entered fitting or validation somewhere across development. We have not scored outer-role predictions; we do not have a globally pristine BreakHis confirmation cohort.
- **Encoder provenance:** Phikon-v2 pretraining includes TCGA. Exact slide overlap is unknown. TCGA is a secondary multi-organ task, not contamination-free external validation of the breast classifier. Frozen ImageNet ResNet is the first provenance control.
- **Novelty:** multiscale attention, missing masks, shared/specific features, and partial transport already have prior art. Splitting a mean into matched/unmatched parts does not create new information.

## Proposed contribution: a falsifiable hypothesis

Keep the strong mean features and add a small, fixed32-dimensional summary of cross-view agreement. Fit a regularized linear head rather than another large transformer. Ask whether the agreement summary improves missing-view classification beyond equally sized extra-feature controls.

The necessary controls are: mean alone, duplicated mean, within-bag variation, random partition, and simple cross-view affinity. Partial transport is optional only if cheaper agreement controls leave an unexplained benefit. This is a proposal; no method result or novelty proof is claimed.

**Go:** benefit survives the declared protocol, exceeds equal-dimension controls, and does not depend on one selected split. **No-go:** random/duplicate controls explain the effect, identity sensitivity removes it, or the second task contradicts it. Report negative evidence and narrow the contribution if necessary.

## Working plan to November18

|When|Deliverable / gate|
|---|---|
|Nov12|Trusted175-run audit, six critiques, registered80/16/32 protocol, evidence packet|
|Nov13-14|Verify frozen ImageNet caches; freeze a minimal native baseline budget and metric policy|
|Nov15-16|Algebra/invariance tests and the smallest candidate plus decisive control, if validity gates pass|
|Nov17|Consolidated findings, failure examples chosen by a stated rule, limitations and go/no-go|
|Nov18 target|Review evidence and choose contribution/data/protocol direction with Professor Dong|

This is a working sequence, not a booked meeting or guarantee of a positive result. No author/conference messages or submissions have been sent. No BRACS access is currently approved.

## Decisions for Professor Dong

1. How should uncertain BreakHis case identities be handled in the final scientific claim?
2. Should the main contribution be a narrow method hypothesis, a rigorous benchmark/robustness study, or a different task?
3. Can an independent breast cohort such as BRACS be obtained through approved access?
4. What evidence and venue scope would justify moving to final confirmation?

## Evidence links and limits

- [Matched baseline report](../CAPACITY_RESULTS.md), [split sensitivity](../SENSITIVITY_RESULTS.md), [comparison contract](../COMPARISON_PLAN.md).
- [Registered TCGA quarantine](../TCGA_QUARANTINE_PROTOCOL.md), [dataset audit](../TCGA_MS_AUDIT.md).
- [Validity critique](../reviews/VALIDITY_CRITIQUE.md), [novelty critique](../reviews/NOVELTY_CRITIQUE.md), [data provenance critique](../reviews/DATA_PROVENANCE_CRITIQUE.md).
- [Integrity follow-up](../reviews/INTEGRITY_FOLLOWUP.md), [protocol follow-up](../reviews/PROTOCOL_FOLLOWUP.md), [anticipated meeting questions](../reviews/MEETING_CLAIMS_FOLLOWUP.md).
- [Pinned Phikon-v2 card](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/README.md), [MagViT](https://arxiv.org/html/2608.16959v1), [M2OTCA](https://doi.org/10.1016/j.media.2026.104082).

Trusted result-ledger SHA256:`{audit['result_ledger_sha256']}`. Native manifest SHA256:`{native['manifest_sha256']}`. Test scoring, independent clinical generalization, and a submission-ready novel paper remain unfinished. Review agents are diagnostic aids, not scientific certification.
'''
    for old,new in [('checked7','checked 7'),('at40','at 40'),('Completed175','Completed 175'),('all175','all 175'),('decoded65','decoded 65'),('secondary16','secondary 16'),('with80','with 80'),(',16',', 16'),(',32',', 32'),('only31','only 31'),('values average15','values average 15'),('fixed32','fixed 32'),('November18','November 18'),('Trusted175','Trusted 175'),('registered80','registered 80'),('for37','for 37')]:brief=brief.replace(old,new)
    if addendum:
        values={a:addendum['arms'][a]['development']['mean_subset_macro_f1'] for a in ('mean','duplicate','random','affinity')}
        update=('## Later November 12 update: fixed native pilot completed\n\n'
                'The four-fit secondary 16-class development pilot is complete. Mean / duplicate / random / affinity mean-subset macro F1: '
                + ' / '.join('%.6f'%values[a] for a in ('mean','duplicate','random','affinity')) + '. '
                'Affinity does not beat the duplicate-feature control; this version does not support an agreement-specific improvement claim. '
                'All 960 prediction records were checked, and the 32 reserved patients remain unscored. '
                '[Full native results](../NATIVE_DEVELOPMENT_RESULTS.md) · [Plain-language project guide](../PROJECT_EXPLAINED.md).\n\n'
                'The following sections and eight-slide deck retain the earlier pre-native-pilot snapshot; the update above supersedes its pending-experiment wording.\n\n')
        title,body=brief.split('\n',1);brief=title+'\n\n'+update+body.lstrip()
    (a.output/'PROFESSOR_BRIEF.md').write_text(brief)
    slides=[
      ('A reliable prediction from incomplete microscope views','<p class="lead">Can a model combine the available views and classify tissue when some zoom levels are missing?</p><div class="flow"><b>40x</b><b>100x</b><b>200x</b><b>400x</b><span>→</span><b>8 subtype scores</b></div><p>MagSet-Histo · Evidence and next decision · Prepared November12,2026</p><p class="notice">Progress package for Professor Dong. November18 is an internal preparation target; meeting unconfirmed. No completed novel-method claim.</p>'),
      ('What we built, and what the counts mean','<div class="cards"><article><strong>6</strong><p>Original files reviewed</p></article><article><strong>7,909</strong><p>BreakHis images checked</p></article><article><strong>175</strong><p>Development configurations</p></article><article><strong>31</strong><p>Distinct validation provider cases</p></article></div><p class="lead">We built the experiment machinery and measured strong simple controls. Runs, fields, and mask combinations do not create new patients.</p><p>Frozen Phikon / ImageNet encoders · all15 missing-view subsets · saved scores and provenance · six critiques from three agents in two rounds.</p>'),
      ('The simple baseline is still hard to beat','<img class="chart" src="matched-development.svg" alt="Matched development results across two inner splits"><p class="small">Matched cases, feature caches, and normalization. Dots are overlapping provider holdouts. These selected validation averages are not final test estimates or significance claims.</p>'),
      ('Three findings changed the plan','<div class="columns"><article><h3>Identity uncertainty</h3><p>Provider cases and numeric stems differ. Strict class-covered three-way stem splitting is infeasible for adenosis.</p></article><article><h3>Global reuse</h3><p>All82 cases have entered fitting or validation somewhere. No outer-role predictions were scored, but there is no globally pristine BreakHis cohort.</p></article><article><h3>Pretraining overlap</h3><p>Phikon used TCGA-source data. TCGA is a secondary task, not clean external breast validation.</p></article></div><p class="notice">Trusted repair: all175 saved sets match actual labels and regenerated split membership. These checks do not resolve clinical identity.</p>'),
      ('Reserve confirmation patients before new training','<div class="split"><span style="flex:80;background:#d8eee7">80 fit</span><span style="flex:16;background:#f2e9ce">16 dev</span><span style="flex:32;background:#dfe6f4">32 sealed</span></div><div class="columns"><article><h3>Same16 classes</h3><p>5 fit +1 dev +2 sealed patients per class. Fixed hash ordering; no salt search.</p></article><article><h3>Fixed4 resolutions</h3><p>0.25 /0.5 /1.0 /2.0mpp. Six incomplete cases remain in fit only.</p></article><article><h3>Preparation only</h3><p>Frozen ImageNet features are allowed for every role. No classifier fitting or reserved-patient scoring yet.</p></article></div><p class="small">A new supervised quarantine, not source-protocol reproduction. Small dev/test sample sizes remain a limitation. Phikon is overlap-sensitive.</p>'),
      ('A small agreement correction: hypothesis only','<p class="lead">Keep the mean baseline. Add a compact summary of which field clues agree across available views.</p><div class="flow"><b>Mean + availability</b><span>+</span><b>Fixed32D agreement contrast</b><span>→</span><b>Regularized linear head</b></div><p>The original mean equals matched plus unmatched mass. Preserving both branches alone is not new information or proof of novelty.</p><p class="notice">Decisive controls: duplicated mean · within-bag variation · random partition · simple cross-view affinity. OT is optional only after these controls.</p>'),
      ('What would justify continuing?','<div class="columns"><article><h3>Continue</h3><p>Improvement exceeds the strongest mean and same-dimension controls under a fixed valid protocol, with all outcomes retained.</p></article><article><h3>Reconsider</h3><p>Random/duplicate controls explain the gain, grouping changes remove it, or another task contradicts the effect.</p></article><article><h3>Final confirmation</h3><p>Freeze method and metrics before accessing the32 reserved patients. Preserve patient dependence across15 subsets.</p></article></div><p class="lead">The next milestone is a defensible go/no-go decision. A new name, extra modules, or more runs does not establish a contribution.</p>'),
      ('Plan and questions for Professor Dong','<ol><li>Now: trusted audit, critics, registered quarantine and this packet.</li><li>Nov13-14: feature-readiness checks and a minimal fixed native baseline plan.</li><li>Nov15-17: candidate correctness, decisive control, failures and limitations.</li><li>Nov18 target: review the evidence and choose the scientific direction.</li></ol><p class="notice">Decisions: case-identity policy · method vs benchmark contribution · independent breast-data access · final evidence/venue scope.</p><p class="small">No BRACS approval yet. No final test result, clinical generalization, or submission-ready novel paper. See PROFESSOR_BRIEF.md and the six linked review files for evidence and primary sources.</p>')]
    sections=[]
    for i,(title,body) in enumerate(slides,1):
        sections.append(f'<section id="slide{i}"><div class="eyebrow">MAGSET-HISTO / RESEARCH REVIEW</div><h1>{html.escape(title)}</h1>{body}<footer><span>Development evidence · November12,2026</span><span>{i} / {len(slides)}</span></footer></section>')
    css='''*{box-sizing:border-box}body{margin:0;background:#e9edef;color:#162d3a;font:19px/1.45 Arial,sans-serif}nav{position:sticky;top:0;background:#142d3b;color:white;padding:12px 4vw;z-index:2;font-size:14px}nav a{color:white;margin-right:20px}section{position:relative;max-width:1200px;min-height:675px;margin:28px auto;padding:42px 55px 65px;background:white;box-shadow:0 5px 25px #102a3514;page-break-after:always}h1{font-size:38px;line-height:1.15;margin:12px 0 28px;max-width:1050px;letter-spacing:-.5px}h3{font-size:22px;margin:0 0 12px}p{margin:16px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:#527482;font-weight:bold}.lead{font-size:25px;line-height:1.4;max-width:1020px}.flow{display:flex;gap:15px;align-items:center;margin:35px 0;flex-wrap:wrap}.flow b{padding:18px;background:#e4f1ed;border:1px solid #b7d5cb;border-radius:9px;font-size:21px}.notice{padding:20px 24px;background:#edf3f6;border-left:5px solid #3d778b;font-size:18px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}.cards article{background:#eef4f1;padding:20px}.cards strong{font-size:47px;color:#147d64}.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}.columns article{background:#f3f6f7;padding:23px}.columns p{font-size:19px}.split{display:flex;margin:30px 0 40px;height:75px;text-align:center;gap:4px;font-weight:bold}.split span{display:flex;justify-content:center;align-items:center}.chart{width:100%;max-height:425px;object-fit:contain}.small{font-size:15px;color:#4d626d}footer{position:absolute;bottom:20px;left:55px;right:55px;display:flex;justify-content:space-between;font-size:12px;color:#6a7d85}ol{padding-left:28px}li{margin:18px 0}@media print{@page{size:landscape;margin:0}nav{display:none}body{background:white}section{margin:0;box-shadow:none;max-width:none;width:100%;height:100vh;min-height:0;padding:35px 50px 60px}h1{font-size:33px}.chart{max-height:390px}p,.columns p{font-size:17px}.lead{font-size:23px}}'''
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MagSet-Histo | Professor evidence review</title><style>'+css+'</style><nav><a href="#slide1">Start</a><a href="#slide3">Results</a><a href="#slide5">New protocol</a><a href="#slide8">Decision</a>Use arrow keys or scroll. Print for a portable copy.</nav>'+''.join(sections)+'''<script>document.addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','PageDown','PageUp'].includes(e.key))return;e.preventDefault();let a=[...document.querySelectorAll('section')],i=a.reduce((best,s,k)=>Math.abs(s.getBoundingClientRect().top)<Math.abs(a[best].getBoundingClientRect().top)?k:best,0);i=Math.max(0,Math.min(a.length-1,i+(['ArrowRight','PageDown'].includes(e.key)?1:-1)));a[i].scrollIntoView({behavior:'smooth'})})</script></html>'''
    for old,new in [('November12,2026','November 12, 2026'),('November18','November 18'),('all15','all 15'),('All82','All 82'),('all175','all 175'),('Same16','Same 16'),('Fixed4','Fixed 4'),('Fixed32D','Fixed 32D'),('the32','the 32'),('across15','across 15'),('Nov13','Nov 13'),('Nov15','Nov 15'),('Nov18','Nov 18')]:page=page.replace(old,new)
    (a.output/'PROFESSOR_DECK.html').write_text(page)
    public={k:v for k,v in audit.items() if k!='records'}
    public['native_manifest_sha256']=native['manifest_sha256'];public['native_protocol_sha256']=file_hash(a.native_protocol)
    if addendum:public['native_development_addendum_sha256']=file_hash(addendum_path)
    if a.native_cache_audit:
        cached=json.loads(a.native_cache_audit.read_text())
        if (cached['images']!=native['fields'] or cached['manifest_sha256']!=native['manifest_sha256']
                or cached['protocol_sha256']!=file_hash(a.native_protocol) or cached['classifier_predictions_computed']!=0):
            raise ValueError('Native cache attestation differs from registered protocol')
        public['native_feature_cache_audit_sha256']=file_hash(a.native_cache_audit)
        public['native_feature_images_verified']=cached['images']
    public['base_commit_at_render']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    public['artifact_builder_sha256']=file_hash(__file__)
    (a.output/'EVIDENCE_LEDGER.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()

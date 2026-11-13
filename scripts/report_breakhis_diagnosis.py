"""Render an accepted diagnosis as a report, identifier-free ledger and optional figure."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

LABELS={'frozen_case_head':'Frozen logistic case head','adapted_case_head':'Adapted logistic case head',
        'aux_mean_logits':'Auxiliary head, mean logits','aux_mean_probabilities':'Auxiliary head, mean probabilities'}


def public_copy(value):
    result=copy.deepcopy(value)
    for run in result['runs']:
        del run['case_ids']
        for role in run['case_scores'].values():
            for score in role.values():
                score['subsets']={name:{k:v[k] for k in ('accuracy','balanced_accuracy','macro_f1','class_balanced_cross_entropy')}
                                  for name,v in score['subsets'].items()}
    def safe(x):
        if isinstance(x,dict):
            if set(x)&{'case_id','case_ids','paths','relative_path'}:raise ValueError('Private identifiers in public ledger')
            for v in x.values():safe(v)
        elif isinstance(x,list):
            for v in x:safe(v)
        elif isinstance(x,str) and ('SOB_' in x or '/Users/' in x or '/data/users' in x):
            raise ValueError('Private path/identifier in public ledger')
    safe(result);return result


def build(value,sha,figure_name=None):
    rows=value['summary'];lines=['# BreakHis diagnosis: a large fitting–validation gap','',
        'November 13, 2026. All ten saved-model diagnoses completed. **The strongest finding is poor transfer from fitting cases to validation cases under the existing provider-case splits.** No model was trained or selected in this analysis.','',
        'In plain language: the model answers practice cases very well, but struggles with the cases left out of each run’s training. Keeping the image classifier or changing these two simple averaging orders does not fix the gap. This is consistent with overfitting/limited generalization; it does not establish one sole cause.','']
    if figure_name:lines+=['![Fitting versus validation accuracy](evidence/'+figure_name+')','',
        'Bars are descriptive five-trial means for each split. The panels have different prediction units; the image panel also equalizes case/zoom influence. Repeated trials are not independent clinical replications.','']
    lines+=['## Saved adapted case classifier','',
        'Balanced accuracy gives every subtype equal importance. Fitting contains46 cases with unequal class counts; validation contains eight cases, one per subtype. Scores average all15 view subsets.','',
        '| Split | Fitting balanced accuracy | Validation balanced accuracy | Validation macroF1 |',
        '| --- | ---: | ---: | ---: |']
    for r in rows:
        a=r['cases']['train']['adapted_case_head'];b=r['cases']['validation']['adapted_case_head']
        lines.append('| %d | %.1f%% | %.1f%% | %.4f |'%(r['seed'],100*a['mean_subset_balanced_accuracy'],100*b['mean_subset_balanced_accuracy'],b['mean_subset_macro_f1']))
    lines+=['','## Saved auxiliary image classifier','',
        'These are canonical, unaugmented cached pictures, with dropout disabled. Class/case/view-balanced field weights match expected training sampling mass. The field target is the inherited provider-case label, not an independently annotated diagnostic region.','',
        '| Split | Fitting weighted accuracy | Validation weighted accuracy | Fitting weighted CE | Validation weighted CE | Raw fitting /validation image accuracy |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        a=r['auxiliary_fields']['train']['class_case_view_balanced'];b=r['auxiliary_fields']['validation']['class_case_view_balanced']
        lines.append('| %d | %.1f%% | %.1f%% | %.3f | %.3f | %.1f%% / %.1f%% |'%(r['seed'],100*a['accuracy'],100*b['accuracy'],a['cross_entropy'],b['cross_entropy'],
            100*r['auxiliary_fields']['train']['raw']['accuracy'],100*r['auxiliary_fields']['validation']['raw']['accuracy']))
    lines+=['','These evaluation losses differ from augmented/dropout-on minibatch training loss. Validation weighted CE is worse than the analytic uniform eight-class probability reference, ln(8)≈2.079. Accuracy and confidence quality describe different failures. No calibration model is fitted.','',
        '## All fixed case readouts','',
        'Every readout uses the same allowed cases and all15 subsets. No winner is selected from these diagnostic probes.','',
        '| Split | Saved readout | Fitting balanced accuracy | Validation balanced accuracy | Validation F1 | All-view validation accuracy |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        for name,label in LABELS.items():
            a=r['cases']['train'][name];b=r['cases']['validation'][name]
            lines.append('| %d | %s | %.1f%% | %.1f%% | %.4f | %.1f%% |'%(r['seed'],label,100*a['mean_subset_balanced_accuracy'],100*b['mean_subset_balanced_accuracy'],b['mean_subset_macro_f1'],100*b['full_view_accuracy']))
    lines+=['','The auxiliary case averages do not provide a consistent rescue. Both saved image-level and case-level classifiers show a large fit–validation gap. Case aggregation has higher validation accuracy than the individual-field score here, but this is a descriptive change of prediction unit, not proof of a causal benefit.','',
        'For a linear head, averaging features before classification equals averaging logits after classification. The executed identity check passed. Consequently, these results do not establish that early averaging alone discarded a recoverable signal. Probability averaging is nonlinear, but these fixed probes remain weak.','',
        '## Subtype recall of the adapted case classifier','',
        '| Subtype | Fit812 | Validation812 | Fit813 | Validation813 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(value['class_names']):
        a,b=rows;v=[r['cases'][role]['adapted_case_head']['class_recall'][i] for r in (a,b) for role in ('train','validation')]
        lines.append('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'%((name.replace('_',' '),)+tuple(100*x for x in v)))
    lines+=['','Phyllodes and adenosis each have one fitting provider case per run. Phyllodes fits perfectly at the case level yet has0 validation recall. Repeated fields do not supply new independent disease examples. Papillary validation recall also remains weak. These observations implicate limited generalization, not a proven biological/labeling explanation.','',
        '## Feature geometry','',
        '| Split | Validation raw-field cosine | Validation raw relative L2 | Validation standardized view-mean cosine | Validation standardized relative L2 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        a=r['feature_drift']['validation']['raw_fields'];b=r['feature_drift']['validation']['own_normalized_view_means']
        lines.append('| %d | %.5f | %.5f | %.5f | %.5f |'%(r['seed'],a['cosine'],a['relative_l2'],b['cosine'],b['relative_l2']))
    lines+=['','All geometry comparisons had valid mass1. The directions remain similar; a cosine near1 is not a percentage of identical features and does not prove that classifier margins are unchanged. Standardized means use each head’s own saved fitting-only normalizer. Geometry does not isolate a cause of classification failure.','',
        '## Verification and scope','',
        'CPU job4968029 completed successfully in70seconds, using two CPU cores and no GPU. There were zero new fits, encoder forwards, raw-image decoding or outer predictions. Immutable diagnosis source `'+value['code_sha']+'`; input comparison SHA256 `'+value['comparison_sha256']+'`. Private diagnosis SHA256 `'+sha+'`.','',
        'Maximum archive replay error%.3g; auxiliary Torch probability error%.3g; affine-pooling logit error%.3g.'%tuple(max(r['replay'][k] for r in value['runs']) for k in ('archived_case_probability_error','auxiliary_torch_probability_error','affine_pooling_logit_error')),'',
        'The independent reviewer recomputes scores, role/path/label weights, both auxiliary case readouts and aggregate arithmetic from saved records. Raw geometry vectors remain on the cluster, so local geometry review checks provenance/bounds/summary arithmetic rather than independently redoing cosine/L2. [Independent review](reviews/BREAKHIS_DIAGNOSIS_REVIEW.md) · [Public aggregate/hash ledger](evidence/BREAKHIS_DIAGNOSIS_RESULTS.json) · [Fixed diagnostic specification](BREAKHIS_DIAGNOSIS_SPEC.md).','',
        '## Decision and next work','',
        'Retain the frozen baseline. The failed adaptation gate is unchanged; do not repeat this recipe for more epochs or choose a diagnostic winner. Before another registered training round, audit split/class coverage and define a compatible supervised image-model baseline with the same roles, class-aware weighting, case readout and15-subset metrics. A smaller end-to-end supervised reference is a hypothesis to test, not a promised fix.','',
        'Stay on BreakHis. All82 provider cases have entered development somewhere; only31 distinct validation cases recur across these arrangements, and clinical identifier relationships remain unresolved. This diagnosis is posthoc on reused development cases, not a fresh clinical test, significance claim, novelty result or a finished conference paper.']
    text='\n'.join(lines)+'\n'
    for old,new in {'contains46':'contains 46','all15':'all 15','has0':'has 0','mass1':'mass 1','job4968029':'job 4968029',
                    'in70seconds':'in 70 seconds','All82':'All 82','only31':'only 31','and15':'and 15'}.items():text=text.replace(old,new)
    return text


def figure(value,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=value['summary'];fig,axes=plt.subplots(1,2,figsize=(9,4.5),sharey=True)
    for ax,title,kind in zip(axes,('Saved case classifier','Saved image classifier'),('case','field')):
        x=[0,1]
        def score(r,role):
            return 100*(r['cases'][role]['adapted_case_head']['mean_subset_balanced_accuracy'] if kind=='case'
                else r['auxiliary_fields'][role]['class_case_view_balanced']['accuracy'])
        for offset,role,color,label in ((-.18,'train','#2563eb','Fitting'),(.18,'validation','#ea580c','Validation')):
            bars=ax.bar([v+offset for v in x],[score(r,role) for r in rows],width=.34,color=color,label=label)
            ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=10)
        ax.set_xticks(x,['Split 812','Split 813']);ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Balanced accuracy (%)')
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.78,wspace=.12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    fig.suptitle('BreakHis: fitting versus validation',fontsize=15,y=.97)
    prefix=Path(prefix);fig.savefig(prefix.with_suffix('.png'),dpi=300);fig.savefig(prefix.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--markdown',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--figure-prefix',type=Path)
    a=p.parse_args();raw=a.diagnosis.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.expected_sha:raise ValueError('Accepted diagnosis hash differs')
    d=json.loads(raw)
    if d['status']!='complete saved-artifact diagnosis' or d['diagnosed_runs']!=10 or any(d[k]!=0 for k in ('new_fits','new_encoder_forwards','outer_predictions')):
        raise ValueError('Incomplete or out-of-scope diagnosis')
    a.markdown.write_text(build(d,a.expected_sha,a.figure_prefix.with_suffix('.png').name if a.figure_prefix else None))
    a.ledger.write_text(json.dumps(public_copy(d),indent=2,allow_nan=False)+'\n')
    if a.figure_prefix:figure(d,a.figure_prefix)

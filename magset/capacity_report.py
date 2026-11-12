"""Verify matched inputs/normalization and report capacity controls without outer scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from .collect import collect, verify_run, load_trusted_context
from .round2_report import METHODS as ALL_METHODS

METHODS=('field-mean-view-mean','field-attention-view-attention')


def assert_matching_context(linear, neural, mean, std, neural_mean, neural_std):
    for field in ('encoder','trial','seed','manifest_sha256','protocol_sha256','class_names','feature_provenance'):
        if linear[field]!=neural[field]:
            raise ValueError('Comparison inputs differ: '+field)
    left={role:sorted(ids) for role,ids in linear['case_ids'].items()}
    right={role:sorted(ids) for role,ids in neural['case_ids'].items()}
    if left!=right:
        raise ValueError('Comparison case assignments differ')
    if not np.array_equal(mean,neural_mean) or not np.array_equal(std,neural_std):
        raise ValueError('Comparison normalization arrays differ')


def build(root,manifest=None,protocol=None):
    import torch
    root=Path(root); rows=[];proof=[]
    trusted=load_trusted_context(manifest,protocol) if manifest is not None or protocol is not None else None
    context={'manifest':manifest,'protocol':protocol}
    for seed in (812,813):
        linear_root=root/'matched-mean-v1'/('split%d'%seed)
        linear=collect(linear_root,seed=seed,encoders=('phikon-v2',),methods=('mean',),**context)
        rows.extend({'policy':'matched-linear','split_seed':seed,**r} for r in linear['summary'])
        original_root=root/'fusion-v1' if seed==812 else root/'fusion-sensitivity-v1/split813-train812'
        original=collect(original_root,seed=seed,encoders=('phikon-v2',),methods=ALL_METHODS,**context)
        for policy in ('original','small','dropout'):
            folder=original_root if policy=='original' else root/'capacity-v1'/policy/('split%d'%seed)
            report=original if policy=='original' else collect(folder,seed=seed,encoders=('phikon-v2',),methods=METHODS,**context)
            for method in METHODS:
                summary=next(r for r in report['summary'] if r['method']==method)
                parameter_counts=set()
                for trial in range(1,6):
                    name='phikon-v2-%s-trial%d-seed%d'%(method,trial,seed)
                    neural,hashes=verify_run(folder/name/'metrics.json',trusted=trusted)
                    linear_name='phikon-v2-mean-trial%d-seed%d'%(trial,seed)
                    lin,_=verify_run(linear_root/linear_name/'metrics.json',trusted=trusted)
                    if lin.get('normalization_policy')!='field-moments-v2' or neural['training_seed']!=812:
                        raise ValueError('Unexpected normalization or training-seed policy')
                    checkpoint=torch.load(folder/name/'checkpoint.pt',map_location='cpu',weights_only=True)
                    with np.load(linear_root/linear_name/'linear-model.npz',allow_pickle=False) as saved:
                        assert_matching_context(lin,neural,saved['mean'],saved['std'],
                            checkpoint['normalization_mean'].numpy(),checkpoint['normalization_std'].numpy())
                    reference,_=verify_run(original_root/name/'metrics.json',trusted=trusted)
                    expected=dict(reference['config'])
                    if policy=='small':expected['hidden']=32
                    if policy=='dropout':expected['dropout']=.5
                    if neural['config']!=expected:
                        raise ValueError('Capacity policy changed an unregistered setting')
                    if len(lin['candidates'])!=4 or len(neural['candidates'])!=4:
                        raise ValueError('Comparison selection budget differs')
                    parameter_counts.add(neural['parameters'])
                    proof.append({'policy':policy,'split_seed':seed,'trial':trial,'method':method,
                                  'normalization_arrays_identical':True,'inputs_cases_equal':True,
                                  'linear_code_sha':lin['code_sha'],'neural_code_sha':neural['code_sha'],**hashes})
                if len(parameter_counts)!=1:
                    raise ValueError('Parameter count changed across trials')
                rows.append({'policy':policy,'split_seed':seed,'parameters':parameter_counts.pop(),**summary})
    return {'status':'complete matched-input validation comparison','new_configurations':50,
            'referenced_neural_configurations':20,'comparisons_checked':len(proof),'summary':rows,'proof':proof,
            'interpretation':'Exploratory validation only; no independent-fold significance or final test claim.'}


def markdown(report):
    lines=['# Normalization-matched linear and capacity controls', '',
           'All 50 new configurations completed and their saved probabilities reproduce reported metrics. '
           'The comparison also references 20 previously completed neural configurations. Input caches, case '
           'assignments, and the actual saved normalization arrays match for every linear/neural comparison. '
           'Each setting receives four validation-selected candidates; no outer-test outcome is used.', '',
           '| Policy / method | Split seed | Parameters (neural) | Mean subset macro F1 | Full-view macro F1 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in report['summary']:
        lines.append('| %s / %s | %d | %s | %.4f | %.4f |'%(row['policy'],row['method'],row['split_seed'],
            str(row.get('parameters','—')),row['mean_validation_subset_macro_f1'],row['mean_validation_full_view_macro_f1']))
    lines += ['', 'The small policy changes only hidden width 128→32. The dropout policy changes only '
              'dropout 0.2→0.5. Other optimization/input settings remain fixed. Neural parameter counts '
              'differ intentionally; a proposed-mechanism comparison will additionally require matched-capacity controls.', '',
              'These remain noisy exploratory results: eight validation cases per trial, overlapping provider '
              'holdouts, and selected hyperparameters. A numerical lead does not establish generalization or novelty.', '']
    lines += ['## What this round establishes', '',
              '| Inner split seed | Matched linear mean | Highest tested neural average |',
              '| --- | ---: | ---: |']
    for seed in (812,813):
        subset=[r for r in report['summary'] if r['split_seed']==seed]
        linear=next(r for r in subset if r['policy']=='matched-linear')
        best=max((r for r in subset if r['policy']!='matched-linear'),key=lambda r:r['mean_validation_subset_macro_f1'])
        lines.append('| %d | %.4f | %.4f |'%(seed,linear['mean_validation_subset_macro_f1'],best['mean_validation_subset_macro_f1']))
    lines += ['', 'The matched linear mean remains higher than the tested neural controls on both inner splits. '
              'Reducing hidden width or increasing dropout did not close that descriptive gap. '
              'The normalization difference between earlier rounds no longer explains this comparison.', '',
              'Retain these results as an exploratory baseline reference. The next research work is to settle '
              'native second-dataset eligibility/metrics and pretraining overlap, complete compatible published-method '
              'comparisons, and specify a defensible candidate mechanism. More architecture trials alone do not '
              'establish a contribution. No outer-test evaluation or second-dataset model result is reported here.', '',
              'Experimental source revision: `ed8db0f49294cb3a19b4ee4e45f8e1a4c6064b08`. '
              'Regenerate using `python -m magset.capacity_report --runs-root /path/to/runs '
              '--manifest /path/to/manifest.csv --protocol /path/to/protocol.json --output /path/to/verified-comparison.json --markdown docs/CAPACITY_RESULTS.md`.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();report=build(a.runs_root,a.manifest,a.protocol)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    a.markdown.parent.mkdir(parents=True,exist_ok=True);a.markdown.write_text(markdown(report))
    print(json.dumps({k:report[k] for k in ('status','comparisons_checked','summary')},indent=2))


if __name__=='__main__':
    main()

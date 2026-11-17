"""Fixed-P binary diagnosis using fitting-group-out queries and saved validation."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_auc_score

SUBSETS=tuple(s for k in range(1,5)for s in combinations(range(4),k))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)

def group_support(cases,training,query):
    require(query in training,'Query must be a fitting case')
    group=cases[query]['group'];support=[i for i in training if cases[i]['group']!=group]
    require(support and query not in support and all(cases[i]['group']!=group for i in support),'Whole group exclusion failed')
    return support

def pool(audit,cases,indices,joint):
    lookup={int(i):p[1:].sum()for i,p in zip(indices,joint)};views=[]
    for c in cases:
        views.append([np.mean([lookup[f['index']]for f in audit['fields']if f['case_index']==c and f['view']==v])for v in range(4)])
    return np.stack([np.asarray(views)[:,s].mean(1)for s in SUBSETS],axis=1)

def summarize(y,p,edges=(0,.2,.4,.6,.8,1.)):
    correct=[];balanced=[];auc=[]
    for k in range(15):
        pred=p[:,k]>=.5;correct.append(float(np.mean(pred==y)));balanced.append(float(np.mean([np.mean(pred[y==label]==label)for label in (0,1)])));auc.append(float(roc_auc_score(y,p[:,k])))
    bins=[];yy=np.repeat(y,15);pp=p.ravel()
    for k,(low,high)in enumerate(zip(edges[:-1],edges[1:])):
        m=(pp>=low)&((pp<high)if k<len(edges)-2 else(pp<=high));bins.append({'low':float(low),'high':float(high),'events':int(m.sum()),'mean_probability':float(pp[m].mean())if m.any()else None,'malignant_fraction':float(yy[m].mean())if m.any()else None})
    clipped=np.clip(pp,1e-12,1-1e-12)
    return {'cases':len(y),'dependent_case_subset_events':len(pp),'mean_accuracy':float(np.mean(correct)),'mean_balanced_accuracy':float(np.mean(balanced)),'mean_roc_auc':float(np.mean(auc)),'brier':float(np.mean((pp-yy)**2)),'cross_entropy':float(-np.mean(yy*np.log(clipped)+(1-yy)*np.log1p(-clipped))),'calibration_bins':bins}

def main():
    p=argparse.ArgumentParser()
    for n in ('producer-source','parent','config','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Refusing existing diagnostic');a.output.mkdir(parents=True);cfg=json.loads(a.config.read_text());start=time.monotonic()
    require(sha(a.config)==sha(Path(__file__).resolve().parents[1]/'configs/breakhis-router-diagnosis-v1.json'),'Registered router config changed')
    result={'status':'router diagnosis failed','test_queries':0,'new_optimized_heads':0,'new_threshold_choices':0,'script_sha256':sha(__file__),'config_sha256':sha(a.config)}
    try:
        require(sha(a.parent/'checked/result.json')==cfg['parent_result_sha256'],'Parent result changed');parent=json.loads((a.parent/'checked/result.json').read_text())
        require((a.producer_source/'.source-sha').read_text().strip()==cfg['producer_code_sha'],'Frozen source changed')
        for n,h in parent['implementation_sha256'].items():require(sha(a.producer_source/n)==h,'Parent scientific source changed')
        audit_path=a.parent/'input-audit/private.json';require(sha(audit_path)==parent['input_audit_sha256'],'Parent audit changed');audit=json.loads(audit_path.read_text());sys.path.insert(0,str(a.producer_source.resolve()))
        from magset.bags import load_bags
        from magset import prior_centered_model as geometry
        from scipy.special import logsumexp
        parent_cfg=a.producer_source/'configs/breakhis-matched-three-way-v1.json';require(sha(parent_cfg)==parent['config_sha256'],'Frozen matched config changed');matched=json.loads(parent_cfg.read_text())
        coverage_path=a.producer_source/matched['midnight_p']['feature_config'];require(sha(coverage_path)==matched['midnight_p']['feature_config_sha256'],'Frozen P coverage registration changed');coverage=json.loads(coverage_path.read_text())
        for mag,pins in coverage['cache_provenance'].items():
            f=Path(audit['coverage_cache'])/('midnight-coverage-v1-'+mag+'.npz')
            require(sha(f)==pins['npz_sha256'] and sha(f.with_suffix('.json'))==pins['json_sha256'],'Pinned P feature cache changed')
        cases,bags,_=load_bags(audit['fields'],Path(audit['coverage_cache']),'midnight-coverage-v1',audit['manifest_sha256']);require([x['case_id']for x in cases]==[x['case_id']for x in audit['cases']],'Case order changed')
        train=audit['roles']['train'];labels=np.array([c['truth5']for c in audit['cases']]);train_p=[];counts=[]
        for q in train:
            group=audit['cases'][q]['group'];support=group_support(audit['cases'],train,q)
            require(q not in support and all(audit['cases'][i]['group']!=group for i in support),'Group query leaked into support')
            prepared=geometry.prepare_case(bags,labels,support,q,leave_query_case_out=False);logs=geometry.field_log_probabilities(prepared,cfg['density_temperature'],'dual','head_specific');lookup={}
            for v,x in enumerate(logs):
                joint=np.exp(x-logsumexp(x,axis=1,keepdims=True));keys=[f['index']for f in audit['fields']if f['case_index']==q and f['view']==v];lookup.update(zip(keys,joint))
            ix=np.array(sorted(lookup),np.int64);train_p.append(pool(audit,[q],ix,np.stack([lookup[int(i)]for i in ix]))[0]);counts.append(len(support))
            print(json.dumps({'training_queries_completed':len(train_p),'total':len(train)}),flush=True)
        train_p=np.stack(train_p)
        sel=a.parent/'models/midnight_p/selection.json';selection=json.loads(sel.read_text());require(sha(sel)==parent['models']['midnight_p']['selection_sha256'],'P selection changed')
        candidate=a.parent/'models/midnight_p/candidates'/('candidate%d'%selection['selected_index']);vp=candidate/'validation-fields.npz'
        require(sha(vp)==selection['candidate_artifact_sha256'][selection['selected_index']]['validation-fields.npz'],'Original validation archive changed')
        with np.load(vp,allow_pickle=False)as d:val_p=pool(audit,audit['roles']['validation'],d['field_indices'],d['joint_probabilities5'])
        yt=(labels[train]>0).astype(int);yv=(labels[audit['roles']['validation']]>0).astype(int)
        np.savez_compressed(a.output/'private-cases.npz',train_case_indices=np.array(train),train_group_out_malignant_probabilities=train_p,train_binary_truth=yt,validation_case_indices=np.array(audit['roles']['validation']),validation_malignant_probabilities=val_p,validation_binary_truth=yv)
        result.update(status='complete training-group-out and validation-only router diagnosis',parent_result_sha256=cfg['parent_result_sha256'],input_audit_sha256=sha(audit_path),fixed_temperature=cfg['density_temperature'],training_group_count=len({audit['cases'][i]['group']for i in train}),support_case_count_range=[min(counts),max(counts)],whole_group_exclusion=True,training_group_out=summarize(yt,train_p,cfg['calibration_bins']),validation=summarize(yv,val_p,cfg['calibration_bins']),elapsed_seconds=time.monotonic()-start,private_array_sha256=sha(a.output/'private-cases.npz'))
    except Exception as e:
        import traceback
        result.update(error_type=type(e).__name__);(a.output/'private-traceback.txt').write_text(traceback.format_exc())
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));return 0 if result['status'].startswith('complete ')else 1

if __name__=='__main__':raise SystemExit(main())

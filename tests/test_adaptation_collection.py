import copy
import json
import unittest
import numpy as np
try:
    import torch
    from magset.collect_adaptation import verify_budget,check_features,digest_state,replay_head,fragment_schema
    from test_adaptation import TinyReader,rows_and_cases,adapt
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Pinned PyTorch required for adaptation artifacts')
class AdaptationCollectionTests(unittest.TestCase):
    def fixture(self):
        rows,cases=rows_and_cases()
        config={'epochs':2,'head_only_epochs':1,'encoder_layers':2,'dimension':4,'batch_size':8,'smoke_epochs':2,'smoke_batches_per_epoch':2,
                'encoder_learning_rate':.00001,'head_learning_rate':.001,'weight_decay':.01}
        state={k:torch.zeros(shape) for k,shape in fragment_schema(config).items()};names=set(state)
        blocks={n for n in names if n.startswith('encoder.')};norms=names-blocks
        subset=lambda keys:{k:v for k,v in state.items() if k in keys}
        proof={'frozen_prefix_sha256':'b'*64,'allowed_before_sha256':digest_state(state),
               'base_last_block_sha256':digest_state(subset(blocks)),'base_final_norm_sha256':digest_state(subset(norms)),
               'initial_base_feature_max_difference':0.,'encoder_reload_max_difference':0.}
        # Construct a final-artifact fixture without training any model.
        for value in state.values():value.add_(.01)
        proof.update(allowed_after_sha256=digest_state(state),encoder_trainable_names=sorted(names),
            encoder_trainable_parameters=sum(v.numel() for v in state.values()))
        common={'loss':1.,'draws':32,'class_draws':[4]*8,'case_draws':{c['case_id']:4 for c in cases},
                'view_draws':{str(m):8 for m in (40,100,200,400)}}
        proof['history']=[{**copy.deepcopy(common),'epoch':1,'encoder_adapting':False,
                           'last_block_sha256':proof['base_last_block_sha256'],'final_norm_sha256':proof['base_final_norm_sha256']},
                          {**copy.deepcopy(common),'epoch':2,'encoder_adapting':True,
                           'last_block_sha256':digest_state(subset(blocks)),'final_norm_sha256':digest_state(subset(norms))}]
        checkpoint={'epoch':2,'history':copy.deepcopy(proof['history']),'encoder_fragment':state,
                    'frozen_prefix_sha256':proof['frozen_prefix_sha256'],'base_allowed_sha256':proof['allowed_before_sha256']}
        head={'1.weight':torch.zeros(8,4),'1.bias':torch.zeros(8)};checkpoint['auxiliary_head']=head
        optimizer=torch.optim.AdamW([{'params':[torch.nn.Parameter(v) for v in state.values()],'lr':.00001},
                                    {'params':[torch.nn.Parameter(v) for v in head.values()],'lr':.001}],weight_decay=.01)
        opt=optimizer.state_dict();n=len(state)
        opt['state']={i:{'step':torch.tensor(4. if i<n else 8.),'exp_avg':torch.zeros_like(v),'exp_avg_sq':torch.zeros_like(v)}
                      for i,v in enumerate(list(state.values())+list(head.values()))}
        checkpoint['optimizer']=opt
        return proof,checkpoint,config,rows,cases

    def test_exact_budget_warmup_and_fragment_checked_independently(self):
        args=self.fixture();verify_budget(*args)
        for mutation in ('draws','phase','warmup','loss','case','fragment'):
            proof,checkpoint,cfg,rows,cases=copy.deepcopy(args)
            if mutation=='draws':proof['history'][1]['draws']-=1
            if mutation=='phase':proof['history'][0]['encoder_adapting']=True
            if mutation=='warmup':proof['history'][0]['last_block_sha256']='a'*64
            if mutation=='loss':proof['history'][1]['loss']=float('nan')
            if mutation=='case':proof['history'][1]['case_draws']['outer-case']=4
            if mutation=='fragment':next(iter(checkpoint['encoder_fragment'].values())).flatten()[0]=float('nan')
            # Keep claimed history internally consistent, then require independent policy rejection.
            checkpoint['history']=copy.deepcopy(proof['history'])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):verify_budget(proof,checkpoint,cfg,rows,cases)

    def test_optimizer_setting_steps_and_truncated_schema_rejected(self):
        args=self.fixture()
        for mutation in ('learning_rate','steps','missing_state','schema'):
            proof,checkpoint,cfg,rows,cases=copy.deepcopy(args)
            if mutation=='learning_rate':checkpoint['optimizer']['param_groups'][0]['lr']=99.
            if mutation=='steps':checkpoint['optimizer']['state'][0]['step']+=1
            if mutation=='missing_state':checkpoint['optimizer']['state']={}
            if mutation=='schema':
                # Rehash a truncated state coherently; prefix-only checking would accept it.
                state=checkpoint['encoder_fragment'];key=next(iter(state));del state[key]
                proof['encoder_trainable_names']=sorted(state);proof['encoder_trainable_parameters']=sum(v.numel() for v in state.values())
                proof['allowed_after_sha256']=digest_state(state)
                proof['history'][-1]['last_block_sha256']=digest_state({k:v for k,v in state.items() if k.startswith('encoder.')})
                checkpoint['history']=copy.deepcopy(proof['history'])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):verify_budget(proof,checkpoint,cfg,rows,cases)

    def test_outer_paths_duplicate_coverage_and_nonfinite_features_rejected(self):
        rows,_=rows_and_cases();fields={'paths':np.array([r['relative_path'] for r in rows]),'features':np.ones((32,4),np.float32)}
        check_features(fields,rows,4)
        for mutation in ('outer','duplicate','nan','dtype'):
            bad=copy.deepcopy(fields)
            if mutation=='outer':bad['paths'][0]='outer.png'
            if mutation=='duplicate':bad['paths'][1]=bad['paths'][0]
            if mutation=='nan':bad['features'][0,0]=float('nan')
            if mutation=='dtype':bad['features']=bad['features'].astype(np.float64)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):check_features(bad,rows,4)

    def test_normalization_and_probabilities_replayed_without_refitting(self):
        import tempfile
        from pathlib import Path
        rng=np.random.default_rng(5);cases=[];paths=[];features=[]
        labels=[i for i,n in enumerate([1,25,6,2,5,3,1,3]) for _ in range(n)]+list(range(8))
        for i,label in enumerate(labels):
            case={'case_id':'c%d'%i,'label':label,'indices':{}}
            for m in (40,100,200,400):
                p='c%d/%d'%(i,m);case['indices'][m]=[p];paths.append(p)
                features.append(np.eye(8)[label]+rng.normal(0,.02,8))
            cases.append(case)
        fields={'paths':np.array(paths),'features':np.asarray(features,np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);adapt.fit_case_head(cases,fields,folder)
            records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            checksum=adapt.file_hash(folder/'linear-model.npz')
            self.assertLess(replay_head(folder,cases,fields,records,checksum),1e-8)
            bad=copy.deepcopy(records);bad[0]['probabilities']=np.roll(bad[0]['probabilities'],1).tolist()
            with self.assertRaises(ValueError):replay_head(folder,cases,fields,bad,checksum)
            altered=copy.deepcopy(fields);altered['features'][0,0]+=1
            with self.assertRaises(ValueError):replay_head(folder,cases,altered,records,checksum)


if __name__=='__main__':unittest.main()

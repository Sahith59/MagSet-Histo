import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
try:
    import torch
    from torch import nn
    import magset.collect_supervised as collect
    import magset.supervised_breakhis as sup
    from test_supervised_breakhis import TinyCNN
    import test_supervised_breakhis as fixture_module
    from test_adaptation import rows_and_cases
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Pinned Torch required for supervised artifact checks')
class SupervisedCollectionTests(unittest.TestCase):
    def fixture(self):
        cfg=fixture_module.SupervisedTests().config();cfg['dimension']=4;rows,cases=rows_and_cases()
        m=TinyCNN();original=copy.deepcopy(m);original.fc=nn.Linear(4,1000);original=original.state_dict()
        state=copy.deepcopy(m.state_dict());initial={k:collect.fingerprint(original,k) for k in ('bn','backbone','early','late')}
        state['conv1.weight']+=.01;state['layer4.0.weight']+=.01
        history=[]
        for epoch in range(1,4):
            h={'epoch':epoch,'loss':1.,'draws':16,'class_draws':[2]*8,'case_draws':{c['case_id']:2 for c in cases},
               'view_draws':{str(m):4 for m in (40,100,200,400)},'backbone_adapting':epoch>2,
               'learning_rates':list(sup.learning_rates(cfg,epoch)),'bn_sha256':initial['bn'],
               'early_sha256':initial['early'] if epoch<=2 else collect.fingerprint(state,'early'),
               'late_sha256':initial['late'] if epoch<=2 else collect.fingerprint(state,'late')};history.append(h)
        backbone=[k for k in state if not k.startswith('fc.') and not k.endswith(('running_mean','running_var','num_batches_tracked'))];head=['fc.1.weight','fc.1.bias'];n=len(backbone)
        opt=torch.optim.AdamW([{'params':[nn.Parameter(state[k]) for k in backbone],'lr':cfg['backbone_learning_rate']},
                               {'params':[nn.Parameter(state[k]) for k in head],'lr':cfg['head_learning_rate']}],weight_decay=cfg['weight_decay']).state_dict()
        opt['state']={i:{'step':torch.tensor(2. if i<n else 6.),'exp_avg':torch.zeros_like(state[k]),'exp_avg_sq':torch.zeros_like(state[k])} for i,k in enumerate(backbone+head)}
        proof={'initial':initial,'history':history,'candidates':[],'best':None,'reload_logit_max_difference':0.,
               **{'final_%s_sha256'%k:collect.fingerprint(state,k) for k in ('bn','early','late')}}
        saved={'model':state,'optimizer':opt,'epoch':3,'history':copy.deepcopy(history),'candidates':[],'best':None,'initial':initial}
        return {'proof':proof},saved,original,cfg,rows,cases

    def test_corrupted_bn_steps_schema_and_trace_rejected(self):
        args=self.fixture();collect.check_budget(*args,True)
        for mutation in ('bn','step','lr','shape','phase','nan_loss'):
            report,saved,original,cfg,rows,cases=copy.deepcopy(args)
            if mutation=='bn':saved['model']['bn.running_mean']+=1
            if mutation=='step':saved['optimizer']['state'][0]['step']+=1
            if mutation=='lr':saved['optimizer']['param_groups'][0]['lr']=99.
            if mutation=='shape':saved['model']['fc.1.weight']=torch.zeros(7,4)
            if mutation=='phase':report['proof']['history'][0]['backbone_adapting']=True
            if mutation=='nan_loss':report['proof']['history'][0]['loss']=float('nan')
            saved['history']=copy.deepcopy(report['proof']['history'])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):collect.check_budget(report,saved,original,cfg,rows,cases,True)

    def test_nan_metric_and_missing_keys_rejected(self):
        collect.equal_metric({'score':[1.,2.]},{'score':[1.,2.]})
        for value in ({'score':[float('nan'),2.]},{'score':[1.]},{'different':[1.,2.]}):
            with self.assertRaises(ValueError):collect.equal_metric({'score':[1.,2.]},value)

    def test_field_path_and_log_probability_integrity(self):
        rows=[{'relative_path':'a'},{'relative_path':'b'}];z=np.array([[1000.,0,0,0,0,0,0,0],[0.,2,3,4,5,6,7,8]])
        from scipy.special import logsumexp
        lp=z-logsumexp(z,axis=1,keepdims=True);values={'paths':np.array(['a','b']),'logits':z,'log_probabilities':lp,'probabilities':np.exp(lp)}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.npz';np.savez(p,**values);collect.load_fields(p,rows)
            for mutation in ('path','nan','log'):
                bad=copy.deepcopy(values)
                if mutation=='path':bad['paths'][0]='x'
                if mutation=='nan':bad['probabilities'][0,0]=float('nan')
                if mutation=='log':bad['log_probabilities'][0,0]+=1
                np.savez(p,**bad)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):collect.load_fields(p,rows)


if __name__=='__main__':unittest.main()

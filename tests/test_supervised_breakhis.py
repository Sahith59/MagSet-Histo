import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
try:
    import torch
    from torch import nn
    import magset.supervised_breakhis as sup
    from test_adaptation import rows_and_cases
except ImportError:
    torch=None

if torch is not None:
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__();self.conv1=nn.Conv2d(3,4,1);self.bn=nn.BatchNorm2d(4)
            self.layer4=nn.Sequential(nn.Conv2d(4,4,1),nn.ReLU());self.fc=nn.Sequential(nn.Dropout(.3),nn.Linear(4,8))
        def forward(self,x):return self.fc(self.layer4(self.bn(self.conv1(x))).mean((2,3)))


@unittest.skipIf(torch is None,'Pinned Torch needed for supervised reference checks')
class SupervisedTests(unittest.TestCase):
    def config(self):
        return {'epochs':4,'head_only_epochs':2,'selection_epochs':[3,4],'smoke_epochs':3,'smoke_batches_per_epoch':2,
                'backbone_learning_rate':.01,'head_learning_rate':.02,'minimum_learning_rate_fraction':.1,
                'weight_decay':.0001,'training_seed':812,'batch_size':8,'image_size':4,'gradient_clip':1.,'label_smoothing':.1}

    def test_warmup_and_bn_buffers_stay_frozen_then_backbone_updates(self):
        torch.manual_seed(1);m=TinyCNN();before={k:sup.state_hash(m,k) for k in ('bn','backbone','early','late')}
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        for adapting in (False,True):
            sup.set_phase(m,adapting);self.assertFalse(m.bn.training);self.assertTrue(m.fc.training)
            opt.zero_grad(set_to_none=True);nn.functional.cross_entropy(m(torch.randn(8,3,4,4)),torch.arange(8)).backward();opt.step()
            self.assertEqual(sup.state_hash(m,'bn'),before['bn'])
            if not adapting:self.assertEqual(sup.state_hash(m,'backbone'),before['backbone'])
            else:
                self.assertNotEqual(sup.state_hash(m,'early'),before['early']);self.assertNotEqual(sup.state_hash(m,'late'),before['late'])
                self.assertIsNotNone(m.bn.weight.grad)

    def test_draw_indices_and_augmentation_are_worker_independent(self):
        weights=torch.tensor([.5,.5],dtype=torch.float64);first=list(sup.DrawSampler(weights,812,3));second=list(sup.DrawSampler(weights,812,3))
        self.assertEqual(first,second);self.assertEqual([d for _,d in first],[0,1])
        limited=sup.DrawSampler(torch.ones(32,dtype=torch.float64),812,3,16)
        self.assertEqual(len(limited),16);self.assertEqual(len(list(limited)),16)
        x=Image.fromarray(np.arange(12*10*3,dtype=np.uint8).reshape(12,10,3));a=sup.augment(x,'p.png',3,8,812)
        torch.manual_seed(77);np.testing.assert_array_equal(a,sup.augment(x,'p.png',3,8,812))
        self.assertGreater(len({sup.augment(x,'p.png',3,d,812).tobytes() for d in range(8)}),1)
        self.assertEqual(tuple(sup.pixels(x,4).shape),(3,4,4))

    def test_registered_cosine_endpoints(self):
        cfg=self.config();self.assertEqual(sup.learning_rates(cfg,1),(.01,.02));self.assertEqual(sup.learning_rates(cfg,3),(.01,.02))
        self.assertAlmostEqual(sup.learning_rates(cfg,4)[0],.001);self.assertAlmostEqual(sup.learning_rates(cfg,4)[1],.002)

    def test_forbidden_row_rejected_before_any_file_open(self):
        rows,cases=rows_and_cases();bad=copy.deepcopy(rows);bad[0]['case_id_raw']='outer'
        with patch.object(sup,'load_checked_image',side_effect=AssertionError('Must not open images')):
            with self.assertRaisesRegex(ValueError,'allowed-role'):sup.stream(bad,'unused',cases,self.config(),0)

    def test_probability_readout_uses_equal_view_mass_and_refuses_outer(self):
        from scipy.special import logsumexp
        cases=[{'case_id':'c','label':1,'indices':{40:['a%d'%i for i in range(10)],100:['b'],200:['d'],400:['e']}}]
        paths=sum([cases[0]['indices'][m] for m in (40,100,200,400)],[]);z=np.full((13,8),-1000.);z[:,1]=0;z[:10,0]=1.;z[10:,0]=-2.
        lp=z-logsumexp(z,axis=1,keepdims=True);fields={'paths':np.array(paths),'log_probabilities':lp}
        records,_=sup.aggregate(cases,fields,'validation');row=next(r for r in records if r['magnifications']==(40,100))
        self.assertEqual(np.argmax(row['probabilities']),1)
        self.assertEqual(np.exp(lp[:11]).mean(0).argmax(),0)
        with self.assertRaises(ValueError):sup.aggregate(cases,fields,'test')
        bad={**fields,'paths':np.array(paths[:-1])}
        with self.assertRaises(ValueError):sup.aggregate(cases,bad,'validation')

    def test_resume_restores_earlier_best_and_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,cases=rows_and_cases(root);cfg=self.config();context={'source':'same','trial':1,'seed':812,'fit':[c['case_id'] for c in cases]}
            torch.manual_seed(9);base=TinyCNN();a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            # Avoid fitting/selection on actual validation; deterministic synthetic score depends only on epoch order.
            def fake_fields(model,rows,root,cases,cfg,device,workers):
                return {'paths':np.array([r['relative_path'] for r in rows]),'log_probabilities':np.full((len(rows),8),-np.log(8.)),'probabilities':np.full((len(rows),8),1/8),'logits':np.zeros((len(rows),8))}
            def fake_aggregate(cases,fields,role):return [],{'mean_subset_macro_f1':.5}
            with patch.object(sup,'score_fields',side_effect=fake_fields),patch.object(sup,'aggregate',side_effect=fake_aggregate):
                full=copy.deepcopy(base);sup.train(full,rows,rows,cases,cases,root,cfg,context,a,torch.device('cpu'),0)
                original=sup.stream
                def interrupted(*args,**kwargs):
                    if args[5]==4:raise RuntimeError('interruption')
                    return original(*args,**kwargs)
                part=copy.deepcopy(base)
                with patch.object(sup,'stream',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'interruption'):
                    sup.train(part,rows,rows,cases,cases,root,cfg,context,b,torch.device('cpu'),0)
                saved=sup.bound_checkpoint(b/'last.pt',context)
                self.assertEqual(saved['best']['epoch'],3)
                # Simulate orphan/corrupt selected-file contents; last's cloned best snapshot must repair it.
                torch.save({'wrong':'orphan'},b/'selected.pt')
                resumed=copy.deepcopy(base);proof=sup.train(resumed,rows,rows,cases,cases,root,cfg,context,b,torch.device('cpu'),0,resume=True)
                self.assertEqual(proof['best']['epoch'],3)
                for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
                chosen=sup.bound_checkpoint(b/'selected.pt',context);self.assertEqual(chosen['epoch'],3)
                self.assertEqual(sup.digest_state(chosen['model']),proof['best']['state_sha256'])
                with self.assertRaisesRegex(ValueError,'binding'):sup.bound_checkpoint(b/'last.pt',{**context,'seed':813})


if __name__=='__main__':unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    import magset.adapt_breakhis as adapt
except ImportError:
    torch=None;adapt=None

from magset.data import CLASS_NAMES,MAGNIFICATIONS,file_hash


if torch is not None:
    class TinyReader(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Module()
            self.encoder.layer=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
            self.layernorm=nn.LayerNorm(4)
        def forward(self,pixel_values):
            x=pixel_values
            for layer in self.encoder.layer:x=torch.tanh(layer(x))
            return SimpleNamespace(last_hidden_state=self.layernorm(x)[:,None,:])


def rows_and_cases(root=None):
    rows=[];cases=[]
    for label in range(8):
        prefix='SOB_B_C%d'%label;raw='14-%d'%label;key=prefix+'-'+raw
        cases.append({'case_id':key,'label':label,'indices':{}})
        for m in MAGNIFICATIONS:
            filename='%s-%s-%d-001.png'%(prefix,raw,m);path=filename
            row={'filename':filename,'relative_path':path,'case_id_raw':raw,'subtype':CLASS_NAMES[label],
                 'magnification':str(m),'width':'4','height':'4'}
            if root is not None:
                Image.new('RGB',(4,4),(label*20+3,m%255,80)).save(Path(root)/path)
                row['sha256']=file_hash(Path(root)/path)
            rows.append(row);cases[-1]['indices'][m]=[path]
    return rows,cases


@unittest.skipIf(adapt is None,'Pinned PyTorch needed for encoder-adaptation checks')
class AdaptationTests(unittest.TestCase):
    def test_class_case_view_sampling_mass_and_forbidden_rows(self):
        rows,cases=rows_and_cases();more=copy.deepcopy(rows[:4])
        for r in more:r['filename']=r['filename'].replace('14-0','14-99');r['case_id_raw']='14-99';r['relative_path']=r['filename']
        cases.append({'case_id':'SOB_B_C0-14-99','label':0,'indices':{}});rows+=more
        # Duplicate fields within one view change per-field weight, not its view mass.
        duplicate=copy.deepcopy(rows[0]);duplicate['filename']=duplicate['filename'].replace('001','002');duplicate['relative_path']=duplicate['filename'];rows.append(duplicate)
        weights=adapt.sampling_weights(rows,cases).numpy();self.assertAlmostEqual(weights.sum(),1)
        for label in range(8):
            ids={c['case_id'] for c in cases if c['label']==label}
            self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if adapt.case_id(r) in ids),1/8)
        first=cases[0]['case_id'];self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if adapt.case_id(r)==first),1/16)
        bad=copy.deepcopy(rows);bad[0]['case_id_raw']='unknown'
        with self.assertRaises(ValueError):adapt.sampling_weights(bad,cases)

    def test_only_whitelisted_encoder_weights_change_after_warmup(self):
        torch.manual_seed(9);model=TinyReader();names=adapt.trainable_names(model)
        before=adapt.fingerprint(model,names,False);last=adapt.fingerprint(model,names,True)
        head=nn.Linear(4,8);opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if n in names],'lr':.01},
                                                {'params':head.parameters(),'lr':.02}])
        x=torch.randn(8,4);y=torch.arange(8)
        adapt.set_phase(model,False);adapt.step(model,head,opt,x,y,1.)
        self.assertEqual(last,adapt.fingerprint(model,names,True));self.assertEqual(before,adapt.fingerprint(model,names,False))
        adapt.set_phase(model,True);adapt.step(model,head,opt,x,y,1.)
        self.assertNotEqual(last,adapt.fingerprint(model,names,True));self.assertEqual(before,adapt.fingerprint(model,names,False))
        self.assertTrue(all(p.requires_grad==(n in names) for n,p in model.named_parameters()))

    def test_fragment_and_checkpoint_role_source_bindings(self):
        model=TinyReader();state=adapt.fragment(model);other=copy.deepcopy(model)
        adapt.restore_fragment(other,state)
        self.assertEqual(adapt.fingerprint(model,adapt.trainable_names(model),True),adapt.fingerprint(other,adapt.trainable_names(other),True))
        bad={**state,'encoder.layer.0.weight':model.encoder.layer[0].weight.detach()}
        with self.assertRaises(ValueError):adapt.restore_fragment(other,bad)
        bad=copy.deepcopy(state);next(iter(bad.values())).flatten()[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):adapt.restore_fragment(other,bad)
        with self.assertRaises(ValueError):adapt.checked_difference(np.array([float('nan')]),np.zeros(1))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.pt';context={'fit':['fit-only'],'config':'fixed','code_sha':'original'}
            torch.save({'context':context},p);adapt.bound_checkpoint(p,context)
            with self.assertRaisesRegex(ValueError,'binding'):adapt.bound_checkpoint(p,{**context,'fit':['validation']})

    def test_hash_augmentation_does_not_depend_on_worker_rng(self):
        image=Image.fromarray(np.arange(36,dtype=np.uint8).reshape(3,4,3))
        a=np.asarray(adapt.flipped(image,'fixed.png',2,812));torch.manual_seed(99)
        np.testing.assert_array_equal(a,np.asarray(adapt.flipped(image,'fixed.png',2,812)))

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,cases=rows_and_cases(root)
            torch.manual_seed(4);initial=TinyReader()
            cfg={'dimension':4,'head_dropout':.2,'encoder_learning_rate':.01,'head_learning_rate':.02,'weight_decay':.01,
                 'epochs':2,'head_only_epochs':1,'training_seed':812,'batch_size':8,'gradient_clip':1.}
            context={'config':cfg,'fit':[c['case_id'] for c in cases],'trial':1,'seed':812};a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            transform=lambda images:torch.tensor([[np.asarray(im).mean()/255, .2,.3,.4] for im in images],dtype=torch.float32)
            first=copy.deepcopy(initial);second=copy.deepcopy(initial)
            adapt.train(first,transform,rows,cases,root,cfg,context,a,torch.device('cpu'),0,False,False)
            original_loader=adapt.loader
            def interrupted(*args,**kwargs):
                if args[5]==2:raise RuntimeError('simulated interruption')
                return original_loader(*args,**kwargs)
            with patch('magset.adapt_breakhis.loader',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    adapt.train(second,transform,rows,cases,root,cfg,context,b,torch.device('cpu'),0,False,False)
            resumed=copy.deepcopy(initial)
            proof=adapt.train(resumed,transform,rows,cases,root,cfg,context,b,torch.device('cpu'),0,False,True)
            for key,value in first.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            self.assertEqual([r['draws'] for r in proof['history']],[32,32])

    def test_adapted_case_head_reload_and_validation_only_predictions(self):
        rng=np.random.default_rng(22);cases=[];paths=[];z=[];counts=[1,25,6,2,5,3,1,3]
        labels=[label for label,n in enumerate(counts) for _ in range(n)]+list(range(8))
        for i,label in enumerate(labels):
            case={'case_id':'case%d'%i,'label':label,'indices':{}}
            for m in MAGNIFICATIONS:
                keys=['case%d/%d/%d'%(i,m,j) for j in range(2)];case['indices'][m]=keys;paths+=keys
                z.extend(np.eye(8)[label]+rng.normal(0,.02,(2,8)))
            cases.append(case)
        with tempfile.TemporaryDirectory() as tmp:
            C,metric,choices,error=adapt.fit_case_head(cases,{'paths':np.asarray(paths),'features':np.asarray(z,dtype=np.float32)},tmp)
            self.assertEqual(len(choices),4);self.assertLessEqual(error,1e-8)
            records=[json.loads(line) for line in (Path(tmp)/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),120);self.assertEqual({r['role'] for r in records},{'validation'})
            self.assertEqual({r['case_id'] for r in records},{'case%d'%i for i in range(46,54)})


if __name__=='__main__':unittest.main()

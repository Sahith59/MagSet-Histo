import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from magset.collect import verify_run
from synthetic_bags import write_fixture
try:
    import torch
    from magset.fusion import HierarchicalFusion
    from magset.train_fusion import all_subset_logits, subset_mask
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Install pinned PyTorch to run neural checks')
class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(812)
        self.x = torch.randn(2,4,5,8)
        self.fields = torch.ones((2,4,5),dtype=torch.bool); self.fields[:,:,4] = False
        self.observed = torch.tensor([[True,False,True,False],[False,True,False,True]])

    def models(self):
        for field,view in itertools.product(('mean','attention'),('mean','gated','attention')):
            yield HierarchicalFusion(8,hidden=8,field_pool=field,view_fusion=view,dropout=0,heads=2).eval()

    def test_absent_nan_fields_padding_and_permutations_cannot_change_prediction(self):
        for model in self.models():
            expected = model(self.x,self.fields,self.observed)
            changed = self.x.clone(); changed[~self.observed] = float('nan')
            changed[~self.fields] = float('nan')
            actual = model(changed,self.fields,self.observed)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
            field_order = torch.tensor([3,0,4,2,1]); scale_order = torch.tensor([3,1,0,2])
            actual = model(self.x[:,:,field_order],self.fields[:,:,field_order],self.observed)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
            actual = model(self.x[:,scale_order],self.fields[:,scale_order],self.observed[:,scale_order],scale_order)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

    def test_all_subsets_match_direct_forward_and_have_finite_gradients(self):
        for model in self.models():
            masks = subset_mask('cpu')
            logits = all_subset_logits(model,self.x,self.fields,masks)
            self.assertEqual(tuple(logits.shape),(15,2,8))
            for i,mask in enumerate(masks):
                expected = model(self.x,self.fields,mask.expand(2,-1))
                torch.testing.assert_close(logits[i],expected,atol=1e-6,rtol=1e-6)
            x = self.x.clone().requires_grad_()
            model(x,self.fields,self.observed).square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(x.grad[~self.observed].abs().sum().item(),0)
            self.assertEqual(x.grad[~self.fields].abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_empty_input_rejected_and_state_dict_reloads(self):
        for model in self.models():
            with self.assertRaises(ValueError):
                model(self.x,self.fields,torch.zeros_like(self.observed))
            fields = self.fields.clone(); fields[0,0] = False
            with self.assertRaises(ValueError):
                model(self.x,fields,self.observed)
            expected = model(self.x,self.fields,self.observed)
            stream = io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
            model.load_state_dict(torch.load(stream,weights_only=True))
            torch.testing.assert_close(model(self.x,self.fields,self.observed),expected)

    def test_training_cli_saved_scores_reload_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,protocol,cache = write_fixture(tmp); root = Path(tmp)
            config = root/'config.json'
            config.write_text(json.dumps({'hidden':8,'heads':2,'dropout':.1,'epochs':2,'batch_size':8,
                'grid':[{'learning_rate':.001,'weight_decay':.01},{'learning_rate':.003,'weight_decay':.01}]}))
            output = root/'run'
            run = subprocess.run([sys.executable,'-m','magset.train_fusion','--manifest',str(manifest),
                '--protocol',str(protocol),'--features',str(cache),'--encoder','phikon-v2','--trial','1',
                '--field-pool','attention','--view-fusion','attention','--config',str(config),'--device','cpu',
                '--output',str(output),'--code-sha','synthetic'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report,_ = verify_run(output/'metrics.json')
            self.assertTrue(report['reload_verified']); self.assertNotIn('test',report)
            saved = torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)
            self.assertEqual(saved['epoch'],2); self.assertIn('optimizer',saved)
            self.assertEqual(len(report['candidates']),2)


if __name__ == '__main__':
    unittest.main()

"""Private-intake and immutable adapter checks; no model fitting."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('swin_adapter',Path(__file__).resolve().parents[1]/'scripts/run_supplied_swin.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    cells=[{'cell_type':'code','source':'# cell%d'%i,'outputs':[],'metadata':{},'execution_count':None}for i in range(15)]
    cells[0]={'cell_type':'markdown','source':'Original introduction','metadata':{}}
    cells[2]['source']="OUT_DIR     = '/kaggle/working'"
    cells[3]['source']="root_dir = '/kaggle/input/breakhisv1/histology_slides/breast'"
    cells[8]['source']='reader=TFSwinModel.from_pretrained(SWIN_CKPT)'
    return {'cells':cells,'metadata':{},'nbformat':4,'nbformat_minor':5}


class SuppliedSwinTests(unittest.TestCase):
    def test_only_paths_and_loading_are_patched(self):
        original=fixture();before=copy.deepcopy(original)
        cells=m.patched_sources(original,'/synthetic/data','/synthetic/out','/synthetic/weights')
        self.assertEqual([i for i,c in enumerate(cells)if c!=original['cells'][i]['source']],[2,3,8])
        self.assertIn('local_files_only=True, use_safetensors=False',cells[8])
        self.assertEqual(original,before)
        original['cells'][8]['source']='unexpected anchor'
        with self.assertRaises(ValueError):m.patched_sources(original,'/a','/b','/c')

    def test_fixed_scope_and_no_automatic_search(self):
        config=Path('configs/supplied-swin-v1.json');x=m.config_receipt(config)
        self.assertEqual(x['mode'],'dual');self.assertEqual(x['split']['seed'],0)
        self.assertEqual(x['training']['maximum_epochs'],40)
        self.assertEqual(x['budget']['production_models'],1)
        with tempfile.TemporaryDirectory()as tmp:
            changed=Path(tmp)/'config.json';changed.write_text(config.read_text()+' ')
            with self.assertRaises(ValueError):m.config_receipt(changed)

    def test_actual_private_notebook_anchors_and_sources(self):
        path=Path('.local/supplied-swin/original.ipynb')
        if not path.exists():self.skipTest('Exact supplied notebook is private')
        original=m.load_original(path);cells=m.patched_sources(original,'/a','/b','/c')
        self.assertEqual(len(cells),15)
        self.assertEqual([i for i,c in enumerate(cells)if c!=''.join(original['cells'][i]['source'])],[2,3,8])
        for i in range(2,15):ast.parse(cells[i])
        self.assertIn('tf.stop_gradient(p_bin)',cells[8])
        self.assertIn("mal = y5_true > 0",cells[9])
        self.assertIn("labels=range(5)",cells[9])

    def test_runtime_and_technical_code_are_valid_python(self):
        for code in (m.runtime('/synthetic/prepared.json',True),m.prefit('/synthetic/audit'),m.EXPORT,m.SMOKE):
            ast.parse(code)
        self.assertIn("assert _gpu_count>=1",m.runtime('/x',True))
        self.assertNotIn("assert _gpu_count>=1",m.runtime('/x',False))
        self.assertIn("'real_test_predictions':0",m.SMOKE)


if __name__=='__main__':unittest.main()

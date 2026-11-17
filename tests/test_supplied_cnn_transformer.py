"""No-TensorFlow adapter checks for notebook scope, order and failure retention."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('supplied_adapter', Path(__file__).resolve().parents[1]/'scripts/run_supplied_cnn_transformer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def notebook_fixture():
    sources = ['# cell%d'%i for i in range(15)]
    sources[1] = "root_dir = '/kaggle/input/breakhisv1/histology_slides/breast'"
    sources[2] = "data_csv = pd.read_csv('/kaggle/working/breakhis_dataset.csv')"
    sources[4] = 'datagen.random_transform(img / 255.)'
    sources[5] = ("        super().__init__(**kwargs)\n        self.attention = sentinel\n"
                  "            'd_model': self.attention.key_dim * self.attention._num_heads,\n"
                  "            'num_heads': self.attention._num_heads,")
    sources[7] = "EPOCHS = 50\nfilepath = '/kaggle/working/cnn_transformer_dualhd.keras'"
    sources[10] = 'X_test_norm = X_test / 255.'
    sources[13] = 'img = X_test[idx]\nmodel.predict(np.expand_dims(img,0))'
    return {'cells':[{'cell_type':'code','source':s,'outputs':[]}for s in sources]}


class SuppliedNotebookAdapterTests(unittest.TestCase):
    def test_only_registered_variant_cells_differ_and_visualization_stays_raw(self):
        original = notebook_fixture()
        faithful = m.patched_sources(original, 'faithful', '/synthetic/data', '/synthetic/output')
        fixed = m.patched_sources(original, 'input_fixed', '/synthetic/data', '/synthetic/output')
        self.assertEqual([i for i,(a,b) in enumerate(zip(faithful,fixed))if a!=b],[4,10])
        self.assertIn('img / 255.',faithful[4]);self.assertNotIn('img / 255.',fixed[4])
        self.assertEqual(faithful[13],original['cells'][13]['source'])
        self.assertEqual(fixed[13],original['cells'][13]['source'])
        self.assertIn('EPOCHS = 50',fixed[7])
        for name in ('d_model','num_heads','dff','dropout_rate'):
            self.assertIn('self.'+name, fixed[5])
        self.assertNotIn('self.attention.key_dim',fixed[5])
        with self.assertRaises(ValueError):m.patched_sources(original,'other','/x','/y')

    def test_actual_private_notebook_matches_anchors_when_available(self):
        path=Path('.local/supplied-cnn-transformer/original.ipynb')
        if not path.exists():self.skipTest('Original supplied notebook is private')
        original=m.original_notebook(path)
        faithful=m.patched_sources(original,'faithful','/data/root','/output')
        fixed=m.patched_sources(original,'input_fixed','/data/root','/output')
        self.assertEqual(len(faithful),15)
        self.assertEqual([i for i,(a,b)in enumerate(zip(faithful,fixed))if a!=b],[4,10])
        for index in (0,3,6,8,9,11,12,13,14):
            self.assertEqual(faithful[index],''.join(original['cells'][index]['source']))

    def test_literal_enumeration_retains_unsorted_class_and_file_order(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for cls in ('benign','malignant'):(root/cls/'SOB'/'subtype'/'40X').mkdir(parents=True)
            real_listdir=m.os.listdir
            def listing(path):
                return ['malignant','benign']if Path(path)==root else real_listdir(path)
            def walking(path):yield (str(Path(path)/'40X'),[],['z.png','a.png'])
            with patch.object(m.os,'listdir',side_effect=listing),patch.object(m.os,'walk',side_effect=walking):
                rows=m.enumerate_images(root)
            self.assertEqual([r['class_label']for r in rows],['malignant','malignant','benign','benign'])
            self.assertEqual([Path(r['image_path']).name for r in rows],['z.png','a.png','z.png','a.png'])

    def test_manifest_binding_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory()as tmp:
            raw=Path(tmp); breast=raw/'breast'; path=breast/'benign'/'SOB'/'adenosis'/'case'/'40X'/'SOB_B_A-test-40-001.png'
            path.parent.mkdir(parents=True);path.write_bytes(b'synthetic-field-bytes')
            row={'relative_path':str(path.relative_to(raw)),'filename':path.name,'broad_class':'benign','subtype':'adenosis',
                 'magnification':'40','sha256':m.file_hash(path),'case_id_raw':'test','patient_group_conservative':'synthetic-group'}
            manifest=raw/'manifest.csv'
            with manifest.open('w',newline='')as h:
                writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            rows=m.bound_rows(manifest,raw,breast,expected_count=1)
            self.assertEqual(rows[0]['relative_path'],row['relative_path'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes'):m.bound_rows(manifest,raw,breast,expected_count=1)

    def test_fixed_split_counts_and_no_order_sorting(self):
        rows=[{'class_label':'benign' if i%8<4 else 'malignant','subclass_label':'type%d'%(i%8)}for i in range(7909)]
        yb,ys,train,test,binary,subtype=m.split_labels(rows)
        self.assertEqual([len(train),len(test)],[6327,1582])
        self.assertFalse(set(train)&set(test));self.assertEqual(len(set(train)|set(test)),7909)
        self.assertEqual(binary,['benign','malignant']);self.assertEqual(len(subtype),8)
        self.assertEqual(ys[:9].tolist(),[0,1,2,3,4,5,6,7,0])
        compile(m.prefit_guard('/audit'),'prefit-guard','exec')
        compile(m.EXPORT_METRICS,'metric-export','exec')
        self.assertIn('_checkpoint_model.predict(X_test_norm[:_reload_count],batch_size=32',m.EXPORT_METRICS)
        self.assertNotIn('X_test / 255',m.EXPORT_METRICS)

    def test_registration_config_is_required_and_exact(self):
        canonical=Path('configs/supplied-cnn-transformer-v1.json')
        self.assertEqual(m.registered_config(canonical)['notebook_sha256'],m.NOTEBOOK_SHA)
        with self.assertRaises(ValueError):m.registered_config(None)
        with tempfile.TemporaryDirectory()as tmp:
            changed=Path(tmp)/'config.json';value=json.loads(canonical.read_text());value['training']['maximum_epochs']=51
            changed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'exact immutable'):m.registered_config(changed)
        # Exercise the actual embedded wrapper without importing TensorFlow.
        function=next(node for node in ast.parse(m.runtime_preamble(False)).body
                      if isinstance(node,ast.FunctionDef)and node.name=='_project_keras_get_file')
        calls=[]
        def original(*args,**kwargs):calls.append((args,kwargs));return 'original-download-result'
        namespace={'_keras_cache_root':'/synthetic/project-cache','_original_keras_get_file':original}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'cache-wrapper','exec'),namespace)
        value=namespace['_project_keras_get_file']('weights.h5',origin='https://example.invalid/official',file_hash='unchanged',cache_dir='/tmp/cache')
        self.assertEqual(value,'original-download-result')
        self.assertEqual(calls,[(('weights.h5',),{'origin':'https://example.invalid/official','file_hash':'unchanged','cache_dir':'/synthetic/project-cache'})])
        self.assertIn("run_line_magic('matplotlib','inline')",m.SMOKE_CODE)
        self.assertIn("savefig('smoke-plot.png')",m.SMOKE_CODE)
        compile(m.SMOKE_CODE,'synthetic-smoke','exec');compile(m.WEIGHT_GUARD,'weight-guard','exec')

    def test_notebook_execution_error_preserves_partial_outputs(self):
        saved={}; kernel=SimpleNamespace(kernel_spec=SimpleNamespace(argv=[]))
        class FailingClient:
            def __init__(self,notebook,**kwargs):self.notebook=notebook;saved['kwargs']=kwargs
            def execute(self,**kwargs):
                self.notebook['partial_output']='preserved';raise RuntimeError('synthetic cell failure')
        fake_nbformat=SimpleNamespace(write=lambda notebook,path:Path(path).write_text(json.dumps(notebook)))
        with tempfile.TemporaryDirectory()as tmp,patch.dict(sys.modules,{'nbformat':fake_nbformat,'nbclient':SimpleNamespace(NotebookClient=FailingClient),'jupyter_client':SimpleNamespace(KernelManager=lambda**kwargs:kernel)}):
            with self.assertRaisesRegex(RuntimeError,'synthetic cell failure'):m.execute_notebook({},tmp)
            self.assertEqual(json.loads((Path(tmp)/'executed.ipynb').read_text())['partial_output'],'preserved')
            self.assertTrue((Path(tmp)/'execution_error.json').exists())
            self.assertEqual(kernel.kernel_spec.argv[0],sys.executable)


if __name__=='__main__':unittest.main()

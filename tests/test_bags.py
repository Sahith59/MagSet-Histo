import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from magset.bags import fit_normalization, load_bags
from magset.collect import collect, verify_run
from magset.data import file_hash, read_manifest
from magset.field_baseline import field_training_data
from synthetic_bags import write_fixture


class BagsTests(unittest.TestCase):
    def test_normalization_excludes_held_out_and_balances_field_counts(self):
        bags = [[np.array([[float(i)],[float(i+2)]]) for _ in range(4)] for i in range(3)]
        mean,std = fit_normalization(bags,[0,1])
        repeated = [[np.repeat(x,7,axis=0) for x in bags[0]],bags[1],
                    [np.full((2,1),1e9) for _ in range(4)]]
        other_mean,other_std = fit_normalization(repeated,[0,1])
        np.testing.assert_allclose(mean,other_mean); np.testing.assert_allclose(std,other_std)
        np.testing.assert_allclose(mean,1.5); np.testing.assert_allclose(std,np.sqrt(1.25))

    def test_field_weights_give_equal_class_mass_and_case_influence(self):
        y = np.array(list(range(8))+[0])
        bags = [[np.ones((i+1,2)) for _ in range(4)] for i in range(9)]
        x,labels,weights = field_training_data(bags,y,np.arange(9),0)
        self.assertEqual(len(x),45); self.assertAlmostEqual(weights.sum(),9)
        for label in range(8):
            self.assertAlmostEqual(weights[labels==label].sum(),9/8)
        self.assertAlmostEqual(weights[0],weights[-9:].sum())

    def test_cache_tampering_and_field_baseline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,protocol,cache = write_fixture(tmp)
            load_bags(read_manifest(manifest),cache,'phikon-v2',file_hash(manifest))
            output = Path(tmp)/'runs'/'field'
            run = subprocess.run([sys.executable,'-m','magset.field_baseline','--manifest',str(manifest),
                '--protocol',str(protocol),'--features',str(cache),'--encoder','phikon-v2','--trial','1',
                '--output',str(output),'--code-sha','synthetic'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report,_ = verify_run(output/'metrics.json')
            self.assertEqual(report['method'],'field-late'); self.assertNotIn('test',report)
            summary = collect(output.parent,encoders=('phikon-v2',),methods=('field-late',),trials=(1,))
            self.assertEqual(summary['verified_runs'],1); self.assertEqual(len(summary['summary']),1)
            meta_path = cache/'phikon-v2-40.json'; meta = json.loads(meta_path.read_text())
            meta['magnification'] = 400; meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_bags(read_manifest(manifest),cache,'phikon-v2',file_hash(manifest))


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from magset.collect import ENCODERS, METHODS
from magset.error_analysis import analyze, markdown
from test_collect import fixture


class ErrorAnalysisTests(unittest.TestCase):
    def test_perfect_predictions_have_no_view_harm_and_reused_cases_count_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'; runs.mkdir()
            index = 0
            for encoder in ENCODERS:
                for method in METHODS:
                    for trial in range(1,6):
                        work = root/('fixture%d'%index); work.mkdir()
                        path,report = fixture(work)
                        report.update(encoder=encoder,method=method,trial=trial)
                        path.write_text(json.dumps(report))
                        path.parent.rename(runs/str(index)); index += 1
            report = analyze(runs)
            self.assertEqual(report['verified_runs'],30)
            for group in report['groups']:
                self.assertEqual(group['view_addition'],{'paired_comparisons':1120,'correct_to_wrong':0,'wrong_to_correct':0})
                self.assertEqual(group['mean_macro_f1_by_view_count'],{str(i):1. for i in range(1,5)})
                self.assertTrue(all(r['distinct_cases']==1 and r['recall']==1 for r in group['class_recall']))
            self.assertNotIn('val0',markdown(report))


if __name__ == '__main__':
    unittest.main()

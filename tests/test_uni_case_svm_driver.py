"""Synthetic full case-SVC workflow and failure-boundary checks."""
import importlib.util
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import uni_case_svm as core
from magset.data import file_hash

spec = importlib.util.spec_from_file_location('case_svc_driver', Path(__file__).resolve().parents[1] / 'scripts/run_uni_case_svm.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    rng = np.random.RandomState(168)
    labels = np.arange(20, dtype=np.int64) % 5; train = np.arange(10, dtype=np.int64)
    z = rng.normal(size=(20, 4, 6)); z[:, :, 0] += labels[:, None]
    truth8 = (0, 1, 3, 4, 5)
    cases = [dict(case_id='synthetic%02d' % i, truth5=int(y), truth8=truth8[y]) for i, y in enumerate(labels)]
    audit = {'cases': cases, 'roles': {'train': train.tolist(), 'validation': list(range(10, 15)), 'test': list(range(15, 20))}}
    context = dict(code_sha='a' * 40, source_archive_sha256='b' * 64, config_sha256='c' * 64,
                   parent_result_sha256='d' * 64, input_audit_sha256='e' * 64)
    return dict(values=z, labels=labels, preprocessing=core.fit_preprocessing(z, train), audit=audit, context=context,
                config={'counts': {'validation_case_subset_records': 600, 'retrospective_case_subset_records': 150}})


class CaseSVCDriverTests(unittest.TestCase):
    def test_full_family_saves_six_key_archives_exports_and_seals_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'study'; state = fixture(); original = core.predict; events = []
            def guarded(bundle, values, indices):
                indices = np.asarray(indices)
                if np.array_equal(indices, np.arange(15, 20)):
                    m.verify_gate(output); events.append('selected-test')
                else:
                    self.assertTrue(np.array_equal(indices, np.arange(10, 15))); events.append('validation')
                return original(bundle, values, indices)
            with redirect_stdout(StringIO()), patch.object(m, 'load_inputs', return_value=state), patch.object(core, 'predict', side_effect=guarded), \
                 patch.object(core, 'fit_bundle', wraps=core.fit_bundle) as fit:
                result = m.run(SimpleNamespace(output=output))
            self.assertEqual(fit.call_count, 8); self.assertEqual(result['head_estimator_fits'], 16)
            self.assertEqual(events, ['validation'] * 16 + ['selected-test'] * 2)
            self.assertEqual(result['validation_case_subset_records'], 600)
            self.assertEqual(result['retrospective_case_subset_records'], 150)
            with np.load(output / 'preprocessing.npz', allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {'case_view_means', 'mean', 'std', 'scale', 'train_case_indices', 'case_indices', 'masks'})
                np.testing.assert_array_equal(saved['mean'], state['values'][:10].mean(0))
            for arm in core.ARMS:
                selected = m.read_json(output / 'arms' / arm / 'selection.json')
                self.assertEqual(selected['selected_index'], int(np.argmax(selected['candidate_validation_scores'])))
                for index, receipt in enumerate(selected['candidate_artifact_sha256']):
                    folder = output / 'candidates' / arm / ('candidate%d' % index)
                    self.assertEqual(set(receipt), set(m.CANDIDATE_FILES))
                    with np.load(folder / 'state.npz', allow_pickle=False) as saved:
                        self.assertIn('binary_dual_coef_', saved.files); self.assertIn('malignant_support_case_indices', saved.files)
                        self.assertTrue(set(saved['binary_support_case_indices']) <= set(range(10)))
                    with np.load(folder / 'validation-probabilities.npz', allow_pickle=False) as saved:
                        self.assertEqual(set(saved.files), set(m.ARCHIVE_KEYS))
                        self.assertEqual(saved['binary_margin'].shape, (5, 15))
                        self.assertEqual(saved['malignant_scores'].shape, (5, 15, 4))
                receipt = m.read_json(output / 'arms' / arm / 'test/test.json')
                self.assertEqual(receipt['status'], 'complete selected retrospective inference')
                self.assertEqual(receipt['selected_setting'], selected['selected_setting'])
            for name, digest in result['artifact_sha256'].items(): self.assertEqual(file_hash(output / name), digest)
            candidate = output / 'candidates/linear/candidate0/state.npz'
            candidate.write_bytes(b'corrupted after seal')
            with self.assertRaisesRegex(ValueError, 'changed after seal'): m.verify_gate(output)

    def test_failed_reload_does_not_seal_or_query_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'failed'; state = fixture()
            with patch.object(m, 'load_inputs', return_value=state), \
                 patch.object(m, 'reload_check', side_effect=ValueError('synthetic reload failure')) as reload, \
                 patch.object(core, 'predict', wraps=core.predict) as predict:
                with self.assertRaisesRegex(ValueError, 'reload failure'): m.run(SimpleNamespace(output=output))
            self.assertEqual(reload.call_count, 1); predict.assert_not_called()
            self.assertFalse((output / 'family-gate.json').exists())
            self.assertFalse((output / 'arms/linear/test').exists())

    def test_existing_output_is_not_modified_or_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing'; output.mkdir(); (output / 'keep.txt').write_text('preserved')
            with patch.object(m, 'load_inputs') as load:
                with self.assertRaisesRegex(ValueError, 'overwrite'): m.run(SimpleNamespace(output=output))
                load.assert_not_called()
            self.assertEqual((output / 'keep.txt').read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()

"""Synthetic CPU checks of matched P supports, selection and sealed test access."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import matched_p as m
from magset import matched_data as md
from magset.data import file_hash


def fixture():
    rng = np.random.RandomState(91)
    truth8 = (0, 1, 3, 4, 5)
    cases = [dict(case_id='synthetic-case-%02d' % i, group='synthetic-group-%02d' % i,
                  truth5=i % 5, truth8=truth8[i % 5]) for i in range(15)]
    bags = [[rng.normal(size=(2, 6)).astype(np.float32) + (i % 5) for _ in range(4)] for i in range(15)]
    fields = []
    for case in range(15):
        for view in range(4):
            for field in range(2):
                fields.append(dict(index=len(fields), case_index=case, view=view,
                                   relative_path='case_%02d/view%d/field%d' % (case, view, field)))
    audit = dict(cases=cases, fields=fields, roles={'train': list(range(5)),
                 'validation': list(range(5, 10)), 'test': list(range(10, 15))})
    return audit, bags, np.array([c['truth5'] for c in cases], dtype=np.int64)


class MatchedPTests(unittest.TestCase):
    def test_field_mapping_normalization_and_query_order_invariance(self):
        audit, bags, labels = fixture()
        support = audit['roles']['train']; queries = audit['roles']['validation']
        indices, values = m._field_probabilities(audit, bags, labels, support, queries, (.03,))
        reversed_indices, reversed_values = m._field_probabilities(audit, bags, labels, support, queries[::-1], (.03,))
        self.assertEqual(indices.tolist(), list(range(40, 80)))
        np.testing.assert_array_equal(indices, reversed_indices)
        np.testing.assert_array_equal(values[0], reversed_values[0])
        np.testing.assert_allclose(values[0].sum(1), 1, atol=1e-14, rtol=0)
        changed = labels.copy(); changed[queries] = (changed[queries] + 1) % 5
        _, unchanged = m._field_probabilities(audit, bags, changed, support, queries, (.03,))
        np.testing.assert_array_equal(values[0], unchanged[0])

    def test_query_preparation_occurs_once_per_case_and_shares_temperatures(self):
        audit, bags, labels = fixture()
        original = m.p.prepare_case
        with patch.object(m.p, 'prepare_case', wraps=original) as prepare:
            _, values = m._field_probabilities(audit, bags, labels, audit['roles']['train'],
                                              audit['roles']['validation'], m.TEMPERATURES)
        self.assertEqual(len(values), 4)
        self.assertEqual([call.args[3] for call in prepare.call_args_list], list(range(5, 10)))
        self.assertTrue(all(call.kwargs == {'leave_query_case_out': False} for call in prepare.call_args_list))
        self.assertTrue(all(call.args[2] == list(range(5)) for call in prepare.call_args_list))
        with self.assertRaises(ValueError):
            m._field_probabilities(audit, bags, labels, [0, 1, 2, 3, 4], [4], (.03,))
        with self.assertRaises(ValueError):
            m._field_probabilities(audit, bags, labels, list(range(5)), [5], (.02,))

    def test_all_twelve_cache_artifacts_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir()
            pins = {}
            for mag in (40, 100, 200, 400):
                base = cache / ('midnight-coverage-v1-%d' % mag)
                paths = {'npz_sha256': base.with_suffix('.npz'), 'json_sha256': base.with_suffix('.json'),
                         'components_sha256': cache / (base.name + '-components.npz')}
                for name, path in paths.items(): path.write_bytes(('%s-%s' % (mag, name)).encode())
                pins[str(mag)] = {name: file_hash(path) for name, path in paths.items()}
            coverage = root / 'coverage.json'
            coverage.write_text(json.dumps(dict(encoder_name='midnight-coverage-v1', dimension=3072,
                                                temperatures=list(m.TEMPERATURES), cache_provenance=pins)))
            readout = root / 'readout.json'
            readout.write_text(json.dumps(dict(base_input_config_sha256=file_hash(coverage),
                                               modes=['head_specific'], temperatures=list(m.TEMPERATURES))))
            config = {'midnight_p': dict(feature_config=str(coverage), feature_config_sha256=file_hash(coverage),
                       readout_config=str(readout), mode='head_specific', temperatures=list(m.TEMPERATURES))}
            m._cache_bindings(cache, config)
            (cache / 'midnight-coverage-v1-200-components.npz').write_bytes(b'changed components')
            with self.assertRaisesRegex(ValueError, 'cache bytes'):
                m._cache_bindings(cache, config)

    def _args_and_inputs(self, root):
        audit, bags, labels = fixture()
        audit_dir = root / 'audit'; audit_dir.mkdir()
        (audit_dir / 'private.json').write_text(json.dumps(audit))
        output = root / 'models' / 'midnight_p'
        args = SimpleNamespace(config=root / 'config.json', audit=audit_dir, cache=root / 'cache',
                               output=output, code_sha='a' * 40, root=root)
        context = {'code_sha': 'a' * 40, 'config_sha256': 'b' * 64, 'source_archive_sha256': 'c' * 64}
        return args, (dict(), audit, context, bags, labels)

    def test_selection_uses_validation_only_and_first_tie_then_hashes_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, inputs = self._args_and_inputs(Path(tmp))
            with patch.object(m, '_load_inputs', return_value=inputs), \
                 patch.object(md, 'summary', return_value={'mean_subset_macro_f1': .5}), \
                 patch.object(m.p, 'prepare_case', wraps=m.p.prepare_case) as prepare:
                selected = m.select(args)
            self.assertEqual(selected['selected_index'], 0)
            self.assertEqual(selected['selected_setting'], .01)
            self.assertEqual(selected['test_predictions'], 0)
            self.assertEqual([call.args[3] for call in prepare.call_args_list], list(range(5, 10)))
            self.assertEqual(selected['support_case_indices'], list(range(5)))
            self.assertEqual(len(selected['candidate_artifact_sha256']), 4)
            result = json.loads((args.output / 'result.json').read_text())
            self.assertEqual(len(result['artifact_sha256']), 13)
            for name, digest in result['artifact_sha256'].items():
                self.assertEqual(file_hash(args.output / name), digest)
            with np.load(args.output / 'candidates/candidate0/validation-fields.npz') as arrays:
                self.assertEqual(set(arrays.files), {'field_indices', 'joint_probabilities5'})
                self.assertEqual(arrays['field_indices'].tolist(), list(range(40, 80)))

    def test_test_gate_runs_before_any_input_load_or_query(self):
        args = SimpleNamespace(root='study', audit='audit', config='config', code_sha='a' * 40)
        with patch.object(md, 'verify_gate', side_effect=ValueError('unsealed'), create=True) as gate, \
             patch.object(m, '_load_inputs') as load, patch.object(m, '_field_probabilities') as query:
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                m.test(args)
        gate.assert_called_once_with('study', 'audit', 'config', 'a' * 40)
        load.assert_not_called(); query.assert_not_called()

    def test_sealed_test_uses_only_selected_temperature_and_rejects_changed_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args, inputs = self._args_and_inputs(root)
            with patch.object(m, '_load_inputs', return_value=inputs), \
                 patch.object(md, 'summary', side_effect=[{'mean_subset_macro_f1': s} for s in (.2, .7, .7, .1)]):
                m.select(args)
            (root / 'test-gate.json').write_text('{"status":"synthetic accepted gate"}')
            args.output = args.output / 'test'
            with patch.object(md, 'verify_gate', return_value={}, create=True), \
                 patch.object(m, '_load_inputs', return_value=inputs), \
                 patch.object(m.p, 'density_cache', wraps=m.p.density_cache) as density:
                result = m.test(args)
            self.assertEqual(result['status'], 'complete selected test inference')
            self.assertEqual([call.args[1] for call in density.call_args_list], [.03] * 5)
            with np.load(args.output / 'fields.npz') as arrays:
                self.assertEqual(arrays['field_indices'].tolist(), list(range(80, 120)))
            metrics = json.loads((args.output / 'metrics.json').read_text())
            self.assertEqual(metrics['distinct_cases'], 5)
            self.assertEqual(len(metrics['subsets']), 15)
            selection_path = args.output.parent / 'selection.json'
            (args.output.parent / 'candidates/candidate0/metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed after selection'):
                m._selected_candidate(selection_path, inputs[2], file_hash(args.audit / 'private.json'), list(range(5)))


if __name__ == '__main__':
    unittest.main()

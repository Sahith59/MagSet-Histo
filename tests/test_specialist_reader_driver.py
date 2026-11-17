"""Synthetic end-to-end specialist artifact/gate checks; no encoder or fitting."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from magset import matched_data as md
from magset import prior_centered_model as parent
from magset import specialist_reader as core
from magset.data import file_hash

spec = importlib.util.spec_from_file_location('specialist_driver', Path(__file__).resolve().parents[1] / 'scripts/run_specialist_reader.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def synthetic_state(root):
    rng = np.random.RandomState(204)
    labels = np.arange(15, dtype=np.int64) % 5; truth8 = (0, 1, 3, 4, 5)
    cases = [dict(case_id='case%02d' % i, group='group%02d' % i, truth5=int(labels[i]), truth8=truth8[labels[i]]) for i in range(15)]
    pbags = [[rng.normal(size=(2, 7)).astype(np.float32) + labels[i] for _ in range(4)] for i in range(15)]
    ubags = [[rng.normal(size=(2, 5)).astype(np.float32) + labels[i] for _ in range(4)] for i in range(15)]
    fields = []
    for case in range(15):
        for view, mag in enumerate((40, 100, 200, 400)):
            for field in range(2):
                fields.append(dict(index=len(fields), case_index=case, view=view, magnification=str(mag),
                                   relative_path='case%02d/view%d/field%d' % (case, view, field)))
    audit = dict(cases=cases, fields=fields, roles={'train': list(range(5)), 'validation': list(range(5, 10)), 'test': list(range(10, 15))})
    refs = {}
    for role in ('validation', 'test'):
        indices = np.array([f['index'] for f in fields if f['case_index'] in audit['roles'][role]], np.int64)
        lookup = {int(i): j for j, i in enumerate(indices)}; joint = np.empty((len(indices), 5))
        for query in audit['roles'][role]:
            prepared = parent.prepare_case(pbags, labels, list(range(5)), query, leave_query_case_out=False)
            logs = parent.field_log_probabilities(prepared, .1, 'dual', 'head_specific')
            for view, values in enumerate(logs):
                keys = [f['index'] for f in fields if f['case_index'] == query and f['view'] == view]
                joint[[lookup[i] for i in keys]] = np.exp(values - logsumexp(values, axis=1, keepdims=True))
        path = root / ('parent-' + role + '.npz'); md.save_predictions(path, indices, joint)
        refs[role] = indices, joint, path
    context = dict(code_sha='a' * 40, source_archive_sha256='b' * 64, config_sha256='c' * 64,
                   parent_result_sha256='d' * 64, input_audit_sha256='e' * 64)
    support = core.prepare_support(ubags, labels, list(range(5)))
    state = dict(config={'identity': {'field_probability_atol': 1e-10, 'case_probability_atol': 1e-10},
                         'counts': {'validation_case_subset_records': 600, 'retrospective_case_subset_records': 150}},
                 context=context, audit=audit, p_bags=pbags, uni_bags=ubags, labels=labels, uni_support=support,
                 support_summary={'p_control': parent.support_summary(pbags, labels, list(range(5)), mode='head_specific'),
                                  'uni2_specialist': core.support_summary(support)})
    return state, refs


class SpecialistDriverTests(unittest.TestCase):
    def test_complete_family_seals_both_before_selected_only_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, refs = synthetic_state(root); output = root / 'new-study'
            args = SimpleNamespace(output=output)
            events = []
            def archive(actual, role):
                self.assertIs(actual, state); events.append(role)
                if role == 'test':
                    gate = m.verify_gate(output)
                    self.assertEqual(set(gate['selections']), set(m.ARMS))
                return refs[role]
            with patch.object(m, 'load_inputs', return_value=state), patch.object(m, 'parent_archive', side_effect=archive), \
                 patch.object(m, 'role_predictions', wraps=m.role_predictions) as predict, \
                 patch.object(core, 'prepare_query', wraps=core.prepare_query) as prepare:
                result = m.run(args)
            self.assertEqual(events, ['validation', 'test'])
            self.assertEqual(result['validation_prediction_archives'], 8)
            self.assertEqual(result['selected_retrospective_prediction_archives'], 2)
            self.assertEqual(result['validation_case_subset_records'], 600)
            self.assertEqual(result['retrospective_case_subset_records'], 150)
            self.assertEqual([len(t) for t in predict.call_args_list[0].args[2].values()], [4, 4])
            self.assertEqual([len(t) for t in predict.call_args_list[1].args[2].values()], [1, 1])
            for call in prepare.call_args_list:
                query = call.args[4]
                self.assertEqual('parent_reference_joint_logs' in call.kwargs, query >= 10)
            for arm in m.ARMS:
                folder = output / 'arms' / arm
                selection = m.read_json(folder / 'selection.json')
                self.assertEqual(selection['selected_index'], int(np.argmax(selection['candidate_validation_scores'])))
                self.assertEqual(len(selection['candidate_artifact_sha256']), 4)
                receipt = m.read_json(folder / 'test/test.json')
                self.assertEqual(set(receipt['artifact_sha256']), {'fields.npz', 'records.jsonl', 'metrics.json'})
                self.assertEqual(receipt['binary_invariance']['binary_decision_disagreements'], 0)
            with np.load(output / 'parent-identity/test-fields.npz') as saved:
                np.testing.assert_array_equal(saved['joint_probabilities5'], refs['test'][1])
            self.assertEqual(m.read_json(output / 'parent-identity/test.json')['new_p_specialist_queries'], 0)
            for name, digest in result['artifact_sha256'].items(): self.assertEqual(file_hash(output / name), digest)
            (output / 'arms/p_control/candidates/candidate0/metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed after sealing'): m.verify_gate(output)

    def test_failed_parent_identity_cannot_seal_or_open_test_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state, refs = synthetic_state(root); output = root / 'failed-study'
            original = m.role_predictions
            def altered(*args):
                values = original(*args); values['p_control'][2][0] = np.roll(values['p_control'][2][0], 1)
                return values
            with patch.object(m, 'load_inputs', return_value=state), \
                 patch.object(m, 'parent_archive', return_value=refs['validation']) as archive, \
                 patch.object(m, 'role_predictions', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'identity failed'): m.run(SimpleNamespace(output=output))
            self.assertEqual([call.args[1] for call in archive.call_args_list], ['validation'])
            self.assertFalse((output / 'family-gate.json').exists())
            self.assertFalse((output / 'arms/p_control/selection.json').exists())

    def test_cache_loader_uses_path_lookup_and_rejects_tampered_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fields = []; cases = [dict(case_id='SOB_B_A-syntheticB', truth8=0, group='gB'),
                                                    dict(case_id='SOB_M_DC-syntheticM', truth8=1, group='gM')]
            for case, (prefix, raw, subtype) in enumerate((('SOB_B_A', 'syntheticB', 'adenosis'), ('SOB_M_DC', 'syntheticM', 'ductal_carcinoma'))):
                for view, mag in enumerate((40, 100, 200, 400)):
                    filename = '%s-%s-%d-001.png' % (prefix, raw, mag)
                    fields.append(dict(relative_path='mag%d/' % mag + filename, filename=filename, case_id_raw=raw,
                        subtype=subtype, patient_group_conservative=cases[case]['group'], case_index=case, view=view,
                        magnification=str(mag), sha256='%064x' % len(fields)))
            fields.sort(key=lambda f: f['relative_path'])
            for index, row in enumerate(fields): row['index'] = index
            encoder = {'dimension': 3, 'revision': 'synthetic'}; pins = {}; manifest = '1' * 64
            for mag in (40, 100, 200, 400):
                rows = [f for f in fields if int(f['magnification']) == mag][::-1]
                values = np.array([[f['case_index'], mag, 9] for f in rows], np.float32)
                target = root / ('synthetic-%d.npz' % mag)
                np.savez_compressed(target, paths=np.array([r['relative_path'] for r in rows]), features=values)
                md.write_json(target.with_suffix('.json'), dict(encoder=encoder, manifest_sha256=manifest,
                    feature_sha256=file_hash(target), images=2, smoke_limit=0, magnification=mag,
                    code_sha='2' * 40, config_sha256='3' * 64))
                pins[str(mag)] = {'npz_sha256': file_hash(target), 'json_sha256': file_hash(target.with_suffix('.json')), 'images': 2}
            cfg = dict(encoder_name='synthetic', cache_provenance=pins, feature_producer_code_sha='2' * 40, extraction_config_sha256='3' * 64)
            bags, aligned = m.load_cache(root, cfg, {'encoder': encoder}, fields, cases, manifest, 3, False)
            self.assertEqual([r['relative_path'] for r in aligned], sorted(r['relative_path'] for r in fields))
            for case in range(2):
                for view, mag in enumerate((40, 100, 200, 400)):
                    np.testing.assert_array_equal(bags[case][view], [[case, mag, 9]])
            (root / 'synthetic-200.npz').write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError, 'cache bytes changed'):
                m.load_cache(root, cfg, {'encoder': encoder}, fields, cases, manifest, 3, False)

    def test_output_guard_and_binary_invariance_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'existing'; existing.mkdir()
            with patch.object(m, 'load_inputs') as load:
                with self.assertRaisesRegex(ValueError, 'overwrite'): m.run(SimpleNamespace(output=existing))
                load.assert_not_called()
            state, refs = synthetic_state(Path(tmp)); ix, joint, _ = refs['validation']
            original = m.records(state, 'validation', ix, joint); changed = joint.copy()
            changed[0, 0] += 1e-4; changed[0, 1:] *= (1 - changed[0, 0]) / joint[0, 1:].sum()
            altered = m.records(state, 'validation', ix, changed)
            with self.assertRaisesRegex(ValueError, 'binary marginals'):
                m.binary_checks(joint, changed, original, altered)


if __name__ == '__main__':
    unittest.main()

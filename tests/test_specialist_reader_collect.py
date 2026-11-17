"""Independent collector arithmetic, provenance and fail-closed regression checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import matched_data as producer_records
from magset import prior_centered_model as parent_math
from magset import specialist_reader as producer_math

spec = importlib.util.spec_from_file_location('specialist_collector', Path(__file__).resolve().parents[1]/'scripts/collect_specialist_reader.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(p_dimension=7, uni_dimension=5):
    rng = np.random.RandomState(72)
    cases = [dict(case_id='synthetic-%02d' % i, group='synthetic-group-%02d' % i,
                  truth5=i % 5, truth8=(0, 1, 3, 4, 5)[i % 5]) for i in range(20)]
    fields = []
    for case in range(20):
        for view, mag in enumerate(m.metrics.MAGNIFICATIONS):
            for row in range(1+(case+view) % 3):
                fields.append(dict(index=len(fields), case_index=case, view=view,
                                   magnification=str(mag), relative_path='synthetic/%02d/%d/%d.png' % (case, mag, row),
                                   sha256=('%064x' % (len(fields)+1))))
    labels = np.asarray([cases[f['case_index']]['truth5'] for f in fields])
    p = (rng.normal(size=(len(fields), p_dimension))+.4*labels[:, None]).astype(np.float32)
    u = (rng.normal(size=(len(fields), uni_dimension))+.6*labels[:, None]).astype(np.float32)
    audit = dict(cases=cases, fields=fields, roles={'train': list(range(10)), 'validation': list(range(10, 15)), 'test': list(range(15, 20))},
                 manifest_sha256='f'*64, coverage_cache='/synthetic/p-cache')
    bags = lambda values: [[values[[f['index'] for f in fields if f['case_index'] == case and f['view'] == view]]
                            for view in range(4)] for case in range(20)]
    return audit, p, u, bags(p), bags(u)


def producer_predictions(audit, pbags, ubags, role, settings):
    labels = np.asarray([c['truth5'] for c in audit['cases']], dtype=np.int64)
    fields = audit['fields']
    indices = np.asarray([f['index'] for f in fields if f['case_index'] in set(audit['roles'][role])], dtype=np.int64)
    positions = {int(index): row for row, index in enumerate(indices)}
    outputs = {(arm, t): np.empty((len(indices), 5)) for arm, temps in settings.items() for t in temps}
    support = producer_math.prepare_support(ubags, labels, audit['roles']['train'])
    for case in audit['roles'][role]:
        query = producer_math.prepare_query(pbags, ubags, labels, audit['roles']['train'], case, support)
        for arm, temps in settings.items():
            for temperature in temps:
                values = producer_math.field_probabilities(query, temperature, 'uni2_c' if arm == 'uni2_specialist' else arm)
                for view in range(4):
                    selected = [f['index'] for f in fields if f['case_index'] == case and f['view'] == view]
                    outputs[arm, temperature][[positions[i] for i in selected]] = values[view]
    return indices, outputs


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def archive(path, indices, values):
    np.savez_compressed(path, field_indices=indices.astype(np.int64), joint_probabilities5=values.astype(np.float64))


def summary_fixture(path, audit, role, indices, values, setting=None):
    path.mkdir(parents=True, exist_ok=True)
    prefix = 'validation-' if role == 'validation' else ''
    archive(path/(prefix+'fields.npz'), indices, values)
    records = producer_records.case_records(audit['fields'], audit['cases'], audit['roles'][role], indices, values, role)
    producer_records.save_records(path/(prefix+'records.jsonl'), records)
    summary = producer_records.summary(records)
    write(path/'metrics.json', {'setting': setting, 'validation': summary} if role == 'validation' else summary)
    return summary


def complete_fixture(tmp):
    root = Path(tmp)/'family'; root.mkdir()
    audit, p, u, pbags, ubags = fixture(3072, 1536)
    cfg = json.loads(Path('configs/breakhis-specialist-reader-v1.json').read_text())
    public = {'code_sha': 'a'*40, 'source_archive_sha256': 'b'*64,
              'config_sha256': 'c'*64, 'parent_result_sha256': 'd'*64,
              'input_audit_sha256': cfg['input_audit']['private_sha256']}
    context = dict(public, p_cache='/synthetic/p', uni_cache='/synthetic/u')
    write(root/'context.json', context)
    context_sha = m.file_hash(root/'context.json')
    labels = np.asarray([c['truth5'] for c in audit['cases']], dtype=np.int64)
    support = {'p_control': parent_math.support_summary(pbags, labels, audit['roles']['train'], mode='head_specific'),
               'uni2_specialist': producer_math.support_summary(producer_math.prepare_support(ubags, labels, audit['roles']['train']))}
    write(root/'support-summary.json', support)
    support_sha = m.file_hash(root/'support-summary.json')
    val_indices, values = producer_predictions(audit, pbags, ubags, 'validation', {a: m.TEMPERATURES for a in m.ARMS})
    parent_val = values['p_control', .1]
    identity = root/'parent-identity'; identity.mkdir()
    archive(identity/'validation-fields.npz', val_indices, parent_val)
    parent_validation = Path(tmp)/'parent/models/midnight_p/candidates/candidate2/validation-fields.npz'
    parent_validation.parent.mkdir(parents=True)
    archive(parent_validation, val_indices, parent_val)
    write(identity/'validation.json', {'status': 'accepted rederived parent P validation identity',
          'parent_archive_sha256': m.file_hash(parent_validation), 'maximum_field_probability_abs': 0.,
          'maximum_case_probability_abs': 0., 'decision_disagreements': 0, 'test_queries': 0})
    selections, hashes = {}, {}
    for arm in m.ARMS:
        scores, receipts = [], []
        folder = root/'arms'/arm
        for i, temperature in enumerate(m.TEMPERATURES):
            target = folder/'candidates'/('candidate%d' % i)
            summary = summary_fixture(target, audit, 'validation', val_indices, values[arm, temperature], temperature)
            scores.append(summary['mean_subset_macro_f1'])
            receipts.append({name: m.file_hash(target/name) for name in m.CANDIDATE_FILES})
        selected = int(np.argmax(scores))
        selection = dict(context, status='selected validation candidate', arm=arm, test_predictions=0,
                         context_sha256=context_sha, support_summary_sha256=support_sha,
                         candidate_validation_scores=scores, selected_index=selected,
                         selected_setting=m.TEMPERATURES[selected], candidate_artifact_sha256=receipts,
                         validation_identity_sha256=m.file_hash(identity/'validation.json'))
        write(folder/'selection.json', selection)
        selections[arm] = selection; hashes[arm] = m.file_hash(folder/'selection.json')
    gate = {'status': 'sealed both specialist selections', 'test_queries': 0, 'selections': hashes,
            'context_sha256': context_sha, 'support_summary_sha256': support_sha,
            'input_audit_sha256': public['input_audit_sha256']}
    write(root/'family-gate.json', gate)
    settings = {a: (selections[a]['selected_setting'],) for a in m.ARMS}
    test_indices, test_values = producer_predictions(audit, pbags, ubags, 'test', settings)
    _, parent_reference = producer_predictions(audit, pbags, ubags, 'test', {'p_control': (.1,)})
    for arm in m.ARMS:
        selection = selections[arm]; folder = root/'arms'/arm/'test'
        summary_fixture(folder, audit, 'test', test_indices, test_values[arm, selection['selected_setting']])
        receipt = dict(status='complete selected retrospective inference', arm=arm,
                       selected_index=selection['selected_index'], selected_setting=selection['selected_setting'],
                       selection_sha256=hashes[arm], gate_sha256=m.file_hash(root/'family-gate.json'),
                       context_sha256=context_sha, support_summary_sha256=support_sha,
                       artifact_sha256={name: m.file_hash(folder/name) for name in ('fields.npz', 'records.jsonl', 'metrics.json')})
        write(folder/'test.json', receipt)
    args = SimpleNamespace(root=root, parent=Path(tmp)/'parent', config=Path('configs/breakhis-specialist-reader-v1.json'),
                           code_sha='a'*40, output=Path(tmp)/'checked')
    references = {'validation': (val_indices, parent_val), 'test': (test_indices, parent_reference['p_control', .1])}
    counts = {r: {'cases': len(ix), 'groups': len(ix), 'fields': sum(f['case_index'] in set(ix) for f in audit['fields'])}
              for r, ix in audit['roles'].items()}
    return args, cfg, context, public, audit, p, u, references, counts


class SpecialistCollectorTests(unittest.TestCase):
    def test_independent_centers_support_hashes_and_all_eight_densities_match_producer(self):
        audit, p, u, pbags, ubags = fixture()
        prepared = m.geometry(audit, p, u)
        labels = np.asarray([c['truth5'] for c in audit['cases']], dtype=np.int64)
        summary = {'p_control': parent_math.support_summary(pbags, labels, audit['roles']['train'], mode='head_specific'),
                   'uni2_specialist': producer_math.support_summary(producer_math.prepare_support(ubags, labels, audit['roles']['train']))}
        self.assertEqual(m.verify_support(summary, prepared, audit, 7, 5)['malignant_cases'], 8)
        indices, actual = m.replay_densities(audit, p, u, prepared, 'validation', {a: m.TEMPERATURES for a in m.ARMS})
        wanted_indices, wanted = producer_predictions(audit, pbags, ubags, 'validation', {a: m.TEMPERATURES for a in m.ARMS})
        np.testing.assert_array_equal(indices, wanted_indices)
        for key in wanted:
            np.testing.assert_allclose(actual[key], wanted[key], rtol=0, atol=1e-12)
        bad = copy.deepcopy(summary); bad['uni2_specialist']['centers_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'UNI support'): m.verify_support(bad, prepared, audit, 7, 5)

    def test_statistics_ignore_held_cases_and_retain_equal_case_centers(self):
        audit, p, u, _, _ = fixture()
        original = m.geometry(audit, p, u)
        changed_p, changed_u = p.copy(), u.copy()
        held = [f['index'] for f in audit['fields'] if f['case_index'] not in audit['roles']['train']]
        changed_p[held] += 1000; changed_u[held] -= 1000
        changed = m.geometry(audit, changed_p, changed_u)
        for head in ('binary', 'p', 'uni'):
            self.assertEqual(m.array_digest(original['centers'][head]), m.array_digest(changed['centers'][head]))
            self.assertEqual(m.array_digest(original['support'][head]), m.array_digest(changed['support'][head]))
        for view in range(4):
            means = [u[[f['index'] for f in audit['fields'] if f['case_index'] == i and f['view'] == view]].astype(np.float64).mean(0)
                     for i in audit['roles']['train']]
            np.testing.assert_array_equal(original['centers']['uni'][view], np.mean(means, axis=0))

    def test_zero_norm_and_stabilized_large_cosines_remain_finite(self):
        np.testing.assert_array_equal(m.normalize(np.ones((2, 3)), np.ones(3)), np.zeros((2, 3)))
        weights = np.array([[.5, 0], [.5, 0], [0, 1]], dtype=np.float64)
        logs = m.density_logs(np.array([[-1., 1., .4], [1., -1., -.8]]), weights, .01)
        self.assertTrue(np.isfinite(logs).all())
        np.testing.assert_allclose(np.exp(logs).sum(1), 1, rtol=0, atol=1e-14)

    def test_nonfinite_raw_cosine_is_rejected_before_clipping(self):
        audit, p, u, _, _ = fixture()
        prepared = m.geometry(audit, p, u)
        prepared['support']['binary'][0][:] = np.inf
        with np.errstate(all='ignore'), self.assertRaisesRegex(ValueError, 'before clipping'):
            m.replay_densities(audit, p, u, prepared, 'validation', {'p_control': (.1,)})

    def test_binary_invariance_checks_every_field_and_hard_half_route(self):
        audit, _, _, _, _ = fixture()
        indices = np.asarray([f['index'] for f in audit['fields'] if f['case_index'] in audit['roles']['validation']], np.int64)
        reference = np.tile([.5, .125, .125, .125, .125], (len(indices), 1))
        changed_type = np.tile([.5, .5, 0, 0, 0], (len(indices), 1))
        result = m.fixed_binary(audit, 'validation', indices, changed_type, reference)
        self.assertEqual(result['hard_route_disagreements'], 0)
        changed = changed_type.copy(); changed[0, :2] += [.001, -.001]
        with self.assertRaisesRegex(ValueError, 'field marginal'): m.fixed_binary(audit, 'validation', indices, changed, reference)
        changed = np.tile([.5000000000001, .4999999999999, 0, 0, 0], (len(indices), 1))
        with self.assertRaisesRegex(ValueError, 'hard routing'): m.fixed_binary(audit, 'validation', indices, changed, reference)

    def test_identity_compares_full_five_distribution_not_just_binary(self):
        audit, _, _, _, _ = fixture()
        indices = np.asarray([f['index'] for f in audit['fields'] if f['case_index'] in audit['roles']['validation']], np.int64)
        q = np.tile([.2, .5, .1, .1, .1], (len(indices), 1))
        self.assertTrue(m.identity_check(audit, indices, q, q)['confusions_identical'])
        other = q[:, [0, 2, 1, 3, 4]]
        with self.assertRaisesRegex(ValueError, 'identity field'): m.identity_check(audit, indices, other, q)

    def test_archive_rejects_order_schema_dtype_probability_and_duplicates(self):
        audit, _, _, _, _ = fixture()
        indices = np.asarray([f['index'] for f in audit['fields'] if f['case_index'] in audit['roles']['validation']], np.int64)
        q = np.tile([.2]*5, (len(indices), 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fields.npz'; archive(path, indices, q)
            self.assertEqual(m.load_field_archive(path, audit, 'validation')[1].shape, q.shape)
            for values in [dict(field_indices=indices[::-1], joint_probabilities5=q),
                           dict(field_indices=indices.astype(float), joint_probabilities5=q),
                           dict(field_indices=indices, joint_probabilities5=q.astype(np.float32)),
                           dict(field_indices=indices, joint_probabilities5=q*2),
                           dict(field_indices=indices, joint_probabilities5=q, extra=np.ones(1))]:
                np.savez_compressed(path, **values)
                with self.assertRaises(ValueError): m.load_field_archive(path, audit, 'validation')

    def test_first_max_selection_and_both_arm_gate_bind_hashes(self):
        context = dict(zip(m.CONTEXT_KEYS, ['a'*40, 'b'*64, 'c'*64, 'd'*64, 'e'*64]))
        selection = dict(context, status='selected validation candidate', arm='uni2_specialist', test_predictions=0,
                         context_sha256='1'*64, support_summary_sha256='2'*64, selected_index=1, selected_setting=.03,
                         candidate_validation_scores=[.1, .8, .8, .7], candidate_artifact_sha256=[{}]*4)
        self.assertEqual(m.verify_selection(selection, [.1, .8, .8, .7], 'uni2_specialist', context, '1'*64, '2'*64), 1)
        for mutation in [dict(selected_index=2, selected_setting=.1), dict(test_predictions=1), dict(support_summary_sha256='3'*64)]:
            with self.assertRaises(ValueError): m.verify_selection(dict(selection, **mutation), [.1, .8, .8, .7], 'uni2_specialist', context, '1'*64, '2'*64)
        hashes = {'p_control': '3'*64, 'uni2_specialist': '4'*64}
        gate = dict(status='sealed both specialist selections', test_queries=0, selections=hashes,
                    context_sha256='1'*64, input_audit_sha256='e'*64, support_summary_sha256='2'*64)
        m.verify_gate(gate, hashes, '1'*64, 'e'*64, '2'*64)
        for mutation in [dict(selections={'p_control': '3'*64}), dict(test_queries=1),
                         dict(context_sha256='0'*64), dict(support_summary_sha256='0'*64)]:
            with self.assertRaises(ValueError): m.verify_gate(dict(gate, **mutation), hashes, '1'*64, 'e'*64, '2'*64)

    def test_alignment_digest_binds_paths_bytes_groups_views_and_original_labels(self):
        audit, _, _, _, _ = fixture()
        original = m.alignment_digest(audit)
        for target, key, value in [('field', 'sha256', '0'*64), ('field', 'relative_path', 'other'),
                                   ('field', 'magnification', '400'), ('case', 'group', 'other'), ('case', 'truth8', 1)]:
            changed = copy.deepcopy(audit)
            changed['fields' if target == 'field' else 'cases'][0][key] = value
            self.assertNotEqual(m.alignment_digest(changed), original)

    def test_cache_loader_aligns_shuffled_paths_and_rejects_changed_bytes_or_lineage(self):
        audit, p, _, _, _ = fixture()
        reference = {'encoder_name': 'synthetic', 'cache_provenance': {}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for mag in m.metrics.MAGNIFICATIONS:
                indices = [f['index'] for f in audit['fields'] if int(f['magnification']) == mag][::-1]
                path = folder/('synthetic-%d.npz' % mag)
                np.savez_compressed(path, paths=np.asarray([audit['fields'][i]['relative_path'] for i in indices]), features=p[indices])
                metadata = {'manifest_sha256': audit['manifest_sha256'], 'magnification': mag,
                            'feature_sha256': m.file_hash(path)}
                write(path.with_suffix('.json'), metadata)
                reference['cache_provenance'][str(mag)] = {'npz_sha256': m.file_hash(path),
                                                          'json_sha256': m.file_hash(path.with_suffix('.json'))}
            np.testing.assert_array_equal(m.load_features(folder, reference, audit, 7), p)
            path = folder/'synthetic-40.npz'
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'cache bytes'): m.load_features(folder, reference, audit, 7)
            path.write_bytes(original)
            metadata_path = path.with_suffix('.json')
            metadata = json.loads(metadata_path.read_text()); metadata['manifest_sha256'] = '0'*64
            write(metadata_path, metadata)
            reference['cache_provenance']['40']['json_sha256'] = m.file_hash(metadata_path)
            with self.assertRaisesRegex(ValueError, 'metadata lineage'): m.load_features(folder, reference, audit, 7)

    def test_complete_mocked_family_replays_without_encoder_or_new_core_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, context, public, audit, p, u, references, counts = complete_fixture(tmp)
            with patch.object(m, 'source_context', return_value=(cfg, context, public, Path(tmp))), \
                 patch.object(m, 'parent_inputs', return_value=(audit, counts, references)), \
                 patch.object(m, 'verify_dataset_context', return_value=({}, {})), \
                 patch.object(m, 'load_features', side_effect=[p, u]):
                result = m.collect(args)
            self.assertEqual(result['status'], 'complete accepted specialist-reader family', (args.output/'private_review.json').read_text())
            self.assertEqual(set(result['arms']), set(m.ARMS))
            self.assertEqual(json.loads((args.output/'private_review.json').read_text())['errors'], {})
            self.assertEqual(result['collector_new_encoder_forwards'], 0)
            self.assertIn('retrospective', result['interpretation'])
            with self.assertRaisesRegex(ValueError, 'existing'): m.collect(args)

    def test_late_failure_does_not_publish_partial_validation_or_follow_up_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, context, public, audit, p, u, references, counts = complete_fixture(tmp)
            receipt = args.root/'arms/uni2_specialist/test/test.json'
            value = json.loads(receipt.read_text()); value['gate_sha256'] = '0'*64; write(receipt, value)
            with patch.object(m, 'source_context', return_value=(cfg, context, public, Path(tmp))), \
                 patch.object(m, 'parent_inputs', return_value=(audit, counts, references)), \
                 patch.object(m, 'verify_dataset_context', return_value=({}, {})), \
                 patch.object(m, 'load_features', side_effect=[p, u]):
                result = m.collect(args)
            self.assertEqual(result['status'], 'failed or incomplete specialist-reader family')
            self.assertNotIn('arms', result)
            self.assertNotIn('progression_screen', result)
            self.assertIn('review', json.loads((args.output/'private_review.json').read_text())['errors'])
            self.assertIn('No model performance is accepted', (args.output/'summary.md').read_text())

    def test_changed_identity_receipt_invalidates_both_selection_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, context, public, audit, p, u, references, counts = complete_fixture(tmp)
            receipt = args.root/'parent-identity/validation.json'
            value = json.loads(receipt.read_text()); value['test_queries'] = 1; write(receipt, value)
            with patch.object(m, 'source_context', return_value=(cfg, context, public, Path(tmp))), \
                 patch.object(m, 'parent_inputs', return_value=(audit, counts, references)), \
                 patch.object(m, 'verify_dataset_context', return_value=({}, {})), \
                 patch.object(m, 'load_features', side_effect=[p, u]):
                result = m.collect(args)
            self.assertNotIn('arms', result)
            error = json.loads((args.output/'private_review.json').read_text())['errors']['review']['message']
            self.assertIn('selection/validation identity bytes', error)

    def test_collector_source_never_imports_specialist_producer_math(self):
        source = Path(m.__file__).read_text()
        self.assertNotIn('import specialist_reader', source)
        self.assertNotIn('from magset.specialist_reader', source)
        args = m.parser().parse_args(['--root', 'family', '--parent', 'parent', '--config', 'recipe', '--code-sha', 'a'*40, '--output', 'checked'])
        self.assertEqual(args.parent, Path('parent'))


if __name__ == '__main__':
    unittest.main()

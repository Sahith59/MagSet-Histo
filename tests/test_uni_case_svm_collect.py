"""Synthetic checks only: no study fitting, private features or test inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.svm import SVC
from magset import uni_case_svm as synthetic_producer

spec = importlib.util.spec_from_file_location('uni_case_collector', Path(__file__).resolve().parents[1]/'scripts/collect_uni_case_svm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic():
    rng = np.random.RandomState(812)
    cases = [{'case_id': 'synthetic-%02d' % i, 'group': 'synthetic-%02d' % i,
              'truth5': i % 5, 'truth8': (0, 1, 3, 4, 5)[i % 5]} for i in range(20)]
    fields = []; features = []
    for case in range(20):
        for view, mag in enumerate(m.metrics.MAGNIFICATIONS):
            for field in range(1+(case+view) % 3):
                fields.append({'index': len(fields), 'case_index': case, 'view': view, 'magnification': str(mag),
                               'relative_path': 'synthetic/%d/%d/%d.png' % (case, mag, field),
                               'sha256': '%064x' % (len(fields)+1)})
                value = rng.normal(size=7); value[0] += case % 5; value[-1] = 1
                features.append(value)
    audit = {'cases': cases, 'fields': fields,
             'roles': {'train': list(range(10)), 'validation': list(range(10, 15)), 'test': list(range(15, 20))}}
    return audit, np.asarray(features, dtype=np.float32)


def fit_synthetic(kernel, c=.1):
    audit, features = synthetic()
    means = m.case_means(audit, features)
    train = np.asarray(audit['roles']['train'], np.int64)
    mean, std, scale = m.normalization(means, train)
    x = m.descriptors(means, mean, scale, train)
    labels = np.array([case['truth5'] for case in audit['cases']], np.int64)
    result = []
    for count in (2, 4):
        selected = train if count == 2 else train[labels[train] > 0]
        y = np.repeat((labels[selected] > 0).astype(np.int64) if count == 2 else labels[selected]-1, 15)
        inputs = m.descriptors(means, mean, scale, selected)
        model = SVC(kernel=kernel, C=c, gamma='scale', class_weight='balanced', probability=False,
                    decision_function_shape='ovr', break_ties=True, random_state=812,
                    tol=1e-4, shrinking=True, max_iter=100000, cache_size=256)
        weights = np.full(len(y), 1/15, np.float64)
        model.fit(inputs, y, sample_weight=weights)
        state = {key: np.asarray(getattr(model, key)).copy() for key in
                 ('support_vectors_', 'dual_coef_', 'intercept_', 'n_support_', 'support_', 'classes_', 'class_weight_', 'n_iter_')}
        for key in ('n_support_', 'support_', 'classes_', 'n_iter_'):
            state[key] = state[key].astype(np.int64)
        effective = weights*model.class_weight_[y]
        state.update(actual_gamma=np.array(model._gamma, np.float64), fit_status_=np.array(model.fit_status_, np.int64),
                     training_case_indices=selected.copy(), support_case_indices=np.repeat(selected, 15)[model.support_],
                     training_features_sha256=np.array(m.array_hash(inputs)),
                     training_targets_sha256=np.array(hashlib.sha256(y.astype('<i8').tobytes()).hexdigest()),
                     sample_weight_sha256=np.array(m.array_hash(weights)), effective_weight_sha256=np.array(m.array_hash(effective)),
                     sample_weight_sum=np.array(weights.sum()), effective_weight_sum=np.array(effective.sum()))
        result.append((state, model, inputs, y, selected))
    return audit, features, means, (mean, std, scale), result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False)+'\n')


def complete_family(tmp):
    """Full production-shaped, sparse synthetic data; no private-study inputs."""
    cfg = json.loads(Path('configs/breakhis-uni-case-svm-v1.json').read_text())
    rng = np.random.RandomState(812)
    cases = []; roles = {}; fields = []; values = []
    for role, counts in cfg['expected_roles'].items():
        start = len(cases)
        for label, count in enumerate(counts['class_cases']):
            for _ in range(count):
                index = len(cases)
                cases.append({'case_id': 'synthetic-%02d' % index, 'group': 'synthetic-%02d' % index,
                              'truth5': label, 'truth8': (0, 1, 3, 4, 5)[label]})
                for view, mag in enumerate(m.metrics.MAGNIFICATIONS):
                    fields.append({'index': len(fields), 'case_index': index, 'view': view,
                                   'magnification': str(mag), 'relative_path': 'synthetic/%d/%d.png' % (index, mag),
                                   'sha256': '%064x' % (len(fields)+1)})
                    embedding = np.zeros(1536, np.float32)
                    embedding[:7] = rng.normal(size=7); embedding[label] += 2
                    values.append(embedding)
        roles[role] = list(range(start, len(cases)))
    audit = {'cases': cases, 'fields': fields, 'roles': roles}
    features = np.asarray(values, np.float32)
    means = m.case_means(audit, features)
    train = np.asarray(roles['train'], np.int64)
    labels = np.asarray([c['truth5'] for c in cases], np.int64)
    preprocessing = synthetic_producer.fit_preprocessing(means, train)
    root = Path(tmp)/'family'; root.mkdir()
    np.savez_compressed(root/'preprocessing.npz', case_view_means=means,
                        **preprocessing, case_indices=np.arange(len(cases), dtype=np.int64), masks=m.MASKS)
    preprocessing_sha = m.file_hash(root/'preprocessing.npz')
    public = dict(zip(m.CONTEXT_KEYS, ['a'*40]+[str(i)*64 for i in range(1, 6)]))
    context = dict(public, preprocessing_sha256=preprocessing_sha)
    write_json(root/'context.json', context); context_sha = m.file_hash(root/'context.json')
    selections = {}; selection_hashes = {}; bundles = {}
    for arm in m.ARMS:
        scores = []; hashes = []; by_index = []
        for index, c in enumerate(m.CS):
            folder = root/'candidates'/arm/('candidate%d' % index); folder.mkdir(parents=True)
            bundle = synthetic_producer.fit_bundle(means, labels, preprocessing, arm, c)
            joblib.dump(bundle, folder/'model.joblib', compress=3)
            np.savez_compressed(folder/'state.npz', **synthetic_producer.state_arrays(bundle))
            archive = synthetic_producer.predict(bundle, means, roles['validation'])
            np.savez_compressed(folder/'validation-probabilities.npz', **archive)
            records = synthetic_producer.case_records(cases, archive, 'validation')
            (folder/'validation-records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            metric = synthetic_producer.summary(cases, archive)
            write_json(folder/'metrics.json', {'setting': c, 'validation': metric})
            write_json(folder/'reload.json', {'status': 'accepted validation model reload', 'binary_margin_max_abs': 0.,
                'malignant_score_max_abs': 0., 'joint_probability_max_abs': 0., 'atol': 1e-10, 'decisions_identical': True,
                'validation_cases': len(roles['validation']), 'masks': 15, 'test_queries': 0,
                'model_sha256': m.file_hash(folder/'model.joblib'), 'state_sha256': m.file_hash(folder/'state.npz')})
            scores.append(metric['mean_subset_macro_f1'])
            hashes.append({name: m.file_hash(folder/name) for name in m.CANDIDATE_FILES}); by_index.append(bundle)
        index = int(np.argmax(scores)); folder = root/'arms'/arm; folder.mkdir(parents=True)
        selection = dict(context, status='selected validation candidate', arm=arm, context_sha256=context_sha,
                         selected_index=index, selected_setting=m.CS[index], candidate_validation_scores=scores,
                         candidate_artifact_sha256=hashes, test_predictions=0)
        write_json(folder/'selection.json', selection)
        selections[arm] = selection; selection_hashes[arm] = m.file_hash(folder/'selection.json'); bundles[arm] = by_index[index]
    gate = {'status': 'sealed both case-SVC selections', 'test_queries': 0, 'selections': selection_hashes,
            'context_sha256': context_sha, 'preprocessing_sha256': preprocessing_sha,
            'input_audit_sha256': cfg['input_audit']['private_sha256']}
    write_json(root/'family-gate.json', gate)
    for arm in m.ARMS:
        folder = root/'arms'/arm/'test'; folder.mkdir()
        archive = synthetic_producer.predict(bundles[arm], means, roles['test'])
        np.savez_compressed(folder/'probabilities.npz', **archive)
        records = synthetic_producer.case_records(cases, archive, 'test')
        (folder/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        write_json(folder/'metrics.json', synthetic_producer.summary(cases, archive))
        selection = selections[arm]; candidate = root/'candidates'/arm/('candidate%d' % selection['selected_index'])
        write_json(folder/'test.json', {'status': 'complete selected retrospective inference', 'arm': arm,
            'selected_index': selection['selected_index'], 'selected_setting': selection['selected_setting'],
            'context_sha256': context_sha, 'preprocessing_sha256': preprocessing_sha,
            'selection_sha256': selection_hashes[arm], 'gate_sha256': m.file_hash(root/'family-gate.json'),
            'selected_model_sha256': m.file_hash(candidate/'model.joblib'), 'selected_state_sha256': m.file_hash(candidate/'state.npz'),
            'artifact_sha256': {name: m.file_hash(folder/name) for name in ('probabilities.npz', 'records.jsonl', 'metrics.json')}})
    counts = {role: {'cases': len(indices), 'groups': len(indices), 'fields': 4*len(indices),
                     'class_cases': np.bincount(labels[indices], minlength=5).tolist()} for role, indices in roles.items()}
    args = SimpleNamespace(root=root, parent=Path(tmp)/'parent', config=Path('configs/breakhis-uni-case-svm-v1.json'),
                           code_sha='a'*40, output=Path(tmp)/'checked')
    return args, cfg, context, public, audit, counts, features


class UniCaseCollectorTests(unittest.TestCase):
    def test_population_normalizer_uses_only_fitting_case_means_and_replaces_small_std(self):
        audit, features = synthetic()
        means = m.case_means(audit, features)
        original = m.normalization(means, audit['roles']['train'])
        changed = means.copy(); changed[10:] += 10000
        for actual, wanted in zip(m.normalization(changed, audit['roles']['train']), original):
            np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(original[1][:, -1], 0)
        np.testing.assert_array_equal(original[2][:, -1], 1)
        np.testing.assert_array_equal(original[0], means[:10].mean(0))
        np.testing.assert_array_equal(original[1], means[:10].std(0, ddof=0))

    def test_case_major_mask_order_norm_then_flags_and_inactive_view_invariance(self):
        audit, features = synthetic(); means = m.case_means(audit, features)
        mean, _, scale = m.normalization(means, audit['roles']['train'])
        actual = m.descriptors(means, mean, scale, [10, 11]).reshape(2, 15, -1)
        np.testing.assert_array_equal(actual[:, :, -4:], np.broadcast_to(.5*m.MASKS, (2, 15, 4)))
        np.testing.assert_allclose(np.linalg.norm(actual[:, :, :-4], axis=-1), 1, rtol=0, atol=1e-14)
        self.assertGreater(np.linalg.norm(actual[0, -1]), 1.)  # no renormalization after flags
        for column, subset in enumerate(m.SUBSETS):
            changed = means.copy()
            for view in set(range(4))-set(subset): changed[10:12, view] += 1000
            np.testing.assert_array_equal(m.descriptors(changed, mean, scale, [10, 11]).reshape(2, 15, -1)[:, column], actual[:, column])

    def test_preprocessing_archive_requires_exact_train_only_arrays_masks_and_dtypes(self):
        audit, features = synthetic(); means = m.case_means(audit, features)
        mean, std, scale = m.normalization(means, audit['roles']['train'])
        arrays = dict(case_view_means=means, mean=mean, std=std, scale=scale,
                      train_case_indices=np.array(audit['roles']['train'], np.int64),
                      case_indices=np.arange(len(audit['cases']), dtype=np.int64), masks=m.MASKS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'preprocessing.npz'; np.savez_compressed(path, **arrays)
            self.assertEqual(m.verify_preprocessing(path, audit, features)[0]['train'].shape, (150, 11))
            for key, value in [('mean', mean+1), ('masks', m.MASKS[::-1]), ('train_case_indices', np.array(audit['roles']['train'], float))]:
                np.savez_compressed(path, **dict(arrays, **{key: value}))
                with self.assertRaises(ValueError): m.verify_preprocessing(path, audit, features)

    def test_dense_binary_and_ovo_ovr_replay_match_sklearn_both_registered_kernels(self):
        for kernel in m.ARMS:
            audit, _, means, normalizer, heads = fit_synthetic(kernel)
            query = m.descriptors(means, normalizer[0], normalizer[2], audit['roles']['validation'])
            for state, model, inputs, labels, cases in heads:
                proof = m.validate_head(state, model, inputs, labels, cases, kernel, .1)
                self.assertEqual(proof['training_cases'], len(cases))
                actual = m.raw_margins(state, query, kernel)
                if len(state['classes_']) == 4:
                    ovr = m.ovr_from_ovo(actual)
                    np.testing.assert_allclose(ovr, model.decision_function(query), rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(ovr.argmax(1), model.predict(query))
                    model.decision_function_shape = 'ovo'
                np.testing.assert_allclose(actual, model.decision_function(query), rtol=0, atol=1e-12)

    def test_head_rejects_wrong_weights_support_gamma_nonconvergence_and_exports(self):
        _, _, _, _, heads = fit_synthetic('rbf')
        state, model, x, y, cases = heads[0]
        for key, value in [('sample_weight_sha256', np.array('0'*64)), ('training_targets_sha256', np.array('0'*64)),
                           ('support_vectors_', state['support_vectors_']+1e-3), ('actual_gamma', state['actual_gamma']+1),
                           ('fit_status_', np.array(1)), ('classes_', state['classes_'][::-1])]:
            changed = dict(state, **{key: value})
            with self.assertRaises(ValueError): m.validate_head(changed, model, x, y, cases, 'rbf', .1)

    def test_score_transform_margin_ties_and_nonfinite_preclip_kernel_rejection(self):
        margins = np.array([-2., 0., 2.]); scores = np.zeros((3, 4))
        joint, labels = m.scores_and_labels(margins, scores)
        np.testing.assert_array_equal(labels, [0, 1, 1])
        np.testing.assert_allclose(joint.sum(1), 1, rtol=0, atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'margin route'):
            m.scores_and_labels(np.array([-1e-30]), np.zeros((1, 4)))
        with np.errstate(all='ignore'), self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.kernel_matrix(np.array([[np.inf]]), np.ones((1, 1)), 'rbf', 1.)

    def test_ovr_ties_and_pairwise_auc_agree_with_independent_definitions(self):
        np.testing.assert_array_equal(m.ovr_from_ovo(np.zeros((1, 6))).argmax(1), [0])
        self.assertEqual(m.auc(np.array([0, 1, 0, 1]), np.array([0., 1., 1., 2.])), .875)
        self.assertIsNone(m.auc(np.ones(2), np.zeros(2)))

    def test_independent_summary_matches_counts_and_fixed_five_class_policy(self):
        audit, _ = synthetic(); y = np.array([case['truth5'] for case in audit['cases'][10:15]])
        prediction = np.repeat(y[:, None], 15, axis=1)
        result = m.summary(audit, 'validation', prediction.astype(float), prediction)
        self.assertEqual(result['mean_subset_accuracy'], 1.)
        self.assertEqual(result['mean_subset_macro_f1'], 1.)
        self.assertEqual(result['case_subset_events'], 75)
        self.assertEqual(result['all_four']['binary']['roc_auc'], 1.)

    def test_full_synthetic_family_and_late_failure_publish_no_partial_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, context, public, audit, counts, features = complete_family(tmp)
            with patch.object(m, 'source_context', return_value=(cfg, context, public, Path(tmp))), \
                 patch.object(m, 'dataset_inputs', return_value=(audit, counts, features)):
                result = m.collect(args)
            self.assertEqual(result['status'], 'complete accepted UNI case-SVC family', (args.output/'private_review.json').read_text())
            self.assertEqual(set(result['arms']), set(m.ARMS))
            self.assertEqual(result['collector_new_fits'], 0)
            self.assertEqual(result['binary_optimization_subproblems'], 56)
            self.assertEqual(json.loads((args.output/'private_review.json').read_text())['errors'], {})
            receipt = args.root/'arms/rbf/test/test.json'; changed = json.loads(receipt.read_text())
            changed['selected_state_sha256'] = '0'*64; write_json(receipt, changed)
            args.output = Path(tmp)/'failed-check'
            with patch.object(m, 'source_context', return_value=(cfg, context, public, Path(tmp))), \
                 patch.object(m, 'dataset_inputs', return_value=(audit, counts, features)):
                result = m.collect(args)
            self.assertEqual(result['status'], 'failed or incomplete UNI case-SVC family')
            self.assertNotIn('arms', result)
            self.assertNotIn('progression_screen', result)
            self.assertIn('No model performance is accepted', (args.output/'summary.md').read_text())

    def test_selection_gate_and_reload_reject_ties_wrong_hashes_or_failed_guards(self):
        context = dict(zip(m.CONTEXT_KEYS, ['a'*40]+[str(i)*64 for i in range(1, 6)]))
        selection = dict(context, status='selected validation candidate', arm='rbf', test_predictions=0,
                         context_sha256='a'*64, preprocessing_sha256='b'*64,
                         candidate_validation_scores=[.1, .8, .8, .2], selected_index=1, selected_setting=1.)
        self.assertEqual(m.verify_selection(selection, 'rbf', [.1, .8, .8, .2], context, 'a'*64, 'b'*64), 1)
        with self.assertRaises(ValueError):
            m.verify_selection(dict(selection, selected_index=2, selected_setting=10.), 'rbf', [.1, .8, .8, .2], context, 'a'*64, 'b'*64)
        hashes = {'linear': 'c'*64, 'rbf': 'd'*64}
        gate = dict(status='sealed both case-SVC selections', test_queries=0, selections=hashes,
                    context_sha256='a'*64, preprocessing_sha256='b'*64, input_audit_sha256='e'*64)
        m.verify_gate(gate, hashes, 'a'*64, 'b'*64, 'e'*64)
        with self.assertRaises(ValueError): m.verify_gate(dict(gate, preprocessing_sha256='f'*64), hashes, 'a'*64, 'b'*64, 'e'*64)
        reload = dict(status='accepted validation model reload', decisions_identical=True, atol=1e-10,
                      binary_margin_max_abs=0., malignant_score_max_abs=0., joint_probability_max_abs=0.,
                      validation_cases=17, masks=15, test_queries=0, model_sha256='a'*64, state_sha256='b'*64)
        m.verify_reload(reload, 'a'*64, 'b'*64, 17)
        for mutation in (dict(binary_margin_max_abs=1e-9), dict(decisions_identical=False), dict(state_sha256='c'*64), dict(test_queries=1)):
            with self.assertRaises(ValueError): m.verify_reload(dict(reload, **mutation), 'a'*64, 'b'*64, 17)

    def test_collector_has_no_new_producer_math_or_fit_calls_and_exposes_cli(self):
        source = Path(m.__file__).read_text()
        self.assertNotIn('from magset import uni_case_svm', source)
        self.assertNotIn('from magset.uni_case_svm', source)
        self.assertNotIn('.fit(', source)
        args = m.parser().parse_args(['--root', 'study', '--parent', 'parent', '--config', 'cfg', '--code-sha', 'a'*40, '--output', 'checked'])
        self.assertEqual(args.parent, Path('parent'))


if __name__ == '__main__':
    unittest.main()

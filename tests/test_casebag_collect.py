"""Independent saved-output verification fixtures; no downloads or real images."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from magset import casebag_collect as checker, five_casebag as producer
    from test_five_casebag import TinyCNN, config as tiny_config
except ImportError:
    torch = None


def cases(count=6):
    labels = [0, 1, 3, 4, 5, 2]
    return [{'case_id': 'synthetic-' + str(i), 'label': labels[i],
             'indices': {view: ['synthetic/%d/%d.png' % (i, view)] for view in (40, 100, 200, 400)}}
            for i in range(count)]


def write_archive(folder, prefix, model, allowed, role):
    # Save arbitrary synthetic sufficient statistics; this intentionally makes
    # no claim that they came from real image encoding.
    features = np.random.default_rng(19).normal(size=(len(allowed), 4, model.dimension))
    records, arrays = producer.records_from_features(model, features, allowed, role)
    archive = {'case_ids': np.asarray([case['case_id'] for case in allowed]),
               'truth8': np.asarray([case['label'] for case in allowed], dtype=np.int64),
               'truth5': np.asarray([checker.MAP8_TO5[case['label']] for case in allowed], dtype=np.int64),
               'view_features': features, 'field_counts': np.ones((len(allowed), 4), dtype=np.int64),
               'descriptors': np.stack([producer.descriptor_array(features, subset) for subset in producer.INDEX_SUBSETS], axis=1),
               'descriptor_subset_masks': np.asarray([sum(1 << i for i in subset) for subset in producer.INDEX_SUBSETS]),
               'descriptor_magnifications': np.asarray(['+'.join(map(str, subset)) for subset in producer.SUBSETS]), **arrays}
    return archive, records, producer.summarize_records(records, model.arm)


@unittest.skipIf(torch is None, 'Torch required for saved-state synthetic fixtures')
class CaseBagCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.cfg = tiny_config()
        cls.cfg['optimization'].update(epochs=50, warmup_epochs=5)
        cls.cfg['selection']['epochs'] = [10, 20, 35, 50]
        cls.cfg['inference'] = {'selected_reload_tolerance': 1e-6}
        cls.fit = cases()
        cls.val = cases(5)
        cls.context = {'source': 'synthetic fixture only'}
        cls.fixtures = {}
        for arm in ('direct', 'dual'):
            folder = cls.root / arm
            folder.mkdir()
            torch.manual_seed(7)
            model = producer.CaseBagModel(TinyCNN(), 4, arm)
            initial = producer.state_hashes(model)
            producer.write_json(folder / 'started.json', cls.context)

            def stream(rows, root, allowed, cfg, epoch, workers=0, max_cases=None):
                order = producer.case_order(len(allowed), cfg['training_seed'], epoch)
                for begin in range(0, len(order), 4):
                    chosen = order[begin:begin + 4]
                    generator = torch.Generator().manual_seed(epoch * 47 + begin)
                    fields = torch.randn(len(chosen), 4, 2, 3, 4, 4, generator=generator)
                    yield (torch.tensor(chosen), torch.tensor([checker.MAP8_TO5[allowed[i]['label']] for i in chosen]),
                           fields, [producer.sampled_paths(allowed[i], cfg['training_seed'], epoch) for i in chosen])

            def score(model, rows, root, allowed, cfg, device, role):
                return write_archive(folder, '', model, allowed, role)

            with patch.object(producer, 'stream_cases', side_effect=stream), \
                    patch.object(producer, 'score_cases', side_effect=score), contextlib.redirect_stdout(io.StringIO()):
                proof = producer.train(model, [], [], cls.fit, cls.val, None, cls.cfg, cls.context,
                                       folder, torch.device('cpu'))
            # The fixture trains a tiny CPU CNN, while lifecycle checking also
            # requires the serialized one-device CUDA RNG container schema.
            # Populate a clearly synthetic byte state; no CUDA execution claim.
            checkpoint = torch.load(folder / 'last.pt', weights_only=True)
            checkpoint['rng']['cuda'] = [torch.zeros(8, dtype=torch.uint8)]
            producer.atomic_save(checkpoint, folder / 'last.pt')
            proof['last_checkpoint_sha256'] = checker.file_hash(folder / 'last.pt')
            selected = torch.load(folder / 'selected.pt', weights_only=True)
            model.load_state_dict(selected['model'])
            proof['reload'] = {'feature_max_abs': 0., 'probability_max_abs': 0.}
            report = {'status': 'complete case-bag validation run', 'context': cls.context, 'proof': proof,
                      'outer_predictions': 0, 'elapsed_seconds': 1., 'slurm_job_id': '123',
                      'peak_gpu_allocated_bytes': 100, 'peak_gpu_reserved_bytes': 200,
                      'selected_feature_replay_max_abs': 0., 'selected_probability_replay_max_abs': 0.}
            for role, allowed in (('train', cls.fit), ('validation', cls.val)):
                archive, records, summary = write_archive(folder, 'selected-' + role, model, allowed, role)
                report[role] = summary
                report[role + '_artifacts'] = producer.save_scores(folder, 'selected-' + role, archive, records)
            report['invocation_field_forwards'] = {
                'training': proof['invocation_training_field_forwards'],
                'candidate_validation': proof['invocation_candidate_validation_field_forwards'],
                'selected_fitting': len(cls.fit) * 4, 'selected_validation': len(cls.val) * 4, 'reload_probes': 8}
            # The patched producer receives empty raw-row lists; account for its
            # actual candidate calls in this synthetic saved-output fixture.
            proof['invocation_candidate_validation_field_forwards'] = len(cls.val) * 4 * 4
            report['invocation_field_forwards']['candidate_validation'] = len(cls.val) * 4 * 4
            producer.write_json(folder / 'metrics.json', report)
            cls.fixtures[arm] = (folder, model, initial)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def verify(self, arm='dual', folder=None):
        original, model, initial = self.fixtures[arm]
        with patch.object(model.backbone, 'forward', side_effect=AssertionError('No CNN forward allowed')), \
                patch.object(producer, 'load_checked_image', side_effect=AssertionError('No image decode allowed')):
            return checker.verify_run(folder or original, arm, 812, 1, self.cfg, self.context,
                                       self.fit, self.val, model, initial, producer)

    def copy_fixture(self, root, arm='dual'):
        import shutil
        return Path(shutil.copytree(self.fixtures[arm][0], Path(root) / 'run'))

    def test_complete_both_arm_states_optimizer_candidates_and_scores_replay_without_images(self):
        for arm in ('direct', 'dual'):
            result = self.verify(arm)
            self.assertEqual(result['committed_epochs'], 50)
            self.assertEqual(result['committed_optimizer_updates'], 100)
            self.assertEqual([c['epoch'] for c in result['candidates']], [10, 20, 35, 50])
            self.assertEqual(result['fitting']['pooled']['records'], 90)
            self.assertEqual(result['validation']['pooled']['records'], 75)
            checker.assert_public_safe(result)

    def test_descriptor_corruption_rejected_even_when_file_hash_rebound(self):
        folder, model, _ = self.fixtures['dual']
        report = checker.read_json(folder / 'metrics.json')
        state = torch.load(folder / 'selected.pt', weights_only=True)['model']
        with tempfile.TemporaryDirectory() as directory:
            copied = self.copy_fixture(directory)
            path = copied / 'selected-validation-features.npz'
            with np.load(path) as archive:
                data = {key: archive[key] for key in archive.files}
            data['descriptors'][0, 0, 0] += 1
            np.savez_compressed(path, **data)
            hashes = dict(report['validation_artifacts'], features_sha256=checker.file_hash(path))
            with self.assertRaisesRegex(ValueError, 'descriptors'):
                checker.verify_scores(copied, 'selected-validation', self.val, 'validation', 'dual', state, model.dimension, hashes)

    def test_explicit_hardroute_disagreement_rejected_even_when_hash_rebound(self):
        folder, model, _ = self.fixtures['dual']
        report = checker.read_json(folder / 'metrics.json')
        state = torch.load(folder / 'selected.pt', weights_only=True)['model']
        with tempfile.TemporaryDirectory() as directory:
            copied = self.copy_fixture(directory)
            path = copied / 'selected-validation-predictions.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['prediction'] = (records[0]['prediction'] + 1) % 5
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            hashes = dict(report['validation_artifacts'], predictions_sha256=checker.file_hash(path))
            with self.assertRaisesRegex(ValueError, 'replay'):
                checker.verify_scores(copied, 'selected-validation', self.val, 'validation', 'dual', state, model.dimension, hashes)

    def test_forged_optimizer_step_and_warmup_backbone_proof_rejected(self):
        for change in ('optimizer', 'warmup'):
            with tempfile.TemporaryDirectory() as directory:
                copied = self.copy_fixture(directory)
                checkpoint = torch.load(copied / 'last.pt', weights_only=True)
                if change == 'optimizer':
                    next(iter(checkpoint['optimizer']['state'].values()))['step'].add_(1)
                else:
                    checkpoint['history'][0]['fingerprints']['backbone'] = '0' * 64
                producer.atomic_save(checkpoint, copied / 'last.pt')
                report = checker.read_json(copied / 'metrics.json')
                report['proof']['last_checkpoint_sha256'] = checker.file_hash(copied / 'last.pt')
                if change == 'warmup':
                    report['proof']['history'] = checkpoint['history']
                producer.write_json(copied / 'metrics.json', report)
                with self.assertRaisesRegex(ValueError, 'update count|Warmup backbone'):
                    self.verify(folder=copied)

    def test_partial_run_and_forbidden_outer_artifact_rejected(self):
        for action in ('partial', 'outer'):
            with tempfile.TemporaryDirectory() as directory:
                copied = self.copy_fixture(directory)
                if action == 'partial':
                    (copied / 'candidate-epoch50.pt').unlink()
                else:
                    (copied / 'outer-predictions.jsonl').write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'artifact set'):
                    self.verify(folder=copied)

    def test_fractional_epoch_or_inconsistent_resumed_candidate_counts_rejected(self):
        for training, candidate in ((8, 0), (0, 20)):
            with tempfile.TemporaryDirectory() as directory:
                copied = self.copy_fixture(directory)
                report = checker.read_json(copied / 'metrics.json')
                report['proof']['invocation_training_field_forwards'] = training
                report['proof']['invocation_candidate_validation_field_forwards'] = candidate
                report['invocation_field_forwards']['training'] = training
                report['invocation_field_forwards']['candidate_validation'] = candidate
                producer.write_json(copied / 'metrics.json', report)
                with self.assertRaisesRegex(ValueError, 'whole epochs|candidate-validation field forwards'):
                    self.verify(folder=copied)

    def test_tied_scores_select_earliest_and_selected_identity_must_match(self):
        for arm in ('direct', 'dual'):
            result = self.verify(arm)
            best = max(result['candidates'], key=lambda c: c['validation']['mean_subset_macro_f1'])
            self.assertEqual(result['selected_epoch'], best['epoch'])
        with tempfile.TemporaryDirectory() as directory:
            copied = self.copy_fixture(directory)
            saved = torch.load(copied / 'selected.pt', weights_only=True)
            saved['epoch'] = 999
            producer.atomic_save(saved, copied / 'selected.pt')
            with self.assertRaisesRegex(ValueError, 'Selected checkpoint'):
                self.verify(folder=copied)

    def test_stable_extreme_hardroute_and_router_tie_use_explicit_decision(self):
        descriptors = np.zeros((1, 8), dtype=np.float32)
        arrays = {'binary_coef': np.zeros((1, 8), dtype=np.float32),
                  'binary_intercept': np.asarray([np.log(.6 / .4)], dtype=np.float32),
                  'malignant_coef': np.zeros((4, 8), dtype=np.float32),
                  'malignant_intercept': np.log([.3, .25, .25, .2]).astype(np.float32)}
        result = checker.replay_scores(arrays, descriptors, 'dual')
        self.assertEqual(result['prediction'][0], 1)
        self.assertEqual(result['probabilities5'][0].argmax(), 0)
        arrays['binary_intercept'][0] = 0
        self.assertEqual(checker.replay_scores(arrays, descriptors, 'dual')['prediction'][0], 1)
        arrays['binary_intercept'][0] = -2000
        result = checker.replay_scores(arrays, descriptors, 'dual')
        self.assertTrue(np.isfinite(result['log_probabilities5']).all())
        self.assertEqual(result['prediction'][0], 0)

    def test_matrix_requires_exact_twenty_fits_and_no_extra_score_files(self):
        cfg = {'seeds': [812, 813], 'trials': [1, 2, 3, 4, 5]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ('direct', 'dual'):
                for seed in cfg['seeds']:
                    for trial in cfg['trials']:
                        folder = root / arm / ('split%d' % seed) / ('trial%d' % trial)
                        folder.mkdir(parents=True)
                        for name in checker.expected_files():
                            (folder / name).touch()
            self.assertEqual(len(checker.verify_matrix_paths(root, cfg)), 20)
            (root / 'outer-predictions.jsonl').touch()
            with self.assertRaisesRegex(ValueError, '20-fit matrix'):
                checker.verify_matrix_paths(root, cfg)
            (root / 'outer-predictions.jsonl').unlink()
            next(root.rglob('selected.pt')).unlink()
            with self.assertRaisesRegex(ValueError, '20-fit matrix'):
                checker.verify_matrix_paths(root, cfg)

    def test_source_archive_commit_bytes_and_symlink_are_checked_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'module.py').write_text('original\n')
            archive = root / 'source.tar'
            with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT, pax_headers={'comment': 'a' * 40}) as stream:
                stream.add(source / 'module.py', arcname='module.py')
            digest = checker.file_hash(archive)
            checker.verify_archive(archive, source, 'a' * 40, digest)
            with self.assertRaisesRegex(ValueError, 'commit identity'):
                checker.verify_archive(archive, source, 'b' * 40, digest)
            (source / 'module.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'differs from archive'):
                checker.verify_archive(archive, source, 'a' * 40, digest)
            (source / 'module.py').unlink()
            (source / 'module.py').symlink_to(archive)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                checker.verify_archive(archive, source, 'a' * 40, digest)

    def test_frozen_reference_gate_and_matched_hierarchy_delta_are_separate(self):
        run = self.verify('dual')
        reference = {}
        runs = []
        for seed in (812, 813):
            base = checker.aggregate_summaries([run['validation']] * 5)
            reference[str(seed)] = dict(base, mean_subset_macro_f1=base['mean_subset_macro_f1'] - .03,
                                       mean_subset_accuracy=base['mean_subset_accuracy'])
            for arm in ('direct', 'dual'):
                for trial in range(1, 6):
                    runs.append(dict(run, arm=arm, seed=seed, trial=trial))
        cfg = {'seeds': [812, 813], 'trials': list(range(1, 6)),
               'comparison_gate': {'margin_mean15_macroF1_each_arrangement': .02}}
        summary, matched, gate = checker.aggregate(runs, reference, cfg)
        self.assertEqual(matched['812']['delta_mean_subset_macro_f1'], 0)
        self.assertAlmostEqual(gate['812']['delta_mean_subset_macro_f1'], .03)
        self.assertEqual(gate['812']['gate_passed'], all(v > 0 for v in summary['812']['dual']['validation']['pooled_class_recall']))
        with self.assertRaisesRegex(ValueError, 'five-trial'):
            checker.aggregate(runs[:-1], reference, cfg)

    def test_private_identifiers_and_paths_rejected_from_public_values(self):
        for value in ({'case_id': 'synthetic'}, {'path': '/private/root'}, {'name': 'SOB_M_DC-123'}):
            with self.assertRaises(ValueError):
                checker.assert_public_safe(value)


if __name__ == '__main__':
    unittest.main()

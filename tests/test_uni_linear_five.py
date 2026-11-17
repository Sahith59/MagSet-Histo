"""UNI field baseline adapter and full fitting-only smoke/replay regression."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import uni_linear_five as runner, uni_linear_collect as collector
from magset.data import file_hash
from magset.five_metrics import MAP8_TO5


def fixture(root):
    config = json.loads(Path('configs/breakhis-uni-linear-v1.json').read_text())
    counts = ([11, 25, 2, 5, 3], [4, 1, 1, 1, 1], [9, 12, 2, 3, 2])
    labels = np.concatenate([np.repeat(np.arange(5), values) for values in counts])
    rng = np.random.default_rng(139)
    bags = [[(rng.normal(size=(1+(i+v)%3, 3))+labels[i]/3).astype(np.float32)
             for v in range(4)] for i in range(82)]
    cases = [{'case_id': 'synthetic%d'%i, 'label': [0, 1, 3, 4, 5][label]} for i, label in enumerate(labels)]
    assert [MAP8_TO5[c['label']] for c in cases] == labels.tolist()
    train, val, excluded = np.arange(46), np.arange(46, 54), np.arange(54, 82)
    roles = {(seed, trial): (train, val, excluded) for seed in (812, 813) for trial in range(1, 6)}
    config.update(encoder_dimension=3, encoder={'dimension': 3}, expected_role_counts=[46, 8, 28],
        fitting_five_support=counts[0], validation_five_support=counts[1], excluded_five_support=counts[2])
    context = {'code_sha': 'a'*40, 'source_archive_sha256': 'b'*64}
    z = np.asarray([[view.mean(0) for view in case] for case in bags], dtype=np.float32)
    inputs = (config, context, {}, cases, bags, z, labels, roles)
    return inputs


class UniLinearTests(unittest.TestCase):
    def test_registration_locks_known_field_solver_and_budget(self):
        config = json.loads(Path('configs/breakhis-uni-linear-v1.json').read_text())
        self.assertIs(runner.validate_config(config), config)
        for key, value in [('C_grid', [.001, .01, .1, 1]), ('encoder_dimension', 3072), ('production_head_fits', 121)]:
            changed = copy.deepcopy(config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): runner.validate_config(changed)

    def test_input_adapter_keeps_accepted_UNI_bags_and_roles_without_old_reader_fallback(self):
        path = Path('configs/breakhis-uni-linear-v1.json')
        own = json.loads(path.read_text())
        original = {'encoder': {'dimension': 1536}, 'C_grid': [99], 'encoder_dimension': 1536}
        context = {'config_sha256': own['base_input_config_sha256'], 'implementation_hashes': {'field_five.py': 'c'*64}}
        bags, cases, roles = object(), object(), object()
        inputs = (original, context, {}, cases, bags, object(), object(), roles)
        args = SimpleNamespace(config=path, cache=Path('synthetic-UNI-cache'), center_cache=Path('synthetic-center-cache'))
        with patch.object(runner.approved, 'load_inputs', return_value=inputs) as loader:
            returned = runner.load_inputs(args)
        self.assertIs(returned[4], bags); self.assertIs(returned[3], cases); self.assertIs(returned[7], roles)
        self.assertEqual(loader.call_args.args[0].config.name, 'breakhis-uni2-h-v1-kernel.json')
        self.assertEqual(loader.call_args.args[0].cache, args.cache)
        self.assertEqual(returned[0]['C_grid'], [.01, .1, 1., 10.])
        self.assertEqual(returned[1]['parent_approved_config_sha256'], own['base_input_config_sha256'])

    def test_complete_smoke_and_real_synthetic_candidate_loop_replay_without_collector_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = fixture(root); config, context, _, cases, bags, z, labels, roles = inputs
            args = SimpleNamespace(output=root/'audit', audit=None, seed=812, trial=1)
            with patch.object(runner, 'load_inputs', return_value=inputs): runner.audit_input(args)
            args.audit = args.output; args.output = root/'smoke'
            with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(runner, 'verify_smoke_acceptance', side_effect=AssertionError('Smoke cannot require itself')):
                smoke = runner.run(args, smoke=True)
            self.assertEqual(smoke['head_fits'], 12)
            self.assertEqual(runner.smoke_query_indices(labels, roles[812, 1][0]).tolist(), [0, 11, 36, 38, 43])
            self.assertFalse(list(args.output.rglob('validation-predictions.jsonl')))
            self.assertTrue(all('selected_index' not in arm for arm in smoke['arms'].values()))
            review = SimpleNamespace(output=root/'acceptance', audit=args.audit, smokes=args.output)
            with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(runner.field, 'fit_arm', side_effect=AssertionError('Reviewer may not fit')):
                receipt = collector.verify_smoke(review)
            self.assertEqual(receipt['verified_case_subset_records'], 600)
            self.assertEqual(receipt['head_fits'], 12)
            args.smokes = args.output; args.smoke_acceptance = review.output/'acceptance.json'; args.output = root/'production'
            with patch.object(runner, 'load_inputs', return_value=inputs):
                report = runner.run(args)
            with patch.object(runner.field, 'fit_arm', side_effect=AssertionError('Collector may not fit')):
                verified = collector.verify_run(args.output, 812, 1, config, context, cases, bags, z, labels, roles,
                    file_hash(args.audit/'audit.json'), file_hash(args.smoke_acceptance))
            self.assertEqual(verified['head_fits'], 12)
            self.assertEqual(len(verified['arms']['dual']['candidates']), 4)
            selected = max(range(4), key=lambda i: report['arms']['dual']['candidates'][i]['validation']['mean_subset_macro_f1'])
            self.assertEqual(verified['arms']['dual']['selected_index'], selected)
            report['arms']['dual']['selected_index'] = (selected+1)%4
            runner.write_json(args.output/'run.json', report)
            with self.assertRaisesRegex(ValueError, 'Selected candidate'):
                collector.verify_run(args.output, 812, 1, config, context, cases, bags, z, labels, roles,
                    file_hash(args.audit/'audit.json'), file_hash(args.smoke_acceptance))

    def test_production_rejects_missing_smoke_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'independently verified'):
            runner.verify_smoke_acceptance(SimpleNamespace(smoke_acceptance=None), {}, {}, 'a'*64)


if __name__ == '__main__': unittest.main()

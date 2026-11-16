"""Ensure coverage uses supplied audited fields while preserving the KDE contract."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magset import coverage_five as m, coverage_collect as collector
from magset import kernel_five as kernel
from magset.five_class import write_json
from magset.five_metrics import FINAL5
from magset.data import CLASS_NAMES
from test_kernel_five import fixture


def bound_config():
    c = json.loads(Path('configs/breakhis-coverage-kernel-v1.json').read_text())
    c.update(feature_producer_code_sha='a'*40, feature_audit_sha256='b'*64,
             cache_provenance={str(v): {'npz_sha256': 'c'*64, 'json_sha256': 'd'*64,
                 'components_sha256': 'e'*64, 'images': n} for v, n in zip((40, 100, 200, 400), (1977, 1977, 1977, 1978))})
    return c


class CoverageKDETests(unittest.TestCase):
    def test_unbound_features_cannot_reach_any_fitting_or_scoring(self):
        c = bound_config(); self.assertIs(m.validate_bindings(c), c)
        for key, value in [('feature_producer_code_sha', None), ('feature_audit_sha256', None), ('cache_provenance', {})]:
            bad = copy.deepcopy(c); bad[key] = value
            with self.assertRaises(ValueError): m.validate_bindings(bad)
        bad = copy.deepcopy(c); bad['cache_provenance']['40']['images'] += 1
        with self.assertRaisesRegex(ValueError, '7909'): m.validate_bindings(bad)
        bad = copy.deepcopy(c); bad['temperatures'] = [.001, .01, .1, 1.]
        with self.assertRaisesRegex(ValueError, 'recipe'): m.validate_bindings(bad)

    def test_delegation_uses_explicit_new_bags_without_loading_single_center_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = fixture(root)
            args = SimpleNamespace(output=root/'coverage', audit=f.audit_dir, seed=812, trial=1)
            with patch.object(m, 'load_inputs', return_value=f.inputs), patch.object(kernel, 'load_inputs', side_effect=AssertionError('Must not revert to original fields')):
                m.run(args)
            report = json.loads((args.output/'run.json').read_text())
            self.assertEqual(report['optimized_head_fits'], 0)
            self.assertEqual(report['code_sha'], f.context['code_sha'])
            for arm in kernel.ARMS:
                self.assertEqual(len(report['arms'][arm]['candidates']), 4)
                self.assertIn('fitting_leave_case_out', report['arms'][arm])

    def test_reference_allows_coverage_change_but_rejects_split_or_core_changes(self):
        config = {'reference_comparison_sha256': 'a'*64, 'fitting_five_support': [11, 25, 2, 5, 3], 'validation_five_support': [4, 1, 1, 1, 1]}
        context = {'manifest_sha256': 'b'*64, 'protocol_sha256': 'c'*64,
                   'implementation_hashes': {k: 'd'*64 for k in ('data.py', 'baseline.py', 'five_metrics.py', 'bags.py', 'kernel_model.py', 'field_five.py')}}
        roles = {(s, t): None for s in (812, 813) for t in range(1, 6)}
        ref = dict(context, class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10,
                   candidate_pipelines=80, outer_predictions=0, encoder={'features': 'original center'})
        ref['runs'] = [{'seed': s, 'trial': t, 'arms': {a: {
            'fitting_leave_case_out': {'distinct_class_support': config['fitting_five_support']},
            'validation': {'distinct_class_support': config['validation_five_support']}}
            for a in kernel.ARMS}} for s, t in roles]
        with patch.object(collector, 'file_hash', return_value='a'*64), patch.object(collector, 'read_json', return_value=ref):
            self.assertIs(collector.reference_comparison(config, context, roles), ref)
            bad = copy.deepcopy(context); bad['implementation_hashes']['kernel_model.py'] = 'e'*64
            with self.assertRaisesRegex(ValueError, 'method/split'): collector.reference_comparison(config, bad, roles)
            bad = dict(context, protocol_sha256='f'*64)
            with self.assertRaisesRegex(ValueError, 'protocol'): collector.reference_comparison(config, bad, roles)


if __name__ == '__main__': unittest.main()

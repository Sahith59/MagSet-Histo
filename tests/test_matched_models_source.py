"""No-TensorFlow checks of matched model intake, scaling and architecture contract."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import matched_models as m


class MatchedModelSourceTests(unittest.TestCase):
    def test_import_does_not_import_tensorflow_or_keras(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; from magset import matched_models; '
            'assert not any(k == "tensorflow" or k.startswith("tensorflow.") or '
            'k == "tf_keras" or k == "keras" for k in sys.modules)'],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_kind_and_missing_weights_fail_before_runtime(self):
        with patch.object(m, '_runtime', side_effect=AssertionError('Runtime must remain lazy')):
            for kind, kwargs in [('other', {}), ('cnn', {}), ('swin', {}),
                                 ('cnn', {'cnn_weights': '/nonexistent/pretrained.h5'})]:
                with self.assertRaises(ValueError):
                    m.build_model(kind, **kwargs)

    def test_pretrained_file_guard_binds_bytes_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'synthetic.h5'
            p.write_bytes(b'original synthetic weight bytes')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(m._checked_file(p, digest), str(p.resolve()))
            p.write_bytes(b'replaced fitted weights')
            with self.assertRaisesRegex(ValueError, 'pinned original'):
                m._checked_file(p, digest)

    def test_numeric_input_scaling_preserves_raw_cnn_and_normalizes_swin_once(self):
        tf = SimpleNamespace(float32=np.float32,
                             cast=lambda value, dtype: np.asarray(value, dtype=dtype),
                             constant=lambda value, dtype: np.array(value, dtype=dtype))
        pixels = np.array([0, 127, 255], dtype=np.uint8)
        with patch.object(m, '_runtime', return_value=(tf, None)):
            np.testing.assert_array_equal(m.normalize_cnn_inputs(pixels), pixels.astype(np.float32))
            expected = (pixels.astype(np.float32) / 255 - np.array(m.IMAGENET_MEAN, np.float32)) / np.array(m.IMAGENET_STD, np.float32)
            np.testing.assert_array_equal(m.normalize_swin_inputs(pixels), expected)

    def test_descriptor_mean_preserves_field_boundary_and_crop_permutation(self):
        tf = SimpleNamespace(reshape=np.reshape, reduce_mean=lambda x, axis: np.mean(x, axis=axis))
        fields = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        with patch.object(m, '_runtime', return_value=(tf, None)):
            expected = np.array([[2, 3], [8, 9]], dtype=np.float32)
            np.testing.assert_array_equal(m.average_crop_descriptors(fields.reshape(6, 2), 2, 2), expected)
            permuted = fields[:, [2, 0, 1], :].reshape(6, 2)
            np.testing.assert_array_equal(m.average_crop_descriptors(permuted, 2, 2), expected)

    def test_swin_loading_accepts_only_exact_deterministic_buffers(self):
        expected = np.array([[(r1-r2+6)*13+(c1-c2+6)
                              for r2 in range(7) for c2 in range(7)]
                             for r1 in range(7) for c1 in range(7)], dtype=np.int32)
        names = ['swin/encoder/layers.%d/blocks.%d/attention/self/relative_position_index:0' % (s, b)
                 for s, depth in enumerate((2, 2, 6, 2)) for b in range(depth)]
        weights = [SimpleNamespace(name='tf_swin_model/' + name, trainable=False,
                                   dtype=np.int32, numpy=lambda: expected.copy()) for name in names]
        backbone = SimpleNamespace(weights=weights)
        info = {'missing_keys': names, 'unexpected_keys': ['classifier'], 'mismatched_keys': []}
        receipt = m._verify_swin_loading(backbone, info, SimpleNamespace(int32=np.int32))
        self.assertEqual(receipt['verified_position_buffers'], 12)
        for changes in ({'missing_keys': names + ['learned_bias']},
                        {'mismatched_keys': ['learned_bias']}, {'unexpected_keys': ['other_layer']},
                        {'error_msgs': ['load failed']}):
            with self.assertRaises(ValueError):
                m._verify_swin_loading(backbone, dict(info, **changes), SimpleNamespace(int32=np.int32))
        weights[0].trainable = True
        with self.assertRaisesRegex(ValueError, 'position-index'):
            m._verify_swin_loading(backbone, info, SimpleNamespace(int32=np.int32))

    def test_source_has_no_mixed_keras_imports_or_remote_pretrained_fallback(self):
        source = Path(m.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('tensorflow.keras', 'keras')))
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name == 'keras' for alias in node.names))
        self.assertIn('local_files_only=True, use_safetensors=False', source)
        self.assertIn('weights=self.cnn_weights', source)
        self.assertNotIn("weights='imagenet'", source)
        self.assertEqual(m.CROPS, 3)
        self.assertEqual(m.IMAGE_SIZE, 224)

    def test_receipt_reports_actual_flags_counts_and_no_weight_paths(self):
        layers = [SimpleNamespace(name='layer%d' % i, trainable=i >= 4) for i in range(64)]
        model = SimpleNamespace(kind='cnn', built=True,
            heads=SimpleNamespace(stop_binary_gradient=False),
            weights=[SimpleNamespace(shape=(2, 3)), SimpleNamespace(shape=(3,))],
            trainable_weights=[SimpleNamespace(shape=(2, 3))],
            non_trainable_weights=[SimpleNamespace(shape=(3,))],
            backbone=SimpleNamespace(layers=layers))
        receipt = m.architecture_receipt(model)
        self.assertEqual(receipt['parameters'], {'all': 9, 'trainable': 6, 'nontrainable': 3})
        self.assertEqual(receipt['trainable_last_backbone_layers'], 60)
        self.assertEqual([x['trainable'] for x in receipt['backbone_layer_flags']], [False] * 4 + [True] * 60)
        self.assertEqual(receipt['outputs'], {'binary_output': 2, 'subtype_output': 4})
        self.assertFalse(receipt['subtype_binary_stop_gradient'])
        with self.assertRaises(ValueError):
            m.architecture_receipt(SimpleNamespace(kind='cnn', built=False))


if __name__ == '__main__':
    unittest.main()

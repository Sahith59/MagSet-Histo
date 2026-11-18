import unittest
from importlib.util import find_spec

import numpy as np
if find_spec("torch") is None:
    raise unittest.SkipTest("These tests require PyTorch")

import torch

from magset.shared_hierarchy import zoom_masks
from scripts.train_single_output_coverage import SingleOutputClassifier, final_labels


class SingleOutputTests(unittest.TestCase):
    def test_replay_casts_descriptor_like_the_stored_model(self):
        from scripts.verify_single_output_coverage import reconstruct

        views = np.full((1, 4, 2), 1.e7, dtype=np.float64)
        views[..., 0] += .125  # Below the float32 spacing around this value.
        views[..., 1] += 2.
        mean = np.full((4, 2), 1.e7, dtype=np.float32)
        scale = np.ones_like(mean)
        model = SingleOutputClassifier(torch.from_numpy(mean), torch.from_numpy(scale))
        masks = zoom_masks()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.weight[0, 0] = 1.
            model.classifier.bias.zero_()
            exact = model(torch.tensor(views, dtype=torch.float32), masks).numpy()
        rebuilt = reconstruct(views, mean, scale, masks.numpy(),
                              model.classifier.weight.detach().numpy(),
                              model.classifier.bias.detach().numpy())
        np.testing.assert_array_equal(rebuilt, exact)

    def test_pooled_malignant_mass_controls_route(self):
        probabilities = torch.tensor([[.40, .10, .25, .10, .15],
                                      [.51, .10, .19, .10, .10]])
        self.assertEqual(final_labels(probabilities.log()).tolist(), [2, 0])

    def test_only_one_trainable_layer_and_masked_views(self):
        model = SingleOutputClassifier(torch.zeros(4, 8), torch.ones(4, 8))
        self.assertEqual(set(dict(model.named_parameters())), {"classifier.weight", "classifier.bias"})
        views = torch.randn(2, 4, 8)
        masks = zoom_masks()
        scores = model(views, masks)
        self.assertEqual(scores.shape, (2, 15, 5))
        torch.testing.assert_close(model(views, masks.flip(0)), scores.flip(1))
        views[:, 1:] = float("nan")
        torch.testing.assert_close(model(views, masks[:1]), scores[:, :1])
        with self.assertRaises(ValueError):
            model(views, masks)

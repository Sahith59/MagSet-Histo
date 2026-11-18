import unittest

import torch

from magset.shared_hierarchy import zoom_masks
from scripts.train_single_output_coverage import SingleOutputClassifier, final_labels


class SingleOutputTests(unittest.TestCase):
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

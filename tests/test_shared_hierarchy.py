import unittest

import torch

from magset.shared_hierarchy import SharedHierarchy, final_labels, training_loss, zoom_masks


class SharedHierarchyTests(unittest.TestCase):
    def test_route_and_conditional_gradient(self):
        logits = torch.tensor([[-3., 10., 0., 0., 0.], [0., 0., 0., 2., 0.]])
        self.assertEqual(final_labels(logits).tolist(), [0, 3])
        with self.assertRaises(ValueError):
            final_labels(torch.full_like(logits, float("nan")))
        labels = torch.arange(5)
        scores = torch.zeros((5, 15, 5), requires_grad=True)
        training_loss(scores, labels).backward()
        self.assertTrue(torch.all(scores.grad[0, :, 1:] == 0))
        self.assertTrue(torch.all(scores.grad[:, :, 0] != 0))
        self.assertTrue(torch.all(scores.grad[1:, :, 1:] != 0))

    def test_only_requested_zooms_affect_a_prediction(self):
        model = SharedHierarchy(torch.zeros(4, 9), torch.ones(4, 9), 8, .2).eval()
        x = torch.randn(2, 4, 9)
        masks = zoom_masks()
        y = model(x, masks)
        self.assertEqual(y.shape, (2, 15, 5))
        torch.testing.assert_close(model(x, masks.flip(0)), y.flip(1))
        x[:, 1:] = float("nan")
        torch.testing.assert_close(model(x, masks[:1]), y[:, :1])
        with self.assertRaises(ValueError):
            model(x, masks[-1:])
        with self.assertRaises(ValueError):
            model(x, torch.zeros(1, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

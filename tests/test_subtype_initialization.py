import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional")
class InitializationTests(unittest.TestCase):
    def test_initial_evidence_matches_equal_case_cosine(self):
        from magset.shared_hierarchy import SharedHierarchy, zoom_masks
        from scripts.run_subtype_initialization import initialize_subtypes

        torch.manual_seed(44)
        data = torch.randn(7, 4, 9)
        labels = torch.tensor([0, 1, 1, 1, 2, 3, 4])
        net = SharedHierarchy(torch.zeros(4, 9), torch.ones(4, 9), 32, .2)
        masks = zoom_masks()
        query = torch.randn(2, 4, 9)
        initialize_subtypes(net, data, labels, masks)
        output = net(query, masks)
        for i, mask in enumerate(masks):
            normalized_query = torch.nn.functional.normalize(query[:, mask].mean(1))
            for label in range(1, 5):
                rows = data[labels == label]
                unit = torch.stack([torch.nn.functional.normalize(rows[:, m].mean(1)) for m in masks])
                direction = torch.nn.functional.normalize(unit.mean((0, 1)), dim=0)
                torch.testing.assert_close(output[:, i, label], normalized_query @ direction,
                                           atol=2e-6, rtol=2e-6)
        with self.assertRaises(ValueError):
            initialize_subtypes(net, data[:-1], labels[:-1], masks)

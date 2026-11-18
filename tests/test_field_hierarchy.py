import io
import unittest
from importlib.util import find_spec

if find_spec("torch") is None:
    raise unittest.SkipTest("These tests require PyTorch")

import torch

from magset.field_hierarchy import FieldHierarchy, PackedFields
from magset.shared_hierarchy import SharedHierarchy, training_loss, zoom_masks


def packed(views, repetitions=None):
    fields, cases, zooms = [], [], []
    for case in range(len(views)):
        for zoom in range(4):
            count = 1 if repetitions is None else repetitions[case][zoom]
            fields.extend([views[case, zoom]] * count)
            cases.extend([case] * count)
            zooms.extend([zoom] * count)
    return PackedFields(torch.stack(fields), torch.tensor(cases), torch.tensor(zooms), views)


class FieldHierarchyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        self.mean = torch.randn(4, 7)
        self.scale = torch.rand(4, 7) + .5
        self.model = FieldHierarchy(self.mean, self.scale, 8, .2).eval()
        self.masks = zoom_masks()

    def test_raw_mean_control_equivalence_and_state_keys(self):
        control = SharedHierarchy(self.mean, self.scale, 8, .2).eval()
        control.load_state_dict(self.model.state_dict(), strict=True)
        normalized = torch.randn(3, 1, 7).expand(-1, 4, -1)
        views = normalized * self.scale + self.mean
        batch = packed(views, [[1, 2, 3, 4], [5, 1, 2, 1], [2, 2, 1, 3]])
        torch.testing.assert_close(self.model(batch, self.masks), control(views, self.masks), atol=1.e-6, rtol=1.e-5)
        self.assertEqual(list(self.model.state_dict()), list(control.state_dict()))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), sum(p.numel() for p in control.parameters()))

    def test_field_and_mask_order_invariance(self):
        batch = packed(torch.randn(3, 4, 7), [[1, 2, 3, 4], [5, 1, 2, 1], [2, 2, 1, 3]])
        order = torch.randperm(len(batch.fields))
        shuffled = PackedFields(batch.fields[order], batch.case[order], batch.zoom[order], batch.views)
        output = self.model(batch, self.masks)
        torch.testing.assert_close(self.model(shuffled, self.masks), output)
        torch.testing.assert_close(self.model(batch, self.masks.flip(0)), output.flip(1))

    def test_variable_field_counts_do_not_change_view_mass(self):
        views = torch.randn(2, 4, 7)
        original = packed(views)
        repeated = packed(views, [[9, 1, 4, 2], [2, 7, 1, 5]])
        torch.testing.assert_close(self.model(original, self.masks), self.model(repeated, self.masks))
        # Replicate every field of a genuinely heterogeneous view equally.
        fields = original.fields.clone()
        perturbation = torch.randn_like(fields) * .2
        heterogeneous = PackedFields(torch.cat((fields + perturbation, fields - perturbation)),
                                     original.case.repeat(2), original.zoom.repeat(2), views)
        replicated = PackedFields(heterogeneous.fields.repeat_interleave(3, dim=0),
                                  heterogeneous.case.repeat_interleave(3), heterogeneous.zoom.repeat_interleave(3), views)
        torch.testing.assert_close(self.model(heterogeneous, self.masks), self.model(replicated, self.masks))

    def test_nonlinearity_precedes_equal_field_and_view_averages(self):
        model = FieldHierarchy(torch.zeros(4, 2), torch.ones(4, 2), 1, 0.).eval()
        with torch.no_grad():
            model.classifier[0].weight.copy_(torch.tensor([[1., 0., .1, .2, .3, .4]]))
            model.classifier[0].bias.fill_(.2)
            model.classifier[3].weight.copy_(torch.arange(1., 6.)[:, None])
            model.classifier[3].bias.copy_(torch.arange(5.) * .25)
        # Both selected raw view means equal [1, 1], so the bag scale is one.
        # Their field counts differ, and the second view has heterogeneous fields.
        batch = PackedFields(torch.tensor([[1., 1.], [-1., 1.], [-1., 1.], [5., 1.]]),
                             torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 1, 1]),
                             torch.ones(1, 4, 2))
        mask = torch.tensor([[True, True, False, False]])
        gelu = torch.nn.functional.gelu
        expected_hidden = (gelu(torch.tensor(1.5)) +
                           (2 * gelu(torch.tensor(-.5)) + gelu(torch.tensor(5.5))) / 3) / 2
        expected = expected_hidden * torch.arange(1., 6.) + torch.arange(5.) * .25
        torch.testing.assert_close(model(batch, mask)[0, 0], expected)
        control = SharedHierarchy(torch.zeros(4, 2), torch.ones(4, 2), 1, 0.).eval()
        control.load_state_dict(model.state_dict())
        self.assertFalse(torch.allclose(model(batch, mask), control(batch.views, mask)))

    def test_inactive_fields_and_views_are_excluded(self):
        batch = packed(torch.randn(2, 4, 7))
        mask = self.masks[:1]
        expected = self.model(batch, mask)
        fields, views = batch.fields.clone(), batch.views.clone()
        fields[batch.zoom != 0] = float("nan")
        views[:, 1:] = float("nan")
        hidden = PackedFields(fields, batch.case, batch.zoom, views)
        torch.testing.assert_close(self.model(hidden, mask), expected)
        keep = batch.zoom == 0
        missing = PackedFields(fields[keep], batch.case[keep], batch.zoom[keep], views)
        torch.testing.assert_close(self.model(missing, mask), expected)
        with self.assertRaises(ValueError):
            self.model(hidden, self.masks[-1:])
        # A selected view may contribute to one mask while being excluded from another.
        altered_views = batch.views.clone()
        altered_fields = batch.fields.clone()
        altered_views[:, 1] += 50
        altered_fields[batch.zoom == 1] += 50
        both = torch.tensor([[True, False, False, False], [False, True, False, False]])
        altered = PackedFields(altered_fields, batch.case, batch.zoom, altered_views)
        torch.testing.assert_close(self.model(altered, both)[:, :1], self.model(batch, both)[:, :1])

    def test_guards_and_zero_bag_stability(self):
        batch = packed(torch.randn(2, 4, 7))
        for masks in (torch.zeros(1, 4, dtype=torch.bool), torch.ones(1, 4), torch.ones(4, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                self.model(batch, masks)
        for field in (float("nan"), float("inf")):
            invalid = batch.fields.clone()
            invalid[0, 0] = field
            with self.assertRaises(ValueError):
                self.model(PackedFields(invalid, batch.case, batch.zoom, batch.views), self.masks)
        invalid_views = batch.views.clone()
        invalid_views[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields, batch.case, batch.zoom, invalid_views), self.masks)
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields.double(), batch.case, batch.zoom, batch.views), self.masks)
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields, batch.case.float(), batch.zoom, batch.views), self.masks)
        invalid_cases = batch.case.clone()
        invalid_cases[0] = len(batch.views)
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields, invalid_cases, batch.zoom, batch.views), self.masks)
        invalid_zoom = batch.zoom.clone()
        invalid_zoom[0] = -1
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields, batch.case, invalid_zoom, batch.views), self.masks)
        keep = batch.zoom != 0
        with self.assertRaises(ValueError):
            self.model(PackedFields(batch.fields[keep], batch.case[keep], batch.zoom[keep], batch.views), self.masks)
        zero = packed(self.mean[None].expand(2, -1, -1).clone())
        self.assertTrue(torch.isfinite(self.model(zero, self.masks)).all())

    def test_shared_gradient_and_reload(self):
        self.model.train()
        views = torch.randn(5, 4, 7)
        batch = packed(views)
        loss = training_loss(self.model(batch, self.masks), torch.arange(5))
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.model.eval()
        restored = FieldHierarchy(self.mean, self.scale, 8, .2).eval()
        saved = io.BytesIO()
        torch.save(self.model.state_dict(), saved)
        saved.seek(0)
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        torch.testing.assert_close(restored(batch, self.masks), self.model(batch, self.masks), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

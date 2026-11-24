import unittest
from unittest.mock import patch

import torch

from peptide_geometry_refiner import PeptideGeometryProjection, PeptideConstrainedLoopPairRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class PeptideGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.bb = torch.randn(5, 4, 3, dtype=torch.double)
        self.anchors = torch.randn(2, 4, 3, dtype=torch.double)
        self.mask = torch.ones(2, 4, dtype=torch.bool)
        self.layer = PeptideGeometryProjection()

    def test_energy_rigid_geometry_and_anchors(self):
        anchors = self.anchors.clone()
        with torch.no_grad():
            result = self.layer(self.bb, self.anchors, self.mask)
        self.assertLess(float(self.layer.energy(result, anchors, self.mask)),
                        float(self.layer.energy(self.bb, anchors, self.mask)))
        torch.testing.assert_close(torch.cdist(result, result), torch.cdist(self.bb, self.bb))
        self.assertTrue(torch.equal(anchors, self.anchors))
        self.assertFalse(result.requires_grad)

    def test_equivariance(self):
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        t = torch.tensor([3., -2., 1.], dtype=torch.double)
        with torch.no_grad():
            a = self.layer(self.bb, self.anchors, self.mask)
            b = self.layer(self.bb @ q+t, self.anchors @ q+t, self.mask)
        torch.testing.assert_close(b, a @ q+t, atol=1e-9, rtol=1e-9)

    def test_backward_and_no_grad_agree(self):
        bb = self.bb.clone().requires_grad_(True)
        result = self.layer(bb, self.anchors, self.mask)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(bb.grad).all())
        self.assertGreater(float(bb.grad.abs().sum()), 0)
        with torch.no_grad():
            expected = self.layer(self.bb, self.anchors, self.mask)
        torch.testing.assert_close(result.detach(), expected)

    def test_masked_nan_single_residue(self):
        anchors = torch.full_like(self.anchors, float('nan'))
        mask = torch.zeros_like(self.mask)
        with torch.no_grad():
            result = self.layer(self.bb[:1], anchors, mask)
        torch.testing.assert_close(result, self.bb[:1])
        bb = self.bb[:1].clone().requires_grad_(True)
        self.layer(bb, anchors, mask).square().sum().backward()
        self.assertTrue(torch.isfinite(bb.grad).all())

    def test_numerical_gradient(self):
        layer = PeptideGeometryProjection(steps=1)
        bb = self.bb[:2].clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda v: layer(v, self.anchors, self.mask), (bb,), fast_mode=True))

    def test_zero_steps(self):
        result = PeptideGeometryProjection(steps=0)(self.bb, self.anchors, self.mask)
        self.assertTrue(torch.equal(result, self.bb))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            self.layer(self.bb*float('nan'), self.anchors, self.mask)
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            self.layer(self.bb, self.anchors, self.mask)

    def test_wrapper_and_checkpoint_compatibility(self):
        baseline = PersistentLoopPairRefiner(width=8, layers=1)
        candidate = PeptideConstrainedLoopPairRefiner(width=8, layers=1)
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        logits = torch.randn(5, 20)
        x = dict(anchors=self.anchors, anchor_mask=self.mask)
        with patch.object(PersistentLoopPairRefiner, 'forward', return_value=dict(backbone=self.bb, sequence_logits=logits)):
            with torch.no_grad():
                result = candidate(x)
        self.assertIs(result['sequence_logits'], logits)
        self.assertIs(result['backbone_before_peptide_projection'], self.bb)
        self.assertFalse(torch.equal(result['backbone'], self.bb))

    def test_full_synthetic_forward_backward(self):
        # Entire candidate, float32, synthetic context only (no dataset access).
        x = dict(base_bb=self.bb.float(), base_logits=torch.randn(5, 20),
                 anchors=self.anchors.float(), anchor_mask=self.mask,
                 context_bb=torch.randn(10, 4, 3), context_mask=torch.ones(10, 4, dtype=torch.bool),
                 context_features=torch.randn(10, 27), antigen_mask=torch.tensor([False]*5+[True]*5))
        for kind, ns, nv in [('surface', 11, 4), ('atom', 10, 2)]:
            x[f'binding_{kind}_xyz'] = torch.randn(12, 3)
            x[f'binding_{kind}_scalars'] = torch.randn(12, ns)
            x[f'binding_{kind}_vectors'] = torch.randn(12, nv, 3)
        original = {k: v.clone() for k, v in x.items()}
        candidate = PeptideConstrainedLoopPairRefiner(width=8, layers=1)
        result = candidate(x)
        loss = result['backbone'].square().mean() + result['sequence_logits'].square().mean()
        loss.backward()
        grads = [p.grad for p in candidate.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in x.items()))
        with torch.no_grad():
            again = candidate(x)
        torch.testing.assert_close(result['backbone'].detach(), again['backbone'])


if __name__ == '__main__':
    unittest.main()

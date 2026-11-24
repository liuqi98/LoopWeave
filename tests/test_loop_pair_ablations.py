import unittest
import torch
from persistent_loop_pair_refiner import LoopPairLayer
from loop_pair_ablation_refiner import AblatedLoopPairLayer


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.h = torch.randn(5, 16)
        self.bb = torch.randn(5, 4, 3)
        self.z = torch.randn(5, 5, 16)
        self.p = torch.randn(5, 20).softmax(-1)
        self.original = LoopPairLayer(16)
        for layer in (self.original.node, self.original.rigid):
            torch.nn.init.normal_(layer.weight, std=.01)

    def make(self, variant):
        model = AblatedLoopPairLayer(16, variant)
        model.load_state_dict(self.original.state_dict())
        return model

    def test_full_matches_original(self):
        expected = self.original(self.h, self.bb, self.z, self.p)
        actual = self.make('full')(self.h, self.bb, self.z, self.p)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_reset_ignores_previous_pair(self):
        model = self.make('reset_pair')
        for a, b in zip(model(self.h,self.bb,self.z,self.p),model(self.h,self.bb,self.z*3,self.p)):
            self.assertTrue(torch.equal(a, b))

    def test_no_sequence_ignores_probabilities(self):
        model = self.make('no_sequence')
        for a, b in zip(model(self.h,self.bb,self.z,self.p),model(self.h,self.bb,self.z,self.p.roll(1,-1))):
            self.assertTrue(torch.equal(a, b))

    def test_no_triangle_ignores_triangle_weights(self):
        model = self.make('no_triangle')
        expected = model(self.h,self.bb,self.z,self.p)
        with torch.no_grad():
            model.left.weight.add_(3)
        actual = model(self.h,self.bb,self.z,self.p)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a,b))
        sum(t.square().mean() for t in actual).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

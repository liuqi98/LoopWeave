import unittest
import torch
from persistent_loop_pair_refiner import LoopPairLayer


class LoopPairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026);self.layer=LoopPairLayer(8).double()
        self.h=torch.randn(5,8,dtype=torch.double);self.bb=torch.randn(5,4,3,dtype=torch.double)
        self.z=torch.zeros(5,5,8,dtype=torch.double);self.p=torch.randn(5,20,dtype=torch.double).softmax(-1)
    def test_identity_and_gradient(self):
        h,bb,z=self.layer(self.h,self.bb,self.z,self.p)
        self.assertTrue(torch.equal(h,self.h));self.assertTrue(torch.equal(bb,self.bb))
        self.assertGreater(float(z.abs().sum()),0)
        (h.square().sum()+bb.square().sum()).backward()
        self.assertGreater(float(self.layer.rigid.weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(self.layer.rigid.weight.grad).all())
    def test_state_used_and_equivariance(self):
        torch.nn.init.normal_(self.layer.rigid.weight,std=.02)
        torch.nn.init.normal_(self.layer.node.weight,std=.02)
        h,bb,z=self.layer(self.h,self.bb,self.z,self.p)
        _,_,z2=self.layer(self.h,self.bb,torch.randn_like(self.z),self.p)
        self.assertFalse(torch.allclose(z,z2))
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.double))
        if torch.det(q)<0:q[:,0]*=-1
        t=torch.tensor([4.,-2.,7.],dtype=torch.double)
        hr,br,zr=self.layer(self.h,self.bb@q+t,self.z,self.p)
        self.assertTrue(torch.allclose(hr,h,atol=1e-9));self.assertTrue(torch.allclose(zr,z,atol=1e-9))
        self.assertTrue(torch.allclose(br,bb@q+t,atol=1e-9))


if __name__=='__main__':unittest.main()

"""Unit checks for projected update, including unused heads and zero gradients."""
import unittest
import torch
from train_p0_guarded import surgery,InjectUpdate

class ProjectionTests(unittest.TestCase):
 def check(self,d,p):
  g,m=surgery([d],[p],[True],.1)
  self.assertLessEqual(m['weighted_ratio_after'],.100001)
  self.assertGreaterEqual(float(torch.dot(d,g[0]-d)),-1e-5)
  self.assertTrue(torch.isfinite(g[0]).all())
  return g,m
 def test_opposition(self):
  g,m=self.check(torch.tensor([1.,0.]),torch.tensor([-10.,2.]));self.assertTrue(m['conflict']);torch.testing.assert_close(g[0],torch.tensor([1.,.1]))
 def test_aligned_cap(self):
  g,m=self.check(torch.tensor([1.,0.]),torch.tensor([100.,0.]));torch.testing.assert_close(g[0],torch.tensor([1.1,0.]))
 def test_small_physics_preserved(self):
  g,m=self.check(torch.tensor([1.,0.]),torch.tensor([.1,0.]));torch.testing.assert_close(g[0],torch.tensor([1.01,0.]))
 def test_zero_data(self):
  g,m=self.check(torch.zeros(2),torch.ones(2));torch.testing.assert_close(g[0],torch.zeros(2))
 def test_zero_physics(self):
  g,m=self.check(torch.ones(2),torch.zeros(2));torch.testing.assert_close(g[0],torch.ones(2))
 def test_unused_head(self):
  g,m=surgery([torch.ones(2),None,None],[torch.ones(2),torch.ones(1),None],[True,False,False],.1);self.assertIsNone(g[-1]);torch.testing.assert_close(g[1],torch.tensor([.1]))
 def test_gradient_injection(self):
  p=torch.nn.Parameter(torch.ones(2));g=torch.tensor([.7,-.4]);loss=InjectUpdate.apply(torch.tensor(3.),[g],p);self.assertEqual(float(loss.detach()),3.);loss.backward();torch.testing.assert_close(p.grad,g)
 def test_random_vectors(self):
  generator=torch.Generator().manual_seed(20261009)
  for _ in range(100):self.check(torch.randn(100,generator=generator),torch.randn(100,generator=generator)*30)

if __name__=='__main__':unittest.main()

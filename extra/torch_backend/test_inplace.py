import unittest
import torch
import extra.torch_backend.backend
device = "tiny"
torch.set_default_device(device)
import numpy as np

class TestTorchBackendInplace(unittest.TestCase):
  def test_zero(self):
    a = torch.ones(4, device=device)
    a.zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_view_zero(self):
    a = torch.ones(4, device=device)
    a.view((2, 2)).zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_slice_zero(self):
    a = torch.ones(4, device=device)
    a[2:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [1,1,0,0])

  def test_slice_permute_zero(self):
    a = torch.ones((3,2), device=device)
    a.permute(1,0)[1:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [[1,0],[1,0],[1,0]])

  def test_slice_fill(self):
    a = torch.empty(4, device=device)
    a[2:].fill_(2)
    np.testing.assert_equal(a.cpu().numpy(), [0,0,2,2])

if __name__ == "__main__":
  unittest.main()

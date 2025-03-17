import unittest
import torch
import tinygrad.frontend.torch
torch.set_default_device("tiny")
import numpy as np

class TestTorchBackendInplace(unittest.TestCase):
  def test_zero(self):
    a = torch.ones(4)
    a.zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_view_zero(self):
    a = torch.ones(4)
    a.view((2, 2)).zero_()
    np.testing.assert_equal(a.cpu().numpy(), [0,0,0,0])

  def test_slice_zero(self):
    a = torch.ones(4)
    a[2:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [1,1,0,0])

  def test_slice_permute_zero(self):
    a = torch.ones((3,2))
    a.permute(1,0)[1:].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [[1,0],[1,0],[1,0]])

  def test_slice_fill(self):
    a = torch.zeros(4)
    a[2:].fill_(2)
    np.testing.assert_equal(a.cpu().numpy(), [0,0,2,2])

  def test_slice_mul(self):
    a = torch.ones(4)
    a[:2] *= 3
    a[2:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [3,3,2,2])

  def test_stacked_mul(self):
    a = torch.ones((3,3))
    b = a[1:,1:].permute(1,0)
    c = b[1:,:]
    b *= 2
    c *= 3
    np.testing.assert_equal(a.cpu().numpy(), [[1,1,1],[1,2,6],[1,2,6]])

  def test_flatten_reshape_add(self):
    a = torch.zeros((2,2,12,32))
    b = a.flatten()
    c = b.reshape((48,32))
    a += 1
    b += 1
    c += 1
    np.testing.assert_equal(c.cpu().numpy(), torch.full((48,32),3).cpu().numpy())

  def test_noncontig(self):
    a = torch.empty_strided((4,4),(1,4), dtype=torch.int64)
    # self.assertFalse(a.is_contiguous()) # TODO: we are contiguous when it's not required
    a.zero_()
    b = a.view((4,4))
    b[1:3,:] += 1
    np.testing.assert_equal(a.cpu().numpy(), [[0]*4,[1]*4,[1]*4,[0]*4])

  def test_topk_inplace(self):
    x = torch.tensor([1, 3, 2, 4])
    sorted_tensor = torch.empty(2, dtype=x.dtype)
    indices_tensor = torch.empty(2, dtype=torch.int64)
    torch.topk(x, k=2, out=(sorted_tensor, indices_tensor))
    np.testing.assert_equal(sorted_tensor.cpu().numpy(), [4, 3])
    np.testing.assert_equal(indices_tensor.cpu().numpy(), [3, 1])

  def test_sort_inplace(self):
    x = torch.tensor([3, 1, 4, 2])
    sorted_tensor = torch.empty_like(x, dtype=x.dtype)
    indices_tensor = torch.empty_like(x, dtype=torch.int64)
    torch.sort(x, out=(sorted_tensor, indices_tensor))
    np.testing.assert_equal(sorted_tensor.cpu().numpy(), [1, 2, 3, 4])
    np.testing.assert_equal(indices_tensor.cpu().numpy(), [1, 3, 0, 2])

if __name__ == "__main__":
  unittest.main()

# simple tests
import unittest
import torch
import numpy as np
import extra.torch_backend.backend  # "tiny" backend is installed

class TestTorchBackend(unittest.TestCase):
  def test_numpy_ones(self):
    a = torch.ones(4, device="tiny")
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_numpy_ones(self):
    a = torch.ones(4, dtype=torch.int32, device="tiny")
    assert a.dtype == torch.int32
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_plus(self):
    a = torch.ones(4, device="tiny")
    b = torch.ones(4, device="tiny")
    c = a+b
    np.testing.assert_equal(c.cpu().numpy(), [2,2,2,2])

  def test_eq(self):
    a = torch.ones(4, device="tiny")
    b = torch.ones(4, device="tiny")
    c = a == b
    print(c.cpu().numpy())

  def test_isfinite(self):
    a = torch.ones(4, device="tiny")
    np.testing.assert_equal(torch.isfinite(a), [True, True, True, True])

  # TODO: why
  def test_str(self):
    a = torch.ones(4, device="tiny")
    print(str(a))

if __name__ == "__main__":
  unittest.main()

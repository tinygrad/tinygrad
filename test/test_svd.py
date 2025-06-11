import numpy as np
import unittest
from tinygrad import Tensor

class TestSVD(unittest.TestCase):
  def test_svd_basic(self):
    A = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (3, 2))
    self.assertEqual(S.shape, (2,))
    self.assertEqual(Vh.shape, (2, 2))
    
    U_full, S_full, Vh_full = Tensor.svd(A, full_matrices=True)
    
    self.assertEqual(U_full.shape, (3, 3))
    self.assertEqual(S_full.shape, (2,))
    self.assertEqual(Vh_full.shape, (2, 2))

  def test_svd_reconstruction(self):
    A = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    S_diag = Tensor.zeros(2, 2).contiguous()
    S_diag[0, 0] = S[0]
    S_diag[1, 1] = S[1]
    A_reconstructed = U @ S_diag @ Vh
    
    diff = (A - A_reconstructed).abs().max()
    self.assertLess(diff.item(), 1e-4)

  def test_svd_singular_values_descending(self):
    A = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    S_np = S.numpy()
    for i in range(len(S_np) - 1):
      self.assertGreaterEqual(S_np[i], S_np[i+1])

  def test_svd_square_matrix(self):
    A = Tensor([[1., 2.], [3., 4.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (2, 2))
    self.assertEqual(S.shape, (2,))
    self.assertEqual(Vh.shape, (2, 2))

  def test_svd_tall_matrix(self):
    A = Tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (4, 2))
    self.assertEqual(S.shape, (2,))
    self.assertEqual(Vh.shape, (2, 2))

  def test_svd_wide_matrix(self):
    A = Tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (2, 2))
    self.assertEqual(S.shape, (2,))
    self.assertEqual(Vh.shape, (2, 4))

  def test_svd_1x1_matrix(self):
    A = Tensor([[5.0]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (1, 1))
    self.assertEqual(S.shape, (1,))
    self.assertEqual(Vh.shape, (1, 1))
    
    np.testing.assert_allclose(S.numpy(), [5.0], atol=1e-6)

  def test_svd_matrix_with_zeros(self):
    A = Tensor([[1., 0.], [0., 0.], [0., 1.]])
    
    U, S, Vh = Tensor.svd(A, full_matrices=False)
    
    self.assertEqual(U.shape, (3, 2))
    self.assertEqual(S.shape, (2,))
    self.assertEqual(Vh.shape, (2, 2))
    
    S_np = S.numpy()
    self.assertTrue(all(s >= 0 for s in S_np))
    for i in range(len(S_np) - 1):
      self.assertGreaterEqual(S_np[i], S_np[i+1])

if __name__ == "__main__":
  unittest.main() 
import numpy as np
import unittest
from tinygrad import Tensor

Tensor.manual_seed(101)

class TestLinAlg(unittest.TestCase):
  def test_svd_general(self): 
    sizes = [(2,2),(5,3),(3,5),(2,2,2,2,3)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      U,S,V = Tensor.svd(a)
      b_shape,m,n = size[0:-2],size[-2],size[-1]
      k= min(m,n)
      s_diag = (S.unsqueeze(-2) * Tensor.eye(k).reshape((1,)*len(b_shape) + (k,k)))
      s_diag = s_diag.expand(b_shape + (k,k)).pad(tuple([(0,0) for _ in range(len(size)-2)] + [(0,m-k), (0,n-k)]))
      reconstructed_tensor = U @ s_diag @ V
      u_identity = (Tensor.eye(m).reshape((1,) * len(b_shape)+(m,m)).expand(b_shape+(m,m)))
      v_identity = (Tensor.eye(n).reshape((1,) * len(b_shape)+(n,n)).expand(b_shape+(n,n)))
      np.testing.assert_allclose(reconstructed_tensor.numpy(),a.numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose((U @ U.transpose(-2,-1)).numpy(),u_identity.numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose((V @ V.transpose(-2,-1)).numpy(),v_identity.numpy(),atol=1e-5,rtol=1e-5)
  def test_svd_nonfull(self):
    sizes = [(2,2),(5,3),(3,5),(2,2,2,2,3)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      U,S,V = Tensor.svd(a,full_matrices=False)
      b_shape,m,n = size[0:-2],size[-2],size[-1]
      k = min(m,n)
      s_diag = (S.unsqueeze(-2) * Tensor.eye(k).reshape((1,) * len(b_shape)+(k,k)).expand(b_shape+(k,k)))
      reconstructed_tensor = U @ s_diag @ V
      u_identity = (Tensor.eye(k).reshape((1,)*len(b_shape)+(k,k)).expand(b_shape+(k,k)))
      v_identity = (Tensor.eye(k).reshape((1,)*len(b_shape)+(k,k)).expand(b_shape+(k,k)))
      #reduced U,V is only orthogonal along smaller dim
      if (m < n): U_r,V_r = U @ U.transpose(-2,-1),V @ V.transpose(-2,-1)
      else: U_r,V_r = U.transpose(-2,-1) @ U, V.transpose(-2,-1) @ V
      np.testing.assert_allclose(reconstructed_tensor.numpy(),a.numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose(U_r.numpy(),u_identity.numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose(V_r.numpy(),v_identity.numpy(),atol=1e-5,rtol=1e-5)
  def test_qr_general(self):
    sizes = [(3,3),(3,6),(6,3),(2,2,2,2,2)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      Q,R = Tensor.qr(a)
      reconstructed_tensor = Q @ R
      Q_size = size[-2]
      identity = Tensor.eye(Q_size).reshape((1,) * (len(size)-2)+(Q_size,Q_size)).expand(size[0:-2]+(Q_size,Q_size))
      np.testing.assert_allclose(reconstructed_tensor.numpy(),a.numpy(),atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose((Q @ Q.transpose(-2,-1)).numpy(),identity.numpy(),atol=1e-5,rtol=1e-5)

if __name__ == "__main__":
  unittest.main() 
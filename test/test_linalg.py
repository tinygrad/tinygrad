import numpy as np
import unittest
from tinygrad import Tensor

class TestLinAlg(unittest.TestCase):
  def test_svd_general(self): 
    sizes = [(2,2),(3,3),(3,6),(6,3)]
    #sizes = [(2,2),(4,2),(2,4),(5,3),(3,5),(2,2,2),(2,2,4),(2,4,2)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      U,S,V = Tensor.svd(a)
      num_single_values= min(size[0],size[1])
      m,n = size
      S_diag = (Tensor.eye(min(m,n)) * S).pad(((0,m-num_single_values),(0,n-num_single_values)))
      reconstructed_tensor = U @ S_diag @ V.transpose()
      np.testing.assert_allclose(reconstructed_tensor.numpy(), a.numpy() , atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose((U @ U.transpose()).numpy(),Tensor.eye(m).numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose((V @ V.transpose()).numpy(),Tensor.eye(n).numpy(),atol=1e-5,rtol=1e-5)
  def test_qr_general(self):
    sizes = [(3,3),(3,6),(6,3),(2,2,2,2,2)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      Q,R = Tensor.qr(a)
      reconstructed_tensor = Q @ R
      Q_size = size[-2]
      identity = Tensor.eye(Q_size).reshape((1,)*(len(size)-2)+(Q_size,Q_size)).expand(size[0:-2]+(Q_size,Q_size))
      np.testing.assert_allclose(reconstructed_tensor.numpy(), a.numpy() , atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose((Q @ Q.transpose(-2,-1)).numpy(),identity.numpy(),atol=1e-5,rtol=1e-5)

if __name__ == "__main__":
  unittest.main() 
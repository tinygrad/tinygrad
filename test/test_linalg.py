import numpy as np
import unittest
from tinygrad import Tensor

 #TODO: create orthogonality test. create reconstruction test if needed
def test_reconstruction(U):
  pass
class TestLinAlg(unittest.TestCase):
  def test_svd_general(self): 
    sizes = [(2,2),(3,3),(3,6),(6,3)]
    #sizes = [(2,2),(4,2),(2,4),(5,3),(3,5),(2,2,2),(2,2,4),(2,4,2)]
    for size in sizes:
      a = Tensor.randn(size[0],size[1]).realize()
      U,S,V = Tensor.svd(a)
      num_single_values= min(size[0],size[1])
      m,n = size
      S_diag = (Tensor.eye(min(m,n)) * S).pad(((0,m-num_single_values),(0,n-num_single_values)))
      reconstructed_tensor = U @ S_diag @ V.transpose()
      np.testing.assert_allclose(reconstructed_tensor.numpy(), a.numpy() , atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose((U @ U.transpose()).numpy(),Tensor.eye(m).numpy(),atol=1e-5,rtol=1e-5)
      np.testing.assert_allclose((V @ V.transpose()).numpy(),Tensor.eye(n).numpy(),atol=1e-5,rtol=1e-5)
  # def test_singular_matrix(): #test nans
  #   pass
  # def test_large_matrix():
  #   pass
  # def test_qr_general():
  #   pass

if __name__ == "__main__":
  unittest.main() 
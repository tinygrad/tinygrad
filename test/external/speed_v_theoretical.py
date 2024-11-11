import unittest, time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context, DEBUG

class TestKernelSpeed(unittest.TestCase):
  def _test_matmul(self, M, N=None, K=None, nv=None, amd=None):
    # (MxK) @ (KxN)
    @TinyJit
    def f(a, b): return (a @ b).realize()

    if N is None: N = M
    if K is None: K = M
    tms = []
    with Context(BEAM=3):
      for _ in range(10):
        with Context(BEAM=0, DEBUG=0):
          a = Tensor.rand(M, K, dtype="half").realize()
          b = Tensor.rand(K, N, dtype="half").realize()
        Device.default.synchronize()
        st = time.perf_counter()
        _c = f(a, b)
        Device.default.synchronize()
        tms.append(time.perf_counter() - st)

    ops = 2 * M * N * K
    tm = min(tms)
    tflops = ops / tm / 1e12

    if DEBUG >= 1:
      print(f"{tm=}")
      print(f"{tflops=}")

    if Device.DEFAULT == "NV":
      if DEBUG >=1: print(f"target: {nv}")
      self.assertGreater(tflops, nv)
    if Device.DEFAULT == "AMD":
      if DEBUG >=1: print(f'target: {amd}')
      self.assertGreater(tflops, amd)

  # TODO: smaller ones has other overhead in synchronize
  # TODO: AMD number can be better (perf level?)
  def test_gemm_1024(self): self._test_matmul(1024, nv=9, amd=7)
  def test_gemm_2048(self): self._test_matmul(2048, nv=50, amd=20)
  def test_gemm_4096(self): self._test_matmul(4096, nv=100, amd=30)
  def test_gemm_8192(self): self._test_matmul(8192, nv=130, amd=50)

  # TODO: add gemv, which is memory bounded


if __name__ == '__main__':
  unittest.main()
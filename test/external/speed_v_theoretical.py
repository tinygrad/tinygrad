import unittest, time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context, DEBUG

class TestKernelSpeed(unittest.TestCase):
  def _test_matmul(self, M, K=None, N=None, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
    # (MxK) @ (KxN)
    @TinyJit
    def f(a, b) -> Tensor: return (a @ b).realize()

    if N is None: N = M
    if K is None: K = M
    tms = []
    with Context(BEAM=3):
      for _ in range(10):
        with Context(BEAM=0, DEBUG=0):
          # TODO: randn is 20% faster than rand for gemv
          a = Tensor.randn(M, K, dtype="half").realize()
          b = Tensor.randn(K, N, dtype="half").realize()
        Device.default.synchronize()
        st = time.perf_counter()
        c = f(a, b)
        Device.default.synchronize()
        tms.append(time.perf_counter() - st)

    ops = 2 * M * N * K
    mems = a.dtype.itemsize * M * K + b.dtype.itemsize * K * N + c.dtype.itemsize * M * N
    tm = min(tms)
    tflops = ops / tm / 1e12
    gbs = mems / tm / 1e9

    if DEBUG >= 1:
      print(f"{tm=:.6f}")
      print(f"{tflops=:.6f}")
      print(f"{gbs=:.3f}")

    if Device.DEFAULT == "NV":
      if nv_tflops is not None:
        if DEBUG >=1: print(f"tflop/s target: {nv_tflops}")
        self.assertGreater(tflops, nv_tflops)
      if nv_gbs is not None:
        if DEBUG >=1: print(f"gb/s target: {nv_gbs}")
        self.assertGreater(gbs, nv_gbs)

    if Device.DEFAULT == "AMD":
      if amd_tflops is not None:
        if DEBUG >=1: print(f"tflop/s target: {amd_tflops}")
        self.assertGreater(tflops, amd_tflops)
      if amd_gbs is not None:
        if DEBUG >=1: print(f"gb/s target: {amd_gbs}")
        self.assertGreater(gbs, amd_gbs)

  # TODO: smaller ones has other overhead in synchronize
  # def test_gemm_1024(self): self._test_matmul(1024, nv_tflops=8, amd_tflops=7)
  # def test_gemm_2048(self): self._test_matmul(2048, nv_tflops=50, amd_tflops=30)
  def test_gemm_4096(self): self._test_matmul(4096, nv_tflops=100, amd_tflops=70)
  def test_gemm_8192(self): self._test_matmul(8192, nv_tflops=130, amd_tflops=70)

  def test_gemv_16384_4096(self): self._test_matmul(16384, 4096, 1, nv_gbs=430, amd_gbs=400)
  def test_gemv_4096_16384(self): self._test_matmul(4096, 16384, 1, nv_gbs=430, amd_gbs=400)

if __name__ == '__main__':
  unittest.main()
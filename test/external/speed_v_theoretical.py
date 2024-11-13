import unittest, time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context, DEBUG
from tinygrad.nn import Conv2d

class TestKernelSpeed(unittest.TestCase):
  def _get_tensor(self, *shape:int):
    with Context(BEAM=0, DEBUG=0):
      # TODO: randn is 20% faster than rand for gemv
      return Tensor.randn(shape, dtype="half").realize()

  def _compare(self, tm, tflops, gbs, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
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

  def _test_matmul(self, M, K=None, N=None, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
    # (MxK) @ (KxN)
    @TinyJit
    def f(a, b) -> Tensor: return (a @ b).realize()

    if N is None: N = M
    if K is None: K = M
    tms = []
    with Context(BEAM=3):
      for _ in range(10):
        a = self._get_tensor(M, K)
        b = self._get_tensor(K, N)
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
    self._compare(tm, tflops, gbs, nv_tflops, nv_gbs, amd_tflops, amd_gbs)

  def _test_conv_3x3(self, BS, CIN, COUT, H, W, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
    @TinyJit
    def f(conv, x) -> Tensor: return conv(x).realize()
    tms = []
    K = 3
    with Context(BEAM=2):
      conv = Conv2d(CIN, COUT, K, padding=1)
      for _ in range(10):
        x = self._get_tensor(BS, CIN, H, W)
        Device.default.synchronize()
        st = time.perf_counter()
        _c = f(conv, x)
        Device.default.synchronize()
        tms.append(time.perf_counter() - st)

    # naive algo
    ops = 2 * BS * CIN * COUT * K * K * H * W
    # TODO: what should this be?
    mems = 0
    tm = min(tms)
    tflops = ops / tm / 1e12
    gbs = mems / tm / 1e9
    self._compare(tm, tflops, gbs, nv_tflops, nv_gbs, amd_tflops, amd_gbs)

  # TODO: smaller ones has other overhead in synchronize
  # def test_gemm_1024(self): self._test_matmul(1024, nv_tflops=8, amd_tflops=7)
  # def test_gemm_2048(self): self._test_matmul(2048, nv_tflops=50, amd_tflops=30)
  def test_gemm_4096(self): self._test_matmul(4096, nv_tflops=100, amd_tflops=70)
  def test_gemm_8192(self): self._test_matmul(8192, nv_tflops=130, amd_tflops=70)

  def test_gemv_16384_4096(self): self._test_matmul(16384, 4096, 1, nv_gbs=430, amd_gbs=400)
  def test_gemv_4096_16384(self): self._test_matmul(4096, 16384, 1, nv_gbs=430, amd_gbs=400)

  def test_conv_3x3_256_32_32_256_256(self): self._test_conv_3x3(256, 32, 32, 256, 256, nv_tflops=30, amd_tflops=22)

if __name__ == '__main__':
  unittest.main()
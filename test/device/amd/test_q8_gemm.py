import functools, unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestQ8Gemm(unittest.TestCase):
  def test_staged_q8(self):
    target = Device[Device.DEFAULT].target[0]
    if target not in (11, 12): self.skipTest("requires RDNA3/4")
    rng = np.random.default_rng(41)
    for M, K, tile, bn in ((32, 32, 16, 64), (33, 32, 32, 64), (32, 33, 32, 64), (32, 32, 32, 32)):
      with self.subTest(invalid=(M, K, tile, bn)), self.assertRaises(AssertionError):
        Tensor.custom_kernel(Tensor.empty(M, 32), Tensor.empty(544, dtype=dtypes.uint16), Tensor.empty(M, K),
                             fxn=functools.partial(_q8_gemm_kernel, token_tile=tile, block_n=bn, rdna4=target == 12))
    # Odd group count exercises GS=1 with padded weight staging for block_n=64.
    for M, N, K, tile, bn, splits in ((32, 144, 288, 32, 64, 1), (64, 128, 512, 64, 64, 1),
                                      (192, 320, 1024, 64, 64, 1), (128, 144, 256, 64, 128, 1),
                                      (352, 320, 10240, 32, 64, 4), (352, 320, 10240, 32, 64, 16)):
      with self.subTest(shape=(M, N, K), tile=tile, block_n=bn, splits=splits):
        scales = rng.uniform(0.001, 0.04, size=(N, K//32, 1)).astype(np.float16)
        quant = rng.integers(-127, 128, size=(N, K//32, 32), dtype=np.int8)
        raw = np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()
        weights = (quant.astype(np.float32)*scales.astype(np.float32)).astype(np.float16).reshape(N, K)
        tw = Tensor(raw).realize()
        for dtype, npdtype in (((dtypes.float32, np.float32),) if splits > 1 else ((dtypes.float32, np.float32), (dtypes.half, np.float16))):
          @TinyJit
          def run(x, w):
            fxn = functools.partial(_q8_gemm_kernel, token_tile=tile, block_n=bn, rdna4=target == 12, k_splits=splits)
            y = Tensor.custom_kernel(Tensor.empty(splits*M, N, dtype=dtype), w, x, fxn=fxn)[0]
            return y.reshape(splits, M, N).sum(0).realize()
          for _ in range(3):
            data = rng.normal(size=(M, K)).astype(np.float16)
            actual = run(Tensor(data).realize(), tw).numpy()
            ref = (data.astype(np.float32) @ weights.astype(np.float32).T).astype(npdtype)
            np.testing.assert_allclose(actual, ref, rtol=0.002, atol=float(np.abs(ref).max())*0.0001)

  def test_tile96_same_bits(self):
    # Q8_TT96: token tiles of 96 (block 128, two groups a stage) give the bits of the tiles of 128 (the same sum order along K)
    target = Device[Device.DEFAULT].target[0]
    if target != 11: self.skipTest("requires RDNA3")
    rng = np.random.default_rng(7)
    for M, N, K in ((1152, 1024, 2560), (768, 512, 1024)):
      scales = rng.uniform(0.001, 0.04, size=(N, K//32, 1)).astype(np.float16)
      quant = rng.integers(-127, 128, size=(N, K//32, 32), dtype=np.int8)
      tw = Tensor(np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()).realize()
      x = Tensor(rng.normal(size=(M, K)).astype(np.float16)).realize()
      outs = [Tensor.custom_kernel(Tensor.empty(M, N, dtype=dtype), tw, x, fxn=functools.partial(_q8_gemm_kernel, token_tile=tile, block_n=128,
              rdna4=False))[0].numpy() for tile in (128, 96) for dtype in (dtypes.float32, dtypes.half)]
      with self.subTest(shape=(M, N, K)):
        np.testing.assert_array_equal(outs[0], outs[2])
        np.testing.assert_array_equal(outs[1], outs[3])

if __name__ == "__main__": unittest.main()

import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_qwen import amd_custom_kernels_supported, ghc_prefill_norm, ghc_prefill_mix, Linear, q8_linear, Q8_0

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestGHCPrefill(unittest.TestCase):
  def test_prefill(self):
    if not amd_custom_kernels_supported("AMD"): self.skipTest("requires AMD custom kernels")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(31)
    C, D, eps = 4, 2560, 1e-6
    w = rng.normal(size=(C, D)).astype(np.float32)
    wi = rng.normal(size=(C, C, D)).astype(np.float32)
    wt, wit = Tensor(w.reshape(-1)).shard(devices).realize(), Tensor(wi.reshape(C, -1)).shard(devices).realize()
    for T, dtype in ((192, dtypes.float16), (2112, dtypes.float32)):
      @TinyJit
      def run(x):
        ys = ghc_prefill_norm(x, wt, wit, C, eps)
        Tensor.realize(*ys)
        return ys
      for iteration in range(3):
        x = rng.normal(size=(T, C, D)).astype(np.float16 if dtype == dtypes.float16 else np.float32)
        x[0], x[1] = 0, 1e-7
        xf = x.astype(np.float32)
        expected = xf / np.sqrt(np.mean(xf*xf, axis=-1, keepdims=True) + eps) * w
        part = np.einsum("tcd,jcd->tcj", expected, wi)
        ys = run(Tensor(x.reshape(T, -1), dtype=dtype).shard(devices, axis=0).realize())
        actual, half, inject = [y.numpy() for y in ys]
        with self.subTest(tokens=T, dtype=dtype, iteration=iteration):
          np.testing.assert_allclose(actual.reshape(T, C, D), expected, rtol=3e-6, atol=2e-6)
          np.testing.assert_allclose(half.reshape(T, C, D).astype(np.float32), expected.astype(np.float16).astype(np.float32),
                                     rtol=.001, atol=1e-5)
          np.testing.assert_allclose(inject, part, rtol=2e-4, atol=5e-4)
    if Device[Device.DEFAULT].target[0] != 11: return
    K = 320
    scales = rng.uniform(.001, .01, size=(C*D, K//32, 1)).astype(np.float16)
    quant = rng.integers(-127, 128, size=(C*D, K//32, 32), dtype=np.int8)
    raw = np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()
    weights = (quant.astype(np.float32)*scales.astype(np.float32)).astype(np.float16).reshape(C*D, K).astype(np.float32)
    tw = Tensor(raw).shard(devices).realize()
    for T in (192, 2112):
      @TinyJit
      def mix(lo, xn): return ghc_prefill_mix(lo, tw, xn).realize()
      for iteration in range(3):
        lo = rng.normal(size=(T, K)).astype(np.float16)
        xn = rng.normal(size=(T, C*D)).astype(np.float32)
        gates = 1 / (1 + np.exp(-(lo.astype(np.float32) @ weights.T)))
        expected = (gates*xn).reshape(T, C, D).mean(1)
        actual = mix(Tensor(lo).shard(devices, axis=0).realize(), Tensor(xn).shard(devices, axis=0).realize()).numpy()
        with self.subTest(mix_tokens=T, iteration=iteration):
          np.testing.assert_allclose(actual, expected, rtol=.002, atol=2e-4)
    # Exercise q8_linear's sequence-sharded split-K reduction, not just the per-device kernel.
    T, N, K = 2112, 320, C*D
    scales = rng.uniform(.001, .01, size=(N, K//32, 1)).astype(np.float16)
    quant = rng.integers(-127, 128, size=(N, K//32, 32), dtype=np.int8)
    raw = np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()
    weights = (quant.astype(np.float32)*scales.astype(np.float32)).astype(np.float16).reshape(N, K).astype(np.float32)
    with Linear.empty_init(): layer = Linear(K, N, bias=False)
    layer.ggml_type, layer.weight = Q8_0, Tensor(raw).shard(devices).realize()
    @TinyJit
    def down(x): return q8_linear(layer, x, split=True).realize()
    for iteration in range(3):
      x = rng.normal(size=(T, K)).astype(np.float16)
      actual = down(Tensor(x).shard(devices, axis=0).realize()).numpy()
      expected = x.astype(np.float32) @ weights.T
      with self.subTest(split_k_iteration=iteration):
        np.testing.assert_allclose(actual, expected, rtol=.002, atol=float(np.abs(expected).max())*1e-4)

  def test_tile16(self):
    # q8_gemm's m_rows (the tiles of the first rows alone) in tiles of 16: the same bits on those rows as all of them in tiles of 32
    if not amd_custom_kernels_supported("AMD"): self.skipTest("requires AMD custom kernels")
    import functools
    from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel
    rng = np.random.default_rng(5)
    C, D, K, T = 4, 2560, 320, 32
    def q8raw(rows, k):
      scales = rng.uniform(.001, .01, size=(rows, k//32, 1)).astype(np.float16)
      quant = rng.integers(-127, 128, size=(rows, k//32, 32), dtype=np.int8)
      return Tensor(np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()).realize()
    up, down = q8raw(C*D, K), q8raw(K, C*D)
    xh = Tensor(rng.normal(size=(T, C*D)).astype(np.float16)).realize()
    lo = Tensor(rng.normal(size=(T, K)).astype(np.float16)).realize()
    xn = Tensor(rng.normal(size=(T, C*D)).astype(np.float32)).realize()
    def run(tt, mr):
      fd = functools.partial(_q8_gemm_kernel, token_tile=tt, block_n=64, k_splits=32, out_features=K, in_features=C*D, m_rows=mr)
      d = Tensor.custom_kernel(Tensor.empty(32*T, K, dtype=dtypes.float32), down, xh, fxn=fd)[0].numpy().reshape(32, T, K)
      fm = functools.partial(_q8_gemm_kernel, token_tile=tt, block_n=32, m_rows=mr)
      m = Tensor.custom_kernel(Tensor.empty(T, D, dtype=dtypes.float32), up, lo, xn, fxn=fm)[0].numpy()
      return d, m
    d0, m0 = run(32, 0)
    for R in (4, 16, 24):
      d1, m1 = run(16 if R <= 16 else 32, R)
      with self.subTest(R=R):
        np.testing.assert_array_equal(d1[:, :R].view(np.uint32), d0[:, :R].view(np.uint32))
        np.testing.assert_array_equal(m1[:R].view(np.uint32), m0[:R].view(np.uint32))

if __name__ == "__main__": unittest.main()

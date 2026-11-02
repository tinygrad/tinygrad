import functools, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.llm.kernels.amd import _moe_combine_kernel, amd_custom_kernels_supported

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestMoeCombine(unittest.TestCase):
  def test_prefetched_combine(self):
    if not amd_custom_kernels_supported("AMD"): self.skipTest("requires AMD custom kernels")
    rng = np.random.default_rng(63)
    T, K, D = 64, 10, 2560
    for by_selection in (False, True):
      for dtype in (dtypes.float32, dtypes.half):
        @TinyJit
        def run(down, pos, probs, extra):
          return Tensor.custom_kernel(Tensor.empty(T, D, dtype=dtype), down, pos, probs, extra,
                                       fxn=functools.partial(_moe_combine_kernel, k=K, by_selection=by_selection))[0].realize()
        for iteration in range(3):
          data = rng.normal(size=(T*K, D)).astype(np.float16)
          pos = rng.permutation(T*K).astype(np.int32)
          pos[rng.random(T*K) < .25] = -1
          probs = rng.random(T*K, dtype=np.float32)
          extra = rng.normal(size=(T, D)).astype(np.float32)
          indices = np.arange(T*K) if by_selection else np.maximum(pos, 0)
          expected = np.zeros((T, D), dtype=np.float64)
          for j in range(K): expected += (data[indices[j::K]].astype(np.float64)*probs[j::K, None]) * (pos[j::K, None] >= 0)
          expected += extra
          actual = run(*[Tensor(x).realize() for x in (data, pos, probs, extra)]).numpy()
          with self.subTest(by_selection=by_selection, dtype=dtype, iteration=iteration):
            np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=.001 if dtype == dtypes.half else 1e-5,
                                       atol=2e-6)

if __name__ == "__main__": unittest.main()

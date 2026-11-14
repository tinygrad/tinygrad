import functools, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd_qwen import moe_linear, q8_quantize, q8_quantize_cache_clear, _quant_decode_kernel
from tinygrad.llm.kernels.amd_qwen import IQ3_S, IQ4_NL, Q8_0, Q5_0, IQ3_XXS, IQ2_S, IQ4_XS

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestMoEDecode(unittest.TestCase):
  def test_narrow_slices_and_five_slots(self):
    if Device[Device.DEFAULT].target[0] != 11: self.skipTest("requires RDNA3")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(78)
    E, S = 8, 10
    def copies(t): return [Tensor(UOp.from_buffer(b, t.dtype, d)).numpy() for b, d in zip(t.uop.buffer.bufs, devices)]
    def quantized(a):
      a = a.reshape(-1, 32)
      d = np.maximum(np.abs(a).max(1, keepdims=True)/127, 1e-8)
      return (np.rint(a/d).clip(-127, 127)*d).reshape(-1, K)
    # (V3.1 UD-IQ4_XS: gate / up IQ3_XXS, IQ4_XS, IQ2_S; down Q5_0, IQ4_NL)
    for typ, K, N, block, size, XS in ((IQ3_S, 2560, 256, 256, 110, 1), (IQ4_NL, 128, 2560, 32, 18, S),
                                      (Q8_0, 128, 2560, 32, 34, S), (Q5_0, 128, 2560, 32, 22, S), (IQ3_XXS, 2560, 256, 256, 98, 1),
                                      (IQ4_XS, 2560, 256, 256, 136, 1), (IQ2_S, 2560, 256, 256, 82, 1)):
      raw = rng.integers(0, 256, (E*N*K//block, size), dtype=np.uint8)
      raw[:, :2] = rng.uniform(.001, .02, (len(raw), 1)).astype(np.float16).view(np.uint8)
      weights = ggml_data_to_tensor(Tensor(raw.ravel(), device="CPU"), E*N*K, typ).numpy().reshape(E, N, K)
      w = Tensor(raw.copy().view(np.uint16).ravel()).shard(devices).realize()
      first = Tensor([0], dtype=dtypes.int32).shard(devices).realize()
      for T in (1, 5, 6):
        @TinyJit
        def run(x, sel): return moe_linear(w, typ, E, x, sel, first).realize()
        @TinyJit
        def dense(x):
          xq = q8_quantize(x.reshape(-1, K)[:T], T, K)
          return Tensor.custom_kernel(Tensor.empty(T, N, device=devices), w, *xq,
                                      fxn=functools.partial(_quant_decode_kernel, out_features=N, in_features=K, ggml_type=Q8_0))[0].realize()
        for iteration in range(3):
          values = rng.normal(size=(1, T, XS, K)).astype(np.float32)
          if iteration == 0: values[..., 0, :] *= 1e-8
          ids = rng.integers(0, E, (1, T, S), dtype=np.int32)
          ids[..., 0], ids[..., -1] = -1, E  # both bounds of the inactive-expert range
          x, sel = [Tensor(a).shard(devices).realize() for a in (values, ids)]
          got = run(x, sel)
          act = quantized(values)
          indices = ids.reshape(-1)
          rows = np.arange(T*S)//S if XS == 1 else np.arange(T*S)
          ref = np.einsum("rk,rnk->rn", act[rows], weights[indices.clip(0, E-1)], dtype=np.float64)
          ref[(indices < 0) | (indices >= E)] = 0
          for rank, actual in enumerate(copies(got)):
            with self.subTest(quant=typ, tokens=T, iteration=iteration, rank=rank):
              np.testing.assert_allclose(actual.reshape(-1, N), ref, rtol=3e-4, atol=max(np.abs(ref).max()*3e-5, 1e-10))
          if typ == Q8_0:
            ref = act[:T].astype(np.float64) @ weights[0].astype(np.float64).T
            for actual in copies(dense(x)):
              np.testing.assert_allclose(actual.reshape(T, N), ref, rtol=3e-4, atol=max(np.abs(ref).max()*3e-5, 1e-10))
          q8_quantize_cache_clear()

if __name__ == "__main__": unittest.main()

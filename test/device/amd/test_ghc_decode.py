import functools, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_qwen import ghc_decode_mix, q8_quantize, q8_quantize_cache_clear, _quant_decode_kernel, Q8_0

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD")
class TestGHCDecode(unittest.TestCase):
  def test_mix_and_quantization(self):
    if Device[Device.DEFAULT].target[0] != 11: self.skipTest("requires RDNA3")
    devices = tuple(f"AMD:{i}" for i in range(getenv("TEST_GPUS", 6)))
    rng = np.random.default_rng(74)
    C, D, R = 4, 2560, 320
    scales = rng.uniform(.001, .02, size=(C*D, R//32, 1)).astype(np.float16)
    quant = rng.integers(-127, 128, size=(C*D, R//32, 32), dtype=np.int8)
    raw = np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel()
    w = Tensor(raw).shard(devices).realize()
    def copies(t): return [Tensor(UOp.from_buffer(b, t.dtype, d)).numpy() for b, d in zip(t.uop.buffer.bufs, devices)]
    for T in (2, 5, 8):
      @TinyJit
      def run(lo, xn, part):
        mixed, inj = ghc_decode_mix(lo, w, xn, part, C)
        packed = q8_quantize(mixed, T, D)
        Tensor.realize(mixed, inj, *packed)
        return mixed, inj, *packed
      for iteration in range(3):
        lo = rng.normal(size=(T, R)).astype(np.float32)
        lo[0] = 0
        lo[-1] = np.linspace(-50, 50, R)
        x = rng.normal(size=(T, C*D)).astype(np.float16)
        p = rng.normal(size=(T, C, C)).astype(np.float32)
        lt, xt, pt = [Tensor(a).shard(devices).realize() for a in (lo, x, p)]
        got = run(lt, xt, pt)
        z = lt*(1/C)
        act = (z*z.sigmoid()).contiguous()
        xq = q8_quantize(act, T, R)
        up = Tensor.custom_kernel(Tensor.empty(T, C*D, device=devices), w, *xq,
                                   fxn=functools.partial(_quant_decode_kernel, out_features=C*D, in_features=R, ggml_type=Q8_0))[0]
        mixed = (up.reshape(T, C, D).sigmoid()*xt.reshape(T, C, D).float()).mean(1).contiguous()
        inj = (pt.sum(1)*(1/C)).contiguous()
        # Independently re-quantize the realized mix, not its cached graph representation.
        q8_quantize_cache_clear()
        expected = (mixed, inj, *q8_quantize(got[0], T, D))
        Tensor.realize(*expected)
        for index, (a, b) in enumerate(zip(got, expected)):
          for rank, (actual, ref) in enumerate(zip(copies(a), copies(b))):
            with self.subTest(tokens=T, iteration=iteration, output=index, rank=rank):
              if index == 0: np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=4e-6)
              else: np.testing.assert_array_equal(actual, ref)
        q8_quantize_cache_clear()

if __name__ == "__main__": unittest.main()

import unittest, numpy as np
from tinygrad import Tensor, Device, nn
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, rms_norm_q8, p2p_comm, q8_quantize, _q8_cache, _p2p_pending

def devices(n:int) -> tuple[str, ...]: return tuple(f"{Device.DEFAULT}:{i}" if i else Device.DEFAULT for i in range(n))
def has_p2p(n:int) -> bool:
  try: return p2p_comm(devices(n)) is not None
  except Exception: return False

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "RDNA3/4 required")
class TestRMSNormQ8(unittest.TestCase):
  def _check(self, n:int):
    Tensor.manual_seed(0)
    D, devs = 5120, devices(n)
    x, w, parts, norm = Tensor.randn(1, 1, D).realize(), Tensor.randn(D).half().realize(), Tensor.randn(n, 1, D).realize(), nn.RMSNorm(D, 1e-6)
    ref = norm(ref_h:=x + parts.sum(0)).numpy() * w.float().numpy()
    if n > 1: x, parts, w = x.shard(devs), parts.shard(devs, 0), w.shard(devs)
    norm.weight = w
    for _ in range(3):  # the exchange alternates its mailbox regions
      a = parts.sum(0).reshape(1, 1, D)
      if n > 1: _p2p_pending[a.uop] = (p2p_comm(devs), parts)  # as registered by a row parallel q8_linear
      h, xn = rms_norm_q8(norm, x, a)
      Tensor.realize(h, xn, *(q8:=_q8_cache[xn.uop]))
      np.testing.assert_allclose(h.numpy(), ref_h.numpy(), atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(xn.numpy(), ref, atol=1e-2, rtol=1e-2)
      for g, r in zip(q8, q8_quantize(xn.contiguous().realize(), 1, D)): np.testing.assert_equal(g.numpy(), r.numpy())

  def test_single(self): self._check(1)
  @unittest.skipUnless(has_p2p(2), "needs p2p")
  def test_p2p_2(self): self._check(2)
  @unittest.skipUnless(has_p2p(4), "needs p2p")
  def test_p2p_4(self): self._check(4)

if __name__ == '__main__': unittest.main()

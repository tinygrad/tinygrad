import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, Device
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, rms_norm_q8, gated_delta_decode, p2p_comm, q8_quantize, _q8_cache

def devices(n:int) -> tuple[str, ...]: return tuple(f"{Device.DEFAULT}:{i}" if i else Device.DEFAULT for i in range(n))

def p2p_supported(n:int) -> bool:
  try: return p2p_comm(devices(n)) is not None
  except Exception: return False

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "RDNA3/4 required")
class TestDecodeKernels(unittest.TestCase):
  def setUp(self): Tensor.manual_seed(0)

  def test_rms_norm_q8(self):
    D = 5120
    norm = nn.RMSNorm(D, 1e-6)
    norm.weight = Tensor.randn(D).half().realize()
    for T in (1, 3):
      x, a = Tensor.randn(1, T, D).realize(), Tensor.randn(1, T, D).realize()
      for dt in (dtypes.float32, dtypes.half):
        for add in (False, True):
          ref_h = x + a if add else x
          ref = norm(ref_h).cast(dt).contiguous().realize()
          h, xn = rms_norm_q8(norm, x, a if add else None, dtype=dt)
          q8 = list(_q8_cache[xn.uop])
          Tensor.realize(h, xn, *q8)
          np.testing.assert_allclose(h.numpy(), ref_h.numpy(), atol=1e-6)
          np.testing.assert_allclose(xn.float().numpy(), ref.float().numpy(), atol=1e-3, rtol=1e-3)
          for g, r in zip(q8, q8_quantize(xn.contiguous().realize(), T, D)): np.testing.assert_allclose(g.numpy(), r.numpy(), atol=1e-6)

  def _rms_norm_p2p(self, n:int):
    D, devs = 5120, devices(n)
    norm = nn.RMSNorm(D, 1e-6)
    w = Tensor.randn(D).half().realize()
    norm.weight = w.shard(devs).contiguous().realize()
    for _ in range(4):
      x, parts = Tensor.randn(1, 1, D).realize(), Tensor.randn(n, 1, D).realize()
      h, xn = rms_norm_q8(norm, x.shard(devs).contiguous().realize(), parts.shard(devs, 0).contiguous().realize())
      ref_h = x + parts.sum(0)
      norm.weight = w
      ref = norm(ref_h).numpy()
      norm.weight = w.shard(devs).contiguous().realize()
      # the partials travel with 15 mantissa bits, every device gets the same bits
      hs = [h.to(d).numpy() for d in devs]
      for r in hs[1:]: np.testing.assert_equal(r, hs[0])
      np.testing.assert_allclose(hs[0], ref_h.numpy(), atol=2e-4, rtol=1e-4)
      np.testing.assert_allclose(xn.to(devs[0]).numpy(), ref, atol=2e-3, rtol=1e-3)

  @unittest.skipUnless(p2p_supported(2), "needs 2 devices with p2p")
  def test_rms_norm_p2p_2(self): self._rms_norm_p2p(2)
  @unittest.skipUnless(p2p_supported(4), "needs 4 devices with p2p")
  def test_rms_norm_p2p_4(self): self._rms_norm_p2p(4)

  def _gated_delta(self, start:int):
    Kh, Vh, D = 8, 24, 128
    C = 2*Kh*D + Vh*D
    qkv, gate = Tensor.randn(1, 1, C).realize(), Tensor.randn(1, 1, Vh*D).realize()
    beta, alpha = Tensor.randn(1, 1, Vh).realize(), Tensor.randn(1, 1, Vh).realize()
    conv_state, state = Tensor.randn(1, 3, C).contiguous().realize(), (Tensor.randn(1, Vh, D, D)*0.1).contiguous().realize()
    conv_w, dt_bias, ssm_a = Tensor.randn(C, 4).half().realize(), Tensor.randn(Vh).half().realize(), (-Tensor.rand(Vh)).half().realize()
    norm = nn.RMSNorm(D, 1e-6)
    norm.weight = Tensor.randn(D).half().realize()
    # reference: the unfused math
    keep = 0 if start == 0 else 1
    win = Tensor.cat(conv_state*keep, qkv, dim=1)
    conv = sum(win[:, i:i+1] * conv_w[:, i] for i in range(4)).silu()
    q, k, v = conv.split([Kh*D, Kh*D, Vh*D], dim=-1)
    q = q.reshape(1, 1, 1, Kh, D).normalize(dim=-1, eps=1e-6).expand(1, 1, 3, Kh, D).reshape(Vh, D) * D**-0.5
    k = k.reshape(1, 1, 1, Kh, D).normalize(dim=-1, eps=1e-6).expand(1, 1, 3, Kh, D).reshape(Vh, D)
    b, a = beta.reshape(Vh).sigmoid(), ((alpha.reshape(Vh) + dt_bias).softplus() * ssm_a).exp()
    S = state.reshape(Vh, D, D) * keep * a.reshape(Vh, 1, 1)
    delta = (v.reshape(Vh, D) - (S * k.reshape(Vh, 1, D)).sum(-1)) * b.reshape(Vh, 1)
    S = S + delta.reshape(Vh, D, 1) * k.reshape(Vh, 1, D)
    zref = (norm((S * q.reshape(Vh, 1, D)).sum(-1)) * gate.reshape(Vh, D).silu()).cast(dtypes.half).reshape(1, 1, Vh*D)
    zref, Sref, csref = zref.numpy(), S.numpy(), win[:, 1:].numpy()
    sp = UOp.variable("start_pos", 0, 100).bind(start)
    z = gated_delta_decode(qkv, gate, beta, alpha, conv_state, state, conv_w, dt_bias, ssm_a, norm, Tensor(sp), Kh, 1e-6)
    zq = list(_q8_cache[z.uop])
    Tensor.realize(z, *zq)
    np.testing.assert_allclose(z.float().numpy(), zref.astype(np.float32), atol=2e-2, rtol=1e-2)
    np.testing.assert_allclose(state.numpy().reshape(Vh, D, D), Sref, atol=1e-5)
    np.testing.assert_allclose(conv_state.numpy(), csref, atol=1e-6)
    for g, r in zip(zq, q8_quantize(z.contiguous().realize(), 1, Vh*D)): np.testing.assert_allclose(g.numpy(), r.numpy(), atol=1e-6)

  def test_gated_delta_decode(self): self._gated_delta(5)
  def test_gated_delta_decode_reset(self): self._gated_delta(0)

if __name__ == '__main__': unittest.main()

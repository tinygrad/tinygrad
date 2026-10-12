import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, Device, TinyJit
from tinygrad.llm.kernels.amd import (amd_custom_kernels_supported, rms_norm_q8, swiglu_q8, gated_delta_decode, attn_decode_prep,
                                      amd_flash_attention_decode, amd_sample, p2p_comm, p2p_allreduce, q8_quantize, _q8_cache, _hash_uniform)
from tinygrad.llm.model import apply_rope, precompute_freqs_cis

def devices(n:int) -> tuple[str, ...]: return tuple(f"{Device.DEFAULT}:{i}" if i else Device.DEFAULT for i in range(n))

def multi_supported(n:int) -> bool:
  try: return p2p_comm(devices(n)) is not None
  except Exception: return False

def q8(t:Tensor) -> list[np.ndarray]: return [x.numpy() for x in _q8_cache[t.uop]]

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
          got_q8 = [t.realize() for t in _q8_cache[xn.uop]]
          np.testing.assert_allclose(h.numpy(), ref_h.numpy(), atol=1e-6)
          np.testing.assert_allclose(xn.float().numpy(), ref.float().numpy(), atol=1e-3, rtol=1e-3)
          for g, r in zip(got_q8, q8_quantize(xn.contiguous().realize(), T, D)): np.testing.assert_allclose(g.numpy(), r.numpy(), atol=1e-6)

  def test_swiglu_q8(self):
    gate, up = Tensor.randn(1, 1, 4352).realize(), Tensor.randn(1, 1, 4352).realize()
    act = swiglu_q8(gate, up)
    got = [t.realize() for t in _q8_cache[act.uop]]
    np.testing.assert_allclose(act.numpy(), (gate.silu()*up).numpy(), atol=1e-5, rtol=1e-5)
    for g, r in zip(got, q8_quantize(act.contiguous().realize(), 1, 4352)): np.testing.assert_allclose(g.numpy(), r.numpy(), atol=1e-6)

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

  def _attention(self, n:int):
    H, KVH, D, R, N, L = 24, 4, 256, 64, 4096, 10
    devs = devices(n)
    q, k, v = Tensor.randn(1, 1, H*2*D).realize(), Tensor.randn(1, 1, KVH*D).realize(), Tensor.randn(1, 1, KVH*D).realize()
    cache = (Tensor.randn(2, 1, KVH, N, D)*0.5).half().contiguous().realize()
    qn, kn = nn.RMSNorm(D, 1e-6), nn.RMSNorm(D, 1e-6)
    qw, kw = Tensor.randn(D).half().realize(), Tensor.randn(D).half().realize()
    freqs = precompute_freqs_cis(R, N, 1e7).realize()
    sp = UOp.variable("start_pos", 0, N-1).bind(L)
    # reference
    qg = q.reshape(1, 1, H, 2, D)
    qq, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(1, 1, H*D)
    qn.weight, kn.weight = qw, kw
    qq, kk = qn(qq.reshape(1, 1, H, D).transpose(1, 2)), kn(k.reshape(1, 1, KVH, D).transpose(1, 2))
    qq = apply_rope(qq[..., :R], freqs[L:L+1]).cat(qq[..., R:], dim=-1)
    kk = apply_rope(kk[..., :R], freqs[L:L+1]).cat(kk[..., R:], dim=-1)
    cn = cache.numpy().copy()
    cn[0, :, :, L], cn[1, :, :, L] = kk.half().numpy()[:, :, 0], v.reshape(1, KVH, D).half().numpy()
    K, V = Tensor(cn[0, :, :, :L+1]).float(), Tensor(cn[1, :, :, :L+1]).float()
    ref = (qq.half().float().scaled_dot_product_attention(K, V, enable_gqa=True).transpose(1, 2).reshape(1, 1, H*D) * gate.sigmoid()).numpy()
    if n > 1:
      q, k, v, cache = (t.shard(devs, ax).contiguous().realize() for t,ax in ((q, 2), (k, 2), (v, 2), (cache, 2)))
      qn.weight, kn.weight, freqs = (t.shard(devs).contiguous().realize() for t in (qw, kw, freqs))
    qr, assigned = attn_decode_prep(q, k, v, qn, kn, freqs, cache, sp, H, True)
    out = amd_flash_attention_decode(qr, assigned, sp+1, N, gate=q)
    np.testing.assert_allclose(out.to(Device.DEFAULT).numpy(), ref, atol=2e-3)
    got = assigned.to(Device.DEFAULT).numpy()
    np.testing.assert_allclose(got[:, :, :, L].astype(np.float32), cn[:, :, :, L].astype(np.float32), atol=2e-3)

  def test_attention_decode(self): self._attention(1)
  @unittest.skipUnless(multi_supported(4), "needs 4 devices with p2p")
  def test_attention_decode_one_kv_head_per_device(self): self._attention(4)

  def test_hash_uniform_open_interval(self):
    x = Tensor([0, 0xffffffff, 0x80000000, 1], dtype=dtypes.uint32)
    u = Tensor(_hash_uniform(x.uop)).numpy()
    self.assertTrue(((u > 0) & (u < 1)).all())

  def _sample(self, n:int):
    devs, V = devices(n), 248320
    temp = Tensor([0.0]).realize()
    dev = devs if n > 1 else devs[0]
    @TinyJit
    def f(l): return amd_sample(l, temp, dev).contiguous()
    for _ in range(20):
      lg = Tensor.randn(1, V).realize()
      tok = f(lg.shard(devs, 1).contiguous().realize() if n > 1 else lg)
      for d in devs: self.assertEqual(int(tok.to(d).numpy()[0, 0]), int(lg.numpy().argmax()))

  def test_sample_greedy(self): self._sample(1)
  @unittest.skipUnless(multi_supported(2), "needs 2 devices with p2p")
  def test_sample_greedy_sharded(self): self._sample(2)

  def test_sample_distribution(self):
    lg = Tensor([[0.0, 1.0, 2.0, 0.5]]).realize()
    counts = np.zeros(4)
    for _ in range(400): counts[amd_sample(lg, Tensor([1.0]), Device.DEFAULT).item()] += 1
    np.testing.assert_allclose(counts/400, np.exp([0, 1, 2, 0.5])/np.exp([0, 1, 2, 0.5]).sum(), atol=0.06)

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT) and multi_supported(2), "needs 2 RDNA3/4 devices with p2p")
class TestP2PAllreduce(unittest.TestCase):
  def _test(self, n:int, exact:bool, D:int=5120):
    devs = devices(n)
    comm = p2p_comm(devs)
    assert comm is not None
    @TinyJit
    def f(x): return p2p_allreduce(comm, x, (1, D), exact=exact).contiguous()
    for _ in range(10):
      x = Tensor.randn(n, D).realize()
      out = f(x.shard(devs, 0).contiguous().realize())
      res = [out.to(d).numpy()[0] for d in devs]
      for r in res[1:]: np.testing.assert_equal(r, res[0])  # every device gets the same bits
      np.testing.assert_allclose(res[0], x.numpy().sum(0), atol=0 if exact and n == 2 else 1e-5 if exact else 2e-4)

  def test_2(self): self._test(2, False)
  def test_2_exact(self): self._test(2, True, D=128)
  @unittest.skipUnless(multi_supported(4), "needs 4 devices with p2p")
  def test_4(self): self._test(4, False)
  @unittest.skipUnless(multi_supported(4), "needs 4 devices with p2p")
  def test_4_exact(self): self._test(4, True, D=128)
  @unittest.skipUnless(multi_supported(3), "needs 3 devices with p2p")
  def test_3(self): self._test(3, False)

if __name__ == '__main__': unittest.main()

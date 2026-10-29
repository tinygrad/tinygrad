#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_res_mix.py: amd.res_mix_norm (residual stream update + mix + rmsnorm + q8 in one kernel) against the tensor path
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.amd import res_mix_norm, _q8_quantize, _q8_cache

def ref(xs:Tensor, wm:Tensor, wn:Tensor, eps:float, norm_eps:float) -> Tensor:  # model.py AttentionResidual.mix then nn.RMSNorm
  scores = (xs.rmsnorm(eps=eps) * wm).sum(-1, keepdim=True)
  h = (scores.softmax(2) * xs).sum(2)
  return h.float().rmsnorm(eps=norm_eps) * wn

rng = np.random.default_rng(0)
def rand(*shape): return Tensor(rng.standard_normal(shape).astype(np.float32) * rng.uniform(0.5, 4, (*shape[:-1], 1)).astype(np.float32))
for T, streams, new_block in [(1, *c) for c in ((2, None), (3, None), (1, True), (2, True), (2, False), (8, True), (9, False))] + \
                              [(2, *c) for c in ((3, None), (2, True), (9, False))]:  # (T: a verify step's tokens)
  xs, out = rand(1, T, streams, 7168), rand(1, T, 1, 7168)
  wm, wn = Tensor(rng.standard_normal(7168).astype(np.float32) * 0.05), Tensor(rng.uniform(0.5, 1.5, 7168).astype(np.float32))
  y, xs2 = res_mix_norm(xs, wm, wn, 1e-6, 1e-5, out=None if new_block is None else out, new_block=bool(new_block))
  q, scale, xsum = _q8_cache[y.uop]
  Tensor.realize(y, xs2, q, scale, xsum)
  want_xs = xs if new_block is None else xs.cat(out, dim=2) if new_block else xs[:, :, :-1].cat(xs[:, :, -1:] + out, dim=2)
  np.testing.assert_allclose(xs2.numpy(), want_xs.numpy(), rtol=1e-6)
  want = ref(want_xs, wm, wn, 1e-6, 1e-5).numpy()
  np.testing.assert_allclose(y.numpy(), want, rtol=1e-4, atol=1e-4)
  q2, scale2, xsum2 = _q8_quantize(Tensor(y.numpy()), tokens=T, in_features=7168)
  np.testing.assert_equal(q.numpy(), q2.numpy())
  np.testing.assert_allclose(scale.numpy(), scale2.numpy(), rtol=1e-6)
  np.testing.assert_allclose(xsum.numpy(), xsum2.numpy())
  print(f"{T} tokens, {streams} streams, update {new_block}: ok, max |y - ref| {np.abs(y.numpy() - want).max():.2e}")

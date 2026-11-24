import unittest

import numpy as np
from tinygrad import Device, Tensor
from tinygrad.renderer.cstyle import HIPRenderer

import slopkernels
from slopkernels.bench import _desc, _run_op


def _amd_supported(dev:str|tuple[str, ...]|None) -> bool:
  d = dev[0] if isinstance(dev, tuple) else dev
  if d is None or d.split(":")[0] != "AMD": return False
  t = getattr(Device[d], "target", None)
  return t is not None and t[0] in (9, 11, 12) and isinstance(Device[d].renderer, HIPRenderer)


def _gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> Tensor:
  target = tuple(int(x) for x in Device[q.device].target)
  srcs = (q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state)
  uir = slopkernels.get("gated_delta_prefill_op", target=target, device=q.device, srcs=tuple(_desc(s) for s in srcs))
  assert uir is not None
  return _run_op(uir, *srcs)


class TestGatedDeltaNetBlock(unittest.TestCase):
  def test_gated_delta_rectangular_state_and_column_decay(self):
    if not _amd_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(size=(1, 1, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta = rng.normal(size=(1, 1, 3, 4)).astype(np.float32), rng.uniform(size=(1, 1, 3)).astype(np.float32)
    alpha, initial = rng.uniform(0.8, 1, size=(1, 1, 3, 32)).astype(np.float32), rng.normal(size=(1, 1, 4, 32)).astype(np.float32)
    expected_state, expected_out = initial.copy(), np.empty_like(v)
    for t in range(3):
      previous, av = expected_state.copy(), alpha[:, :, t, None, :]
      delta = (v[:, :, t] - (previous * av * k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      expected_state = previous * av + delta[..., None] * k[:, :, t, None, :]
      expected_out[:, :, t] = (previous * av * q[:, :, t, None, :]).sum(-1) + delta * (q[:, :, t] * k[:, :, t]).sum(-1)
    state = Tensor(initial).contiguous().realize()
    out = _gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()

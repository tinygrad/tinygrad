import unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.kernels.amd import gated_delta_prefill, amd_custom_kernels_supported

class TestGatedDeltaNetBlock(unittest.TestCase):
  def test_gated_delta_rectangular_state_and_column_decay(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(size=(1, 1, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta = rng.normal(size=(1, 1, 3, 4)).astype(np.float32), rng.uniform(size=(1, 1, 3)).astype(np.float32)
    alpha, initial = rng.uniform(0.8, 1, size=(1, 1, 3, 32)).astype(np.float32), rng.normal(size=(1, 1, 4, 32)).astype(np.float32)
    expected_state, expected_out = initial.copy(), np.empty_like(v)
    for t in range(3):
      previous, av = expected_state.copy(), alpha[:, :, t, None, :]
      delta = (v[:, :, t] - (previous*av*k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      expected_state = previous*av + delta[..., None]*k[:, :, t, None, :]
      expected_out[:, :, t] = (previous*av*q[:, :, t, None, :]).sum(-1) + delta*(q[:, :, t]*k[:, :, t]).sum(-1)
    state = Tensor(initial).contiguous().realize()
    out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def test_gated_delta_state_slots(self):
    # with src: start from state slot src, the state after token t goes to slot t, the other slots are untouched
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(0)
    q, k = (rng.normal(size=(1, 2, 3, 32)).astype(np.float32)*0.3 for _ in range(2))
    v, beta = rng.normal(size=(1, 2, 3, 8)).astype(np.float32), rng.uniform(0.1, 0.9, size=(1, 2, 3)).astype(np.float32)
    alpha, slots = rng.uniform(0.5, 1, size=(1, 2, 3, 1)).astype(np.float32), rng.normal(size=(4, 1, 2, 8, 32)).astype(np.float32)
    expected_state, expected_out, expected_slots = slots[1].copy(), np.empty_like(v), slots.copy()
    for t in range(3):
      decayed = expected_state * alpha[:, :, t, None, :]
      delta = (v[:, :, t] - (decayed*k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      expected_slots[t] = expected_state = decayed + delta[..., None]*k[:, :, t, None, :]
      expected_out[:, :, t] = (expected_state*q[:, :, t, None, :]).sum(-1)
    state, src = Tensor(slots).contiguous().realize(), UOp.variable("spec_j", 0, 3).bind(1)
    out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state, src=Tensor(src)).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_slots, rtol=1e-4, atol=1e-4)

if __name__ == "__main__": unittest.main()

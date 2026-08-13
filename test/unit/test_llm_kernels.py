import unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.model import gated_delta_prefill

def numpy_ref(q, k, v, beta, alpha, initial):
  state, out = initial.copy(), np.empty_like(v)
  for t in range(q.shape[2]):
    av = alpha[:, :, t, :, None] if alpha.ndim == 4 else alpha[:, :, t, None, None]
    sa = alpha[:, :, t] if alpha.ndim == 4 else alpha[:, :, t, None]
    previous = state.copy()
    delta = (v[:, :, t] - (previous*k[:, :, t, None]).sum(-1)*sa) * beta[:, :, t, None]
    state = previous*av + delta[..., None]*k[:, :, t, None, :]
    out[:, :, t] = (previous*q[:, :, t, None]).sum(-1)*sa + delta*(q[:, :, t]*k[:, :, t]).sum(-1, keepdims=True)
  return out, state

class TestGatedDeltaPrefill(unittest.TestCase):
  def _make(self, B, H, T, V, K, alpha_4d=False, seed=42):
    rng = np.random.default_rng(seed)
    # normalize like the model does: with raw unit-norm keys the delta rule is stable, random keys make it diverge
    q, k = (rng.normal(size=(B, H, T, K)).astype(np.float32) for _ in range(2))
    k = k / np.maximum(np.sqrt((k*k).sum(-1, keepdims=True)), 1e-6)
    v, beta = rng.normal(size=(B, H, T, V)).astype(np.float32), rng.uniform(size=(B, H, T)).astype(np.float32)
    alpha = rng.uniform(0.9, 1, size=(B, H, T, V) if alpha_4d else (B, H, T)).astype(np.float32)
    initial = rng.normal(size=(B, H, V, K)).astype(np.float32)
    return q, k, v, beta, alpha, initial

  def test_rectangular_state_and_row_decay(self):
    q, k, v, beta, alpha, initial = self._make(1, 1, 3, 4, 32, alpha_4d=True)
    expected_out, expected_state = numpy_ref(q, k, v, beta, alpha, initial)
    state = Tensor(initial).contiguous().realize()
    out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def test_prefill_matches_single_steps(self):
    # one T=32 kernel call must match 32 sequential T=1 calls with in-place state
    q, k, v, beta, alpha, initial = self._make(1, 4, 32, 128, 128)
    state_a = Tensor(initial).contiguous().realize()
    out_a = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state_a).realize()
    outs, state_b = [], Tensor(initial).contiguous().realize()
    for t in range(32):
      outs.append(gated_delta_prefill(Tensor(q[:, :, t:t+1]), Tensor(k[:, :, t:t+1]), Tensor(v[:, :, t:t+1]),
                                      Tensor(beta[:, :, t:t+1]), Tensor(alpha[:, :, t:t+1]), state_b).realize())
    np.testing.assert_allclose(out_a.numpy(), Tensor.stack(*outs, dim=2).squeeze(3).numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state_a.numpy(), state_b.numpy(), rtol=1e-4, atol=1e-4)

  def test_start_pos_zero_resets_state(self):
    q, k, v, beta, alpha, initial = self._make(1, 2, 5, 8, 16)
    # garbage state must be ignored when start_pos binds to 0
    garbage = np.full_like(initial, 1.0e9)
    def run(sp, init):
      state = Tensor(init).contiguous().realize()
      start_pos = Tensor(UOp.variable("start_pos", 0, 63).bind(sp))
      return gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state, start_pos).realize(), state
    out_reset, state_reset = run(0, garbage)
    expected_out, expected_state = numpy_ref(q, k, v, beta, alpha, np.zeros_like(initial))
    np.testing.assert_allclose(out_reset.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state_reset.numpy(), expected_state, rtol=1e-4, atol=1e-4)
    # nonzero start_pos must use the provided state
    out_cont, state_cont = run(3, initial)
    expected_out, expected_state = numpy_ref(q, k, v, beta, alpha, initial)
    np.testing.assert_allclose(out_cont.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state_cont.numpy(), expected_state, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  unittest.main()

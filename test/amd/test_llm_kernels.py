import unittest
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.llm.amd_kernels import fused_gate_up, gdn_recurrent_update_conv, q8_lmhead_gumbel_argmax

DIM, HIDDEN, VOCAB, Q8_BLOCK, Q8_BLOCK_BYTES = 1024, 3584, 248320, 32, 34
GDN_HV, GDN_V, GDN_K = 16, 128, 128

def make_q8_blocks(rng:np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  blocks = DIM // Q8_BLOCK
  scales = rng.uniform(2.0e-4, 1.0e-3, size=(HIDDEN, blocks)).astype(np.float16)
  qs = rng.integers(-127, 128, size=(HIDDEN, blocks, Q8_BLOCK), dtype=np.int8)
  raw = np.empty((HIDDEN, blocks, Q8_BLOCK_BYTES), dtype=np.uint8)
  raw[:, :, :2] = scales.view(np.uint8).reshape(HIDDEN, blocks, 2)
  raw[:, :, 2:] = qs.view(np.uint8)
  return raw.reshape(-1), scales.astype(np.float32), qs.astype(np.float32)

def q8_matvec(scales:np.ndarray, qs:np.ndarray, x:np.ndarray) -> np.ndarray:
  xb = x.reshape(DIM // Q8_BLOCK, Q8_BLOCK)
  return (qs * scales[:, :, None] * xb[None, :, :]).sum(axis=(1, 2), dtype=np.float32)

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD only")
class TestLLMAMDKernels(unittest.TestCase):
  def test_fused_gate_up_q8_correctness(self):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(DIM,)).astype(np.float32)
    gate_raw, gate_scales, gate_qs = make_q8_blocks(rng)
    up_raw, up_scales, up_qs = make_q8_blocks(rng)

    gate = q8_matvec(gate_scales, gate_qs, x)
    up = q8_matvec(up_scales, up_qs, x)
    expected = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32)

    out = fused_gate_up(Tensor(x, device="AMD"), Tensor(gate_raw, device="AMD").contiguous(),
                        Tensor(up_raw, device="AMD").contiguous()).numpy()
    np.testing.assert_allclose(out, expected, rtol=2e-4, atol=2e-4)

  def test_q8_lmhead_gumbel_argmax_correctness(self):
    blocks = DIM // Q8_BLOCK
    target = 12345
    raw = np.zeros((VOCAB, blocks, Q8_BLOCK_BYTES), dtype=np.uint8)
    raw[target, :, :2] = np.array([1], dtype=np.float16).view(np.uint8)
    raw[target, :, 2:] = 1

    hidden = np.ones((DIM,), dtype=np.float32)
    out = q8_lmhead_gumbel_argmax(Tensor(hidden, device="AMD"), Tensor(raw.reshape(-1), device="AMD").contiguous(),
                                  Tensor([0.0], device="AMD")).numpy()
    self.assertEqual(int(out.reshape(-1)[0]), target)

  def test_gdn_recurrent_update_conv_correctness(self):
    rng = np.random.default_rng(1)
    state = rng.uniform(-0.1, 0.1, size=(GDN_HV, GDN_V, GDN_K)).astype(np.float32)
    conv_out = rng.uniform(-0.5, 0.5, size=(3 * GDN_HV * GDN_K,)).astype(np.float32)
    alpha = rng.uniform(0.8, 1.0, size=(GDN_HV,)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=(GDN_HV,)).astype(np.float16)

    expected_state = state.copy()
    expected_core = np.empty((GDN_HV, GDN_V), dtype=np.float32)
    for hv in range(GDN_HV):
      q = conv_out[hv * GDN_K:(hv + 1) * GDN_K]
      k = conv_out[GDN_HV * GDN_K + hv * GDN_K:GDN_HV * GDN_K + (hv + 1) * GDN_K]
      q = q / np.sqrt((q * q).sum()) * (GDN_K ** -0.5)
      k = k / np.sqrt((k * k).sum())
      for row in range(GDN_V):
        v = conv_out[2 * GDN_HV * GDN_K + hv * GDN_V + row]
        h = expected_state[hv, row] * alpha[hv]
        updated = h + k * ((v - (h * k).sum()) * float(beta[hv]))
        expected_state[hv, row] = updated
        expected_core[hv, row] = (updated * q).sum()

    state_t = Tensor(state, device="AMD").contiguous()
    out = gdn_recurrent_update_conv(state_t, Tensor(conv_out, device="AMD"), Tensor(alpha, device="AMD"),
                                    Tensor(beta, device="AMD")).numpy().reshape(GDN_HV, GDN_V)
    np.testing.assert_allclose(out, expected_core, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(state_t.numpy(), expected_state, rtol=2e-4, atol=2e-4)

if __name__ == "__main__":
  unittest.main()

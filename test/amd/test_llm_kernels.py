import unittest
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.llm.amd_kernels import fused_gate_up

DIM, HIDDEN, Q8_BLOCK, Q8_BLOCK_BYTES = 1024, 3584, 32, 34

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

if __name__ == "__main__":
  unittest.main()

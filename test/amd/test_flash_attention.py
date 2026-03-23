#!/usr/bin/env python3
"""Test amd_flash_attention custom kernel on real hardware and in the emulator."""
import unittest, subprocess, os, sys
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context, GlobalCounters

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestFlashAttentionHW(unittest.TestCase):
  """Flash attention on real hardware — correctness check against reference."""

  def test_flash_attention_correctness(self):
    from extra.gemm.amd_flash_attention import amd_flash_attention
    B, H, N, D = 1, 1, 64, 64
    q = Tensor.rand(B, H, N, D).cast(dtypes.half)
    k = Tensor.rand(B, H, N, D).cast(dtypes.half)
    v = Tensor.rand(B, H, N, D).cast(dtypes.half)
    o = Tensor.empty(B, H, N, D, dtype=dtypes.float)
    with Context(DEBUG=0): Tensor.realize(q, k, v)
    q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, N, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, N, D)
    GlobalCounters.reset()
    tst = Tensor.custom_kernel(o_flat, q_flat, k_flat, v_flat, fxn=amd_flash_attention)[0].realize()
    with Context(DEBUG=0):
      ref = q.float().scaled_dot_product_attention(k.float(), v.float()).reshape(B*H, N, D).realize()
      err = (ref - tst).square().mean().item()
    self.assertLess(err, 1e-2, f"flash attention mse {err} too high")

class TestFlashAttentionEmulator(unittest.TestCase):
  """Flash attention custom kernel in the Python RDNA3 emulator."""

  def test_flash_attention_emulator(self):
    """Run flash attention kernel through the emulator with correctness check (relaxed tolerance for emulator precision)."""
    test_code = '''
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, GlobalCounters
from extra.gemm.amd_flash_attention import amd_flash_attention

B, H, N, D = 1, 1, 64, 64
q = Tensor.rand(B, H, N, D).cast(dtypes.half)
k = Tensor.rand(B, H, N, D).cast(dtypes.half)
v = Tensor.rand(B, H, N, D).cast(dtypes.half)
o = Tensor.empty(B, H, N, D, dtype=dtypes.float)
with Context(DEBUG=0): Tensor.realize(q, k, v)
q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, N, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, N, D)
GlobalCounters.reset()
tst = Tensor.custom_kernel(o_flat, q_flat, k_flat, v_flat, fxn=amd_flash_attention)[0].realize()
with Context(DEBUG=0):
  ref = q.float().scaled_dot_product_attention(k.float(), v.float()).reshape(B*H, N, D).realize()
  err = (ref - tst).square().mean().item()
# Emulator has higher precision drift from f16/exp2 approximations vs real hardware
assert err < 0.1, f"flash attention mse {err} too high"
print("PASS")
'''
    env = os.environ.copy()
    env["AMD"] = "1"
    env["MOCKGPU"] = "1"
    env["PYTHON_REMU"] = "1"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run([sys.executable, "-c", test_code], env=env, capture_output=True, text=True, timeout=180)
    self.assertEqual(result.returncode, 0, f"emulator failed (rc={result.returncode}): {result.stderr[-500:]}")
    self.assertIn("PASS", result.stdout)

if __name__ == "__main__":
  unittest.main()

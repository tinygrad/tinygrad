import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.quant import MXFP4_VALUES, dequantize_mxfp4, quantize_dequantize_mxfp8, quantize_mxfp4, quantize_mxfp4_cpu

class TestMXFormats(unittest.TestCase):
  def test_mxfp4_known_codes_and_scale(self):
    values = np.array(MXFP4_VALUES * 2, dtype=np.float32)
    packed, scale = quantize_mxfp4(Tensor(values))
    # Positive and negative zero are numerically identical, so nearest-value encoding canonicalizes to +0.
    np.testing.assert_array_equal(packed.numpy(), np.array([0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe] * 2, dtype=np.uint8))
    np.testing.assert_array_equal(scale.numpy(), np.array([127], dtype=np.uint8))
    np.testing.assert_array_equal(dequantize_mxfp4(packed, scale, dtypes.float32).numpy(), values)

  def test_mxfp4_block_scales_and_zero(self):
    x = Tensor(np.array([0.0]*32 + [12.0, -12.0] + [0.0]*30, dtype=np.float32))
    packed, scale = quantize_mxfp4(x)
    np.testing.assert_array_equal(scale.numpy(), np.array([127, 128], dtype=np.uint8))
    np.testing.assert_allclose(dequantize_mxfp4(packed, scale, dtypes.float32).numpy(), x.numpy())

  def test_mxfp4_scale_rounds_amax_over_format_max(self):
    # OCP E8M0 scale selection rounds log2(amax / 6), rather than flooring the
    # input exponent. At this boundary the two rules differ by a factor of two.
    x = Tensor(np.array([8.0] + [0.0]*31, dtype=np.float32))
    packed, scale = quantize_mxfp4(x)
    np.testing.assert_array_equal(scale.numpy(), np.array([127], dtype=np.uint8))
    self.assertEqual(dequantize_mxfp4(packed, scale, dtypes.float32).numpy()[0], 6.0)

  def test_mxfp4_cpu_converter_matches_tensor_path(self):
    x = Tensor(np.linspace(-13, 13, 64*32, dtype=np.float32).reshape(64, 32))
    packed, scale = quantize_mxfp4(x)
    cpu_packed, cpu_scale = quantize_mxfp4_cpu(x)
    np.testing.assert_array_equal(cpu_packed.numpy(), packed.numpy())
    np.testing.assert_array_equal(cpu_scale.numpy(), scale.numpy())

  def test_mxfp4_midpoints_round_to_even(self):
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    x = Tensor(np.pad(np.concatenate((midpoints, -midpoints)), (0, 18)))
    packed, scale = quantize_mxfp4(x)
    expected = np.pad(np.array([0, 1, 1, 2, 2, 4, 4, 0, -1, -1, -2, -2, -4, -4], dtype=np.float32), (0, 18))
    np.testing.assert_array_equal(dequantize_mxfp4(packed, scale, dtypes.float32).numpy(), expected)

  def test_mxfp8_roundtrip_and_dtype(self):
    # All E4M3-exact values remain exact after extracting a shared exponent.
    x = Tensor(np.array(([0.0, 0.5, 1.0, 1.5, 2.0, -3.0, 4.0, -6.0] * 4), dtype=np.float32))
    out = quantize_dequantize_mxfp8(x)
    self.assertEqual(out.dtype, dtypes.bfloat16)
    np.testing.assert_array_equal(out.float().numpy(), x.numpy())

  def test_mxfp8_subnormal_and_rounding(self):
    x = np.zeros(32, dtype=np.float32)
    x[:5] = [1.0, 1.0625, 1.07, 2**-9, 2**-10]
    out = quantize_dequantize_mxfp8(Tensor(x), dtype=dtypes.float32).numpy()
    # amax / 448 rounds to an E8M0 scale of 2**-9, saturating the largest
    # values while retaining the E4M3 subnormal quantum for this block.
    np.testing.assert_array_equal(out[:5], [0.875, 0.875, 0.875, 2**-9, 2**-10])

  def test_mxfp8_uses_full_e4m3_range(self):
    x = np.zeros(32, dtype=np.float32)
    x[:4] = [448.0, 416.0, 400.0, -448.0]
    np.testing.assert_array_equal(quantize_dequantize_mxfp8(Tensor(x), dtype=dtypes.float32).numpy()[:4], [448.0, 416.0, 384.0, -448.0])

  def test_mxfp8_scale_rounds_amax_over_format_max(self):
    x = np.zeros(32, dtype=np.float32)
    x[0] = 512.0
    self.assertEqual(quantize_dequantize_mxfp8(Tensor(x), dtype=dtypes.float32).numpy()[0], 448.0)

if __name__ == "__main__": unittest.main()

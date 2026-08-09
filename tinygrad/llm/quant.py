from tinygrad import Tensor, dtypes

MX_BLOCK_SIZE = 32
MXFP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

def _e8m0_scale(scale:Tensor) -> Tensor:
  """Decode an OCP E8M0 scale byte. 127 encodes 2**0."""
  return (scale.cast(dtypes.float32) - 127.0).exp2()

def quantize_mxfp4(x:Tensor) -> tuple[Tensor, Tensor]:
  """Quantize the last dimension to OCP MXFP4 (E2M1 values, E8M0 scale, block size 32)."""
  if x.shape[-1] % MX_BLOCK_SIZE: raise ValueError(f"MXFP4 requires a multiple-of-32 last dimension, got {x.shape}")
  *outer, k = x.shape
  blocks = x.float().reshape(*outer, k//MX_BLOCK_SIZE, MX_BLOCK_SIZE)
  amax = blocks.abs().max(axis=-1)
  # Match the OCP/MLX reference: quantize amax / E2M1_MAX to the nearest E8M0
  # power of two. This deliberately differs from extracting exponent bits: blocks
  # whose maximum is near a power-of-two boundary can select a scale 2x smaller.
  exponent = (amax.maximum(1e-38).div(6.0).log2().round()).clamp(-127, 127)
  scale = (amax == 0).where(127, exponent + 127).cast(dtypes.uint8)
  normalized = blocks / _e8m0_scale(scale).unsqueeze(-1)
  # Midpoint bins avoid materializing a 16x larger distance tensor while preserving nearest-value encoding.
  magnitude = normalized.abs()
  # OCP formats use round-to-nearest-even: at alternating midpoints, the upper code has an even mantissa LSB.
  code = sum(((magnitude >= midpoint) if upper_even else (magnitude > midpoint)).cast(dtypes.uint8)
             for midpoint, upper_even in ((0.25, False), (0.75, True), (1.25, False), (1.75, True),
                                           (2.5, False), (3.5, True), (5.0, False)))
  code = ((normalized < 0) & (code != 0)).where(code + 8, code).reshape(*outer, k)
  # Safetensors has no nibble dtype. Store the earlier element in the low nibble.
  packed = code[..., ::2] + code[..., 1::2] * 16
  return packed.contiguous(), scale.contiguous()

def quantize_mxfp4_cpu(x:Tensor) -> tuple[Tensor, Tensor]:
  """CPU converter fast path for large checkpoints. Inference itself does not depend on numpy."""
  import numpy as np
  if x.shape[-1] % MX_BLOCK_SIZE: raise ValueError(f"MXFP4 requires a multiple-of-32 last dimension, got {x.shape}")
  array = x.float().numpy()
  blocks = array.reshape(*array.shape[:-1], array.shape[-1]//MX_BLOCK_SIZE, MX_BLOCK_SIZE)
  amax = np.max(np.abs(blocks), axis=-1)
  exponent = np.clip(np.rint(np.log2(np.maximum(amax, 1e-38) / 6.0)), -127, 127)
  scale = np.where(amax == 0, 127, exponent + 127).astype(np.uint8)
  normalized = blocks / np.exp2(scale.astype(np.float32) - 127)[..., None]
  magnitude = np.abs(normalized)
  code = sum(((magnitude >= midpoint) if upper_even else (magnitude > midpoint)).astype(np.uint8)
             for midpoint, upper_even in ((0.25, False), (0.75, True), (1.25, False), (1.75, True),
                                           (2.5, False), (3.5, True), (5.0, False)))
  code = np.where((normalized < 0) & (code != 0), code + 8, code).astype(np.uint8).reshape(array.shape)
  packed = code[..., ::2] + code[..., 1::2] * 16
  return Tensor(packed), Tensor(scale)

def dequantize_mxfp4(packed:Tensor, scale:Tensor, dtype=dtypes.bfloat16) -> Tensor:
  """Decode the packed representation emitted by quantize_mxfp4."""
  if packed.shape[-1] != scale.shape[-1] * 16:
    raise ValueError(f"incompatible MXFP4 values/scales: {packed.shape} and {scale.shape}")
  lo = packed - packed.div(16, rounding_mode="trunc") * 16
  hi = packed.div(16, rounding_mode="trunc")
  code = Tensor.stack(lo, hi, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1]*2)
  values = Tensor(MXFP4_VALUES, dtype=dtypes.float32, device=packed.device)[code]
  scales = _e8m0_scale(scale).unsqueeze(-1).expand(*scale.shape, MX_BLOCK_SIZE).reshape(*scale.shape[:-1], scale.shape[-1]*MX_BLOCK_SIZE)
  return (values * scales).cast(dtype)

def quantize_dequantize_mxfp8(x:Tensor, dtype=dtypes.bfloat16) -> Tensor:
  """Apply the Kimi expert-activation MXFP8 E4M3/E8M0 round trip in 32-value blocks."""
  if x.shape[-1] % MX_BLOCK_SIZE: raise ValueError(f"MXFP8 requires a multiple-of-32 last dimension, got {x.shape}")
  *outer, k = x.shape
  blocks = x.float().reshape(*outer, k//MX_BLOCK_SIZE, MX_BLOCK_SIZE)
  amax = blocks.abs().max(axis=-1)
  # As for MXFP4, the E8M0 scale is nearest-power-of-two(amax / E4M3_MAX).
  exponent = (amax.maximum(1e-38).div(448.0).log2().round()).clamp(-127, 127)
  scale = (amax == 0).where(127, exponent + 127).cast(dtypes.uint8)
  normalized = blocks / _e8m0_scale(scale).unsqueeze(-1)
  # Software OCP E4M3 rounding is required on gfx1100 (RDNA3 has no native FP8 dtype).
  # E4M3 has three explicit mantissa bits and a minimum normal exponent of -6;
  # using e=-6 also gives the 2**-9 subnormal quantum.
  magnitude = normalized.abs().clamp(max_=448.0)
  elem_exp = magnitude.maximum(2**-9).log2().floor().clamp(-6, 8)
  quantum = (elem_exp - 3).exp2()
  quantized = (magnitude / quantum).round() * quantum
  quantized = (normalized < 0).where(-quantized, quantized).clamp(-448.0, 448.0)
  return (quantized * _e8m0_scale(scale).unsqueeze(-1)).reshape(*outer, k).cast(dtype)

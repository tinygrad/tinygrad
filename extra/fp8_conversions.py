# based on https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp
import struct
import math

def float_to_fp8(x: float, fp8_interpretation: str) -> int:
  assert fp8_interpretation in ["E4M3", "E5M2"], "Invalid fp8 interpretation"
  if x is math.nan:
    return math.nan
  xbits, = struct.unpack('Q', struct.pack('d', x))

  if fp8_interpretation == "E4M3":
    FP8_EXP_BIAS = 7
    FP8_SIGNIFICAND_BITS = 4
    FP8_MANTISSA_MASK = 0x7
    FP8_MINDENORM_O2 = 0x3F50000000000000
    FP8_OVERFLOW_THRESHOLD = 0x407D000000000000
    FP8_MAXNORM = 0x7E
    FP8_MINNORM = 0x3F90000000000000
  else:  # E5M2
    FP8_EXP_BIAS = 15
    FP8_SIGNIFICAND_BITS = 3
    FP8_MANTISSA_MASK = 0x3
    FP8_MINDENORM_O2 = 0x3EE0000000000000
    FP8_OVERFLOW_THRESHOLD = 0x40EE000000000000 - 1
    FP8_MAXNORM = 0x7B
    FP8_MINNORM = 0x3F10000000000000

  DP_INF_BITS = 0x7FF0000000000000
  FP8_DP_HALF_ULP = 1 << (53 - FP8_SIGNIFICAND_BITS - 1)

  sign = ((xbits >> 63) & 1) << 7
  exp = (((xbits >> 52) & 0x7FF) - 1023 + FP8_EXP_BIAS) & 0xFF
  mantissa = (xbits >> (53 - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK
  absx = xbits & 0x7FFFFFFFFFFFFFFF

  if absx <= FP8_MINDENORM_O2:
    res = 0
  elif absx > DP_INF_BITS:
    if fp8_interpretation == "E4M3":
      res = 0x7F
    else:
      res = 0x7E | mantissa
  elif absx > FP8_OVERFLOW_THRESHOLD:
    res = FP8_MAXNORM
  elif absx >= FP8_MINNORM:
    res = ((exp << (FP8_SIGNIFICAND_BITS - 1)) | mantissa) & 0xFF
    round_bits = xbits & ((FP8_DP_HALF_ULP << 1) - 1)
    if (round_bits > FP8_DP_HALF_ULP) or (round_bits == FP8_DP_HALF_ULP and (mantissa & 1)):
      res = (res + 1) & 0xFF
  else:
    shift = 1 - exp
    mantissa |= 1 << (FP8_SIGNIFICAND_BITS - 1)
    res = (mantissa >> shift) & 0xFF
    round_bits = (xbits | (1 << (53 - 1))) & ((FP8_DP_HALF_ULP << (shift + 1)) - 1)
    if (round_bits > (FP8_DP_HALF_ULP << shift)) or (round_bits == (FP8_DP_HALF_ULP << shift) and (res & 1)):
      res = (res + 1) & 0xFF

  res |= sign
  return res

def fp8_to_float(x: int, fp8_interpretation: str) -> float:
  assert fp8_interpretation in ["E4M3", "E5M2"], "Invalid fp8 interpretation"
  if x is math.nan:
    return math.nan
  ur = x << 8

  if fp8_interpretation == "E5M2":
    if (ur & 0x7FFF) > 0x7C00:
      ur = 0x7FFF
  elif fp8_interpretation == "E4M3":
    sign = ur & 0x8000
    exponent = ((ur & 0x7800) >> 1) + 0x2000
    mantissa = (ur & 0x0700) >> 1
    absx = x & 0x7F

    if absx == 0x7F:
      ur = 0x7FFF
    elif exponent == 0x2000:
      if mantissa != 0:
        mantissa <<= 1
        while (mantissa & 0x0400) == 0:
          mantissa <<= 1
          exponent -= 0x0400
        mantissa &= 0x03FF
      else:
        exponent = 0
      ur = (sign | exponent) | mantissa
    else:
      ur = (sign | exponent) | mantissa

  half_bytes = struct.pack('<H', ur)
  float32_val = struct.unpack('e', half_bytes)[0]
  return float(float32_val)

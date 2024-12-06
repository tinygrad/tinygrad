import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.device import Device, is_dtype_supported

def bit_extract(x, e, s) -> Tensor: 
  mask = (1 << (e - s + 1)) - 1
  return (x >> s) & mask

def u16_to_f16(x):
  sign = bit_extract(x, 15, 15).float()
  exponent = bit_extract(x, 14, 10).float()
  fraction = bit_extract(x, 9, 0).float()
  return sign.where(-1, 1) * exponent.where((exponent - 15.0).exp2() * (1 + fraction / 1024.0), 6.103515625e-5 * (fraction / 1024.0))

def u32_to_f16(oo):
  f1 = u16_to_f16(oo>>16)
  f2 = u16_to_f16(oo&0xFFFF)
  return Tensor.cat(f2.reshape(-1, 1), f1.reshape(-1, 1), dim=1).flatten()

if __name__ == "__main__":
  a = Tensor.randn(50, dtype=dtypes.float16, device=None if is_dtype_supported(dtypes.float16) else "CLANG:0")
  f16_as_u32 = a.bitcast(dtypes.uint32) if is_dtype_supported(dtypes.float16) else a.bitcast(dtypes.uint32).to(Device.DEFAULT) 
  f16 = u32_to_f16(f16_as_u32[:50])

  ref = a.numpy()
  out = f16.numpy().astype(np.float16)

  print(ref-out)

  np.testing.assert_allclose(out, ref)

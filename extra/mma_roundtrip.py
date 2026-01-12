from extra.assembly.amd.autogen.rdna4.ins import *
from extra.assembly.amd.test.test_roundtrip import compile_asm

tests = [
  ("v_wmma_bf16_16x16x16_bf16 v[0:7], v[8:15], v[8:15], 1",
   v_wmma_bf16_16x16x16_bf16(v[0:7], v[8:15], v[8:15], 1)),

  ("v_wmma_f16_16x16x16_f16 v[0:7], v[8:15], v[8:15], 1",
   v_wmma_f16_16x16x16_f16(v[0:7], v[8:15], v[8:15], 1)),

  ("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[8:15], 1",
   v_wmma_f32_16x16x16_bf16(v[0:7], v[8:15], v[8:15], 1)),

  ("v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[8:15], 1",
   v_wmma_f32_16x16x16_f16(v[0:7], v[8:15], v[8:15], 1)),

  ("v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[8:9], 1",
   v_wmma_i32_16x16x16_iu4(v[0:7], v[8:9], v[8:9], 1)),

  ("v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[8:11], 1",
   v_wmma_i32_16x16x16_iu8(v[0:7], v[8:11], v[8:11], 1)),
]

for t, i in tests:
  t = compile_asm(t, arch="rdna3")
  try: assert t == i.to_bytes()
  except Exception as e:
    ref = i.__class__.from_bytes(t)
    for k in i._fields:
      print(f"{k}={getattr(i, k)} {getattr(ref, k)}")
      raise e

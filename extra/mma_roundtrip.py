from extra.assembly.amd.autogen.cdna.ins import *
from extra.assembly.amd.test.test_roundtrip import compile_asm

tests = [
  ("v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[0:1], 1",
   v_mfma_f32_16x16x16_f16(v[0:3], v[0:1], v[0:1], 1, clmp=1)),

  ("v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[0:1], 1",
   v_mfma_f32_16x16x16_bf16(v[0:3], v[0:1], v[0:1], 1, clmp=1)),

  ("v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], 1",
   v_mfma_f32_16x16x32_f16(v[0:3], v[0:3], v[0:3], 1, clmp=1)),

  ("v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], 1",
   v_mfma_f32_16x16x32_bf16(v[0:3], v[0:3], v[0:3], 1, clmp=1)),

  ("v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[0:7], 1",
   v_mfma_f32_16x16x128_f8f6f4(v[0:3], v[0:7], v[0:7], 1, clmp=1)),

  ("v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[0:5], 1, cbsz:2 blgp:2",
   v_mfma_f32_16x16x128_f8f6f4(v[0:3], v[0:5], v[0:5], 1, neg=2, neg_hi=2)),

  ("v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:3], v[0:3], 1, cbsz:4 blgp:4",
   v_mfma_f32_16x16x128_f8f6f4(v[0:3], v[0:3], v[0:3], 1, neg=4, neg_hi=4)),
]

for t, i in tests:
  t = compile_asm(t, arch="cdna")
  try: assert t == i.to_bytes()
  except Exception as e:
    ref = i.__class__.from_bytes(t)
    for k in i._fields:
      print(f"{k}={getattr(i, k)} {getattr(ref, k)}")
      raise e

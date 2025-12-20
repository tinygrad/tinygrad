#!/usr/bin/env python
import unittest

from tinygrad.codegen import full_rewrite
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.uop.ops import UOp, Ops

class TestAMDRDNA4FP8Codegen(unittest.TestCase):
  def _render_fp8_cast_kernel(self, fp8_dtype):
    a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    b = UOp(Ops.DEFINE_GLOBAL, fp8_dtype.ptr(), (), 1)
    c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 2)
    idx = UOp.const(dtypes.int, 0)

    f = a.index(idx, ptr=True).load(dtype=dtypes.float)
    q = f.cast(fp8_dtype)
    dq = q.cast(dtypes.float)

    uops = full_rewrite(UOp.sink(b.index(idx, ptr=True).store(q), c.index(idx, ptr=True).store(dq)))
    return AMDRenderer("gfx1201").render(uops)

  def test_fp8e4m3_casts_use_builtins_gfx1201(self):
    src = self._render_fp8_cast_kernel(dtypes.fp8e4m3)
    self.assertIn("__builtin_amdgcn_cvt_pk_fp8_f32", src)
    self.assertIn("__builtin_amdgcn_cvt_f32_fp8", src)

  def test_fp8e5m2_casts_use_builtins_gfx1201(self):
    src = self._render_fp8_cast_kernel(dtypes.fp8e5m2)
    self.assertIn("__builtin_amdgcn_cvt_pk_bf8_f32", src)
    self.assertIn("__builtin_amdgcn_cvt_f32_bf8", src)

if __name__ == "__main__":
  unittest.main()

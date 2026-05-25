from tinygrad.renderer.isa import ISARenderer
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, UPat, PatternMatcher

from tinygrad.dtype import dtypes
from tinygrad.uop import Ops

from tinygrad.runtime.autogen.amd.rdna3 import ins as RDNA3Ins

pre_isel_matcher = PatternMatcher([])
isel_matcher = PatternMatcher([

  (UPat(Ops.PARAM, name="x"), lambda x: None),
  (UPat(Ops.SPECIAL, name="x"), lambda x: None), ## not sure what to do about this yet as register spec is unimpl 

  (UPat(
    Ops.LOAD,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="offset")))), name="x"),
   lambda x, base, offset: x.ins(RDNA3Ins.global_load_b32, src=(offset, base))),

  (UPat.var("a", dtypes.float32) + UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_add_f32_e32, src = (a,b))),
  (UPat.var("a", dtypes.float32) * UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_mul_f32_e32, src = (a,b))),

  (UPat(
    Ops.STORE,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="offset"))),
      UPat(name="val")), name="x"),
   lambda x, base, offset, val: x.ins(RDNA3Ins.global_store_b32, src=(offset, val, base))),

  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=(x.ins(RDNA3Ins.s_endpgm, src=x.src),)) if not x.src or x.src[0].op is not Ops.INS else None),
])
post_regalloc_matcher = PatternMatcher([])

class RDNA3Renderer(ISARenderer):
  shared_max = 65536
  global_max = (2147483647, 65535, 65535)
  global_prod_max = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher

  def __init__(self, target:Target):
      super().__init__(target)
      from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
      self.compiler = AMDLLVMCompiler(target.arch)

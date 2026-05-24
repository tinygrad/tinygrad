from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, UPat, PatternMatcher

from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp

from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm

pre_isel_matcher = PatternMatcher([])
isel_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=(x.ins(s_endpgm(), src=x.src),)) if not x.src or x.src[0].op is not Ops.INS else None),
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

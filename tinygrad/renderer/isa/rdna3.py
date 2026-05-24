from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, UPat, PatternMatcher

pre_isel_matcher = PatternMatcher([])
isel_matcher = PatternMatcher([])
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

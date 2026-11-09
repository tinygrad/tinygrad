from tinygrad.uop.ops import UOp, PatternMatcher
from tinygrad.uop.render import render_uir
from tinygrad.renderer.x86 import ISARenderer

class X86DSLRenderer(ISARenderer):
  device = "CPU"
  has_local = False
  global_max = (1, 0, 0)
  pre_isel_matcher = PatternMatcher([])
  isel_matcher = PatternMatcher([])
  pre_regalloc_matcher = PatternMatcher([])
  post_regalloc_matcher = PatternMatcher([])

  def render(self, uops:list[UOp]) -> str: return render_uir(uops)
  def asm_str(self, uops:list[UOp], function_name:str) -> str: return self.render(uops)

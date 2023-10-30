import functools
from typing import List

from tinygrad.codegen.linearizer import UOps, Linearizer
from tinygrad.features.image import fix_schedule_for_images
from tinygrad.helpers import DEBUG

class CompilerStack:
  def __init__(self, name, linearizer_opts, renderer, pass_list): self.name, self.linearizer_opts, self.renderer, self.pass_list = name, linearizer_opts, renderer, pass_list
  def compile(self, kernel_name, uops: List[UOps]):
    prg, runtime_args = self.renderer(kernel_name, uops)
    if DEBUG >= 4: print(prg)
    ret = functools.reduce(lambda ir, f: f(kernel_name, ir), self.pass_list, prg)
    return ret, runtime_args
  def make_lin(self, ast): return Linearizer(ast, self.linearizer_opts)
  def compile_ast(self, ast):
    lin = self.make_lin(ast)
    lin.hand_coded_optimizations()
    lin.linearize()
    return self.compile(lin.function_name, lin.uops)
  def plus_pass(self, comp_pass): return CompilerStack(self.name, self.linearizer_opts, self.renderer, self.pass_list + [comp_pass])

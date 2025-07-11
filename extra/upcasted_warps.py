# play with upcasted warps
from tinygrad import Tensor, Device
from tinygrad.uop.ops import KernelInfo
from tinygrad.opt import get_optimized_ast
from tinygrad.opt.kernel import OptOps, Opt
from tinygrad.engine.realize import get_program

if __name__ == "__main__":
  renderer = Device.default.renderer

  N = 64
  b = Tensor.empty(N,N)
  c = b.sum(axis=1)
  opts = (Opt(OptOps.UNROLL, 0, 32),)
  ast = c.schedule()[-1].ast
  ast = ast.replace(arg=KernelInfo(opts_to_apply=opts))
  ast = get_optimized_ast(ast, renderer)
  prg = get_program(ast, renderer)
  print(prg.src)



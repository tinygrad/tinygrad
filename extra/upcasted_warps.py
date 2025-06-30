# play with upcasted warps
from tinygrad import Tensor, Device
from tinygrad.uop.ops import KernelInfo
from tinygrad.opt import get_optimized_ast
from tinygrad.opt.kernel import OptOps, Opt
from tinygrad.engine.realize import get_program

if __name__ == "__main__":

  renderer = Device.default.renderer
  N = 64
  a = Tensor.empty(N,N)

  out = a.sum(axis=1)
  ast = out.schedule()[-1].ast
  opts = tuple()
  opts += (Opt(OptOps.UPCAST, 0, 8),)
  opts += (Opt(OptOps.UNROLL, 0, 8),)
  ast = ast.replace(arg=KernelInfo(opts_to_apply=opts))
  ast = get_optimized_ast(ast, renderer)
  prg = get_program(ast, renderer)
  print(prg.src)

  out = a.sum(axis=1)
  ast = out.schedule()[-1].ast
  opts = tuple()
  opts += (Opt(OptOps.UNROLL, 0, 8),)
  opts += (Opt(OptOps.UPCAST, 0, 8),)
  ast = ast.replace(arg=KernelInfo(opts_to_apply=opts))
  ast = get_optimized_ast(ast, renderer)
  prg = get_program(ast, renderer)
  print(prg.src)

  # gemm
  """
  b = Tensor.empty(N,N)
  # metal TC
  #opts = (Opt(OptOps.UPCAST, 0, 2), # not the warp
  #        Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2), Opt(OptOps.UPCAST, 1, 2),
  #        Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2))
  # new TC should just be able to extract from this and swizzle as needed
  opts = (Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UPCAST, 1, 8), Opt(OptOps.UNROLL, 0, 8))
  c = (a@b)
  ast = c.schedule()[-1].ast
  ast = ast.replace(arg=KernelInfo(opts_to_apply=opts))
  ast = get_optimized_ast(ast, renderer)
  prg = get_program(ast, renderer)
  print(prg.src)
  """

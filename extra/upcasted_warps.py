# play with upcasted warps
from tinygrad import Tensor, Device
from tinygrad.uop.ops import KernelInfo
from tinygrad.opt import get_optimized_ast
from tinygrad.opt.kernel import OptOps, Opt
from tinygrad.engine.realize import get_program

if __name__ == "__main__":
  renderer = Device.default.renderer
  N = 64

  """
  a = Tensor.empty(N,N)

  out = (a + 1) #.sum(axis=2)
  ast = out.schedule()[-1].ast
  opts = tuple()
  opts += (Opt(OptOps.UPCAST, 0, 32),)
  ast = ast.replace(arg=KernelInfo(opts_to_apply=opts))
  ast = get_optimized_ast(ast, renderer)
  prg = get_program(ast, renderer)
  print(prg.src)
  """

  # how you split the store determines everything if you don't allow cross warp comms.
  # actually not everything, there's also the split before the horizontal (unrolled) reduces

  # new flow
  #  - pull out any dimensions from the store that you want to upcast.
  #  - decide how you want to assign them to registers. GPUs have a 512-byte memory LOAD/STORE which loads into 4 regs. see BUFFER_LOAD_B128
  #    - the loads and stores can be shuffled, but only in restrictive ways. in kernels without reduces, the store determines everything
  #    - it loads 16 bytes from up 32 different places = 512 bytes
  #  - in kernels with reduces, you now have more flexibility. the final target of the reduce must be what is stored
  #    - warp dimensions can be in the reduce (this is GROUP)

  # every dimension can be assigned to <global, local, loop, upcast, warp>

  """
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
  """

  # gemm
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

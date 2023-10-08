import sys
from tqdm import tqdm
from tinygrad.helpers import dedup
from tinygrad.codegen.linearizer import Linearizer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf = float('inf')

if __name__ == "__main__":
  ast_strs = dedup(open(sys.argv[1]).read().strip().split("\n"))
  print(f"loaded {len(ast_strs)} kernels")

  for ast_str in tqdm(ast_strs):
    ast = eval(ast_str)
    lin = Linearizer(ast)
    preopt = lin.colored_shape()
    lin.hand_coded_optimizations()
    postopt = lin.colored_shape()
    lin.linearize()
    try:
      gflops = lin.info.flops/1e9
    except Exception:
      # TODO: fix symbolic
      gflops = float('nan')
    print(f"{len(lin.uops)} uops, {lin.global_size} {lin.local_size}, {gflops:.2f} GFLOPS", preopt, "->", postopt)
    #for u in lin.uops: print(u)


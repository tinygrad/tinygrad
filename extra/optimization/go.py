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
  asts = [eval(x) for x in dedup(open(sys.argv[1]).read().strip().split("\n"))]
  print(f"loaded {len(asts)} kernels")

  for ast in tqdm(asts):
    lin = Linearizer(ast)
    #preopt = lin.colored_shape()
    lin.hand_coded_optimizations()
    #postopt = lin.colored_shape()
    #print(preopt, "->", postopt)
    lin.linearize()
    #for u in lin.uops: print(u)


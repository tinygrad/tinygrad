from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.device import Buffer, Device
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, BinaryOps, MetaOps, ReduceOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.dtype import dtypes
from tinygrad.engine.graph import print_tree

# Temporary test of split reduce
# Equivalent ast to NOOPT Tensor.rand(256,255).realize().sum()

ast = LazyOp(op=MetaOps.KERNEL, src=(
        LazyOp(op=BufferOps.STORE, src=(
          LazyOp(op=ReduceOps.SUM, src=(
            LazyOp(op=BufferOps.LOAD, src=(), 
          arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(256, 255), strides=(255, 1), offset=0, mask=None, contiguous=True),)))),), 
        arg=(0, 1)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)))),), 
      arg=None)

k = Kernel(ast, opts=Device["CLANG"].renderer)
k.apply_opt(Opt(OptOps.SPLIT, 0, 256))
newast = k.get_optimized_ast()
print_tree(newast)
k1, k2 = Kernel.split(newast, k.opts)
k1.linearize()
k2.linearize()

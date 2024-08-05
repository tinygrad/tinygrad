from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.device import Buffer, Device
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, BinaryOps, MetaOps, ReduceOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.dtype import dtypes
from tinygrad.engine.graph import print_tree

# Temporary test of some split reduces

ast = LazyOp(op=MetaOps.KERNEL, src=(
        LazyOp(op=BufferOps.STORE, src=(
          LazyOp(op=ReduceOps.SUM, src=(
            LazyOp(op=BufferOps.LOAD, src=(), 
          arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(256, 255), strides=(255, 1), offset=0, mask=None, contiguous=True),)))),), 
        arg=(0, 1)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)))),), 
      arg=None)

ast2 = LazyOp(op=MetaOps.KERNEL, src=(
         LazyOp(op=BufferOps.STORE, src=(
           LazyOp(op=ReduceOps.SUM, src=(
             LazyOp(op=BufferOps.LOAD, src=(), 
           arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(4, 255, 256), strides=(65280, 256, 1), offset=0, mask=None, contiguous=True),)))),),
         arg=(0, 1, 2)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))),), 
       arg=None)
  

k = Kernel(ast, opts=Device["CLANG"].renderer)
k.apply_opt(Opt(OptOps.SPLIT, 0, 256))
k1, k2 = k.split()

k1.linearize()
k2.linearize()

k = Kernel(ast2, opts=Device["CLANG"].renderer)
k.apply_opt(Opt(OptOps.SPLIT, 0, 255))
k2, k1 = k.split()

k1.linearize()
k2.linearize()

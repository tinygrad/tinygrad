import numpy as np

from tinygrad.runtime.ops_gpu import CLCodegen
from tinygrad.codegen.assembly import AssemblyCodegen

from tinygrad.helpers import LazyNumpyArray, dtypes
from tinygrad.ops import LazyOp, BinaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker

ones = LazyNumpyArray.from_np(np.ones((3,), np.float32))

#target = "GPU"
target = "RDNA"

b1 = LazyBuffer.fromCPU(ones, target)
b2 = LazyBuffer.fromCPU(ones, target)

out = LazyBuffer(target, ShapeTracker((3,)), BinaryOps, LazyOp(BinaryOps.ADD, (b1, b2)), dtypes.float32)
print(out.toCPU())

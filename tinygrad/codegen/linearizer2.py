from typing import List, Tuple, Union
from tinygrad.shape.symbolic import sint
from tinygrad.ops import LazyOp, MemBuffer, ConstBuffer, BufferOps
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.graph import print_tree
from tinygrad.helpers import flatten, dedup

# Linearizer replacement
# 1. add "groups" and LocalBuffers to the LazyOp, still not a list (can skip for now). this can split into multiple blueprints. pad lives here?
# 2. create a ShapeTracker "Blueprint"
# 3. Linearize
# 4. generate UOps from the Ops

# this is a subset of "OptOps"
# at start, axes are either "global" or "reduce"
# these loads exist apart from the actual implementation

# "Linearizer" actually does three things
#    "grouping" = this changes the LazyOp structure
#    "tvm schedule" = stuff that deals with the ShapeTracker and axis mappings. This is Blueprint
#    "tvm lower" = LazyOp + Blueprint -> UOps

def linearize_lazyop(x:LazyOp) -> List[LazyOp]:
  return flatten([linearize_lazyop(x) for x in x.src]) + [x]

# global > local > group > reduce > upcast > unroll

# full reduce example (grouping):
#  LOAD MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(16384,), strides=(1,), offset=0, mask=None, contiguous=True),)))
#  SUM (1,)
#  STORE MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)))
# A = (16384,)
# split
# A  = (256, 64)
# Al = (256, 1)
#  TODO: this should be two blueprints in one kernel
#  LOAD MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(256, 64), strides=(1,), offset=0, mask=None, contiguous=True),)))
#  SUM (256, 1)
#  STORE LocalBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(256, 1), strides=(1,0), offset=0, mask=None, contiguous=True),)))
#  BARRIER (note: 256 is now a reduce)
#  NOTE: UPCASTMID is just an UPCAST on this LOAD
#  LOAD LocalBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(256, 1), strides=(1,0), offset=0, mask=None, contiguous=True),)))
#  SUM (1, 1)
#  STORE MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1,1), strides=(0,0), offset=0, mask=None, contiguous=True),)))

# matmul example: (A*B).sum(axis=2)
# C = (1024, 1024, 1)
# A = (1024, 1024, 1024)
# B = (1024, 1024, 1024)
# output_shape = (1024, 1024, 1)
# full_shape   = (1024, 1024, 1024)

# UNROLL axis=2 (4)
# C = (1024, 1024, 1)
# A = (1024, 1024, 256, 4)
# B = (1024, 1024, 256, 4)

# LOCAL axis=0 (32)
# C = (32, 32 1024, 1)
# A = (32, 32, 1024, 256, 4)
# B = (32, 32, 1024, 256, 4)
#     glb lcl  glb   red  un

# UPCAST axis=0 (4)
# C = (256, 4,  1024, 1)
# A = (256, 4,  1024, 256, 4)
# B = (256, 4,  1024, 256, 4)
#      glb  up  glb   red  un



class Linearizer:
  def __init__(self, ast:LazyOp, opts:LinearizerOptions):
    self.opts = opts
    self.ast = ast
    self.name = "billy"
    self.global_size = [1,1,1]
    self.local_size = [1,1,1]

    # the code lives here
    self.uops = []

  def required_optimizations(self): pass
  def hand_coded_optimizations(self): pass
  def apply_tensor_cores(self, use_tensor_cores=1): return False

  def linearize(self):
    ops = linearize_lazyop(self.ast)
    print("**** ops *****")
    for i,o in enumerate(ops): print(i, o.op, o.arg, [ops.index(x) for x in o.src])
    print("**** ops *****")

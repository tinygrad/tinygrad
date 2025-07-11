# write a version of kernel that uses rewrite rules to apply OptOps

# we need a new op to track the AxisType.
# we can also use this and know the dtype count of the ops to not have to push the expander through them.
# these are like the names of the axes in halide

from tinygrad.uop.ops import UOp
from tinygrad.opt.kernel import Opt
from tinygrad.shape.shapetracker import ShapeTracker

# all Opts need to be assigned to either a SINK, STORE or a REDUCE_AXIS.
# TODO: explore how Halide is doing this sort of assignment

# LOAD: SMEM (if locals)
# REDUCE_AXIS: TC (if fed by a mul or cast-mul), UNROLL, GROUP (if locals), GROUPTOP (if locals)
# SINK: NOLOCALS (if locals), LOCAL (if locals), SWAP, PADTO, UPCAST
# TODO: multistore kernels

def apply_optops(ast:UOp, opts:list[Opt]):
  st = ShapeTracker(ast.shape)
  for o in opts:
    pass


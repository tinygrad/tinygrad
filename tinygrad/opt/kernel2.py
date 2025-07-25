# write a version of kernel that uses rewrite rules to apply OptOps

# we need a new op to track the AxisType.
# we can also use this and know the dtype count of the ops to not have to push the expander through them.
# one of the problems with knowing the dtype count is that
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

# so aside from TC, we can do this all with range splitting. for TC, we'd have to upcast warps
# without upcasted warps, we'd need to somehow know where the warp is

# we *have* to upcast the warps. there's no point in doing this refactor without it
# the old question, do we left justify or right justify the warp?

# <upcasts> <warp> <unrolls...> is the input to the reduce, but we can quickly shift it (or not if it's a down shuffle)
# we can also add a "which ones to reduce" pattern to the reduce maybe. or maybe it's just better to do with a GEP
# the reduce is expanded quickly enough that it doesn't matter
# we could also switch the reduce (but not reduce_axis) to remove the high dimensions, aka more nonlocal
# we could even switch reduce_axis...like it sort of makes more sense that way.

# on GPU, what if upcast actually just splits the op? at the end of the day that's what it does, splits into registers
# similarly with UNROLL
# but the problem with splitting is loads, you want to be able to use that 4x load whenever possible

# alternatively, we allow multidimensional dtypes where the dtype is the shape

#  - decide how you want to assign them to registers. GPUs have a 512-byte memory LOAD/STORE which loads into 4 regs. see BUFFER_LOAD_B128
#    - the loads and stores can be shuffled, but only in restrictive ways. in kernels without reduces, the store determines everything
#    - it loads 16 bytes from up 32 different places = 512 bytes

# three things the new opt absolutely needs to support
#  - upcasted warps
#  - smem with async copy (even TMA maybe, does AMD have a TMA?)
#  - locals double buffering (what is this in UOps?)

# basically everything needed to make flash attention and GEMM fast on CDNA4

# <globals> <locals> (<unrolls/upcasts> <warp>) <loop reduce>

def apply_optops(ast:UOp, opts:list[Opt]):
  st = ShapeTracker(ast.shape)
  for o in opts:
    pass


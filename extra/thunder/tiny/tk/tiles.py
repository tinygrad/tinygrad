import math
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

from extra.thunder.tiny.tk import WARP_THREADS

global_slot = 0
def gl(shape, dtype):
  global global_slot
  global_slot += 1
  return UOp.placeholder(shape, dtype, slot=global_slot-1)

shared_slot = 0
def st(shape, dtype):
  global shared_slot
  shared_slot += 1
  return UOp.placeholder(shape, dtype, addrspace=AddrSpace.LOCAL, slot=shared_slot-1)

TILE_ROW_DIM, TILE_COL_DIM = 16, 16
RT_BASE_TILE_NE = TILE_ROW_DIM * TILE_COL_DIM
RT_BASE_TILE_NEPT = RT_BASE_TILE_NE // WARP_THREADS
register_slot = 0
def rt(shape, dtype):
  assert len(shape) == 2

  height = shape[0] // TILE_ROW_DIM
  width = shape[1] // TILE_COL_DIM

  global register_slot
  register_slot += 1
  return UOp.placeholder((height, width, RT_BASE_TILE_NEPT), dtype, addrspace=AddrSpace.REG, slot=register_slot-1)

def rv(length, dtype, layout="naive"):
  tiles = length // TILE_ROW_DIM
  match layout:
    case "naive":
      inner_dim = 1
      outer_dim = (tiles + 1) // 2
    case "ortho":
      inner_dim = 1
      outer_dim = tiles
    case _: raise NotImplementedError(f"rv layout {layout} not implemented")

  global register_slot
  register_slot += 1
  return UOp.placeholder((outer_dim, inner_dim), dtype, addrspace=AddrSpace.REG, slot=register_slot-1)

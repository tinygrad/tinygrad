import math, functools
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.tiles import TILE_ROW_DIM, TILE_COL_DIM, RT_BASE_TILE_NEPT, shared_slot

class Group:
  def __init__(self, warps:int, threadIdx_x:UOp):
    self.warps = warps
    self.group_threads = warps * WARP_THREADS
    self.threadIdx_x = threadIdx_x

  # helpers
  @property
  def laneid(self): return self.threadIdx_x % self.group_threads
  @property
  def warpid(self): return self.laneid // WARP_THREADS
  @property
  def groupid(self): return self.threadIdx_x // self.group_threads

  # ops that only work on a single warp

  clear_rid = 1000
  def clear(self, reg:UOp, value:float=0):
    assert self.warps == 1

    i = UOp.range(reg.size, Group.clear_rid)
    Group.clear_rid += 1
    return reg.reshape((reg.size,))[i].set(value, end=i).after(reg).reshape(reg.shape)

  def zero(self, reg:UOp): return self.clear(reg, 0)
  def neg_inf(self, reg:UOp): return self.clear(reg, -math.inf)

  copy_rid = 300
  def copy(self, dst:UOp, src:UOp):
    assert self.warps == 1

    assert dst.shape == src.shape
    assert cast(PtrDType, dst.dtype).addrspace == AddrSpace.REG
    assert cast(PtrDType, src.dtype).addrspace == AddrSpace.REG

    rngs_for_shape = tuple(UOp.range(dim, Group.copy_rid + i) for i, dim in enumerate(dst.shape))
    Group.copy_rid += len(dst.shape)

    dst_store = dst[*rngs_for_shape].store(src[*rngs_for_shape].cast(dst.dtype.base)).end(*rngs_for_shape)

    return dst.after(dst_store).reshape(dst.shape)

  mma_rid = 600
  def mma_AB(self, c:UOp, a:UOp, b:UOp, after=True):
    assert self.warps == 1

    mma_i_height = UOp.range(c.shape[-3], Group.mma_rid)
    mma_i_width = UOp.range(c.shape[-2], Group.mma_rid+1)
    mma_i_inner = UOp.range(a.shape[-2], Group.mma_rid+2, AxisType.REDUCE)
    Group.mma_rid += 3

    wmma_arg = ("WMMA_8_16_16_bfloat16_float", (8, 16, 16), dtypes.bfloat16, dtypes.float, "CUDA", 32, (((4, 2), (3, 2), (8, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ())

    a_in = UOp.vectorize(*[a[mma_i_height, mma_i_inner, i] for i in range(8)])
    b_in1 = UOp.vectorize(*([b[mma_i_inner, mma_i_width, i] for i in range(2)] + [b[mma_i_inner, mma_i_width, 4+i] for i in range(2)]))
    c_out1 = UOp.vectorize(*[c[mma_i_height, mma_i_width, i] for i in range(4)])
    b_in2 = UOp.vectorize(*([b[mma_i_inner, mma_i_width, 2+i] for i in range(2)] + [b[mma_i_inner, mma_i_width, 6+i] for i in range(2)]))
    c_out2 = UOp.vectorize(*[c[mma_i_height, mma_i_width, 4+i] for i in range(4)])

    out1 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in1, c_out1), arg=wmma_arg)
    out2 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in2, c_out2), arg=wmma_arg)
    c_i = [c[mma_i_height, mma_i_width, i].store(out1.gep(i)) for i in range(4)] + [c[mma_i_height, mma_i_width, 4+i].store(out2.gep(i)) for i in range(4)]
    c_store = UOp.group(*c_i).end(mma_i_height, mma_i_width, mma_i_inner)

    return c.after(c_store).reshape(c.shape) if after else c_store

  def mma_ABt(self, c:UOp, a:UOp, b:UOp, after=True):
    assert self.warps == 1

    mma_i_height = UOp.range(c.shape[-3], Group.mma_rid)
    mma_i_width = UOp.range(c.shape[-2], Group.mma_rid+1)
    mma_i_inner = UOp.range(a.shape[-2], Group.mma_rid+2, AxisType.REDUCE)
    Group.mma_rid += 3

    wmma_arg = ("WMMA_8_16_16_bfloat16_float", (8, 16, 16), dtypes.bfloat16, dtypes.float, "CUDA", 32, (((4, 2), (3, 2), (8, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ())

    a_in = UOp.vectorize(*[a[mma_i_height, mma_i_inner, i] for i in range(8)])
    b_in1 = UOp.vectorize(*([b[mma_i_width, mma_i_inner, i] for i in range(2)] + [b[mma_i_width, mma_i_inner, 4+i] for i in range(2)]))
    c_out1 = UOp.vectorize(*[c[mma_i_height, mma_i_width, i] for i in range(4)])
    b_in2 = UOp.vectorize(*([b[mma_i_width, mma_i_inner, 2+i] for i in range(2)] + [b[mma_i_width, mma_i_inner, 6+i] for i in range(2)]))
    c_out2 = UOp.vectorize(*[c[mma_i_height, mma_i_width, 4+i] for i in range(4)])

    out1 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in1, c_out1), arg=wmma_arg)
    out2 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in2, c_out2), arg=wmma_arg)
    c_i = [c[mma_i_height, mma_i_width, i].store(out1.gep(i)) for i in range(4)] + [c[mma_i_height, mma_i_width, 4+i].store(out2.gep(i)) for i in range(4)]
    c_store = UOp.group(*c_i).end(mma_i_height, mma_i_width, mma_i_inner)

    return c.after(c_store).reshape(c.shape) if after else c_store

  map_rid = 400
  def map(self, a:UOp, op:Callable[[UOp], UOp]|Callable[[UOp, tuple], UOp]):
    assert self.warps == 1

    rngs_for_shape = tuple(UOp.range(dim, Group.map_rid + i) for i, dim in enumerate(a.shape))
    Group.map_rid += len(a.shape)

    if op.__code__.co_argcount == 1:
      to_store = op(a[*rngs_for_shape])
    else:
      to_store = op(a[*rngs_for_shape], rngs_for_shape)

    a_store = a[*rngs_for_shape].store(to_store).end(*rngs_for_shape)
    return a.after(a_store).reshape(a.shape)

  red_rid = 500
  def row_reduce(self, vec:UOp, src:UOp, op:Callable[[UOp, UOp], UOp]):
    assert self.warps == 1

    red_i_height = UOp.range(src.shape[-3], Group.red_rid)
    red_i_width = UOp.range(src.shape[-2], Group.red_rid+1)
    red_i_inner = UOp.range(RT_BASE_TILE_NEPT, Group.red_rid+2, AxisType.REDUCE)
    Group.red_rid += 3

    global shared_slot
    red_local = UOp.placeholder((self.group_threads,), src.dtype.base, addrspace=AddrSpace.LOCAL, slot=shared_slot)
    shared_slot += 1

    # initial reduce in registers
    vec_store = vec[red_i_height, 0].store(op(vec.after(UOp.group(red_i_width, red_i_inner))[red_i_height, 0], src[red_i_height, red_i_width, red_i_inner])).end(red_i_height, red_i_width, red_i_inner)
    vec = vec.after(vec_store).reshape(vec.shape)

    # store to shared memory
    red_local_store = red_local[self.laneid].store(vec[red_i_height, 0])
    red_local = red_local.after(red_local_store).reshape(red_local.shape)

    # final reduce from shared memory
    offset = (self.laneid + 16) % 32
    red_local_i = (offset // 16) * 16 + (offset % 16)
    vec_store = vec[red_i_height, 0].store(op(vec[red_i_height, 0], red_local[red_local_i])).end(red_i_height)

    return vec.after(vec_store).reshape(vec.shape)

  # ops that can work across multiple warps

  LOAD_INNER = 8
  load_rid = 100
  def load(self, dst:UOp, src:UOp, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0):
    if cast(PtrDType, dst.dtype).addrspace == AddrSpace.REG:
      srcf = src.flatten(-2)

      load_i_height = UOp.range(dst.shape[-3], Group.load_rid)
      load_i_width = UOp.range(dst.shape[-2], Group.load_rid+1)
      load_i_inner = UOp.range(RT_BASE_TILE_NEPT, Group.load_rid+2)
      Group.load_rid += 3

      row = (self.warpid * dst.shape[-3] + load_i_height) * TILE_ROW_DIM + (self.laneid % 16)
      col = load_i_width * TILE_COL_DIM + (self.laneid // 16) * 8
      src_i_last = row * src.shape[-1] + col + load_i_inner

      dst_store = dst[*dst_idxs, load_i_height, load_i_width, load_i_inner].store(srcf[*idxs[:-2], src_i_last]).end(load_i_height, load_i_width, load_i_inner)
    else:
      dstf = dst.flatten(-2)

      srcf = src.flatten()
      row_stride = prod(src.shape[axis+1:])

      idxs = tuple(idx * dst.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * dst.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
      src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

      memcpy_per_row = dst.shape[-1] // Group.LOAD_INNER
      total_calls = prod(dst.shape[-2:]) // (self.group_threads * Group.LOAD_INNER)

      load_i_outer = UOp.range(total_calls, Group.load_rid)
      load_i_inner = UOp.range(Group.LOAD_INNER, Group.load_rid+1)
      Group.load_rid += 2

      load_idx = load_i_outer * self.group_threads + self.laneid
      row = load_idx // memcpy_per_row
      col = (load_idx * Group.LOAD_INNER) % dst.shape[-1]

      dst_i = row * dst.shape[-1] + col + load_i_inner
      src_i += row * row_stride + col + load_i_inner

      dst_store = dstf[*dst_idxs, dst_i].store(srcf[src_i]).end(load_i_outer, load_i_inner)

    return dst.after(dst_store.barrier()).reshape(dst.shape)

  STORE_INNER = 8
  store_rid = 200
  def store(self, dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0, after=True):
    if cast(PtrDType, src.dtype).addrspace == AddrSpace.REG:
      dstf = dst.flatten(-2)

      store_i_height = UOp.range(src.shape[-3], Group.store_rid)
      store_i_width = UOp.range(src.shape[-2], Group.store_rid+1)
      store_i_inner = UOp.range(RT_BASE_TILE_NEPT, Group.store_rid+2)
      Group.store_rid += 3

      row = (self.warpid * src.shape[-3] + store_i_height) * TILE_ROW_DIM + (self.laneid % 16)
      col = store_i_width * TILE_COL_DIM + (self.laneid // 16) * 8
      dst_i_last = row * dst.shape[-1] + col + store_i_inner

      dst_store = dstf[*idxs[:-2], dst_i_last].store(src[*src_idxs, store_i_height, store_i_width, store_i_inner]).end(store_i_height, store_i_width, store_i_inner)
    else:
      dstf = dst.flatten()
      row_stride = prod(dst.shape[axis+1:])

      idxs = tuple(idx * src.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * src.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
      dst_i = ((idxs[0] * dst.shape[-3] + idxs[1]) * dst.shape[-2] + idxs[2]) * dst.shape[-1] + idxs[3]

      srcf = src.flatten(-2)

      memcpy_per_row = src.shape[-1] // Group.STORE_INNER
      total_calls = prod(src.shape[-2:]) // (self.group_threads * Group.STORE_INNER)

      store_i_outer = UOp.range(total_calls, Group.store_rid)
      store_i_inner = UOp.range(Group.STORE_INNER, Group.store_rid+1)
      Group.store_rid += 2

      load_idx = store_i_outer * self.group_threads + self.laneid
      row = load_idx // memcpy_per_row
      col = (load_idx * Group.STORE_INNER) % src.shape[-1]

      src_i = row * src.shape[-1] + col + store_i_inner
      dst_i += row * row_stride + col + store_i_inner

      dst_store = dstf[dst_i].store(srcf[*src_idxs, src_i]).end(store_i_outer, store_i_inner)

    return dst.after(dst_store).reshape(dst.shape) if after else dst_store

warp_ = functools.partial(Group, 1)
warpgroup_ = functools.partial(Group, 4)

import math
from typing import cast
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

WARP_THREADS = 32

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

clear_rid = 16
def clear(reg:UOp, value:float=0):
  global clear_rid
  i = UOp.range(reg.size, clear_rid)
  clear_rid += 1
  return reg.reshape((reg.size,))[i].set(value, end=i).after(reg).reshape(reg.shape)

def zero(reg:UOp): return clear(reg, 0)
def neg_inf(reg:UOp): return clear(reg, -math.inf)

LOAD_INNER = 8
load_rid = 100
def load(dst:UOp, src:UOp, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0):
  global load_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  # flatten dst
  if cast(PtrDType, dst.dtype).addrspace == AddrSpace.REG:
    srcf = src.flatten(-2)

    load_i_height = UOp.range(dst.shape[-3], load_rid)
    load_i_width = UOp.range(dst.shape[-2], load_rid+1)
    load_i_inner = UOp.range(RT_BASE_TILE_NEPT, load_rid+2)
    load_rid += 3

    row = (warpid * dst.shape[-3] + load_i_height) * TILE_ROW_DIM + (laneid % 16)
    col = load_i_width * TILE_COL_DIM + (laneid // 16) * 8
    src_i_last = row * src.shape[-1] + col + load_i_inner

    dst_store = dst[*dst_idxs, load_i_height, load_i_width, load_i_inner].store(srcf[*idxs[:-2], src_i_last]).end(load_i_height, load_i_width, load_i_inner)
  else:
    dstf = dst.flatten(-2)

    srcf = src.flatten()
    row_stride = prod(src.shape[axis+1:])

    idxs = tuple(idx * dst.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
    idxs = tuple(idx * dst.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
    src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

    memcpy_per_row = dst.shape[-1] // LOAD_INNER
    total_calls = prod(dst.shape[-2:]) // (NUM_WORKERS * WARP_THREADS * LOAD_INNER)

    load_i_outer = UOp.range(total_calls, load_rid)
    load_i_inner = UOp.range(LOAD_INNER, load_rid+1)
    load_rid += 2

    load_idx = load_i_outer * (NUM_WORKERS * WARP_THREADS) + laneid
    row = load_idx // memcpy_per_row
    col = (load_idx * LOAD_INNER) % dst.shape[-1]

    dst_i = row * dst.shape[-1] + col + load_i_inner
    src_i += row * row_stride + col + load_i_inner

    dst_store = dstf[*dst_idxs, dst_i].store(srcf[src_i]).end(load_i_outer, load_i_inner)

  barrier = UOp.barrier(dst_store)

  return dst.after(barrier).reshape(dst.shape)

STORE_INNER = 8
store_rid = 200
def store(dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0, after=True):
  global store_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  # flatten src
  if cast(PtrDType, src.dtype).addrspace == AddrSpace.REG:
    dstf = dst.flatten(-2)

    store_i_height = UOp.range(src.shape[-3], store_rid)
    store_i_width = UOp.range(src.shape[-2], store_rid+1)
    store_i_inner = UOp.range(RT_BASE_TILE_NEPT, store_rid+2)
    store_rid += 3

    row = (warpid * src.shape[-3] + store_i_height) * TILE_ROW_DIM + (laneid % 16)
    col = store_i_width * TILE_COL_DIM + (laneid // 16) * 8
    dst_i_last = row * dst.shape[-1] + col + store_i_inner

    dst_store = dstf[*idxs[:-2], dst_i_last].store(src[*src_idxs, store_i_height, store_i_width, store_i_inner]).end(store_i_height, store_i_width, store_i_inner)
  else:
    dstf = dst.flatten()
    row_stride = prod(dst.shape[axis+1:])

    idxs = tuple(idx * src.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
    idxs = tuple(idx * src.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
    dst_i = ((idxs[0] * dst.shape[-3] + idxs[1]) * dst.shape[-2] + idxs[2]) * dst.shape[-1] + idxs[3]

    srcf = src.flatten(-2)

    memcpy_per_row = src.shape[-1] // STORE_INNER
    total_calls = prod(src.shape[-2:]) // (NUM_WORKERS * WARP_THREADS * STORE_INNER)

    store_i_outer = UOp.range(total_calls, store_rid)
    store_i_inner = UOp.range(STORE_INNER, store_rid+1)
    store_rid += 2

    load_idx = store_i_outer * (NUM_WORKERS * WARP_THREADS) + laneid
    row = load_idx // memcpy_per_row
    col = (load_idx * STORE_INNER) % src.shape[-1]

    src_i = row * src.shape[-1] + col + store_i_inner
    dst_i += row * row_stride + col + store_i_inner

    dst_store = dstf[dst_i].store(srcf[*src_idxs, src_i]).end(store_i_outer, store_i_inner)

  return dst.after(dst_store).reshape(dst.shape) if after else dst_store

def mma_ABt_base(d:UOp, a:UOp, b:UOp, c:UOp):
  pass

def mma_ABt(d:UOp, a:UOp, b:UOp, c:UOp):
  return d

def mma_AB_base(d:UOp, a:UOp, b:UOp, c:UOp):
  pass

def mma_AB(d:UOp, a:UOp, b:UOp, c:UOp):
  return d

def row_reduce(row_accum:UOp, src:UOp, src_accum:UOp):
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  leader = threadIdx_x & 0x1C



NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 1, 16, 1, 64

ROWS = 16 * (64 // D)

def ker():
  # define special indices
  blockIdx_x = UOp.special(N // (ROWS*NUM_WORKERS), "gidx0")
  blockIdx_y = UOp.special(H, "gidx1")
  blockIdx_z = UOp.special(B, "gidx2")
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")

  warpid = threadIdx_x // WARP_THREADS
  laneid = threadIdx_x % WARP_THREADS

  # kernel
  o = gl((B, N, H, D), dtypes.bfloat16)
  q = gl((B, N, H, D), dtypes.bfloat16)
  k = gl((B, N, H, D), dtypes.bfloat16)
  v = gl((B, N, H, D), dtypes.bfloat16)

  workerid = warpid

  batch, head, q_seq = blockIdx_z, blockIdx_y, blockIdx_x# * NUM_WORKERS + workerid

  k_smem = st((ROWS, D), dtypes.bfloat16)
  v_smem = st((ROWS, D), dtypes.bfloat16)
  qo_smem = st((ROWS, D), dtypes.bfloat16)

  q_reg = rt((ROWS, D), dtypes.bfloat16)
  k_reg = rt((ROWS, D), dtypes.bfloat16)
  v_reg = rt((D, ROWS), dtypes.bfloat16)
  o_reg = rt((ROWS, D), dtypes.float32)
  att_block = rt((ROWS, ROWS), dtypes.float32)
  att_block_mma = rt((ROWS, ROWS), dtypes.bfloat16)
  max_vec_last = rt((ROWS, 1), dtypes.float32)
  max_vec = rt((ROWS, 1), dtypes.float32)
  norm_vec = rt((ROWS, 1), dtypes.float32)

  max_vec = neg_inf(max_vec)
  norm_vec = zero(norm_vec)
  o_reg = zero(o_reg)

  # load q tile
  qo_smem = load(qo_smem, q, (), (batch, q_seq, head, 0), axis=1)
  q_reg = load(q_reg, qo_smem)

  height_rng = UOp.range(q_reg.shape[-3], 1)
  width_rng = UOp.range(q_reg.shape[-2], 2)
  subtile_rng = UOp.range(RT_BASE_TILE_NEPT, 3)
  q_reg_store = q_reg[height_rng, width_rng, subtile_rng].store(q_reg[height_rng, width_rng, subtile_rng] * ((1.0 / math.sqrt(D)) * (1.0 / math.log(2)))).end(height_rng, width_rng, subtile_rng)
  q_reg = q_reg.after(q_reg_store).reshape(q_reg.shape)

  outer_kv_rng = UOp.range(N // ROWS, 0)

  k_smem = load(k_smem, k, (), (batch, outer_kv_rng, head, 0), axis=1)
  v_smem = load(v_smem, v, (), (batch, outer_kv_rng, head, 0), axis=1)

  k_reg = load(k_reg, k_smem)
  att_block = zero(att_block)
  # TODO: mma_ABt
  att_block = mma_ABt(att_block, q_reg, k_reg, att_block)

  max_vec_last = max_vec # TODO: need copy?
  # max_vec = max(att_block, max_vec)
  # att_block = (att_block + max_vec * UOp.const(dtypes.float32, -1.)).exp2() # TODO: that is stupid
  # max_vec_last = (max_vec_last + max_vec * UOp.const(dtypes.float32, -1.)).exp2()
  # norm_vec = norm_vec * max_vec_last
  # norm_vec = norm_vec + att_block.sum()

  att_block_mma = att_block # TODO: def need copy
  v_reg = load(v_reg, v_smem)

  # o_reg = o_reg * max_vec_last
  o_reg = mma_AB(o_reg, att_block_mma, v_reg, o_reg)

  # o_reg = o_reg / norm_vec

  qo_smem = store(qo_smem, q_reg)
  o = store(o, qo_smem, (batch, q_seq, head, 0), (), axis=1, after=False)

  sink = o

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  with Context(DEBUG=0):
    q = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    # q = Tensor.arange(B * N * H * D).reshape(B, N, H, D).cast(dtypes.bfloat16).contiguous()
    k = Tensor.randn(B, N, H, D, dtype="bfloat16")
    v = Tensor.randn(B, N, H, D, dtype="bfloat16")
    out = Tensor.empty(B, N, H, D, dtype="bfloat16")
    Tensor.realize(q, k, v, out)

  sink = ker()
  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (out, q, k, v)])

  GlobalCounters.reset()
  times = []
  for _ in range(5):
    et = ei.run(wait=True)
    times.append(et)
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  print(f"{attn_flops/(min(times)*1e12):2f} TFLOPS")

  print(out.tolist())

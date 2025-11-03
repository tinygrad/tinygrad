import math
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv, prod

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

register_slot = 0
def rt(shape, dtype):
  global register_slot
  register_slot += 1
  return UOp.placeholder(shape, dtype, addrspace=AddrSpace.REG, slot=register_slot-1)

clear_rid = 16
def clear(reg:UOp, value:float=0):
  global clear_rid
  i = UOp.range(reg.size, clear_rid)
  clear_rid += 1
  return reg[i].set(value, end=i)

def zero(reg:UOp): return clear(reg, 0)
def neg_inf(reg:UOp): return clear(reg, -math.inf)

LOAD_INNER = 1
load_rid = 100
def load(dst:UOp, src:UOp, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0):
  global load_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // WARP_THREADS
  laneid = threadIdx_x % WARP_THREADS

  # permute src so that axis and axis + 1 and last
  perm = list(range(len(src.shape)))
  src_axis = perm.pop(axis)
  src_axis1 = perm.pop(axis)
  perm += [src_axis, src_axis1]
  srcp = src.permute(tuple(perm))
  srcp = srcp.reshape(srcp.shape[:-2] + (prod(srcp.shape[-2:]),))

  # flatten dst
  dstf = dst.reshape(dst.shape[:-2] + (prod(dst.shape[-2:]),))

  load_i_outer = UOp.range(dst.size // (WARP_THREADS * LOAD_INNER), load_rid)
  load_i_inner = UOp.range(LOAD_INNER, load_rid+1, AxisType.UPCAST)
  load_rid += 2

  dst_i = warpid * (WARP_THREADS * LOAD_INNER) + laneid * LOAD_INNER + load_i_outer * (WARP_THREADS * LOAD_INNER) + load_i_inner
  src_last_i = dst_i
  if len(dst.shape) != len(src.shape):
    src_last_i += idxs[-2] * src.shape[-1] + idxs[-1]

  dst_store = dstf[*dst_idxs, dst_i].store(src[*idxs[:-2], src_last_i]).end(load_i_outer, load_i_inner)

  barrier = UOp.barrier(dst_store)

  return dst.after(barrier).reshape(dst.shape)

STORE_INNER = 1
store_rid = 200
def store(dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0, after=True):
  global store_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // WARP_THREADS
  laneid = threadIdx_x % WARP_THREADS

  perm = list(range(len(dst.shape)))
  dst_axis = perm.pop(axis)
  dst_axis1 = perm.pop(axis)
  perm += [dst_axis, dst_axis1]
  dstp = dst.permute(tuple(perm))
  dstp = dstp.reshape(dstp.shape[:-2] + (prod(dstp.shape[-2:]),))

  # flatten src
  srcf = src.reshape(src.shape[:-2] + (prod(src.shape[-2:]),))

  store_i_outer = UOp.range(dst.size // (WARP_THREADS * STORE_INNER), store_rid)
  store_i_inner = UOp.range(STORE_INNER, store_rid+1, AxisType.UPCAST)
  store_rid += 2

  src_i = warpid * (WARP_THREADS * LOAD_INNER) + laneid * LOAD_INNER + store_i_outer * (WARP_THREADS * LOAD_INNER) + store_i_inner
  dst_last_i = src_i
  if len(dst.shape) != len(src.shape):
    dst_last_i += idxs[-2] * dst.shape[-1] + idxs[-1]

  dst_store = dstp[*idxs[:-2], dst_last_i].store(srcf[*src_idxs, src_i]).end(store_i_outer, store_i_inner)

  return dst.after(dst_store).reshape(dst.shape) if after else dst_store

WARP_THREADS = 32

NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 1, 64, 1, 64

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

  batch, head, q_seq = blockIdx_z, blockIdx_y, blockIdx_x * NUM_WORKERS + workerid

  k_smem = st((ROWS, D), dtypes.bfloat16)
  v_smem = st((ROWS, D), dtypes.bfloat16)
  qo_smem = st((ROWS, D), dtypes.bfloat16)

  q_reg = rt((ROWS, D), dtypes.bfloat16)
  k_reg = rt((ROWS, D), dtypes.bfloat16)
  v_reg = rt((D, ROWS), dtypes.bfloat16)
  o_reg = rt((ROWS, D), dtypes.float32)
  att_block = rt((ROWS, ROWS), dtypes.float32)
  att_block_mma = rt((ROWS, ROWS), dtypes.bfloat16)
  max_vec_last = rt((ROWS,), dtypes.float32)
  max_vec = rt((ROWS,), dtypes.float32)
  norm_vec = rt((ROWS,), dtypes.float32)

  max_vec = neg_inf(max_vec)
  norm_vec = zero(norm_vec)
  o_reg = zero(o_reg)

  outer_kv_rng = UOp.range(N // ROWS, 0)

  # load q tile
  qo_smem = load(qo_smem, q, (), (batch, q_seq, head, 0), axis=1)
  q_reg = load(q_reg, qo_smem)

  qo_smem = store(qo_smem, q_reg)
  o = store(o, qo_smem, (batch, q_seq, head, 0), (), axis=1, after=False)

  # q_reg = q_reg.set(qo_smem[workerid])
  #
  # sink = o[batch, q_seq, head, 0].store(q_reg)

  sink = o

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  with Context(DEBUG=0):
    q = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
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
    print(ei.prg)
    times.append(et)
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  print(f"{attn_flops/(min(times)*1e12):2f} TFLOPS")

  print(out.tolist())

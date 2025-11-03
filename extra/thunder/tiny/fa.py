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

zero_rid = 16
def zero(reg:UOp):
  global zero_rid
  i = UOp.range(reg.size, zero_rid)
  zero_rid += 1
  return reg[i].set(0, end=i)

LOAD_INNER = 1
load_rid = 100
def load(dst:UOp, src:UOp, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0):
  assert len(idxs) == len(src.shape)

  global load_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  laneid = threadIdx_x

  src_last_i = idxs[-2] * src.shape[-1] + idxs[-1]

  # permute src so that axis and axis + 1 and last
  perm = list(range(len(src.shape)))
  src_axis = perm.pop(axis)
  src_axis1 = perm.pop(axis)
  perm += [src_axis, src_axis1]
  src = src.permute(tuple(perm))
  src = src.reshape(src.shape[:-2] + (prod(src.shape[-2:]),))

  # flatten dst
  dst = dst.reshape(dst.shape[:-2] + (prod(dst.shape[-2:]),))

  load_i_outer = UOp.range(dst.size // (WARP_THREADS * LOAD_INNER), load_rid)
  load_i_inner = UOp.range(LOAD_INNER, load_rid+1, AxisType.UPCAST)
  load_rid += 2

  dst_i = load_i_outer * (WARP_THREADS * LOAD_INNER) + laneid * LOAD_INNER + load_i_inner
  src_last_i = src_last_i + dst_i

  dst_store = dst[*dst_idxs, dst_i].store(src[*idxs[:-2], src_last_i]).end(load_i_outer, load_i_inner)

  barrier = UOp.barrier(dst_store)

  return dst.after(barrier)

STORE_INNER = 1
store_rid = 200
def store(dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0):
  assert len(idxs) == len(dst.shape)

  global store_rid

  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  laneid = threadIdx_x

  dst_last_i = idxs[-2] * dst.shape[-1] + idxs[-1]

  perm = list(range(len(dst.shape)))
  dst_axis = perm.pop(axis)
  dst_axis1 = perm.pop(axis)
  perm += [dst_axis, dst_axis1]
  dstp = dst.permute(tuple(perm))
  dstp = dstp.reshape(dstp.shape[:-2] + (prod(dstp.shape[-2:]),))

  # flatten src
  src = src.reshape(src.shape[:-2] + (prod(src.shape[-2:]),))

  store_i_outer = UOp.range(dst.size // (WARP_THREADS * STORE_INNER), store_rid)
  store_i_inner = UOp.range(STORE_INNER, store_rid+1, AxisType.UPCAST)
  store_rid += 2

  src_i = store_i_outer * (WARP_THREADS * LOAD_INNER) + laneid * LOAD_INNER + store_i_inner
  dst_last_i = dst_last_i + src_i

  dst_store = dstp[*idxs[:-2], dst_last_i].store(src[*src_idxs, src_i]).end(store_i_outer, store_i_inner)

  return dst_store

WARP_THREADS = 32

NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 1, 32, 1, 64

ROWS = 16 * (128 // D)

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
  qo_smem = st((NUM_WORKERS, ROWS, D), dtypes.bfloat16)

  q_reg = rt((ROWS, D), dtypes.bfloat16)
  k_reg = rt((ROWS, D), dtypes.bfloat16)
  v_reg = rt((D, ROWS), dtypes.bfloat16)
  o_reg = rt((ROWS, D), dtypes.float32)
  att_block = rt((ROWS, ROWS), dtypes.float32)
  att_block_mma = rt((ROWS, ROWS), dtypes.bfloat16)
  max_vec_last = rt((ROWS,), dtypes.float32)
  max_vec = rt((ROWS,), dtypes.float32)
  norm_vec = rt((ROWS,), dtypes.float32)

  q_reg = zero(q_reg)
  o_reg = zero(o_reg)

  outer_kv_rng = UOp.range(N // ROWS, 0)

  qo_smem = load(qo_smem, q, (workerid,), (batch, q_seq, head, 0), axis=1)

  o = store(o, qo_smem, (batch, q_seq, head, 0), (workerid,), axis=1)

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
  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (q, k, v, out)])

  GlobalCounters.reset()
  times = []
  with Context(DEBUG=2):
    for _ in range(5):
      et = ei.run(wait=True)
      times.append(et)
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  print(f"{attn_flops/(min(times)*1e12):2f} TFLOPS")

  print(out.tolist())

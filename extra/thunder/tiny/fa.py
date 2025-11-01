from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

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

def load(dst:UOp, src:UOp, idxs:tuple[UOp,...]=()):
  pass

WARP_THREADS = 32

NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 16, 1024, 16, 64

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

  q_loads_per_thread = (ROWS * D) // WARP_THREADS
  q_loads_per_thread_i = UOp.range(q_loads_per_thread, 16)
  q_glb_seq_i = q_seq * ROWS + (laneid * WARP_THREADS + q_loads_per_thread_i)
  qo_smem_store = qo_smem[workerid, ].store(q[batch, q_glb_seq_i, head, 0]).end()

  barrier = UOp.barrier(qo_smem_store)
  qo_smem = qo_smem.after(barrier)

  q_reg = q_reg.set(qo_smem[workerid]).end(i)

  sink = o[batch, q_seq, head, 0].store(q_reg)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  with Context(DEBUG=0):
    q = Tensor.randn(B, N, H, D, dtype="bfloat16")
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

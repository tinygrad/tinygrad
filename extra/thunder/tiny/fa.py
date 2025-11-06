import math
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.tiles import gl, st, rt, rv
from extra.thunder.tiny.tk.group import Group, warp_, warpgroup_

NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 1, 16, 1, 64

ROWS = 16 * (64 // D)
BLOCK_SIZE=16

def test_ker():
  # define special indices
  blockIdx_x = UOp.special(N // BLOCK_SIZE, "gidx0")
  blockIdx_y = UOp.special(N // BLOCK_SIZE, "gidx1")
  blockIdx_z = UOp.special(1, "gidx2")
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")

  warp = warp_(threadIdx_x)
  warpgroup = warpgroup_(threadIdx_x)

  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)

  # kernel
  c = gl((1, 1, N, N), dtypes.float32)
  a = gl((1, 1, N, N), dtypes.bfloat16)
  b = gl((1, 1, N, N), dtypes.bfloat16)

  a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
  b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
  c_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

  a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
  b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
  c_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

  col, row = blockIdx_x, blockIdx_y

  c_reg = warp.zero(c_reg)

  outer_k_rng = UOp.range(N // BLOCK_SIZE, 0)
  a_smem = warp.load(a_smem, a, (), (0, 0, row, outer_k_rng), axis=2)
  b_smem = warp.load(b_smem, b, (), (0, 0, outer_k_rng, col), axis=2)

  a_reg = warp.load(a_reg, a_smem)
  b_reg = warp.load(b_reg, b_smem)

  c_reg_store = warp.mma_AB(c_reg, a_reg, b_reg, after=False).end(outer_k_rng).barrier()
  c_reg = c_reg.after(c_reg_store).reshape(c_reg.shape)

  c_smem = warp.store(c_smem, c_reg)
  c = warp.store(c, c_smem, (0, 0, row, col), (), axis=2, after=False)

  sink = c
  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def ker():
  # define special indices
  blockIdx_x = UOp.special(N // (ROWS*NUM_WORKERS), "gidx0")
  blockIdx_y = UOp.special(H, "gidx1")
  blockIdx_z = UOp.special(B, "gidx2")
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")

  warp = warp_(threadIdx_x)
  warpgroup = warpgroup_(threadIdx_x)

  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)

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
  max_vec_last = rv(ROWS, dtypes.float32, "ortho")
  max_vec = rv(ROWS, dtypes.float32, "ortho")
  norm_vec = rv(ROWS, dtypes.float32, "ortho")

  max_vec = warp.neg_inf(max_vec)
  norm_vec = warp.zero(norm_vec)
  o_reg = warp.zero(o_reg)

  # load q tile
  qo_smem = warp.load(qo_smem, q, (), (batch, q_seq, head, 0), axis=1)
  q_reg = warp.load(q_reg, qo_smem)

  q_reg = warp.map(q_reg, lambda x: x * ((1.0 / math.sqrt(D)) * (1.0 / math.log(2))))

  outer_kv_rng = UOp.range(N // ROWS, 0)

  k_smem = warp.load(k_smem, k, (), (batch, outer_kv_rng, head, 0), axis=1)
  v_smem = warp.load(v_smem, v, (), (batch, outer_kv_rng, head, 0), axis=1)

  k_reg = warp.load(k_reg, k_smem)
  att_block = warp.zero(att_block).after(outer_kv_rng)
  att_block = warp.mma_ABt(att_block, q_reg, k_reg)

  max_vec_last = warp.copy(max_vec_last, max_vec)
  max_vec = warp.row_reduce(max_vec, att_block, lambda a, b: a.maximum(b))
  att_block = warp.map(att_block, lambda x, idx: (x + max_vec[idx[0], 0] * UOp.const(dtypes.float32, -1.)).exp2())
  max_vec_last = warp.map(max_vec_last, lambda x, idx: (x + max_vec[*idx] * UOp.const(dtypes.float32, -1.)).exp2())
  norm_vec = warp.map(norm_vec, lambda x, idx: x * max_vec_last[*idx])
  norm_vec = warp.row_reduce(norm_vec, att_block, lambda a, b: a + b)

  att_block_mma = warp.copy(att_block_mma, att_block)
  v_reg = warp.load(v_reg, v_smem)

  o_reg = warp.map(o_reg, lambda x, idx: x * max_vec_last[idx[0], 0])
  o_reg_store = warp.mma_AB(o_reg, att_block_mma, v_reg, after=False).end(outer_kv_rng).barrier()
  o_reg = o_reg.after(o_reg_store).reshape(o_reg.shape)

  o_reg = warp.map(o_reg, lambda x, idx: x / norm_vec[idx[0], 0])

  qo_smem = warp.store(qo_smem, o_reg)
  o = warp.store(o, qo_smem, (batch, q_seq, head, 0), (), axis=1, after=False)

  sink = o
  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  with Context(DEBUG=0):
    q = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    v = Tensor.arange(B * N * H * D).reshape(B, N, H, D).cast(dtypes.bfloat16).contiguous()
    k = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    # v = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    out = Tensor.empty(B, N, H, D, dtype="bfloat16")
    Tensor.realize(q, k, v, out)

    # a = Tensor.ones(1, 1, N, N, dtype="bfloat16").contiguous()
    # b = Tensor.ones(1, 1, N, N, dtype="bfloat16").contiguous()
    # c = Tensor.empty(1, 1, N, N, dtype="float32")
    # Tensor.realize(a, b, c)

  sink = ker()
  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (out, q, k, v)])
  # ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (c, a, b)])

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
  # print(c.tolist())

  # ref = q.scaled_dot_product_attention(k, v)
  # print(ref.tolist())


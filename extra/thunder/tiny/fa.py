import math
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import gl, st, rt, rv

NUM_WORKERS = 1
PIPE_STAGES = 3

B, N, H, D = 2, 32, 2, 64

ROWS = 16 * (64 // D)

def fa_ker():
  with Kernel((N // (ROWS*NUM_WORKERS), H, B), NUM_WORKERS * WARP_THREADS) as ker:
    warp = ker.warp

    # kernel
    o = gl((B, N, H, D), dtypes.bfloat16)
    q = gl((B, N, H, D), dtypes.bfloat16)
    k = gl((B, N, H, D), dtypes.bfloat16)
    v = gl((B, N, H, D), dtypes.bfloat16)

    workerid = ker.warpid

    batch, head, q_seq = ker.blockIdx_z, ker.blockIdx_y, ker.blockIdx_x * NUM_WORKERS + workerid

    k_smem = st((ROWS, D), dtypes.bfloat16)
    v_smem = st((ROWS, D), dtypes.bfloat16)
    qo_smem = st((ROWS, D), dtypes.bfloat16)

    q_reg = rt((ROWS, D), dtypes.bfloat16)
    k_reg = rt((ROWS, D), dtypes.bfloat16)
    v_reg = rt((ROWS, D), dtypes.bfloat16)
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

    for kv_idx in ker.range(N // ROWS):
      k_smem = warp.load(k_smem, k, (), (batch, kv_idx, head, 0), axis=1)
      v_smem = warp.load(v_smem, v, (), (batch, kv_idx, head, 0), axis=1)

      k_reg = warp.load(k_reg, k_smem)
      att_block = warp.zero(att_block.after(kv_idx))
      att_block = warp.mma_ABt(att_block, q_reg, k_reg)

      max_vec_last = warp.copy(max_vec_last.after(kv_idx), max_vec)
      max_vec = warp.row_reduce(max_vec, att_block, lambda a, b: a.maximum(b))
      att_block = warp.map(att_block, lambda x, idx: (x - max_vec[idx[0], 0, (idx[2]%4)//2]).exp2())
      max_vec_last = warp.map(max_vec_last, lambda x, idx: (x - max_vec[*idx]).exp2())
      norm_vec = warp.map(norm_vec, lambda x, idx: x * max_vec_last[*idx])
      norm_vec = warp.row_reduce(norm_vec, att_block, lambda a, b: a + b)

      att_block_mma = warp.copy(att_block_mma.after(kv_idx), att_block)
      v_reg = warp.load(v_reg, v_smem, transpose=True)

      o_reg = warp.map(o_reg, lambda x, idx: x * max_vec_last[idx[0], 0, (idx[2]%4)//2])
      o_reg = warp.mma_AB(o_reg, att_block_mma, v_reg)
    o_reg = ker.endrange()

    o_reg = warp.map(o_reg, lambda x, idx: x / norm_vec[idx[0], 0, (idx[2]%4)//2])

    qo_smem = warp.store(qo_smem, o_reg)
    o = warp.store(o, qo_smem, (batch, q_seq, head, 0), (), axis=1)

    return ker.finish()

if __name__ == "__main__":
  with Context(DEBUG=0):
    q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
    k = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
    v = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
    out = Tensor.empty(B, N, H, D, dtype=dtypes.bfloat16)
    Tensor.realize(q, k, v, out)

  sink = fa_ker()
  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (out, q, k, v)])

  GlobalCounters.reset()
  times = []
  for _ in range(5):
    et = ei.run(wait=True)
    times.append(et)
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  print(f"{attn_flops/(min(times)*1e12):3f} TFLOPS")

  out = out.float()

  q_permuted = q.permute(0, 2, 1, 3)
  k_permuted = k.permute(0, 2, 1, 3)
  v_permuted = v.permute(0, 2, 1, 3)
  ref = q_permuted.scaled_dot_product_attention(k_permuted, v_permuted).float()
  ref = ref.permute(0, 2, 1, 3)

  print((ref - out).mean().item(), (ref - out).max().item())

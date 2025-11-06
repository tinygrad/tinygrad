import math
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
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

def rv(length, dtype, layout="naive"):
  tiles = length // TILE_ROW_DIM
  match layout:
    case "naive":
      inner_dim = 1
      outer_dim = (tiles + 1) // 2
    case "ortho":
      inner_dim = 1
      outer_dim = tiles

  global register_slot
  register_slot += 1
  return UOp.placeholder((outer_dim, inner_dim), dtype, addrspace=AddrSpace.REG, slot=register_slot-1)

clear_rid = 1000
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
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  global load_rid
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

  return dst.after(dst_store.barrier()).reshape(dst.shape)

STORE_INNER = 8
store_rid = 200
def store(dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0, after=True):
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  global store_rid
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

copy_rid = 300
def copy(dst:UOp, src:UOp):
  assert dst.shape == src.shape
  assert cast(PtrDType, dst.dtype).addrspace == AddrSpace.REG
  assert cast(PtrDType, src.dtype).addrspace == AddrSpace.REG

  global copy_rid
  rngs_for_shape = tuple(UOp.range(dim, copy_rid + i) for i, dim in enumerate(dst.shape))
  copy_rid += len(dst.shape)

  dst_store = dst[*rngs_for_shape].store(src[*rngs_for_shape].cast(dst.dtype.base)).end(*rngs_for_shape)

  return dst.after(dst_store).reshape(dst.shape)

mma_rid = 600
def mma_AB(c:UOp, a:UOp, b:UOp, after=True):
  global mma_rid
  mma_i_height = UOp.range(c.shape[-3], mma_rid)
  mma_i_width = UOp.range(c.shape[-2], mma_rid+1)
  mma_i_inner = UOp.range(a.shape[-2], mma_rid+2, AxisType.REDUCE)
  mma_rid += 3

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

def mma_ABt(c:UOp, a:UOp, b:UOp, after=True):
  global mma_rid
  mma_i_height = UOp.range(c.shape[-3], mma_rid)
  mma_i_width = UOp.range(c.shape[-2], mma_rid+1)
  mma_i_inner = UOp.range(a.shape[-2], mma_rid+2, AxisType.REDUCE)
  mma_rid += 3

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
def map(a:UOp, op:Callable[[UOp], UOp]|Callable[[UOp, tuple], UOp]):
  global map_rid
  rngs_for_shape = tuple(UOp.range(dim, map_rid + i) for i, dim in enumerate(a.shape))
  map_rid += len(a.shape)

  if op.__code__.co_argcount == 1:
    to_store = op(a[*rngs_for_shape])
  else:
    to_store = op(a[*rngs_for_shape], rngs_for_shape)

  a_store = a[*rngs_for_shape].store(to_store).end(*rngs_for_shape)
  return a.after(a_store).reshape(a.shape)

red_rid = 500
def row_reduce(vec:UOp, src:UOp, op:Callable[[UOp, UOp], UOp]):
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")
  laneid = threadIdx_x % (NUM_WORKERS * WARP_THREADS)

  global red_rid
  red_i_height = UOp.range(src.shape[-3], red_rid)
  red_i_width = UOp.range(src.shape[-2], red_rid+1)
  red_i_inner = UOp.range(RT_BASE_TILE_NEPT, red_rid+2, AxisType.REDUCE)
  red_rid += 3

  global shared_slot
  red_local = UOp.placeholder((NUM_WORKERS * WARP_THREADS,), src.dtype.base, addrspace=AddrSpace.LOCAL, slot=shared_slot)
  shared_slot += 1

  # initial reduce in registers
  vec_store = vec[red_i_height, 0].store(op(vec.after(UOp.group(red_i_width, red_i_inner))[red_i_height, 0], src[red_i_height, red_i_width, red_i_inner])).end(red_i_height, red_i_width, red_i_inner)
  vec = vec.after(vec_store).reshape(vec.shape)

  # store to shared memory
  red_local_store = red_local[laneid].store(vec[red_i_height, 0])
  red_local = red_local.after(red_local_store).reshape(red_local.shape)

  # final reduce from shared memory
  offset = (laneid + 16) % 32
  red_local_i = (offset // 16) * 16 + (offset % 16)
  vec_store = vec[red_i_height, 0].store(op(vec[red_i_height, 0], red_local[red_local_i])).end(red_i_height)

  return vec.after(vec_store).reshape(vec.shape)

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

  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS % WARP_THREADS)

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

  c_reg = zero(c_reg)

  outer_k_rng = UOp.range(N // BLOCK_SIZE, 0)
  a_smem = load(a_smem, a, (), (0, 0, row, outer_k_rng), axis=2)
  b_smem = load(b_smem, b, (), (0, 0, outer_k_rng, col), axis=2)

  a_reg = load(a_reg, a_smem)
  b_reg = load(b_reg, b_smem)

  c_reg_store = mma_AB(c_reg, a_reg, b_reg, after=False).end(outer_k_rng).barrier()
  c_reg = c_reg.after(c_reg_store).reshape(c_reg.shape)

  c_smem = store(c_smem, c_reg)
  c = store(c, c_smem, (0, 0, row, col), (), axis=2, after=False)

  sink = c
  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def ker():
  # define special indices
  blockIdx_x = UOp.special(N // (ROWS*NUM_WORKERS), "gidx0")
  blockIdx_y = UOp.special(H, "gidx1")
  blockIdx_z = UOp.special(B, "gidx2")
  threadIdx_x = UOp.special(NUM_WORKERS * WARP_THREADS, "lidx0")

  warpid = threadIdx_x // (NUM_WORKERS * WARP_THREADS)
  laneid = threadIdx_x % (NUM_WORKERS % WARP_THREADS)

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

  max_vec = neg_inf(max_vec)
  norm_vec = zero(norm_vec)
  o_reg = zero(o_reg)

  # load q tile
  qo_smem = load(qo_smem, q, (), (batch, q_seq, head, 0), axis=1)
  q_reg = load(q_reg, qo_smem)

  q_reg = map(q_reg, lambda x: x * ((1.0 / math.sqrt(D)) * (1.0 / math.log(2))))

  outer_kv_rng = UOp.range(N // ROWS, 0)

  k_smem = load(k_smem, k, (), (batch, outer_kv_rng, head, 0), axis=1)
  v_smem = load(v_smem, v, (), (batch, outer_kv_rng, head, 0), axis=1)

  k_reg = load(k_reg, k_smem)
  att_block = zero(att_block).after(outer_kv_rng)
  att_block = mma_ABt(att_block, q_reg, k_reg)

  max_vec_last = copy(max_vec_last, max_vec)
  max_vec = row_reduce(max_vec, att_block, lambda a, b: a.maximum(b))
  att_block = map(att_block, lambda x, idx: (x + max_vec[idx[0], 0] * UOp.const(dtypes.float32, -1.)).exp2())
  max_vec_last = map(max_vec_last, lambda x, idx: (x + max_vec[*idx] * UOp.const(dtypes.float32, -1.)).exp2())
  norm_vec = map(norm_vec, lambda x, idx: x * max_vec_last[*idx])
  norm_vec = row_reduce(norm_vec, att_block, lambda a, b: a + b)

  att_block_mma = copy(att_block_mma, att_block)
  v_reg = load(v_reg, v_smem)

  o_reg = map(o_reg, lambda x, idx: x * max_vec_last[idx[0], 0])
  o_reg = mma_AB(o_reg, att_block_mma, v_reg)

  o_reg = o_reg.after(o_reg.end(outer_kv_rng)).reshape(o_reg.shape)

  o_reg = map(o_reg, lambda x, idx: x / norm_vec[idx[0], 0])

  qo_smem = store(qo_smem, o_reg)
  o = store(o, qo_smem, (batch, q_seq, head, 0), (), axis=1, after=False)

  sink = o
  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  with Context(DEBUG=0):
    # q = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    # v = Tensor.arange(B * N * H * D).reshape(B, N, H, D).cast(dtypes.bfloat16).contiguous()
    # k = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    # # v = Tensor.ones(B, N, H, D, dtype="bfloat16").contiguous()
    # out = Tensor.empty(B, N, H, D, dtype="bfloat16")
    # Tensor.realize(q, k, v, out)

    a = Tensor.ones(1, 1, N, N, dtype="bfloat16").contiguous()
    b = Tensor.ones(1, 1, N, N, dtype="bfloat16").contiguous()
    c = Tensor.empty(1, 1, N, N, dtype="float32")
    Tensor.realize(a, b, c)

  sink = test_ker()
  # ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (out, q, k, v)])
  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (c, a, b)])

  GlobalCounters.reset()
  times = []
  for _ in range(5):
    et = ei.run(wait=True)
    times.append(et)
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  print(f"{attn_flops/(min(times)*1e12):2f} TFLOPS")

  # print(out.tolist())
  print(c.tolist())

  # ref = q.scaled_dot_product_attention(k, v)
  # print(ref.tolist())


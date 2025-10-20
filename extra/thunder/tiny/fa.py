from tinygrad import Device, Tensor, Context
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, AxisType, PatternMatcher, UPat, pm_lower_index_dtype, GroupOp, KernelInfo
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import prod
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.codegen.simplify import pm_flatten_range

TILE_DIM = 32

# tinykittens
range_num = 0
def rng(x, typ=AxisType.LOOP) -> UOp:
  global range_num
  range_num += 1
  return UOp.range(x, range_num-1, typ)

def glbl(nm, dtype, sz): return UOp(Ops.DEFINE_GLOBAL, dtype.ptr(prod(sz), AddrSpace.GLOBAL), arg=nm).reshape(sz)
def st(nm, dtype, sz): return UOp(Ops.DEFINE_LOCAL, dtype.ptr(prod(sz), AddrSpace.LOCAL), arg=nm).reshape(sz)
def rt(nm, dtype, sz): return UOp(Ops.DEFINE_REG, dtype.ptr(prod(sz), AddrSpace.REG), arg=nm).reshape(sz)

def zero(reg:UOp, *endrngs):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  return reg[*rngs].store(UOp.const(reg.dtype.base, 0.0), *rngs, *endrngs, dtype=reg.dtype).reshape(reg.shape)

def load(reg:UOp, gl:UOp, idxs=(), reg_idxs=(), reshape=True, barriers=()):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  rngs = list(reg_idxs) + rngs
  sto = reg[*rngs].store(gl[*grngs].load(*barriers), *rngs, dtype=reg.dtype)
  barrier = sto.barrier()
  if reshape: sto = sto.reshape(reg.shape)
  return sto, barrier

def store(gl:UOp, reg:UOp, idxs=(), gl_idxs=(), reshape=False, barriers=()):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  grngs = list(gl_idxs) + grngs
  sto = gl[*grngs].store(reg[*rngs].load(*barriers), *rngs, dtype=gl.dtype)
  barrier = sto.barrier()
  if reshape: sto = sto.reshape(gl.shape)
  return sto, barrier

def mma_AB(outacc:UOp, a:UOp, b:UOp, *endrngs):
  assert a.shape[1] == b.shape[0]
  # meta::unroll_i_j_in_range -- split on TILE_DIM
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in outacc.shape]
  red = rng(a.shape[1]//TILE_DIM, AxisType.REDUCE)*TILE_DIM
  # meta::unroll_i_in_range -- split reduce on TILE_DIM
  rngs = [x+rng(TILE_DIM) for x in rngs]
  red = red + rng(TILE_DIM, AxisType.REDUCE)
  acc = outacc[*rngs].load(red) + a[rngs[0],red].load() * b[red,rngs[1]].load()
  return outacc[*rngs].store(acc, *rngs, red, *endrngs, dtype=outacc.dtype).reshape(outacc.shape)

def mma_ABt(outacc:UOp, a:UOp, b:UOp, *endrngs):
  assert a.shape[1] == b.shape[1]
  # meta::unroll_i_j_in_range -- split on TILE_DIM
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in outacc.shape]
  red = rng(a.shape[1]//TILE_DIM, AxisType.REDUCE)*TILE_DIM
  # meta::unroll_i_in_range -- split reduce on TILE_DIM
  rngs = [x+rng(TILE_DIM) for x in rngs]
  red = red + rng(TILE_DIM, AxisType.REDUCE)
  acc = outacc[*rngs].load(red) + a[rngs[0],red].load() * b[red,rngs[1]].load()
  return outacc[*rngs].store(acc, *rngs, red, *endrngs, dtype=outacc.dtype).reshape(outacc.shape)

WARP_THREADS = 32

NUM_WORKERS = 4
PIPE_STAGES = 3
LOAD_BLOCKS = NUM_WORKERS // 2

B, N, H, D = 16, 1024, 16, 64

if __name__ == "__main__":
  block_id_z = UOp.range(B, -4, AxisType.GLOBAL)
  block_id_y = UOp.range(H, -3, AxisType.GLOBAL)
  block_id_x = UOp.range((N + TILE_DIM*NUM_WORKERS - 1) // (TILE_DIM*NUM_WORKERS), -2, AxisType.GLOBAL)

  thread_id_x = UOp.range(WARP_THREADS * NUM_WORKERS, -1, AxisType.LOCAL)

  lane_id = thread_id_x % (2 * WARP_THREADS)
  warp_id = lane_id // WARP_THREADS

  batch = block_id_z
  head = block_id_y
  q_seq = block_id_x * NUM_WORKERS

  gl_o = glbl("gl0_o", dtypes.bfloat16, (B, N, H, D))
  gl_q = glbl("gl1_q", dtypes.bfloat16, (B, N, H, D))
  gl_k = glbl("gl2_k", dtypes.bfloat16, (B, N, H, D))
  gl_v = glbl("gl3_v", dtypes.bfloat16, (B, N, H, D))

  k_smem = st("k_smem", dtypes.bfloat16, (LOAD_BLOCKS, PIPE_STAGES, TILE_DIM, D))
  v_smem = st("v_smem", dtypes.bfloat16, (LOAD_BLOCKS, PIPE_STAGES, TILE_DIM, D))
  qo_smem = st("qo_smem", dtypes.bfloat16, (NUM_WORKERS, TILE_DIM, D))

  q_reg = rt("q_reg", dtypes.bfloat16, (TILE_DIM, D))
  k_reg = rt("k_reg", dtypes.bfloat16, (TILE_DIM, D))
  v_reg = rt("v_reg", dtypes.bfloat16, (D, TILE_DIM))
  o_reg = rt("o_reg", dtypes.float, (TILE_DIM, D))
  att_block = rt("att_block", dtypes.float, (TILE_DIM, TILE_DIM))
  att_block_mma = rt("att_block_mma", dtypes.bfloat16, (TILE_DIM, TILE_DIM))

  # qo_smem, barrier = load(qo_smem, gl_q, (batch, q_seq, head), (warp_id,))
  # q_reg, barrier = load(q_reg, qo_smem, (warp_id,), barriers=(barrier,))
  #
  # qo_smem, barrier = store(qo_smem, q_reg, (), (warp_id,), reshape=True, barriers=(barrier,))
  # sink, barrier = store(gl_o, qo_smem, (warp_id, batch, q_seq, head), (), barriers=(barrier,))

  q_reg, barrier = load(q_reg, gl_q, (batch, q_seq, head), ())
  sink, barrier = store(gl_o, q_reg, (batch, q_seq, head), (), barriers=(barrier,))

  sink = graph_rewrite(sink, pm_mops+pm_flatten_range, name="pm_mops")

  from tinygrad.codegen.gpudims import pm_add_gpudims
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=Device.default.renderer, name="gpudims")

  pm_lower_index_dtype_simple = PatternMatcher([
    (UPat(GroupOp.All, dtype=dtypes.index, name="x"), lambda x: x.replace(dtype=dtypes.int))
  ])
  sink = graph_rewrite(sink, pm_lower_index_dtype_simple, name="index_dtype")

  from tinygrad.codegen import rewrites_for_linearizer, apply_rewrites
  lin = apply_rewrites(sink, rewrites_for_linearizer)
  src = Device.default.renderer.render(lin.arg.lst)
  print(src)

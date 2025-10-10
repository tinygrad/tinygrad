from tinygrad import Device, Tensor, Context
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, AxisType, PatternMatcher, UPat, pm_lower_index_dtype, GroupOp, KernelInfo
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import prod
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.codegen.simplify import pm_flatten_range

TILE_DIM = 8
N_BLOCK = 4
K_BLOCK = 2
M_BLOCK = 4

#M = N = K = 4096
M = N = K = 1024

range_num = 0
def rng(x, typ=AxisType.LOOP) -> UOp:
  global range_num
  range_num += 1
  return UOp.range(x, range_num-1, typ)

def glbl(nm, dtype, sz): return UOp(Ops.DEFINE_GLOBAL, dtype.ptr(prod(sz), AddrSpace.GLOBAL), arg=nm).reshape(sz)
def rt(nm, dtype, sz): return UOp(Ops.DEFINE_REG, dtype.ptr(prod(sz), AddrSpace.REG), arg=nm).reshape(sz)

def zero(reg:UOp, *endrngs):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  return reg[*rngs].store(UOp.const(reg.dtype.base, 0.0), *rngs, *endrngs, dtype=reg.dtype).reshape(reg.shape)

def load(reg:UOp, gl:UOp, *idxs):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  return reg[*rngs].store(gl[*grngs].load(), *rngs, dtype=reg.dtype).reshape(reg.shape)

def store(gl:UOp, reg:UOp, *idxs):
  rngs = [rng(s//TILE_DIM)*TILE_DIM for s in reg.shape]
  rngs = [x+rng(TILE_DIM) for x in rngs]

  # TODO: why does this not have shape?
  #rngs = [rng(s) for s in (N_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM)]
  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  return gl[*grngs].store(reg[*rngs].load(), *rngs)

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

if __name__ == "__main__":
  # TODO: support string ranges
  tg_id_y = UOp.range(M // (M_BLOCK * TILE_DIM), -3, AxisType.GLOBAL if Device.DEFAULT != "CPU" else AxisType.LOOP)
  tg_id_x = UOp.range(N // (N_BLOCK * TILE_DIM), -2, AxisType.GLOBAL if Device.DEFAULT != "CPU" else AxisType.LOOP)

  gl_d = glbl("gl0_d", dtypes.float, (N, M))
  gl_a = glbl("gl1_a", dtypes.float, (N, K))
  gl_b = glbl("gl2_b", dtypes.float, (K, M))

  a_reg = rt("a_reg", dtypes.float, (N_BLOCK*TILE_DIM, K_BLOCK*TILE_DIM))
  b_reg = rt("b_reg", dtypes.float, (K_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM))
  d_reg = rt("d_reg", dtypes.float, (N_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM))
  d_reg = zero(d_reg, tg_id_y, tg_id_x)

  k = UOp.range(K // (K_BLOCK * TILE_DIM), -1, AxisType.REDUCE)
  a_reg = load(a_reg, gl_a, tg_id_y, k)
  b_reg = load(b_reg, gl_b, k, tg_id_x)
  d_reg = mma_AB(d_reg, a_reg, b_reg, k)
  sink = store(gl_d, d_reg, tg_id_y, tg_id_x).sink(arg=KernelInfo())

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
  #exit(0)

  from tinygrad.engine.realize import CompiledRunner, ExecItem
  from tinygrad.renderer import ProgramSpec

  ps = ProgramSpec("test", src, Device.DEFAULT, sink, lin.arg.lst, [1,1,1], [1,1,1])
  run = CompiledRunner(ps)

  a = Tensor.randn(N, N)
  b = Tensor.randn(N, N)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  ei = ExecItem(run, [x.uop.buffer.ensure_allocated() for x in (c,a,b)])
  with Context(DEBUG=2):
    for i in range(5): ei.run()
    for i in range(5): ref = (a@b).realize()
  print((ref-c).mean().item())

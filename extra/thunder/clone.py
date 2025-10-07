from tinygrad import Device
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, PatternMatcher, AxisType
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import prod
from tinygrad.schedule.rangeify import pm_mops

TILE_DIM = 8
N_BLOCK = 4
K_BLOCK = 2
M_BLOCK = 4

M = N = K = 4096

range_num = 0
def rng(x, typ=AxisType.LOOP) -> UOp:
  global range_num
  range_num += 1
  return UOp.range(x, range_num-1, typ)

def glbl(nm, dtype, sz): return UOp(Ops.DEFINE_GLOBAL, dtype.ptr(prod(sz), AddrSpace.GLOBAL), arg=nm).reshape(sz)
def rt(nm, dtype, sz): return UOp(Ops.DEFINE_REG, dtype.ptr(prod(sz), AddrSpace.REG), arg=nm).reshape(sz)

def zero(reg:UOp):
  rngs = [rng(s) for s in reg.shape]
  return reg[*rngs].store(UOp.const(reg.dtype.base, 0.0), *rngs)

def load(reg:UOp, gl:UOp, *idxs):
  rngs = [rng(s) for s in reg.shape]
  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  return reg[*rngs].store(gl[*grngs].load(), *rngs) #.forced_reshape(reg.shape)

def store(gl:UOp, reg:UOp, *idxs):
  #rngs = [rng(s) for s in reg.shape]
  # TODO: why does this not have shape?
  rngs = [rng(s) for s in (N_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM)]
  grngs = [i*(r.vmax+1)+r for i,r in zip(idxs,rngs)]
  return gl[*grngs].store(reg[*rngs].load(), *rngs) #.forced_reshape(reg.shape)

def mma_AB(outacc:UOp, a:UOp, b:UOp, *endrngs):
  assert a.shape[1] == b.shape[0]
  rngs = [rng(s) for s in outacc.shape]
  red = rng(a.shape[1], AxisType.REDUCE)
  return outacc[*rngs].store(outacc[*rngs].load(red) + a[rngs[0],red]*b[red,rngs[1]], *rngs, *endrngs, red)

if __name__ == "__main__":
  # TODO: support string ranges
  tg_id_y = UOp.range(M // (M_BLOCK * TILE_DIM), -3, AxisType.GLOBAL)
  tg_id_x = UOp.range(N // (N_BLOCK * TILE_DIM), -2, AxisType.GLOBAL)

  gl_a = glbl("gl_a", dtypes.float, (N, K))
  gl_b = glbl("gl_b", dtypes.float, (K, M))
  gl_d = glbl("gl_d", dtypes.float, (N, M))

  a_reg = rt("a_reg", dtypes.float, (N_BLOCK*TILE_DIM, K_BLOCK*TILE_DIM))
  b_reg = rt("b_reg", dtypes.float, (K_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM))
  d_reg = rt("d_reg", dtypes.float, (N_BLOCK*TILE_DIM, M_BLOCK*TILE_DIM))
  d_reg = zero(d_reg)

  k = UOp.range(K // (K_BLOCK * TILE_DIM), -1, AxisType.REDUCE)
  a_reg = load(a_reg, gl_a, tg_id_y, k)
  b_reg = load(b_reg, gl_b, k, tg_id_x)
  d_reg = mma_AB(d_reg, a_reg, b_reg, k)

  sink = store(gl_d, d_reg, tg_id_y, tg_id_x).sink()
  sink = graph_rewrite(sink, pm_mops)

  #from tinygrad.codegen import full_rewrite
  #lst = full_rewrite(sink)

  from tinygrad.codegen import rewrites_for_linearizer, apply_rewrites
  lin = apply_rewrites(sink, rewrites_for_linearizer)
  src = Device.default.renderer.render(lin.arg.lst)
  print(src)



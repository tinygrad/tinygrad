from tinygrad import UOp, dtypes
from tinygrad.uop.ops import AxisType, Ops, KernelInfo, AddrSpace
from extra.gemm.amd_uop_matmul import test_matmul

N = 2048

# metal has an 8x8 tensor core

def hand_spec_tc_cores():
  gx = UOp.special(N // 8, "gidx0")
  gy = UOp.special(N // 8, "gidx1")
  warp = UOp.special(32, "lidx0")

  c = UOp.placeholder(dtypes.float, (N, N), slot=0).reshape((N//8, 2, 2, 2, N//8, 2, 2, 2))
  a = UOp.placeholder(dtypes.float, (N, N), slot=1).reshape((N//8, 2, 2, 2, N//8, 2, 2, 2))
  b = UOp.placeholder(dtypes.float, (N, N), slot=2).reshape((N//8, 2, 2, 2, N//8, 2, 2, 2))

  gk = UOp.range(N // 8, 0, AxisType.REDUCE)

  # indexing for tensor cores
  l = [(warp//2**i)%2 for i in range(5)]
  def mat_idx(g0, g1, u): return [g0, l[4], l[2], l[1], g1, l[3], l[0], u]

  a_tc = UOp.vectorize(*[a[*mat_idx(gx, gk, i)] for i in range(2)])
  b_tc = UOp.vectorize(*[b[*mat_idx(gk, gy, i)] for i in range(2)])

  acc = UOp.placeholder(dtypes.float, (2,), slot=0, addrspace=AddrSpace.REG)
  acc = acc[0].set(0.0)
  acc = acc[1].set(0.0)

  # TODO: make this simple
  wmma_arg = ('WMMA_8_8_8_float_float', (8, 8, 8), dtypes.float, dtypes.float, 'METAL', 32, (((3, 2),), ((3, 2),), ((3, 2),)), ())

  acc_load = UOp.vectorize(acc.after(gk)[0], acc.after(gk)[1])
  out = UOp(Ops.WMMA, dtypes.float.vec(2), (a_tc, b_tc, acc_load), arg=wmma_arg)

  sink = UOp.group(*[acc[i].store(out.gep(i)) for i in range(2)]).end(gk)

  sink = UOp.group(*[c.after(sink)[*mat_idx(gx, gy, i)].store(acc[i]) for i in range(2)])
  return sink.sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  test_matmul(hand_spec_tc_cores(), N=N)

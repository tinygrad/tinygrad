import os
import numpy as np
np.set_printoptions(linewidth=1000000)
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor, Context, dtypes, UOp
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

N = 16
M = 16
K = 32

N,M,K = 256,256,64

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  # A = (M x K)
  # B = (K x N)
  # C = (M x N)

  # check it's proper matmul
  assert C.shape[0] == A.shape[0]
  assert C.shape[1] == B.shape[1]
  assert A.shape[1] == B.shape[0]

  print(C.shape, A.shape, B.shape)

  # split out the tensor core (checks divisbility too)
  A = A.reshape((M//16, 16, K//32, 32))
  B = B.reshape((K//32, 32, N//16, 16))
  C = C.reshape((M//16, 16, N//16, 16))

  K_loop = UOp.range(K//32, 0, AxisType.REDUCE)

  gx, gy = UOp.special(M//16, "gidx0"), UOp.special(N//16, "gidx1")
  warp = UOp.special(64, "lidx0")

  A_in = UOp.vectorize(*[A[gx, warp%16, K_loop, (warp//16)*8+i] for i in range(8)])
  B_in = UOp.vectorize(*[B[K_loop, (warp//16)*8+i, gy, warp%16] for i in range(8)])

  # init the acc
  acc = UOp.placeholder((4,), dtypes.float, 0, AddrSpace.REG)
  acc = acc[init_l:=UOp.range(4, 1)].set(0.0, end=init_l)

  # do the wmma
  acc_load = UOp.vectorize(*[acc.after(K_loop)[i] for i in range(4)])
  wmma_arg = ('WMMA_16_16_32_half_float', (16, 16, 32), dtypes.half, dtypes.float, 'AMD', 64, ((), (), ((3, 2), (2, 2))), ())
  out = UOp(Ops.WMMA, dtypes.float.vec(4), (A_in, B_in, acc_load), arg=wmma_arg)

  # store back the acc
  acc = acc.after(UOp.group(*[acc[i].store(out.gep(i)) for i in range(4)]).end(K_loop))

  store = UOp.group(*[C[gx, (warp//16)*4+i, gy, warp%16].store(acc[i]) for i in range(4)])
  #store = UOp.group(*[C[gx, (warp//16)*4+i, gy, warp%16].store(out.gep(i)) for i in range(4)])
  return store.sink(arg=KernelInfo(name="custom_gemm", opts_to_apply=()))

if __name__ == "__main__":
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a,b)

  ref = a.dot(b, dtype=dtypes.float)
  ref.realize()

  tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
  tst.realize()

  with Context(DEBUG=0):
    assert Tensor.isclose(ref, tst, atol=1e-2).all().item(), "matrix not close"

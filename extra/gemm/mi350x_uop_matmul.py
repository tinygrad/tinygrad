import os
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor, Context, dtypes, UOp
from tinygrad.uop.ops import AxisType, KernelInfo

N = 16
M = 16
K = 32

#N,M,K = 4,4,4

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  # check it's proper matmul
  assert C.shape[0] == A.shape[0]
  assert C.shape[1] == B.shape[1]
  assert A.shape[1] == B.shape[0]

  #print(C, A, B)
  #print(C.shape, A.shape, B.shape)

  i, j = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1)
  k = UOp.range(A.shape[1], 2, AxisType.REDUCE)

  # zero out output matrix
  C = C[i,j].set(0)

  # do matmul
  C = C[i,j].set(C.after(k)[i,j] + (A[i,k] * B[k,j]).cast(dtypes.float))

  # end loops
  ast = C.end(i, j, k).sink(arg=KernelInfo(name="custom_gemm", opts_to_apply=()))
  return ast

if __name__ == "__main__":
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a,b)

  ref = a@b
  ref.realize()

  tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
  tst.realize()

  #print(ref.numpy())
  #print(tst.numpy())
  #print(Tensor.isclose(ref, tst, atol=1e-2).numpy())

  with Context(DEBUG=0):
    assert Tensor.isclose(ref, tst, atol=1e-2).all().item(), "matrix not close"




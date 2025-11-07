import os
import numpy as np
np.set_printoptions(linewidth=1000000)
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor, Context, dtypes, UOp, GlobalCounters
from tinygrad.helpers import DEBUG, getenv
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, AxisType, KernelInfo, Ops

WARP_SIZE = 64

# Reg tile sizes (tensor cores)
TC_M = 16
TC_N = 16
TC_K = 32

N,M,K = 4096,4096,4096

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64

WARPGROUP_SIZE = 1
BLOCK_M = BLOCK_M * WARPGROUP_SIZE

TID_SIZE = WARPGROUP_SIZE*WARP_SIZE

def copy(dest:UOp, src:UOp, rng:int, set=False, upcast=()):
  assert dest.shape == src.shape
  rngs = [UOp.range(s, rng+i, AxisType.UPCAST if i in upcast else AxisType.LOOP) for i,s in enumerate(src.shape)]
  copy = dest[*rngs].store(src[*rngs]).end(*rngs)
  return dest.after(copy) if set else copy

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  gx, gy = UOp.special(M//BLOCK_M, "gidx0"), UOp.special(N//BLOCK_N, "gidx1")
  K_outer_loop = UOp.range(K//BLOCK_K, 0, AxisType.REDUCE)

  # split out the globals into blocks
  C = C.reshape((M//BLOCK_M, BLOCK_M, N//BLOCK_N, BLOCK_N))
  A = A.reshape((M//BLOCK_M, BLOCK_M, K//BLOCK_K, BLOCK_K))[gx, :, K_outer_loop, :]
  B = B.reshape((K//BLOCK_K, BLOCK_K, N//BLOCK_N, BLOCK_N))[K_outer_loop, :, gy, :]

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  tid = UOp.special(TID_SIZE, "lidx0")

  A_view = A.reshape(-1, TID_SIZE, 8)
  B_view = B.reshape(-1, TID_SIZE, 8)

  # A: read BM x BK tiles (permute on store into locals)
  #BM_As_stride = (BLOCK_M + 1)
  #As = UOp.placeholder((BLOCK_K, BM_As_stride), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_M))
  #As_view = As.permute(1,0).reshape(-1, TID_SIZE, 8)
  As = UOp.placeholder((BLOCK_M, BLOCK_K), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to(BLOCK_K, BLOCK_N)
  As_view = As.reshape(-1, TID_SIZE, 8)

  Bs = UOp.placeholder((BLOCK_K, BLOCK_N+4), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to(BLOCK_K, BLOCK_N)
  Bs_view = Bs.reshape(-1, TID_SIZE, 8)

  outer_copy = UOp.range(A_view.shape[0], 100)
  inner_copy = UOp.range(A_view.shape[2], 101, AxisType.UPCAST)
  As_store = As_view[outer_copy, tid, inner_copy].store(A_view[outer_copy, tid, inner_copy])
  Bs_store = Bs_view[outer_copy, tid, inner_copy].store(B_view[outer_copy, tid, inner_copy])

  As_store = As[0,0].store(0)
  #Bs_store = Bs[0,0].store(0)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(UOp.group(As_store, Bs_store).end(outer_copy, inner_copy))
  #As, Bs = As.after(barrier), Bs.after(barrier)

  sink = barrier.end(K_outer_loop)

  sink = C.after(sink)[0,0,0,0].store(As[0,0]+Bs[0,0])

  #return UOp.sink(As, Bs)
  return sink.sink(arg=KernelInfo(name="custom_gemm", opts_to_apply=())).simplify()

if __name__ == "__main__":
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a,b)


  GlobalCounters.reset()
  with Context(DEBUG=max(2, DEBUG.value), DEVECTORIZE=2):
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    tst.realize()
  print(f"{(N*M*K*2 / GlobalCounters.time_sum_s)*1e-12:.2f} REAL TFLOPS")


  with Context(DEBUG=0):
    ref = a.dot(b, dtype=dtypes.float)
    ref.realize()
    #print(ref.numpy())
    #print(tst.numpy())
    assert Tensor.isclose(ref, tst, atol=1e-2).all().item(), "matrix not close"

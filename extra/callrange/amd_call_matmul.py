from typing import Callable
from tinygrad import UOp, dtypes, Device, Tensor, getenv, function
from tinygrad.uop.ops import AxisType, AddrSpace

def simple_function(fxn:Callable[..., UOp]) -> Callable[..., UOp]:
  def wrapper(*args:UOp) -> UOp:
    params:list[UOp] = [x.param_like(i) for i,x in enumerate(args)]
    return fxn(*params).call(*args)
  return wrapper

N = getenv("N", 4096)
M = K = N

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_N = 128   # columns of C (N-dim) per block
BLOCK_M = 128   # rows of C (M-dim) per block
BLOCK_K = 8     # K-slice per block iteration

@simple_function
def slice_matmul(c_regs, a_local, b_local):
  # 128x128, 8x128, 8x128
  pass

@simple_function
def local_matmul(c:UOp, a:UOp, b:UOp):
  # accumulate in registers
  a_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  b_local = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  c_regs = UOp.placeholder(c.shape, dtypes.float, slot=2, addrspace=AddrSpace.REG)
  k_tile = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)*BLOCK_K

  # assign = store + after
  slice_matmul(c_regs.assign(0), a_local.assign(a[:, k_tile:k_tile+BLOCK_K].permute(1,0)), b_local.assign(b[k_tile:k_tile+BLOCK_K, :]))
  return c.store(c_regs)

@simple_function
def global_matmul(c, a, b):
  gx = UOp.range(N//BLOCK_N, 0, AxisType.GLOBAL) * BLOCK_N
  gy = UOp.range(M//BLOCK_M, 0, AxisType.GLOBAL) * BLOCK_M
  return local_matmul(c[gx:gx+BLOCK_N, gy:gy+BLOCK_M], a[gx:gx+BLOCK_N, :], b[:, gy:gy+BLOCK_M])

if __name__ == "__main__":
  C = UOp.new_buffer(Device.DEFAULT, N*M, dtypes.float).reshape(N,M)
  A = UOp.new_buffer(Device.DEFAULT, N*K, dtypes.float).reshape(N,K)
  B = UOp.new_buffer(Device.DEFAULT, K*M, dtypes.float).reshape(K,M)
  global_matmul(C, A, B).realize()

  # input matmuls
  #c = UOp.param(0, dtypes.float, (N, M))
  #a = UOp.param(1, dtypes.float, (N, K))
  #b = UOp.param(2, dtypes.float, (K, M))


  #ba = a.rearrange("(n bn) (k bk) -> n k bn bk", bn=BLOCK_N, bk=BLOCK_K)[gx, k_tile_range]
  #bb = b.rearrange("(k bk) (m bm) -> k m bk bm", bk=BLOCK_K, bm=BLOCK_M)[k_tile_range, gy]
  #bc = c.rearrange("(n bn) (m bm) -> n m bn bm", bn=BLOCK_N, bm=BLOCK_M)[gx, gy]

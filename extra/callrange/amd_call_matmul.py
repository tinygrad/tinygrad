from typing import Callable
from tinygrad import UOp, dtypes, Device, Tensor, getenv, function
from tinygrad.uop.ops import AxisType, AddrSpace

def simple_function(fxn:Callable[..., UOp]) -> Callable[..., UOp]:
  def wrapper(*args:UOp) -> UOp:
    params:list[UOp] = [x.param_like(i) for i,x in enumerate(args)]
    return fxn(*params).call(*args)
  return wrapper

THREADS_PER_BLOCK = 128
WARP_SIZE = 32

# Register tile sizes (per-thread accumulator tile of C)
TN = 4     # columns per thread
TM = 4     # rows per thread

WAVE_TILE_N = 128
WAVE_TILE_M = 32

LANES_PER_WAVE_X = 8
LANES_PER_WAVE_Y = 4
ITERS_PER_WAVE_N = 4 #WAVE_TILE_N // (LANES_PER_WAVE_X * TN)
ITERS_PER_WAVE_M = 2 #WAVE_TILE_M // (LANES_PER_WAVE_Y * TM)

WAVES_IN_BLOCK_Y = 4
WAVES_IN_BLOCK_X = 1


N = getenv("N", 4096)
M = K = N

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_N = 128   # columns of C (N-dim) per block
BLOCK_M = 128   # rows of C (M-dim) per block
BLOCK_K = 8     # K-slice per block iteration

@simple_function
def slice_matmul(c_regs, a_local, b_local):
  # 2x
  A_col = UOp.placeholder((ITERS_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder((ITERS_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)


  pass

@simple_function
def compute_local(c:UOp, a_local:UOp, b_local:UOp) -> UOp:
  # this is the LID level on the GPU, here we can define regs
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
  waveIdx = (tid // WARP_SIZE) % WAVES_IN_BLOCK_X
  waveIdy = (tid // WARP_SIZE) // WAVES_IN_BLOCK_X
  assert waveIdy.vmax+1 == WAVES_IN_BLOCK_Y

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WAVE_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WAVE_X
  assert laneIdy.vmax+1 == LANES_PER_WAVE_Y

  A_col = UOp.placeholder((ITERS_PER_WAVE_M*TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder((ITERS_PER_WAVE_N*TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)

  # do the math
  A_col = A_col.assign(a_local[k_tile].reshape(WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM)[waveIdy, :, laneIdy, :].flatten())
  B_row = B_row.assign(b_local[k_tile].reshape(WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN)[waveIdx, :, laneIdx, :].flatten())
  c_regs += A_col.reshape(-1, 1) * B_row.reshape(1, -1)  #
  c_regs



@simple_function
def load_local(a_local, b_local, a_global, b_global):
  # NOTE: it ends this range, so there's a BARRIER
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
  return UOp.group(
    a_local[:, tid].store(a_global[tid, :]),
    b_local[:, tid].store(b_global[:, tid]))

@simple_function
def reg_matmul(c_regs, a_local, b_local):
  A_col = UOp.placeholder((ITERS_PER_WAVE_M*TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder((ITERS_PER_WAVE_N*TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)



@simple_function
def local_matmul(c:UOp, a:UOp, b:UOp, a_local:UOp, b_local:UOp):
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
  waveIdx = (tid // WARP_SIZE) % WAVES_IN_BLOCK_X
  waveIdy = (tid // WARP_SIZE) // WAVES_IN_BLOCK_X
  laneIdx = (tid % WARP_SIZE) % LANES_PER_WAVE_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WAVE_X

  # this is the LID level on the GPU, this (and below) is where we define REGs
  c_regs = UOp.placeholder((ITERS_PER_WAVE_M*TM, ITERS_PER_WAVE_N*TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)

  # 128x128, Kx128, Kx128
  k_tile = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)*BLOCK_K
  fxn = reg_matmul(c_regs.assign(0),
                   a_local[:, tid].assign(a[k_tile:k_tile+BLOCK_K, tid]),
                   b_local[:, tid].assign(b[k_tile:k_tile+BLOCK_K, tid]))

  # do math
  c = c.reshape(WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM,
                WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN)
  return c[waveIdy, :, laneIdy, :, waveIdx, :, laneIdx, :].store(c_regs.after(fxn))

@simple_function
def global_matmul(c:UOp, a:UOp, b:UOp):
  # this is the GID level on the GPU, this is where we define LOCAL buffers shared across lids
  gx = UOp.range(N//BLOCK_N, 0, AxisType.GLOBAL) * BLOCK_N
  gy = UOp.range(M//BLOCK_M, 1, AxisType.GLOBAL) * BLOCK_M
  a_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  b_local = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  return local_matmul(c[gx:gx+BLOCK_N, gy:gy+BLOCK_M], a.permute(1,0)[:, gx:gx+BLOCK_N], b[:, gy:gy+BLOCK_M], a_local, b_local)

  #ll = load_local(a_local, b_local, a.permute(1,0)[:, gx:gx+BLOCK_N], b[:, gy:gy+BLOCK_M])
  #return compute_local(c[gx:gx+BLOCK_N, gy:gy+BLOCK_M], a_local.after(ll), b_local.after(ll))

if __name__ == "__main__":
  # this is the outer lvel on the GPU, this is where we define GLOBAL buffers
  C = Tensor.empty(N, M)
  A = Tensor.randn(N, K)
  B = Tensor.randn(K, M)
  c_out = C.call(A, B, fxn=global_matmul).numpy()

  #C = UOp.new_buffer(Device.DEFAULT, N*M, dtypes.float).reshape(N,M)
  #A = UOp.new_buffer(Device.DEFAULT, N*K, dtypes.float).reshape(N,K)
  #B = UOp.new_buffer(Device.DEFAULT, K*M, dtypes.float).reshape(K,M)
  #global_matmul(C, A, B).realize()

  # input matmuls
  #c = UOp.param(0, dtypes.float, (N, M))
  #a = UOp.param(1, dtypes.float, (N, K))
  #b = UOp.param(2, dtypes.float, (K, M))


  #ba = a.rearrange("(n bn) (k bk) -> n k bn bk", bn=BLOCK_N, bk=BLOCK_K)[gx, k_tile_range]
  #bb = b.rearrange("(k bk) (m bm) -> k m bk bm", bk=BLOCK_K, bm=BLOCK_M)[k_tile_range, gy]
  #bc = c.rearrange("(n bn) (m bm) -> n m bn bm", bn=BLOCK_N, bm=BLOCK_M)[gx, gy]

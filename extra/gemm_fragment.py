"""
tilelang-style matmul_relu written with tinygrad UOp APIs.

Demonstrates that tilelang's T.alloc_fragment is expressible with existing
tinygrad primitives: a per-thread REG buffer, wrapped in one Ops.UNSHARD per
sharded axis over the LOCAL thread-grid ranges to form the full logical tile.
Here the 64 threads are an 8x8 grid and each thread owns an 8x8 sub-tile --
the 2-D fragment layout tilelang infers. The kernel is written against the
full-tile UNSHARD view, and multi_pm (the same pass that lowers multi-device
UNSHARDs) resolves it into per-thread shard code.

Reference tilelang kernel:

    @tilelang.jit
    def matmul_relu(A, B, block_M=64, block_N=64, block_K=64,
                    dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const('M, N, K')
        C = T.empty([M, N], dtype)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

API mapping (tilelang -> tinygrad UOps, idioms from test/backend/test_custom_kernel.py):

    T.Kernel(gx, gy, threads=T)    -> AxisType.GLOBAL ranges (blocks) + AxisType.LOCAL ranges (thread grid)
    T.alloc_shared(shape, dtype)   -> UOp.placeholder(shape, dtype, slot, AddrSpace.LOCAL)
    T.alloc_fragment(shape, dt)    -> per-thread REG placeholder, wrapped in one Ops.UNSHARD per sharded axis over
                                      the AxisType.LOCAL ranges: fragment.unshard((axis_y, axis_x), (ty, tx)).
                                      The full logical tile is the shard with each sharded axis multiplied by its
                                      range size, exactly like device sharding, but the sharding axes are thread
                                      axes carried by the RANGE metadata instead of a device tuple. C_local[i, j]
                                      with [i, j] in this thread's shard is INDEX on the UNSHARD, which multi_pm
                                      resolves into INDEX on the per-thread REG shard, axis by axis.
    T.copy(gmem_slice, smem)       -> smem[thread_idx].set(gmem_slice[thread_idx], end=copy_rng). set returns the
                                      smem tile AFTER the copy; the implicit-barrier pass turns the store->load
                                      dependency of the loop that consumes it into a workgroup barrier
    T.gemm (no WMMA)               -> C_local[..].set(C_local.after(k)[..] + a_shared[..] * b_shared[..], end=k)
                                      with k a loop-carried LOOP range (codegen builds the register accumulator
                                      from this self-referential store automatically)
    T.copy(fragment, gmem)         -> gmem.index(gidx).store(C_local[..]).end(all_ranges)
    UNSHARD lowering               -> multi_pm in codegen (full_rewrite_to_sink): INDEX/AFTER/STORE ops on the
                                      full-tile view become per-thread shard ops, no UNSHARD survives into the program.
"""

from tinygrad.dtype import dtypes, AddrSpace, DType
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.helpers import cdiv, getenv
from tinygrad.tensor import Tensor

# ---------------------------------------------------------------------------
# tilelang builtins, expressed with tinygrad UOp APIs
# ---------------------------------------------------------------------------

def alloc_shared(shape:tuple[int, ...], dtype:DType) -> UOp:
  """T.alloc_shared: one LOCAL buffer shared by all threads in the block."""
  return UOp.placeholder(tuple(shape), dtype, next(UOp.unique_num), AddrSpace.LOCAL)

def alloc_fragment(shape:tuple[int, ...], dtype:DType, axes:tuple[int, ...], rngs:tuple[UOp, ...]) -> UOp:
  """T.alloc_fragment: per-thread REG fragment + UNSHARD over the LOCAL thread grid.

  Each thread privately owns shape[axis]//threads elements along every sharded
  axis in a REG buffer. The UNSHARDs over the LOCAL thread ranges present the
  full logical tile: full_shape = shard_shape with each sharded axis multiplied
  by its range size. This is exactly how UNSHARD carries a DEVICE axis today,
  except the sharding axes are thread axes carried by the RANGE metadata.
  """
  assert len(axes) == len(rngs)
  assert all(tnum.op is Ops.RANGE and tnum.arg[-1] is AxisType.LOCAL for tnum in rngs), "fragments shard over LOCAL ranges"
  assert all(shape[a] % (int(rng.vmax)+1) == 0 for a, rng in zip(axes, rngs))
  by_axis = dict(zip(axes, rngs))
  shard_shape = tuple(s // (int(by_axis[i].vmax)+1) if i in by_axis else s for i, s in enumerate(shape))
  fragment = UOp.placeholder(shard_shape, dtype, next(UOp.unique_num), AddrSpace.REG)
  return fragment.unshard(axes, rngs)

# ---------------------------------------------------------------------------
# GEMM kernel: C = relu(A @ B), float inputs (fp16 or fp32), fp32 fragment accumulator, no WMMA
# ---------------------------------------------------------------------------

# 64x64 output tile per block, 64 threads as an 8x8 grid; each thread owns an 8x8 fragment sub-tile
# (the 2-D per-thread layout tilelang infers for this GEMM)
BLOCK_M = BLOCK_N = BLOCK_K = 64
TY = TX = 8
THREADS = TY * TX
TM = BLOCK_M // TY   # fragment rows per thread
TN = BLOCK_N // TX   # fragment columns per thread

def matmul_relu_kernel(c:UOp, a:UOp, b:UOp) -> UOp:
  """C[M, N] = relu(A[M, K] @ B[K, N]) -- one 64x64 tile per block, locals + a 2-D fragment."""
  M, K = a.shape
  K2, N = b.shape
  assert K == K2 and a.dtype == b.dtype == c.dtype and not dtypes.is_int(a.dtype)
  assert not (K % BLOCK_K or M % BLOCK_M or N % BLOCK_N), "test sizes must be multiples of the block sizes"

  # with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=64) as (bx, by):
  bx = UOp.range(cdiv(N, BLOCK_N), 0, AxisType.GLOBAL)
  by = UOp.range(cdiv(M, BLOCK_M), 1, AxisType.GLOBAL)
  ty = UOp.range(TY, 2, AxisType.LOCAL)
  tx = UOp.range(TX, 3, AxisType.LOCAL)

  # A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
  # B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
  A_shared = alloc_shared((BLOCK_M, BLOCK_K), a.dtype)
  B_shared = alloc_shared((BLOCK_K, BLOCK_N), b.dtype)

  # C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype) -- an 8x8 REG tile per thread of the 8x8 grid
  C_local = alloc_fragment((BLOCK_M, BLOCK_N), dtypes.float32, (0, 1), (ty, tx))

  # T.clear(C_local) -- each thread zeroes its own fragment sub-tile
  ic, jc = UOp.range(TM, 4, AxisType.LOOP), UOp.range(TN, 5, AxisType.LOOP)
  C_loc = C_local[ty*TM + ic, tx*TN + jc].set(0.0, end=(ic, jc))

  # for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
  # (num_stages pipelining is async copy + multi-buffering; this is the synchronous single-buffer version)
  ko = UOp.range(cdiv(K, BLOCK_K), 6, AxisType.LOOP)

  # T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared) -- each thread copies its own 8x8 sub-tile
  # set returns the tile AFTER the copy; codegen turns that store->load dependency into a workgroup barrier
  iar, ka = UOp.range(TM, 7, AxisType.LOOP), UOp.range(TN, 8, AxisType.LOOP)
  A_store = A_shared[ty*TM + iar, tx*TN + ka].store(a[by*BLOCK_M + ty*TM + iar, ko*BLOCK_K + tx*TN + ka]).end(iar, ka)

  # T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
  kb, ibr = UOp.range(TM, 9, AxisType.LOOP), UOp.range(TN, 10, AxisType.LOOP)
  B_store = B_shared[ty*TM + kb, tx*TN + ibr].store(b[ko*BLOCK_K + ty*TM + kb, bx*BLOCK_N + tx*TN + ibr]).end(kb, ibr)

  # get the shared after the stores (single barrier)
  A_shared = A_shared.after(A_store, B_store)
  B_shared = B_shared.after(A_store, B_store)

  # T.gemm(A_shared, B_shared, C_local), no WMMA -- per-thread accumulate over its fragment sub-tile.
  # identical to custom_gemm: a self-referential store over the loop-carried kk range,
  # which codegen turns into a register accumulator
  ir, kk = UOp.range(TM, 11, AxisType.LOOP), UOp.range(BLOCK_K, 12, AxisType.LOOP)
  jj = UOp.range(TN, 13, AxisType.LOOP)
  acc = C_loc.after(kk)[ty*TM + ir, tx*TN + jj] + A_shared[ty*TM + ir, kk].cast(dtypes.float32) * B_shared[kk, tx*TN + jj].cast(dtypes.float32)
  # closing the ko loop here too; codegen adds the barrier so no thread overwrites the tiles while others still read them
  C_loc = C_loc[ty*TM + ir, tx*TN + jj].set(acc, end=(kk, ir, jj, ko))

  # for i, j in T.Parallel(BLOCK_M, BLOCK_N): C_local[i, j] = T.max(C_local[i, j], 0)
  # T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N]) -- per-thread store of the fragment shard (relu fused into it)
  # LOOP: these loops are the per-thread output layout; convert_loop_to_global must not globalize them
  ie, je = UOp.range(TM, 14, AxisType.LOOP), UOp.range(TN, 15, AxisType.LOOP)
  c_st = c[by*BLOCK_M + ty*TM + ie, bx*BLOCK_N + tx*TN + je].store(C_loc[ty*TM + ie, tx*TN + je].relu().cast(c.dtype))

  # all open ranges are closed at the final store (ko was closed above).
  # the fragment UNSHARDs go to codegen as is: multi_pm there resolves the full-tile view into per-thread shard code
  return c_st.end(je, ie, tx, ty, bx, by).sink(arg=KernelInfo(name="matmul_relu", opts_to_apply=()))

# ---------------------------------------------------------------------------
# python wrapper: same signature as the tilelang function
# ---------------------------------------------------------------------------

def matmul_relu(a:Tensor, b:Tensor) -> Tensor:
  """C = relu(A @ B), fp16 in/out with an fp32 fragment accumulator."""
  c = Tensor.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
  return c.custom_kernel(a, b, fxn=matmul_relu_kernel)[0]

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  from tinygrad import Device
  assert Device[Device.DEFAULT].renderer.has_local, "this GPU-style kernel needs a backend with local memory (LOCAL ranges + barriers)"
  M = K = N = getenv("N", 256)  # 4x4 grid of 64x64 tiles, 4 K chunks

  a = Tensor.randn(M, K, dtype=dtypes.float16).contiguous()
  b = Tensor.randn(K, N, dtype=dtypes.float16).contiguous()
  ref = (a.float() @ b.float()).relu().realize()

  out = matmul_relu(a, b).realize()

  import numpy as np
  np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-1, rtol=1e-2)
  print("matmul_relu passed!")

"""Shared UOp generation helpers for WMMA GEMM kernels."""
from dataclasses import dataclass
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops, AxisType, AddrSpace, KernelInfo
from tinygrad.renderer.cstyle import CUDARenderer

@dataclass
class WmmaVariant:
  name: str
  smem_bytes: int

  # tile / math params
  M_TILE: int            # block output size in M
  N_TILE: int            # block output size in N
  K_TILE: int            # K elements consumed per outer loop iteration
  MMA_K: int             # per-WMMA k (16)
  M_FRAGS: int           # "am" dimension (2 vs 4 in fp16-acc)
  N_FRAGS: int           # "bn" dimension (8 for both)

  # block / thread config
  block_threads: int     # 128 vs 256
  warps_m: int           # number of warp-groups along M
  warps_n: int           # number of warp-groups along N

  # pipeline
  stages: int            # 1, 2, or 3
  loop_prefetch_delta: int  # usually stages-1
  swizzled_smem: bool

  # memory / epilogue
  epilogue_type: str     # "direct", "smem_float4", "smem_half2"
  N_PAD: int             # 132 (float) or 136 (half)

  # dtypes
  acc_dtype: str         # "float" or "half"

  # smem stage sizes
  A_stage_bytes: int
  B_stage_bytes: int

FLAT_VARIANT = WmmaVariant(
  name="flat_smem_input",
  smem_bytes=24576,
  M_TILE=64, N_TILE=128, K_TILE=64, MMA_K=16,
  M_FRAGS=2, N_FRAGS=8,
  block_threads=128, warps_m=2, warps_n=2,
  stages=1, loop_prefetch_delta=1, swizzled_smem=False,
  epilogue_type="direct", N_PAD=132,
  acc_dtype="float",
  A_stage_bytes=8192, B_stage_bytes=16384
)

MAX_FP32_VARIANT = WmmaVariant(
  name="max",
  smem_bytes=49152,
  M_TILE=64, N_TILE=128, K_TILE=64, MMA_K=16,
  M_FRAGS=2, N_FRAGS=8,
  block_threads=128, warps_m=2, warps_n=2,
  stages=2, loop_prefetch_delta=2, swizzled_smem=True,
  epilogue_type="smem_float4", N_PAD=132,
  acc_dtype="float",
  A_stage_bytes=8192, B_stage_bytes=16384
)

MAX_FP16_VARIANT = WmmaVariant(
  name="max_fp16",
  smem_bytes=73728,
  M_TILE=256, N_TILE=128, K_TILE=32, MMA_K=16,
  M_FRAGS=4, N_FRAGS=8,
  block_threads=256, warps_m=4, warps_n=2,
  stages=3, loop_prefetch_delta=2, swizzled_smem=True,
  epilogue_type="smem_half2", N_PAD=136,
  acc_dtype="half",
  A_stage_bytes=16384, B_stage_bytes=8192
)

# Keep backward compatibility
MAX_VARIANT = MAX_FP32_VARIANT

class WmmaUOpBuilder:
  """Builder for WMMA GEMM kernels using UOps."""

  def __init__(self, M: int, N: int, K: int, variant: WmmaVariant):
    self.M, self.N, self.K = M, N, K
    self.variant = variant
    self.uops: list[UOp] = []
    self.seen_uops: set = set()
    self.commit_counter = 0
    self.wait_counter = 0
    self.barrier_counter = 0

  def u(self, op_or_uop, *args, **kwargs) -> UOp:
    out = op_or_uop if isinstance(op_or_uop, UOp) else UOp(op_or_uop, *args, **kwargs)
    if out in self.seen_uops: return out
    for src in out.src: self.u(src)
    self.uops.append(out)
    self.seen_uops.add(out)
    return out

  def idx(self, buf, off=0): return self.u(buf.index(off if isinstance(off, UOp) else UOp.const(dtypes.int, off), ptr=True))

  def cp_async(self, dst_ptr: UOp, src_ptr: UOp) -> UOp:
    return self.u(Ops.CUSTOM, dtypes.void, (dst_ptr, src_ptr), arg="__pipeline_memcpy_async({0}, {1}, 16);")

  def cp_commit(self) -> UOp:
    self.commit_counter += 1
    return self.u(Ops.CUSTOM, dtypes.void, (), arg=f"__pipeline_commit(); /* {self.commit_counter} */")

  def cp_wait_prior(self, n: int) -> UOp:
    self.wait_counter += 1
    return self.u(Ops.CUSTOM, dtypes.void, (), arg=f"__pipeline_wait_prior({n}); /* {self.wait_counter} */")

  def barrier(self) -> UOp:
    self.barrier_counter += 1
    return self.u(Ops.CUSTOM, dtypes.void, (), arg=f"__syncthreads(); /* {self.barrier_counter} */")

  def decompose_threads(self, threads):
    """Decompose linear thread ID into warp-group coordinates."""
    v = self.variant
    if v.block_threads == 128:
      return threads // 64, (threads // 32) % 2, threads % 32
    else:  # 256-thread layout
      return (threads // 32) % v.warps_m, threads // 128, threads % 32

  def compute_global_a_off(self, threads, grid_m):
    """Compute global A offset based on variant."""
    v, K = self.variant, self.K
    if v.block_threads == 128:
      return grid_m * v.M_TILE * K + (threads % 8) * 8 + (threads // 8) * K
    return grid_m * v.M_TILE * K + (threads % 4) * 8 + ((threads // 4) % 2) * (8 * 16 * K) + (threads // 8) * K

  def compute_global_b_off(self, threads, grid_n):
    """Compute global B offset based on variant."""
    return grid_n * self.variant.N_TILE + (threads % 16) * 8 + (threads // 16) * self.N

  def compute_store_smem_a_off(self, threads):
    """Compute SMEM store offset for A."""
    if not self.variant.swizzled_smem:
      return (threads % 8) * 8 + (threads // 8) * 64
    return (threads // 8) * 64 + (((threads * 8) ^ threads) & 56)

  def compute_store_smem_b_off(self, threads):
    """Compute SMEM store offset for B."""
    if not self.variant.swizzled_smem:
      return (threads % 16) * 8 + (threads // 16) * 128
    # store_smem_b_off = ((threads / 16) * 128) + ((((threads / 16) % 8) * 8) ^ ((threads % 16) * 8))
    return (threads // 16) * 128 + (((threads // 16) % 8 * 8) ^ ((threads % 16) * 8))

  def emit_prefetch_fp32(self, smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop):
    """Emit cp_async calls for one K-block prefetch (64x128 fp32 variant)."""
    K = self.K
    for r in [0, 16, 32, 48]:
      self.cp_async(self.idx(smem_a, store_smem_a_off + r*64), self.idx(a_uop, global_a_off + r*K))
    for r in range(0, 64, 8):
      self.cp_async(self.idx(smem_b, store_smem_b_off + r*128), self.idx(b_uop, global_b_off + r*self.N))

  def emit_prefetch_fp16(self, smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop):
    """Emit cp_async calls for one K-block prefetch (256x128 fp16 variant)."""
    for r in [0, 32, 64, 96]:
      self.cp_async(self.idx(smem_a, store_smem_a_off + r*64), self.idx(a_uop, global_a_off + r*self.K))
    for r in [0, 16]:
      self.cp_async(self.idx(smem_b, store_smem_b_off + r*128), self.idx(b_uop, global_b_off + r*self.N))

  def emit_prefetch(self, smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop):
    """Emit cp_async calls for one K-block prefetch."""
    if self.variant.block_threads == 128:
      self.emit_prefetch_fp32(smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
    else:
      self.emit_prefetch_fp16(smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)

  def emit_wmma_block(self, k_idx, smem_a, smem_b, ld_a_offsets, ld_b_offsets, acc_regs, a_frags, b_frags, wmma_arg):
    """Emit ldmatrix + WMMA ops for one k-tile (16 elements in K dimension)."""
    variant = self.variant
    acc_scalar_dtype = dtypes.float if variant.acc_dtype == "float" else dtypes.half

    # ldmatrix loads for A and B
    for i in range(variant.M_FRAGS):
      self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags[i]), self.idx(smem_a, ld_a_offsets[k_idx][i])), arg="__ldmatrix_a_elems({0}, {1});")
    for pair in range(4):
      self.u(Ops.CUSTOM, dtypes.void,
        (self.idx(b_frags[2*pair]), self.idx(b_frags[2*pair+1]), self.idx(smem_b, ld_b_offsets[k_idx][pair])), arg="__ldmatrix_b_elems({0}, {1}, {2});")

    # M_FRAGS x N_FRAGS WMMA ops
    for am in range(variant.M_FRAGS):
      a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags[am]),), tag=f"load_a_{k_idx}_{am}")
      for bn in range(variant.N_FRAGS):
        b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags[bn]),), tag=f"load_b_{k_idx}_{bn}")
        c_ptr = self.idx(acc_regs[am*variant.N_FRAGS + bn])
        c_val = self.u(Ops.LOAD, acc_scalar_dtype.vec(4), (c_ptr,), tag=f"load_c_{k_idx}_{am}_{bn}")
        out_val = self.u(Ops.WMMA, acc_scalar_dtype.vec(4), (a_val, b_val, c_val), arg=wmma_arg)
        self.u(Ops.STORE, dtypes.void, (c_ptr, out_val))

  def compute_flat_ldmatrix_offsets(self, wg_m, wg_n, wg_threads):
    """Compute ldmatrix offsets for flat (non-swizzled) SMEM layout."""
    load_smem_a_base = wg_m * 16*64 + (wg_threads % 8) * 64 + (wg_threads // 8 % 2) * 64*8 + (wg_threads // 16) * 8
    load_smem_b_base = wg_n * 16 + (wg_threads % 8) * 128 + (wg_threads // 8 % 2) * 128*8 + (wg_threads // 16) * 8

    ld_a_offsets, ld_b_offsets = [], []
    for k_idx in range(self.variant.K_TILE // self.variant.MMA_K):
      ld_a_offsets.append([self.u(load_smem_a_base + k_idx*16 + i*32*64) for i in range(self.variant.M_FRAGS)])
      ld_b_offsets.append([self.u(load_smem_b_base + k_idx*16*128 + pair*32) for pair in range(4)])
    return ld_a_offsets, ld_b_offsets

  def compute_swizzled_ldmatrix_offsets_fp32(self, wg_m, wg_n, threads):
    """Compute ldmatrix offsets for swizzled SMEM layout (fp32 accumulator kernel)."""
    load_smem_a_row = (wg_m * 16 + threads % 16) * 64
    load_smem_a_phase = threads // 16 % 2
    load_smem_b_row = (threads % 16) * 128
    load_smem_b_phase = wg_n * 2 + threads // 16 % 2

    ld_a_offsets, ld_b_offsets = [], []
    for k_idx in range(self.variant.K_TILE // self.variant.MMA_K):
      a_off_0 = self.u(load_smem_a_row + ((load_smem_a_phase + k_idx*2) ^ (threads % 8)) * 8)
      ld_a_offsets.append([a_off_0, self.u(a_off_0 + 32*64)])
      ld_b_offsets.append([self.u(load_smem_b_row + k_idx*16*128 + ((load_smem_b_phase + pair*4) ^ (threads % 8)) * 8) for pair in range(4)])
    return ld_a_offsets, ld_b_offsets

  def compute_swizzled_ldmatrix_offsets_fp16(self, wg_m, wg_n, wg_threads, threads):
    """Compute ldmatrix offsets for swizzled SMEM layout (fp16 accumulator kernel)."""
    load_smem_a_row = (wg_m * 16 + threads % 16) * 64
    load_smem_a_phase = threads // 16 % 2
    load_smem_b_row = (threads % 16) * 128
    load_smem_b_phase = wg_n * 2 + wg_threads // 16

    ld_a_offsets, ld_b_offsets = [], []
    for k_idx in range(self.variant.K_TILE // self.variant.MMA_K):
      a_frag_offsets = []
      for frag in range(self.variant.M_FRAGS):
        frag_phase_base = (frag // 2) * 4
        a_frag_offsets.append(self.u(load_smem_a_row + (frag % 2) * 64*64 + ((load_smem_a_phase + k_idx*2 + frag_phase_base) ^ (threads % 8)) * 8))
      ld_a_offsets.append(a_frag_offsets)
      ld_b_offsets.append([self.u(load_smem_b_row + k_idx*16*128 + ((load_smem_b_phase + pair*4) ^ (threads % 8)) * 8) for pair in range(4)])
    return ld_a_offsets, ld_b_offsets

  def compute_ldmatrix_offsets(self, wg_m, wg_n, wg_threads, threads):
    """Compute ldmatrix offsets based on variant."""
    variant = self.variant
    if not variant.swizzled_smem:
      return self.compute_flat_ldmatrix_offsets(wg_m, wg_n, wg_threads)
    elif variant.block_threads == 128:
      return self.compute_swizzled_ldmatrix_offsets_fp32(wg_m, wg_n, threads)
    else:
      return self.compute_swizzled_ldmatrix_offsets_fp16(wg_m, wg_n, wg_threads, threads)

  def emit_direct_epilogue(self, acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads):
    """Emit direct stores to global memory (flat_smem_input epilogue)."""
    v = self.variant
    wg_c_off = grid_m * v.M_TILE * self.N + grid_n * v.N_TILE + wg_m * 16*self.N + wg_n * 16
    thread_c_off = (wg_threads % 4) * 2 + (wg_threads // 4 % 8) * self.N

    for am in range(v.M_FRAGS):
      for bn in range(v.N_FRAGS):
        c_vec = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(acc_regs[am*v.N_FRAGS + bn]),))
        for k in range(4):
          self.u(Ops.STORE, dtypes.void,
            (self.idx(c_uop, wg_c_off + thread_c_off + am * 32*self.N + [0, 1, 8*self.N, 8*self.N + 1][k] + ((bn % 2) + 4 * (bn // 2))*8),
             self.u(Ops.GEP, dtypes.float, (c_vec,), (k,))))

  def emit_smem_float4_epilogue(self, acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem):
    """Emit smem-based float4 epilogue (max fp32 kernel)."""
    N_PAD = self.variant.N_PAD
    smem_d = smem.cast(dtypes.float.ptr(AddrSpace.LOCAL))
    smem_d_off = wg_m * 16*N_PAD + wg_n * 16 + (wg_threads % 4) * 2 + (wg_threads // 4 % 8) * N_PAD
    load_smem_d_off = (threads % 32) * 4 + (threads // 32) * N_PAD
    global_d_base = grid_m * self.variant.M_TILE * self.N + grid_n * self.variant.N_TILE + (threads % 32) * 4 + (threads // 32) * self.N

    bn_col_map = [0, 1, 4, 5, 8, 9, 12, 13]
    for am, d_regs_name in enumerate(["d_0", "d_1"]):
      for bn in range(8):
        c_vec = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(acc_regs[am*8 + bn]),))
        for k in range(4):
          self.u(Ops.STORE, dtypes.void,
            (self.idx(smem_d, smem_d_off + bn_col_map[bn]*8 + [0, 1, 8*N_PAD, 8*N_PAD + 1][k]),
             self.u(Ops.GEP, dtypes.float, (c_vec,), (k,))))
      self.barrier()

      d_regs = [self.u(Ops.DEFINE_REG, dtypes.float.vec(4).ptr(1, AddrSpace.REG), (), f"{d_regs_name}_{i}") for i in range(8)]
      for i, row_off in enumerate([0, 4, 8, 12, 16, 20, 24, 28]):
        self.u(Ops.STORE, dtypes.void, (self.idx(d_regs[i]),
          self.u(Ops.LOAD, dtypes.float.vec(4),
            (self.u(self.idx(smem_d, load_smem_d_off + row_off * N_PAD).cast(dtypes.float.vec(4).ptr(AddrSpace.LOCAL))),),
            tag=f"load_{d_regs_name}_{i}")))
      self.barrier()

      rows = [0, 4, 8, 12, 16, 20, 24, 28] if am == 0 else [32, 36, 40, 44, 48, 52, 56, 60]
      for i, row in enumerate(rows):
        self.u(Ops.STORE, dtypes.void,
          (self.u(self.idx(c_uop, global_d_base + row*self.N).cast(dtypes.float.vec(4).ptr())),
           self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(d_regs[i]),), tag=f"store_{d_regs_name}_{i}")))

  def emit_smem_half2_epilogue(self, acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem):
    """Emit smem-based half2/half8 epilogue (max fp16 kernel)."""
    v, W = self.variant, self.variant.N_PAD
    smem32_d = smem.cast(dtypes.half.vec(2).ptr(AddrSpace.LOCAL))
    smem128_d = smem.cast(dtypes.half.vec(8).ptr(AddrSpace.LOCAL))
    out128_d = c_uop.cast(dtypes.half.vec(8).ptr())

    smem32_d_off = wg_m * 8 * (W // 2) + wg_n * 8 + (wg_threads // 4) * (W // 2) + wg_threads % 4
    smem128_d_read_off = (threads // 16) * (W // 8) + threads % 16
    out128_d_off = grid_m * v.M_TILE * (self.N // 8) + grid_n * (v.N_TILE // 8) + (threads // 128) * 16 * (self.N // 8) + (threads // 16 % 8) * (self.N // 8) + threads % 16

    bn_col_map = [0, 1, 4, 5, 8, 9, 12, 13]
    for am in range(v.M_FRAGS):
      for half_idx, (gep0, gep1) in enumerate([(0, 1), (2, 3)]):
        self.barrier()
        for bn in range(v.N_FRAGS):
          c_vec = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(acc_regs[am * v.N_FRAGS + bn]),))
          h2_val = self.u(Ops.VECTORIZE, dtypes.half.vec(2), (self.u(Ops.GEP, dtypes.half, (c_vec,), (gep0,)), self.u(Ops.GEP, dtypes.half, (c_vec,), (gep1,))))
          self.u(Ops.STORE, dtypes.void, (self.idx(smem32_d, smem32_d_off + bn_col_map[bn] * 4), h2_val))
        self.barrier()

        out_off_base = out128_d_off + am * 64 * (self.N // 8)
        row_offsets = [(0, 0), (32, 16 * (W // 8))] if half_idx == 0 else [(8, 0), (40, 16 * (W // 8))]
        for out_row, smem_row in row_offsets:
          self.u(Ops.STORE, dtypes.void, (self.idx(out128_d, out_off_base + out_row * (self.N // 8)),
            self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(smem128_d, smem128_d_read_off + smem_row),), tag=f"epi_{am}_{out_row}_load")))
    self.barrier()

  def build(self) -> list[UOp]:
    """Build the complete WMMA kernel UOps."""
    variant = self.variant
    M, N, K = self.M, self.N, self.K

    # Determine dtypes based on variant
    acc_scalar_dtype = dtypes.float if variant.acc_dtype == "float" else dtypes.half
    out_dtype = dtypes.float if variant.acc_dtype == "float" else dtypes.half

    # Global definitions
    c_uop = self.u(Ops.DEFINE_GLOBAL, out_dtype.ptr(), (), 0)
    a_uop = self.u(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 1)
    b_uop = self.u(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 2)

    # Grid/thread indices
    grid_m = self.u(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, M // variant.M_TILE),), "gidx0")
    grid_n = self.u(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, N // variant.N_TILE),), "gidx1")
    threads = self.u(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, variant.block_threads),), "lidx0")

    wg_m, wg_n, wg_threads = self.decompose_threads(threads)

    # Global offsets
    global_a_off = self.compute_global_a_off(threads, grid_m)
    global_b_off = self.compute_global_b_off(threads, grid_n)

    # SMEM store offsets
    store_smem_a_off = self.compute_store_smem_a_off(threads)
    store_smem_b_off = self.compute_store_smem_b_off(threads)

    # Shared memory definition
    # For large shared memory (>48KB), we use extern __shared__ which requires dynamic allocation
    if variant.smem_bytes > 49152:
      # Emit extern shared memory declaration as a CUSTOM statement, then define a local pointer
      self.u(Ops.CUSTOM, dtypes.void, (), arg="extern __shared__ __align__(16) signed char dyn_smem[];")
      smem = self.u(Ops.CUSTOMI, dtypes.int8.ptr(variant.smem_bytes, AddrSpace.LOCAL), (), arg="dyn_smem")
    else:
      smem = self.u(Ops.DEFINE_LOCAL, dtypes.int8.ptr(variant.smem_bytes, AddrSpace.LOCAL), (), "smem")

    # Create SMEM stage pointers
    smem_a_stages, smem_b_stages, offset = [], [], 0
    for s in range(variant.stages):
      smem_a_stages.append((smem + UOp.const(dtypes.int, offset)).cast(dtypes.half.ptr(AddrSpace.LOCAL)))
      offset += variant.A_stage_bytes
    for s in range(variant.stages):
      smem_b_stages.append((smem + UOp.const(dtypes.int, offset)).cast(dtypes.half.ptr(AddrSpace.LOCAL)))
      offset += variant.B_stage_bytes

    # Compute ldmatrix offsets
    ld_a_offsets, ld_b_offsets = self.compute_ldmatrix_offsets(wg_m, wg_n, wg_threads, threads)

    # Register definitions
    num_acc_regs = variant.M_FRAGS * variant.N_FRAGS
    acc_regs = [self.u(Ops.DEFINE_REG, acc_scalar_dtype.vec(4).ptr(1, AddrSpace.REG), (), f"acc_reg_{i}") for i in range(num_acc_regs)]
    a_frags = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_{i}") for i in range(variant.M_FRAGS)]
    b_frags = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_{i}") for i in range(variant.N_FRAGS)]

    # Initialize accumulators to zero
    for i in range(num_acc_regs):
      self.u(Ops.STORE, dtypes.void, (self.idx(acc_regs[i]), UOp(Ops.VECTORIZE, acc_scalar_dtype.vec(4), (UOp.const(acc_scalar_dtype, 0.0),)*4)))

    # WMMA descriptor
    if variant.acc_dtype == "float":
      wmma_arg = ("WMMA_8_16_16_half_float", (8, 16, 16), dtypes.half, dtypes.float, "CUDA", 32, (((0, 8),), ((0, 4),), ((0, 4),)), ())
    else:
      wmma_arg = ("WMMA_8_16_16_half_half", (8, 16, 16), dtypes.half, dtypes.half, "CUDA", 32, (((0, 8),), ((0, 4),), ((0, 4),)), ())

    num_k_blocks = K // variant.K_TILE

    self.barrier()

    if variant.stages == 1:
      # Single-buffer, 1-stage pipeline (flat_smem_input)
      smem_a, smem_b = smem_a_stages[0], smem_b_stages[0]
      self.emit_prefetch(smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off += variant.K_TILE
      global_b_off += variant.K_TILE * N
      self.barrier()

      block_k = self.u(Ops.RANGE, dtypes.int, (UOp.const(dtypes.int, num_k_blocks),), (0, AxisType.LOOP))
      self.cp_wait_prior(0)
      self.barrier()
      for k_idx in range(variant.K_TILE // variant.MMA_K):
        self.emit_wmma_block(k_idx, smem_a, smem_b, ld_a_offsets, ld_b_offsets, acc_regs, a_frags, b_frags, wmma_arg)
      self.barrier()

      cond = block_k < num_k_blocks - 1
      if_uop = self.u(Ops.IF, dtypes.void, (cond,))
      self.emit_prefetch(smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off + block_k * variant.K_TILE, global_b_off + block_k * variant.K_TILE * N, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_uop,))
      self.cp_commit()
      self.u(Ops.END, dtypes.void, (block_k,))

    elif variant.stages == 2:
      # Double-buffer, 2-stage pipeline (max fp32)
      self.emit_prefetch(smem_a_stages[0], smem_b_stages[0], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off += variant.K_TILE
      global_b_off += variant.K_TILE * N
      self.barrier()

      self.emit_prefetch(smem_a_stages[1], smem_b_stages[1], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off += variant.K_TILE
      global_b_off += variant.K_TILE * N
      self.cp_wait_prior(0)
      self.barrier()

      block_k = self.u(Ops.RANGE, dtypes.int, (UOp.const(dtypes.int, num_k_blocks),), (0, AxisType.LOOP))
      cond_odd = (block_k % 2).eq(0)
      smem_a_curr = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_odd, smem_a_stages[1], smem_a_stages[0]))
      smem_b_curr = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_odd, smem_b_stages[1], smem_b_stages[0]))

      for k_idx in range(variant.K_TILE // variant.MMA_K):
        self.emit_wmma_block(k_idx, smem_a_curr, smem_b_curr, ld_a_offsets, ld_b_offsets, acc_regs, a_frags, b_frags, wmma_arg)
      self.barrier()

      cond_prefetch = block_k < num_k_blocks - 2
      if_prefetch = self.u(Ops.IF, dtypes.void, (cond_prefetch,))
      self.emit_prefetch(smem_a_curr, smem_b_curr, store_smem_a_off, store_smem_b_off, global_a_off + block_k * variant.K_TILE, global_b_off + block_k * variant.K_TILE * N, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_prefetch,))
      self.cp_commit()

      cond_wait = block_k < num_k_blocks - 1
      if_wait = self.u(Ops.IF, dtypes.void, (cond_wait,))
      self.cp_wait_prior(1)
      self.barrier()
      self.u(Ops.END, dtypes.void, (if_wait,))
      self.u(Ops.END, dtypes.void, (block_k,))

    else:  # stages == 3
      # Triple-buffer, 3-stage pipeline (max fp16)
      a_frags_k0 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_k0_{i}") for i in range(variant.M_FRAGS)]
      a_frags_k1 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_k1_{i}") for i in range(variant.M_FRAGS)]
      b_frags_k0 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_k0_{i}") for i in range(variant.N_FRAGS)]
      b_frags_k1 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_k1_{i}") for i in range(variant.N_FRAGS)]

      self.emit_prefetch(smem_a_stages[0], smem_b_stages[0], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off += variant.K_TILE
      global_b_off += variant.K_TILE * N

      self.emit_prefetch(smem_a_stages[1], smem_b_stages[1], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off += variant.K_TILE
      global_b_off += variant.K_TILE * N

      self.cp_wait_prior(1)
      self.barrier()

      for i in range(variant.M_FRAGS):
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k0[i]), self.idx(smem_a_stages[0], ld_a_offsets[0][i])), arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(b_frags_k0[2*pair]), self.idx(b_frags_k0[2*pair+1]), self.idx(smem_b_stages[0], ld_b_offsets[0][pair])), arg="__ldmatrix_b_elems({0}, {1}, {2});")

      block_k = self.u(Ops.RANGE, dtypes.int, (UOp.const(dtypes.int, num_k_blocks),), (0, AxisType.LOOP))
      phase_k = block_k % 3
      next_phase_k = (block_k + 1) % 3
      store_phase_k = (block_k + 2) % 3

      # Current tile SMEM pointers
      cond_0 = phase_k.eq(0)
      cond_1 = phase_k.eq(1)
      smem_a_curr = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_curr = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_1, smem_b_stages[1], smem_b_stages[2]))))

      # Next tile SMEM pointers
      cond_n0 = next_phase_k.eq(0)
      cond_n1 = next_phase_k.eq(1)
      smem_a_next = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_n0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_n1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_next = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_n0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_n1, smem_b_stages[1], smem_b_stages[2]))))

      # Store (prefetch) tile SMEM pointers
      cond_s0 = store_phase_k.eq(0)
      cond_s1 = store_phase_k.eq(1)
      smem_a_store = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_s0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_s1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_store = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_s0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_s1, smem_b_stages[1], smem_b_stages[2]))))

      # 1) Load K=1 fragments for the current tile into dedicated K=1 registers
      for i in range(variant.M_FRAGS):
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k1[i]), self.idx(smem_a_curr, ld_a_offsets[1][i])), arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        self.u(Ops.CUSTOM, dtypes.void,
          (self.idx(b_frags_k1[2*pair]), self.idx(b_frags_k1[2*pair+1]), self.idx(smem_b_curr, ld_b_offsets[1][pair])), arg="__ldmatrix_b_elems({0}, {1}, {2});")

      # 2) MMA for K=0 using previously loaded K=0 fragments
      for am in range(variant.M_FRAGS):
        a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags_k0[am]),), tag=f"mma_k0_a_{am}")
        for bn in range(variant.N_FRAGS):
          b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags_k0[bn]),), tag=f"mma_k0_b_{bn}")
          c_ptr = self.idx(acc_regs[am * variant.N_FRAGS + bn])
          c_val = self.u(Ops.LOAD, acc_scalar_dtype.vec(4), (c_ptr,), tag=f"mma_k0_c_{am}_{bn}")
          out_val = self.u(Ops.WMMA, acc_scalar_dtype.vec(4), (a_val, b_val, c_val), arg=wmma_arg)
          self.u(Ops.STORE, dtypes.void, (c_ptr, out_val))

      # 3) Prefetch next tile
      cond_prefetch = block_k < num_k_blocks - 2
      if_prefetch = self.u(Ops.IF, dtypes.void, (cond_prefetch,))
      self.emit_prefetch(smem_a_store, smem_b_store, store_smem_a_off, store_smem_b_off, global_a_off + block_k * variant.K_TILE, global_b_off + block_k * variant.K_TILE * N, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_prefetch,))
      self.cp_commit()

      # 4) Wait for the "next" tile (block_k+1) to become available and sync
      self.cp_wait_prior(1)
      self.barrier()

      # 5) Load K=0 fragments for the next tile into K=0 registers (used as K=0 on next iteration)
      for i in range(variant.M_FRAGS):
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k0[i]), self.idx(smem_a_next, ld_a_offsets[0][i])), arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        self.u(Ops.CUSTOM, dtypes.void,
          (self.idx(b_frags_k0[2*pair]), self.idx(b_frags_k0[2*pair+1]), self.idx(smem_b_next, ld_b_offsets[0][pair])), arg="__ldmatrix_b_elems({0}, {1}, {2});")

      # 6) MMA for K=1 using the K=1 fragments loaded at the top of the loop
      for am in range(variant.M_FRAGS):
        a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags_k1[am]),), tag=f"mma_k1_a_{am}")
        for bn in range(variant.N_FRAGS):
          b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags_k1[bn]),), tag=f"mma_k1_b_{bn}")
          c_ptr = self.idx(acc_regs[am * variant.N_FRAGS + bn])
          c_val = self.u(Ops.LOAD, acc_scalar_dtype.vec(4), (c_ptr,), tag=f"mma_k1_c_{am}_{bn}")
          out_val = self.u(Ops.WMMA, acc_scalar_dtype.vec(4), (a_val, b_val, c_val), arg=wmma_arg)
          self.u(Ops.STORE, dtypes.void, (c_ptr, out_val))

      self.u(Ops.END, dtypes.void, (block_k,))

    # Epilogue
    self.cp_wait_prior(0)
    self.barrier()

    if variant.epilogue_type == "direct":
      self.emit_direct_epilogue(acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads)
    elif variant.epilogue_type == "smem_float4":
      self.emit_smem_float4_epilogue(acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem)
    else:  # smem_half2
      self.emit_smem_half2_epilogue(acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem)

    self.u(Ops.SINK, dtypes.void, (), arg=KernelInfo(name=f"nv_{variant.name}_uop"))

    return self.uops


def build_wmma_uops(M: int, N: int, K: int, variant: WmmaVariant) -> list[UOp]:
  """Build WMMA GEMM UOps for the given variant."""
  builder = WmmaUOpBuilder(M, N, K, variant)
  return builder.build()

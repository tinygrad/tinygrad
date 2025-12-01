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

  def u(self, op, *args, **kwargs) -> UOp:
    out = UOp(op, *args, **kwargs)
    if out not in self.seen_uops:
      self.uops.append(out)
      self.seen_uops.add(out)
    return out

  def s_const(self, v: int) -> UOp:
    out = UOp.const(dtypes.index, v)
    if out not in self.seen_uops:
      self.uops.append(out)
      self.seen_uops.add(out)
    return out

  def f_const(self, v: float) -> UOp:
    out = UOp.const(dtypes.float, v)
    if out not in self.seen_uops:
      self.uops.append(out)
      self.seen_uops.add(out)
    return out

  def h_const(self, v: float) -> UOp:
    out = UOp.const(dtypes.half, v)
    if out not in self.seen_uops:
      self.uops.append(out)
      self.seen_uops.add(out)
    return out

  def add(self, a, b): return self.u(Ops.ADD, dtypes.int, (a, b))
  def mul(self, a, b): return self.u(Ops.MUL, dtypes.int, (a, b))
  def xor(self, a, b): return self.u(Ops.XOR, dtypes.int, (a, b))
  def and_(self, a, b): return self.u(Ops.AND, dtypes.int, (a, b))
  def idx(self, buf, val, dtype=None): return self.u(Ops.INDEX, dtype if dtype else buf.dtype, (buf, val))
  def cast(self, val, dtype): return self.u(Ops.CAST, dtype, (val,))

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
    variant = self.variant
    if variant.block_threads == 128:
      wg_m = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(64)))
      wg_n = self.u(Ops.MOD, dtypes.int, (self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(32))), self.s_const(2)))
      wg_threads = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(32)))
    else:  # 256-thread layout
      wg_threads = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(32)))
      wg_m = self.u(Ops.MOD, dtypes.int, (self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(32))), self.s_const(variant.warps_m)))
      wg_n = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(128)))
    return wg_m, wg_n, wg_threads

  def compute_global_a_off(self, threads, grid_m):
    """Compute global A offset based on variant."""
    variant = self.variant
    K = self.K
    if variant.block_threads == 128:
      # 64x128 fp32 accumulator formula
      t_mod_8 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(8)))
      t_div_8 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(8)))
      return self.add(self.add(
        self.mul(self.mul(grid_m, self.s_const(variant.M_TILE)), self.s_const(K)),
        self.mul(t_mod_8, self.s_const(8))),
        self.mul(t_div_8, self.s_const(K)))
    else:
      # 256x128 fp16 accumulator formula
      # ((grid_m * 256) * K) + ((threads % 4) * 8) + (((threads / 4) % 2) * 8 * 16 * K) + ((threads / 8) * K)
      t_mod_4 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(4)))
      t_div_4 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(4)))
      t_div_4_mod_2 = self.u(Ops.MOD, dtypes.int, (t_div_4, self.s_const(2)))
      t_div_8 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(8)))
      term1 = self.mul(self.mul(grid_m, self.s_const(variant.M_TILE)), self.s_const(K))
      term2 = self.mul(t_mod_4, self.s_const(8))
      term3 = self.mul(t_div_4_mod_2, self.s_const(8 * 16 * K))
      term4 = self.mul(t_div_8, self.s_const(K))
      return self.add(self.add(term1, term2), self.add(term3, term4))

  def compute_global_b_off(self, threads, grid_n):
    """Compute global B offset based on variant."""
    variant = self.variant
    N = self.N
    t_mod_16 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(16)))
    t_div_16 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(16)))
    return self.add(self.add(
      self.mul(grid_n, self.s_const(variant.N_TILE)),
      self.mul(t_mod_16, self.s_const(8))),
      self.mul(t_div_16, self.s_const(N)))

  def compute_store_smem_a_off(self, threads):
    """Compute SMEM store offset for A."""
    variant = self.variant
    if not variant.swizzled_smem:
      t_mod_8 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(8)))
      t_div_8 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(8)))
      return self.add(self.mul(t_mod_8, self.s_const(8)), self.mul(t_div_8, self.s_const(64)))
    else:
      # store_smem_a_off = ((threads / 8) * 64) + (((threads * 8) ^ threads) & 56)
      t_div_8 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(8)))
      threads_mul_8 = self.mul(threads, self.s_const(8))
      swiz_a = self.and_(self.xor(threads_mul_8, threads), self.s_const(56))
      return self.add(self.mul(t_div_8, self.s_const(64)), swiz_a)

  def compute_store_smem_b_off(self, threads):
    """Compute SMEM store offset for B."""
    variant = self.variant
    t_mod_16 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(16)))
    t_div_16 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(16)))
    if not variant.swizzled_smem:
      return self.add(self.mul(t_mod_16, self.s_const(8)), self.mul(t_div_16, self.s_const(128)))
    else:
      # store_smem_b_off = ((threads / 16) * 128) + ((((threads / 16) % 8) * 8) ^ ((threads % 16) * 8))
      t_div_16_mod_8 = self.u(Ops.MOD, dtypes.int, (t_div_16, self.s_const(8)))
      t_div_16_mod_8_mul_8 = self.mul(t_div_16_mod_8, self.s_const(8))
      t_mod_16_mul_8 = self.mul(t_mod_16, self.s_const(8))
      swiz_b = self.xor(t_div_16_mod_8_mul_8, t_mod_16_mul_8)
      return self.add(self.mul(t_div_16, self.s_const(128)), swiz_b)

  def emit_prefetch_fp32(self, smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop):
    """Emit cp_async calls for one K-block prefetch (64x128 fp32 variant)."""
    K = self.K
    for r in [0, 16, 32, 48]:
      self.cp_async(
        self.idx(smem_a, self.add(store_smem_a_off, self.s_const(r*64))),
        self.idx(a_uop, self.add(global_a_off, self.s_const(r*K))),
      )
    for r in range(0, 64, 8):
      self.cp_async(
        self.idx(smem_b, self.add(store_smem_b_off, self.s_const(r*128))),
        self.idx(b_uop, self.add(global_b_off, self.s_const(r*self.N))),
      )

  def emit_prefetch_fp16(self, smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop):
    """Emit cp_async calls for one K-block prefetch (256x128 fp16 variant)."""
    K = self.K
    # A: 4 rows at 0, 32, 64, 96 stride
    for r in [0, 32, 64, 96]:
      self.cp_async(
        self.idx(smem_a, self.add(store_smem_a_off, self.s_const(r*64))),
        self.idx(a_uop, self.add(global_a_off, self.s_const(r*K))),
      )
    # B: 2 rows at 0, 16 stride
    for r in [0, 16]:
      self.cp_async(
        self.idx(smem_b, self.add(store_smem_b_off, self.s_const(r*128))),
        self.idx(b_uop, self.add(global_b_off, self.s_const(r*self.N))),
      )

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

    # ldmatrix loads for A
    for i in range(variant.M_FRAGS):
      smem_ptr_a = self.idx(smem_a, ld_a_offsets[k_idx][i])
      self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags[i], self.s_const(0)), smem_ptr_a),
        arg="__ldmatrix_a_elems({0}, {1});")

    # ldmatrix loads for B (pairs)
    for pair in range(4):
      smem_ptr_b = self.idx(smem_b, ld_b_offsets[k_idx][pair])
      self.u(Ops.CUSTOM, dtypes.void,
        (self.idx(b_frags[2*pair], self.s_const(0)), self.idx(b_frags[2*pair+1], self.s_const(0)), smem_ptr_b),
        arg="__ldmatrix_b_elems({0}, {1}, {2});")

    # M_FRAGS x N_FRAGS WMMA ops
    for am in range(variant.M_FRAGS):
      a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags[am], self.s_const(0)),), tag=f"load_a_{k_idx}_{am}")
      for bn in range(variant.N_FRAGS):
        acc_index = am*variant.N_FRAGS + bn
        b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags[bn], self.s_const(0)),), tag=f"load_b_{k_idx}_{bn}")
        c_ptr = self.idx(acc_regs[acc_index], self.s_const(0))
        c_val = self.u(Ops.LOAD, acc_scalar_dtype.vec(4), (c_ptr,), tag=f"load_c_{k_idx}_{am}_{bn}")
        out_val = self.u(Ops.WMMA, acc_scalar_dtype.vec(4), (a_val, b_val, c_val), arg=wmma_arg)
        self.u(Ops.STORE, dtypes.void, (c_ptr, out_val))

  def compute_flat_ldmatrix_offsets(self, wg_m, wg_n, wg_threads):
    """Compute ldmatrix offsets for flat (non-swizzled) SMEM layout."""
    variant = self.variant
    wg_threads_mod_8 = self.u(Ops.MOD, dtypes.int, (wg_threads, self.s_const(8)))
    wg_threads_div_8 = self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(8)))
    wg_threads_div_8_mod_2 = self.u(Ops.MOD, dtypes.int, (wg_threads_div_8, self.s_const(2)))
    wg_threads_div_16 = self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(16)))

    lsa_term1 = self.mul(wg_m, self.s_const(16*64))
    lsa_term2 = self.mul(wg_threads_mod_8, self.s_const(64))
    lsa_term3 = self.mul(wg_threads_div_8_mod_2, self.s_const(64*8))
    lsa_term4 = self.mul(wg_threads_div_16, self.s_const(8))
    load_smem_a_base = self.add(self.add(lsa_term1, lsa_term2), self.add(lsa_term3, lsa_term4))

    lsb_term1 = self.mul(wg_n, self.s_const(16))
    lsb_term2 = self.mul(wg_threads_mod_8, self.s_const(128))
    lsb_term3 = self.mul(wg_threads_div_8_mod_2, self.s_const(128*8))
    load_smem_b_base = self.add(self.add(lsb_term1, lsb_term2), self.add(lsb_term3, lsa_term4))

    # Build offset tables [k_idx][frag_idx]
    num_k_sub = variant.K_TILE // variant.MMA_K
    ld_a_offsets = []
    ld_b_offsets = []
    for k_idx in range(num_k_sub):
      # A offsets: base + k_idx*16, and +32*64 for second fragment
      a_off_base = self.add(load_smem_a_base, self.s_const(k_idx*16))
      ld_a_offsets.append([self.add(a_off_base, self.s_const(i*32*64)) for i in range(variant.M_FRAGS)])
      # B offsets: 4 pairs, each offset by 32 in the N dimension, k advances by 16*128
      b_off_base = self.add(load_smem_b_base, self.s_const(k_idx*16*128))
      ld_b_offsets.append([self.add(b_off_base, self.s_const(pair*32)) for pair in range(4)])

    return ld_a_offsets, ld_b_offsets

  def compute_swizzled_ldmatrix_offsets_fp32(self, wg_m, wg_n, threads):
    """Compute ldmatrix offsets for swizzled SMEM layout (fp32 accumulator kernel)."""
    variant = self.variant
    threads_mod_8 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(8)))
    threads_mod_16 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(16)))
    threads_div_16 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(16)))
    threads_div_16_mod_2 = self.u(Ops.MOD, dtypes.int, (threads_div_16, self.s_const(2)))

    # load_smem_a_row = ((wg_m * 16) + (threads % 16)) * 64
    load_smem_a_row = self.mul(self.add(self.mul(wg_m, self.s_const(16)), threads_mod_16), self.s_const(64))
    # load_smem_a_phase = (threads / 16) % 2
    load_smem_a_phase = threads_div_16_mod_2

    # load_smem_b_row = (threads % 16) * 128
    load_smem_b_row = self.mul(threads_mod_16, self.s_const(128))
    # load_smem_b_phase = (wg_n * 2) + ((threads / 16) % 2)
    load_smem_b_phase = self.add(self.mul(wg_n, self.s_const(2)), threads_div_16_mod_2)

    num_k_sub = variant.K_TILE // variant.MMA_K
    ld_a_offsets = []
    ld_b_offsets = []

    for k_idx in range(num_k_sub):
      # A: load_smem_a_row + (((load_smem_a_phase + k_idx*2) ^ (threads % 8)) * 8)
      phase_k = self.add(load_smem_a_phase, self.s_const(k_idx*2))
      a_swiz = self.mul(self.xor(phase_k, threads_mod_8), self.s_const(8))
      a_off_0 = self.add(load_smem_a_row, a_swiz)
      ld_a_offsets.append([a_off_0, self.add(a_off_0, self.s_const(32*64))])

      # B: 4 pairs with phase offsets 0, 4, 8, 12
      b_row_k = self.add(load_smem_b_row, self.s_const(k_idx*16*128))
      b_pair_offsets = []
      for pair in range(4):
        phase_b = self.add(load_smem_b_phase, self.s_const(pair*4))
        b_swiz = self.mul(self.xor(phase_b, threads_mod_8), self.s_const(8))
        b_pair_offsets.append(self.add(b_row_k, b_swiz))
      ld_b_offsets.append(b_pair_offsets)

    return ld_a_offsets, ld_b_offsets

  def compute_swizzled_ldmatrix_offsets_fp16(self, wg_m, wg_n, wg_threads, threads):
    """Compute ldmatrix offsets for swizzled SMEM layout (fp16 accumulator kernel)."""
    variant = self.variant
    threads_mod_8 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(8)))
    threads_mod_16 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(16)))
    threads_div_16 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(16)))
    threads_div_16_mod_2 = self.u(Ops.MOD, dtypes.int, (threads_div_16, self.s_const(2)))
    wg_threads_div_16 = self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(16)))

    # load_smem_a_row = ((wg_m * 16) + (threads % 16)) * 64
    load_smem_a_row = self.mul(self.add(self.mul(wg_m, self.s_const(16)), threads_mod_16), self.s_const(64))
    # load_smem_a_phase = (threads / 16) % 2
    load_smem_a_phase = threads_div_16_mod_2

    # load_smem_b_row = (threads % 16) * 128
    load_smem_b_row = self.mul(threads_mod_16, self.s_const(128))
    # load_smem_b_phase = (wg_n * 2) + (wg_threads / 16)
    load_smem_b_phase = self.add(self.mul(wg_n, self.s_const(2)), wg_threads_div_16)

    num_k_sub = variant.K_TILE // variant.MMA_K  # 2 for fp16 variant
    ld_a_offsets = []
    ld_b_offsets = []

    # fp16 variant has K=0 and K=1 offsets for A/B
    # k_0 uses phases 0, 4 for A; k_1 uses phases 2, 6 for A
    for k_idx in range(num_k_sub):
      # A offsets for all 4 M fragments
      a_frag_offsets = []
      for frag in range(variant.M_FRAGS):
        # phase offset: frag 0,1 use base phase, frag 2,3 use base phase + 4
        frag_phase_base = (frag // 2) * 4
        phase_k = self.add(load_smem_a_phase, self.s_const(k_idx*2 + frag_phase_base))
        a_swiz = self.mul(self.xor(phase_k, threads_mod_8), self.s_const(8))
        a_row = self.add(load_smem_a_row, self.s_const((frag % 2) * 64*64))
        a_frag_offsets.append(self.add(a_row, a_swiz))
      ld_a_offsets.append(a_frag_offsets)

      # B: 4 pairs with phase offsets 0, 4, 8, 12, k advances by 16*128
      b_row_k = self.add(load_smem_b_row, self.s_const(k_idx*16*128))
      b_pair_offsets = []
      for pair in range(4):
        phase_b = self.add(load_smem_b_phase, self.s_const(pair*4))
        b_swiz = self.mul(self.xor(phase_b, threads_mod_8), self.s_const(8))
        b_pair_offsets.append(self.add(b_row_k, b_swiz))
      ld_b_offsets.append(b_pair_offsets)

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
    variant = self.variant
    wg_c_off_term1 = self.mul(self.mul(grid_m, self.s_const(variant.M_TILE)), self.s_const(self.N))
    wg_c_off_term2 = self.mul(grid_n, self.s_const(variant.N_TILE))
    wg_c_off_term3 = self.mul(wg_m, self.s_const(16*self.N))
    wg_c_off_term4 = self.mul(wg_n, self.s_const(16))
    wg_c_off = self.add(self.add(wg_c_off_term1, wg_c_off_term2), self.add(wg_c_off_term3, wg_c_off_term4))

    tc_term1 = self.mul(self.u(Ops.MOD, dtypes.int, (wg_threads, self.s_const(4))), self.s_const(2))
    tc_term2 = self.mul(self.u(Ops.MOD, dtypes.int, (self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(4))), self.s_const(8))), self.s_const(self.N))
    thread_c_off = self.add(tc_term1, tc_term2)

    for am in range(variant.M_FRAGS):
      for bn in range(variant.N_FRAGS):
        vec_idx = am*variant.N_FRAGS + bn
        c_vec = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(acc_regs[vec_idx], self.s_const(0)),))
        for k in range(4):
          val = self.u(Ops.GEP, dtypes.float, (c_vec,), (k,))
          k_off = [0, 1, 8*self.N, 8*self.N + 1][k]
          idx_val = wg_c_off
          if am == 1:
            idx_val = self.add(idx_val, self.s_const(32*self.N))
          idx_val = self.add(idx_val, thread_c_off)
          bn_mult = (bn % 2) + 4 * (bn // 2)
          idx_val = self.add(idx_val, self.s_const(k_off + bn_mult*8))
          self.u(Ops.STORE, dtypes.void, (self.idx(c_uop, idx_val), val))

  def emit_smem_float4_epilogue(self, acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem):
    """Emit smem-based float4 epilogue (max fp32 kernel)."""
    variant = self.variant
    N_PAD = variant.N_PAD
    smem_d = self.cast(smem, dtypes.float.ptr(AddrSpace.LOCAL))

    # smem_d_off = (wg_m * 16 * N_PAD) + (wg_n * 16) + ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * N_PAD)
    wg_threads_mod_4 = self.u(Ops.MOD, dtypes.int, (wg_threads, self.s_const(4)))
    wg_threads_div_4 = self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(4)))
    wg_threads_div_4_mod_8 = self.u(Ops.MOD, dtypes.int, (wg_threads_div_4, self.s_const(8)))

    smem_d_off = self.add(
      self.add(self.mul(wg_m, self.s_const(16*N_PAD)), self.mul(wg_n, self.s_const(16))),
      self.add(self.mul(wg_threads_mod_4, self.s_const(2)), self.mul(wg_threads_div_4_mod_8, self.s_const(N_PAD)))
    )

    # Store first half of accumulators (acc_frag_0_*) to smem
    bn_col_map = [0, 1, 4, 5, 8, 9, 12, 13]
    for bn in range(8):
      vec_idx = bn  # am=0
      c_vec = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(acc_regs[vec_idx], self.s_const(0)),))
      col = bn_col_map[bn]
      for k in range(4):
        val = self.u(Ops.GEP, dtypes.float, (c_vec,), (k,))
        k_off = [0, 1, 8*N_PAD, 8*N_PAD + 1][k]
        off = self.add(smem_d_off, self.s_const(col*8 + k_off))
        self.u(Ops.STORE, dtypes.void, (self.idx(smem_d, off), val))

    self.barrier()

    # load_smem_d_off = ((threads % 32) * 4) + ((threads / 32) * N_PAD)
    threads_mod_32 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(32)))
    threads_div_32 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(32)))
    load_smem_d_off = self.add(self.mul(threads_mod_32, self.s_const(4)), self.mul(threads_div_32, self.s_const(N_PAD)))

    # Load 8 float4s for first half
    d_0_regs = [self.u(Ops.DEFINE_REG, dtypes.float.vec(4).ptr(1, AddrSpace.REG), (), f"d_0_{i}") for i in range(8)]
    for i in range(8):
      row_off = [0, 4, 8, 12, 16, 20, 24, 28][i]
      ptr = self.idx(smem_d, self.add(load_smem_d_off, self.s_const(row_off * N_PAD)))
      self.u(Ops.CUSTOM, dtypes.void, (self.idx(d_0_regs[i], self.s_const(0)), ptr),
        arg=f"{{{{float4 tmp = *((float4*){{1}}); *{{0}} = tmp;}}}} /* load_d0_{i} */")

    self.barrier()

    # Store second half of accumulators (acc_frag_1_*) to smem
    for bn in range(8):
      vec_idx = 8 + bn  # am=1
      c_vec = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(acc_regs[vec_idx], self.s_const(0)),))
      col = bn_col_map[bn]
      for k in range(4):
        val = self.u(Ops.GEP, dtypes.float, (c_vec,), (k,))
        k_off = [0, 1, 8*N_PAD, 8*N_PAD + 1][k]
        off = self.add(smem_d_off, self.s_const(col*8 + k_off))
        self.u(Ops.STORE, dtypes.void, (self.idx(smem_d, off), val))

    self.barrier()

    # Load 8 float4s for second half
    d_1_regs = [self.u(Ops.DEFINE_REG, dtypes.float.vec(4).ptr(1, AddrSpace.REG), (), f"d_1_{i}") for i in range(8)]
    for i in range(8):
      row_off = [0, 4, 8, 12, 16, 20, 24, 28][i]
      ptr = self.idx(smem_d, self.add(load_smem_d_off, self.s_const(row_off * N_PAD)))
      self.u(Ops.CUSTOM, dtypes.void, (self.idx(d_1_regs[i], self.s_const(0)), ptr),
        arg=f"{{{{float4 tmp = *((float4*){{1}}); *{{0}} = tmp;}}}} /* load_d1_{i} */")

    self.barrier()

    # Write to global memory
    global_d_base = self.add(
      self.add(self.mul(self.mul(grid_m, self.s_const(variant.M_TILE)), self.s_const(self.N)), self.mul(grid_n, self.s_const(variant.N_TILE))),
      self.add(self.mul(threads_mod_32, self.s_const(4)), self.mul(threads_div_32, self.s_const(self.N)))
    )

    # First half: rows 0,4,8,12,16,20,24,28
    for i, row in enumerate([0, 4, 8, 12, 16, 20, 24, 28]):
      d_val = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(d_0_regs[i], self.s_const(0)),), tag=f"store_d0_{i}")
      ptr = self.idx(c_uop, self.add(global_d_base, self.s_const(row*self.N)))
      self.u(Ops.CUSTOM, dtypes.void, (ptr, d_val), arg=f"*((float4*){{0}}) = {{1}}; /* store_d0_{i} */")

    # Second half: rows 32,36,40,44,48,52,56,60
    for i, row in enumerate([32, 36, 40, 44, 48, 52, 56, 60]):
      d_val = self.u(Ops.LOAD, dtypes.float.vec(4), (self.idx(d_1_regs[i], self.s_const(0)),), tag=f"store_d1_{i}")
      ptr = self.idx(c_uop, self.add(global_d_base, self.s_const(row*self.N)))
      self.u(Ops.CUSTOM, dtypes.void, (ptr, d_val), arg=f"*((float4*){{0}}) = {{1}}; /* store_d1_{i} */")

  def emit_smem_half2_epilogue(self, acc_regs, c_uop, grid_m, grid_n, wg_m, wg_n, wg_threads, threads, smem):
    """Emit smem-based half2/half8 epilogue (max fp16 kernel)."""
    variant = self.variant
    SMEM_N_WIDTH = variant.N_PAD  # 136

    smem32_d = self.cast(smem, dtypes.half.vec(2).ptr(AddrSpace.LOCAL))
    smem128_d = self.cast(smem, dtypes.half.vec(8).ptr(AddrSpace.LOCAL))
    out128_d = self.cast(c_uop, dtypes.half.vec(8).ptr())

    # smem32_d_write_off = (wg_m * 8 * (SMEM_N_WIDTH / 2)) + (wg_n * (16 / 2))
    smem32_d_write_off = self.add(
      self.mul(wg_m, self.s_const(8 * (SMEM_N_WIDTH // 2))),
      self.mul(wg_n, self.s_const(16 // 2))
    )
    # smem32_d_thread_off = ((wg_threads / 4) * (SMEM_N_WIDTH / 2)) + (wg_threads % 4)
    wg_threads_div_4 = self.u(Ops.IDIV, dtypes.int, (wg_threads, self.s_const(4)))
    wg_threads_mod_4 = self.u(Ops.MOD, dtypes.int, (wg_threads, self.s_const(4)))
    smem32_d_thread_off = self.add(
      self.mul(wg_threads_div_4, self.s_const(SMEM_N_WIDTH // 2)),
      wg_threads_mod_4
    )

    # smem128_d_read_off = ((threads / 16) * (SMEM_N_WIDTH / 8)) + (threads % 16)
    threads_div_16 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(16)))
    threads_mod_16 = self.u(Ops.MOD, dtypes.int, (threads, self.s_const(16)))
    smem128_d_read_off = self.add(
      self.mul(threads_div_16, self.s_const(SMEM_N_WIDTH // 8)),
      threads_mod_16
    )

    # out128_d_off = ((grid_m * 256) * (N / 8)) + (grid_n * (128 / 8)) + ...
    threads_div_128 = self.u(Ops.IDIV, dtypes.int, (threads, self.s_const(128)))
    threads_div_16_mod_8 = self.u(Ops.MOD, dtypes.int, (threads_div_16, self.s_const(8)))
    out128_d_off = self.add(
      self.add(
        self.mul(self.mul(grid_m, self.s_const(variant.M_TILE)), self.s_const(self.N // 8)),
        self.mul(grid_n, self.s_const(variant.N_TILE // 8))
      ),
      self.add(
        self.add(
          self.mul(threads_div_128, self.s_const(16 * (self.N // 8))),
          self.mul(threads_div_16_mod_8, self.s_const(self.N // 8))
        ),
        threads_mod_16
      )
    )

    # Column mapping for acc fragments: bn -> column multiplier
    bn_col_map = [0, 1, 4, 5, 8, 9, 12, 13]

    # Process each M fragment (0..3)
    for am in range(variant.M_FRAGS):
      # Write first half (x, y) of each half4 acc to SMEM
      self.barrier()
      for bn in range(variant.N_FRAGS):
        vec_idx = am * variant.N_FRAGS + bn
        c_vec = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(acc_regs[vec_idx], self.s_const(0)),))
        col = bn_col_map[bn]
        # half2(x, y)
        x_val = self.u(Ops.GEP, dtypes.half, (c_vec,), (0,))
        y_val = self.u(Ops.GEP, dtypes.half, (c_vec,), (1,))
        h2_val = self.u(Ops.VECTORIZE, dtypes.half.vec(2), (x_val, y_val))
        off = self.add(self.add(smem32_d_write_off, smem32_d_thread_off), self.s_const(col * 4))
        self.u(Ops.STORE, dtypes.void, (self.idx(smem32_d, off), h2_val))

      # Read and write first two 8-element chunks
      self.barrier()
      out_off_base = self.add(out128_d_off, self.s_const(am * 64 * (self.N // 8)))
      # out128_d[out128_d_off + (0 * (N / 8))] = smem128_d[smem128_d_read_off]
      ptr0 = self.idx(smem128_d, smem128_d_read_off)
      val0 = self.u(Ops.LOAD, dtypes.half.vec(8), (ptr0,), tag=f"epi_{am}_0_load")
      out_ptr0 = self.idx(out128_d, out_off_base)
      self.u(Ops.STORE, dtypes.void, (out_ptr0, val0))
      # out128_d[out128_d_off + (32 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))]
      ptr1 = self.idx(smem128_d, self.add(smem128_d_read_off, self.s_const(16 * (SMEM_N_WIDTH // 8))))
      val1 = self.u(Ops.LOAD, dtypes.half.vec(8), (ptr1,), tag=f"epi_{am}_32_load")
      out_ptr1 = self.idx(out128_d, self.add(out_off_base, self.s_const(32 * (self.N // 8))))
      self.u(Ops.STORE, dtypes.void, (out_ptr1, val1))

      # Write second half (z, w) of each half4 acc to SMEM
      self.barrier()
      for bn in range(variant.N_FRAGS):
        vec_idx = am * variant.N_FRAGS + bn
        c_vec = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(acc_regs[vec_idx], self.s_const(0)),))
        col = bn_col_map[bn]
        # half2(z, w)
        z_val = self.u(Ops.GEP, dtypes.half, (c_vec,), (2,))
        w_val = self.u(Ops.GEP, dtypes.half, (c_vec,), (3,))
        h2_val = self.u(Ops.VECTORIZE, dtypes.half.vec(2), (z_val, w_val))
        off = self.add(self.add(smem32_d_write_off, smem32_d_thread_off), self.s_const(col * 4))
        self.u(Ops.STORE, dtypes.void, (self.idx(smem32_d, off), h2_val))

      # Read and write second two 8-element chunks
      self.barrier()
      # out128_d[out128_d_off + (8 * (N / 8))] = smem128_d[smem128_d_read_off]
      ptr2 = self.idx(smem128_d, smem128_d_read_off)
      val2 = self.u(Ops.LOAD, dtypes.half.vec(8), (ptr2,), tag=f"epi_{am}_8_load")
      out_ptr2 = self.idx(out128_d, self.add(out_off_base, self.s_const(8 * (self.N // 8))))
      self.u(Ops.STORE, dtypes.void, (out_ptr2, val2))
      # out128_d[out128_d_off + (40 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))]
      ptr3 = self.idx(smem128_d, self.add(smem128_d_read_off, self.s_const(16 * (SMEM_N_WIDTH // 8))))
      val3 = self.u(Ops.LOAD, dtypes.half.vec(8), (ptr3,), tag=f"epi_{am}_40_load")
      out_ptr3 = self.idx(out128_d, self.add(out_off_base, self.s_const(40 * (self.N // 8))))
      self.u(Ops.STORE, dtypes.void, (out_ptr3, val3))

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
    grid_m = self.u(Ops.SPECIAL, dtypes.int, (self.s_const(M // variant.M_TILE),), "gidx0")
    grid_n = self.u(Ops.SPECIAL, dtypes.int, (self.s_const(N // variant.N_TILE),), "gidx1")
    threads = self.u(Ops.SPECIAL, dtypes.int, (self.s_const(variant.block_threads),), "lidx0")

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
    smem_a_stages = []
    smem_b_stages = []
    offset = 0
    for s in range(variant.stages):
      smem_a_stages.append(self.cast(self.add(smem, self.s_const(offset)), dtypes.half.ptr(AddrSpace.LOCAL)))
      offset += variant.A_stage_bytes
    for s in range(variant.stages):
      smem_b_stages.append(self.cast(self.add(smem, self.s_const(offset)), dtypes.half.ptr(AddrSpace.LOCAL)))
      offset += variant.B_stage_bytes

    # Compute ldmatrix offsets
    ld_a_offsets, ld_b_offsets = self.compute_ldmatrix_offsets(wg_m, wg_n, wg_threads, threads)

    # Register definitions
    num_acc_regs = variant.M_FRAGS * variant.N_FRAGS
    acc_regs = [self.u(Ops.DEFINE_REG, acc_scalar_dtype.vec(4).ptr(1, AddrSpace.REG), (), f"acc_reg_{i}") for i in range(num_acc_regs)]
    a_frags = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_{i}") for i in range(variant.M_FRAGS)]
    b_frags = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_{i}") for i in range(variant.N_FRAGS)]

    # Initialize accumulators to zero
    if variant.acc_dtype == "float":
      zero = self.f_const(0.0)
      v_zero = self.u(Ops.VECTORIZE, dtypes.float.vec(4), (zero, zero, zero, zero))
    else:
      zero = self.h_const(0.0)
      v_zero = self.u(Ops.VECTORIZE, dtypes.half.vec(4), (zero, zero, zero, zero))
    for i in range(num_acc_regs):
      self.u(Ops.STORE, dtypes.void, (self.idx(acc_regs[i], self.s_const(0)), v_zero))

    # WMMA descriptor
    if variant.acc_dtype == "float":
      wmma_arg = ("WMMA_8_16_16_half_float", (8, 16, 16), dtypes.half, dtypes.float, "CUDA", 32, (((0, 8),), ((0, 4),), ((0, 4),)), ())
    else:
      wmma_arg = ("WMMA_8_16_16_half_half", (8, 16, 16), dtypes.half, dtypes.half, "CUDA", 32, (((0, 8),), ((0, 4),), ((0, 4),)), ())

    num_k_blocks = K // variant.K_TILE

    self.barrier()

    if variant.stages == 1:
      # Single-buffer, 1-stage pipeline (flat_smem_input)
      smem_a = smem_a_stages[0]
      smem_b = smem_b_stages[0]
      self.emit_prefetch(smem_a, smem_b, store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off_loop = self.add(global_a_off, self.s_const(variant.K_TILE))
      global_b_off_loop = self.add(global_b_off, self.s_const(variant.K_TILE * N))
      self.barrier()

      # Main loop
      block_k = self.u(Ops.RANGE, dtypes.int, (self.s_const(num_k_blocks),), (0, AxisType.LOOP))
      self.cp_wait_prior(0)
      self.barrier()

      num_k_sub = variant.K_TILE // variant.MMA_K
      for k_idx in range(num_k_sub):
        self.emit_wmma_block(k_idx, smem_a, smem_b, ld_a_offsets, ld_b_offsets, acc_regs, a_frags, b_frags, wmma_arg)

      self.barrier()

      # Prefetch next iteration
      cond = self.u(Ops.CMPLT, dtypes.bool, (block_k, self.s_const(num_k_blocks - 1)))
      if_uop = self.u(Ops.IF, dtypes.void, (cond,))
      curr_ga_off = self.add(global_a_off_loop, self.mul(block_k, self.s_const(variant.K_TILE)))
      curr_gb_off = self.add(global_b_off_loop, self.mul(block_k, self.s_const(variant.K_TILE * N)))
      self.emit_prefetch(smem_a, smem_b, store_smem_a_off, store_smem_b_off, curr_ga_off, curr_gb_off, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_uop,))
      self.cp_commit()
      self.u(Ops.END, dtypes.void, (block_k,))

    elif variant.stages == 2:
      # Double-buffer, 2-stage pipeline (max fp32)
      # First prefetch to stage 0
      self.emit_prefetch(smem_a_stages[0], smem_b_stages[0], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off = self.add(global_a_off, self.s_const(variant.K_TILE))
      global_b_off = self.add(global_b_off, self.s_const(variant.K_TILE * N))
      self.barrier()

      # Second prefetch to stage 1
      self.emit_prefetch(smem_a_stages[1], smem_b_stages[1], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off_loop = self.add(global_a_off, self.s_const(variant.K_TILE))
      global_b_off_loop = self.add(global_b_off, self.s_const(variant.K_TILE * N))
      self.cp_wait_prior(0)
      self.barrier()

      # Main loop
      block_k = self.u(Ops.RANGE, dtypes.int, (self.s_const(num_k_blocks),), (0, AxisType.LOOP))

      # Select buffer based on block_k parity
      blk_mod_2 = self.u(Ops.MOD, dtypes.int, (block_k, self.s_const(2)))
      cond_odd = self.u(Ops.CMPEQ, dtypes.bool, (blk_mod_2, self.s_const(0)))
      smem_a_curr = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_odd, smem_a_stages[1], smem_a_stages[0]))
      smem_b_curr = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_odd, smem_b_stages[1], smem_b_stages[0]))

      num_k_sub = variant.K_TILE // variant.MMA_K
      for k_idx in range(num_k_sub):
        self.emit_wmma_block(k_idx, smem_a_curr, smem_b_curr, ld_a_offsets, ld_b_offsets, acc_regs, a_frags, b_frags, wmma_arg)

      self.barrier()

      # Prefetch next+1 iteration (2 ahead)
      cond_prefetch = self.u(Ops.CMPLT, dtypes.bool, (block_k, self.s_const(num_k_blocks - 2)))
      if_prefetch = self.u(Ops.IF, dtypes.void, (cond_prefetch,))
      curr_ga_off = self.add(global_a_off_loop, self.mul(block_k, self.s_const(variant.K_TILE)))
      curr_gb_off = self.add(global_b_off_loop, self.mul(block_k, self.s_const(variant.K_TILE * N)))
      self.emit_prefetch(smem_a_curr, smem_b_curr, store_smem_a_off, store_smem_b_off, curr_ga_off, curr_gb_off, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_prefetch,))
      self.cp_commit()

      # Wait for next iteration's data
      cond_wait = self.u(Ops.CMPLT, dtypes.bool, (block_k, self.s_const(num_k_blocks - 1)))
      if_wait = self.u(Ops.IF, dtypes.void, (cond_wait,))
      self.cp_wait_prior(1)
      self.barrier()
      self.u(Ops.END, dtypes.void, (if_wait,))

      self.u(Ops.END, dtypes.void, (block_k,))

    else:  # stages == 3
      # Triple-buffer, 3-stage pipeline (max fp16)
      # Separate fragment registers for K=0 and K=1, matching a_frag_*_k_0/1 and b_frag_*_k_0/1
      a_frags_k0 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_k0_{i}") for i in range(variant.M_FRAGS)]
      a_frags_k1 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(8).ptr(1, AddrSpace.REG), (), f"a_frag_k1_{i}") for i in range(variant.M_FRAGS)]
      b_frags_k0 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_k0_{i}") for i in range(variant.N_FRAGS)]
      b_frags_k1 = [self.u(Ops.DEFINE_REG, dtypes.half.vec(4).ptr(1, AddrSpace.REG), (), f"b_frag_k1_{i}") for i in range(variant.N_FRAGS)]

      # Prefetch first tile to stage 0
      self.emit_prefetch(smem_a_stages[0], smem_b_stages[0], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off = self.add(global_a_off, self.s_const(variant.K_TILE))
      global_b_off = self.add(global_b_off, self.s_const(variant.K_TILE * N))

      # Prefetch second tile to stage 1
      self.emit_prefetch(smem_a_stages[1], smem_b_stages[1], store_smem_a_off, store_smem_b_off, global_a_off, global_b_off, a_uop, b_uop)
      self.cp_commit()
      global_a_off = self.add(global_a_off, self.s_const(variant.K_TILE))
      global_b_off = self.add(global_b_off, self.s_const(variant.K_TILE * N))

      # Wait on first prefetch (stage 0) and sync
      self.cp_wait_prior(1)
      self.barrier()

      # Pre-load K=0 elements for the first tile (from stage 0) into K=0 registers
      for i in range(variant.M_FRAGS):
        smem_ptr_a = self.idx(smem_a_stages[0], ld_a_offsets[0][i])
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k0[i], self.s_const(0)), smem_ptr_a),
          arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        smem_ptr_b = self.idx(smem_b_stages[0], ld_b_offsets[0][pair])
        self.u(Ops.CUSTOM, dtypes.void,
          (self.idx(b_frags_k0[2*pair], self.s_const(0)), self.idx(b_frags_k0[2*pair+1], self.s_const(0)), smem_ptr_b),
          arg="__ldmatrix_b_elems({0}, {1}, {2});")

      # Base offsets for later global prefetches
      global_a_off_loop = global_a_off
      global_b_off_loop = global_b_off

      # Main loop over K-blocks
      block_k = self.u(Ops.RANGE, dtypes.int, (self.s_const(num_k_blocks),), (0, AxisType.LOOP))

      # Select current, next, and store stages based on block_k % 3
      phase_k = self.u(Ops.MOD, dtypes.int, (block_k, self.s_const(3)))
      next_phase_k = self.u(Ops.MOD, dtypes.int, (self.add(block_k, self.s_const(1)), self.s_const(3)))
      store_phase_k = self.u(Ops.MOD, dtypes.int, (self.add(block_k, self.s_const(2)), self.s_const(3)))

      # Current tile SMEM pointers
      cond_0 = self.u(Ops.CMPEQ, dtypes.bool, (phase_k, self.s_const(0)))
      cond_1 = self.u(Ops.CMPEQ, dtypes.bool, (phase_k, self.s_const(1)))
      smem_a_curr = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_curr = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_1, smem_b_stages[1], smem_b_stages[2]))))

      # Next tile SMEM pointers
      cond_n0 = self.u(Ops.CMPEQ, dtypes.bool, (next_phase_k, self.s_const(0)))
      cond_n1 = self.u(Ops.CMPEQ, dtypes.bool, (next_phase_k, self.s_const(1)))
      smem_a_next = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_n0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_n1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_next = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_n0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_n1, smem_b_stages[1], smem_b_stages[2]))))

      # Store (prefetch) tile SMEM pointers
      cond_s0 = self.u(Ops.CMPEQ, dtypes.bool, (store_phase_k, self.s_const(0)))
      cond_s1 = self.u(Ops.CMPEQ, dtypes.bool, (store_phase_k, self.s_const(1)))
      smem_a_store = self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_s0, smem_a_stages[0],
        self.u(Ops.WHERE, smem_a_stages[0].dtype, (cond_s1, smem_a_stages[1], smem_a_stages[2]))))
      smem_b_store = self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_s0, smem_b_stages[0],
        self.u(Ops.WHERE, smem_b_stages[0].dtype, (cond_s1, smem_b_stages[1], smem_b_stages[2]))))

      # 1) Load K=1 fragments for the current tile into dedicated K=1 registers
      for i in range(variant.M_FRAGS):
        smem_ptr_a = self.idx(smem_a_curr, ld_a_offsets[1][i])
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k1[i], self.s_const(0)), smem_ptr_a),
          arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        smem_ptr_b = self.idx(smem_b_curr, ld_b_offsets[1][pair])
        self.u(Ops.CUSTOM, dtypes.void,
          (self.idx(b_frags_k1[2*pair], self.s_const(0)), self.idx(b_frags_k1[2*pair+1], self.s_const(0)), smem_ptr_b),
          arg="__ldmatrix_b_elems({0}, {1}, {2});")

      # 2) MMA for K=0 using previously loaded K=0 fragments
      for am in range(variant.M_FRAGS):
        a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags_k0[am], self.s_const(0)),), tag=f"mma_k0_a_{am}")
        for bn in range(variant.N_FRAGS):
          acc_index = am * variant.N_FRAGS + bn
          b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags_k0[bn], self.s_const(0)),), tag=f"mma_k0_b_{bn}")
          c_ptr = self.idx(acc_regs[acc_index], self.s_const(0))
          c_val = self.u(Ops.LOAD, acc_scalar_dtype.vec(4), (c_ptr,), tag=f"mma_k0_c_{am}_{bn}")
          out_val = self.u(Ops.WMMA, acc_scalar_dtype.vec(4), (a_val, b_val, c_val), arg=wmma_arg)
          self.u(Ops.STORE, dtypes.void, (c_ptr, out_val))

      # 3) Prefetch next+2 iteration (store tile) if needed
      cond_prefetch = self.u(Ops.CMPLT, dtypes.bool, (block_k, self.s_const(num_k_blocks - 2)))
      if_prefetch = self.u(Ops.IF, dtypes.void, (cond_prefetch,))
      curr_ga_off = self.add(global_a_off_loop, self.mul(block_k, self.s_const(variant.K_TILE)))
      curr_gb_off = self.add(global_b_off_loop, self.mul(block_k, self.s_const(variant.K_TILE * N)))
      self.emit_prefetch(smem_a_store, smem_b_store, store_smem_a_off, store_smem_b_off, curr_ga_off, curr_gb_off, a_uop, b_uop)
      self.u(Ops.END, dtypes.void, (if_prefetch,))
      self.cp_commit()

      # 4) Wait for the "next" tile (block_k+1) to become available and sync
      self.cp_wait_prior(1)
      self.barrier()

      # 5) Load K=0 fragments for the next tile into K=0 registers
      #    (this will be used as "K=0" on the next loop iteration)
      for i in range(variant.M_FRAGS):
        smem_ptr_a = self.idx(smem_a_next, ld_a_offsets[0][i])
        self.u(Ops.CUSTOM, dtypes.void, (self.idx(a_frags_k0[i], self.s_const(0)), smem_ptr_a),
          arg="__ldmatrix_a_elems({0}, {1});")
      for pair in range(4):
        smem_ptr_b = self.idx(smem_b_next, ld_b_offsets[0][pair])
        self.u(Ops.CUSTOM, dtypes.void,
          (self.idx(b_frags_k0[2*pair], self.s_const(0)), self.idx(b_frags_k0[2*pair+1], self.s_const(0)), smem_ptr_b),
          arg="__ldmatrix_b_elems({0}, {1}, {2});")

      # 6) MMA for K=1 using the K=1 fragments loaded at the top of the loop
      for am in range(variant.M_FRAGS):
        a_val = self.u(Ops.LOAD, dtypes.half.vec(8), (self.idx(a_frags_k1[am], self.s_const(0)),), tag=f"mma_k1_a_{am}")
        for bn in range(variant.N_FRAGS):
          acc_index = am * variant.N_FRAGS + bn
          b_val = self.u(Ops.LOAD, dtypes.half.vec(4), (self.idx(b_frags_k1[bn], self.s_const(0)),), tag=f"mma_k1_b_{bn}")
          c_ptr = self.idx(acc_regs[acc_index], self.s_const(0))
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

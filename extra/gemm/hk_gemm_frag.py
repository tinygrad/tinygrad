"""
HipKittens hk_bf16_gemm (extra/thunder/amd/gemm_bf16.cpp) reimplemented with tinygrad UOps.

C[M, N] (bf16) = A[M, K] @ B[N, K]^T, fp32 accumulation, exactly the kittens kernel shape:
  - 256x256 output tile per workgroup, K_STEP=64
  - 8 warps in a 2x4 grid, each warp owns a 128x64 warp-tile
  - v_mfma_f32_16x16x32_bf16 on CDNA4 (gfx950) / v_wmma_f32_16x16x16_bf16 (wave32, gfx12) on RDNA4,
    fp32 accumulators
  - shared tiles As/Bs with the kittens st_16x32_s swizzle (16x32 subtiles of 1024B)
  - K stages (STAGES=1: synchronous single buffer)

Validated on gfx1201 hardware (exact for identity-B, rounding-level noise otherwise), rendered
and compiled to gfx950 with comgr for assembly comparison against gemm_bf16.cpp.

What is NOT expressible vs the kittens C++:
  - explicit s_waitcnt vmcnt()/lgkmcnt() pipelining and s_setprio: tinygrad models async copy
    overlap with slot dependencies and emits full workgroup barriers; instruction scheduling
    is left to clang/LLVM
  - direct-to-LDS global loads (buffer_load_lds): tinygrad goes global->reg->LDS

Pipelining status: STAGES=2 gives the kittens-shaped double-buffered pipeline (2 x 64KB LDS
like gemm_bf16.cpp, copies overlap the previous pair's mma's), written with FA/gemm_fragment
conventions: LDS buffers are (2, tile) placeholders indexed by symbolic parity (ko % 2),
which sidesteps static slot choice, fill iterations, predication and duplicate static stores.
It is validated on the CDNA4 emulator for even K/64 >= 4 with a SINGLE workgroup
(bit-close to stages=1 across amt in {4, 6, 8, 10, 14, 16}).

Known remaining bugs (as of the origin/master merge):
  - MULTI-WORKGROUP launches miscompute the pipeline (all workgroups hit the
    "accumulator reads the pre-loop value" failure, while a single workgroup with the
    identical instruction stream computes correctly). Not yet root-caused; suspicious areas:
    per-WG scheduling of the pipeline's explicit barrier+prefetch interleave in the mock
    emulator, or grid-shape-dependent codegen of the parity/prologue copy.
  - Odd or small tile counts (amt < 4 or odd amt) vary by run (looks like a copy/read race
    in the prefetch's clamped tail writes) and currently fall back to stages=1.
See the gating logic in hk_bf16_gemm_kernel for what currently selects the pipeline.

Known blockers found along the way (as of this branch):
  1. pm_split_ranges refactors AxisType.LOCAL ranges against index expressions, scrambling
     the logical<->hardware lane mapping WMMA fragments are written against -> use UOp.special.
  2. memory_coalescing (late/coalesce.py) assumes a single static store per (buffer, index):
     a double-buffered LDS slot written by prologue *and* in-loop copies asserts
     "attempting multiple stores". That blocks the stages=2 pipeline below unless DMC=1
     (which also drops adjacent-store vectorization on the copy-out side).
  3. RDNA4 (gfx12) uses 8-element accumulator fragments, so the 8x4 tile grid needs
     256 fp32 acc registers per thread -> guaranteed spills (0.85 TF vs 96 TF default on
     gfx1201). The kernel is right-sized for CDNA4 (fragsz 4 -> 128 acc regs).

Lane layouts (RDNA4 verified with probing on gfx1201 hardware; CDNA from the mfma docs):
  CDNA (64 thr/warp, 16x16x32):  A/B frag: tile-row = l%16, k = (l//16)*8+i (i in 0..7)
  RDNA4 (32 thr/warp, 16x16x16): A/B frag: tile-row = l%16, k = (l//16)*4+(i%4)+8*(i//4) (i in 0..7)
  both:                        acc frag: CDNA m=(l//16)*4+i (i<4) / RDNA4 m=(l//16)*8+i (i<8), n=l%16

The RDNA4 fragment k-set {k0..3, k0+8..11} is not contiguous, so on RDNA4 the LDS column layout
is block-permuted (4-element blocks within each 16-col group are stored as [0,2,1,3]) making
every fragment 8 contiguous halves (one 16B chunk) on both archs; the copy path applies the
same permutation.

NOTE: thread ids come from UOp.special (like mi350x_uop_matmul.py), not an AxisType.LOCAL
RANGE: pm_split_ranges refactors LOCAL ranges after index expressions and scrambles the
logical<->hardware lane mapping the WMMA fragments are written against.
NOTE 2: WMMA operand/accumulator fragments must carry the fragment length in their UOp shape.
NOTE 3: swizzled addresses are written in provably-contiguous "base + vector-offset" form,
otherwise the devectorizer emits scalar ds_read_u16/ds_write_b16.
"""
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.renderer import Estimates
from tinygrad.helpers import getenv, cdiv

# ---- tile shape (identical to gemm_bf16.cpp; HK_TILE=128 overrides for small-LDS devices) ----
BLOCK_M = BLOCK_N = getenv("HK_TILE", 256)
K_STEP = 64
WARPS_M, WARPS_N = 2, 4
NUM_WARPS = WARPS_M * WARPS_N                                # 8
WARP_TILE_M, WARP_TILE_N = BLOCK_M // WARPS_M, BLOCK_N // WARPS_N   # 128 x 64 (64 x 32 at HK_TILE=128)

def arch_params(arch:str):
  is_cdna = arch.startswith("gfx9")
  if is_cdna:   # CDNA mfma 16x16x32 bf16: in-frag 8, acc 4 (16,16,16 on the acc side)
    return dict(warp_threads=64, dims=(16,16,32), frag_in=8, frag_out=4,
                acc_m=lambda l, i: (l//16)*4 + i, kperm=None, copy_vec=8)
  # RDNA4 wmma 16x16x16 (wave32, gfx12 layout): in-frag 8 (permuted in LDS), acc 8
  return dict(warp_threads=32, dims=(16,16,16), frag_in=8, frag_out=8,
              acc_m=lambda l, i: (l//16)*8 + i, kperm=(0,2,1,3), copy_vec=4)

# ---- kittens st_16x32_s swizzle (byte offset: off ^ (((off % 1024) >> 9) << 5)) ----
# halves-index form: within a 1024B (16x32) subtile, halves-index bit4 ^= row bit3,
# written in "base + vector-offset" form so the devectorizer can prove contiguity.
def st_half_base(r, c, tile_cols:int):
  """swizzled halves-index pre-vector-offset; c is the logical (permuted) column, 4-aligned."""
  subtile_id = (r//16) * (tile_cols//32) + (c//32)
  r16 = r % 16
  flip = (r16 >> 3) & 1
  return subtile_id*512 + r16*32 + (((c % 32) >> 2) ^ (flip << 2)) * 4

def hk_bf16_gemm_kernel(C:UOp, A:UOp, B:UOp, *, arch:str, stages:int=1) -> UOp:
  """C = A @ B^T ; A is (M,K), B is (N,K), C is (M,N). HipKittens tile shape."""
  M, K = A.shape
  N, K2 = B.shape
  assert K == K2 and A.dtype == B.dtype == dtypes.bfloat16 and C.dtype == dtypes.bfloat16
  assert not (M % BLOCK_M or N % BLOCK_N or K % K_STEP), f"dims must be multiples of {(BLOCK_M, BLOCK_N, K_STEP)}"

  ap = arch_params(arch)
  warp_threads, dims = ap["warp_threads"], ap["dims"]
  FRAG_IN, FRAG_OUT, kperm, acc_m, CPV = ap["frag_in"], ap["frag_out"], ap["kperm"], ap["acc_m"], ap["copy_vec"]
  NUM_THREADS = NUM_WARPS * warp_threads
  TC_M, TC_N, TC_K = dims
  MT, NT = WARP_TILE_M // TC_M, WARP_TILE_N // TC_N               # 8, 4 tiles per warp
  # permute 4-half blocks within each 16-col group (RDNA4: [0,2,1,3] = swap middle blocks)
  def perm_col(c):
    if kperm is None: return c
    return (c & ~15) | ((((c>>2) & 1) << 1 | ((c>>3) & 1)) << 2) | (c & 3)

  bx, by = UOp.special(N//BLOCK_N, "gidx0"), UOp.special(M//BLOCK_M, "gidx1")
  lane = UOp.special(warp_threads, "lidx0")
  warp = UOp.special(NUM_WARPS, "lidx1")
  warp_row, warp_col = warp // WARPS_N, warp % WARPS_N
  tid = warp*warp_threads + lane

  def smem(slot) -> UOp: return UOp.placeholder((BLOCK_M*K_STEP,), dtypes.bfloat16, slot, AddrSpace.LOCAL)
  As = [smem(2*i) for i in range(stages)]
  Bs = [smem(2*i+1) for i in range(stages)]

  # per-warp accumulator: (MT x NT) 16x16 tiles of FRAG_OUT fp32 per thread
  acc = UOp.placeholder((MT, NT, FRAG_OUT), dtypes.float32, 12, AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))   # FA-style init: self-store, keeps the value flow loop-carried

  # global -> LDS copy: CPV halves per op (16B on CDNA, 8B on RDNA4), thread-major coalescing
  OPS_PER_TILE = BLOCK_M*K_STEP//CPV
  OPR = K_STEP//CPV
  def copy_tile(dst:UOp, src:UOp, base_row:UOp, base_col:UOp, slot:int) -> UOp:
    ir = UOp.range(cdiv(OPS_PER_TILE, NUM_THREADS), slot, AxisType.LOOP)
    j = UOp.range(CPV, slot+1, AxisType.UPCAST)
    chunk = ir*NUM_THREADS + tid
    r, cb = chunk // OPR, chunk % OPR          # row, 4/8-col block
    return dst[st_half_base(r, perm_col(cb*CPV), K_STEP) + j].store(src[base_row + r, base_col + cb*CPV + j]).end(ir, j)

  def load_stage(sidx:int, ko, slot:int, barrier:bool) -> tuple[UOp, UOp]:
    A_r = copy_tile(As[sidx], A, by*BLOCK_M, ko*K_STEP, slot)
    B_r = copy_tile(Bs[sidx], B, bx*BLOCK_N, ko*K_STEP, slot+10)
    bar = UOp.barrier(A_r, B_r) if barrier else UOp.group(A_r, B_r)
    return As[sidx].after(bar), Bs[sidx].after(bar)

  # ---- pipelined path (stages=2) ----
  NIT = cdiv(OPS_PER_TILE, NUM_THREADS)      # copy ops per thread per tile
  def gload_write_tile(dst:UOp, src:UOp, base_row:UOp, kt, slot:int) -> UOp:
    """store one global tile into an LDS slot (loads and stores share the vec range j)."""
    j = UOp.range(CPV, slot, AxisType.UPCAST)
    def one(ir:int) -> UOp:
      chunk = ir*NUM_THREADS + tid
      r, cb = chunk // OPR, chunk % OPR
      return dst[st_half_base(r, perm_col(cb*CPV), K_STEP) + j].store(src[base_row + r, kt*K_STEP + cb*CPV + j])
    return UOp.group(*[one(ir) for ir in range(NIT)]).end(j)

  def compute(acc:UOp, A_l:UOp, B_l:UOp, afters:tuple[UOp, ...], pred:UOp|None=None) -> UOp:
    """One K_STEP=64 iteration: (K_STEP//TC_K) k-chunks unrolled, (MT x NT) mma each, like the kittens main loop.

    pred (optional): a loop-range condition; accumulator stores are predicated on it so the
    first (fill) iteration of a software pipeline can run the body with garbage LDS contents
    without contaminating the accumulator."""
    arow = warp_row*WARP_TILE_M + lane % 16    # fragment tile row in the LDS tile (m)
    brow = warp_col*WARP_TILE_N + lane % 16    # (n)
    ja = UOp.range(FRAG_IN, 701, AxisType.UPCAST)
    jb = UOp.range(FRAG_IN, 702, AxisType.UPCAST)
    acc_k = acc.after(*afters) if afters else acc
    last_store = None
    # in the permuted layout every fragment is 8 contiguous halves starting at an 8-aligned col
    for kk in range(K_STEP//TC_K):
      cc = kk*(TC_K//FRAG_IN) + (lane // 16)   # fragment chunk col (8 halves)
      a_frags = [A_l[st_half_base(arow + mt*16, cc*8, K_STEP) + ja].contract(ja) for mt in range(MT)]
      b_frags = [B_l[st_half_base(brow + nt*16, cc*8, K_STEP) + jb].contract(jb) for nt in range(NT)]
      for mt in range(MT):
        for nt in range(NT):
          cur = acc_k[mt, nt]
          out = UOp.wmma(a_frags[mt], b_frags[nt], cur, dims, 'AMD', warp_threads)
          if pred is not None: out = pred.where(cur, out)
          last_store = acc_k[mt, nt].store(out)
          acc_k = acc_k.after(last_store)
    return last_store

  # ---- K loop ----
  amt = cdiv(K, K_STEP)
  # stages=2 pipeline is validated on the CDNA4 emulator for K % 128 == 0, amt >= 4, and a
  # single workgroup (amt in {4, 6, 8, 10, 14, 16} sweep-verified bit-close to stages=1).
  # Multi-workgroup launches currently miscompute the pipeline (see DIAGNOSIS in the docstring).
  multi_wg = (M // BLOCK_M) * (N // BLOCK_N) > 1
  _stages = stages if (K % (2*K_STEP) == 0 and amt >= 4 and not multi_wg) else 1
  if _stages == 1:
    ko = UOp.range(amt, 600, AxisType.LOOP)
    A_l, B_l = load_stage(0, ko, 100, barrier=True)
    last = compute(acc, A_l, B_l, afters=(ko,))
    acc = acc.after(last.barrier().end(ko))
  else:
    # Double-buffered pipeline on FA/gemm_fragment conventions: each LDS buffer is a
    # (2, tile) placeholder indexed by symbolic parity (ko % 2) -- no static slot choice,
    # no duplicate static stores (memory_coalescing-safe), no fill iteration, no predication.
    def smem2(slot) -> UOp: return UOp.placeholder((2, BLOCK_M*K_STEP), dtypes.bfloat16, slot, AddrSpace.LOCAL)
    A_l, B_l = smem2(0), smem2(1)

    def copy_stage(dst:UOp, src:UOp, base_row:UOp, kt, slot:int) -> UOp:
      """store one global tile into dst (a (tile,) LDS view, already parity-indexed)."""
      j = UOp.range(CPV, slot, AxisType.UPCAST)
      ir = UOp.range(cdiv(OPS_PER_TILE, NUM_THREADS), slot+1, AxisType.LOOP)
      chunk = ir*NUM_THREADS + tid
      r, cc = chunk // OPR, chunk % OPR
      return dst[st_half_base(r, perm_col(cc*CPV), K_STEP) + j].store(src[base_row + r, kt*K_STEP + cc*CPV + j]).end(ir, j)

    # prologue: tile 0 into slot 0 of both buffers, barrier before first read
    g0 = UOp.group(copy_stage(A_l[0], A, by*BLOCK_M, UOp.const(dtypes.weakint, 0), 100),
                   copy_stage(B_l[0], B, bx*BLOCK_N, UOp.const(dtypes.weakint, 0), 110))
    bar0 = UOp.barrier(g0)
    # Double-buffered pipeline with symbolic-parity slot indexing (FA/gemm_fragment
    # conventions): slot ko%2 holds k-tile ko; compute of tile ko overlaps the prefetch
    # copy of tile ko+1 into the other slot. No fill iteration, no predication, one static
    # store per slot scope, add_war_barrier protects each hand-off.
    ko = UOp.range(amt, 600, AxisType.LOOP)
    pr, pn = ko % 2, (ko+1) % 2                      # slot of the tile being computed / being prefetched
    kt_next = UOp.minimum(ko+1, amt-1)               # clamped tail prefetch (its data is unused)
    last = compute(acc, A_l[pr].after(bar0, ko), B_l[pr].after(bar0, ko), afters=(ko,))
    ga = UOp.group(copy_stage(A_l[pn].after(last), A, by*BLOCK_M, kt_next, 130),
                   copy_stage(B_l[pn].after(last), B, bx*BLOCK_N, kt_next, 140))
    ko_end = UOp.group(last, ga).end(ko)
    acc = acc.after(ko_end)

  # ---- epilogue: per-thread fragment stores, cast to bf16 (scalar per fragment element) ----
  mt, nt = UOp.range(MT, 801, AxisType.LOOP), UOp.range(NT, 802, AxisType.LOOP)
  def store_i(i:int) -> UOp:
    crow = by*BLOCK_M + warp_row*WARP_TILE_M + mt*16 + acc_m(lane, i)
    ccol = bx*BLOCK_N + warp_col*WARP_TILE_N + nt*16 + lane % 16
    return C[crow, ccol].store(acc[mt, nt, i].cast(dtypes.bfloat16))
  out_st = UOp.group(*[store_i(i) for i in range(FRAG_OUT)])
  return out_st.end(mt, nt).sink(arg=KernelInfo(name="hk_bf16_gemm",
    estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K+M*N)*2)))

def hk_bf16_gemm_tiny(a:Tensor, b:Tensor, stages:int=1) -> Tensor:
  """C = a @ b.T for bf16 a (M,K), b (N,K) with the HipKittens-shaped tinygrad kernel."""
  arch = Device[a.device].renderer.target.arch
  c = Tensor.empty(a.shape[0], b.shape[0], dtype=dtypes.bfloat16, device=a.device)
  return c.custom_kernel(a, b, fxn=lambda C, A, B: hk_bf16_gemm_kernel(C, A, B, arch=arch, stages=stages))[0]

if __name__ == "__main__":
  import numpy as np
  from tinygrad import Device
  M = N = K = 512
  # exact test: B = identity -> C must equal A bit-exactly
  a = Tensor.randn(M, K, dtype=dtypes.bfloat16).contiguous()
  bid = Tensor(np.eye(K, N, dtype=np.float32), dtype=dtypes.bfloat16).contiguous()
  cid = hk_bf16_gemm_tiny(a, bid, stages=getenv("STAGES", 1)).realize()
  assert np.array_equal(cid.float().numpy(), a.float().numpy()), "identity test failed"
  # real test: bf16 gemm vs fp32 reference, rounding-level noise
  b = Tensor.randn(N, K, dtype=dtypes.bfloat16).contiguous()
  c = hk_bf16_gemm_tiny(a, b, stages=getenv("STAGES", 1)).realize()
  ref = (a @ b.T).float().realize()
  err = (c.float() - ref).abs().max().item()
  print(f"identity exact, random max err: {err:.5f}")

  # match the real HipKittens hk_bf16_gemm on the (mock) CDNA4 emulator at small sizes.
  # run from the repo root with:  DEV=MOCK+AMD:HIP:gfx950 HK_COMPARE=1 python extra/gemm/hk_gemm_frag.py
  if getenv("HK_COMPARE"):
    from extra.gemm.cdna_asm_gemm import asm_gemm
    assert Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950"), "needs CDNA4 (mock emulator or hardware)"
    def compare(M:int, N:int, K:int, seed:int=0, identity:bool=False):
      np.random.seed(seed)
      An = np.random.randn(M, K)
      Bn = np.eye(K, N) if identity else np.random.randn(K, N)      # (K,N) as expected by asm_gemm
      A = Tensor(An, dtype=dtypes.bfloat16).contiguous()
      B = Tensor(Bn, dtype=dtypes.bfloat16).contiguous()             # (K,N) for asm_gemm
      c_hkc = asm_gemm(A, B).realize().float().numpy()               # real HipKittens hk_bf16_gemm
      c_hkt = hk_bf16_gemm_tiny(A, B.T.contiguous()).realize().float().numpy()
      ref64 = A.float().numpy().astype(np.float64) @ B.float().numpy()
      tag = "ident" if identity else "rand "
      print(f"({M},{N},{K}) {tag}: tiny-vs-kittens {np.abs(c_hkt-c_hkc).max():9.6f}  "
            f"tiny-vs-fp64 {np.abs(c_hkt-ref64).max():9.6f}  kittens-vs-fp64 {np.abs(c_hkc-ref64).max():9.6f}")
      assert np.abs(c_hkt - ref64).max() < 0.26, "tiny kernel must match fp64 at rounding level"
      assert np.abs(c_hkt - c_hkc).max() < 0.51, "tiny kernel must match hipkittens"
    # NOTE: hk_bf16_gemm requires K % 128 == 0 (its prologue+epilogue unconditionally touch
    # k-tiles num_tiles-1 and num_tiles-2); at other K it reads wrong-but-in-bounds global
    # memory on the emulator and on real hardware, so only K%128==0 sizes are checked here.
    compare(256, 256, 128, seed=1)   # single workgroup
    compare(256, 256, 256, seed=2)
    compare(512, 512, 128, seed=3)   # multi workgroup
    compare(256, 256, 128, identity=True)   # bit-exact check

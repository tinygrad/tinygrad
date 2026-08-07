# RDNA3 128x128 half GEMM using WMMA (gfx1100).
# Hand-tuned A/B LDS + v_wmma_f32_16x16x16_f16. Explicit experiment (not wired into Tensor.matmul):
#   from extra.gemm.rdna3_asm_wmma_gemm import can_use_rdna3_wmma_gemm, rdna3_wmma_gemm
#   if can_use_rdna3_wmma_gemm(a, b): c = rdna3_wmma_gemm(a, b)
#   DEV=AMD:AMD python extra/gemm/rdna3_asm_wmma_gemm.py
import functools, time
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.renderer.amd.dsl import s, v, VCC_LO, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import *

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 16
TILES_M, TILES_N = 4, 4
THREADS, ELEM = 128, 2
LDS_A_ROW = BLOCK_K*ELEM   # 32
LDS_B_ROW = BLOCK_N*ELEM  # 256
LDS_A_SIZE = BLOCK_M * LDS_A_ROW  # 4096
LDS_B_SIZE = BLOCK_K * LDS_B_ROW  # 4096
LDS_SIZE = LDS_A_SIZE + LDS_B_SIZE  # 8192
LDS_B_OFF = LDS_A_SIZE
# RDNA3 WMMA operands are 8 VGPRs wide (RDNA4 is 4). With 4 A + 4 B frags that is
# 32+32 regs; the old RDNA4-derived map collided FB+16/FB+24 onto ACC. Repacked:
# FA 44-75, FB 76-107, DA 108-115, DB 116-123, ACC 128-255 (top tile ends at 255).
ACC, DA, DB, FA, FB, ET = 128, 108, 116, 44, 76, 10

def build_kernel(N, arch='gfx1100', out_f32=False):
  assert N % BLOCK_M == 0 and N >= 256
  NO_ALU, NO_DS, NO_GLOBAL = getenv("NO_ALU", 0), getenv("NO_DS", 0), getenv("NO_GLOBAL", 0)
  I, L, B = [], {}, []
  def e(i): I.append(i); return i
  def label(n): L[n] = sum(i.size() for i in I)
  def br(i, t): B.append((len(I)-1, t))

  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  # RDNA3: wg ids in s[2:4] (see amd_lib WGID). RDNA4 uses ttmp[9]/ttmp[7] instead.
  e(s_lshl_b32(s[10], s[2], 7)); e(s_lshl_b32(s[11], s[3], 7))
  e(s_mov_b32(s[12], N)); e(s_lshl_b32(s[13], s[12], 2 if out_f32 else 1))  # C row stride in bytes
  e(s_mul_i32(s[14], s[12], BLOCK_K*ELEM))
  e(s_add_i32(s[17], s[12], -2*BLOCK_K))  # loop bound

  e(v_and_b32_e32(v[1], 31, v[0])); e(v_lshrrev_b32_e32(v[2], 5, v[0]))
  e(v_and_b32_e32(v[3], 1, v[2])); e(v_lshrrev_b32_e32(v[2], 1, v[2]))

  e(v_lshlrev_b32_e32(v[4], 5, v[0]))
  # B is stored col-major in LDS (transpose): off = LDS_B_OFF + col*32 + k*2, col=(tid%8)*16+j.
  # v[5] base = LDS_B_OFF + (tid%8)*512 + (tid/8)*2; the 16 cols scatter at stride 32 on store.
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[5], 9, v[48]))   # (tid%8)*512
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_lshlrev_b32_e32(v[48], 1, v[48]))  # (tid/8)*2
  e(v_add_nc_u32_e32(v[5], v[5], v[48])); e(v_add_nc_u32_e32(v[5], LDS_B_OFF, v[5]))

  e(v_add_nc_u32_e32(v[48], s[11], v[0]))
  e(v_mul_lo_u32(v[6], v[48], N*ELEM)); e(v_mov_b32_e32(v[7], 0))
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_mul_lo_u32(v[8], v[48], N*ELEM))
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[48], 5, v[48]))
  e(v_add_nc_u32_e32(v[8], v[8], v[48]))
  e(s_mul_i32(s[15], s[10], ELEM)); e(v_add_nc_u32_e32(v[8], s[15], v[8]))
  e(v_mov_b32_e32(v[9], 0))

  # RDNA3 WMMA needs the A/B fragment REPLICATED in lanes 0-15 and 16-31 (elements_per_thread=16
  # -> 2 copies of the 16x16 tile). So the read addr uses only lane%16; lanes 16-31 read the same
  # LDS as 0-15. Adding a (lane//16)*16 term (RDNA4 layout) fed non-replica data -> undefined odd rows.
  LLA, LLB = 40, 43
  e(v_and_b32_e32(v[50], 15, v[1]))
  e(v_lshlrev_b32_e32(v[LLA], 5, v[50]))
  e(v_lshlrev_b32_e32(v[52], 11, v[2]))
  e(v_add_nc_u32_e32(v[LLA], v[LLA], v[52]))
  e(v_lshlrev_b32_e32(v[LLB], 5, v[50]))
  e(v_lshlrev_b32_e32(v[52], 11, v[3]))
  e(v_add_nc_u32_e32(v[LLB], v[LLB], v[52]))
  e(v_add_nc_u32_e32(v[LLB], LDS_B_OFF, v[LLB]))

  def store_a_lds():
    for i in range(2): e(ds_store_b128(addr=v[4], data0=v[DA+i*4:DA+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
  def store_b_lds():
    # transpose: the 16 cols this thread holds (packed 2/reg in DB) scatter at col stride 32.
    for j in range(16):
      off = j*32; op = ds_store_b16 if j%2==0 else ds_store_b16_d16_hi
      e(op(addr=v[5], data0=v[DB+j//2], offset0=off&0xFF, offset1=off>>8))

  def load_a(tm):
    aoff = tm * 16 * LDS_A_ROW
    b = FA + tm * 8
    e(ds_load_b128(vdst=v[b:b+3], addr=v[LLA], offset0=aoff&0xFF, offset1=aoff>>8))
    e(ds_load_b128(vdst=v[b+4:b+7], addr=v[LLA], offset0=(aoff+16)&0xFF, offset1=(aoff+16)>>8))

  def load_b(bi):
    boff = bi * 512  # col-major: fragment bi = 16 cols at bi*16, contiguous k
    b = FB + bi * 8
    e(ds_load_b128(vdst=v[b:b+3], addr=v[LLB], offset0=boff&0xFF, offset1=boff>>8))
    e(ds_load_b128(vdst=v[b+4:b+7], addr=v[LLB], offset0=(boff+16)&0xFF, offset1=(boff+16)>>8))

  for i in range(0, 128, 2):
    e(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[ACC+i], vdsty=v[ACC+i+1], srcx0=0, srcy0=0))
  e(s_mov_b32(s[16], 0))

  if not NO_GLOBAL:
    for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], addr=v[6], saddr=s[4:5], offset=i*16))
    for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], addr=v[8], saddr=s[6:7], offset=i*16))
    e(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  if not NO_DS:
    store_a_lds(); store_b_lds()
  if not NO_GLOBAL:
    e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
    e(v_add_nc_u32_e32(v[8], s[14], v[8]))

  def compute_block():
    # RDNA3: issue all LDS reads, wait once, then all WMMAs. The RDNA4-style interleaved
    # load_b(2/3) with partial s_waitcnt_lgkmcnt(1) made "load finished before WMMA reads it"
    # timing-dependent on hardware -> nondeterministic NaN. One wait(0) is correct and simpler.
    if not NO_DS:
      for tm in range(TILES_M): load_a(tm)
      for bi in range(TILES_N): load_b(bi)
      e(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
    if not NO_ALU:
      for tn in range(TILES_N):
        for tm in range(TILES_M):
          ac = ACC + (tm*TILES_N+tn)*8
          e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+tn*8:FB+tn*8+7], src2=v[ac:ac+7]))

  def emit_iter_body(load_set='AB'):
    if not NO_DS:
      e(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
      e(s_barrier())
    if not NO_GLOBAL:
      if 'A' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], addr=v[6], saddr=s[4:5], offset=i*16))
        e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
      if 'B' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], addr=v[8], saddr=s[6:7], offset=i*16))
        e(v_add_nc_u32_e32(v[8], s[14], v[8]))
    compute_block()
    if not NO_GLOBAL and not NO_DS: e(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
    if not NO_DS:
      # single-buffered LDS: barrier so every wave finished reading (WMMAs) this block
      # before any wave overwrites LDS with the next block. Without this, waves race and
      # a fast wave clobbers LDS a slow wave is still reading -> nondeterministic NaN/inf.
      e(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)); e(s_barrier())
      store_a_lds(); store_b_lds()
    e(s_add_i32(s[16], s[16], BLOCK_K))

  label('LOOP')
  emit_iter_body(load_set='AB')
  emit_iter_body(load_set='AB')
  e(s_cmp_lt_i32(s[16], s[17])); e(s_cbranch_scc1(simm16=0)); br(I[-1], 'LOOP')

  emit_iter_body(load_set='AB')

  if not NO_DS:
    e(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
    e(s_barrier())
  compute_block()

  label('EPILOGUE')
  e(v_and_b32_e32(v[ET], 15, v[1]))
  # RDNA3 WMMA: lanes 0-15 -> even rows, 16-31 -> odd rows (not RDNA4's +8 row band).
  e(v_lshrrev_b32_e32(v[ET+1], 4, v[1]))
  e(v_lshlrev_b32_e32(v[ET+2], 6, v[2])); e(v_add_nc_u32_e32(v[ET+2], s[11], v[ET+2]))
  e(v_lshlrev_b32_e32(v[ET+3], 6, v[3])); e(v_add_nc_u32_e32(v[ET+3], s[10], v[ET+3]))
  e(v_add_nc_u32_e32(v[ET+3], v[ET+3], v[ET]))

  for tm in range(TILES_M):
    for tn in range(TILES_N):
      ac = ACC + (tm*TILES_N+tn)*8; r_off, c_off = tm*16, tn*16
      e(v_add_nc_u32_e32(v[ET+6], r_off, v[ET+2])); e(v_add_nc_u32_e32(v[ET+6], v[ET+1], v[ET+6]))
      e(v_mul_lo_u32(v[ET+4], v[ET+6], s[12])); e(v_add_nc_u32_e32(v[ET+4], v[ET+4], v[ET+3]))
      if c_off: e(v_add_nc_u32_e32(v[ET+4], c_off, v[ET+4]))
      e(v_lshlrev_b32_e32(v[ET+4], 2 if out_f32 else 1, v[ET+4]))
      for elem in range(8):
        if out_f32:
          e(global_store_b32(addr=v[ET+4], data=v[ac+elem], saddr=s[8:9]))
        else:
          e(v_cvt_f16_f32_e32(v[ET+7], v[ac+elem]))
          e(global_store_b16(addr=v[ET+4], data=v[ET+7], saddr=s[8:9]))
        if elem < 7:
          e(v_add_nc_u32_e32(v[ET+4], s[13], v[ET+4]))
          e(v_add_nc_u32_e32(v[ET+4], s[13], v[ET+4]))  # WMMA acc elems are 2 rows apart

  e(s_waitcnt_vscnt(sdst=NULL, simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())

  for idx, target in B:
    off = (L[target] - sum(i.size() for i in I[:idx+1])) // 4
    assert -32768 <= off <= 32767; I[idx].simm16 = off
  return I

def assemble_kernel(n: int, arch: str = 'gfx1100', out_f32: bool = False) -> tuple[int, int]:
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDRenderer
  insts = build_kernel(n, arch, out_f32=out_f32)
  r = AMDRenderer(Target(device='AMD', arch=arch, renderer='AMD'))
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))
  blob = r.asm(UOp(Ops.PROGRAM, src=(UOp.sink(), lin)), lin)
  return len(insts), len(blob)

def can_use_rdna3_wmma_gemm(a: Tensor, b: Tensor) -> bool:
  """Square half GEMM on RDNA3 ISA renderer, tile-aligned, N>=256."""
  if a.dtype != dtypes.half or b.dtype != dtypes.half: return False
  if a.ndim != 2 or b.ndim != 2: return False
  M, K = a.shape
  K2, N = b.shape
  if K != K2 or M != N or M != K: return False
  if M < 256 or M % BLOCK_M or N % BLOCK_N or K % BLOCK_K: return False
  try:
    from tinygrad.renderer.isa import ISARenderer
    ren = Device[a.device].renderer
    if not isinstance(ren, ISARenderer): return False
    arch = getattr(ren, "arch", None) or getattr(getattr(ren, "target", None), "arch", "") or ""
    if not str(arch).startswith("gfx11"): return False
  except Exception:
    return False
  return True

def _renderer_arch(device) -> str:
  ren = Device[device].renderer
  return getattr(ren, "arch", None) or getattr(getattr(ren, "target", None), "arch", None) or "gfx1100"

def _asm_kernel_fxn(n: int, arch: str, out_dtype):
  out_f32 = out_dtype == dtypes.float
  insts = build_kernel(n, arch, out_f32=out_f32)
  grid = (n // BLOCK_N, n // BLOCK_M, 1)
  lds_size = max(LDS_SIZE, 65536 // getenv("LIMIT_OCC", 2))
  c_bytes = n * n * out_dtype.itemsize
  def asm_kernel(A, B, C):
    gidxs = [UOp.special(sz, f"gidx{i}") for i, sz in enumerate(grid)]
    lidxs = [UOp.special(THREADS, "lidx0")]
    lds = UOp.placeholder((lds_size,), dtypes.uint8, 0, AddrSpace.LOCAL)
    sink = UOp.sink(A.base, B.base, C.base, lds, *gidxs, *lidxs,
                    arg=KernelInfo(name=colored("rdna3_wmma", "cyan"),
                                   estimates=Estimates(ops=n*n*n*2, mem=n*n*2*2+c_bytes)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))
  return asm_kernel

@functools.cache
def _cached_asm_kernel_fxn(n: int, arch: str, out_f32: bool):
  return _asm_kernel_fxn(n, arch, dtypes.float if out_f32 else dtypes.half)

def rdna3_wmma_gemm(a: Tensor, b: Tensor, out_dtype=dtypes.half) -> Tensor:
  """Hand WMMA GEMM. out_dtype half or float."""
  assert can_use_rdna3_wmma_gemm(a, b), f"rdna3_wmma_gemm: unsupported shapes/dtypes {a.shape}@{b.shape} {a.dtype}"
  from tinygrad.dtype import to_dtype
  out_dtype = to_dtype(out_dtype)
  assert out_dtype in (dtypes.half, dtypes.float), out_dtype
  n = int(a.shape[0])
  arch = _renderer_arch(a.device)
  c = Tensor.empty(n, n, dtype=out_dtype, device=a.device)  # kernel overwrites all of C
  return Tensor.custom_kernel(a, b, c, fxn=_cached_asm_kernel_fxn(n, arch, out_dtype == dtypes.float))[2]

def try_rdna3_wmma_matmul(a: Tensor, b: Tensor, dtype=None) -> Tensor|None:
  """Optional explicit fast path helper. Returns None if ineligible."""
  if not can_use_rdna3_wmma_gemm(a, b): return None
  from tinygrad.dtype import to_dtype
  out = dtypes.half if dtype is None else to_dtype(dtype)
  if out not in (dtypes.half, dtypes.float): return None
  return rdna3_wmma_gemm(a, b, out_dtype=out)

def run_matmul(n: int | None = None):
  n = int(n if n is not None else getenv("N", 4096))
  dev = Device[Device.DEFAULT]
  arch = getattr(dev.renderer, "arch", "gfx1100")
  print(f"N={n}  Device arch: {arch}")
  rng = np.random.default_rng(42)
  a = Tensor(rng.random((n, n), dtype=np.float32).astype(np.float16))
  b = Tensor(rng.random((n, n), dtype=np.float32).astype(np.float16))
  Tensor.realize(a, b)
  assert can_use_rdna3_wmma_gemm(a, b)
  c = rdna3_wmma_gemm(a, b)
  linear = c.schedule_linear()
  print(f"Grid: ({n//BLOCK_N}, {n//BLOCK_M}, 1), Local: ({THREADS}, 1, 1)")

  ets = []
  with Context(DEBUG=int(getenv("DEBUG", 0))):
    for _ in range(int(getenv("CNT", 5))):
      t0 = time.perf_counter()
      run_linear(linear)
      Device[Device.DEFAULT].synchronize()
      ets.append(time.perf_counter() - t0)
  best = min(ets)
  if best > 0:
    print(f"REAL TFLOPS {n*n*n*2 / best * 1e-12:.2f}  ({best*1000:.2f} ms best of {len(ets)})")
  else:
    print(f"timing unavailable (ets={ets})")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    Device[Device.DEFAULT].synchronize()
    c_np, a_np, b_np = c.float().numpy(), a.float().numpy(), b.float().numpy()
    ref = a_np @ b_np
    err = np.sqrt(np.mean((c_np - ref)**2)) / np.sqrt(np.mean(ref**2))
    print(f"relative RMSE {err:.6f}  (sample={c_np[0,0]:.4g})")
    if err != err or err > 0.05: raise RuntimeError(f"matmul is wrong! RMSE={err}")

if __name__ == "__main__":
  if getenv("COMPILE_ONLY", 0):
    n = int(getenv("N", 4096))
    ni, nb = assemble_kernel(n)
    print(f"compile OK: {ni} insts, {nb} byte elf")
  else:
    run_matmul()

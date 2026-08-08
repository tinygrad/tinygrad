from __future__ import annotations
import struct

from tinygrad.device import CompileError
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.helpers import Target, getenv, prod, is_image_shape
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext, Register, greg
from tinygrad.renderer import Renderer
from tinygrad.renderer.amd.dsl import Reg, s, v, NULL, EXEC, VCC, Inst
from tinygrad.runtime.autogen.amd.rdna3 import ins as r3
from tinygrad.runtime.autogen.amd.rdna3.enum import VOPDOp
from tinygrad.codegen.decomp.op import fast_idiv
from tinygrad.codegen.opt import tc
from tinygrad.schedule.rangeify import BufferizeOpts
from tinygrad.uop import Ops, FastEnum, auto
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat

# RDNA3: kernarg in s[0:1], local ids packed in v0. Even SGPR bases for 64-bit kernarg loads.
# WGID follows USER_SGPR_COUNT: s2 when count=2 (1D locals); s15 when gfx1100 pads to 15 (2D locals).
KERNARG_REG = s[0:1]
WGID = tuple(Register(f"s{i}", i) for i in range(2, 5))  # 1D default; 2D uses s15+ via _wgid_reg
LID = tuple(Register(f"v{i}", 256+i) for i in range(3))
# Skip s16 — reserved as WGID_Y when USER_SGPR=15 (gfx1100 2D locals).
SGPR = tuple(Register(f"s{i}", i) for i in range(6, 104, 2) if i != 16)
VGPR = tuple(Register(f"v{i}", 256+i) for i in range(3, 254))
# Hand kernel parks ACC at v128+; use v126..v253 (128 regs) so low VGPRs stay contiguous for addresses.
WMMA_ACC_VGPR = VGPR[123:]
# Disjoint pools: LDS half2 loads stay low; PACK_F16 early-clobber dests stay high (product-8 fix).
# Under ALLOW_UPCAST16 ACC grows to v126..v253 and overlaps the high PACK band — use mid PACK then.
LLOAD_VGPR = VGPR[:120]        # v3..v122
PACK_F16_VGPR = VGPR[187:246]  # v190..v248 (default)
PACK_F16_VGPR_UP16 = VGPR[61:123]  # v64..v125 — below ACC when WMMA=16
LLOAD_VGPR_UP16 = VGPR[:61]        # v3..v63
# v254/v255: per-instruction VGPR scratch; s104:105: EXEC save/restore or SALU compare scratch.
TMP_VDATA, TMP_VADDR = v[254], v[255]
TMP_EXEC = s[104:105]
TMP_SDATA0, TMP_SDATA1 = s[104], s[105]

def _allow_upcast16() -> bool:
  # Off under TC_LDS_AB — product 16 still spills there.
  return bool(getenv("ALLOW_UPCAST16", 0 if getenv("TC_LDS_AB", 0) else 1))

class AMDOps(FastEnum):
  LABEL = auto()
  DEFINE = auto()
  SCRATCH_BASE = auto()
  SCRATCH_SIZE = auto()
  SCRATCH_ADDR = auto()
  KERNARG = auto()
  MOV = auto()
  PACK = auto()
  EXTRACT = auto()
  ADD = auto()
  SUB = auto()
  MUL = auto()
  MULACC = auto()
  CAST = auto()
  RECIPROCAL = auto()
  EXP2 = auto()
  LOG2 = auto()
  SQRT = auto()
  TRUNC = auto()
  SIN = auto()
  MAX = auto()
  SHL = auto()
  SHR = auto()
  AND = auto()
  OR = auto()
  XOR = auto()
  CMPLT = auto()
  CMPNE = auto()
  CMPEQ = auto()
  WHERE = auto()
  LOAD = auto()
  STORE = auto()
  ATOMIC_ADD = auto()
  LDS_BASE = auto()
  LLOAD = auto()
  LSTORE = auto()
  SLOAD = auto()
  SSTORE = auto()
  REG_STORE = auto()
  BARRIER = auto()
  FILL = auto()
  SPILL = auto()
  CMP_GE = auto()
  BRANCH = auto()
  CBRANCH_SCC1 = auto()
  IF_MASK = auto()
  END_MASK = auto()
  PACK_F16 = auto()
  WMMA = auto()

_F32_UNARY = {AMDOps.RECIPROCAL: r3.v_rcp_f32_e32, AMDOps.EXP2: r3.v_exp_f32_e32, AMDOps.LOG2: r3.v_log_f32_e32,
              AMDOps.SQRT: r3.v_sqrt_f32_e32, AMDOps.TRUNC: r3.v_trunc_f32_e32}
_ISEL_UNARY = {Ops.RECIPROCAL: AMDOps.RECIPROCAL, Ops.EXP2: AMDOps.EXP2, Ops.LOG2: AMDOps.LOG2, Ops.SQRT: AMDOps.SQRT,
               Ops.TRUNC: AMDOps.TRUNC, Ops.SIN: AMDOps.SIN}

def _elem_count(u:UOp) -> int:
  """Logical vector element count. INS shape is always scalar; width lives in the opcode/srcs."""
  if u.op is Ops.AFTER: return _elem_count(u.src[0])
  if u.op is Ops.SHRINK and len(u.src) > 2 and u.src[2].op is Ops.CONST: return int(u.src[2].arg)
  if u.op is Ops.INS:
    if u.arg is AMDOps.WMMA: return 8
    if u.arg is AMDOps.PACK: return len(u.src)
    if u.arg is AMDOps.PACK_F16:
      # Vec-load form: srcs are half×n LOAD/LLOAD (see _wmma_ab_vec_loads); else EXTRACT/scalar list.
      if _pack_f16_is_vec_load(u): return sum(_elem_count(s) for s in u.src)
      return len(u.src)
    if u.arg is AMDOps.EXTRACT: return 1
    if u.arg in (AMDOps.LOAD, AMDOps.LLOAD, AMDOps.SLOAD):
      return int(u.src[2].arg) if len(u.src) > 2 and u.src[2].op is Ops.CONST else 1
    if u.arg is AMDOps.MOV and u.src and u.src[0].op is not Ops.SPECIAL: return _elem_count(u.src[0])
    return 1
  try: return u.max_numel()
  except (ValueError, RuntimeError): return 1

def _reg_slots(u:UOp) -> int:
  """VGPR/SGPR slots occupied by u. PACK_F16 packs 2 halves per slot."""
  if u.op is Ops.AFTER: return _reg_slots(u.src[0])
  if u.op is Ops.INS:
    if u.arg is AMDOps.WMMA: return 8
    if u.arg is AMDOps.PACK: return len(u.src)
    if u.arg is AMDOps.PACK_F16:
      if _pack_f16_is_vec_load(u): return sum(_reg_slots(s) for s in u.src)
      return max(1, len(u.src) // 2)
    if u.arg is AMDOps.EXTRACT: return 1
    if u.arg is AMDOps.FILL:
      return int(u.src[1].arg) if len(u.src) > 1 and u.src[1].op is Ops.CONST else 1
    if u.arg in (AMDOps.LOAD, AMDOps.LLOAD, AMDOps.SLOAD):
      return max(1, (u.dtype.itemsize * _elem_count(u) + 3) // 4)
    if u.arg in (AMDOps.STORE, AMDOps.LSTORE, AMDOps.SSTORE): return _reg_slots(u.src[2])
    if u.arg is AMDOps.SPILL: return _reg_slots(u.src[1])
    if u.arg is AMDOps.MOV and u.src and u.src[0].op is not Ops.SPECIAL: return _reg_slots(u.src[0])
    return max(1, (u.dtype.itemsize + 3) // 4)
  return max(1, (u.dtype.itemsize * _elem_count(u) + 3) // 4)

def _mem_itemsize(dt:DType) -> int: return dt.itemsize
def _load_count_src(n:int) -> UOp: return UOp.const(n, dtypes.int32).rtag()
def _reg_to_amd(reg:Register, sz:int=1) -> Reg:
  if reg.index >= 256:
    idx = reg.index - 256
    return v[idx] if sz == 1 else v[idx:idx+sz-1]
  return s[reg.index] if sz == 1 else s[reg.index:reg.index+sz-1]
def _reg_lane(reg:Register, lane:int) -> Reg:
  return _reg_to_amd(Register(f"{reg.name}_{lane}", reg.index + lane))
def _parallel_vmov(moves:list[tuple[Reg, Reg|int|float]]) -> list:
  pending = [(dst, src) for dst,src in moves if not isinstance(src, Reg) or dst != src]
  ret = []
  while pending:
    # Pair zero-imm into VOPD on even/odd VGPR banks (hand WMMA ACC init).
    if len(pending) >= 2:
      (d0, s0), (d1, s1) = pending[0], pending[1]
      n0 = d0.offset - 256 if d0.sz == 1 and 256 <= d0.offset < 512 else None
      n1 = d1.offset - 256 if d1.sz == 1 and 256 <= d1.offset < 512 else None
      if (n0 is not None and n1 == n0 + 1 and n0 % 2 == 0 and
          not isinstance(s0, Reg) and not isinstance(s1, Reg) and s0 == 0 and s1 == 0):
        ret.append(r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=d0, vdsty=d1, srcx0=0, srcy0=0))
        pending = pending[2:]
        continue
    src_regs = {src for _,src in pending if isinstance(src, Reg)}
    for i,(dst,src) in enumerate(pending):
      if not isinstance(src, Reg) or dst not in src_regs:
        ret.append(r3.v_mov_b32_e32(dst, src))
        pending.pop(i)
        break
    else:
      src = pending[0][1]
      if not isinstance(src, Reg): raise RuntimeError("parallel copy cycle without a register source")
      ret.append(r3.v_mov_b32_e32(TMP_VDATA, src))
      pending = [(dst, TMP_VDATA if isinstance(s, Reg) and s == src else s) for dst,s in pending]
  return ret

def _src(x:UOp):
  if x.op is Ops.AFTER: return _src(x.src[0])
  if x.op is Ops.CONST:
    if x.dtype is dtypes.float32: return float(x.arg)
    if x.dtype is dtypes.float16: return struct.unpack("H", struct.pack("e", float(x.arg)))[0]
    return int(x.arg)
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg src {x}")
  if _elem_count(x) > 1: return _reg_lane(greg(x), 0)
  return _reg_to_amd(greg(x), _reg_slots(x))

def _dst(x:UOp) -> Reg:
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg dst {x}")
  return _reg_to_amd(greg(x), _reg_slots(x))

def _full_src(x:UOp) -> Reg:
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg src {x}")
  return _reg_to_amd(greg(x), _reg_slots(x))

def _reg_chunk(reg:Register, off:int, slots:int) -> Reg:
  return _reg_to_amd(Register(reg.name, reg.index+off, _cons=reg.cons), slots)

def _global_load_insts(u:UOp, addr:Reg) -> list:
  slots, sc = _reg_slots(u), u.dtype.scalar()
  saddr = _src(u.src[0])
  if slots == 1:
    if (load:=_global_load(u.dtype, _elem_count(u))) is None: raise CompileError(f"no global load {u.dtype}")
    return [load(_dst(u), addr, saddr=saddr)]
  # multi-VGPR by byte width: float×2/×4 or half×4/×8 → B64/B128 (ungated only; coalesce rejects gated)
  if sc not in (dtypes.float16, dtypes.float32): raise CompileError(f"no vec global load {u.dtype}")
  if not isinstance(greg(u), Register): raise CompileError(f"expected reg dst {u}")
  if slots == 2: return [r3.global_load_b64(_dst(u), addr, saddr=saddr)]
  if slots == 4: return [r3.global_load_b128(_dst(u), addr, saddr=saddr)]
  if slots == 8:
    # offset+16 keeps addr live (no TMP bump) so the pair can s_clause.
    return [r3.s_clause(simm16=1),
            r3.global_load_b128(_reg_chunk(greg(u), 0, 4), addr, saddr=saddr),
            r3.global_load_b128(_reg_chunk(greg(u), 4, 4), addr, saddr=saddr, offset=16)]
  raise CompileError(f"no global load {u.dtype}")

def _apply_byte_off(addr:Reg, byte_off:int, idx:UOp|None=None, itemsize:int=1) -> tuple[list, Reg, int]:
  """GLOBAL offset is 0..4095; peel larger byte_off into addr via v_lshl_add / v_add."""
  if byte_off <= 0: return [], addr, 0
  if byte_off <= 0xfff: return [], addr, byte_off
  if idx is not None and itemsize in (2, 4):
    # (elem_base << shift) + byte_off — one op, avoids ADD-then-LSHL + addr VGPRs.
    return [r3.v_lshl_add_u32(TMP_VADDR, _src(idx), itemsize.bit_length() - 1, byte_off)], TMP_VADDR, 0
  return [r3.v_add_nc_u32_e64(TMP_VADDR, byte_off, addr)], TMP_VADDR, 0

class _StoreAddrCache:
  """CSE C-store bases into 4096-byte pages (one scale/add per page)."""
  __slots__ = ("key", "page")
  def __init__(self): self.clear()
  def clear(self): self.key, self.page = None, None
  def addr(self, idx:UOp, itemsize:int, byte_off:int) -> tuple[list, Reg, int]:
    src = _src(idx)
    key = (getattr(src, "offset", id(src)), itemsize)
    page, rem = divmod(max(byte_off, 0), 0x1000)
    if self.key == key and self.page == page: return [], TMP_VADDR, rem
    if itemsize == 1:
      pre:list = [r3.v_mov_b32_e32(TMP_VADDR, src)] if page == 0 else \
                 [r3.v_add_nc_u32_e64(TMP_VADDR, page * 0x1000, src)]
    elif itemsize in (2, 4):
      shift = itemsize.bit_length() - 1
      pre = [r3.v_lshl_add_u32(TMP_VADDR, src, shift, page * 0x1000)] if page else \
            [r3.v_lshlrev_b32_e64(TMP_VADDR, shift, src)]
    else:
      raise CompileError(f"bad addr scale {itemsize}")
    self.key, self.page = key, page
    return pre, TMP_VADDR, rem

def _global_store_insts(u:UOp, addr:Reg, byte_off:int=0) -> list:
  val = u.src[2]
  slots, sc = _reg_slots(val), val.dtype.scalar()
  saddr = _src(u.src[0])
  # byte_off already clamped to 0..4095 by _apply_byte_off; larger went through v_lshl_add/v_add.
  off_kw = {"offset": byte_off} if 0 < byte_off <= 0xfff else {}
  if slots == 1:
    if (store:=_global_store(val.dtype, _elem_count(val))) is None: raise CompileError(f"no global store {val.dtype}")
    dpre, data = _vgpr_data(TMP_VDATA, val)
    return dpre + [store(addr=addr, data=data, saddr=saddr, **off_kw)]
  if sc not in (dtypes.float16, dtypes.float32): raise CompileError(f"no vec global store {val.dtype}")
  if not isinstance(greg(val), Register): raise CompileError(f"expected reg src {val}")
  if slots == 2: return [r3.global_store_b64(addr=addr, data=_full_src(val), saddr=saddr, **off_kw)]
  if slots == 4: return [r3.global_store_b128(addr=addr, data=_full_src(val), saddr=saddr, **off_kw)]
  if slots == 8:
    lo, hi = _reg_chunk(greg(val), 0, 4), _reg_chunk(greg(val), 4, 4)
    if off_kw:
      o = off_kw["offset"]
      if o + 16 <= 0xfff:
        return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr, offset=o),
                r3.global_store_b128(addr=addr, data=hi, saddr=saddr, offset=o+16)]
      # Second b128 bumps TMP_VADDR — caller must drop store-addr CSE.
      return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr, offset=o),
              r3.v_add_nc_u32_e64(TMP_VADDR, 16, addr),
              r3.global_store_b128(addr=TMP_VADDR, data=hi, saddr=saddr, offset=o)]
    return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr),
            r3.global_store_b128(addr=addr, data=hi, saddr=saddr, offset=16)]
  raise CompileError(f"no global store {val.dtype}")

def _reg_idxs(x:UOp) -> set[int]:
  if x.op is Ops.AFTER: return set().union(*(_reg_idxs(s) for s in x.src))
  if not isinstance(greg(x), Register): return set()
  return set(range(greg(x).index, greg(x).index + _reg_slots(x)))

def _wait_for_domain(domain:str, cnt:int=0):
  if domain == "vm": return r3.s_waitcnt_vmcnt(sdst=NULL, simm16=cnt)
  if domain == "lgkm": return r3.s_waitcnt_lgkmcnt(sdst=NULL, simm16=cnt)
  if domain == "vs": return r3.s_waitcnt_vscnt(sdst=NULL, simm16=cnt)
  raise CompileError(f"unknown wait domain {domain}")

def _wait_domain_for_load(u:UOp) -> str|None:
  if u.op is not Ops.INS: return None
  if u.arg in (AMDOps.LOAD, AMDOps.SLOAD, AMDOps.FILL): return "vm"
  if u.arg in (AMDOps.KERNARG, AMDOps.LLOAD): return "lgkm"
  # PACK_F16 may emit global u16+d16_hi itself (LLVM-style strided B).
  if u.arg is AMDOps.PACK_F16 and _pack_f16_has_d16_hi(u): return "vm"
  return None

def _wait_domain_for_store(u:UOp) -> str|None:
  # RDNA3: vector store completion is vscnt. Track global stores for end/branch drain only
  # (hand-kernel style). Scratch uses inline vscnt; LDS stores scoreboard on lgkm (flush before BARRIER).
  if u.op is not Ops.INS: return None
  if u.arg is AMDOps.STORE: return "vs"
  if u.arg is AMDOps.LSTORE: return "lgkm"
  return None

def _store_src_regs(u:UOp) -> set[int]:
  # Sentinel: any outstanding global/LDS store. Do not scoreboard TMP_VADDR — addr is sampled at issue.
  if u.arg in (AMDOps.STORE, AMDOps.LSTORE): return {-1}
  return set()

def _needs_vm_flush(u:UOp) -> bool:
  # Packs/extracts and int addr ALU can run with VMEM still in flight.
  if u.op is not Ops.INS: return False
  if u.arg in (AMDOps.WMMA, AMDOps.STORE, AMDOps.ATOMIC_ADD, AMDOps.SSTORE, AMDOps.SPILL): return True
  if u.arg in (AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV): return False
  if u.arg in (AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.ADD, AMDOps.SUB, AMDOps.MUL,
               AMDOps.CMP_GE, AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ):
    return u.dtype.scalar() not in dtypes.ints and u.dtype.scalar() is not dtypes.bool
  return True

_SCALAR_LOAD = {
  dtypes.bool: (r3.global_load_u8, r3.scratch_load_u8, r3.ds_load_u8),
  dtypes.uint8: (r3.global_load_u8, r3.scratch_load_u8, r3.ds_load_u8),
  dtypes.int8: (r3.global_load_i8, r3.scratch_load_i8, r3.ds_load_i8),
  dtypes.uint16: (r3.global_load_u16, r3.scratch_load_u16, r3.ds_load_u16),
  dtypes.int16: (r3.global_load_i16, r3.scratch_load_i16, r3.ds_load_i16),
  dtypes.float16: (r3.global_load_u16, r3.scratch_load_u16, r3.ds_load_u16),
  dtypes.uint32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
  dtypes.int32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
  dtypes.float32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
}
_SCALAR_STORE = {
  dtypes.bool: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.uint8: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.int8: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.uint16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.int16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.float16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.uint32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
  dtypes.int32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
  dtypes.float32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
}
_WIDE_LOAD = {8: (r3.global_load_b64, r3.ds_load_b64), 16: (r3.global_load_b128, r3.ds_load_b128)}
_WIDE_STORE = {8: (r3.global_store_b64, r3.ds_store_b64), 16: (r3.global_store_b128, r3.ds_store_b128)}

def _mem_load(kind:int, dt:DType, n:int=1):
  if n > 1:
    nbytes = dt.itemsize * n
    # half×2 → B32; half×4/×8 → B64/B128 for global+LDS (PACK_F16 clobber fixed; gated stays scalar).
    # scratch stays half2 — no wide scratch half path yet.
    if dt.scalar() is dtypes.float16:
      if nbytes == 4: return (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32)[kind]
      if kind != 1 and nbytes in _WIDE_LOAD: return _WIDE_LOAD[nbytes][0 if kind == 0 else 1]
      return None
    if dt is not dtypes.float32: return None
    wide_ops = _WIDE_LOAD.get(nbytes)
    return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
  scalar_ops = _SCALAR_LOAD.get(dt)
  return None if scalar_ops is None else scalar_ops[kind]

def _mem_store(kind:int, dt:DType, n:int=1):
  if n > 1:
    nbytes = dt.itemsize * n
    if dt.scalar() is dtypes.float16:
      if nbytes == 4: return (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32)[kind]
      if kind != 1 and nbytes in _WIDE_STORE: return _WIDE_STORE[nbytes][0 if kind == 0 else 1]
      return None
    if dt is not dtypes.float32: return None
    wide_ops = _WIDE_STORE.get(nbytes)
    return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
  scalar_ops = _SCALAR_STORE.get(dt)
  return None if scalar_ops is None else scalar_ops[kind]

def _global_load(dt:DType, n:int=1): return _mem_load(0, dt, n)
def _global_store(dt:DType, n:int=1): return _mem_store(0, dt, n)
def _scratch_load(dt:DType, n:int=1): return _mem_load(1, dt, n)
def _scratch_store(dt:DType, n:int=1): return _mem_store(1, dt, n)
def _local_load(dt:DType, n:int=1): return _mem_load(2, dt, n)
def _local_store(dt:DType, n:int=1): return _mem_store(2, dt, n)

def _scaled_addr(dst:Reg, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  src = _src(idx)
  if itemsize == 1:
    return ([], src) if isinstance(src, Reg) and src.offset >= 256 else ([r3.v_mov_b32_e32(dst, src)], dst)
  if itemsize not in (2, 4): raise CompileError(f"bad addr scale {itemsize}")
  return [r3.v_lshlrev_b32_e64(dst, itemsize.bit_length()-1, src)], dst

def _masked_addr(pre:list, addr:Reg, masked:bool) -> tuple[list, Reg]:
  if not masked or addr.offset < 256: return pre, addr
  return pre + [r3.v_cndmask_b32_e32(TMP_VADDR, 0, addr)], TMP_VADDR

def _vgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [], src
  return [r3.v_mov_b32_e32(tmp, src)], tmp

def _sgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [r3.v_readfirstlane_b32_e32(tmp, src)], tmp
  return [], src

def _buf_ref(x:UOp, lds:bool) -> bool:
  if x.op is Ops.AFTER: return _buf_ref(x.src[0], lds)
  if lds: return (x.op is Ops.BUFFER and x.addrspace is AddrSpace.LOCAL) or (x.op is Ops.INS and x.arg is AMDOps.LDS_BASE)
  return (x.op is Ops.BUFFER and x.addrspace is AddrSpace.REG) or (x.op is Ops.INS and x.arg is AMDOps.SCRATCH_ADDR)
def _is_lds_ref(x:UOp) -> bool: return _buf_ref(x, True)
def _is_scratch_ref(x:UOp) -> bool: return _buf_ref(x, False)

def _lds_itemsize(x:UOp) -> int:
  return x.dtype.itemsize

def _lds_size_bytes(x:UOp) -> int:
  return x.max_numel() * x.dtype.itemsize

def _align(x:int, a:int) -> int: return x + (-x % a)

def _lds_offsets(ctx:IselContext) -> dict[int, int]:
  if (offsets:=ctx.scratch.get("lds_offsets")) is None:
    offsets, slots, off = {}, set(), 0
    for b in sorted([u for u in ctx.uses if u.op is Ops.BUFFER and u.addrspace is AddrSpace.LOCAL], key=lambda u: u.arg.slot):
      if b.arg.slot in slots: raise CompileError(f"duplicate lds slot {b.arg.slot}")
      slots.add(b.arg.slot)
      off = _align(off, _lds_itemsize(b))
      offsets[b.arg.slot] = off
      off += _lds_size_bytes(b)
    ctx.scratch["lds_offsets"] = offsets
  return offsets

def _lds_base(ctx:IselContext, x:UOp) -> UOp|None:
  if x.addrspace is not AddrSpace.LOCAL: return None
  return UOp(Ops.INS, dtypes.uint32,
             (UOp.const(_lds_size_bytes(x), dtypes.int32).rtag(), UOp.const(_lds_offsets(ctx)[x.arg.slot], dtypes.int32).rtag()),
             AMDOps.LDS_BASE)

def _lds_base_offset(x:UOp) -> int:
  if x.op is Ops.AFTER: return _lds_base_offset(x.src[0])
  if x.op is Ops.INS and x.arg is AMDOps.LDS_BASE: return x.src[1].arg
  return 0

def _scratch_base_offset(x:UOp) -> int:
  if x.op is Ops.AFTER: return _scratch_base_offset(x.src[0])
  if x.op is Ops.INS and x.arg is AMDOps.SCRATCH_ADDR: return x.src[0].arg
  return 0

def _local_addr(base:UOp, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  pre, addr = _scaled_addr(TMP_VADDR, idx, itemsize)
  if (off:=_lds_base_offset(base)) == 0: return pre, addr
  return pre + [r3.v_add_nc_u32_e64(TMP_VADDR, off, addr)], TMP_VADDR

def _scratch_addr(base:UOp, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  pre, addr = _scaled_addr(TMP_VADDR, idx, itemsize)
  if (off:=_scratch_base_offset(base)) == 0: return pre, addr
  return pre + [r3.v_add_nc_u32_e64(TMP_VADDR, off, addr)], TMP_VADDR

def _reg_buffer_base(x:UOp) -> UOp|None:
  if x.op is Ops.AFTER: return _reg_buffer_base(x.src[0])
  return x if x.op is Ops.BUFFER and x.addrspace is AddrSpace.REG else None

def _reg_mem_key(base:UOp, idx:UOp) -> tuple[UOp, int]|None:
  if (buf:=_reg_buffer_base(base)) is None: return None
  if (off:=_const_int(idx)) is None or off < 0: return None
  return buf, off

def _is_zero_val(val:UOp) -> bool:
  if val.op is Ops.CONST: return val.arg == 0
  if val.op is Ops.INS and val.arg is AMDOps.MOV and val.src and val.src[0].op is Ops.CONST:
    return val.src[0].arg == 0
  return False

def _is_identity_sload(val:UOp, key:tuple[UOp, int]) -> bool:
  return val.op is Ops.INS and val.arg is AMDOps.SLOAD and _reg_mem_key(val.src[0], val.src[1]) == key

def _is_identity_load(val:UOp, addr:UOp) -> bool:
  if val.op is not Ops.LOAD or not val.src: return False
  load_addr = val.src[0]
  if load_addr.op not in (Ops.INDEX, Ops.SHRINK) or addr.op not in (Ops.INDEX, Ops.SHRINK): return False
  return _reg_mem_key(load_addr.src[0], load_addr.src[1]) == _reg_mem_key(addr.src[0], addr.src[1])

def _compute_amd_skip(uops:list[UOp]) -> set[UOp]:
  buffer_offset_stores: dict[tuple[UOp, int], list[tuple[UOp, bool, bool]]] = {}
  for u in uops:
    if u.op is Ops.INS and u.arg is AMDOps.SSTORE and len(u.src) >= 3:
      if (key:=_reg_mem_key(u.src[0], u.src[1])) is None: continue
      val = u.src[2]
      is_identity, is_zero = _is_identity_sload(val, key), _is_zero_val(val)
    elif u.op is Ops.STORE and len(u.src) >= 2:
      addr = u.src[0]
      if addr.op not in (Ops.INDEX, Ops.SHRINK): continue
      if (key:=_reg_mem_key(addr.src[0], addr.src[1])) is None: continue
      val = u.src[1]
      is_identity, is_zero = _is_identity_load(val, addr), _is_zero_val(val)
    else: continue
    buffer_offset_stores.setdefault(key, []).append((u, is_identity, is_zero))
  dead_offsets = {key for key, stores in buffer_offset_stores.items()
                  if all(is_identity or is_zero for _, is_identity, is_zero in stores)}
  skip: set[UOp] = set()
  identity_loads: set[UOp] = set()
  for key, stores in buffer_offset_stores.items():
    for store, is_identity, is_zero in stores:
      if is_identity:
        skip.add(store)
        val = store.src[2] if store.op is Ops.INS else store.src[1]
        if val.op is Ops.INS and val.arg is AMDOps.SLOAD: identity_loads.add(val)
        elif val.op is Ops.LOAD: identity_loads.add(val)
      elif key in dead_offsets and is_zero: skip.add(store)
  for u in uops:
    for src in u.src:
      if src in identity_loads and u not in skip: identity_loads.discard(src)
  skip |= identity_loads
  return skip

def _fused_d16_hi_loads(uops:list[UOp]) -> set[UOp]:
  # PACK_F16 emits u16+d16_hi into the pack VGPR — scalar LOADs are encode-only skip.
  # Do not put these in _compute_amd_skip: pre_regalloc `_promote_reg_access` drops skip INS
  # from the linear list (`return x, []`), which breaks live ranges for PACK srcs.
  fused: set[UOp] = set()
  for u in uops:
    if u.op is not Ops.INS or u.arg is not AMDOps.PACK_F16: continue
    if _pack_f16_is_vec_load(u) or len(u.src) < 2 or len(u.src) % 2: continue
    for i in range(len(u.src) // 2):
      lo, hi = u.src[2 * i], u.src[2 * i + 1]
      if _pack_f16_d16_hi_pair(lo, hi): fused.update((lo, hi))
  if not fused: return set()
  uses: dict[UOp, int] = {}
  for u in uops:
    for su in u.src:
      if su in fused: uses[su] = uses.get(su, 0) + 1
  return {ld for ld in fused if uses.get(ld, 0) <= 1}

def _amd_skip(ctx:PreRegAllocContext) -> set[UOp]:
  if "skip" not in ctx.scratch and ctx.uops: ctx.scratch["skip"] = _compute_amd_skip(ctx.uops)
  return ctx.scratch.get("skip") or set()

def _amd_fused_d16(ctx:PreRegAllocContext) -> set[UOp]:
  if "fused_d16" not in ctx.scratch and ctx.uops: ctx.scratch["fused_d16"] = _fused_d16_hi_loads(ctx.uops)
  return ctx.scratch.get("fused_d16") or set()

def _const_int(x:UOp) -> int|None:
  if x.op is Ops.CONST: return int(x.arg)
  if x.op is Ops.INS and x.arg is AMDOps.MOV and x.src and x.src[0].op is Ops.CONST: return int(x.src[0].arg)
  return None

def _is_wmma_acc_reload_pack(cin:UOp, ctx:PreRegAllocContext|None=None) -> bool:
  if cin.op is not Ops.INS or cin.arg is not AMDOps.PACK or len(cin.src) != 8: return False
  if all(s.op is Ops.INS and s.arg is AMDOps.SLOAD for s in cin.src): return True
  # LDS product-16: cin is PACK of zero MOVs (register path uses SLOAD reload).
  if all(s.op is Ops.INS and s.arg is AMDOps.MOV and s.src and s.src[0].op is Ops.CONST and
         s.src[0].dtype.scalar() is dtypes.float and float(s.src[0].arg) == 0.0 for s in cin.src):
    return True
  # SLOAD may already have been rewritten to EXTRACT from the zero-init packs
  if ctx is not None and all(s.op is Ops.INS and s.arg is AMDOps.EXTRACT for s in cin.src):
    tiles = ctx.scratch.get("wmma_acc_tiles") or {}
    tile_inits = set(tiles.values())
    return bool(tile_inits) and all(s.src[0] in tile_inits for s in cin.src)
  return False

def _wmma_acc_buffers(ctx:PreRegAllocContext) -> set[UOp]:
  """REG acc buffers too large for scalar promotion (>64) that feed WMMA ACC VGPRs."""
  if (cached:=ctx.scratch.get("wmma_acc_buffers")) is not None: return cached
  bufs: set[UOp] = set()
  for u in ctx.uops or []:
    if u.op is not Ops.INS or u.arg is not AMDOps.WMMA: continue
    pack = u.src[0]
    if not _is_wmma_acc_reload_pack(pack): continue
    for slot in pack.src:
      if slot.op is Ops.INS and slot.arg is AMDOps.SLOAD:
        if (base:=_reg_buffer_base(slot.src[0])) is None: continue
        if 64 < base.max_numel() <= 128: bufs.add(base)
  # LDS zero-cin path: packs are MOV zeros, so discover oversized REG via SLOAD/SSTORE traffic.
  if not bufs and any(u.op is Ops.INS and u.arg is AMDOps.WMMA and _is_wmma_acc_reload_pack(u.src[0])
                      for u in (ctx.uops or []) if u.src):
    for u in ctx.uops or []:
      if u.op is not Ops.INS or u.arg not in (AMDOps.SLOAD, AMDOps.SSTORE): continue
      if (base:=_reg_buffer_base(u.src[0])) is None: continue
      if 64 < base.max_numel() <= 128: bufs.add(base)
  ctx.scratch["wmma_acc_buffers"] = bufs
  return bufs

def _wmma_slot_tile_lane(idx:int) -> tuple[int, int]:
  # 4×4 UPCAST packs floats as tile=(idx//32)*4+(idx%4), lane=(idx%32)//4
  return (idx // 32) * 4 + (idx % 4), (idx % 32) // 4

def _wmma_acc_zero_inits(uops:list[UOp]) -> tuple[list[UOp], dict[int, UOp], dict[int, tuple[UOp, int]]]:
  """Zero-init WMMA ACC packs before the K-loop.

  Returns (inits, tile->init, reg_idx->(init, lane)).
  tile->init uses the 4×4 interleaved formula. Consecutive product-16 SLOAD packs collide
  on first-idx tile keys (4 keys for 16 packs), so epilogue SLOADs use reg_idx->init.
  """
  ctx = PreRegAllocContext(uops)
  bufs = _wmma_acc_buffers(ctx)
  if not bufs: return [], {}, {}
  seen: set[int] = set()
  inits: list[UOp] = []
  tiles: dict[int, UOp] = {}
  idx_map: dict[int, tuple[UOp, int]] = {}
  next_tile = 0
  for u in uops:
    if u.op is not Ops.INS or u.arg is not AMDOps.WMMA: continue
    pack = u.src[0]
    if not _is_wmma_acc_reload_pack(pack): continue
    if not isinstance(pack.tag, tuple) or not pack.tag: continue
    if (tid:=id(pack.tag)) in seen: continue
    # SLOAD-cin: tile from REG indices. Zero-MOV cin: enumerate in expand order.
    sload_idxs: list[int|None] = []
    if all(s.op is Ops.INS and s.arg is AMDOps.SLOAD for s in pack.src):
      if not any((b:=_reg_buffer_base(s.src[0])) is not None and b in bufs for s in pack.src): continue
      sload_idxs = [_const_int(s.src[1]) for s in pack.src]
      if any(i is None for i in sload_idxs): continue
      tile, _ = _wmma_slot_tile_lane(sload_idxs[0])  # type: ignore[arg-type]
    else:
      tile = next_tile
      next_tile += 1
    seen.add(tid)
    init = UOp(Ops.INS, dtypes.float, tuple(UOp.const(0.0, dtypes.float32) for _ in range(8)), AMDOps.PACK, pack.tag)
    inits.append(init)
    tiles[tile] = init
    for lane, idx in enumerate(sload_idxs):
      if idx is not None: idx_map[idx] = (init, lane)
  return inits, tiles, idx_map

def _reg_promotable_buffers(ctx:PreRegAllocContext) -> set[UOp]:
  if (promotable:=ctx.scratch.get("reg_promotable")) is not None: return promotable
  bases, bad, seen_store = set(), set(), set()
  wmma_bufs = _wmma_acc_buffers(ctx)
  for u in ctx.uops or []:
    if u.op is not Ops.INS or u.arg not in (AMDOps.SLOAD, AMDOps.SSTORE): continue
    if (base:=_reg_buffer_base(u.src[0])) is None: continue
    bases.add(base)
    if base in wmma_bufs: continue  # handled by WMMA ACC aliasing
    idx = _const_int(u.src[1])
    dt = u.dtype if u.arg is AMDOps.SLOAD else u.src[2].dtype
    n = _elem_count(u) if u.arg is AMDOps.SLOAD else _elem_count(u.src[2])
    if idx is None or idx < 0 or idx >= base.max_numel() or base.max_numel() > 64 or n != 1 or dt.itemsize > 4:
      bad.add(base)
      continue
    key = (base, idx)
    if u.arg is AMDOps.SLOAD and key not in seen_store: bad.add(base)
    if u.arg is AMDOps.SSTORE: seen_store.add(key)
  ctx.scratch["reg_promotable"] = promotable = bases - bad
  ctx.scratch["reg_values"] = {}
  ctx.scratch["reg_n"] = 0
  return promotable

def _reg_promote_slot(ctx:PreRegAllocContext, base:UOp, idx:UOp) -> tuple[UOp, int]|None:
  buf = _reg_buffer_base(base)
  if buf is None or buf not in _reg_promotable_buffers(ctx): return None
  return None if (slot:=_const_int(idx)) is None else (buf, slot)

def _new_promoted_reg(ctx:PreRegAllocContext, val:UOp) -> UOp:
  n = ctx.scratch["reg_n"]
  ctx.scratch["reg_n"] = n + 1
  return UOp(Ops.INS, val.dtype, (val,), AMDOps.MOV, (Register(f"reg{n}", 0, _cons=VGPR),))

def _peel_add_imm(idx:UOp, itemsize:int, max_byte:int=0xffff, deep:bool=False) -> tuple[UOp, int]:
  """Peel ADD+imm from an index into a byte offset. Keeps one address base live.
  deep=True folds nested ADD+const chains (WMMA C stores: ADD(ADD(base,1024),16))."""
  total, cur = 0, idx
  while True:
    is_add = cur.op is Ops.ADD or (cur.op is Ops.INS and cur.arg is AMDOps.ADD)
    if not is_add or len(cur.src) != 2: break
    peeled = False
    for base_i, imm_i in ((0, 1), (1, 0)):
      if (c := _const_int(cur.src[imm_i])) is None: continue
      byte = total + c * itemsize
      if 0 <= byte <= max_byte:
        total, cur, peeled = byte, cur.src[base_i], True
        break
    if not peeled or not deep: break
  return (cur, total) if total else (idx, 0)

def _ds_off(byte_off:int) -> dict:
  return {"offset0": byte_off & 0xff, "offset1": (byte_off >> 8) & 0xff}

def _mem_byte_off(u:UOp, src_i:int=3) -> int:
  return int(u.src[src_i].arg) if len(u.src) > src_i and u.src[src_i].op is Ops.CONST else 0

def _lds_byte_off(u:UOp) -> int:
  return _mem_byte_off(u, 3)

def _load_ins(x:UOp, a:UOp, alt:UOp|None=None, gate:UOp|None=None) -> UOp:
  n = x.max_numel()
  if alt is not None and gate is not None:
    raw = UOp(Ops.LOAD, x.dtype, (a,))
    if n == 1: return gate.where(raw, alt)
    return UOp(Ops.STACK, x.dtype, tuple(
      gate.where(raw.index(UOp.const(i, dtypes.weakint)), alt.index(UOp.const(i, dtypes.weakint)) if alt.max_numel() > 1 else alt)
      for i in range(n)))
  count = _load_count_src(n)
  if _is_lds_ref(a.src[0]):
    if _local_load(x.dtype, n) is None and not (x.dtype.scalar() is dtypes.half and n == 16):
      raise CompileError(f"no lds load {x.dtype} x{n}")
    idx, off = _peel_add_imm(a.src[1], _mem_itemsize(x.dtype))
    src = (a.src[0], idx, count) if off == 0 else (a.src[0], idx, count, UOp.const(off, dtypes.int32).rtag())
    return x.ins(AMDOps.LLOAD, dtype=x.dtype, src=src)
  if _is_scratch_ref(a.src[0]):
    if _scratch_load(x.dtype, n) is None: raise CompileError(f"no scratch load {x.dtype} x{n}")
    return x.ins(AMDOps.SLOAD, dtype=x.dtype, src=(a.src[0], a.src[1], count))
  if _global_load(x.dtype, n) is None and not (x.dtype.scalar() is dtypes.half and n == 16):
    raise CompileError(f"no global load {x.dtype} x{n}")
  return x.ins(AMDOps.LOAD, dtype=x.dtype, src=(a.src[0], a.src[1], count))

def _store_ins(x:UOp, a:UOp, val:UOp) -> UOp:
  # Bottom-up isel can match STORE before INDEX→EXTRACT on vec/WMMA values. Defer (return None)
  # so the INDEX child is rewritten first; only raise when the value is already final.
  def try_store(check, op, peel:bool=False, max_byte:int=0xffff, deep:bool=False):
    if check(val.dtype, _elem_count(val)) is None:
      if val.op is Ops.INDEX and _elem_count(val.src[0]) > 1: return None
      raise CompileError(f"no store {val.dtype}")
    if peel:
      # Peel base+imm so one addr VGPR is shared (TC_LDS C-stores: 64 ADD(base,imm) → few bases).
      idx, off = _peel_add_imm(a.src[1], _mem_itemsize(val.dtype), max_byte=max_byte, deep=deep)
      src = (a.src[0], idx, val) if off == 0 else (a.src[0], idx, val, UOp.const(off, dtypes.int32).rtag())
      return x.ins(op, src=src)
    return x.ins(op, src=(a.src[0], a.src[1], val))
  if _is_lds_ref(a.src[0]): return try_store(_local_store, AMDOps.LSTORE, peel=True)
  if _is_scratch_ref(a.src[0]): return try_store(_scratch_store, AMDOps.SSTORE)
  # Soft-peel any ADD+imm (incl. nested). Emit uses GLOBAL offset when ≤4095 else v_lshl_add.
  # (Hard-peel-only-≤4095 left ~120 addr VGPRs + LSHL/store for WMMA C.)
  return try_store(_global_store, AMDOps.STORE, peel=True, max_byte=0x7fffffff, deep=True)

def _lane_const(x:UOp) -> int|None:
  if x.op is Ops.CONST: return x.arg
  if x.op is Ops.INS and x.arg is AMDOps.MOV and len(x.src) == 1 and x.src[0].op is Ops.CONST: return x.src[0].arg
  return None

def _extract_vec_lane(x:UOp) -> UOp|None:
  if len(x.src) != 2 or (lane:=_lane_const(x.src[1])) is None: return None
  if x.src[0].op is Ops.WMMA:
    return UOp(Ops.INS, dtypes.float32, (x.src[0], UOp.const(lane, dtypes.int32).rtag()), AMDOps.EXTRACT)
  n = _elem_count(x.src[0])
  if n == 1 and lane == 0 and x.src[0].addrspace in (None, AddrSpace.ALU): return x.src[0]
  if n == 1: return None
  sc = x.src[0].dtype.scalar()
  if sc not in (dtypes.float32, dtypes.float16): raise CompileError(f"no extract from {x.src[0].dtype}")
  if not 0 <= lane < n: raise CompileError(f"lane {lane} oob for {x.src[0].dtype} x{n}")
  return UOp(Ops.INS, sc, (x.src[0], UOp.const(lane, dtypes.int32).rtag()), AMDOps.EXTRACT)

def _pack_vec(x:UOp) -> UOp|None:
  if x.max_numel() == 1 and len(x.src) <= 1: return None
  if len(x.src) != x.max_numel(): raise CompileError(f"pack size {len(x.src)} != {x.max_numel()}")
  if x.dtype is dtypes.float32:
    return UOp(Ops.INS, dtypes.float32, x.src, AMDOps.PACK, x.tag)
  if x.dtype is dtypes.float16:
    if len(x.src) % 2: raise CompileError(f"half pack needs even len, got {len(x.src)}")
    return UOp(Ops.INS, dtypes.half, x.src, AMDOps.PACK_F16, x.tag)
  raise CompileError(f"no pack {x.dtype}")

def _pack_f16_is_vec_load(u:UOp) -> bool:
  """PACK_F16(half×n LOAD/LLOAD, ...) from _wmma_ab_vec_loads — not scalar-LOAD B packs."""
  def is_vec_mem(s:UOp) -> bool:
    if s.op is Ops.LOAD and s.max_numel() >= 2: return True
    return s.op is Ops.INS and s.arg in (AMDOps.LOAD, AMDOps.LLOAD) and _elem_count(s) >= 2
  return bool(u.src) and all(is_vec_mem(s) for s in u.src)

def _wmma_ab_vec_loads(elems:tuple[UOp, ...]) -> tuple[UOp, ...]|None:
  # STACK of INDEX(half×n LOAD, 0..n-1)... → PACK srcs are the Ops.LOAD nodes (isel tags them once).
  if len(elems) != 16: return None
  loads: list[UOp] = []
  i = 0
  while i < 16:
    e = elems[i]
    if e.op is not Ops.INDEX or len(e.src) != 2 or e.src[1].op is not Ops.CONST: return None
    base, lane0 = e.src[0], int(e.src[1].arg)
    if lane0 != 0 or base.op is not Ops.LOAD: return None
    n = base.max_numel()
    if n < 2 or i + n > 16: return None
    for j in range(n):
      ej = elems[i + j]
      if (ej.op is not Ops.INDEX or ej.src[0] is not base or ej.src[1].op is not Ops.CONST or
          int(ej.src[1].arg) != j): return None
    loads.append(base)
    i += n
  return tuple(loads) if loads else None

def _wmma_stack_operand(src:UOp, idx:int) -> UOp:
  # Coalesced half×16 frag may arrive as Ops.LOAD (STACK folded) or STACK of INDEX.
  if idx < 2 and src.dtype.scalar() is dtypes.half and src.max_numel() == 16 and src.op is Ops.LOAD:
    return UOp(Ops.INS, dtypes.half, (src,), AMDOps.PACK_F16)
  if idx < 2 and src.op is Ops.INS and src.arg is AMDOps.LOAD and _elem_count(src) == 16:
    return UOp(Ops.INS, dtypes.half, (src,), AMDOps.PACK_F16)
  if src.op is not Ops.STACK: raise CompileError(f"wmma src must be stack, got {src.op}")
  n, sc = len(src.src), src.dtype
  if idx < 2 and n == 16 and sc is dtypes.half:
    if (loads := _wmma_ab_vec_loads(src.src)) is not None:
      return UOp(Ops.INS, dtypes.half, loads, AMDOps.PACK_F16)
    return UOp(Ops.INS, dtypes.half, src.src, AMDOps.PACK_F16)
  if idx == 2 and n == 8 and sc is dtypes.float:
    return UOp(Ops.INS, dtypes.float, src.src, AMDOps.PACK)
  raise CompileError(f"bad wmma stack idx={idx} len={n} dtype={src.dtype}")

def _isel_wmma(ctx:IselContext, x:UOp) -> UOp:
  a, b = (_wmma_stack_operand(s, i) for i, s in enumerate(x.src[:2]))
  # accumulator is the init STACK for the first WMMA, or a chained prior WMMA result when UNROLL
  # fuses K iterations. is_two_address coalesces dst with src[0] (=acc), so the chain shares one reg.
  cin = x.src[2]
  if cin.op is Ops.WMMA or (cin.op is Ops.INS and cin.arg is AMDOps.WMMA):
    c = cin
  else:
    # fresh zero accumulator. WMMA is two-address (D is written over C), so each independent output
    # tile (from UPCAST) needs its OWN accumulator register. the zero-init STACK is identical across
    # tiles and dedups to one UOp -> one reg, which the two-address coalesce can only satisfy for one
    # tile. pre-assign a unique vreg so each tile's accumulator stays distinct.
    c = _wmma_stack_operand(cin, 2).replace(tag=(ctx.vreg(WMMA_ACC_VGPR),))
  return UOp(Ops.INS, dtypes.float if x.dtype is dtypes.float else x.dtype, (c, a, b), AMDOps.WMMA, x.tag)

def _wmma_inst(u:UOp):
  dt_in, dt_out = u.src[1].dtype.scalar(), u.dtype.scalar()
  if dt_in is dtypes.half and dt_out is dtypes.float: return r3.v_wmma_f32_16x16x16_f16
  raise CompileError(f"no wmma {dt_in} -> {dt_out}")

def _pack_f16_half2_load(lo:UOp, hi:UOp) -> tuple[UOp, int]|None:
  # EXTRACT(LLOAD, 2k)/EXTRACT(LLOAD, 2k+1) rebuilds the same half2 VGPR word — MOV it.
  # LLOAD-only: global LOAD shares the general VGPR pool with EXTRACT; hi LSHR can
  # clobber the load before PACK MOVs it. Parking LOAD in PACK pool also failed
  # (wide_regalloc collapsed distinct LOADs onto one VGPR).
  if not (lo.op is Ops.INS and lo.arg is AMDOps.EXTRACT and hi.op is Ops.INS and hi.arg is AMDOps.EXTRACT): return None
  if lo.src[0] is not hi.src[0]: return None
  base = lo.src[0]
  if base.op is not Ops.INS or base.arg is not AMDOps.LLOAD: return None
  if not isinstance(greg(base), Register): return None
  lo_lane, hi_lane = _lane_const(lo.src[1]), _lane_const(hi.src[1])
  if lo_lane is None or hi_lane != lo_lane + 1 or lo_lane % 2: return None
  return base, lo_lane // 2

def _pack_f16_d16_hi_pair(lo:UOp, hi:UOp) -> bool:
  # Two scalar global half LOADs → global_load_u16 + global_load_d16_hi_b16 into one VGPR (LLVM).
  # Default off: AMD_D16_HI=1 can hang ones@ones on gfx1100 (wait timeout); v_pack path is correct.
  if not getenv("AMD_D16_HI", 0): return False
  if not (lo.op is Ops.INS and lo.arg is AMDOps.LOAD and hi.op is Ops.INS and hi.arg is AMDOps.LOAD): return False
  if _elem_count(lo) != 1 or _elem_count(hi) != 1: return False
  if lo.dtype.scalar() is not dtypes.half or hi.dtype.scalar() is not dtypes.half: return False
  if _is_lds_ref(lo.src[0]) or _is_scratch_ref(lo.src[0]): return False
  if _is_lds_ref(hi.src[0]) or _is_scratch_ref(hi.src[0]): return False
  return True

def _pack_f16_has_d16_hi(u:UOp) -> bool:
  if _pack_f16_is_vec_load(u) or len(u.src) < 2 or len(u.src) % 2: return False
  return any(_pack_f16_d16_hi_pair(u.src[2 * i], u.src[2 * i + 1]) for i in range(len(u.src) // 2))

def _pack_f16_emit_d16_burst(u:UOp) -> list|None:
  # LLVM-style: hoist byte-addr scales, then s_clause + tight VMEM bursts (no LSHL between loads).
  # Lo: scale into pack lane, load u16 (dest-as-addr). Hi: scale hi idx in place (idx dead after), d16_hi.
  if not _pack_f16_has_d16_hi(u): return None
  n = len(u.src) // 2
  pairs = [(u.src[2 * i], u.src[2 * i + 1]) for i in range(n)]
  if not all(_pack_f16_d16_hi_pair(lo, hi) for lo, hi in pairs): return None
  item = _mem_itemsize(pairs[0][0].dtype)
  if item != 2: return None
  base = greg(u)
  ret: list = []
  # --- lo u16: addr in pack lane ---
  for i, (lo, _) in enumerate(pairs):
    dst = _reg_lane(base, i)
    pre, addr = _scaled_addr(dst, lo.src[1], item)
    if addr != dst: ret += pre + [r3.v_mov_b32_e32(dst, addr)]
    else: ret += pre
  if n: ret.append(r3.s_clause(simm16=n - 1))
  for i, (lo, _) in enumerate(pairs):
    dst = _reg_lane(base, i)
    ret.append(r3.global_load_u16(dst, dst, saddr=_src(lo.src[0])))
  # --- hi d16_hi: scale hi element idx in place (idx dead after), then clause+loads ---
  hi_addrs: list[Reg] = []
  scaled: set[int] = set()
  for _, hi in pairs:
    src = _src(hi.src[1])
    if not isinstance(src, Reg) or src.offset < 256:
      # SGPR/imm index — fall back to TMP path for this pack
      return None
    if src.offset not in scaled:
      ret.append(r3.v_lshlrev_b32_e64(src, 1, src))
      scaled.add(src.offset)
    hi_addrs.append(src)
  if n: ret.append(r3.s_clause(simm16=n - 1))
  for i, (_, hi) in enumerate(pairs):
    ret.append(r3.global_load_d16_hi_b16(_reg_lane(base, i), hi_addrs[i], saddr=_src(hi.src[0])))
  return ret

def _pack_f16_identity_load(u:UOp) -> UOp|None:
  # PACK_F16(EXTRACT(L,0)..EXTRACT(L,2n-1)) with L = half×n LLOAD → reuse L's VGPRs.
  if len(u.src) < 2 or len(u.src) % 2: return None
  base: UOp|None = None
  for i in range(len(u.src) // 2):
    got = _pack_f16_half2_load(u.src[2*i], u.src[2*i+1])
    if got is None: return None
    b, slot = got
    if slot != i: return None
    if base is None: base = b
    elif base is not b: return None
  if base is None or _reg_slots(base) < len(u.src) // 2: return None
  return base

def _pack_f16_insts(u:UOp) -> list:
  # Vec-load form: PACK_F16(LOAD/LLOAD[, ...]) — bitcast half2 words into WMMA src VGPRs.
  if _pack_f16_is_vec_load(u) or (len(u.src) == 1 and u.src[0].op is Ops.INS and
      u.src[0].arg in (AMDOps.LOAD, AMDOps.LLOAD) and _reg_slots(u.src[0]) == _reg_slots(u)):
    ret: list = []
    off = 0
    for src in u.src:
      for j in range(_reg_slots(src)):
        s, d = _reg_lane(greg(src), j), _reg_lane(greg(u), off + j)
        if s != d: ret.append(r3.v_mov_b32_e32(d, s))
      off += _reg_slots(src)
    return ret
  # Regalloc remat may replace a vec-load PACK src with a scalar MOV bind (same phys).
  if len(u.src) == 1:
    src = u.src[0]
    if isinstance(greg(src), Register) and isinstance(greg(u), Register) and greg(src).index == greg(u).index:
      return []
    return [r3.v_mov_b32_e32(_dst(u), _src(src))]
  if len(u.src) < 2 or len(u.src) % 2: raise CompileError(f"pack_f16 needs even src, got {len(u.src)}")
  if (burst := _pack_f16_emit_d16_burst(u)) is not None: return burst
  ret, d16_hi = [], []
  for i in range(len(u.src) // 2):
    lo, hi = u.src[2*i], u.src[2*i+1]
    if (got := _pack_f16_half2_load(lo, hi)) is not None:
      base, slot = got
      src_slot, dst_slot = _reg_lane(greg(base), slot), _reg_lane(greg(u), i)
      if src_slot != dst_slot: ret.append(r3.v_mov_b32_e32(dst_slot, src_slot))
      continue
    # Fallback when burst cannot in-place scale hi idx (SGPR/imm).
    if _pack_f16_d16_hi_pair(lo, hi):
      dst = _reg_lane(greg(u), i)
      pre0, a0 = _scaled_addr(TMP_VADDR, lo.src[1], _mem_itemsize(lo.dtype))
      pre1, a1 = _scaled_addr(TMP_VDATA, hi.src[1], _mem_itemsize(hi.dtype))
      ret += pre0 + [r3.global_load_u16(dst, a0, saddr=_src(lo.src[0]))]
      d16_hi += pre1 + [r3.global_load_d16_hi_b16(dst, a1, saddr=_src(hi.src[0]))]
      continue
    # Always v_pack from lo/hi. Do not shortcut through the half2 load VGPR: EXTRACT(hi)
    # may LSHR that load in place, so a later mov from the load reg sees [hi,0] (2026-07-20).
    pre0, a = _vgpr_data(TMP_VDATA, lo)
    pre1, b = _vgpr_data(TMP_VADDR, hi)
    ret += pre0 + pre1 + [r3.v_pack_b32_f16(_reg_lane(greg(u), i), a, b)]
  return ret + d16_hi

AMD_ATOMIC_ADD = "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
def _atomic_add_ins(x:UOp) -> UOp|None:
  if x.arg != AMD_ATOMIC_ADD: return None
  if len(x.src) != 2 or x.src[0].op is not Ops.INDEX: raise CompileError(f"bad atomic {x}")
  a, val = x.src
  if val.dtype.scalar() is not dtypes.float32: raise CompileError(f"f32 atomic only, got {val.dtype}")
  if _is_lds_ref(a.src[0]) or _is_scratch_ref(a.src[0]): raise CompileError("global atomic only")
  return x.ins(AMDOps.ATOMIC_ADD, src=(a.src[0], a.src[1], val))

def _need_yi(ctx:IselContext) -> bool:
  # lidx1/lidx2 → ENABLE_VGPR_WORKITEM_ID≥1 → gfx1100 USER_SGPR pad to 15 (elf.py).
  return any(u.op is Ops.SPECIAL and u.arg.startswith("lidx") and u.arg[-1] in "12" for u in ctx.func_args)

def _wgid_reg(ctx:IselContext, dim:int) -> Register:
  # HSA: workgroup IDs are placed immediately after user SGPRs.
  base = 15 if _need_yi(ctx) else 2
  return Register(f"s{base + dim}", base + dim)

def _special_reg(name:str, ctx:IselContext|None=None) -> Register:
  if len(name) != 5 or name[:4] not in ("lidx", "gidx") or name[-1] not in "012":
    raise CompileError(f"bad special {name}")
  if name[0] == "l": return LID[int(name[-1])]
  if ctx is None: return WGID[int(name[-1])]
  return _wgid_reg(ctx, int(name[-1]))

def _emit_lidx(dst, dim:int) -> list:
  # gfx11+: packed work-item IDs in v0 (X:0..9, Y:10..19, Z:20..29). Unpacked v1/v2 are unused.
  v0 = _reg_to_amd(LID[0])
  return [r3.v_bfe_u32(dst, v0, dim * 10, 10)]

def _kernarg_offset(ctx:IselContext, x:UOp) -> int:
  params = [u for u in ctx.func_args if u.op is Ops.PARAM]
  bufs = [u for u in params if u.arg.addrspace is not AddrSpace.ALU]
  vals = [u for u in params if u.arg.addrspace is AddrSpace.ALU]
  if x.arg.addrspace is AddrSpace.ALU:
    return len(bufs) * 8 + next(i for i,u in enumerate(vals) if u.arg == x.arg) * 4
  return next(i for i,u in enumerate(bufs) if u.arg == x.arg) * 8

def _alloc_vregs(ctx:IselContext, x:UOp, sgpr_pool:tuple[Register, ...], vgpr_pool:tuple[Register, ...]) -> UOp|None:
  if isinstance(x.tag, tuple): return None
  if x.op is Ops.BUFFER:
    return x.replace(src=tuple(s.rtag() for s in x.src), tag=None) if x.addrspace is AddrSpace.REG else None
  if x.arg in (AMDOps.DEFINE, AMDOps.SCRATCH_SIZE, AMDOps.SCRATCH_ADDR, AMDOps.LDS_BASE,
               AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) or x.dtype is dtypes.void:
    return x.replace(tag=None) if x.arg in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) and x.tag is not None else None
  if x.arg is AMDOps.KERNARG: return x.replace(tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.PARAM:
    if x.arg.addrspace is AddrSpace.ALU: return x.replace(src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
    return x.replace(dtype=dtypes.uint64, src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.SPECIAL:
    # gidx → WGID SGPR (s2 or s15). lidx → normal VGPR; MOV emit unpacks from packed v0.
    if x.arg.startswith("gidx"): return x.replace(tag=(ctx.vreg(_special_reg(x.arg, ctx)),))
    return None
  if x.arg is AMDOps.PACK_F16:
    # Vec-load PACK shares the general VGPR pool with its LOAD so two-address coalesce can alias.
    if _pack_f16_is_vec_load(x):
      return x.replace(tag=(ctx.vreg(vgpr_pool),))
    if (base := _pack_f16_identity_load(x)) is not None and isinstance(base.tag, tuple):
      return x.replace(tag=base.tag)
    return x.replace(tag=(ctx.vreg(PACK_F16_VGPR_UP16 if _allow_upcast16() else PACK_F16_VGPR),))
  if x.arg is AMDOps.LLOAD:
    return x.replace(tag=(ctx.vreg(LLOAD_VGPR_UP16 if _allow_upcast16() else LLOAD_VGPR),))
  return x.replace(tag=(ctx.vreg(vgpr_pool),))

def _gated_load(addr:UOp, alt:UOp, gate:UOp, x:UOp) -> UOp|None:
  if addr.op is not Ops.INDEX or len(addr.src) != 2: return None
  safe_addr = addr.replace(src=(addr.src[0], gate.where(addr.src[1], addr.src[1].const_like(0))))
  return gate.where(safe_addr.load(dtype=x.dtype), alt.cast(x.dtype) if alt.dtype != x.dtype else alt)

def _pow2_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.arg <= 0 or c.arg & (c.arg - 1) or (x.dtype not in dtypes.uints and x.vmin < 0): return None
  return x & UOp.const(c.arg - 1, x.dtype)

class _AMDFastDivRenderer(Renderer):
  def __init__(self): super().__init__(Target("NULL", ""))
  def supported_dtypes(self) -> set[DType]: return {dtypes.int32, dtypes.uint32}

def _const_cdiv(x:UOp, c:UOp) -> UOp|None:
  return fast_idiv(_AMDFastDivRenderer(), x, c.arg) if c.arg > 0 and x.vmin >= 0 else None

def _const_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.arg <= 0 or x.vmin < 0: return None
  if (q:=_const_cdiv(x, c)) is None: return None
  return x - q * UOp.const(c.arg, x.dtype)

def _bool_not(x:UOp) -> UOp:
  return x.where(UOp.const(False, dtypes.bool), UOp.const(True, dtypes.bool))

def _u32_divmod(n:UOp, d:UOp) -> tuple[UOp, UOp]:
  zero, one = UOp.const(0, dtypes.uint32), UOp.const(1, dtypes.uint32)
  q, r = zero, zero
  for i in range(31, -1, -1):
    r = (r << one.const_like(1)) | ((n >> UOp.const(i, dtypes.uint32)) & one)
    ge = _bool_not(r < d)
    q = q | ge.where(one << UOp.const(i, dtypes.uint32), zero)
    r = ge.where(r - d, r)
  return q, r

def _var_divmod(x:UOp, d:UOp, op:UOp) -> UOp|None:
  if x.dtype != d.dtype or x.dtype.scalar() not in (dtypes.int32, dtypes.uint32): return None
  if x.dtype.scalar() is dtypes.uint32:
    q, r = _u32_divmod(x, d)
    return q if op.op is Ops.CDIV else r
  zero = UOp.const(0, dtypes.int32)
  xneg, dneg = x < zero, d < zero
  ax, ad = xneg.where(zero - x, x).cast(dtypes.uint32), dneg.where(zero - d, d).cast(dtypes.uint32)
  q, r = _u32_divmod(ax, ad)
  q, r = q.cast(dtypes.int32), r.cast(dtypes.int32)
  return xneg.where(zero - r, r) if op.op is Ops.CMOD else (xneg ^ dneg).where(zero - q, q)

def _narrow_var_divmod(x:UOp, d:UOp, op:UOp) -> UOp|None:
  if x.dtype != d.dtype or x.dtype.scalar() not in dtypes.ints or x.dtype.itemsize >= 4: return None
  wide = dtypes.int32 if x.dtype.scalar() in dtypes.sints else dtypes.uint32
  return UOp(op.op, wide, (x.cast(wide), d.cast(wide))).cast(x.dtype)

def _cmp_bool_const(x:UOp, m:UOp, c:UOp) -> UOp:
  keep = (x.op is Ops.CMPNE and c.arg is False) or (x.op is Ops.CMPEQ and c.arg is True)
  return m if keep else m.where(UOp.const(False, dtypes.bool), UOp.const(True, dtypes.bool))

def _bool_flag(x:UOp) -> UOp:
  return x.where(UOp.const(True, dtypes.bool), UOp.const(False, dtypes.bool))

def _materialize_flags(x:UOp, idx:tuple[int, ...]|None=None) -> UOp|None:
  src, changed = list(x.src), False
  for i in idx or range(len(src)):
    if src[i].op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ):
      src[i] = _bool_flag(src[i])
      changed = True
  return x.replace(src=tuple(src)) if changed else None

def _materialize_compare_flags(x:UOp) -> UOp|None: return _materialize_flags(x)
def _materialize_store_compare_flag(x:UOp) -> UOp|None:
  return _materialize_flags(x, (1,)) if len(x.src) >= 2 and x.src[1].op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) else None
def _materialize_where_value_flags(x:UOp) -> UOp|None: return _materialize_flags(x, (1, 2))

def _materialize_bool_where(m:UOp, a:UOp, b:UOp) -> UOp|None:
  if m.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): return None
  return UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPNE, dtypes.bool, (m, UOp.const(False, dtypes.bool))), a, b))

def _is_foldable(ctx:IselContext, x:UOp, s:UOp) -> bool: return len(ctx.uses[s]) == x.src.count(s) == 1

def _fused_mulacc(ctx:IselContext, a:UOp, b:UOp, c:UOp) -> UOp|None:
  return a.ins(AMDOps.MULACC, src=(*a.src, b)) if _is_foldable(ctx, c, a) else None

def _promote_f16_unary(x:UOp, d:UOp) -> UOp:
  return UOp(x.op, dtypes.float32, (d.cast(dtypes.float32),)).cast(dtypes.float16)

def _int_cast(y:UOp, x:UOp) -> UOp|None:
  if x.dtype.itemsize == y.dtype.itemsize: return x.replace(op=Ops.NOOP)
  return x.ins(AMDOps.CAST, src=(y,))

pre_isel_matcher = PatternMatcher([
  (UPat(Ops.INDEX, name="addr").load(UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool), name="x"), _gated_load),
  (UPat((Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN), dtype=dtypes.float16, src=(UPat.var("d"),), name="x"),
   _promote_f16_unary),
  (UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar("c"))), _const_cdiv),
  (UPat(Ops.CMOD, src=(UPat.var("x", dtypes.ints), UPat.cvar("c"))), _pow2_cmod),
  (UPat(Ops.CMOD, src=(UPat.var("x", dtypes.ints), UPat.cvar("c"))), _const_cmod),
  (UPat((Ops.CDIV, Ops.CMOD), src=(UPat.var("x", dtypes.ints), UPat.var("d", dtypes.ints)), name="op"), _narrow_var_divmod),
  (UPat((Ops.CDIV, Ops.CMOD), src=(UPat.var("x", (dtypes.int32, dtypes.uint32)), UPat.var("d", (dtypes.int32, dtypes.uint32))), name="op"),
   _var_divmod),
  (UPat((Ops.CMPNE, Ops.CMPEQ), src=(UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="m"), UPat.cvar("c", dtypes.bool)), name="x"),
   _cmp_bool_const),
  (UPat((Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ), dtype=dtypes.bool, name="x"), _materialize_compare_flags),
  (UPat(Ops.STORE, name="x"), _materialize_store_compare_flag),
  (UPat(Ops.WHERE, name="x"), _materialize_where_value_flags),
  (UPat.var("m", dtypes.bool).cast(dtypes.ints+(dtypes.float16, dtypes.float32), name="x"),
   lambda m,x: m.where(UOp.const(1, x.dtype), UOp.const(0, x.dtype))),
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")), _materialize_bool_where),
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"), _int_cast),
  (UPat.var("y", dtypes.ints).cast(dtypes.float16), lambda y: y.cast(dtypes.float32).cast(dtypes.float16)),
  (UPat.var("y", dtypes.float16).cast(dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat(Ops.BITCAST, name="x"), lambda x: x.replace(op=Ops.NOOP)),
])

def make_isel_matcher(sgpr_pool:tuple[Register, ...]=SGPR, vgpr_pool:tuple[Register, ...]=VGPR) -> PatternMatcher:
  return PatternMatcher([
    (UPat((Ops.ADD, Ops.SUB), src=(UPat(Ops.INS, arg=AMDOps.SCRATCH_BASE), UPat.cvar()), name="x"),
     lambda x: UOp(Ops.INS, dtypes.void, (x.src[1],), AMDOps.SCRATCH_SIZE)),
    (UPat(Ops.INDEX, src=(UPat(Ops.INS, arg=AMDOps.SCRATCH_BASE), UPat.cvar("off")), name="x"),
     lambda off,x: UOp(Ops.INS, dtypes.uint32, (off,), AMDOps.SCRATCH_ADDR)),
    (UPat(Ops.RANGE, src=(UPat.cvar("c"),), allow_any_len=True, name="x"), lambda c,x:
     x.replace(dtype=dtypes.uint32, src=(UOp.const(c.arg, dtypes.uint32).rtag(),) + x.src[1:])),
    (UPat(Ops.RANGE, name="x"), lambda ctx,x,sgpr_pool=sgpr_pool:
     x.replace(dtype=dtypes.uint32, tag=(ctx.vreg(sgpr_pool),)) if not isinstance(x.tag, tuple) else None),
    (UPat(Ops.PARAM, name="x"), lambda ctx,x:
     UOp(Ops.INS, dtypes.uint64 if x.arg.addrspace is not AddrSpace.ALU else dtypes.uint32,
         (UOp.const(_kernarg_offset(ctx, x), dtypes.int32).rtag(),), AMDOps.KERNARG, None)
     if not isinstance(x.tag, tuple) else None),
    (UPat(Ops.BUFFER, name="x"), lambda ctx,x: _lds_base(ctx, x)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x,vgpr_pool=vgpr_pool:
     None if x.tag is not None else
     UOp(Ops.INS, dtypes.uint32, (x.rtag(),), AMDOps.MOV,
         (ctx.vreg(vgpr_pool if x.arg.startswith("lidx") else _special_reg(x.arg, ctx)),))),
    (UPat(Ops.INDEX, name="x"), _extract_vec_lane),
    (UPat(Ops.STACK, name="x"), _pack_vec),
    # Int/bool CONST stay as CONST (_src inlines / _vgpr_data temps at use). Avoids
    # long-lived VGPR MOVs that dominate UPCAST4 spill. Float still needs a MOV VGPR.
    (UPat.cvar("x", (dtypes.float16, dtypes.float32)), lambda x:
     x.ins(AMDOps.MOV, src=(x.rtag(),)) if not x.tag else None),
    ((UPat(Ops.MUL, (dtypes.float16, dtypes.float32), name="a") + UPat.var("b")).named("c"), _fused_mulacc),
    ((UPat(dtype=dtypes.ints+(dtypes.bool, dtypes.float16, dtypes.float32)) + UPat()).named("x"), lambda x: x.ins(AMDOps.ADD)),
    (UPat(Ops.SUB, dtype=dtypes.ints+(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.SUB)),
    ((UPat(dtype=dtypes.ints+(dtypes.float16, dtypes.float32)) * UPat()).named("x"), lambda x: x.ins(AMDOps.MUL)),
    (UPat(Ops.MULACC, dtype=(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.MULACC)),
    *((UPat.var("y", fr).cast(to, name="x"), lambda y,x,op=op: x.ins(op, src=(y,)))
      for fr,to,op in ((dtypes.float16, dtypes.float32, AMDOps.CAST), (dtypes.float32, dtypes.float16, AMDOps.CAST),
                       (dtypes.ints, dtypes.float32, AMDOps.CAST), (dtypes.float32, dtypes.ints, AMDOps.CAST))),
    *((UPat(op, dtype=dtypes.float32, name="x"), lambda x,op=op,amd=amd: x.ins(amd)) for op,amd in _ISEL_UNARY.items()),
    (UPat(Ops.MAX, dtype=dtypes.ints+(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.MAX)),
    ((UPat(dtype=dtypes.ints) << UPat()).named("x"), lambda x: x.ins(AMDOps.SHL)),
    ((UPat(dtype=dtypes.ints) >> UPat()).named("x"), lambda x: x.ins(AMDOps.SHR)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) & UPat()).named("x"), lambda x: x.ins(AMDOps.AND)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) | UPat()).named("x"), lambda x: x.ins(AMDOps.OR)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) ^ UPat()).named("x"), lambda x: x.ins(AMDOps.XOR)),
    (UPat(Ops.CMPLT, name="x"), lambda x: x.ins(AMDOps.CMPLT, dtype=dtypes.bool)),
    (UPat(Ops.CMPNE, name="x"), lambda x: x.ins(AMDOps.CMPNE, dtype=dtypes.bool)),
    (UPat(Ops.CMPEQ, name="x"), lambda x: x.ins(AMDOps.CMPEQ, dtype=dtypes.bool)),
    (UPat.var("m").where(UPat.var("a"), UPat.var("b")).named("x"), lambda m,a,b,x: x.ins(AMDOps.WHERE, src=(m, a, b))),
    (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool)), name="x"), _load_ins),
    (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"),), name="x"), _load_ins),
    (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("val")), name="x"), _store_ins),
    (UPat(Ops.CUSTOM, name="x"), _atomic_add_ins),
    (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(AMDOps.BARRIER)),
    (UPat(Ops.WMMA, name="x"), lambda ctx,x: _isel_wmma(ctx, x)),
    (UPat((Ops.INS, Ops.BUFFER), name="x"), lambda ctx,x,sgpr_pool=sgpr_pool,vgpr_pool=vgpr_pool: _alloc_vregs(ctx, x, sgpr_pool, vgpr_pool)),
  ])

isel_matcher = make_isel_matcher()

def _loop_label(x:UOp) -> str: return "_".join(str(i) for i in x.arg[:-1])

def _lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = _loop_label(x)
  acc = x.ins(AMDOps.MOV, dtype=dtypes.uint32, src=(UOp.const(0, dtypes.uint32).rtag(),))
  label = UOp(Ops.INS, dtypes.void, arg=AMDOps.LABEL, tag=f".LOOP_{loop_label}")
  cmp = UOp(Ops.INS, dtypes.void, (acc, x.src[0]), AMDOps.CMP_GE)
  jump_out = UOp(Ops.INS, dtypes.void, (cmp,), AMDOps.CBRANCH_SCC1, tag=f".LOOP_OUT_{loop_label}")
  ctx.loop_label[acc] = loop_label
  return acc, [acc, label, cmp, jump_out]

def _lower_end(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = ctx.loop_label[x.src[1]]
  jmp = UOp(Ops.INS, dtypes.void, arg=AMDOps.BRANCH, tag=f".LOOP_{loop_label}")
  return jmp, [
    x.src[1].ins(AMDOps.ADD, dtype=dtypes.uint32, src=(x.src[1], UOp.const(1, dtypes.uint32).rtag())),
    jmp,
    UOp(Ops.INS, dtypes.void, arg=AMDOps.LABEL, tag=f".LOOP_OUT_{loop_label}")]

def _lower_reg_store(x:UOp) -> tuple[UOp, list[UOp]]:
  acc, val = x.src
  if acc.op is Ops.INS and acc.arg is AMDOps.FILL:
    # spilled acc: write update back to scratch slot
    sp = UOp(Ops.INS, dtypes.void, (acc.src[0], val), AMDOps.SPILL)
    return sp, [sp]
  if not isinstance(greg(acc), Register) or greg(acc).index < 256:
    raise CompileError(f"bad reg store acc {acc}")
  st = UOp(Ops.INS, val.dtype, (val,), AMDOps.MOV, (greg(acc),))
  return st, [st]

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: _lower_range(ctx, x)),
  (UPat(Ops.END, name="x"), lambda ctx,x: _lower_end(ctx, x)),
  (UPat(Ops.INS, arg=AMDOps.REG_STORE, name="x"), lambda x: _lower_reg_store(x)),
  (UPat((Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.SPECIAL, Ops.SINK, Ops.GROUP), name="x"), lambda x: (x, [])),
])

def _vcc_rematerialize(ctx, x:UOp):
  _flags = (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ)
  flag_def = x if x.arg in _flags else \
             x.src[0] if x.arg in (AMDOps.WHERE, AMDOps.IF_MASK) and x.src[0].op is Ops.INS and x.src[0].arg in _flags else None
  if flag_def is None: return None
  # VCC is implicit; rematerialize compares before WHERE/IF_MASK consumers
  if flag_def is not x: return x, [flag_def, x]
  if ctx.lock is not None and ctx.lock is not flag_def: ctx.clobbered.add(ctx.lock)
  ctx.lock = flag_def
  if flag_def not in ctx.clobbered: return None
  ctx.clobbered.remove(flag_def)
  return x, [flag_def, x]

def _lower_late_index(x:UOp) -> tuple[UOp, list[UOp]]: return x, []
def _store_addr(a:UOp) -> UOp:
  return a if a.op in (Ops.INDEX, Ops.SHRINK) else UOp(Ops.INDEX, a.dtype, (a, UOp.const(0, dtypes.int32).rtag()))

def _lower_late_store(ctx, x:UOp, a:UOp, val:UOp, gate:UOp|None=None) -> tuple[UOp, list[UOp]]:
  if x in _amd_skip(ctx): return x, []
  st = _store_ins(x, _store_addr(a), val)
  if gate is None: return st, [st]
  mif = UOp(Ops.INS, dtypes.void, (gate,), AMDOps.IF_MASK)
  remat = _vcc_rematerialize(ctx, mif)
  pre = remat[1] if remat is not None else [mif]
  mend = UOp(Ops.INS, dtypes.void, (mif,), AMDOps.END_MASK)
  return mend, [*pre, st, mend]

def _promote_wmma_acc_pack(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  """Redirect oversized WMMA acc reload PACK to the pre-loop zero-init with the same tag."""
  if not _is_wmma_acc_reload_pack(x, ctx): return None
  if not isinstance(x.tag, tuple) or not x.tag: return None
  inits = ctx.scratch.get("wmma_acc_inits") or {}
  if (init:=inits.get(id(x.tag))) is None: return None
  left = ctx.scratch.get("wmma_packs_left")
  if left is None: ctx.scratch["wmma_packs_left"] = left = len(inits)
  ctx.scratch["wmma_packs_left"] = left - 1
  if left - 1 == 0: ctx.scratch["wmma_past_acc"] = True
  return init, []  # cin aliases pre-loop zero pack; physical ACC updated in-place by WMMA

def _promote_reg_buffer(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x.addrspace is not AddrSpace.REG or x not in _reg_promotable_buffers(ctx): return None
  return x, []

def _promote_reg_access(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x in _amd_skip(ctx): return x, []
  # Fused B LOADs stay in the list (address rewrite) but drop VGPR tags — PACK emits u16+d16_hi.
  if x in _amd_fused_d16(ctx):
    nx = x.replace(tag=None)
    return nx, [nx]
  if x.arg is AMDOps.SSTORE:
    if (buf:=_reg_buffer_base(x.src[0])) is not None and buf in _wmma_acc_buffers(ctx):
      return x, []  # acc stays in WMMA VGPR across K-loop
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1])) is None: return None
    val = x.src[2]
    reg_values = ctx.scratch["reg_values"]
    if slot not in reg_values:
      reg_values[slot] = acc = _new_promoted_reg(ctx, val)
      return acc, [acc]
    acc = reg_values[slot]
    st = UOp(Ops.INS, dtypes.void, (acc, val), AMDOps.REG_STORE)
    return acc, [st]
  if x.arg is AMDOps.SLOAD:
    if (buf:=_reg_buffer_base(x.src[0])) is not None and buf in _wmma_acc_buffers(ctx):
      # Loop-body SLOADs only feed WMMA PACK (redirected); post-loop reads need EXTRACT.
      if not ctx.scratch.get("wmma_past_acc"): return x, []
      if (idx:=_const_int(x.src[1])) is None: return None
      # Prefer exact REG-index map (product-16: interleaved tile keys collide at 0..7).
      idx_map = ctx.scratch.get("wmma_acc_idx_map") or {}
      if (got:=idx_map.get(idx)) is not None: init, lane = got
      else:
        tile, lane = _wmma_slot_tile_lane(idx)
        tiles = ctx.scratch.get("wmma_acc_tiles") or {}
        if (init:=tiles.get(tile)) is None: return None
      n = ctx.scratch.get("wmma_ext_n", 0)
      ctx.scratch["wmma_ext_n"] = n + 1
      ext = UOp(Ops.INS, dtypes.float32, (init, UOp.const(lane, dtypes.int32).rtag()), AMDOps.EXTRACT,
                (Register(f"wmma_ext{n}", 0, _cons=VGPR),))
      return ext, [ext]
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1])) is None: return None
    loaded = ctx.scratch["reg_values"].get(slot)
    if loaded is None: return None
    return loaded, []
  return None

def _lower_late_if(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  mif = UOp(Ops.INS, dtypes.void, (x.src[0],), AMDOps.IF_MASK)
  remat = _vcc_rematerialize(ctx, mif)
  return remat if remat is not None else (mif, [mif])

def _lower_late_endif(x:UOp) -> tuple[UOp, list[UOp]]:
  # Keep END_MASK after the guarded store. Source-less INS nodes are hoisted before regalloc.
  mend = UOp(Ops.INS, dtypes.void, x.src, AMDOps.END_MASK)
  return mend, [mend]

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), _lower_late_index),
  (UPat(Ops.STORE, src=(UPat.var("a"), UPat.var("val"), UPat.var("gate", dtype=dtypes.bool)), name="x"), _lower_late_store),
  (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("val")), name="x"), _lower_late_store),
  (UPat(Ops.STORE, src=(UPat.var("a"), UPat.var("val")), name="x"), _lower_late_store),
  (UPat(Ops.INS, arg=AMDOps.PACK, name="x"), _promote_wmma_acc_pack),
  (UPat(Ops.BUFFER, name="x"), _promote_reg_buffer),
  (UPat(Ops.INS, name="x"), _promote_reg_access),
  (UPat(Ops.IF, name="x"), _lower_late_if),
  (UPat(Ops.ENDIF, name="x"), _lower_late_endif),
  (UPat(Ops.INS, name="x"), _vcc_rematerialize),
])


_ALU2: dict[AMDOps, tuple] = {
  AMDOps.ADD: (r3.v_add_f16_e32, r3.v_add_f32_e32, r3.s_add_u32, r3.v_add_nc_u32_e64),
  AMDOps.SUB: (r3.v_sub_f16_e32, r3.v_sub_f32_e32, r3.s_sub_u32, r3.v_sub_nc_u32_e64),
  AMDOps.MUL: (r3.v_mul_f16_e32, r3.v_mul_f32_e32, r3.v_mul_lo_u32, r3.v_mul_lo_u32),
}
def _alu2(u:UOp):
  f16, f32, sgpr, vint = _ALU2[u.arg]
  d = _dst(u)
  sc = u.dtype.scalar()
  if sc is dtypes.float16: return [f16(d, _src(u.src[0]), _src(u.src[1]))]
  if sc is dtypes.float32: return [f32(d, _src(u.src[0]), _src(u.src[1]))]
  if greg(u).index < 256: return [sgpr(d, _src(u.src[0]), _src(u.src[1]))]
  # VOP: one src must be VGPR; materialize src0 if it's an imm/SGPR.
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  return pre + [vint(d, a, _src(u.src[1]))]
def _max(u:UOp):
  d = _dst(u)
  sc = u.dtype.scalar()
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  b = _src(u.src[1])
  if sc is dtypes.float16: return pre + [r3.v_max_f16_e32(d, a, b)]
  if sc is dtypes.float32: return pre + [r3.v_max_f32_e32(d, a, b)]
  if sc in dtypes.sints: return pre + [r3.v_max_i32_e64(d, a, b)]
  return pre + [r3.v_max_u32_e64(d, a, b)]
def _cmp_lt(u:UOp):
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  sc = u.src[0].dtype.scalar()
  if sc is dtypes.float16: cmp = r3.v_cmp_gt_f16_e32
  elif sc is dtypes.float32: cmp = r3.v_cmp_gt_f32_e32
  elif sc in dtypes.sints: cmp = r3.v_cmp_gt_i32_e32
  else: cmp = r3.v_cmp_gt_u32_e32
  return pre + [cmp(_src(u.src[1]), a)]
def _cmp_ne(u:UOp):
  pre, b = _vgpr_data(TMP_VDATA, u.src[1])
  sc = u.src[0].dtype.scalar()
  if sc is dtypes.float16: return pre + [r3.v_cmp_neq_f16_e32(_src(u.src[0]), b)]
  if sc is dtypes.float32: return pre + [r3.v_cmp_neq_f32_e32(_src(u.src[0]), b)]
  return pre + [r3.v_cmp_ne_u32_e32(_src(u.src[0]), b)]
def _cmp_eq(u:UOp):
  pre, b = _vgpr_data(TMP_VDATA, u.src[1])
  sc = u.src[0].dtype.scalar()
  if sc is dtypes.float16: return pre + [r3.v_cmp_eq_f16_e32(_src(u.src[0]), b)]
  if sc is dtypes.float32: return pre + [r3.v_cmp_eq_f32_e32(_src(u.src[0]), b)]
  return pre + [r3.v_cmp_eq_u32_e32(_src(u.src[0]), b)]

_MASKED_MEM = (AMDOps.LOAD, AMDOps.STORE, AMDOps.LLOAD, AMDOps.LSTORE, AMDOps.SLOAD, AMDOps.SSTORE)

def insts_for_uop(u:UOp, skip:set[UOp]|None=None, masked:bool=False, store_addr_cache:_StoreAddrCache|None=None):
  if u.op is not Ops.INS or (skip and u in skip): return []
  if isinstance(u.arg, Inst): return [u.arg]
  match u.arg:
    case (AMDOps.LABEL | AMDOps.BRANCH | AMDOps.CBRANCH_SCC1 | AMDOps.DEFINE | AMDOps.SCRATCH_BASE |
          AMDOps.SCRATCH_SIZE | AMDOps.SCRATCH_ADDR | AMDOps.LDS_BASE):
      return []
    case AMDOps.KERNARG:
      off = u.src[0].arg
      load = r3.s_load_b64(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off) if u.dtype.itemsize == 8 else \
             r3.s_load_b32(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off)
      return [load]
    case AMDOps.MOV:
      if not u.src: return []
      if u.src[0].op is Ops.SPECIAL:
        name = u.src[0].arg
        if name.startswith("lidx"): return _emit_lidx(_dst(u), int(name[-1]))
        return []  # gidx coalesced onto WGID SGPR
      if (sregs:=_reg_idxs(u.src[0])) and sregs == _reg_idxs(u): return []
      if _reg_slots(u) > 1:
        if not isinstance(greg(u.src[0]), Register): raise CompileError(f"expected vec reg src {u}")
        return _parallel_vmov([(_reg_lane(greg(u), i), _reg_lane(greg(u.src[0]), i)) for i in range(_reg_slots(u))])
      if greg(u).index < 256: return [r3.s_mov_b32(_dst(u), _src(u.src[0]))]
      return [r3.v_mov_b32_e32(_dst(u), _src(u.src[0]))]
    case AMDOps.PACK:
      if u.dtype is not dtypes.float32: raise CompileError(f"f32 pack only, got {u.dtype}")
      return _parallel_vmov([(_reg_lane(greg(u), i), _src(s)) for i,s in enumerate(u.src)])
    case AMDOps.PACK_F16:
      return _pack_f16_insts(u)
    case AMDOps.WMMA:
      acc, src0, src1 = u.src[0], u.src[1], u.src[2]
      vdst = _reg_to_amd(greg(acc), 8)
      return [_wmma_inst(u)(vdst=vdst, src0=_reg_to_amd(greg(src0), 8), src1=_reg_to_amd(greg(src1), 8), src2=vdst)]
    case AMDOps.EXTRACT:
      lane = u.src[1].arg
      src = u.src[0]
      if not isinstance(greg(src), Register): raise CompileError(f"expected vec reg src {u}")
      sc = src.dtype.scalar()
      if sc is dtypes.float32:
        lane_src = _reg_lane(greg(src), lane)
        return [] if isinstance(greg(u), Register) and greg(u).index == greg(src).index+lane else [r3.v_mov_b32_e32(_dst(u), lane_src)]
      if sc is dtypes.float16:
        # two f16s per VGPR; high lane needs a 16-bit shift
        slot, hi = divmod(int(lane), 2)
        lane_src = _reg_lane(greg(src), slot)
        if not hi:
          return [] if isinstance(greg(u), Register) and greg(u).index == greg(src).index+slot else [r3.v_mov_b32_e32(_dst(u), lane_src)]
        # Never LSHR in place: lo EXTRACT / PACK_F16 may still need the full half2 in src.
        if isinstance(greg(u), Register) and greg(u).index == greg(src).index+slot:
          return [r3.v_lshrrev_b32_e64(TMP_VDATA, 16, lane_src), r3.v_mov_b32_e32(_dst(u), TMP_VDATA)]
        return [r3.v_lshrrev_b32_e64(_dst(u), 16, lane_src)]
      raise CompileError(f"f16/f32 extract only, got {src.dtype}")
    case AMDOps.ADD | AMDOps.SUB | AMDOps.MUL:
      return _alu2(u)
    case AMDOps.MULACC:
      if u.dtype.scalar() is dtypes.float16: return [r3.v_fma_f16(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
      if u.dtype.scalar() is dtypes.float32: return [r3.v_fma_f32(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
      raise CompileError(f"f16/f32 mulacc only, got {u.dtype}")
    case AMDOps.CAST:
      pre, cast_src = _vgpr_data(TMP_VDATA, u.src[0])
      if u.dtype.scalar() in dtypes.ints and u.src[0].dtype.scalar() in dtypes.ints:
        if u.dtype.itemsize > 4 or u.src[0].dtype.itemsize > 4: raise CompileError(f"no cast {u.src[0].dtype} -> {u.dtype}")
        narrow = u.src[0].dtype if u.src[0].dtype.itemsize <= u.dtype.itemsize else u.dtype
        if narrow.scalar() in dtypes.uints: return pre + [r3.v_and_b32_e32(_dst(u), (1 << (narrow.itemsize * 8)) - 1, cast_src)]
        shift = 32 - narrow.itemsize * 8
        return pre + [r3.v_lshlrev_b32_e64(_dst(u), shift, cast_src), r3.v_ashrrev_i32_e64(_dst(u), shift, _dst(u))]
      if u.dtype.scalar() is dtypes.float32 and u.src[0].dtype.scalar() is dtypes.float16:
        return pre + [r3.v_cvt_f32_f16_e32(_dst(u), cast_src)]
      if u.src[0].dtype.scalar() is dtypes.float32 and u.dtype.scalar() is dtypes.float16:
        return pre + [r3.v_cvt_f16_f32_e32(_dst(u), cast_src)]
      if u.dtype.scalar() is dtypes.float32 and u.src[0].dtype.scalar() in dtypes.ints:
        cast_op = r3.v_cvt_f32_i32_e32 if u.src[0].dtype.scalar() in dtypes.sints else r3.v_cvt_f32_u32_e32
        return pre + [cast_op(_dst(u), cast_src)]
      if u.src[0].dtype.scalar() is dtypes.float32 and u.dtype.scalar() in dtypes.ints:
        cast_op = r3.v_cvt_i32_f32_e32 if u.dtype.scalar() in dtypes.sints else r3.v_cvt_u32_f32_e32
        return pre + [cast_op(_dst(u), cast_src)]
      raise CompileError(f"no cast {u.src[0].dtype} -> {u.dtype}")
    case AMDOps.RECIPROCAL:
      if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"f32 rcp only, got {u.dtype}")
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      dst = _dst(u)
      return pre + [r3.v_rcp_f32_e32(TMP_VADDR, val), r3.v_mul_f32_e32(dst, val, TMP_VADDR),
                    r3.v_sub_f32_e32(dst, 1.0, dst), r3.v_fma_f32(dst, TMP_VADDR, dst, TMP_VADDR),
                    r3.v_cmp_eq_f32_e32(dst, dst), r3.v_cndmask_b32_e32(dst, TMP_VADDR, dst)]
    case AMDOps.EXP2 | AMDOps.LOG2 | AMDOps.SQRT | AMDOps.TRUNC:
      if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"f32 {u.arg.name} only, got {u.dtype}")
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [_F32_UNARY[u.arg](_dst(u), val)]
    case AMDOps.SIN:
      if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"f32 sin only, got {u.dtype}")
      val = _src(u.src[0])
      pre = [] if isinstance(val, Reg) and val == TMP_VADDR else [r3.v_mov_b32_e32(TMP_VADDR, val)]
      return pre + [r3.v_mul_f32_e32(TMP_VDATA, 0.15915494309189535, TMP_VADDR),
                    r3.v_add_f32_e32(TMP_VDATA, 0.5, TMP_VDATA),
                    r3.v_fract_f32_e32(_dst(u), TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VDATA, TMP_VDATA, _dst(u)),
                    r3.v_mul_f32_e32(_dst(u), 6.28125, TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VADDR, TMP_VADDR, _dst(u)),
                    r3.v_mul_f32_e32(_dst(u), 0.0019353071795864769, TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VADDR, TMP_VADDR, _dst(u)),
                    r3.v_mul_f32_e32(TMP_VDATA, 0.15915494309189535, TMP_VADDR),
                    r3.v_sin_f32_e32(_dst(u), TMP_VDATA)]
    case AMDOps.MAX:
      return _max(u)
    case AMDOps.SHL:
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_lshlrev_b32_e64(_dst(u), _src(u.src[1]), a)]
    case AMDOps.SHR:
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      if u.dtype.scalar() in dtypes.sints: return pre + [r3.v_ashrrev_i32_e64(_dst(u), _src(u.src[1]), a)]
      return pre + [r3.v_lshrrev_b32_e64(_dst(u), _src(u.src[1]), a)]
    case AMDOps.AND:
      pre, b = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_and_b32_e32(_dst(u), _src(u.src[0]), b)]
    case AMDOps.OR:
      pre, b = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_or_b32_e32(_dst(u), _src(u.src[0]), b)]
    case AMDOps.XOR:
      pre, b = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_xor_b32_e32(_dst(u), _src(u.src[0]), b)]
    case AMDOps.CMPLT:
      return _cmp_lt(u)
    case AMDOps.CMPNE:
      return _cmp_ne(u)
    case AMDOps.CMPEQ:
      return _cmp_eq(u)
    case AMDOps.WHERE:
      pre, true_val = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_cndmask_b32_e32(_dst(u), _src(u.src[2]), true_val)]
    case AMDOps.LOAD:
      pre, addr = _scaled_addr(_dst(u) if _reg_slots(u) == 1 else TMP_VADDR, u.src[1], _mem_itemsize(u.dtype))
      pre, addr = _masked_addr(pre, addr, masked)
      return pre + _global_load_insts(u, addr)
    case AMDOps.STORE:
      itemsize, byte_off = _mem_itemsize(u.src[2].dtype), _mem_byte_off(u)
      if store_addr_cache is not None and not masked:
        pre, addr, byte_off = store_addr_cache.addr(u.src[1], itemsize, byte_off)
      elif byte_off > 0xfff:
        pre, addr, byte_off = _apply_byte_off(TMP_VADDR, byte_off, u.src[1], itemsize)
      else:
        pre, addr = _scaled_addr(TMP_VADDR, u.src[1], itemsize)
        pre2, addr, byte_off = _apply_byte_off(addr, byte_off)
        pre = pre + pre2
      pre, addr = _masked_addr(pre, addr, masked)
      return pre + _global_store_insts(u, addr, byte_off)
    case AMDOps.ATOMIC_ADD:
      if u.src[2].dtype is not dtypes.float32: raise CompileError(f"f32 atomic only, got {u.src[2].dtype}")
      pre, addr = _scaled_addr(TMP_VADDR, u.src[1], u.src[2].dtype.itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
      return pre + dpre + [r3.global_atomic_add_f32(addr=addr, data=data, saddr=_src(u.src[0]), vdst=TMP_VDATA),
                           r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
    case AMDOps.LLOAD:
      slots = _reg_slots(u)
      pre, addr = _local_addr(u.src[0], u.src[1], _mem_itemsize(u.dtype))
      pre, addr = _masked_addr(pre, addr, masked)
      off = _lds_byte_off(u)
      if slots == 8 and u.dtype.scalar() is dtypes.half:
        return pre + [r3.ds_load_b128(vdst=_reg_chunk(greg(u), 0, 4), addr=addr, **_ds_off(off)),
                      r3.ds_load_b128(vdst=_reg_chunk(greg(u), 4, 4), addr=addr, **_ds_off(off + 16))]
      if (local_load:=_local_load(u.dtype, _elem_count(u))) is None: raise CompileError(f"no lds load {u.dtype}")
      return pre + [local_load(vdst=_dst(u), addr=addr, **_ds_off(off))]
    case AMDOps.LSTORE:
      if (local_store:=_local_store(u.src[2].dtype, _elem_count(u.src[2]))) is None: raise CompileError(f"no lds store {u.src[2].dtype}")
      pre, addr = _local_addr(u.src[0], u.src[1], _mem_itemsize(u.src[2].dtype))
      pre, addr = _masked_addr(pre, addr, masked)
      dpre, data = ([], _full_src(u.src[2])) if _reg_slots(u.src[2]) > 1 else _vgpr_data(TMP_VDATA, u.src[2])
      # No per-store lgkmcnt — scoreboard flushes once before BARRIER / LLOAD use (hand-style).
      return pre + dpre + [local_store(addr=addr, data0=data, **_ds_off(_lds_byte_off(u)))]
    case AMDOps.SLOAD:
      slots = _reg_slots(u)
      pre, addr = _scratch_addr(u.src[0], u.src[1], u.dtype.itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      if slots > 1:
        if u.dtype is not dtypes.float32: raise CompileError(f"no vec scratch load {u.dtype}")
        return pre + [r3.scratch_load_b32(addr=addr, vdst=_reg_lane(greg(u), i), offset=i*4, sve=1) for i in range(slots)]
      if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"no scratch load {u.dtype}")
      return pre + [scratch_load(addr=addr, vdst=_dst(u), offset=0, sve=1)]
    case AMDOps.SSTORE:
      slots = _reg_slots(u.src[2])
      pre, addr = _scratch_addr(u.src[0], u.src[1], u.src[2].dtype.itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      if slots > 1:
        if u.src[2].dtype is not dtypes.float32: raise CompileError(f"no vec scratch store {u.src[2].dtype}")
        return pre + [r3.scratch_store_b32(addr=addr, data=_reg_lane(greg(u.src[2]), i), offset=i*4, sve=1) for i in range(slots)] + \
               [r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
      if (scratch_store:=_scratch_store(u.src[2].dtype)) is None:
        raise CompileError(f"no scratch store {u.src[2].dtype}")
      dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
      return pre + dpre + [scratch_store(addr=addr, data=data, offset=0, sve=1), r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
    case AMDOps.BARRIER:
      return [r3.s_barrier()]
    case AMDOps.FILL:
      if greg(u).index < 256: raise CompileError("no sgpr scratch fill")
      slots = _reg_slots(u)
      if u.src[0].arg + (slots-1)*4 >= 4096: raise CompileError("scratch fill oob")
      if slots > 1:
        # raw VGPR lanes (f32 packs or half2 LDS loads)
        loads = [r3.scratch_load_b32(addr=TMP_VADDR, vdst=_reg_lane(greg(u), i), offset=u.src[0].arg + i*4, sve=1)
                 for i in range(slots)]
        return [r3.v_mov_b32_e32(TMP_VADDR, 0)] + loads
      if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"no scratch fill {u.dtype}")
      return [r3.v_mov_b32_e32(TMP_VADDR, 0), scratch_load(addr=TMP_VADDR, vdst=_dst(u), offset=u.src[0].arg, sve=1)]
    case AMDOps.SPILL:
      if greg(u.src[1]).index < 256: raise CompileError("no sgpr scratch spill")
      slots = _reg_slots(u.src[1])
      if u.src[0].arg + (slots-1)*4 >= 4096: raise CompileError("scratch spill oob")
      if slots > 1:
        # raw VGPR lanes (f32 packs or half2 LDS loads)
        return [r3.v_mov_b32_e32(TMP_VADDR, 0)] + \
               [r3.scratch_store_b32(addr=TMP_VADDR, data=_reg_lane(greg(u.src[1]), i), offset=u.src[0].arg + i*4, sve=1)
                for i in range(slots)] + \
               [r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
      if (scratch_store:=_scratch_store(u.src[1].dtype)) is None:
        raise CompileError(f"no scratch spill {u.src[1].dtype}")
      return [r3.v_mov_b32_e32(TMP_VADDR, 0), scratch_store(addr=TMP_VADDR, data=_src(u.src[1]), offset=u.src[0].arg, sve=1),
              r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
    case AMDOps.CMP_GE:
      pre0, a = _sgpr_data(TMP_SDATA0, u.src[0])
      pre1, b = _sgpr_data(TMP_SDATA1, u.src[1])
      return pre0 + pre1 + [r3.s_cmp_ge_u32(a, b)]
    case AMDOps.IF_MASK:
      if u.src[0].op is Ops.INS and u.src[0].arg in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ):
        return [r3.s_and_saveexec_b64(TMP_EXEC, VCC)]
      pre, gate = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_cmp_ne_u32_e32(0, gate), r3.s_and_saveexec_b64(TMP_EXEC, VCC)]
    case AMDOps.END_MASK:
      return [r3.s_mov_b64(EXEC, TMP_EXEC)]
  raise CompileError(f"cannot encode {u.arg}")

def _hoist_lloads_before_extracts(ops:list[UOp]) -> list[UOp]:
  # Hand kernel: issue all ds_loads, one lgkmcnt, then use. Linearize emits LLOAD+EXTRACT pairs;
  # hoist LLOADs in each streak so the scoreboard waits once for the batch.
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if u.op is Ops.INS and u.arg is AMDOps.LLOAD:
      j, lloads, extracts = i, [], []
      while j < len(ops):
        v = ops[j]
        if v.op is Ops.INS and v.arg is AMDOps.LLOAD:
          lloads.append(v)
          j += 1
        elif v.op is Ops.INS and v.arg is AMDOps.EXTRACT:
          extracts.append(v)
          j += 1
        else: break
      out.extend(lloads)
      out.extend(extracts)
      i = j
    else:
      out.append(u)
      i += 1
  return out

# Addr / pack ops that can issue while a prior WMMA's inputs stay live.
_SINKABLE_PAST_WMMA = frozenset({
  AMDOps.LOAD, AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV,
  # ADD excluded: sinking past loop S_ADD reorders the trip counter vs the last WMMA.
  AMDOps.SUB, AMDOps.MUL, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR,
})

def _sink_wmma_past_loads(ops:list[UOp]) -> list[UOp]:
  # Sink WMMA (+ its ACC EXTRACTs) past independent loads — not past peer WMMAs.
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and u.arg is AMDOps.WMMA):
      i += 1
      continue
    wmma_dst = _reg_idxs(u)
    wmma_src = set().union(*(_reg_idxs(s) for s in u.src))
    # Keep EXTRACTs of this ACC glued to the WMMA so they don't block the sink.
    end = i + 1
    while end < len(out):
      v = out[end]
      if not (v.op is Ops.INS and v.arg is AMDOps.EXTRACT): break
      if not (_reg_idxs(v.src[0]) & wmma_dst): break
      end += 1
    block = end - i
    j = end
    while j < len(out):
      v = out[j]
      if v.op is not Ops.INS or v.arg not in _SINKABLE_PAST_WMMA: break
      # Keep tile-local schedule: don't sink past scalar half A loads. Otherwise all A packs
      # first and B B128 lands after a full wait — loses A/B VMEM overlap vs LLVM.
      if v.arg is AMDOps.LOAD and v.dtype.scalar() is dtypes.half and _elem_count(v) == 1: break
      v_dst = _reg_idxs(v)
      v_src = set().union(*(_reg_idxs(s) for s in v.src))
      if v_src & wmma_dst: break
      # Don't sink past a reload of this WMMA's inputs (e.g. next A tile into same VGPRs).
      if v_dst & wmma_src: break
      if v_dst & wmma_dst: break
      j += 1
    if j > end:
      chunk = out[i:end]
      del out[i:end]
      insert_at = j - block
      out[insert_at:insert_at] = chunk
      i = insert_at
      continue
    i += 1
  return out

def _hoist_loads_before_wmma(ops:list[UOp]) -> list[UOp]:
  # Bubble LOAD/PACK_F16 (+ int addr) above preceding WMMAs when independent.
  # Must not clobber a WMMA's A/B/ACC — UPCAST≥4 reuses PACK VGPRs across tiles; hoisting
  # the next tile's load above a prior WMMA left only the last A in those regs (wrong results).
  if not any(u.op is Ops.INS and u.arg is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and u.arg in (AMDOps.LOAD, AMDOps.PACK_F16)):
      i += 1
      continue
    # Only hoist wide B (half×8+) / vec-load packs above WMMA — not scalar A loads.
    # Hoisting scalar A above prior WMMA collapsed the schedule to all-A-then-B (no A/B overlap).
    if u.arg is AMDOps.LOAD and u.dtype.scalar() is dtypes.half and _elem_count(u) == 1:
      i += 1
      continue
    if u.arg is AMDOps.PACK_F16 and not _pack_f16_is_vec_load(u):
      i += 1
      continue
    # Grow a hoistable prefix of addr ALU ending at this LOAD/PACK (and following PACK).
    start = i
    while start > 0:
      p = out[start - 1]
      if p.op is Ops.INS and p.arg in (AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.SHL, AMDOps.SHR,
                                       AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.MOV) and \
         p.dtype.scalar() in dtypes.ints:
        start -= 1
        continue
      break
    end = i + 1
    if end < len(out) and out[end].op is Ops.INS and out[end].arg is AMDOps.PACK_F16:
      end += 1
    chunk_src = set().union(*(set().union(*(_reg_idxs(s) for s in out[k].src)) for k in range(start, end)))
    chunk_dst = set().union(*(_reg_idxs(out[k]) for k in range(start, end)))
    dest = start
    while dest > 0:
      p = out[dest - 1]
      if p.op is Ops.INS and p.arg is AMDOps.EXTRACT:
        # Walk through ACC EXTRACTs glued after WMMA, but never past EXTRACTs of a still-live
        # vector LOAD whose VGPRs the hoisted chunk would clobber (half 8×8: B addr ADDs into
        # A’s B128 dests → sq_intr hang / wrong mul).
        ext_src = set().union(*(_reg_idxs(s) for s in p.src))
        if chunk_dst & (ext_src | _reg_idxs(p)): break
        dest -= 1
        continue
      if p.op is Ops.INS and p.arg is AMDOps.WMMA:
        wmma_src = set().union(*(_reg_idxs(s) for s in p.src))
        if _reg_idxs(p) & chunk_src: break
        if chunk_dst & (wmma_src | _reg_idxs(p)): break
        dest -= 1
        continue
      break
    if dest < start:
      chunk = out[start:end]
      del out[start:end]
      out[dest:dest] = chunk
      i = dest + len(chunk)
      continue
    i += 1
  return out

def _vm_load_count(insts:list) -> int:
  return sum(1 for i in insts if (n:=getattr(i, "op_name", "")) and
             (n.startswith("GLOBAL_LOAD") or n.startswith("SCRATCH_LOAD") or
              n.startswith("BUFFER_LOAD") or n.startswith("FLAT_LOAD")))

def _split_scale_and_loads(emitted:list) -> tuple[list, list]:
  """Split dest-as-addr scalar load emit into addr ALU vs trailing VMEM loads."""
  i = len(emitted)
  while i > 0 and _vm_load_count([emitted[i - 1]]): i -= 1
  return emitted[:i], emitted[i:]

def _clauseable_half_gload(u:UOp, skip:set[UOp], mask_depth:int) -> bool:
  # Scalar half global LOAD with dest-as-addr (no mask/TMP). Streak → hoist scales + s_clause.
  if u in skip or mask_depth or u.op is not Ops.INS or u.arg is not AMDOps.LOAD: return False
  return _reg_slots(u) == 1 and u.dtype.scalar() is dtypes.half

def insts_from_linear(lin:UOp):
  ops = list(lin.src)
  skip = _compute_amd_skip(ops) | _fused_d16_hi_loads(ops)
  mask_depth = 0
  # vm: (dest_regs, n_vmem_ops) in issue order — soft vmcnt must count ops, not UOps
  # (PACK_F16 can emit 16 loads; treating that as 1 desyncs vmcnt → MMU faults).
  pending_vm: list[tuple[set[int], int]] = []
  pending: dict[str, set[int]] = {"lgkm": set(), "vs": set()}
  items, targets, byte = [], {}, 0
  store_addr_cache = _StoreAddrCache()
  def emit(inst):
    nonlocal byte
    items.append(inst)
    byte += len(inst.to_bytes())
  def wait_vm(allow:int=0):
    total = sum(n for _, n in pending_vm)
    if not total: return
    allow = max(0, min(allow, total))
    emit(_wait_for_domain("vm", allow))
    done = total - allow
    while done > 0 and pending_vm:
      regs, n = pending_vm[0]
      if n <= done:
        pending_vm.pop(0)
        done -= n
      else:
        pending_vm[0] = (regs, n - done)
        done = 0
  def flush(*domains:str):
    for domain in domains:
      if domain == "vm": wait_vm(0)
      elif pending[domain]:
        emit(_wait_for_domain(domain))
        pending[domain].clear()
  def flush_regs(regs:set[int]):
    need_i = next((i for i in range(len(pending_vm) - 1, -1, -1) if pending_vm[i][0] & regs), -1)
    if need_i >= 0: wait_vm(sum(n for _, n in pending_vm[need_i + 1:]))
    if pending["lgkm"] & regs: flush("lgkm")
  def note_vm(regs:set[int], insts:list):
    if (n:=_vm_load_count(insts)) and regs: pending_vm.append((regs, n))
  def _pending_src(regs:set[int]) -> bool:
    return any(pr & regs for pr, _ in pending_vm) or bool(pending["lgkm"] & regs)
  scheduled = _hoist_loads_before_wmma(_sink_wmma_past_loads(_hoist_lloads_before_extracts(ops)))
  oi = 0
  while oi < len(scheduled):
    u = scheduled[oi]
    if u.op is Ops.INS and u.arg is AMDOps.LABEL:
      flush("vm", "lgkm", "vs")
      store_addr_cache.clear()
      targets[u.tag] = byte
      oi += 1
      continue
    if u.op is Ops.INS and u.arg in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1):
      flush("vm", "lgkm", "vs")
      store_addr_cache.clear()
      inst = r3.s_branch(0) if u.arg is AMDOps.BRANCH else r3.s_cbranch_scc1(0)
      items.append((inst, u.tag, byte))
      byte += len(inst.to_bytes())
      oi += 1
      continue
    if _needs_vm_flush(u):
      # Soft allow can no-op when dest ACC intersects an early pending batch. Drain VM before WMMA.
      # Also drain lgkm on WMMA srcs — TC_LDS_AB feeds A/B from DS_LOAD; skipping that wait
      # left WMMA reading in-flight LDS data (NaN/inf). Hand kernel waits lgkmcnt(0) first.
      if u.op is Ops.INS and u.arg is AMDOps.WMMA:
        flush("vm")
        flush_regs(set().union(*(_reg_idxs(s) for s in u.src)))
      else: flush_regs(set().union(*(_reg_idxs(s) for s in u.src), _reg_idxs(u)))
    if u.op is Ops.INS and u.arg is AMDOps.IF_MASK:
      store_addr_cache.clear()
      emitted = list(insts_for_uop(u, skip))
      for inst in emitted: emit(inst)
      mask_depth += 1
      if (domain:=_wait_domain_for_load(u)) is not None:
        if domain == "vm": note_vm(_reg_idxs(u), emitted)
        else: pending[domain] |= _reg_idxs(u)
      oi += 1
      continue
    # LDS stores must complete before s_barrier / next LLOAD (hand: waitcnt then barrier).
    if u.op is Ops.INS and u.arg is AMDOps.BARRIER:
      flush("lgkm")
    elif u.op is Ops.INS and u.arg is AMDOps.LLOAD and -1 in pending["lgkm"]:
      flush("lgkm")
    masked = mask_depth > 0 and u.op is Ops.INS and u.arg in _MASKED_MEM
    if u in skip:
      if u.op is Ops.INS and u.arg is AMDOps.END_MASK: mask_depth -= 1
      oi += 1
      continue
    # Cluster scalar half loads: all dest-as-addr scales, then s_clause + tight VMEM (LLVM-style B).
    # ~+40% @2048 vs no clause on gfx1100 — always on (no opt-out knob).
    if _clauseable_half_gload(u, skip, mask_depth):
      j = oi + 1
      while j < len(scheduled) and _clauseable_half_gload(scheduled[j], skip, mask_depth): j += 1
      if j - oi >= 2:
        parts = [list(insts_for_uop(scheduled[k], skip, False)) for k in range(oi, j)]
        if all(_vm_load_count(p) == 1 for p in parts):
          store_addr_cache.clear()
          scales, loads = [], []
          for p in parts:
            sc, ld = _split_scale_and_loads(p)
            scales.extend(sc)
            loads.extend(ld)
          for inst in scales: emit(inst)
          emit(r3.s_clause(simm16=len(loads) - 1))
          for inst in loads: emit(inst)
          for k, p in enumerate(parts):
            note_vm(_reg_idxs(scheduled[oi + k]), p)
          oi = j
          continue
    is_store = u.op is Ops.INS and u.arg is AMDOps.STORE
    emitted = list(insts_for_uop(u, skip, masked, store_addr_cache if is_store else None))
    # VALU copy of an outstanding VMEM/LDS dest must wait first (PACK/MOV across pools).
    if emitted and u.op is Ops.INS and u.arg in (AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV):
      src = set().union(*(_reg_idxs(s) for s in u.src))
      if src and _pending_src(src): flush_regs(src)
    # EXTRACT between C-stores often emits nothing (pack+lane alias); only clobber CSE on real emits.
    # Keep page CSE across CAST (uses TMP_VDATA, not TMP_VADDR) — cast-before-store otherwise
    # re-scales the C base for every half store (~100 extra V_LSHL_ADD).
    # half×16 STORE may V_ADD into TMP_VADDR for the second b128 — drop page CSE.
    if is_store and any(getattr(i, "op_name", "") == "V_ADD_NC_U32_E64" for i in emitted):
      store_addr_cache.clear()
    elif not is_store and any(getattr(i, "vdst", None) == TMP_VADDR for i in emitted):
      store_addr_cache.clear()
    for inst in emitted: emit(inst)
    if u.op is Ops.INS and u.arg is AMDOps.END_MASK: mask_depth -= 1
    if (domain:=_wait_domain_for_load(u)) is not None:
      if domain == "vm": note_vm(_reg_idxs(u), emitted)
      else: pending[domain] |= _reg_idxs(u)
    if (domain:=_wait_domain_for_store(u)) is not None:
      pending[domain] |= _store_src_regs(u)
    oi += 1
  # Drain outstanding global stores before s_endpgm (appended by the renderer).
  flush("vs")
  insts = []
  for item in items:
    if isinstance(item, tuple):
      inst, target, branch_byte = item
      if target not in targets: raise CompileError(f"missing branch {target}")
      delta = (targets[target] - (branch_byte + len(inst.to_bytes()))) // 4
      if not -0x8000 <= delta <= 0x7fff: raise CompileError(f"branch oob {target}")
      inst.simm16 = delta & 0xffff
    else: inst = item
    insts.append(inst)
  return insts

# ***** TC_LDS_AB staging (codegen hooks via AMDRenderer.pm_stage_wmma_ab) *****
_WMMA_LDS_AXES, _WMMA_LDS_LOOP_BASE, _WMMA_TC = (AxisType.LOCAL, AxisType.WARP), 200, 16

def _range_size(r:UOp) -> int:
  return int(r.src[0].vmax) if r.src[0].op is Ops.CONST else int(r.vmax) + 1

def _linearize_ranges(axes:list[UOp]) -> UOp:
  out = axes[0]
  for a in axes[1:]: out = out * _range_size(a) + a
  return out

def _tid_axes(coop:list[UOp]) -> list[UOp]|None:
  # LOCALs by range id, WARP last in the product. gpudims maps WARP→lidx0 so
  # tid == hardware linear id when local_size is (32, …).
  locals_ = sorted([u for u in coop if u.arg[1] is AxisType.LOCAL], key=lambda u: u.arg[0])
  warps = [u for u in coop if u.arg[1] is AxisType.WARP]
  if len(locals_) < 2 or len(warps) != 1: return None
  if _range_size(warps[0]) != 32: return None
  return locals_ + warps

def _index_row_stride(idx:UOp) -> int|None:
  e = idx.src[1] if idx.op is Ops.INDEX else idx
  if e.op is not Ops.ADD: return None
  for side in e.src:
    if side.op is Ops.MUL:
      for t in side.src:
        if t.op is Ops.CONST and int(t.arg) > 1: return int(t.arg)
  return None

def _delinearize_ranges(linear:UOp, axes:list[UOp]) -> dict[UOp, UOp]:
  """Map a flat index onto axes (last axis fastest)."""
  subs, rem = {}, linear
  for a in reversed(axes):
    sz = _range_size(a)
    subs[a] = rem % sz
    rem = rem // sz
  return subs

def _bounce_a_shared(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                     as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Shared A via tid bufferize: LDS[tid,k]=A[g*block+tid,k], read [major,k].
  # STACK(8)×chunk → GLOBAL B128 (scalar ept is U16 and loses to frag-wide on INS).
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  buf, stride = ab.src[0], _index_row_stride(ab)
  if stride is None: return None
  reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
  grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
  if len(reds) != 1 or len(grids) != 1: return None
  k_tile, g_wg = reds[0], grids[0]
  op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
  if op_local is None: return None
  tile_prod = prod(_range_size(t) for t in tile) if tile else 1
  block = _range_size(op_local) * tile_prod * _WMMA_TC
  fsz = prod(_range_size(f) for f in frag)
  if block != threads or fsz != _WMMA_TC or fsz % 8: return None
  vec = 8
  chunk = UOp.range(fsz // vec, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
  elems = [buf.index((g_wg * block + tid) * stride + (k_tile * _WMMA_TC + chunk * vec + j)) for j in range(vec)]
  staged = UOp.stack(*elems).bufferize(*tid_axes, chunk, arg=BufferizeOpts(None, AddrSpace.LOCAL))
  # Flat 1D like B: major*16+k peels to base+imm for K-contig B128 LLOADs.
  flat = staged.reshape(threads * fsz)
  major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
  k_r = _linearize_ranges(as_up_frag)
  read = flat.index(major * fsz + k_r)
  return read.contract(*as_up) if as_up else read

def _bounce_frag_wide(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                      as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Identity fill along frag (unit-stride operand, typically A). WEAK(fsz/8)×STACK(8).
  # Drop N-LOCAL from bufferize when A only uses one LOCAL (M): threads that differ only in
  # N-local write the same cells (A independent of ln) → ~2× smaller A LDS.
  fsz, vec = prod(_range_size(f) for f in frag), 8
  if not frag or fsz % vec: return None
  tile_w = [r.replace(arg=(r.arg[0] + _WMMA_LDS_LOOP_BASE + i * 50 + n, AxisType.WEAK)) for n, r in enumerate(tile)]
  chunk = UOp.range(fsz // vec, _WMMA_LDS_LOOP_BASE + i * 50 + 40, AxisType.WEAK)
  elems = []
  for j in range(vec):
    sub = dict(zip(tile, tile_w))
    sub.update(_delinearize_ranges(chunk * vec + j, frag))
    elems.append(ab.substitute(sub))
  ab_locals = [u for u in coop if u.arg[1] is AxisType.LOCAL and any(x is u for x in ab.toposort())]
  write_coop = [u for u in coop if u.arg[1] is not AxisType.LOCAL or u in ab_locals]
  staged = UOp.stack(*elems).bufferize(*write_coop, *tile_w, chunk, arg=BufferizeOpts(None, AddrSpace.LOCAL))
  flat = staged.reshape(*[_range_size(x) for x in write_coop + tile_w], fsz)
  frag_lin = _linearize_ranges(as_up_frag) if len(as_up_frag) > 1 else as_up_frag[0]
  indexed = flat.index(*write_coop, *as_up_tile, frag_lin)
  return indexed.contract(*as_up) if as_up else indexed

def _bounce_tid_wide(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                     as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Tid-partitioned fill for strided B: STACK(8) GLOBAL B128 + scatter DS_STORE to (n,k) LDS.
  # Flat LDS + shared base+imm offsets → one addr VGPR (isel _peel_add_imm); K-contig reads B128.
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  if threads < 32 or threads % 32: return None
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  buf, stride = ab.src[0], _index_row_stride(ab)
  if stride is None: return None
  reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
  # WEAK = former LOOP (#17283). GLOBAL covers workgroup tiles after TC.
  grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
  if len(reds) != 1 or len(grids) != 1: return None
  k_tile, g_wg = reds[0], grids[0]
  op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
  if op_local is None: return None
  tile_prod = prod(_range_size(t) for t in tile) if tile else 1
  block = _range_size(op_local) * tile_prod * _WMMA_TC
  vec = 8
  ept_n = (_WMMA_TC * block) // threads
  if ept_n < vec or ept_n % vec or (_WMMA_TC * block) != threads * ept_n: return None
  t_per_k = block // ept_n
  if t_per_k < 1 or block != t_per_k * ept_n or threads != _WMMA_TC * t_per_k: return None
  k, n_base = tid // t_per_k, (tid % t_per_k) * ept_n
  # Flat (n,k) row-major: addr = n*16+k. One base + j*16 peels to ds_store offset.
  local = UOp.placeholder((block * _WMMA_TC,), ab.dtype.scalar(), slot=100 + i, addrspace=AddrSpace.LOCAL)
  elems, stores = [], []
  # ept_n==vec (default 2×2): no chunk range — keeps addr math short-lived (avoids spills).
  if ept_n == vec:
    base = n_base * _WMMA_TC + k
    for j in range(vec):
      elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n_base + j)))
      stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
    flat = local.after(UOp.group(*stores).end(*tid_axes))
  else:
    chunk = UOp.range(ept_n // vec, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
    base = (n_base + chunk * vec) * _WMMA_TC + k
    for j in range(vec):
      n = n_base + chunk * vec + j
      elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n)))
      stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
    flat = local.after(UOp.group(*stores).end(*tid_axes, chunk))
  major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
  k_r = _linearize_ranges(as_up_frag)
  read = flat.index(major * _WMMA_TC + k_r)
  return read.contract(*as_up) if as_up else read

def stage_wmma_ab_bounce(wmma:UOp, coop:list[UOp]) -> UOp|None:
  # Hybrid bounce: shared A (tid-fill, major-read) when block==threads; else frag-wide A; tid-wide B.
  # expand_wmma slices STACK(16*tile,). Tile product ≤8 (LLOAD/PACK VGPR pools).
  news: list[UOp] = []
  changed = False
  for i, ab in enumerate(wmma.src[:2]):
    if any(u.op is Ops.STAGE for u in ab.toposort()):
      news.append(ab)
      continue
    # Both A and B must be GLOBAL INDEX to stage. Computed operands (eye/WHERE) stay
    # unstaged; mixing with a staged peer breaks expand_broadcast (eye@B IndexError).
    if ab.op is not Ops.INDEX or ab.addrspace != AddrSpace.GLOBAL: return None
    if not any(u in coop for u in ab.toposort() if u.op is Ops.RANGE): return None
    frag_rns = {rn for rn, _ in wmma.arg[4][i]}
    frag = list(dict.fromkeys(u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[0] in frag_rns))
    tile = list(dict.fromkeys(u for u in ab.toposort()
      if u.op is Ops.RANGE and u.arg[1] in (AxisType.UPCAST, AxisType.UNROLL) and u.arg[0] not in frag_rns))
    as_up_tile = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in tile]
    as_up_frag = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in frag]
    as_up = as_up_tile + as_up_frag
    if i == 0 and (ret := _bounce_a_shared(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    elif i == 1 and (ret := _bounce_tid_wide(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    elif (ret := _bounce_frag_wide(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    else:
      read_axes = tile + frag
      write_axes = [r.replace(arg=(r.arg[0] + _WMMA_LDS_LOOP_BASE + i * 50 + (25 if n < len(tile) else 0), AxisType.WEAK))
                    for n, r in enumerate(read_axes)]
      sval = ab.substitute(dict(zip(read_axes, write_axes))) if write_axes else ab
      staged = sval.bufferize(*coop, *write_axes, arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*coop, *as_up)
      news.append(staged.contract(*as_up) if as_up else staged)
    changed = True
  if not changed: return None
  _in0, _in1, out0 = wmma.arg[4]
  return wmma.replace(src=(news[0], news[1], wmma.src[2]), arg=(*wmma.arg[:4], ((), (), out0)))

def stage_wmma_ab_tid(wmma:UOp, coop:list[UOp]) -> UOp|None:
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  if threads < 32 or threads % 32: return None
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  news: list[UOp] = []
  changed = False
  for i, ab in enumerate(wmma.src[:2]):
    if any(u.op is Ops.STAGE for u in ab.toposort()):
      news.append(ab)
      continue
    if ab.op is not Ops.INDEX or ab.addrspace != AddrSpace.GLOBAL: return None
    if not any(u in coop for u in ab.toposort() if u.op is Ops.RANGE): return None
    buf, stride = ab.src[0], _index_row_stride(ab)
    if stride is None: return None
    frag_rns = {rn for rn, _ in wmma.arg[4][i]}
    frag = list(dict.fromkeys(u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[0] in frag_rns))
    tile = list(dict.fromkeys(u for u in ab.toposort()
      if u.op is Ops.RANGE and u.arg[1] in (AxisType.UPCAST, AxisType.UNROLL) and u.arg[0] not in frag_rns))
    if not frag or prod(_range_size(f) for f in frag) != _WMMA_TC: return None
    reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
    grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
    if len(reds) != 1 or len(grids) != 1: return None
    k_tile, g_wg = reds[0], grids[0]
    op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
    if op_local is None: return None
    tile_prod = prod(_range_size(t) for t in tile) if tile else 1
    block = _range_size(op_local) * tile_prod * _WMMA_TC
    ept_n = (block * _WMMA_TC) // threads if i == 0 else (_WMMA_TC * block) // threads
    if ept_n < 1 or (block * _WMMA_TC if i == 0 else _WMMA_TC * block) != threads * ept_n: return None
    ept = UOp.range(ept_n, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
    as_up_tile = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in tile]
    as_up_frag = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in frag]
    as_up = as_up_tile + as_up_frag
    major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
    k_r = _linearize_ranges(as_up_frag)
    if i == 0:
      if block != threads: return None
      gval = buf.index((g_wg * block + tid) * stride + (k_tile * _WMMA_TC + ept))
      staged = gval.bufferize(*tid_axes, ept, arg=BufferizeOpts(None, AddrSpace.LOCAL))
      read = staged.reshape(threads, ept_n).index(major, k_r)
    else:
      # B transpose: STACK(8) GLOBAL B128 + flat LDS base+imm scatter (same as bounce).
      t_per_k = block // ept_n
      if ept_n % 8 or t_per_k < 1 or block != t_per_k * ept_n or threads != _WMMA_TC * t_per_k: return None
      vec = 8
      k, n_base = tid // t_per_k, (tid % t_per_k) * ept_n
      local = UOp.placeholder((block * _WMMA_TC,), ab.dtype.scalar(), slot=100 + i, addrspace=AddrSpace.LOCAL)
      elems, stores = [], []
      if ept_n == vec:
        base = n_base * _WMMA_TC + k
        for j in range(vec):
          elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n_base + j)))
          stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
        read = local.after(UOp.group(*stores).end(*tid_axes)).index(major * _WMMA_TC + k_r)
      else:
        chunk = UOp.range(ept_n // vec, _WMMA_LDS_LOOP_BASE + i * 50 + 1, AxisType.WEAK)
        base = (n_base + chunk * vec) * _WMMA_TC + k
        for j in range(vec):
          n = n_base + chunk * vec + j
          elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n)))
          stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
        read = local.after(UOp.group(*stores).end(*tid_axes, chunk)).index(major * _WMMA_TC + k_r)
    news.append(read.contract(*as_up) if as_up else read)
    changed = True
  if not changed: return None
  _in0, _in1, out0 = wmma.arg[4]
  return wmma.replace(src=(news[0], news[1], wmma.src[2]), arg=(*wmma.arg[:4], ((), (), out0)))

def stage_wmma_ab_to_local(wmma:UOp) -> UOp|None:
  if wmma.op is not Ops.WMMA: return None
  coop = list(dict.fromkeys(u for u in wmma.toposort() if u.op is Ops.RANGE and u.arg[1] in _WMMA_LDS_AXES))
  if not any(u.arg[1] == AxisType.LOCAL for u in coop): return None
  if getenv("TC_LDS_TID", 0):
    if (ret := stage_wmma_ab_tid(wmma, coop)) is not None: return ret
  return stage_wmma_ab_bounce(wmma, coop)

pm_stage_wmma_ab = PatternMatcher([(UPat(Ops.WMMA, name="wmma"), stage_wmma_ab_to_local)])

def apply_tc_hand_opts(tk, rngs):
  from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
  lds_ab = getenv("TC_LDS_AB", 0)
  # Register path: ALLOW_UPCAST16 defaults on → product-16 (4×4). LOCAL=4 (beats LOCAL=2 on gfx1100).
  # LDS path keeps ALLOW_UPCAST16 off (spills); product-8 + LOCAL=2×2 remains the LDS default.
  up_cap = getenv("TC_UPCAST", 4)
  loc_cap = getenv("TC_LOCAL", 2 if lds_ab else 4)
  up16 = _allow_upcast16()
  max_tiles = min(getenv("TC_UPCAST_TILES", 16 if up16 else 8), 8 if not up16 else 10**9)
  if lds_ab and getenv("ALLOW_LDS_PRODUCT8", 1) == 0:
    up_cap, max_tiles = min(up_cap, 2), min(max_tiles, 4)
  local_dims, loc_szs = ([0, 1], [8, 4, 2]) if lds_ab else ([0], [4, 2])
  def do_local():
    if lds_ab:
      # Asymmetric LOCAL (≤loc_cap × ≤2). Symmetric 4×4 TDRs on display GPUs.
      for tc_dim, cap in ((0, loc_cap), (1, min(2, loc_cap) if loc_cap else 0)):
        if not cap: continue
        if (szs := [sz for sz in loc_szs if sz <= cap and rngs[tc_dim].src[0].divides(sz) is not None]):
          try: rngs[tc_dim] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
          except KernelOptError: pass
      return
    for tc_dim in local_dims:
      if (szs := [sz for sz in loc_szs if sz <= loc_cap and rngs[tc_dim].src[0].divides(sz) is not None]):
        try: rngs[tc_dim] = tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
        except KernelOptError: pass
  def do_upcast() -> int:
    tiles = 1
    for i, tc_dim in enumerate([1, 0]):
      other = rngs[0 if tc_dim == 1 else 1]
      # Avoid single-axis UPCAST=4 (product-4 as 4×1): wrong on register path. Leave a factor ≥2
      # for the other dim when it can still upcast.
      szs = []
      for sz in [5, 4, 3, 2]:
        if sz > up_cap or tiles * sz > max_tiles or rngs[tc_dim].src[0].divides(sz) is None: continue
        rem = max_tiles // (tiles * sz)
        if i == 0 and rem == 1 and other.src[0].divides(2) is not None and sz > 2: continue
        szs.append(sz)
      if szs:
        rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
        tiles *= szs[0]
    return tiles
  if lds_ab:
    do_local()
    tiles = do_upcast()
    # UNROLL multiplies WMMA STACK tiles; past max_tiles expand soft-fails then unroll_axis
    # IndexErrors. Only apply when the upcast×unroll product still fits the LDS expand budget.
    if (ku := getenv("TC_LDS_UNROLL", 0)) and tk.unrollable_dims and tiles * ku <= max_tiles:
      try: tk.apply_opt(Opt(OptOps.UNROLL, 0, ku))
      except KernelOptError: pass
  else:
    do_upcast()
    do_local()
  # TC_LDS_GROUP omitted: GROUP≥2 on WMMA half GEMM needs >64KB LDS (always KernelOptError).

def llvm_tc_hand_opts(tk, rngs):
  # AMDLLVM: stock TC schedule unless TC_LDS_AB (shared staging with AMDRenderer).
  if getenv("TC_LDS_AB", 0): return apply_tc_hand_opts(tk, rngs)
  from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
  for tc_dim in [1, 0]:
    if (szs := [sz for sz in [5, 4, 3, 2] if rngs[tc_dim].src[0].divides(sz) is not None]):
      rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
  if (szs := [sz for sz in [4, 2] if rngs[0].src[0].divides(sz) is not None]):
    try: tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
    except KernelOptError: pass

def install_amdllvm_tc(cls):
  cls.pm_stage_wmma_ab = pm_stage_wmma_ab
  cls.apply_tc_hand_opts = lambda self, tk, rngs: llvm_tc_hand_opts(tk, rngs)

# Re-export INDEX mops (tests / callers); definition lives in codegen.late.index_mops.
from tinygrad.codegen.late.index_mops import pm_index_mops, _index_through_reshape, _index_through_permute  # noqa: F401,E402

_WMMA_AB_WIDTH = 16
# Serialize A-tile batches so earlier LDS A packs die before later ones load (VGPR pressure).
# Product-8 OK with batch 2 + disjoint LLOAD/PACK pools. Product-16 still wrong on gfx1100
# (mse~450, spills) even with TC_LDS_A_BATCH=1 — AFTER does not constrain live ranges enough.

def expand_wmma_lds_tiles(u, a, b, c, done_arg, unroll_axis, ctx):
  # TC_LDS_AB pre-contracts to STACK(16*tile,); slice per-tile STACK(16,) here.
  # Serialize A-tile batches with AFTER so earlier LDS A packs die before later ones load.
  if a.op is not Ops.STACK or len(a.src) <= _WMMA_AB_WIDTH or len(a.src) % _WMMA_AB_WIDTH != 0: return None
  ta, tb = len(a.src) // _WMMA_AB_WIDTH, (len(b.src) // _WMMA_AB_WIDTH) if b.op is Ops.STACK else 1
  # Soft-fail past expand budget (8 default; 16 with ALLOW_UPCAST16 / register default).
  max_prod = 16 if _allow_upcast16() else 8
  if ta * tb > max_prod: return None
  a_batch = getenv("TC_LDS_A_BATCH", 2)
  c_stk = c if c.op is Ops.STACK else UOp.stack(*[c.index(UOp.const(i, dtypes.weakint)) for i in range(c.max_numel())])
  wmmas: list[UOp] = []
  prev_batch: UOp|None = None
  for i0 in range(0, ta, a_batch):
    batch: list[UOp] = []
    for i in range(i0, min(i0 + a_batch, ta)):
      aa_elems = a.src[i*_WMMA_AB_WIDTH:(i+1)*_WMMA_AB_WIDTH]
      if prev_batch is not None:
        aa_elems = tuple(UOp(Ops.AFTER, e.dtype, (e, prev_batch)) for e in aa_elems)
      aa = UOp.stack(*aa_elems)
      for j in range(tb):
        bb = UOp.stack(*b.src[j*_WMMA_AB_WIDTH:(j+1)*_WMMA_AB_WIDTH]) if b.op is Ops.STACK else b
        batch.append(u.replace(src=(aa, bb, c_stk), arg=done_arg))
    wmmas.extend(batch)
    prev_batch = UOp.stack(*batch)
  return unroll_axis(ctx, UOp.stack(*wmmas).reshape(ta, tb, c_stk.max_numel()), u.arg[4][2])

def isa_float4_coalesce(uops, ctx):
  from tinygrad.renderer.isa import ISARenderer
  from collections import defaultdict
  f4 = ctx.float4_dtypes if isinstance(ctx, ISARenderer) else None
  if f4 is None: return None, True, False, None
  if any(u.dtype.scalar() in (dtypes.bfloat16, *dtypes.fp8s) for u in uops): return f4, True, True, None
  uses = defaultdict(list)
  for u in uops:
    for su in u.src: uses[su].append(u)
  for u in uops:
    if u.op not in {Ops.LOAD, Ops.STORE} or is_image_shape(u.src[0].src[0]._shape) or u.src[0].op is not Ops.INDEX: continue
    buf = u.src[0].src[0]
    if buf.addrspace is AddrSpace.REG: continue  # weakfloat ACC stores must not disable global half B128
    value_dtype = u.src[1].dtype if u.op is Ops.STORE else u.dtype
    if buf.dtype.scalar() not in f4 or value_dtype.scalar() not in (*f4, dtypes.weakfloat): return f4, False, False, uses
  return f4, True, False, uses

def isa_float4_mem_ok(f4, float4_safe, buf, value_dtype, u, uses, valid) -> bool:
  if f4 is None: return True
  if not float4_safe or buf.dtype.scalar() != value_dtype.scalar(): return False
  if u.op is Ops.LOAD and any(v.op in {Ops.CAST, Ops.BITCAST} for v in uses[u]): return False
  if u.op is Ops.STORE and u.src[1].op is Ops.BITCAST: return False
  return bool(valid.op is Ops.CONST and valid.val is True)

def scratch_inst_size(inst) -> int:
  if type(inst).__name__ not in {"SCRATCH", "VSCRATCH"}: return 0
  name = getattr(inst, "op_name", "")
  if "_B128" in name: return 16
  if "_B96" in name: return 12
  if "_B64" in name: return 8
  if any(x in name for x in ("_B32", "_U32", "_I32")): return 4
  if any(x in name for x in ("_B16", "_U16", "_I16", "_D16")): return 2
  if any(x in name for x in ("_B8", "_U8", "_I8")): return 1
  return 0

def scan_elf_regs(insts, scratch_size_fn=scratch_inst_size):
  from tinygrad.renderer.amd.dsl import Reg, FixedBitField
  from tinygrad.runtime.autogen.amd.common import OpType
  max_vgpr, max_sgpr, max_accvgpr, private_segment_size = 0, 0, 0, 0
  _ACCVGPR_TYPES = {OpType.OPR_ACCVGPR, OpType.OPR_SRC_ACCVGPR}
  for inst in insts:
    if (scratch_size:=scratch_size_fn(inst)) != 0:
      private_segment_size = max(private_segment_size, getattr(inst, "ioffset", getattr(inst, "offset", 0)) + scratch_size)
    accvgpr_fields: set[str] = set()
    for opr_name, (_, _, opr_type) in inst.operands.items():
      if opr_type in _ACCVGPR_TYPES: accvgpr_fields.add(opr_name)
      elif opr_type in {OpType.OPR_VGPR_OR_ACCVGPR, OpType.OPR_SRC_VGPR_OR_ACCVGPR, OpType.OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST}:
        if getattr(inst, 'acc_cd', 0) == 1: accvgpr_fields.add(opr_name)
    for name, field in inst._fields:
      if isinstance(field, FixedBitField): continue
      val = getattr(inst, name)
      if not isinstance(val, Reg): continue
      if 256 <= val.offset < 512:
        if name in accvgpr_fields: max_accvgpr = max(max_accvgpr, (val.offset - 256) + val.sz)
        else: max_vgpr = max(max_vgpr, (val.offset - 256) + val.sz)
      elif val.offset < 106: max_sgpr = max(max_sgpr, val.offset + val.sz)
  return max_vgpr, max_sgpr, max_accvgpr, private_segment_size

def scan_elf_meta(prg:UOp, private_segment_size:int=0):
  sink, n_bufs, n_vars, kernarg_size, param_sizes, lds_size, gids, lids = prg.src[0], 0, 0, 0, {}, 0, set(), set()
  for u in sink.toposort():
    if u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU:
      n_vars += 1
      param_sizes[u.arg.slot] = u.dtype.itemsize
    elif u.op is Ops.PARAM:
      n_bufs += 1
      param_sizes[u.arg.slot] = 8
    elif u.op is Ops.INS and getattr(u.arg, "name", None) == "KERNARG":
      kernarg_size = max(kernarg_size, u.src[0].arg + u.dtype.itemsize)
    elif u.op is Ops.INS and getattr(u.arg, "name", None) == "SCRATCH_SIZE":
      private_segment_size = max(private_segment_size, u.src[0].arg)
    elif u.op is Ops.BUFFER and u.addrspace is AddrSpace.LOCAL:
      lds_size += u.max_numel() * u.dtype.itemsize
    elif u.op is Ops.INS and getattr(u.arg, "name", None) == "LDS_BASE":
      lds_size = max(lds_size, (u.src[1].arg if len(u.src) > 1 else lds_size) + u.src[0].arg)
    elif u.op is Ops.SPECIAL and u.arg.startswith("gidx"): gids.add(int(u.arg[-1]))
    elif u.op is Ops.SPECIAL and u.arg.startswith("lidx") and u.vmax > 0: lids.add(int(u.arg[-1]))
  if len(prg.src) > 1 and prg.src[1].op is Ops.LINEAR:
    for u in prg.src[1].src:
      if u.op is Ops.INS and getattr(u.arg, "name", None) == "SCRATCH_SIZE":
        private_segment_size = max(private_segment_size, u.src[0].arg)
  return n_bufs, n_vars, kernarg_size, param_sizes, lds_size, gids, lids, private_segment_size

class AMDRenderer(ISARenderer):
  device = "AMD"
  has_local = True
  has_shared = True
  supports_float4 = True
  float4_dtypes = (dtypes.float32, dtypes.half)
  wide_regalloc = True
  preferred_reduce_group = 16
  global_max = (0x8fffffff, 0x8fffffff, 0x8fffffff)
  # 2D locals (WARP=lidx0 → e.g. (32,4,1)); needs gfx1100 USER_SGPR=15 (elf.py).
  local_max = (1024, 1024, 64)
  local_prod_max = 1024
  shared_max = 65536
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pm_stage_wmma_ab = pm_stage_wmma_ab
  _code_ops = (Ops.ADD, Ops.SUB, Ops.MUL, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN, Ops.MAX,
               Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ)
  code_for_op = {op: (lambda: None) for op in _code_ops}

  def __init__(self, target:Target):
    if not target.arch.startswith("gfx11"): raise RuntimeError(f"AMDRenderer is RDNA3/gfx11 only, got {target.arch}")
    super().__init__(target)
    self.tensor_cores = [x for x in tc.get_amd(target.arch) if (x.dtype_in, x.dtype_out) == (dtypes.half, dtypes.float)]

  def apply_tc_hand_opts(self, tk, rngs):
    apply_tc_hand_opts(tk, rngs)

  def coalesce_gate(self, uops): return isa_float4_coalesce(uops, self)
  def coalesce_mem_ok(self, f4, float4_safe, buf, value_dtype, u, uses, valid):
    return isa_float4_mem_ok(f4, float4_safe, buf, value_dtype, u, uses, valid)
  def coalesce_vec_lengths(self, buf, f4):
    # 16 halves = one WMMA A frag (2×B128); PACK identity-aliases the load.
    if buf.dtype == dtypes.half: return [16, 8, 4, 2]
    return None

  def prepare_pre_regalloc(self, lst:list[UOp]) -> tuple[list[UOp], dict]:
    inits, tiles, idx_map = _wmma_acc_zero_inits(lst)
    if not inits: return lst, {}
    loop_i = next((i for i,u in enumerate(lst) if u.op is Ops.RANGE), 0)
    return lst[:loop_i] + inits + lst[loop_i:], {
      "wmma_acc_inits": {id(u.tag): u for u in inits},
      "wmma_acc_tiles": tiles,
      "wmma_acc_idx_map": idx_map,
    }

  def is_two_address(self, x:UOp) -> bool:
    if x.op is not Ops.INS: return False
    if x.arg is AMDOps.WMMA: return True
    # PACK_F16(half×16 LOAD) — coalesce onto the load (hand FA/FB).
    return x.arg is AMDOps.PACK_F16 and _pack_f16_is_vec_load(x) and len(x.src) == 1
  def prefer_phys(self, x:UOp, src_phys:list) -> Register|None:
    # Float EXTRACT from a multi-VGPR pack/WMMA → alias onto pack+lane (kill C-store movs).
    if x.op is not Ops.INS or x.arg is not AMDOps.EXTRACT or x.dtype.scalar() is not dtypes.float32: return None
    if not src_phys or src_phys[0] is None or not isinstance(x.tag, tuple): return None
    if (lane := _lane_const(x.src[1])) is None: return None
    want = src_phys[0].index + int(lane)
    return next((r for r in x.tag[0].cons if r.index == want), None)

  def after_pre_regalloc(self, lst:list[UOp]) -> list[UOp]:
    """Schedule each f32→f16 CAST immediately before its STORE.

    Product-16 epilogue otherwise keeps 128 half temps live at once; regalloc spills them into
    live WMMA ACC (v126+) and clobbers unread lanes (half rows 62–63).
    """
    uses: dict[UOp, list[UOp]] = {}
    for u in lst:
      for src in u.src: uses.setdefault(src, []).append(u)
    store_cast: dict[UOp, UOp] = {}  # store -> cast
    for u in lst:
      if u.op is not Ops.INS or u.arg is not AMDOps.CAST: continue
      if u.dtype.scalar() is not dtypes.float16 or not u.src or u.src[0].dtype.scalar() is not dtypes.float32: continue
      us = uses.get(u, [])
      if len(us) == 1 and us[0].op is Ops.INS and us[0].arg is AMDOps.STORE and len(us[0].src) > 2 and us[0].src[2] is u:
        store_cast[us[0]] = u
    if not store_cast: return lst
    skip = set(store_cast.values())
    out: list[UOp] = []
    for u in lst:
      if u in skip: continue
      if u in store_cast: out.append(store_cast[u])
      out.append(u)
    return out
  def _pure_addr(self, x:UOp) -> bool:
    if x.op in (Ops.CONST, Ops.SPECIAL): return True
    if x.op is not Ops.INS or x.dtype.scalar() not in (dtypes.int32, dtypes.uint32): return False
    if x.arg is AMDOps.MOV and x.src: return self._pure_addr(x.src[0])
    if x.arg in (AMDOps.ADD, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR):
      return all(self._pure_addr(s) for s in x.src)
    return False
  def rematerialize(self, x:UOp) -> bool:
    if x.op is not Ops.INS: return False
    # Under TC_LDS_AB: remat LDS half EXTRACTs (and LLOAD bases if ALLOW_UPCAST16).
    # Address remat defaults ON under LDS — EXTRACT-only leaves addr spills and breaks mock
    # (and is no faster on gfx1100). AMD_REMAT_ADDR=0 opts out for experiments.
    if getenv("TC_LDS_AB", 0) and getenv("AMD_REMAT", 1):
      if (x.arg is AMDOps.EXTRACT and x.dtype.scalar() is dtypes.half and x.src and
          x.src[0].op is Ops.INS and x.src[0].arg is AMDOps.LLOAD):
        return True
      if getenv("ALLOW_UPCAST16", 0) and x.arg is AMDOps.LLOAD and x.dtype.scalar() is dtypes.half:
        return True
    if not getenv("AMD_REMAT_ADDR", 1 if getenv("TC_LDS_AB", 0) else 0): return False
    if x.dtype.scalar() not in (dtypes.int32, dtypes.uint32): return False
    return x.arg is not AMDOps.MOV and self._pure_addr(x)
  def keep_remat(self, x:UOp) -> bool:
    # Pure-addr remats under TC_LDS: without sticky, SHR/AND remat ~60× and SHL/ADD flood the loop.
    return x.op is Ops.INS and x.arg in (AMDOps.SHR, AMDOps.AND, AMDOps.SHL, AMDOps.ADD)
  def remat(self, x:UOp, reg:Register, src_regs:list[Register|None]) -> UOp:
    nsrc = [s if r is None else UOp(Ops.INS, s.dtype, (), AMDOps.MOV, (r,)) for s, r in zip(x.src, src_regs)]
    return x.replace(src=tuple(nsrc), tag=(reg,))
  def bind(self, dtype, reg:Register) -> UOp: return UOp(Ops.INS, dtype, (), AMDOps.MOV, (reg,))
  def stack_pointer(self) -> UOp: return UOp(Ops.INS, dtypes.uint32, arg=AMDOps.SCRATCH_BASE)
  def register_slots(self, x:UOp, vreg:Register|None=None) -> int:
    if vreg is None or not all(c.index >= 256 for c in vreg.cons): return 1
    return _reg_slots(x)
  def copy(self, x:UOp, reg): return UOp(Ops.INS, x.dtype, (x,), AMDOps.MOV, (reg,))
  def spill(self, disp:UOp, x:UOp) -> UOp:
    if greg(x).index < 256: raise CompileError("no sgpr spill")
    return UOp(Ops.INS, dtypes.void, (disp, x), AMDOps.SPILL)
  def fill(self, disp:UOp, x:UOp, reg) -> UOp:
    if reg.index < 256: raise CompileError("no sgpr fill")
    return UOp(Ops.INS, x.dtype, (disp, UOp.const(_reg_slots(x), dtypes.int32).rtag()), AMDOps.FILL, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    ret = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is AMDOps.LABEL: ret.append(f"{u.tag}:")
      elif u.arg in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1): ret.append(f"  {u.arg.name.lower()} {u.tag}")
      else: ret.append(f"  {u.arg.name.lower()} " + ", ".join(str(greg(s) or s.arg) for s in u.src))
    return "\n".join(ret)

  def render(self, uops:list[UOp]) -> str: return self.asm_str(uops, "kernel")
  def _insts_for_uop(self, u:UOp): return insts_for_uop(u)
  def _insts_from_linear(self, lin:UOp): return insts_from_linear(lin)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    insts = self._insts_from_linear(lin)
    insts.append(r3.s_endpgm())
    nlin = lin.replace(src=tuple(UOp(Ops.INS, arg=i) for i in insts))
    return assemble_linear(prg, nlin, self.target.arch)

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.float16, dtypes.float32}

# ***** wide (AMD) linear-scan helpers *****
def wide_alloc(cons, i, nslots, allowed, live, lr, uops_len, slots_fn, pinned=frozenset()):
  allowed_idxs = {r.index for r in allowed}
  live_span = {vr: set(range(r.index, r.index + slots_fn(vr))) for vr, r in live.items()}
  def blockers(reg):
    occupied = set(range(reg.index, reg.index+nslots))
    return tuple(vr for vr, sp in live_span.items() if occupied & sp)
  def next_use(vr): return next((j-i for j in lr[vr] if j >= i), uops_len)
  candidates = [(r, blockers(r)) for r in cons
                if all(x in allowed_idxs for x in range(r.index, r.index+nslots))
                and not (set(range(r.index, r.index+nslots)) & pinned)]
  if not candidates:  # no unpinned window (wide nslots / fill) — steal via blockers
    candidates = [(r, blockers(r)) for r in cons if all(x in allowed_idxs for x in range(r.index, r.index+nslots))]
  if not candidates: raise CompileError(f"wide_alloc: no free regs ({nslots} slots)")
  reg,vregs = max(candidates, key=lambda rv: min((next_use(vv) for vv in rv[1]), default=uops_len))
  for vr in vregs: live.pop(vr, None)
  return reg

def wide_restore(ctx, v, r, i):
  if v in ctx.remat:
    vd, src_regs = ctx.vdef(v), []  # type: ignore[var-annotated]
    for su in vd.src:
      if su.op is Ops.CONST or not isinstance(sv:=greg(su), Register): src_regs.append(None)
      else: src_regs.append(ctx.reals[i][sv])
    return ctx.ren.remat(vd, r, src_regs)
  return ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r)

def wide_regalloc_rewrite(ctx, x:UOp):
  i = ctx.regalloc_i
  if x.op in {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}: return None
  if x.op in (Ops.LOAD, Ops.STORE) and not ctx.insert_before.get(i):
    spilled = any(i in ctx.reals and ((vr:=greg(ctx.uops[i].src[j])) in ctx.spills or vr in ctx.remat)
                  for j in range(len(x.src)))
    if not spilled and i not in (ctx.first_real_idx, ctx.last_real_idx): return None
  nsrc = []
  for j,su in enumerate(x.src):
    vr = greg(ctx.uops[i].src[j]) if i in ctx.reals else None
    if isinstance(vr, Register) and vr in ctx.remat:
      # Actual remat runs in `before` via insert_before; source is just a bind to that phys.
      nsrc.append(ctx.ren.bind(ctx.vdef(vr).dtype, ctx.reals[i][vr]))
    elif isinstance(vr, Register) and vr in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[vr], ctx.vdef(vr), ctx.reals[i][vr]))
    else: nsrc.append(su)
  ndefs = tuple(ctx.reals[i][vr] for vr in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.BUFFER: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)
  before = [wide_restore(ctx, vr, r, i) for vr,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[vr], nx) for vr in x.tag if vr in ctx.spills] if isinstance(x.tag, tuple) else []
  if ctx.stack_size > 0:
    sp, offset = ctx.ren.stack_pointer(), UOp(Ops.CONST, ctx.ren.stack_pointer().dtype, arg=ctx.stack_size)
    if i == ctx.first_real_idx: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, src=(sp, offset), tag=sp.tag))] + before
    elif i == ctx.last_real_idx: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, src=(sp, offset), tag=sp.tag))]
  return nx, before + [nx] + after

# ***** public install (was thin isa/amd.py) *****
def _install_hooks():
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  import tinygrad.codegen as cg
  cg.expand_wmma_lds_hook = expand_wmma_lds_tiles
  install_amdllvm_tc(AMDLLVMRenderer)
_install_hooks()

# Gabriel/x86-style aliases
RDNA3Renderer = AMDRenderer
RDNA3Ops = AMDOps

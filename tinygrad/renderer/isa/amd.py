from __future__ import annotations
import struct

from tinygrad.device import CompileError
from tinygrad.dtype import dtypes, DType, AddrSpace, PtrDType
from tinygrad.helpers import Target
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext, Register
from tinygrad.renderer.amd.dsl import Reg, s, v, NULL, EXEC, VCC
from tinygrad.runtime.autogen.amd.rdna3 import ins as r3
from tinygrad.codegen.decomp.op import fast_idiv
from tinygrad.uop import Ops, FastEnum, auto
from tinygrad.uop.ops import PatternMatcher, UOp, UPat

# RDNA3 starts every wave with the kernarg segment pointer in s[0:1]. With
# ENABLE_SGPR_WORKGROUP_ID_X/Y/Z, workgroup ids follow in s[2:4]. v0-v2 are local ids x/y/z.
KERNARG_REG = s[0:1]
WGID = tuple(Register(f"s{i}", i) for i in range(2, 5))
LID = tuple(Register(f"v{i}", 256+i) for i in range(3))
# Allocate scalar temporaries on even bases so 64-bit kernarg pointer loads don't overlap.
# The odd SGPRs are intentionally left as the high half of potential 64-bit pairs.
SGPR = tuple(Register(f"s{i}", i) for i in range(6, 104, 2))
VGPR = tuple(Register(f"v{i}", 256+i) for i in range(3, 254))
# v254/v255 are reserved for synthetic store-data/address materialization and scratch spills.
# They are per-instruction scratch temps only; nothing may expect them to stay live across linear IR uops.
TMP_VDATA = v[254]
TMP_VADDR = v[255]
# s104:105 is outside the allocator pool and serves two non-overlapping purposes:
#  - TMP_EXEC: cleanup gated stores save EXEC across exactly one store (s_and_saveexec -> store -> s_mov EXEC).
#  - TMP_SDATA0/1: single-instruction scalar scratch for uniform VGPR values before a SALU compare.
# These never alias in time: CMP_GE's readfirstlane+compare is a tight produce/consume at loop boundaries and
# never lands between an IF_MASK save and its END_MASK restore (the masked region only ever contains the store).
TMP_EXEC = s[104:105]
TMP_SDATA0, TMP_SDATA1 = s[104], s[105]

class AMDOps(FastEnum):
  LABEL = auto()
  DEFINE = auto()
  SCRATCH_BASE = auto()
  SCRATCH_SIZE = auto()
  SCRATCH_ADDR = auto()
  KERNARG = auto()
  MOV = auto()
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

_F32_UNARY = {AMDOps.RECIPROCAL: r3.v_rcp_f32_e32, AMDOps.EXP2: r3.v_exp_f32_e32, AMDOps.LOG2: r3.v_log_f32_e32,
              AMDOps.SQRT: r3.v_sqrt_f32_e32, AMDOps.TRUNC: r3.v_trunc_f32_e32}

def _is_float(dt:DType) -> bool: return dt.scalar() in (dtypes.float16, dtypes.float32)
def _reg_to_amd(reg:Register, sz:int=1) -> Reg:
  if reg.index >= 256:
    idx = reg.index - 256
    return v[idx] if sz == 1 else v[idx:idx+sz-1]
  return s[reg.index] if sz == 1 else s[reg.index:reg.index+sz-1]

def _src(x:UOp):
  if x.op is Ops.AFTER: return _src(x.src[0])
  if x.op is Ops.CONST:
    if x.dtype.scalar() is dtypes.float32: return float(x.arg)
    if x.dtype.scalar() is dtypes.float16: return struct.unpack("H", struct.pack("e", float(x.arg)))[0]
    return int(x.arg)
  if not isinstance(x.reg, Register): raise CompileError(f"AMD renderer expected register source for {x}")
  return _reg_to_amd(x.reg, 2 if x.dtype.itemsize == 8 else 1)

def _dst(x:UOp) -> Reg:
  if not isinstance(x.reg, Register): raise CompileError(f"AMD renderer expected destination register for {x}")
  return _reg_to_amd(x.reg, 2 if x.dtype.itemsize == 8 else 1)

def _reg_idxs(x:UOp) -> set[int]:
  if x.op is Ops.AFTER: return set().union(*(_reg_idxs(s) for s in x.src))
  if not isinstance(x.reg, Register): return set()
  return set(range(x.reg.index, x.reg.index + max(1, (x.dtype.itemsize + 3) // 4)))

def _wait_for_domain(domain:str):
  return r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0) if domain == "vm" else r3.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)

def _wait_domain_for_load(u:UOp) -> str|None:
  if u.op is not Ops.INS: return None
  if u.arg in (AMDOps.LOAD, AMDOps.SLOAD, AMDOps.FILL): return "vm"
  if u.arg in (AMDOps.KERNARG, AMDOps.LLOAD): return "lgkm"
  return None

def _global_load(dt:DType):
  return {
    dtypes.bool: r3.global_load_u8, dtypes.uint8: r3.global_load_u8, dtypes.int8: r3.global_load_i8,
    dtypes.uint16: r3.global_load_u16, dtypes.int16: r3.global_load_i16, dtypes.float16: r3.global_load_u16,
    dtypes.uint32: r3.global_load_b32, dtypes.int32: r3.global_load_b32, dtypes.float32: r3.global_load_b32,
  }.get(dt.scalar())

def _global_store(dt:DType):
  return {
    dtypes.bool: r3.global_store_b8, dtypes.uint8: r3.global_store_b8, dtypes.int8: r3.global_store_b8,
    dtypes.uint16: r3.global_store_b16, dtypes.int16: r3.global_store_b16, dtypes.float16: r3.global_store_b16,
    dtypes.uint32: r3.global_store_b32, dtypes.int32: r3.global_store_b32, dtypes.float32: r3.global_store_b32,
  }.get(dt.scalar())

def _scratch_load(dt:DType):
  return {
    dtypes.bool: r3.scratch_load_u8, dtypes.uint8: r3.scratch_load_u8, dtypes.int8: r3.scratch_load_i8,
    dtypes.uint16: r3.scratch_load_u16, dtypes.int16: r3.scratch_load_i16, dtypes.float16: r3.scratch_load_u16,
    dtypes.uint32: r3.scratch_load_b32, dtypes.int32: r3.scratch_load_b32, dtypes.float32: r3.scratch_load_b32,
  }.get(dt.scalar())

def _scratch_store(dt:DType):
  return {
    dtypes.bool: r3.scratch_store_b8, dtypes.uint8: r3.scratch_store_b8, dtypes.int8: r3.scratch_store_b8,
    dtypes.uint16: r3.scratch_store_b16, dtypes.int16: r3.scratch_store_b16, dtypes.float16: r3.scratch_store_b16,
    dtypes.uint32: r3.scratch_store_b32, dtypes.int32: r3.scratch_store_b32, dtypes.float32: r3.scratch_store_b32,
  }.get(dt.scalar())

def _local_load(dt:DType):
  return {
    dtypes.bool: r3.ds_load_u8, dtypes.uint8: r3.ds_load_u8, dtypes.int8: r3.ds_load_i8,
    dtypes.uint16: r3.ds_load_u16, dtypes.int16: r3.ds_load_i16, dtypes.float16: r3.ds_load_u16,
    dtypes.uint32: r3.ds_load_b32, dtypes.int32: r3.ds_load_b32, dtypes.float32: r3.ds_load_b32,
  }.get(dt.scalar())

def _local_store(dt:DType):
  return {
    dtypes.bool: r3.ds_store_b8, dtypes.uint8: r3.ds_store_b8, dtypes.int8: r3.ds_store_b8,
    dtypes.uint16: r3.ds_store_b16, dtypes.int16: r3.ds_store_b16, dtypes.float16: r3.ds_store_b16,
    dtypes.uint32: r3.ds_store_b32, dtypes.int32: r3.ds_store_b32, dtypes.float32: r3.ds_store_b32,
  }.get(dt.scalar())

def _scaled_addr(dst:Reg, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  src = _src(idx)
  if itemsize == 1:
    return ([], src) if isinstance(src, Reg) and src.offset >= 256 else ([r3.v_mov_b32_e32(dst, src)], dst)
  if itemsize not in (2, 4): raise CompileError(f"AMDRenderer does not support address scale for itemsize {itemsize}")
  return [r3.v_lshlrev_b32_e64(dst, itemsize.bit_length()-1, src)], dst

def _vgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [], src
  return [r3.v_mov_b32_e32(tmp, src)], tmp

def _sgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [r3.v_readfirstlane_b32_e32(tmp, src)], tmp
  return [], src

def _is_lds_ref(x:UOp) -> bool:
  if x.op is Ops.AFTER: return _is_lds_ref(x.src[0])
  return (x.op is Ops.BUFFER and x.addrspace is AddrSpace.LOCAL) or (x.op is Ops.INS and x.arg is AMDOps.LDS_BASE)

def _is_scratch_ref(x:UOp) -> bool:
  if x.op is Ops.AFTER: return _is_scratch_ref(x.src[0])
  return (x.op is Ops.BUFFER and x.addrspace is AddrSpace.REG) or (x.op is Ops.INS and x.arg is AMDOps.SCRATCH_ADDR)

def _lds_itemsize(x:UOp) -> int:
  return x.ptrdtype.base.itemsize if isinstance(x.dtype, PtrDType) else x.dtype.itemsize

def _lds_size_bytes(x:UOp) -> int:
  return x.ptrdtype.size * x.ptrdtype.base.itemsize if isinstance(x.dtype, PtrDType) else x.max_numel() * x.dtype.itemsize

def _align(x:int, a:int) -> int: return x + (-x % a)

def _lds_offsets(ctx:IselContext) -> dict[int, int]:
  if not hasattr(ctx, "amd_lds_offsets"):
    offsets, slots, off = {}, set(), 0
    for b in sorted([u for u in ctx.uses if u.op is Ops.BUFFER and u.addrspace is AddrSpace.LOCAL], key=lambda u: u.arg.slot):
      if b.arg.slot in slots: raise CompileError(f"AMDRenderer got duplicate LDS buffer slot {b.arg.slot}")
      slots.add(b.arg.slot)
      off = _align(off, _lds_itemsize(b))
      offsets[b.arg.slot] = off
      off += _lds_size_bytes(b)
    ctx.amd_lds_offsets = offsets
  return ctx.amd_lds_offsets

def _lds_base(ctx:IselContext, x:UOp) -> UOp|None:
  if x.addrspace is not AddrSpace.LOCAL: return None
  return UOp(Ops.INS, dtypes.uint32,
             (UOp.const(dtypes.int32, _lds_size_bytes(x)).rtag(), UOp.const(dtypes.int32, _lds_offsets(ctx)[x.arg.slot]).rtag()),
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

def _const_int(x:UOp) -> int|None:
  if x.op is Ops.CONST: return int(x.arg)
  if x.op is Ops.INS and x.arg is AMDOps.MOV and x.src and x.src[0].op is Ops.CONST: return int(x.src[0].arg)
  return None

def _reg_promotable_buffers(ctx:PreRegAllocContext) -> set[UOp]:
  if hasattr(ctx, "amd_reg_promotable"): return ctx.amd_reg_promotable
  bases, bad, seen_store = set(), set(), set()
  for u in ctx.uops or []:
    if u.op is not Ops.INS or u.arg not in (AMDOps.SLOAD, AMDOps.SSTORE): continue
    if (base:=_reg_buffer_base(u.src[0])) is None: continue
    bases.add(base)
    idx = _const_int(u.src[1])
    dt = u.dtype if u.arg is AMDOps.SLOAD else u.src[2].dtype
    if idx is None or idx < 0 or idx >= base.max_numel() or base.max_numel() > 64 or dt.count != 1 or dt.itemsize > 4:
      bad.add(base)
      continue
    key = (base, idx)
    if u.arg is AMDOps.SLOAD and key not in seen_store: bad.add(base)
    if u.arg is AMDOps.SSTORE: seen_store.add(key)
  ctx.amd_reg_promotable = bases - bad
  ctx.amd_reg_values = {}
  ctx.amd_reg_n = 0
  return ctx.amd_reg_promotable

def _reg_promote_slot(ctx:PreRegAllocContext, base:UOp, idx:UOp) -> tuple[UOp, int]|None:
  base = _reg_buffer_base(base)
  if base is None or base not in _reg_promotable_buffers(ctx): return None
  return None if (slot:=_const_int(idx)) is None else (base, slot)

def _new_promoted_reg(ctx:PreRegAllocContext, val:UOp) -> UOp:
  n = ctx.amd_reg_n
  ctx.amd_reg_n += 1
  return UOp(Ops.INS, val.dtype, (val,), AMDOps.MOV, (Register(f"reg{n}", 0, _cons=VGPR),))

def _load_ins(x:UOp, a:UOp) -> UOp:
  if _is_lds_ref(a.src[0]):
    if _local_load(x.dtype) is None: raise CompileError(f"AMDRenderer does not support LDS loads for {x.dtype}")
    return x.ins(AMDOps.LLOAD, src=(a.src[0], a.src[1]))
  if _is_scratch_ref(a.src[0]):
    if _scratch_load(x.dtype) is None: raise CompileError(f"AMDRenderer does not support scratch loads for {x.dtype}")
    return x.ins(AMDOps.SLOAD, src=(a.src[0], a.src[1]))
  if _global_load(x.dtype) is None: raise CompileError(f"AMDRenderer does not support global loads for {x.dtype}")
  return x.ins(AMDOps.LOAD, src=(a.src[0], a.src[1]))

def _store_ins(x:UOp, a:UOp, val:UOp) -> UOp:
  if _is_lds_ref(a.src[0]):
    if _local_store(val.dtype) is None: raise CompileError(f"AMDRenderer does not support LDS stores for {val.dtype}")
    return x.ins(AMDOps.LSTORE, src=(a.src[0], a.src[1], val))
  if _is_scratch_ref(a.src[0]):
    if _scratch_store(val.dtype) is None: raise CompileError(f"AMDRenderer does not support scratch stores for {val.dtype}")
    return x.ins(AMDOps.SSTORE, src=(a.src[0], a.src[1], val))
  if _global_store(val.dtype) is None: raise CompileError(f"AMDRenderer does not support global stores for {val.dtype}")
  return x.ins(AMDOps.STORE, src=(a.src[0], a.src[1], val))

AMD_ATOMIC_ADD = "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
def _atomic_add_ins(x:UOp) -> UOp|None:
  if x.arg != AMD_ATOMIC_ADD: return None
  if len(x.src) != 2 or x.src[0].op is not Ops.INDEX: raise CompileError(f"AMDRenderer cannot lower custom atomic {x}")
  a, val = x.src
  if val.dtype.scalar() is not dtypes.float32: raise CompileError(f"AMDRenderer only supports f32 atomic add, got {val.dtype}")
  if _is_lds_ref(a.src[0]) or _is_scratch_ref(a.src[0]): raise CompileError("AMDRenderer only supports global atomic add")
  return x.ins(AMDOps.ATOMIC_ADD, src=(a.src[0], a.src[1], val))

def _special_reg(name:str) -> Register:
  if len(name) != 5 or name[:4] not in ("lidx", "gidx") or name[-1] not in "012":
    raise CompileError(f"AMD renderer only supports lidx0/1/2 and gidx0/1/2 SPECIAL now, got {name}")
  return LID[int(name[-1])] if name[0] == "l" else WGID[int(name[-1])]

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
  if x.arg in (AMDOps.DEFINE, AMDOps.SCRATCH_SIZE, AMDOps.SCRATCH_ADDR, AMDOps.LDS_BASE, AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) or x.dtype is dtypes.void:
    return x.replace(tag=None) if x.arg in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) and x.tag is not None else None
  if x.arg is AMDOps.KERNARG: return x.replace(tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.PARAM:
    if x.arg.addrspace is AddrSpace.ALU: return x.replace(src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
    return x.replace(dtype=dtypes.uint64, src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.SPECIAL:
    return x.replace(tag=(ctx.vreg(_special_reg(x.arg)),))
  return x.replace(tag=(ctx.vreg(vgpr_pool),))

def _gated_load(addr:UOp, alt:UOp, gate:UOp, x:UOp) -> UOp|None:
  if addr.op is not Ops.INDEX or len(addr.src) != 2: return None
  safe_addr = addr.replace(src=(addr.src[0], gate.where(addr.src[1], addr.src[1].const_like(0))))
  return gate.where(safe_addr.load(dtype=x.dtype), alt.cast(x.dtype) if alt.dtype != x.dtype else alt)

def _pow2_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.arg <= 0 or c.arg & (c.arg - 1) or (x.dtype not in dtypes.uints and x.vmin < 0): return None
  return x & UOp.const(x.dtype, c.arg - 1)

class _AMDFastDivRenderer:
  def supported_dtypes(self): return {dtypes.int32, dtypes.uint32}

def _const_cdiv(x:UOp, c:UOp) -> UOp|None:
  return fast_idiv(_AMDFastDivRenderer(), x, c.arg) if c.arg > 0 and x.vmin >= 0 else None

def _const_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.arg <= 0 or x.vmin < 0: return None
  if (q:=_const_cdiv(x, c)) is None: return None
  return x - q * UOp.const(x.dtype, c.arg)

def _bool_not(x:UOp) -> UOp:
  return x.where(UOp.const(dtypes.bool, False), UOp.const(dtypes.bool, True))

def _u32_divmod(n:UOp, d:UOp) -> tuple[UOp, UOp]:
  zero, one = UOp.const(dtypes.uint32, 0), UOp.const(dtypes.uint32, 1)
  q, r = zero, zero
  for i in range(31, -1, -1):
    r = (r << one.const_like(1)) | ((n >> UOp.const(dtypes.uint32, i)) & one)
    ge = _bool_not(r < d)
    q = q | ge.where(one << UOp.const(dtypes.uint32, i), zero)
    r = ge.where(r - d, r)
  return q, r

def _var_divmod(x:UOp, d:UOp, op:UOp) -> UOp|None:
  if x.dtype != d.dtype or x.dtype.scalar() not in (dtypes.int32, dtypes.uint32): return None
  if x.dtype.scalar() is dtypes.uint32:
    q, r = _u32_divmod(x, d)
    return q if op.op is Ops.CDIV else r
  zero = UOp.const(dtypes.int32, 0)
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
  return m if keep else m.where(UOp.const(dtypes.bool, False), UOp.const(dtypes.bool, True))

def _materialize_compare_flags(x:UOp) -> UOp|None:
  src, changed = [], False
  for s in x.src:
    if s.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ):
      src.append(s.where(UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, False)))
      changed = True
    else: src.append(s)
  return x.replace(src=tuple(src)) if changed else None

def _materialize_store_compare_flag(x:UOp) -> UOp|None:
  if len(x.src) < 2 or x.src[1].op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): return None
  return x.replace(src=(x.src[0], x.src[1].where(UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, False)), *x.src[2:]))

def _materialize_where_value_flags(x:UOp) -> UOp|None:
  src, changed = list(x.src), False
  for i in (1, 2):
    if src[i].op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ):
      src[i] = src[i].where(UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, False))
      changed = True
  return x.replace(src=tuple(src)) if changed else None

def _materialize_bool_where(m:UOp, a:UOp, b:UOp) -> UOp|None:
  if m.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): return None
  return UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPNE, dtypes.bool, (m, UOp.const(dtypes.bool, False))), a, b))

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
   lambda m,x: m.where(UOp.const(x.dtype, 1), UOp.const(x.dtype, 0))),
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")), _materialize_bool_where),
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"), _int_cast),
  # RDNA has no direct int<->f16 convert; route through f32 so the int<->f32 and f16<->f32 isel legs handle it.
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
     x.replace(dtype=dtypes.uint32, src=(UOp.const(dtypes.uint32, c.arg).rtag(),) + x.src[1:])),
    (UPat(Ops.RANGE, name="x"), lambda ctx,x,sgpr_pool=sgpr_pool:
     x.replace(dtype=dtypes.uint32, tag=(ctx.vreg(sgpr_pool),)) if not isinstance(x.tag, tuple) else None),
    (UPat(Ops.PARAM, name="x"), lambda ctx,x:
     UOp(Ops.INS, dtypes.uint64 if x.arg.addrspace is not AddrSpace.ALU else dtypes.uint32,
         (UOp.const(dtypes.int32, _kernarg_offset(ctx, x)).rtag(),), AMDOps.KERNARG, None)
     if not isinstance(x.tag, tuple) else None),
    (UPat(Ops.BUFFER, name="x"), lambda ctx,x: _lds_base(ctx, x)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x:
     UOp(Ops.INS, dtypes.uint32, (x.rtag(),), AMDOps.MOV, (ctx.vreg(_special_reg(x.arg)),)) if x.tag is None else None),
    (UPat.cvar("x", dtypes.ints+(dtypes.bool, dtypes.float16, dtypes.float32)), lambda x:
     x.ins(AMDOps.MOV, src=(x.rtag(),)) if not x.tag else None),
    ((UPat(Ops.MUL, (dtypes.float16, dtypes.float32), name="a") + UPat.var("b")).named("c"), _fused_mulacc),
    ((UPat(dtype=dtypes.ints+(dtypes.bool, dtypes.float16, dtypes.float32)) + UPat()).named("x"), lambda x: x.ins(AMDOps.ADD)),
    (UPat(Ops.SUB, dtype=dtypes.ints+(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.SUB)),
    ((UPat(dtype=dtypes.ints+(dtypes.float16, dtypes.float32)) * UPat()).named("x"), lambda x: x.ins(AMDOps.MUL)),
    (UPat(Ops.MULACC, dtype=(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.MULACC)),
    (UPat.var("y", dtypes.float16).cast(dtypes.float32, name="x"), lambda y,x: x.ins(AMDOps.CAST, src=(y,))),
    (UPat.var("y", dtypes.float32).cast(dtypes.float16, name="x"), lambda y,x: x.ins(AMDOps.CAST, src=(y,))),
    (UPat.var("y", dtypes.ints).cast(dtypes.float32, name="x"), lambda y,x: x.ins(AMDOps.CAST, src=(y,))),
    (UPat.var("y", dtypes.float32).cast(dtypes.ints, name="x"), lambda y,x: x.ins(AMDOps.CAST, src=(y,))),
    (UPat(Ops.RECIPROCAL, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.RECIPROCAL)),
    (UPat(Ops.EXP2, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.EXP2)),
    (UPat(Ops.LOG2, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.LOG2)),
    (UPat(Ops.SQRT, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.SQRT)),
    (UPat(Ops.TRUNC, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.TRUNC)),
    (UPat(Ops.SIN, dtype=dtypes.float32, name="x"), lambda x: x.ins(AMDOps.SIN)),
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
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, name="a"),), name="x"), _load_ins),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="a"), UPat.var("val")), name="x"), _store_ins),
    (UPat(Ops.CUSTOM, name="x"), _atomic_add_ins),
    (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(AMDOps.BARRIER)),
    (UPat((Ops.INS, Ops.BUFFER), name="x"), lambda ctx,x,sgpr_pool=sgpr_pool,vgpr_pool=vgpr_pool: _alloc_vregs(ctx, x, sgpr_pool, vgpr_pool)),
  ])

isel_matcher = make_isel_matcher()

def _loop_label(x:UOp) -> str: return "_".join(str(i) for i in x.arg[:-1])

def _lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = _loop_label(x)
  acc = x.ins(AMDOps.MOV, dtype=dtypes.uint32, src=(UOp.const(dtypes.uint32, 0).rtag(),))
  label = UOp(Ops.INS, dtypes.void, arg=AMDOps.LABEL, tag=f".LOOP_{loop_label}")
  cmp = UOp(Ops.INS, dtypes.void, (acc, x.src[0]), AMDOps.CMP_GE)
  jump_out = UOp(Ops.INS, dtypes.void, (cmp,), AMDOps.CBRANCH_SCC1, tag=f".LOOP_OUT_{loop_label}")
  ctx.loop_label[acc] = loop_label
  return acc, [acc, label, cmp, jump_out]

def _lower_end(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = ctx.loop_label[x.src[1]]
  jmp = UOp(Ops.INS, dtypes.void, arg=AMDOps.BRANCH, tag=f".LOOP_{loop_label}")
  return jmp, [
    x.src[1].ins(AMDOps.ADD, dtype=dtypes.uint32, src=(x.src[1], UOp.const(dtypes.uint32, 1).rtag())),
    jmp,
    UOp(Ops.INS, dtypes.void, arg=AMDOps.LABEL, tag=f".LOOP_OUT_{loop_label}")]

def _lower_reg_store(x:UOp) -> tuple[UOp, list[UOp]]:
  acc, val = x.src
  if acc.op is Ops.INS and acc.arg is AMDOps.FILL:
    # the promoted accumulator was spilled: regalloc gave it a stable scratch slot (acc reloads from acc.src[0]),
    # so write the update straight back to that slot. This degrades to scratch accumulation, but stays correct.
    sp = UOp(Ops.INS, dtypes.void, (acc.src[0], val), AMDOps.SPILL)
    return sp, [sp]
  if not isinstance(acc.reg, Register) or acc.reg.index < 256:
    raise CompileError(f"AMDRenderer expected promoted REG accumulator in VGPR, got {acc}")
  st = UOp(Ops.INS, val.dtype, (val,), AMDOps.MOV, (acc.reg,))
  return st, [st]

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: _lower_range(ctx, x)),
  (UPat(Ops.END, name="x"), lambda ctx,x: _lower_end(ctx, x)),
  (UPat(Ops.INS, arg=AMDOps.REG_STORE, name="x"), lambda x: _lower_reg_store(x)),
  (UPat((Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.SPECIAL, Ops.SINK, Ops.GROUP), name="x"), lambda x: (x, [])),
])

def _vcc_rematerialize(ctx, x:UOp):
  flag_def = x if x.arg in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) else \
             x.src[0] if x.arg in (AMDOps.WHERE, AMDOps.IF_MASK) and x.src[0].op is Ops.INS and x.src[0].arg in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) else None
  if flag_def is None: return None
  # VCC is a single implicit register. Rematerialize compare flags at every consumer so loops and unrelated compares
  # cannot leave a WHERE/IF_MASK reading a stale condition from a previous instruction or previous loop iteration.
  if flag_def is not x: return x, [flag_def, x]
  if ctx.lock is not None and ctx.lock is not flag_def: ctx.clobbered.add(ctx.lock)
  ctx.lock = flag_def
  if flag_def not in ctx.clobbered: return None
  ctx.clobbered.remove(flag_def)
  return x, [flag_def, x]

def _lower_late_index(x:UOp) -> tuple[UOp, list[UOp]]:
  return x, []

def _lower_late_store(x:UOp, a:UOp, val:UOp) -> tuple[UOp, list[UOp]]:
  st = _store_ins(x, a, val)
  return st, [st]

def _promote_reg_buffer(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x.addrspace is not AddrSpace.REG or x not in _reg_promotable_buffers(ctx): return None
  return x, []

def _promote_reg_access(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x.arg is AMDOps.SSTORE:
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1])) is None: return None
    val = x.src[2]
    if slot not in ctx.amd_reg_values:
      ctx.amd_reg_values[slot] = acc = _new_promoted_reg(ctx, val)
      return acc, [acc]
    acc = ctx.amd_reg_values[slot]
    st = UOp(Ops.INS, dtypes.void, (acc, val), AMDOps.REG_STORE)
    return acc, [st]
  if x.arg is AMDOps.SLOAD:
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1])) is None: return None
    return (acc, []) if (acc:=ctx.amd_reg_values.get(slot)) is not None else None
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
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, name="a"), UPat.var("val")), name="x"), _lower_late_store),
  (UPat(Ops.BUFFER, name="x"), _promote_reg_buffer),
  (UPat(Ops.INS, name="x"), _promote_reg_access),
  (UPat(Ops.IF, name="x"), _lower_late_if),
  (UPat(Ops.ENDIF, name="x"), _lower_late_endif),
  (UPat(Ops.INS, name="x"), _vcc_rematerialize),
])

class AMDRenderer(ISARenderer):
  device = "AMD"
  has_local = True
  has_shared = True
  supports_float4 = False
  global_max = (0x8fffffff, 0x8fffffff, 0x8fffffff)
  local_max = (1024, 1, 1)
  local_prod_max = 1024
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {op: (lambda: None) for op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN, Ops.MAX, Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ)}

  def __init__(self, target:Target):
    if not target.arch.startswith("gfx11"): raise RuntimeError(f"AMDRenderer is RDNA3/gfx11 only, got {target.arch}")
    super().__init__(target)

  def stack_pointer(self) -> UOp: return UOp(Ops.INS, dtypes.uint32, arg=AMDOps.SCRATCH_BASE)
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), AMDOps.MOV, (reg,))
  def spill(self, disp:UOp, x:UOp) -> UOp:
    if x.reg.index < 256: raise CompileError("AMDRenderer does not support SGPR spills yet")
    return UOp(Ops.INS, dtypes.void, (disp, x), AMDOps.SPILL)
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    if reg.index < 256: raise CompileError("AMDRenderer does not support SGPR fills yet")
    return UOp(Ops.INS, x.dtype, (disp,), AMDOps.FILL, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    ret = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is AMDOps.LABEL: ret.append(f"{u.tag}:")
      elif u.arg in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1): ret.append(f"  {u.arg.name.lower()} {u.tag}")
      else: ret.append(f"  {u.arg.name.lower()} " + ", ".join(str(s.reg or s.arg) for s in u.src))
    return "\n".join(ret)

  def render(self, uops:list[UOp]) -> str: return self.asm_str(uops, "kernel")

  def _insts_for_uop(self, u:UOp):
    if u.op is not Ops.INS: return []
    match u.arg:
      case AMDOps.LABEL | AMDOps.BRANCH | AMDOps.CBRANCH_SCC1 | AMDOps.DEFINE | AMDOps.SCRATCH_BASE | AMDOps.SCRATCH_SIZE | AMDOps.SCRATCH_ADDR | AMDOps.LDS_BASE:
        return []
      case AMDOps.KERNARG:
        off = u.src[0].arg
        load = r3.s_load_b64(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off) if u.dtype.itemsize == 8 else \
               r3.s_load_b32(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off)
        return [load]
      case AMDOps.MOV:
        if not u.src or u.src[0].op is Ops.SPECIAL: return []
        if (sregs:=_reg_idxs(u.src[0])) and sregs == _reg_idxs(u): return []
        if u.reg.index < 256: return [r3.s_mov_b32(_dst(u), _src(u.src[0]))]
        return [r3.v_mov_b32_e32(_dst(u), _src(u.src[0]))]
      case AMDOps.ADD:
        if u.dtype.scalar() is dtypes.float16: return [r3.v_add_f16_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.dtype.scalar() is dtypes.float32: return [r3.v_add_f32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.reg.index < 256: return [r3.s_add_u32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        # E64 integer forms accept SGPR/inline constant operands and keep uint32 wraparound semantics.
        return [r3.v_add_nc_u32_e64(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.SUB:
        if u.dtype.scalar() is dtypes.float16: return [r3.v_sub_f16_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.dtype.scalar() is dtypes.float32: return [r3.v_sub_f32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.reg.index < 256: return [r3.s_sub_u32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        # E64 integer forms accept SGPR/inline constant operands and keep uint32 wraparound semantics.
        return [r3.v_sub_nc_u32_e64(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.MUL:
        if u.dtype.scalar() is dtypes.float16: return [r3.v_mul_f16_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.dtype.scalar() is dtypes.float32: return [r3.v_mul_f32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        # RDNA3 autogen exposes this as the VOP3 form, which accepts SGPR operands.
        return [r3.v_mul_lo_u32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.MULACC:
        if u.dtype.scalar() is dtypes.float16: return [r3.v_fma_f16(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
        if u.dtype.scalar() is dtypes.float32: return [r3.v_fma_f32(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
        raise CompileError(f"AMDRenderer only supports MULACC for float16/float32, got {u.dtype}")
      case AMDOps.CAST:
        pre, src = _vgpr_data(TMP_VDATA, u.src[0])
        if u.dtype.scalar() in dtypes.ints and u.src[0].dtype.scalar() in dtypes.ints:
          if u.dtype.itemsize > 4 or u.src[0].dtype.itemsize > 4: raise CompileError(f"AMDRenderer cannot cast {u.src[0].dtype} to {u.dtype}")
          # Equal-width int casts are dropped to NOOP in pre-isel, so the narrower type always sets the result width:
          # widening zero/sign-extends from the source width, narrowing truncates to the destination width.
          narrow = u.src[0].dtype if u.src[0].dtype.itemsize <= u.dtype.itemsize else u.dtype
          if narrow.scalar() in dtypes.uints: return pre + [r3.v_and_b32_e32(_dst(u), (1 << (narrow.itemsize * 8)) - 1, src)]
          shift = 32 - narrow.itemsize * 8
          return pre + [r3.v_lshlrev_b32_e64(_dst(u), shift, src), r3.v_ashrrev_i32_e64(_dst(u), shift, _dst(u))]
        if u.dtype.scalar() is dtypes.float32 and u.src[0].dtype.scalar() is dtypes.float16:
          return pre + [r3.v_cvt_f32_f16_e32(_dst(u), src)]
        if u.src[0].dtype.scalar() is dtypes.float32 and u.dtype.scalar() is dtypes.float16:
          return pre + [r3.v_cvt_f16_f32_e32(_dst(u), src)]
        if u.dtype.scalar() is dtypes.float32 and u.src[0].dtype.scalar() in dtypes.ints:
          op = r3.v_cvt_f32_i32_e32 if u.src[0].dtype.scalar() in dtypes.sints else r3.v_cvt_f32_u32_e32
          return pre + [op(_dst(u), src)]
        if u.src[0].dtype.scalar() is dtypes.float32 and u.dtype.scalar() in dtypes.ints:
          op = r3.v_cvt_i32_f32_e32 if u.dtype.scalar() in dtypes.sints else r3.v_cvt_u32_f32_e32
          return pre + [op(_dst(u), src)]
        raise CompileError(f"AMDRenderer cannot cast {u.src[0].dtype} to {u.dtype}")
      case AMDOps.RECIPROCAL:
        if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"AMDRenderer only supports RECIPROCAL for float32, got {u.dtype}")
        pre, val = _vgpr_data(TMP_VDATA, u.src[0])
        dst = _dst(u)
        # V_RCP_F32 is approximate; one Newton-Raphson step keeps f32 division/truncation from drifting by one near exact quotients.
        return pre + [r3.v_rcp_f32_e32(TMP_VADDR, val), r3.v_mul_f32_e32(dst, val, TMP_VADDR),
                      r3.v_sub_f32_e32(dst, 1.0, dst), r3.v_fma_f32(dst, TMP_VADDR, dst, TMP_VADDR)]
      case AMDOps.EXP2 | AMDOps.LOG2 | AMDOps.SQRT | AMDOps.TRUNC:
        if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"AMDRenderer only supports {u.arg.name} for float32, got {u.dtype}")
        pre, val = _vgpr_data(TMP_VDATA, u.src[0])
        return pre + [_F32_UNARY[u.arg](_dst(u), val)]
      case AMDOps.SIN:
        if u.dtype.scalar() is not dtypes.float32: raise CompileError(f"AMDRenderer only supports SIN for float32, got {u.dtype}")
        val = _src(u.src[0])
        pre = [] if isinstance(val, Reg) and val == TMP_VADDR else [r3.v_mov_b32_e32(TMP_VADDR, val)]
        # RDNA V_SIN_F32 takes turns, not radians. Reduce in radians with split 2*pi constants before
        # converting to turns; this is still lightweight but avoids the worst f32 multiply-only reduction error.
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
        if u.dtype.scalar() is dtypes.float16: return [r3.v_max_f16_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        if u.dtype.scalar() is dtypes.float32: return [r3.v_max_f32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
        op = r3.v_max_i32_e64 if u.dtype.scalar() in dtypes.sints else r3.v_max_u32_e64
        return [op(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.SHL:
        return [r3.v_lshlrev_b32_e64(_dst(u), _src(u.src[1]), _src(u.src[0]))]
      case AMDOps.SHR:
        op = r3.v_ashrrev_i32_e64 if u.dtype.scalar() in dtypes.sints else r3.v_lshrrev_b32_e64
        return [op(_dst(u), _src(u.src[1]), _src(u.src[0]))]
      case AMDOps.AND:
        return [r3.v_and_b32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.OR:
        return [r3.v_or_b32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.XOR:
        return [r3.v_xor_b32_e32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      case AMDOps.CMPLT:
        cmp = r3.v_cmp_gt_f16_e32 if u.src[0].dtype.scalar() is dtypes.float16 else \
              r3.v_cmp_gt_f32_e32 if u.src[0].dtype.scalar() is dtypes.float32 else \
              r3.v_cmp_gt_i32_e32 if u.src[0].dtype.scalar() in dtypes.sints else r3.v_cmp_gt_u32_e32
        pre, a = _vgpr_data(TMP_VDATA, u.src[0])
        return pre + [cmp(_src(u.src[1]), a)]
      case AMDOps.CMPNE:
        pre, b = _vgpr_data(TMP_VDATA, u.src[1])
        if u.src[0].dtype.scalar() is dtypes.float16: return pre + [r3.v_cmp_neq_f16_e32(_src(u.src[0]), b)]
        if u.src[0].dtype.scalar() is dtypes.float32: return pre + [r3.v_cmp_neq_f32_e32(_src(u.src[0]), b)]
        return pre + [r3.v_cmp_ne_u32_e32(_src(u.src[0]), b)]
      case AMDOps.CMPEQ:
        pre, b = _vgpr_data(TMP_VDATA, u.src[1])
        if u.src[0].dtype.scalar() is dtypes.float16: return pre + [r3.v_cmp_eq_f16_e32(_src(u.src[0]), b)]
        if u.src[0].dtype.scalar() is dtypes.float32: return pre + [r3.v_cmp_eq_f32_e32(_src(u.src[0]), b)]
        return pre + [r3.v_cmp_eq_u32_e32(_src(u.src[0]), b)]
      case AMDOps.WHERE:
        pre, true_val = _vgpr_data(TMP_VDATA, u.src[1])
        return pre + [r3.v_cndmask_b32_e32(_dst(u), _src(u.src[2]), true_val)]
      case AMDOps.LOAD:
        if (global_load:=_global_load(u.dtype)) is None: raise CompileError(f"AMDRenderer does not support global loads for {u.dtype}")
        pre, addr = _scaled_addr(_dst(u), u.src[1], u.dtype.itemsize)
        return pre + [global_load(_dst(u), addr, saddr=_src(u.src[0]))]
      case AMDOps.STORE:
        if (global_store:=_global_store(u.src[2].dtype)) is None: raise CompileError(f"AMDRenderer does not support global stores for {u.src[2].dtype}")
        pre, addr = _scaled_addr(TMP_VADDR, u.src[1], u.src[2].dtype.itemsize)
        dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
        return pre + dpre + [global_store(addr=addr, data=data, saddr=_src(u.src[0]))]
      case AMDOps.ATOMIC_ADD:
        if u.src[2].dtype.scalar() is not dtypes.float32: raise CompileError(f"AMDRenderer only supports f32 atomic add, got {u.src[2].dtype}")
        pre, addr = _scaled_addr(TMP_VADDR, u.src[1], u.src[2].dtype.itemsize)
        dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
        return pre + dpre + [r3.global_atomic_add_f32(addr=addr, data=data, saddr=_src(u.src[0]), vdst=TMP_VDATA),
                             r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
      case AMDOps.LLOAD:
        if (local_load:=_local_load(u.dtype)) is None: raise CompileError(f"AMDRenderer does not support LDS loads for {u.dtype}")
        pre, addr = _local_addr(u.src[0], u.src[1], u.dtype.itemsize)
        return pre + [local_load(vdst=_dst(u), addr=addr)]
      case AMDOps.LSTORE:
        if (local_store:=_local_store(u.src[2].dtype)) is None: raise CompileError(f"AMDRenderer does not support LDS stores for {u.src[2].dtype}")
        pre, addr = _local_addr(u.src[0], u.src[1], u.src[2].dtype.itemsize)
        dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
        return pre + dpre + [local_store(addr=addr, data0=data), r3.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)]
      case AMDOps.SLOAD:
        if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"AMDRenderer does not support scratch loads for {u.dtype}")
        pre, addr = _scratch_addr(u.src[0], u.src[1], u.dtype.itemsize)
        return pre + [scratch_load(addr=addr, vdst=_dst(u), offset=0, sve=1)]
      case AMDOps.SSTORE:
        if (scratch_store:=_scratch_store(u.src[2].dtype)) is None: raise CompileError(f"AMDRenderer does not support scratch stores for {u.src[2].dtype}")
        pre, addr = _scratch_addr(u.src[0], u.src[1], u.src[2].dtype.itemsize)
        dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
        return pre + dpre + [scratch_store(addr=addr, data=data, offset=0, sve=1), r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
      case AMDOps.BARRIER:
        return [r3.s_barrier()]
      case AMDOps.FILL:
        if u.reg.index < 256: raise CompileError("AMDRenderer does not support SGPR scratch fills yet")
        if u.src[0].arg >= 4096: raise CompileError("AMDRenderer scratch fill offset exceeds 13-bit immediate range")
        if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"AMDRenderer does not support scratch fills for {u.dtype}")
        return [r3.v_mov_b32_e32(TMP_VADDR, 0), scratch_load(addr=TMP_VADDR, vdst=_dst(u), offset=u.src[0].arg)]
      case AMDOps.SPILL:
        if u.src[1].reg.index < 256: raise CompileError("AMDRenderer does not support SGPR scratch spills yet")
        if u.src[0].arg >= 4096: raise CompileError("AMDRenderer scratch spill offset exceeds 13-bit immediate range")
        if (scratch_store:=_scratch_store(u.src[1].dtype)) is None: raise CompileError(f"AMDRenderer does not support scratch spills for {u.src[1].dtype}")
        return [r3.v_mov_b32_e32(TMP_VADDR, 0), scratch_store(addr=TMP_VADDR, data=_src(u.src[1]), offset=u.src[0].arg),
                r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
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
    raise CompileError(f"AMDRenderer cannot encode {u.arg}")

  def _insts_from_linear(self, lin:UOp):
    items, targets, byte, pending = [], {}, 0, {"vm": set(), "lgkm": set()}
    def emit(inst):
      nonlocal byte
      items.append(inst)
      byte += len(inst.to_bytes())
      name = getattr(inst, "op_name", "")
      if name == "S_WAITCNT_VMCNT": pending["vm"].clear()
      elif name == "S_WAITCNT_LGKMCNT": pending["lgkm"].clear()
    def flush(*domains:str):
      for domain in domains:
        if pending[domain]: emit(_wait_for_domain(domain))
    def flush_regs(regs:set[int]):
      if pending["vm"] & regs: flush("vm")
      if pending["lgkm"] & regs: flush("lgkm")
    for u in lin.src:
      if u.op is Ops.INS and u.arg is AMDOps.LABEL:
        flush("vm", "lgkm")
        targets[u.tag] = byte
        continue
      if u.op is Ops.INS and u.arg in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1):
        flush("vm", "lgkm")
        inst = r3.s_branch(0) if u.arg is AMDOps.BRANCH else r3.s_cbranch_scc1(0)
        items.append((inst, u.tag, byte))
        byte += len(inst.to_bytes())
        continue
      flush_regs(set().union(*(_reg_idxs(s) for s in u.src), _reg_idxs(u)))
      for inst in self._insts_for_uop(u):
        emit(inst)
      if (domain:=_wait_domain_for_load(u)) is not None:
        pending[domain] |= _reg_idxs(u)
    insts = []
    for item in items:
      if isinstance(item, tuple):
        inst, target, branch_byte = item
        if target not in targets: raise CompileError(f"AMDRenderer missing branch target {target}")
        delta = (targets[target] - (branch_byte + len(inst.to_bytes()))) // 4
        if not -0x8000 <= delta <= 0x7fff: raise CompileError(f"AMDRenderer branch target {target} out of SOPP range")
        inst.simm16 = delta & 0xffff
      else: inst = item
      insts.append(inst)
    return insts

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    insts = self._insts_from_linear(lin)
    insts.append(r3.s_endpgm())
    nlin = lin.replace(src=tuple(UOp(Ops.INS, arg=i) for i in insts))
    return assemble_linear(prg, nlin, self.target.arch)

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.float16, dtypes.float32}

from tinygrad.dtype import dtypes, AddrSpace, truncate, DType, InvalidType, to_storage_scalar
from tinygrad.codegen.opt import tc
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg, range_str, GroupOp, graph_rewrite
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef
from tinygrad.renderer.cstyle import create_non_native_float_pats, pm_manual_bf16_cast
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
from tinygrad.renderer.amd.elf import assemble_linear
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
import itertools, functools
from dataclasses import dataclass, field
from enum import Enum, auto

# ---- (UOp, dtype) -> Instruction tables ----
dt_to_isa = { dtypes.int32:"i32", dtypes.uint32:"u32", dtypes.float32:"f32", dtypes.float64:"f64", dtypes.float16:"f16", dtypes.int16:"i16", dtypes.uint16:"u16", dtypes.uint64:"u64", dtypes.int64:"i64", dtypes.bfloat16:"bf16", dtypes.uint8:"u8", dtypes.int8:"i8" }
isa_to_dt = { v:k for k,v in dt_to_isa.items() }

# (uop, prefix, opcodes, support 32 and 64 bit encoding (e32/e64 branches with keys))
# TODO: fold MAX, MIN, GT, GE etcw.. ins patterns where possible in isel
insdefs = [
  (Ops.MAX, "v_max", ["f32_e32", "i32_e32", "u32_e32", "f64", "f16_e32"], False),
  (Ops.ADD, "v_add", ["f16_e32", "f32_e32", "f64", "nc_i32", "nc_u32_e32", "nc_u16", "nc_i16"], False),
  (Ops.SUB, "v_sub", ["f16_e32", "f32_e32", "nc_i32", "nc_i16", "nc_u16", "nc_u32_e32"], False),
  (Ops.MUL, "v_mul", ["f16_e32", "f32_e32", "f64", "lo_u32", "lo_u16"], False), # TODO: mul i16?
  (Ops.LOG2, "v_log", ["f16_e32", "f32_e32"], False),
  (Ops.EXP2, "v_exp", ["f16_e32", "f32_e32"], False),
  (Ops.SQRT, "v_sqrt", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.RECIPROCAL, "v_rcp", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.TRUNC, "v_trunc", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.CMPLT, "v_cmp_lt", ["f16", "f32", "f64", "u32", "u64", "i32", "i64", "u16", "i16"], True),
  (Ops.CMPNE, "v_cmp", ["neq_f16", "neq_f32", "neq_f64", "ne_u32", "ne_u64", "ne_i32", "ne_i64", "ne_i16", "ne_u16"], True),
  (Ops.CMPEQ, "v_cmp_eq", ["f16", "f32", "f64", "u16", "u32", "u64", "i16", "i32", "i64"], True)
]

def _build_ins_table(srcs):
  def _extract_dt(ss): return isa_to_dt[next(s for s in ss.split('_') if s in isa_to_dt)]
  def _extract_ins(prefix, code, nenc:int|None=None):
    s = f"{prefix}_{code}"
    if nenc is not None: s += f"_e{nenc}"
    return getattr(RDNA3Ops, s)
  tbl = {}
  for op, pref, codes, bothenc in srcs:
    if bothenc: tbl[op] = { n : { _extract_dt(code) : _extract_ins(pref, code, n) for code in codes } for n in [32, 64] }
    else: tbl[op] = { _extract_dt(code) : _extract_ins(pref, code) for code in codes }
  return tbl

OP_INS = _build_ins_table(insdefs)
V_RSQ = { dtypes.float32:RDNA3Ops.v_rsq_f32_e32, dtypes.float64:RDNA3Ops.v_rsq_f64_e32, dtypes.float16:RDNA3Ops.v_rsq_f16_e32}
V_FMA = { dtypes.float16:RDNA3Ops.v_fma_f16, dtypes.float32:RDNA3Ops.v_fma_f32, dtypes.float64:RDNA3Ops.v_fma_f64 }
# V_MIN = { dtypes.float32:RDNA3Ops.v_min_f32_e32, dtypes.float16:RDNA3Ops.v_min_f16_e32, dtypes.uint32:RDNA3Ops.v_min_u32_e32, dtypes.int32:RDNA3Ops.v_min_i32_e32, dtypes.float64:RDNA3Ops.v_min_f64 }
V_LSHL = { 2:RDNA3Ops.v_lshlrev_b16, 4:RDNA3Ops.v_lshlrev_b32_e32, 8:RDNA3Ops.v_lshlrev_b64 }
V_LSHR = { 2:RDNA3Ops.v_lshrrev_b16, 4:RDNA3Ops.v_lshrrev_b32_e32, 8:RDNA3Ops.v_lshrrev_b64 }
V_ASHR = { 4:RDNA3Ops.v_ashrrev_i32_e32, 8:RDNA3Ops.v_ashrrev_i64 }

# ---- helpers ----
lane_ctr = itertools.count()
def def_reg(dt, reg:Register|tuple[Register,...]): return UOp.placeholder((1,), dt, next(lane_ctr), AddrSpace.REG).replace(tag=(reg,) if isinstance(reg,Register) else reg)
def const(v, dt:DType=dtypes.uint32) -> UOp: return UOp.const((v if isinstance(v, InvalidType) else truncate[dt](v)), dt).rtag()
def is_const(x:UOp): return is_const(x.src[0]) if x.op in {Ops.CAST, Ops.BITCAST, Ops.AFTER} else x.op is Ops.CONST
def to_vgpr(x:UOp) -> UOp: return vmov(x) if is_const(x) else x
def getsign(u:UOp, nbits):
  return UOp(Ops.SHR, dtypes.int32 if nbits <= 32 else dtypes.int64, src=(u, const(31 if nbits <= 32 else 63, dtypes.uint16))).bitcast(u.dtype)
def vmov(x:UOp, r:VRegister|Register|None=None) -> UOp:
  nx = x.ins(RDNA3Ops.v_mov_b16_e32 if x.dtype.itemsize == 2 and dtypes.is_float(x.dtype) else RDNA3Ops.v_mov_b32_e32, src=(x,))
  return nx.rtag() if r is None else nx.replace(tag=(r,))
def smux(dt:DType, sdt:DType, udt:DType): return udt if dtypes.is_unsigned(dt) else sdt

# ---- register classes/kernel init state ----
VGPRS = tuple(Register(f"v{i}", i, size=4) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i, size=4) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
GP_SGPRS, GP_VGPRS = tuple(SGPRS[5:]), tuple(VGPRS[1:])
VCC, EXEC = Register("vcc", 0, size=4), Register("exec_lo", 0, size=4)
FLAT_SCRATCH_LO, FLAT_SCRATCH_HI = Register("flat_scratch_lo", 0, size=4), Register("flat_scratch_hi", 0, size=4)

kernarg_ptr = def_reg(dtypes.uint64, KERNARG_PTR)
execop, vccop = def_reg(dtypes.uint32, EXEC), def_reg(dtypes.uint32, VCC)
flat_scratch_ptr = (def_reg(dtypes.uint32, FLAT_SCRATCH_LO), def_reg(dtypes.uint32, FLAT_SCRATCH_HI))

# ---- register movement helpers ----
def packb16(lo:UOp, hi:UOp):
  if dtypes.is_float(lo.dtype): return UOp(Ops.INS, arg=RDNA3Ops.v_pack_b32_f16, src=(lo,hi))
  lo = lo & const(0xFFFF) # mask off upper half
  return _vop3(UOp(Ops.INS, arg=RDNA3Ops.v_lshl_or_b32, src=(hi, const(16, dtypes.int32), lo)))

# TODO: replicate this for b8
# stack of 16 bit loads -> load directly into high/low halfs
def load_into_stack(ctx, x:UOp) -> UOp:
  if x.src[0].src[0].addrspace is not AddrSpace.GLOBAL: return None
  out = []
  vp = ctx.vreg(GP_VGPRS, width=len(x.src)//2)
  for l in range(0, len(x.src), 2):
    vr = vp.sub(l//2)
    lo,hi = x.src[l], x.src[l+1]
    lo,hi = load(ctx, lo, lo.src[0]), load(ctx, hi, hi.src[0])
    def _mopc(u:UOp, opc) -> UOp: return u.replace(src=(u.src[0].replace(arg=opc),) + u.src[1:])
    lo,hi = _mopc(lo, RDNA3Ops.global_load_d16_b16).replace(tag=(vr,)), _mopc(hi, RDNA3Ops.global_load_d16_hi_b16).replace(tag=(vr,))
    out.append(hi.after(lo))
  return UOp.group(*out, dtype=x.dtype, tag=(vp,))

def stack2regs(x:UOp):
  nregs, mvs = ((len(x.src) * x.dtype.itemsize) + 3) // 4, []
  for i in range(nregs):
    if x.dtype.itemsize == 2:
      if i*2+1 < len(x.src): mvs.append(packb16(x.src[i*2], x.src[i*2+1]))
      else: mvs.append(vmov(x.src[i*2]))
    elif x.dtype.itemsize == 1:
      def _pk(j:int):
        p = x.src[i*4+j].bitcast(dtypes.uint32) & const(0xFF)
        return p if j == 0 else p << const(8 * j)
      out = _pk(0)
      for j in range(3): out = out | _pk(j+1)
      mvs.append(out)
    else: mvs.append(vmov(x.src[i]))
  return UOp.group(*mvs, dtype=x.dtype) if len(mvs) > 1 else mvs[0].replace(dtype=x.dtype)

# NOTE: should this just be triggered in to_vgpr??
def gethalf(x:UOp, buf:UOp, idx:UOp):
  bb = buf
  while bb.op is Ops.AFTER: bb = bb.src[0]
  # only trigger on value uses, ex. b16 alu stack inputs/outputs
  # NOT index into memory/buffers
  if bb.op is Ops.BUFFER: return None
  b32 = buf.index(const(idx.val // 2, dtypes.int32)).replace(dtype=dtypes.uint32)
  # NOTE: manual construction, needs to be cleaned
  if idx.val % 2 != 0: return UOp(Ops.BITCAST, src=(UOp(Ops.SHR, src=(b32, const(16))),), arg=x.dtype)
  else: return x.ins(RDNA3Ops.v_mov_b16_e32, src=(b32,))

# ---- operand legalization wrappers ----
def _vop3(x:UOp):
  lits = [s for s in x.src if s.op is Ops.CONST]
  return x if len(lits) <= 1 else x.replace(src=tuple([vmov(s) if s in lits[1:] else s for s in x.src]))

rev_op_order = { RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshlrev_b16, RDNA3Ops.v_lshlrev_b64, RDNA3Ops.v_lshrrev_b32_e32, RDNA3Ops.v_lshrrev_b16, RDNA3Ops.v_lshrrev_b64, RDNA3Ops.v_ashrrev_i32_e32, RDNA3Ops.v_ashrrev_i64 }
def _vop2(ctx, x:UOp):
  if x.arg in rev_op_order: x = x.replace(src=x.src[2::-1] + x.src[2:])
  if not is_const(x.src[1]): return x # TODO: should check positive vgpr, sgpr cant be used in vrsc1
  rest = x.src[2:] if len(x.src) > 2 else ()
  non_commutative = x.arg in set(OP_INS[Ops.SUB].values()) | rev_op_order
  if not non_commutative and not is_const(x.src[0]): return x.replace(src=(x.src[1], x.src[0]) + rest)
  return x.replace(src=(x.src[0], vmov(x.src[1])) + rest)

# TODO: allocate vgpr / sgpr based on op group (x.arg.func)?
def alloc_vregs(ctx, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], VRegister): return None

  if x.op is Ops.GROUP:
    vreg = ctx.vreg(GP_VGPRS, width=len(x.src))
    # TODO: replace all references to src edges to avoid duplicates because of tag changes
    return x.replace(tag=(vreg,), src=tuple(s.replace(tag=(vreg.sub(i),)) for i,s in enumerate(x.src)))

  elif isinstance(x.tag, tuple):
    cons, width = x.tag if isinstance(x.tag[0], tuple) else (x.tag, 1)
    vr = ctx.vreg(cons, width=width)
  else:
    vr = ctx.vreg(GP_VGPRS, width=max(x.dtype.itemsize // 4, 1))
  return x.replace(tag=(vr,))

# https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
# TODO: batch param loading? ex. s_load_b128
# NOTE: codegen doesnt know to place param s_loads early, delay lowering like load/store
def abi(ctx, x:UOp) -> UOp|None:
  if x.op is Ops.SPECIAL:
    dim = int(x.arg[-1])
    if x.arg[0] == 'g': return vmov(x.replace(tag=(WGIDS[dim],), dtype=dtypes.uint32)).rtag()
    else: # granulated work item ids, packed into 3 10 bit fields in v0, extract with bfe
      return x.ins(RDNA3Ops.v_bfe_u32, dtype=dtypes.uint32, src=(x.replace(tag=WIIDS), const(10 * dim), const(10)))
  offs = sum(8 if u.op == Ops.PARAM else 4 for u in ctx.func_args[:ctx.func_args.index(x)])
  addr = (kernarg_ptr, const(offs))
  if x.addrspace is AddrSpace.ALU: return vmov(x.replace(src=addr, tag=ctx.vreg(GP_SGPRS)))
  return x.replace(dtype=dtypes.ulong, src=addr, tag=ctx.vreg(GP_SGPRS, width=2, alignment=2),)

# ----- memory access ----
# GLOBAL_ADDR = VADDR_U64 + IMMOFFS_u16
# NOTE: manual SHL construction to avoid none shape error mixing with Ops.INS? fix this somehow
def fold_global(base:UOp, idx:UOp): # (voff, ioffs)
  disp_scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(disp_scale.bit_length() - 1, dtypes.int32)
  vaddr, offs = idx, const(0, dtypes.uint16)
  def foldable(v:int) -> bool: return -(1 << 12) <= v < (1 << 12)
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST and foldable((_offs := idx.src[1].val * disp_scale)):
    vaddr, offs = idx.src[0], const(_offs, dtypes.int16)
    vaddr = int_to_int64(vaddr << shft, dtypes.uint64)
    return (UOp(Ops.ADD, dtype=dtypes.uint64, src=(vaddr, base.bitcast(dtype=dtypes.uint64))), offs)
  else: # saddr + vaddr
    return (to_vgpr(vaddr) << shft, base)

# LDS_ADDR = VGPR_ADDR_u32 + imm_byte_offset_u16
# TODO: actually calculate lds offset per seperate BUFFER, (ctx.func_args)
def fold_lds(base:UOp, idx:UOp): # (vaddr, ioffs)
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(0),)), const(idx.arg * scale, dtypes.uint16), base)
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0].cast(dtypes.uint32), const(idx.src[1].arg * scale, dtypes.uint16), base)
  shft = const(scale.bit_length() - 1)
  offs = UOp(Ops.SHL, dtypes.uint32, src=(idx,shft))
  return (offs, const(0, dtypes.uint16), base)

def fold_address(x:UOp): return fold_lds(*x.src[:2]) if x.addrspace is AddrSpace.LOCAL else fold_global(*x.src[:2])

def load(ctx, x:UOp, idx:UOp):
  if idx.addrspace is AddrSpace.REG:
    return x.replace(tag=ctx.regptr(idx, GP_VGPRS, width=(idx.dtype.itemsize+3)//4)) if x.tag is None else None
  oidx = idx
  while idx.op is Ops.AFTER: idx=idx.src[0]
  n = idx.src[-1].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.src[0].dtype.itemsize
  suffix = "b" if sz > 2 else "u" if dtypes.is_unsigned(x.dtype) or dtypes.is_float(x.dtype) else "i"
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_load_{suffix}{sz*8}")
  vp = ctx.vreg(GP_VGPRS, width=(sz+3)//4)
  addr = UOp(Ops.NOOP, src=fold_address(idx), arg=opc)
  if oidx.op is Ops.AFTER: addr = addr.replace(src=(addr.src[0].after(*oidx.src[1:]),) + addr.src[1:])
  return x.replace(src=(addr, *x.src[1:]), tag=(vp,))

def lower_gated_load(ctx, x:UOp, addr:UOp, alt:UOp, gate:UOp):
  init = [ctx.ren.copy(s, rdef(x).sub(i)) for i,s in enumerate(alt.src)] if alt.op is Ops.GROUP else [ctx.ren.copy(alt, rdef(x))]
  load = UOp(Ops.INS, x.dtype, arg=addr.arg, src=addr.src, tag=x.tag)
  mif = UOp(Ops.INS, arg=RDNA3Ops.s_and_saveexec_b32, src=(gate,), tag=ctx.vreg(GP_SGPRS))
  return load, init + [mif, load, UOp(Ops.ENDIF, src=(mif,))]

def store(ctx, x:UOp, idx:UOp, val:UOp):
  if idx.addrspace is AddrSpace.REG:
    vregs, i = ctx.regptr(idx, GP_VGPRS, width=(idx.dtype.itemsize+3)//4), idx.src[1].val
    if val.op is Ops.GROUP:
      if idx.dtype.itemsize == 8: return ctx.ren.copy(UOp.group(val.src[i*2], val.src[i*2+1], dtype=idx.dtype), vregs[i])
      else: return ctx.ren.copy(val.src[idx.src[1].val].after(val, idx), vregs[idx.src[1].val])
    else: return ctx.ren.copy(val.after(idx).replace(dtype=idx.dtype), *vregs)
  oidx = idx
  while idx.op is Ops.AFTER: idx = idx.src[0]
  n = idx.src[-1].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.dtype.itemsize
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_store_b{sz*8}")
  addr = UOp(Ops.NOOP, src=fold_address(idx), arg=opc)
  if oidx.op is Ops.AFTER: addr = addr.replace(src=(addr.src[0].after(*oidx.src[1:]),) + addr.src[1:])
  return UOp(Ops.STORE, src=(addr, to_vgpr(val)) + x.src[2:])

# ------ ALU ------
def cmp(ctx, x:UOp):
  _mask_cmp = { Ops.CMPNE:RDNA3Ops.s_xor_b32, Ops.XOR:RDNA3Ops.s_xor_b32, Ops.OR: RDNA3Ops.s_or_b32, Ops.AND:RDNA3Ops.s_and_b32, Ops.CMPLT: RDNA3Ops.s_and_not1_b32, Ops.CMPEQ:RDNA3Ops.s_xnor_b32 }
  scmp = x.src[0].dtype is dtypes.bool and x.src[1].dtype is dtypes.bool
  dt = x.src[0].dtype if x.src[0].op is not Ops.AFTER else x.src[0].src[0].dtype
  ins = _mask_cmp[x.op] if scmp else OP_INS[x.op][64][dt]
  if scmp and x.op is Ops.CMPLT: x=x.replace(src=(x.src[1], x.src[0]))
  x = x.ins(ins, tag=GP_SGPRS)
  return x if scmp else _vop3(x)

def arith64(ctx, x:UOp):
  a, b = x.src
  ins_lo = RDNA3Ops.v_add_co_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_u32
  ins_hi = RDNA3Ops.v_add_co_ci_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_ci_u32
  narrow = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  vreg = ctx.vreg(GP_VGPRS, width=2) # NOTE: after causes a problem for auto allocating group reg?
  lo = UOp(Ops.INS, dtype=dtypes.uint32, arg=ins_lo, src=(a.index(0), b.index(0)), tag=(vreg.sub(0),))
  hi = UOp(Ops.INS, dtype=narrow, arg=ins_hi, src=(a.index(1), b.index(1), vccop, lo), tag=(vreg.sub(1),)).after(lo)
  return UOp.group(lo, hi, dtype=x.dtype).replace(tag=(vreg,))

# a64 * b64 = (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo) =  a_hi * 2^32 * b_lo + b_hi * 2^32 * a_hi + a_lo * b_lo
def mul64(ctx, x:UOp):
  def _mad(a:UOp, b:UOp, c:UOp=const(0, x.dtype)): return UOp(Ops.INS, x.dtype, arg=RDNA3Ops.v_mad_u64_u32, src=(a,b,c))
  def _up(x:UOp): return x.ins(RDNA3Ops.v_lshlrev_b64, src=(const(32, dtypes.int32),x))
  a, b = x.src
  p1 = _up(_mad(a.index(1), b.index(0)))
  p2 = _up(_mad(a.index(0), b.index(1)))
  p3 = arith64(ctx, UOp(Ops.ADD, x.dtype, src=(p1,p2)))
  return _mad(a.index(0), b.index(0), p3)

# TODO: fold const 64 as imms here?, shift hi, mask lo
def bitwise64(ctx, x:UOp, ins):
  a, b = x.src
  lo = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.index(0), b.index(0)))
  hi = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.index(1), b.index(1)))
  return UOp.group(lo, hi, dtype=x.dtype)

# Allows embedding special alu instructions ex. mul_hi without introducing
# Ops.INS which have None shape and cause alu() _broadcast to error
def _aluhint(x:UOp, hint): return x.replace(arg=hint)
def _mulhi(a:UOp, b:UOp, signed:bool) -> UOp:
  return _aluhint(UOp(Ops.MUL, dtypes.uint32, src=(a, b)), RDNA3Ops.v_mul_hi_i32 if signed else RDNA3Ops.v_mul_hi_u32)
# use explicit SUBs (not x + y*-1): this runs post-decomp so nothing repairs the weak -1, and weak consts break isel vreg sizing
def _sub(x:UOp, y:UOp) -> UOp: return UOp(Ops.SUB, x.dtype, src=(x, y))

def idiv(ctx, x:UOp):
  signed = not dtypes.is_unsigned(x.dtype)
  dt = dtypes.uint32 if x.dtype.itemsize <= 4 else dtypes.uint64
  a, b = x.src[0].cast(dt), x.src[1].cast(dt)
  if signed:  # take abs values, remember result sign (getsign yields 0 or -1)
    sa, sb = getsign(a, x.dtype.itemsize*8), getsign(b, x.dtype.itemsize*8)
    a, b, sign = (a + sa) ^ sa, (b + sb) ^ sb, sa ^ sb
  if dt is dtypes.uint32:
    # unsigned 32-bit: f32 reciprocal -> fixed-point m ~ 2^32/b; two signed-int NR steps bias m low, so q = mulhi(a, m-1)
    # never overshoots (errs a few low) and the exact remainder a - q*b is closed with a few conditional bumps. no f64.
    u32, one, zero = dtypes.uint32, const(1, dtypes.uint32), const(0, dtypes.uint32)
    m = (b.cast(dtypes.float32).reciprocal() * const(float(2**32), dtypes.float32)).cast(u32)
    for _ in range(2): m = m + _mulhi(m, _sub(zero, b * m), signed=True)   # m += mulhi_i(m, 2^32 - b*m)
    q = _mulhi(a, _sub(m, one), signed=False)
    q = (b < const(3, u32)).where(UOp(Ops.SHR, u32, src=(a, _sub(b, one))), q)  # b in {1,2}: m>=2^31 breaks signed mulhi
    r = _sub(a, q * b)
    for _ in range(3):                                                     # q is at most 3 low; bump while remainder >= b
      over = (r < b).logical_not()
      q, r = q + over.where(one, zero), _sub(r, over.where(b, zero))
  else:
    # 64-bit: fp64 reciprocal + Newton-Raphson (~2^53), quotient within 1, rtne + integer fixups. https://arxiv.org/pdf/2207.08420
    ad, bd = a.cast(dtypes.double), b.cast(dtypes.double)
    invbd0 = b.cast(dtypes.float).reciprocal().cast(dtypes.double)
    invbd = (bd * (invbd0 * const(-1.0, dtypes.double)) + const(1.0, dtypes.double)) * invbd0 + invbd0
    q1 = _aluhint((ad * invbd).trunc(), RDNA3Ops.v_rndne_f64_e32).cast(dtypes.uint64)  # todo: hacky, not really trunc
    r1 = _sub(a.cast(dtypes.int64), b.cast(dtypes.int64) * q1.cast(dtypes.int64))
    q3 = _aluhint((r1.cast(dtypes.double) * invbd).trunc(), RDNA3Ops.v_rndne_f64_e32).cast(dtypes.int64)
    r3 = _sub(r1, b.cast(dtypes.int64) * q3)
    q0 = q1 + (r3 < const(0, dtypes.int64)).where(_sub(q3, const(1, dtypes.int64)), q3).cast(dtypes.uint64)
    is_big, is_one = b.cast(dtypes.int64) < const(0, dtypes.int64), b <= const(1, dtypes.uint64)  # b >= 2^63 | b <= 1
    q = (is_one | is_big).where(is_big.where((a >= b).cast(dtypes.uint64), a), q0)
  if signed: q = _sub(q ^ sign, sign)  # (q ^ sign) - sign negates q iff sign is all-ones
  return q if q.dtype == x.dtype else q.cast(x.dtype)

def alu(ctx, x:UOp): # alu arg used for machine instruction overrides, ex. mul_hi for cdiv
  ins = x.arg if isinstance(x.arg, functools.partial) else OP_INS[x.op][x.dtype]
  return x.ins(ins) if len(x.src) == 1 else _vop2(ctx, x.ins(ins))

def render_wmma(ctx, wmma:UOp):
  a,b,acc = wmma.src
  srcdt = dt_to_isa[wmma.arg[1]]
  if wmma.arg[1] in dtypes.int8s: srcdt = "iu8"
  ins = getattr(RDNA3Ops, f"v_wmma_{dt_to_isa[wmma.dtype]}_16x16x16_{srcdt}")
  return UOp(Ops.INS, arg=ins, dtype=wmma.dtype, src=(a,b,acc), tag=(ctx.vreg(GP_VGPRS, width=8),))

# ---- casting utilities -----
def int_to_int64(y:UOp, tdt:DType):
  hi = vmov(const(0)) if dtypes.is_unsigned(y.dtype) else getsign(to_vgpr(y), y.dtype.itemsize*8)
  return UOp.group(vmov(y), hi, dtype=tdt)

# NOTE: use v_bfe instead of hand rolled masking
def intcast(y:UOp, x:UOp):
  if y.dtype.itemsize == x.dtype.itemsize: return y if y.dtype == x.dtype else y.bitcast(x.dtype)  # same size: noop or retype
  if x.dtype.itemsize > y.dtype.itemsize:
    if x.dtype.itemsize == 2: return (y & const(0xFFFF, y.dtype)).bitcast(x.dtype)
    return (y & const(0xFFFFFFFF, y.dtype)).bitcast(x.dtype)
  if y.dtype.itemsize <= 4 and x.dtype.itemsize < y.dtype.itemsize: # masked narrow
    if x.dtype.itemsize == 2: return (y & const(0xFFFF, y.dtype)).bitcast(x.dtype)
    return (y & const(0xFF, y.dtype)).bitcast(x.dtype)

# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp#L3691
def f64_to_int64(y:UOp, tdt:DType):
  hi_dt = dtypes.uint32 if dtypes.is_unsigned(tdt) else dtypes.int32
  tr = UOp(Ops.TRUNC, dtypes.float64, src=(y,))
  hi_f = tr.ins(RDNA3Ops.v_ldexp_f64, src=(tr,const(-32, dtypes.int16)))
  hi_f = UOp(Ops.INS, dtypes.float64, arg=RDNA3Ops.v_floor_f64_e32, src=(hi_f,))
  lo_f = hi_f.ins(RDNA3Ops.v_ldexp_f64, src=(hi_f, const(32, dtypes.int16))) # tr - hi_f * 2 ^ 32
  lo_f = UOp(Ops.ADD, dtypes.float64, src=(tr, UOp(Ops.MUL, dtypes.float64, src=(lo_f, const(-1., dtypes.float64)))))
  return UOp.group(lo_f.cast(dtypes.uint32), hi_f.cast(hi_dt), dtype=tdt)

# TODO: currently only 53 bit precision (f64 mantissa), could do better
def long2double(x:UOp):
  lo = x.index(0).replace(dtype=dtypes.uint32).cast(dtypes.float64)
  hi = x.index(1).replace(dtype=dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32).cast(dtypes.float64)
  hi = hi.ins(RDNA3Ops.v_ldexp_f64, src=(hi,const(32, dtypes.int16)))
  return UOp(Ops.ADD, dtype=dtypes.float64, src=(lo,hi))

def const64(x:UOp):
  v = x.val.bits if dtypes.is_float(x.dtype) else x.val
  hi_dt = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  return UOp.group(vmov(const(v)), vmov(const(v >> 32, hi_dt)), dtype=x.dtype)

# ---- control flow ----
def restoreexec(mask:UOp) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_or_b32, src=(execop,mask), tag=(EXEC,))
def label(ctx, name:str) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_nop, tag=name)

def lower_range(ctx, x:UOp):
  bnd, mask = x.src[0], x.src[-1]
  acc = x.ins(RDNA3Ops.v_mov_b32_e32, src=(const(0),))
  ctx.loop_label[acc] = range_str(x)
  ctx.exec_mask[acc] = mask
  ctx.range_bnd[acc] = bnd
  loop_body = label(ctx, f".LOOP_BODY_{range_str(x)}")
  return acc, [acc, loop_body]

def lower_end(ctx, x:UOp, acc:UOp):
  loop_end = label(ctx, f".LOOP_END_{ctx.loop_label[acc]}")
  inc = UOp(Ops.INS, arg=RDNA3Ops.v_add_nc_u32_e32, src=(const(1), acc), tag=acc.tag)
  jmp = UOp(Ops.INS, arg=RDNA3Ops.s_cbranch_execnz, tag=f".LOOP_BODY_{ctx.loop_label[acc]}")
  pred = UOp(Ops.INS, arg=RDNA3Ops.v_cmpx_lt_u32_e64, src=(acc,ctx.range_bnd[acc]), tag=(EXEC,))
  return inc, [inc, pred, jmp, loop_end, restoreexec(ctx.exec_mask[acc])]

# ---- lowering passes ----
extra_matcher = PatternMatcher([
  (UPat.cvar("x", dtype=dtypes.bfloat16), lambda x: const(x.val if isinstance(x.val, InvalidType) else to_storage_scalar(x.val, dtypes.bfloat16), dtypes.uint16).bitcast(dtypes.bfloat16)),
  (UPat(Ops.EXP2, dtypes.double, src=(UPat.var("d"),)), xexp2),
  (UPat(Ops.LOG2, dtypes.double, src=(UPat.var("d"),)), xlog2),
  (UPat(Ops.CMOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - b * a.alu(Ops.CDIV, b)), # hack from x86
]) + pm_manual_bf16_cast + create_non_native_float_pats((dtypes.bfloat16,)) + tc.pm_validate_wmma_rdna3

pm_float_to_int = PatternMatcher([
  (UPat.var("y", dtypes.half).cast((dtypes.double,)+dtypes.int32s+dtypes.int64s, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", dtypes.half).cast(dtypes.int8s, name="x"), lambda y,x: y.cast(smux(x.dtype, dtypes.int32, dtypes.uint32)).bitcast(x.dtype)),
  (UPat.var("y", dtypes.float32).cast(dtypes.int16s+dtypes.int8s, name="x"), lambda y,x: y.cast(smux(x.dtype, dtypes.int32, dtypes.uint32))),
  (UPat.var("y", dtypes.float32).cast(dtypes.int64s, name="x"), lambda y,x: y.cast(smux(x.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.double).cast(dtypes.half), lambda y: y.float().half()),
  (UPat.var("y", dtypes.double).cast(dtypes.int16s+dtypes.int8s, name="x"), lambda y,x: y.float().cast(smux(x.dtype, dtypes.int32, dtypes.uint32))),
  (UPat.var("y", dtypes.double).cast(dtypes.int64s).named("x"), lambda y,x: f64_to_int64(y, x.dtype)),
])

pm_int_to_float = PatternMatcher([
  (UPat.var("y", dtypes.int32s).cast(dtypes.half), lambda y: y.float().cast(dtypes.half)),
  (UPat.var("y", dtypes.int8s).cast(dtypes.half), lambda y: y.cast(smux(y.dtype, dtypes.int16, dtypes.uint16)).cast(dtypes.half)),
  (UPat.var("y", dtypes.int8s+dtypes.int16s).cast((dtypes.float,dtypes.double), name="x"), lambda y,x: y.cast(smux(y.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.int64s).cast((dtypes.float, dtypes.half), name="x"), lambda y,x: long2double(y).cast(dtypes.float).cast(x.dtype)),
  (UPat.var("x", dtypes.int64s).cast(dtypes.float64), long2double),
])

pre_isel_matcher = PatternMatcher([
  (UPat(Ops.STACK, name="x"), lambda x: stack2regs(x) if len(x.src) and not (x.dtype.itemsize == 2 and all(s.op is Ops.LOAD for s in x.src)) else None),
  # --- bool repr ---
  # NOTE: booleans get passed around as sgpr masks in between loads and stores, but are converted / realized at mem ops to u8
  (UPat(Ops.STORE, src=(UPat.var("buf"), UPat.var("val", dtype=dtypes.bool)), allow_any_len=True, name="x"), \
    lambda buf,val,x: x.replace(src=(buf,val.cast(dtypes.uint32)) + x.src[2:])),
  (UPat(Ops.LOAD, dtypes.bool, allow_any_len=True, name="x"), lambda x: x.replace(dtype=dtypes.uint32) != 0),
  (UPat(Ops.BUFFER, dtypes.bool, name="x"), lambda x: x.replace(dtype=dtypes.uint8) if x.addrspace is AddrSpace.REG else None),
  (UPat.cvar("x", dtypes.bool), lambda x: x.ins(RDNA3Ops.s_mov_b32, src=(const((1 << 32) - 1 if x.val else 0),), tag=GP_SGPRS)),
  # TODO: use bfe/bi to unpack/pack once we have batched loads/stores
  (UPat.var("y", dtypes.bool).cast(name="x"), lambda y,x: y.where(const(1, x.dtype), const(0, x.dtype))),
  # --- int8 alu is int16 ---
  (UPat(GroupOp.ALU, dtypes.int8s, name="x"), lambda x: x.replace(dtype=smux(x.dtype, dtypes.int16, dtypes.uint16))),
  (UPat(GroupOp.Comparison, src=(UPat.var("y", dtype=dtypes.int8s), UPat()), name="x"),
    lambda x,y: x.replace(src=(y.bitcast(smux(y.dtype, dtypes.int16, dtypes.uint16)), x.src[1]))),
  # -- int -> int casts ---
  (UPat.var("y", dtypes.int8s+dtypes.int16s+dtypes.int32s).cast(dtypes.int64s, name="x"), lambda y,x: int_to_int64(y, x.dtype)),
  (UPat.var("y", dtypes.int64s).cast(dtypes.int16s+dtypes.int8s+dtypes.int32s, name="x"),
    lambda y,x: y.index(0).replace(dtype=smux(y.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.ints).cast(dtypes.ints).named("x"), intcast),
  # narrowing long goes through b32
  (UPat(Ops.MUL, dtypes.int16, name="x"), lambda x: x.replace(dtype=dtypes.int32)),
  # --- 64 bit semantics ---
  (UPat(Ops.CONST, (dtypes.float64, dtypes.long, dtypes.ulong), name="x"), const64),
  (UPat(Ops.WHERE, src=(UPat.var("pred"), UPat.var("a", dtype=(dtypes.ulong,dtypes.long,dtypes.float64)), UPat.var("b"))),
    lambda pred,a,b: UOp.group(pred.where(a.index(0),b.index(0)), pred.where(a.index(1), b.index(1)), dtype=a.dtype) if a.op is not Ops.INDEX else None),
  (UPat((Ops.SHR, Ops.SHL), dtypes.int64s+(dtypes.float64,), src=(UPat(), UPat.cvar("y")), name="x"), # prevent 64 bit immediate from being realized into 2 regs for shift
    lambda y,x: x.replace(src=(x.src[0], y.replace(dtype=dtypes.uint32)))),
  # shift distance must be in single vgpr
  (UPat((Ops.SHR, Ops.SHL), dtypes.int64s, src=(UPat.var("val"), UPat.var("shft")), name="x"),
    lambda x,val,shft: x.replace(src=(val, shft.cast(dtypes.uint32)))),
  # --- other ---
  (UPat(Ops.CDIV, name="x"), idiv),
  # NOTE: this exposes issues with vgpr value representation invariants, if a value takes up less than 32 bits either we dont care about
  # what else is in there, could be garbage, or it has to be masked at boundaries and sign extended carefully etc... so it can be operated on
  (UPat((Ops.CAST, Ops.BITCAST), dtypes.uchar, src=(UPat.var("y", dtype=dtypes.int8),)), \
    lambda y: (y & const((1 << 8) - 1, dtypes.uint8)).replace(dtype=dtypes.uint8)),
  (UPat((Ops.CAST, Ops.BITCAST), dtypes.ushort, src=(UPat.var("y", dtype=dtypes.int16),)), \
    lambda y: (y & const((1 << 16) - 1, dtypes.uint16)).replace(dtype=dtypes.uint16)),
  # hack?
  (UPat(Ops.MAX, dtypes.int64s, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda a,b,x: (a < b).where(b, a).replace(dtype=x.dtype)),
  (UPat(Ops.MUL, dtypes.int32, name="x"), lambda x: x.replace(dtype=dtypes.uint32).bitcast(dtypes.int32)),
  (UPat(Ops.MAX, dtypes.int16s, name="x"), lambda x: x.replace(dtype=smux(x.dtype, dtypes.int32, dtypes.uint32)).bitcast(x.dtype)),
]) + pm_float_to_int + pm_int_to_float

pm_alu_fusion = PatternMatcher([
  (UPat().sqrt().named("x").reciprocal(), lambda x: x.ins(V_RSQ[x.dtype]) if x.dtype in V_RSQ else None),
  ((UPat(Ops.MUL, dtypes.floats, name="a") + UPat.var("b")).named("x"),
    lambda ctx,a,b,x: _vop3(x.ins(V_FMA[a.dtype], src=a.src + (b,)))),
  (UPat(Ops.ADD, dtypes.uint32, src=(UPat(Ops.ADD, name="y"), UPat.var("b")), name="x"),
    lambda ctx,x,y,b: _vop3(x.ins(RDNA3Ops.v_add3_u32, src=y.src + (b,)))),
])

isel_matcher = pm_alu_fusion + PatternMatcher([
  # TODO: make this general
  (UPat(Ops.STACK, dtypes.int16s+(dtypes.half,dtypes.bfloat16), src=UPat(Ops.LOAD), name="x"), load_into_stack),
  # --- control flow ---
  # how to remove positional arg contracts, make inter-lowering semantics explicit so its clear what src edges represent
  (UPat(Ops.RANGE, name="x"), \
    lambda ctx,x: x.replace(src=x.src + (UOp(Ops.INS, arg=RDNA3Ops.s_mov_b32, src=(execop,), tag=ctx.vreg(GP_SGPRS)),))
    if x.src[-1].op is not Ops.INS else None),
  # add exec mask edge to src
  (UPat(Ops.END, src=(UPat(), UPat.var("rng")), name="x"), \
    lambda x,rng: x.replace(src=(x.src[0],rng,rng.src[-1])) if rng.src[-1].op is Ops.INS else None),
  # --- double precis bit alu ---
  (UPat(Ops.MUL, dtypes.int64s, name="x"), mul64),
  (UPat((Ops.ADD, Ops.SUB), dtypes.int64s, name="x"), arith64),
  # --- general alu ---
  (UPat(Ops.SHR, name="x"), lambda ctx,x: _vop2(ctx, x.ins(V_LSHR[max(2, x.dtype.itemsize)] \
    if dtypes.is_unsigned(x.dtype) else V_ASHR[max(4, x.dtype.itemsize)]))),
  (UPat(Ops.SHL, name="x"), lambda ctx,x: _vop2(ctx, x.ins(V_LSHL[max(2, x.dtype.itemsize)]))),
  (UPat(GroupOp.Comparison|{Ops.XOR, Ops.AND, Ops.OR}, dtypes.bool, name="x"), cmp),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), name="x"), lambda ctx,x: \
    _vop2(ctx, x.ins(getattr(RDNA3Ops, f"v_{x.op.name.lower()}_b32_e32"))) \
    if x.dtype.itemsize < 8 else bitwise64(ctx, x, getattr(RDNA3Ops, f"v_{x.op.name.lower()}_b32_e32"))),
  (UPat(Ops.WHERE, dtypes.bool, src=(UPat.var("mask"), UPat.var("a"), UPat.var("b")), name="x"),
    lambda mask,a,b,x: (mask & a) | (~mask & b)),
  (UPat.var("pred").where(UPat.var("a"), UPat.var("b")).named("x"), lambda pred,a,b,x:
    _vop3(x.ins(RDNA3Ops.v_cndmask_b32_e64 if x.dtype.itemsize >= 4 else RDNA3Ops.v_cndmask_b16, src=(b,a,pred)))),
  (UPat(GroupOp.Binary|GroupOp.Unary, name="x"), alu),
  (UPat(Ops.WMMA, name="wmma"), render_wmma),
  (UPat.var("y").cast(name="x"), lambda y,x: x.ins(getattr(RDNA3Ops, f"v_cvt_{dt_to_isa[x.dtype]}_{dt_to_isa[y.dtype]}_e32"))),
  # --- mem ops ---
  (UPat((Ops.INDEX, Ops.SHRINK)).or_after("idx").store(UPat.var("val"), allow_any_len=True).named("x"), store),
  (UPat((Ops.INDEX, Ops.SHRINK)).or_after("idx").load(allow_any_len=True, name="x"), load),
  # --- other ---
  (UPat((Ops.SPECIAL, Ops.PARAM), name="x"), lambda ctx,x: abi(ctx,x)
    if not any(isinstance(v,(VRegister, Register)) for v in rdefs(x)) else None),
  (UPat((Ops.INS, Ops.GROUP, Ops.RANGE), name="x"), alloc_vregs),
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  (UPat(Ops.INDEX, (dtypes.half,) + dtypes.int16s, src=(UPat.var("buf"), UPat.cvar("idx")), name="x"), gethalf),
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),
  (UPat(name="x").bitcast().named("y"), lambda x,y: x if y.tag is None else x.replace(tag=y.tag)),
])

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ((nx := x.ins(RDNA3Ops.s_load_b32 if x.addrspace is AddrSpace.ALU else RDNA3Ops.s_load_b64)), [nx])),
  # Lower a gated load as one adjacent sequence after linearization so the alt initialization cannot escape its CFG block.
  (UPat(Ops.LOAD, src=(UPat(Ops.NOOP, name="addr"), UPat.var("alt"), UPat.var("gate")), name="x"), lower_gated_load),
  (UPat(Ops.LOAD, src=(UPat(Ops.NOOP, name="addr"),), name="x"), lambda x,addr: ((nx := x.ins(addr.arg).replace(src=addr.src)), [nx])),
  (UPat(Ops.STORE, src=(UPat(Ops.NOOP, name="addr"), UPat.var("val")), name="x"), lambda addr,x,val: ((nx := x.ins(addr.arg).replace(src=addr.src + (val,))), [nx])),
  # assign SGPRS exec masks to the linearized graph now that gated store (IF/ENDIFS) are present
  (UPat(Ops.IF, src=(UPat.var("gate"),), allow_any_len=True, name="x"), lambda ctx,x,gate: \
    ((nx := UOp(Ops.INS, arg=RDNA3Ops.s_and_saveexec_b32, src=(gate,), tag=ctx.vreg(GP_SGPRS))), [nx])),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.cvar("c")), name="x"), lambda x,buf,c:
    (x.replace(tag=(rdefs(buf)[c.val],)), []) if c.val < len(rdefs(buf)) else None),
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, src=(UPat(), UPat.var("acc"), UPat()), name="x"), lower_end),
  (UPat(Ops.ENDIF, src=(UPat.var("mif"),)), lambda mif: ((nx := restoreexec(mif)), [nx])),
  # slightly hacky, forces do_assemble in codegen but might hide incomplete lowering
  (UPat(GroupOp.All - {Ops.INS}, name="x"), lambda x: (x, [])),
])

# NOTE: hacky fixes, find cleaner way to conform to isa
def encode(ctx, x:UOp):
  import tinygrad.renderer.amd.dsl as dsl
  if x.arg in [RDNA3Ops.s_nop, RDNA3Ops.s_endpgm]: return x.replace(arg=x.arg())
  dmap = { "vcc" : dsl.VCC, "exec_lo" : dsl.EXEC_LO, "v" : dsl.v, "s" : dsl.s  }
  def _route(r:Register):
    assert isinstance(r, Register)
    return dmap[r.name] if r.name in dmap else dmap[r.name[0]]
  def _immorreg(x:UOp):
    while x.op is Ops.AFTER: x=x.src[0]
    return x.val if x.op is Ops.CONST else _fuse(rdefs(x))
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc, oprs = x.arg, x.arg.func, x.arg.args[0].name.lower(), x.src
  kw = args = None

  if group is RDNA3Ops.SMEM: kw = dict(sdata=_fuse(rdefs(x)), sbase=_fuse(rdefs(oprs[0])), soffset=dsl.NULL, offset=oprs[-1].arg)
  elif group is RDNA3Ops.SOPK: args = [dsl.NULL, oprs[0].arg]
  elif group is RDNA3Ops.SCRATCH:
    kw = dict(offset=_immorreg(oprs[0]))
    if rdef(x) is not None: kw["vdst"] = _fuse(rdefs(x))
    else: kw["data"] = _fuse(rdefs(oprs[1]))
  elif group is RDNA3Ops.GLOBAL:
    kw = dict(addr=_immorreg(oprs[0]))#,  offset=_immorreg(oprs[1]))
    if oprs[1].op is Ops.CONST: kw["offset"] = _immorreg(oprs[1])
    else: kw["saddr"] = _fuse(rdefs(oprs[1]))
    if rdef(x) is None: kw["data"]=_fuse(rdefs(oprs[2]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.DS:
    offs = _immorreg(oprs[1])
    kw = dict(addr=_immorreg(oprs[0]), offset0=offs&0xFF, offset1=offs>>8)
    if rdef(x) is None: kw["data0"]=_fuse(rdefs(oprs[3]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.VOP3SD: kw = dict(sdst=_immorreg(vccop), vdst=_fuse(rdefs(x)), **{f"src{i}":_immorreg(u) for i,u in enumerate(oprs[:3])})
  elif group is RDNA3Ops.VOPC: args = [_immorreg(u) for u in oprs]
  elif group is RDNA3Ops.VOP3P:
    kw = {f"src{i}":_immorreg(oprs[i]) for i in range(3)}
    kw["vdst"] = _fuse(rdefs(x))
    def _signed(dt:DType): return not (dtypes.is_unsigned(dt) or dtypes.is_float(dt))
    kw["neg"] = _signed(oprs[0].dtype) | (_signed(oprs[1].dtype) << 1)
  elif group in [RDNA3Ops.VOP3, RDNA3Ops.VOP2, RDNA3Ops.VOP1, RDNA3Ops.SOP1, RDNA3Ops.SOP2, RDNA3Ops.VOP3_SDST]: # alu
    if group in [RDNA3Ops.VOP1, RDNA3Ops.SOP1]: oprs = oprs[:1]
    if group in [RDNA3Ops.VOP2, RDNA3Ops.SOP2]: oprs = oprs[:2]
    args = [_fuse(rdefs(x))] + [_immorreg(u) for u in oprs]
  elif group is RDNA3Ops.SOPP: args = (oprs[0].val,) if len(oprs) > 0 and oprs[0].op is Ops.CONST else (0,)
  else: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={opc}")
  return x.replace(arg=(enc(**kw) if kw is not None else enc(*args)))

class CntType(Enum):
  DS_CNT = auto(); LOAD_CNT = auto(); STORE_CNT = auto()

  def get(u:UOp):
    if u.arg.func in { RDNA3Ops.GLOBAL, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }:
      return CntType.STORE_CNT if u.dtype is dtypes.void else CntType.LOAD_CNT
    if u.arg.func in { RDNA3Ops.SMEM, RDNA3Ops.DS }: return CntType.DS_CNT
    return None

@dataclass
class RDNA3LinearCtx:
  loop_label: dict[UOp, str] = field(default_factory=dict)
  # TODO: remove these
  exec_mask: dict[UOp, UOp] = field(default_factory=dict)
  range_bnd: dict[UOp, UOp] = field(default_factory=dict)

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  extra_matcher = extra_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.SHR, Ops.SHL, Ops.MAX)}
  post_regalloc_ctx = RDNA3LinearCtx()
  def __init__(self, target:Target):
    super().__init__(target)
    self.tensor_cores = tc.get_amd(target.arch)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s}
  def is_two_address(self, x:UOp) -> bool: return False
  def asm_str(self, uops:list[UOp], function_name:str) -> str: return ""

  # use scratch memory space for spilling thread local memory, addressed as:
  # SCRATCH_BASE + Swizzle(addr, tid) 12 bit ioffs, ensure no overflow!
  def spill(self, spill_offset:int, x:UOp) -> UOp:
    raise NotImplementedError()
    sz = x.dtype.itemsize # TODO: handle GROUP case
    opc = getattr(RDNA3Ops, f"scratch_store_b{sz*8}")
    ioffs = const(spill_offset, dtypes.uint32)
    return UOp(Ops.INS, arg=opc, src=(ioffs,x))

  def fill(self, spill_offset:int, x:UOp, regs:tuple[Register,...]) -> UOp:
    raise NotImplementedError()
    ioffs = const(spill_offset, dtypes.uint32)
    sz = x.dtype.itemsize
    suffix = "b" if sz > 2 else "u" if dtypes.is_unsigned(x.dtype) or dtypes.is_float(x.dtype) else "i"
    opc = getattr(RDNA3Ops, f"scratch_load_{suffix}{sz*8}")
    return UOp(Ops.INS, x.dtype, arg=opc, src=(ioffs,), tag=regs)

  def copy(self, u:UOp, r:VRegister|Register) -> UOp:
    if u.dtype.itemsize == 8:
      return UOp.group(vmov(u.index(0), r.sub(0)), vmov(u.index(1), r.sub(1)), dtype=u.dtype, tag=(r,))
    return vmov(u,r)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    deps: set[Register] = set()
    nuops = []
    # s_waitcnt
    for u in lin.src:
      if any(r in deps for s in u.src for r in rdefs(s)):
        nuops.append(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt, src=(const(0, dtypes.uint16),)))
        deps.clear()
      if (tp := CntType.get(u)) is not None and tp in [CntType.DS_CNT, CntType.LOAD_CNT]:
        deps.update(rdefs(u))
      nuops.append(u)

    # s_clause
    # NOTE: do the grouped instructions need to share src?
    loads: dict[int, UOp] = {}
    stores: dict[int, UOp] = {}
    for i,u in enumerate(nuops):
      if u.arg.func is RDNA3Ops.GLOBAL:
        if u.dtype is dtypes.void: stores[i] = u
        else: loads[i] = u

    def gather(instances:dict[int, UOp]) -> dict[int, int]:
      clauses: dict[int, int] = {}
      start, last = None, None
      for k in sorted(instances.keys()):
        if last is not None and k > last+1: last, start = None, None
        if start is None: start = k
        last = k
        clauses[start] = clauses.setdefault(start, 0) + 1
      return clauses

    dp = 0
    for p,l in (gather(loads) | gather(stores)).items():
      if l > 1:
        nuops.insert(p+dp, UOp(Ops.INS, arg=RDNA3Ops.s_clause, src=(const(l-1),)))
        dp += 1

    pc = 0
    targets: dict[str, int] = {}
    upc: dict[UOp, int] = {}
    uops = nuops.copy()
    nuops = []
    for u in uops:
      if u.arg is RDNA3Ops.s_nop and isinstance(u.tag, str): targets[u.tag] = pc
      else:
        upc[u] = pc = pc + (u := encode(self,u)).arg.size()
        nuops.append(u)

    lin = lin.replace(src=tuple([u if not isinstance(u.tag, str) else \
      u.replace(arg=RDNA3Ops.SOPP(u.arg.op, (targets[u.tag] - upc[u]) // 4)) for u in nuops]))
    return assemble_linear(prg, lin, self.target.arch, scratch_size=self.spill_size)

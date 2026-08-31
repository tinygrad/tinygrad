from tinygrad.dtype import dtypes, AddrSpace, truncate, DType, InvalidType, to_storage_scalar, ConstFloat
from tinygrad.codegen.opt import tc
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg, range_str, GroupOp, graph_rewrite
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef, PreRegallocContext
from tinygrad.renderer.cstyle import create_non_native_float_pats, pm_manual_bf16_cast
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
from tinygrad.codegen.decomp.op import fast_idiv
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.cstyle import HIPRenderer
import tinygrad.renderer.amd.dsl as dsl
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
import itertools, functools, struct, math
from dataclasses import dataclass, field
from enum import Enum, auto

# ---- (UOp, dtype) -> Instruction tables ----
dt_to_isa = { dtypes.int32:"i32", dtypes.uint32:"u32", dtypes.float32:"f32", dtypes.float64:"f64", dtypes.float16:"f16", dtypes.int16:"i16", dtypes.uint16:"u16", dtypes.uint64:"u64", dtypes.int64:"i64", dtypes.bfloat16:"bf16", dtypes.uint8:"u8", dtypes.int8:"i8" }
isa_to_dt = { v:k for k,v in dt_to_isa.items() }

# (uop, prefix, opcodes, support 32 and 64 bit encoding (e32/e64 branches with keys))
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
V_FMA = { dtypes.float16:RDNA3Ops.v_fma_f16, dtypes.float32:RDNA3Ops.v_fma_f32, dtypes.float64:RDNA3Ops.v_fma_f64, dtypes.uint64:RDNA3Ops.v_mad_u64_u32, dtypes.int64:RDNA3Ops.v_mad_i64_i32 }
V_LSHL = { 2:RDNA3Ops.v_lshlrev_b16, 4:RDNA3Ops.v_lshlrev_b32_e32, 8:RDNA3Ops.v_lshlrev_b64 }
V_LSHR = { 2:RDNA3Ops.v_lshrrev_b16, 4:RDNA3Ops.v_lshrrev_b32_e32, 8:RDNA3Ops.v_lshrrev_b64 }
V_ASHR = { 4:RDNA3Ops.v_ashrrev_i32_e32, 8:RDNA3Ops.v_ashrrev_i64 }
S_CMP = { Ops.CMPNE:RDNA3Ops.s_xor_b32, Ops.XOR:RDNA3Ops.s_xor_b32, Ops.OR: RDNA3Ops.s_or_b32, Ops.AND:RDNA3Ops.s_and_b32, Ops.CMPLT: RDNA3Ops.s_and_not1_b32, Ops.CMPEQ:RDNA3Ops.s_xnor_b32 }

# ---- helpers ----
lane_ctr = itertools.count()
def def_reg(dt, reg:Register|tuple[Register,...]):
  return UOp.placeholder((1,), dt, next(lane_ctr), AddrSpace.REG).replace(tag=(reg,) if isinstance(reg,Register) else reg)
def const(v, dt:DType=dtypes.uint32) -> UOp: return UOp.cconst((v if isinstance(v, InvalidType) else truncate[dt](v)), dt).rtag()
def gep(u:UOp, i:int) -> UOp: return u.bitcast(dtypes.uint32).index(UOp.cconst(i, dtypes.uint32))
def const_val(x:UOp):
  strong = x.dtype
  while x.op is not Ops.CONST: x = x.src[0]
  if isinstance(x.val, ConstFloat) and strong is dtypes.half: return struct.unpack('H', struct.pack('e', x.val))[0]
  return x.val
def is_const(x:UOp): return is_const(x.src[0]) if x.op in {Ops.CAST, Ops.BITCAST, Ops.AFTER} else x.op is Ops.CONST
def to_vgpr(x:UOp) -> UOp: return vmov(x) if is_const(x) else x
def smux(dt:DType, sdt:DType, udt:DType): return udt if dtypes.is_unsigned(dt) else sdt
def vmov(x:UOp, r:VRegister|Register|None=None) -> UOp:
  if isinstance(r, VRegister): assert r.width == 1
  nx = x.ins(RDNA3Ops.v_mov_b16_e64 if x.dtype is dtypes.half else RDNA3Ops.v_mov_b32_e32, src=(x,))
  return nx.rtag() if r is None else nx.replace(tag=(r,))
def restoreexec(mask:UOp) -> UOp: return UOp(Ops.INS, src=(execop,mask), arg=(RDNA3Ops.s_or_b32, dtypes.void), tag=(EXEC,))
def label(ctx, name:str) -> UOp: return UOp(Ops.INS, arg=(RDNA3Ops.s_nop, dtypes.void), tag=name)
def rafter(x:UOp, bitcast=False) -> UOp:
  return rafter(x.src[0]) if x.op in ({Ops.AFTER, Ops.BITCAST} if bitcast else {Ops.AFTER}) else x
def multireg(*src, dtype: DType, vr:VRegister|None=None) -> UOp:
  # stack of 32 bit register values/value producing instructions
  # grouped by order to be assigned contiguous register slice
  return UOp(Ops.STACK, src=tuple(s.bitcast(dtypes.uint32) for s in src), tag=(vr,) if vr else None).bitcast(dtype)
# TODO: only expand to double vgpr if not used as src and not foldable?
# - realize into registers on demand if in stack or in to_vgpr
def const64(x:UOp, c:UOp):
  v = c.val.bits if dtypes.is_float(x.dtype) else c.val
  return multireg(vmov(const(v)), vmov(const(v >> 32)), dtype=x.dtype)

# ---- register classes/ABI regs ---
VGPRS = tuple(Register(f"v{i}", i, size=4) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i, size=4) for i in range(106))
GP_SGPRS = tuple(SGPRS[5:])
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
VCC, EXEC = Register("vcc", 0, size=4), Register("exec_lo", 0, size=4)
execop, vccop, kernarg_ptr = def_reg(dtypes.uint32, EXEC), def_reg(dtypes.uint32, VCC), def_reg(dtypes.uint64, KERNARG_PTR)

# ---- register granularity/serialization helpers ----
def packb16(lo:UOp, hi:UOp):
  if dtypes.is_float(lo.dtype): return lo.ins(RDNA3Ops.v_pack_b32_f16, dtype=dtypes.uint32, src=(lo,hi))
  lo, hi = lo.cast(dtypes.uint32), hi.cast(dtypes.uint32)
  return UOp(Ops.INS, arg=(RDNA3Ops.v_bfi_b32, dtypes.uint32), src=(const(0xFFFF), lo, hi << 16))

def stack2regs(ctx, x:UOp):
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
      for j in range(1, min(4, len(x.src)-i*4)): out = out | _pk(j)
      mvs.append(out)
  return multireg(*mvs, dtype=x.dtype) if len(mvs) > 1 else mvs[0].bitcast(x.dtype)

def unpack(buf:UOp, idx:UOp, c:UOp):
  if rafter(buf).op in {Ops.BUFFER, Ops.PARAM}: return None
  opc, sz = RDNA3Ops.v_bfe_i32 if idx.dtype in dtypes.sints else RDNA3Ops.v_bfe_u32, idx.dtype.itemsize
  return UOp(Ops.INS, arg=(opc, idx.dtype), src=(gep(buf, (c.val*sz)//4), const((c.val%(4//sz))*sz*8), const(sz*8)))

# ---- operand legalization wrappers ----
def can_fold_lit(c:UOp):
  v = const_val(rafter(c))
  # ConstFloat hashes by bits so it never hits _FLOAT_ENC, treat it as a literal
  if isinstance(v, float): return not isinstance(v, ConstFloat) and v in dsl.SrcField._FLOAT_ENC
  if isinstance(v, int): return 0 <= v <= 64 or -16 <= v < 0
  return False

def lvop3(x:UOp):
  lits = [s for s in x.src if is_const(s) and not can_fold_lit(s)]
  return None if len(lits) == 1 else x.replace(src=tuple([vmov(s) if s in lits[1:] else s for s in x.src]))

rev_op_order = { RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshlrev_b16, RDNA3Ops.v_lshlrev_b64, RDNA3Ops.v_lshrrev_b32_e32, RDNA3Ops.v_lshrrev_b16, RDNA3Ops.v_lshrrev_b64, RDNA3Ops.v_ashrrev_i32_e32, RDNA3Ops.v_ashrrev_i64 }
commutative_ins = {i for op in (Ops.ADD, Ops.MUL, Ops.MAX) for i in OP_INS[op].values()}
def lvop2(x:UOp, swap_only=False):
  if not is_const(x.src[1]): return None # TODO: should check positive vgpr, sgpr cant be used in vrsc1
  rest = x.src[2:] if len(x.src) > 2 else ()
  non_commutative = x.arg[0] in set(OP_INS[Ops.SUB].values()) | rev_op_order
  if not non_commutative and not is_const(x.src[0]): return x.replace(src=(x.src[1], x.src[0]) + rest)
  # VOP3 encodes a const in src1 fine, it only ever wants the commutative swap above, never the vmov
  return None if swap_only else x.replace(src=(x.src[0], vmov(x.src[1])) + rest)

def alloc_vregs(ctx, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], VRegister): return None

  if isinstance(x.tag, tuple): cons, width = x.tag if isinstance(x.tag[0], tuple) else (x.tag, 1)
  else: cons, width = ctx.gp_vgprs, ((x.dtype.itemsize+3) // 4) * (len(x.src) if x.op is Ops.STACK else 1)
  return x.replace(tag=(ctx.ren.vreg(cons, width=width),))

def abi(ctx, x:UOp) -> UOp|None:
  if x.tag is True: return None

  if x.op is Ops.SPECIAL:
    if x.arg[0] == 'g': return vmov(def_reg(dtypes.uint32, WGIDS[int(x.arg[-1])])).after(x.rtag())
    return x.ins(RDNA3Ops.v_bfe_u32, dtype=dtypes.uint32, src=(def_reg(dtypes.uint32, WIIDS), const(10*int(x.arg[-1])), const(10)))

  # NOTE: carries PARAM op through meta src edge to preserve program info
  offs = const(sum(8 if u.op == Ops.PARAM else u.dtype.itemsize for u in ctx.func_args[:ctx.func_args.index(x)]))
  src = (kernarg_ptr, offs, x.rtag())
  if x.addrspace is AddrSpace.ALU: return vmov(UOp(Ops.INS, src=src, arg=(RDNA3Ops.s_load_b32, x.dtype), tag=GP_SGPRS))
  vr = ctx.ren.vreg(GP_SGPRS, width=2, alignment=2)
  return UOp(Ops.INS, src=src, arg=(RDNA3Ops.s_load_b64, dtypes.ulong), tag=(vr,))

# ----- memory access ----
def fold_global(base:UOp, idx:UOp):
  disp_scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(disp_scale.bit_length() - 1, dtypes.int32)
  vaddr, offs = idx, const(0, dtypes.uint16)
  def foldable(v:int) -> bool: return -(1 << 12) <= v < (1 << 12)
  if idx.op is Ops.ADD and is_const(idx.src[1]) and foldable((_offs := idx.src[1].src[0].val * disp_scale)):
    vaddr, offs = idx.src[0], const(_offs, dtypes.int16)
    if shft.src[0].val > 0: vaddr <<= shft
    vaddr = int_to_int64(vaddr, dtypes.uint64)
    return (vaddr + base.bitcast(dtypes.uint64), offs)
  else:
    vaddr = to_vgpr(vaddr)
    if shft.src[0].val > 0: vaddr <<= shft
    return (vaddr, base)

# TODO: actually calculate lds offset per seperate BUFFER, (ctx.func_args)
def fold_lds(base:UOp, idx:UOp): # (vaddr, ioffs)
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(scale.bit_length() - 1, dtypes.int32)
  # offset0/offset1 form one 16 bit unsigned immediate, anything outside that has to stay in the addr vgpr
  def foldable(v:int) -> bool: return 0 <= v < (1 << 16)
  if is_const(idx) and foldable(v := const_val(idx) * scale): return (vmov(const(0)), const(v, dtypes.uint16), base)
  if idx.op is Ops.ADD and is_const(idx.src[1]) and foldable(v := const_val(idx.src[1]) * scale):
    return ((idx.src[0] << shft if scale > 1 else idx.src[0]).cast(dtypes.uint32), const(v, dtypes.uint16), base)
  return (idx << shft, const(0, dtypes.uint16), base)

def fold_address(x:UOp): return fold_lds(*x.src[:2]) if x.addrspace is AddrSpace.LOCAL else fold_global(*x.src[:2])

def load(ctx, x:UOp, idx:UOp):
  n = idx.src[-1].src[0].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.src[0].dtype.itemsize
  suffix = "b" if sz > 2 else "i" if x.dtype in dtypes.sints else "u"
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_load_{suffix}{sz*8}")
  ctx.ren.semantic_op[opc]=Ops.LOAD
  return x.ins(opc, src=fold_address(rafter(idx, True))+x.src[1:], tag=(ctx.ren.vreg(ctx.gp_vgprs, width=(sz+3)//4),))

def store(ctx, x:UOp, idx:UOp, val:UOp):
  n = idx.src[-1].src[0].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.dtype.itemsize
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_store_b{sz*8}")
  ctx.ren.semantic_op[opc]=Ops.STORE
  return x.ins(opc, src=fold_address(rafter(idx, True))+(to_vgpr(val),*x.src[2:]))

def lower_gated_load(ctx, x:UOp):
  alt, gate = x.src[-2:]
  mask = gate.ins(RDNA3Ops.s_and_saveexec_b32, dtype=dtypes.uint32, src=(gate,), tag=(ctx.ren.vreg(GP_SGPRS),))
  x = x.replace(src=x.src[:-2])
  return x, ctx.ren.vcopy(alt, rdef(x))[1] + [mask, x, restoreexec(mask)]

def lower_gated_store(ctx, x:UOp):
  mask = x.src[-1].ins(RDNA3Ops.s_and_saveexec_b32, dtype=dtypes.uint32, src=(x.src[-1],), tag=(ctx.ren.vreg(GP_SGPRS),))
  x = x.replace(src=x.src[:-1])
  return x, [mask, x, restoreexec(mask)]

# ------ ALU ------
# TODO: remove this, run const64 at isel time fix f64 test that breaks
def as_u64(u:UOp) -> UOp:
  return const64(u.cast(dtypes.uint64), rafter(u, True)) if is_const(u) else u.cast(dtypes.uint64)

def arith64(ctx, x:UOp):
  a, b = x.src
  a, b = as_u64(a), as_u64(b)
  ins_lo = RDNA3Ops.v_add_co_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_u32
  ins_hi = RDNA3Ops.v_add_co_ci_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_ci_u32
  lo = UOp(Ops.INS, src=(gep(a,0), gep(b,0)), arg=(ins_lo, dtypes.uint32))
  hi = UOp(Ops.INS, src=(gep(a,1), gep(b,1), vccop, lo), arg=(ins_hi, dtypes.uint32))
  hi = hi.after(lo)
  return multireg(lo, hi, dtype=x.dtype)

def mul64(ctx, x:UOp):
  def _mad(a:UOp, b:UOp, c:UOp=const(0, dtypes.uint64)): return UOp(Ops.MULACC, src=(a, b, c))
  a, b = x.src
  a, b = as_u64(a), as_u64(b)
  p1 = _mad(gep(a,1), gep(b,0)).bitcast(dtypes.uint64) << 32
  p2 = _mad(gep(a,0), gep(b,1)).bitcast(dtypes.uint64) << 32
  return _mad(gep(a,0), gep(b,0), p1 + p2).bitcast(x.dtype)

def mulhi32(a:UOp, b:UOp) -> UOp: return ((a.cast(dtypes.uint64) * b.cast(dtypes.uint64)) >> 32).cast(dtypes.uint32)
	# return UOp(Ops.INS, dtypes.uint32, src=(a,b), arg=RDNA3Ops.v_mul_hi_u32)
def mulhi64(a:UOp, b:UOp) -> UOp:
  def mul32(a:UOp, b:UOp) -> UOp: return multireg(a*b, mulhi32(a,b), dtype=dtypes.uint64)
  a0, a1, b0, b1 = gep(a,0), gep(a,1), gep(b,0), gep(b,1)
  t = mul32(a1,b0) + mulhi32(a0,b0).cast(dtypes.uint64)
  return mul32(a1,b1) + (t >> 32) + ((mul32(a0,b1) + t.cast(dtypes.uint32).cast(dtypes.uint64)) >> 32)

def idiv32(x:UOp, a:UOp, b:UOp) -> UOp:
  if (signed := not dtypes.is_unsigned(x.dtype)):
    # sign extend to 32 bits first, abs() of a narrower type overflows at its min (e.g. abs(int16 -32768))
    a, b = a.cast(dtypes.int32), b.cast(dtypes.int32)
    s = ((a ^ b) >> UOp.const(31, dtypes.int32)).bitcast(dtypes.uint32)
    a, b = a.abs(), b.abs()
  a, b = a.cast(dtypes.uint32), b.cast(dtypes.uint32)
  z = (b.float().reciprocal() * UOp.const(2**32 - 256, dtypes.float32)).cast(dtypes.uint32)
  # NOTE: bitcast to int32 so mulhi32 sign-extends: the Newton correction 2**32 - b*z is signed
  z = z + mulhi32(z, (b*z).bitcast(dtypes.int32).neg())
  q = mulhi32(a, z)
  r = a - q*b
  q, r = (r < b).where(q, q + 1), (r < b).where(r, r - b)
  q, r = (r < b).where(q, q + 1), (r < b).where(r, r - b)
  if signed: q = (q ^ s) - s
  return q.cast(x.dtype)

def idiv64(x:UOp, a:UOp, b:UOp) -> UOp:
  if (signed := not dtypes.is_unsigned(x.dtype)):
    sa, sb = (a.bitcast(dtypes.int64) >> 63).bitcast(dtypes.uint64), (b.bitcast(dtypes.int64) >> 63).bitcast(dtypes.uint64)
    a, b, s = (a.cast(dtypes.uint64)^sa)-sa, (b.cast(dtypes.uint64)^sb)-sb, sa^sb
  lo, hi = b.cast(dtypes.uint32), (b >> 32).cast(dtypes.uint32)
  m = (hi.float()*UOp.const(2**32, dtypes.float32)+lo.float()).reciprocal() * UOp.const(2**64-2**42, dtypes.float32)
  t = (m * UOp.const(2**-32, dtypes.float32)).alu(Ops.TRUNC)
  z = (t.cast(dtypes.uint32).cast(dtypes.uint64) << 32) | (t*UOp.const(-2**32, dtypes.float32)+m).cast(dtypes.uint32).cast(dtypes.uint64)
  for _ in range(2): z += mulhi64(z, b.const_like(0).alu(Ops.SUB, b)*z)
  q = mulhi64(a, z)
  r = a - b*q
  for _ in range(2): q, r = (c:=(r < b).logical_not()).where(q+1, q), c.where(r-b, r)
  if signed: q = (q ^ s).alu(Ops.SUB, s)
  return q.cast(x.dtype)

def bitwise64(ctx, x:UOp, ins):
  a, b = x.src
  lo = UOp(Ops.INS, src=(gep(a,0), gep(b,0)), arg=(ins, dtypes.uint32))
  hi = UOp(Ops.INS, src=(gep(a,1), gep(b,1)), arg=(ins, dtypes.uint32))
  return multireg(lo, hi, dtype=x.dtype)

def render_wmma(ctx, wmma:UOp):
  a,b,acc = wmma.src
  srcdt = dt_to_isa[wmma.arg[1]]
  if wmma.arg[1] in dtypes.int8s: srcdt = "iu8"
  ins = getattr(RDNA3Ops, f"v_wmma_{dt_to_isa[wmma.dtype]}_16x16x16_{srcdt}")
  return UOp(Ops.INS, src=(a,b,acc), arg=(ins, wmma.dtype), tag=(ctx.ren.vreg(ctx.gp_vgprs, width=8),))

# ---- casting utilities -----
def int_to_int64(y:UOp, tdt:DType):
  hi = vmov(const(0)) if dtypes.is_unsigned(y.dtype) else to_vgpr(y) >> max(y.dtype.itemsize*8, 32)-1
  return multireg(vmov(y), hi, dtype=tdt)

def f64_to_i64(y:UOp, tdt:DType):
  hi_dt = smux(tdt, dtypes.int32, dtypes.uint32)
  tr = y.trunc()
  hi_f = tr.ins(RDNA3Ops.v_ldexp_f64, src=(tr,const(-32)))
  hi_f = UOp(Ops.INS, src=(hi_f,), arg=(RDNA3Ops.v_floor_f64_e32, dtypes.float64))
  lo_f = hi_f.ins(RDNA3Ops.v_ldexp_f64, src=(hi_f, const(32)))
  lo_f = tr + lo_f * const(-1., dtypes.float64)
  return multireg(lo_f.cast(dtypes.uint32), hi_f.cast(hi_dt), dtype=tdt)

# TODO: automatically fuse f64 * 2^n -> ldexp
def i64_to_f64(x:UOp):
  lo = gep(x, 0).double()
  hi = gep(x, 1).bitcast(smux(x.dtype, dtypes.int, dtypes.uint)).double()
  hi = hi.ins(RDNA3Ops.v_ldexp_f64, src=(hi,const(32)))
  return lo + hi

# ---- control flow ----
def lower_range(ctx, x:UOp):
  if x.src[0].op is Ops.NOOP: return x, [label(ctx, f".LOOP_BODY_{range_str(x)}")] # loop
  acc = x.ins(RDNA3Ops.v_mov_b32_e32, src=(const(0),))
  ctx.loop_label[acc] = range_str(x)
  return acc, [acc, label(ctx, f".LOOP_BODY_{range_str(x)}")]

def lower_end(ctx, x:UOp):
  if x.src[-3].src[0].op is Ops.NOOP: # loop
    rng, pred, mask = x.src[-3:]
    jmp = UOp(Ops.INS, arg=(RDNA3Ops.s_cbranch_execnz, dtypes.void), tag=f".LOOP_BODY_{range_str(rng)}")
    pred = UOp(Ops.INS, src=(pred,), arg=(RDNA3Ops.s_mov_b32, dtypes.void), tag=(EXEC,))
    return pred, [pred, jmp, restoreexec(mask)]
  else:
    acc,bnd,mask = x.src[-3:]
    loop_end = label(ctx, f".LOOP_END_{ctx.loop_label[acc]}")
    inc = acc.ins(RDNA3Ops.v_add_nc_u32_e32, src=(const(1), acc), tag=acc.tag)
    jmp = UOp(Ops.INS, arg=(RDNA3Ops.s_cbranch_execnz, dtypes.void), tag=f".LOOP_BODY_{ctx.loop_label[acc]}")
    pred = UOp(Ops.INS, src=(acc,bnd), arg=(RDNA3Ops.v_cmpx_lt_u32_e64, dtypes.void), tag=(EXEC,))
    return inc, [inc, pred, jmp, loop_end, restoreexec(mask)]

# ---- lowering passes ----
int1regs = dtypes.int8s + dtypes.int16s + dtypes.int32s
extra_matcher = PatternMatcher([
  # NOTE: runs before casted const
  (UPat.cvar("c", dtypes.bfloat16), lambda c: UOp.const(c.val if isinstance(c.val, InvalidType) else
    to_storage_scalar(c.val, dtypes.bfloat16), dtypes.uint16).bitcast(dtypes.bfloat16)),
  # NOTE: DISABLE_FAST_IDIV=1 by default, copy patterns here
  (UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar("d"))),
    lambda ctx,x,d: fast_idiv(ctx, x, d.val) if x.vmin >= 0 or x.dtype in dtypes.uints else None),
  (UPat(Ops.CMOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - b * a.alu(Ops.CDIV, b)),
  (UPat(Ops.CDIV, dtypes.int32s+dtypes.int16s+dtypes.int8s, (UPat.var("a"), UPat.var("b")), name="x"), idiv32),
  (UPat(Ops.EXP2, dtypes.double, src=(UPat.var("d"),)), xexp2),
  (UPat(Ops.LOG2, dtypes.double, src=(UPat.var("d"),)), xlog2),
]) + pm_manual_bf16_cast + create_non_native_float_pats((dtypes.bfloat16,)) + tc.pm_validate_wmma_rdna3

pm_float_to_int = PatternMatcher([
  (UPat.var("y", dtypes.half).cast((dtypes.double,)+dtypes.int32s+dtypes.int64s, name="x"),
    lambda y,x: y.float().cast(x.dtype)),
  (UPat.var("y", (dtypes.float32, dtypes.half)).cast(dtypes.ints, name="x"),
    lambda y,x: y.cast(smux(x.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.double).cast(dtypes.int16s+dtypes.int8s, name="x"),
    lambda y,x: y.float().cast(smux(x.dtype, dtypes.int32, dtypes.uint32)).bitcast(x.dtype)),
  (UPat.var("y", dtypes.double).cast(dtypes.int64s).named("x"), lambda y,x: f64_to_i64(y, x.dtype)),
])

pm_int_to_float = PatternMatcher([
  (UPat.var("y", dtypes.int32s).cast(dtypes.half), lambda y: y.float().half()),
  (UPat.var("y", dtypes.int8s).cast(dtypes.half),
    lambda y: y.cast(smux(y.dtype, dtypes.int16, dtypes.uint16)).half()),
  (UPat.var("y", dtypes.int8s+dtypes.int16s).cast((dtypes.float32,dtypes.double), name="x"),
    lambda y,x: y.cast(smux(y.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.int64s).cast((dtypes.float, dtypes.half), name="x"),
    lambda y,x: y.double().float().cast(x.dtype)),
  (UPat.var("x", dtypes.int64s).cast(dtypes.float64), i64_to_f64),
])

pre_isel_matcher = PatternMatcher([
  # --- bools are lane masks ---
  (UPat(Ops.STORE, src=(UPat.var("buf"), UPat.var("val", dtype=dtypes.bool)), allow_any_len=True, name="x"), \
    lambda buf,val,x: x.replace(src=(buf,val.cast(dtypes.uint32)) + x.src[2:])),
  (UPat(Ops.LOAD, dtypes.bool, src=(UPat.var("buf"),), allow_any_len=True, name="x"),
    lambda buf,x: x.replace(src=(buf.bitcast(dtypes.uchar),) +
      (() if len(x.src) == 1 else (x.src[1].cast(dtypes.uchar), x.src[2]))).cast(dtypes.bool)),
  # --- int8 alu is int16 for now ---
  (UPat(GroupOp.ALU-{Ops.WHERE}, dtypes.int8s, name="x"),
    lambda x: (upcast := tuple(s.cast(smux(x.dtype, dtypes.int16, dtypes.uint16)) for s in x.src))[0].alu(x.op, *upcast[1:]).bitcast(x.dtype)),
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.int8s), UPat()), name="x"),
    lambda x: x.replace(src=tuple(s.cast(smux(x.dtype, dtypes.int16, dtypes.uint16)) for s in x.src))),
  # -- int -> int casts ---
  (UPat.var("y", dtypes.int64s).cast(int1regs, name="x"), lambda y,x: gep(y, 0).bitcast(x.dtype)),
  (UPat.var("y", int1regs).cast(dtypes.int64s, name="x"), lambda y,x: int_to_int64(y, x.dtype)),
  (UPat.var("y", dtypes.double).cast(dtypes.half), lambda y: y.float().half()),
  # --- other ---
  # prevent 64 bit shift from being realized into 2 regs
  (UPat((Ops.SHR, Ops.SHL), src=(UPat.var("val"), UPat.var("n", dtypes.int64s+(dtypes.float64,))), name="x"),
    lambda val,x,n: x.replace(src=(val, n.cast(dtypes.uint32)))),
  (UPat(Ops.INDEX, (dtypes.half,dtypes.bfloat16)+dtypes.int8s+dtypes.int16s,
    src=(UPat.var("buf"), UPat.cvar("c").cast()), name="idx"), unpack),
  (UPat(Ops.CDIV, dtypes.int64s, (UPat.var("a"), UPat.var("b")), name="x"), idiv64),
  (UPat(Ops.MUL, (dtypes.int16,dtypes.int32), src=(UPat.var("a"), UPat.var("b")), name="x"), lambda a,b,x:
    (a.cast(dtypes.uint32) * b.cast(dtypes.uint32)).cast(x.dtype)),
  (UPat(Ops.MAX, dtypes.int16s, name="x"), lambda x:
    (upcast := tuple(s.cast(smux(x.dtype, dtypes.int32, dtypes.uint32)) for s in x.src))[0].alu(Ops.MAX, *upcast[1:]).bitcast(x.dtype)),
  (UPat.cvar("x").cast(dtypes.bool), lambda x: x.ins(RDNA3Ops.s_mov_b32, src=(const((1 << 32) - 1 if x.val else 0),), tag=GP_SGPRS)),
  (UPat.var("x").cast(dtypes.bool), lambda x: x.alu(Ops.CMPEQ, const(1, x.dtype))),
  (UPat.cvar("c").cast((dtypes.float64,)+dtypes.int64s, name="x"), const64),
  (UPat.var("y", dtypes.bool).cast(name="x"), lambda y,x: y.where(const(1, x.dtype), const(0, x.dtype))),
  (UPat(Ops.STACK, name="x"), lambda x: x.replace(src=tuple(vmov(s) if is_const(s) and s.dtype.itemsize < 8 else s for s in x.src))
    if any(is_const(s) for s in x.src) else None),
  (UPat(Ops.MAX, dtypes.int64s, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: (a < b).where(b, a)),
  (UPat(Ops.MULACC, dtypes.ints, src=(UPat.var("a"), UPat.var("b"), UPat.var("c"))), lambda a,b,c: a*b + c),
]) + pm_float_to_int + pm_int_to_float

isel_matcher = PatternMatcher([
  # --- control flow ---
  (UPat(Ops.RANGE, name="rng"), lambda ctx,rng:
    rng.replace(src=rng.src + (execop.ins(RDNA3Ops.s_mov_b32, dtype=dtypes.uint32, src=(execop,), tag=GP_SGPRS),),
    tag=ctx.ren.vreg(ctx.gp_vgprs)) if rng.tag is None else None),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng"), UPat()), name="x"),
    lambda x,rng: x.replace(src=(x.src[0],rng,x.src[-1],rng.src[-1])) if rng.tag is not None else None),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng")), name="x"), \
    lambda x,rng: x.replace(src=(x.src[0],rng,rng.src[0],rng.src[-1])) if rng.tag is not None else None),
  # --- double precision alu ---
  (UPat(Ops.MUL, dtypes.int64s, name="x"), mul64),
  (UPat((Ops.ADD, Ops.SUB), dtypes.int64s, name="x"), arith64),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), dtypes.int64s+(dtypes.float64,), name="x"),
    lambda ctx,x: bitwise64(ctx, x, getattr(RDNA3Ops, f"v_{x.op.name.lower()}_b32_e32"))),
  # --- operator fusion ---
  (UPat().sqrt().named("x").reciprocal(), lambda x: x.ins(V_RSQ[x.dtype]) if x.dtype in V_RSQ else None),
  (UPat(Ops.MULACC, dtypes.floats+dtypes.int64s, name="x"), lambda x: x.ins(V_FMA[x.dtype])),
  (UPat(Ops.ADD, dtypes.uint32, src=(UPat(Ops.ADD, name="y"), UPat.var("b")), name="x"),
    lambda ctx,x,y,b: x.ins(RDNA3Ops.v_add3_u32, src=y.src + (b,))),
  # --- general alu ---
  (UPat(Ops.SHR, name="x"), lambda x: x.ins(V_LSHR[max(2, x.dtype.itemsize)] \
    if dtypes.is_unsigned(x.dtype) else V_ASHR[max(4, x.dtype.itemsize)],
    src=x.src[2::-1])),
  (UPat(Ops.SHL, name="x"), lambda x: x.ins(V_LSHL[max(2, x.dtype.itemsize)], src=x.src[2::-1])),
  (UPat(GroupOp.Comparison|{Ops.XOR, Ops.AND, Ops.OR}, dtypes.bool, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
    lambda a,b,x: x.ins(S_CMP[x.op], src=(b,a) if x.op is Ops.CMPLT else (a,b), tag=GP_SGPRS)),
  (UPat(GroupOp.Comparison|{Ops.XOR, Ops.AND, Ops.OR}, dtypes.bool, name="x"), lambda x:
    x.ins(OP_INS[x.op][64][rafter(x.src[0]).dtype], tag=GP_SGPRS)),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), name="x"), lambda ctx,x: x.ins(getattr(RDNA3Ops, f"v_{x.op.name.lower()}_b32_e32"))),
  (UPat(Ops.WHERE, dtypes.bool, src=(UPat.var("mask"), UPat.var("a"), UPat.var("b")), name="x"),
    lambda mask,a,b,x: (mask & a) | (~mask & b)),
  (UPat(Ops.WHERE, src=(UPat.var("pred"), UPat.var("a", dtype=dtypes.int64s+(dtypes.float64,)), UPat.var("b"))),
    lambda pred,a,b: multireg(pred.where(gep(a,0),gep(b,0)), pred.where(gep(a,1), gep(b,1)), dtype=a.dtype)),
  (UPat.var("pred").where(UPat.var("a"), UPat.var("b")).named("x"), lambda pred,a,b,x:
    x.ins(RDNA3Ops.v_cndmask_b32_e64 if x.dtype.itemsize >= 4 else RDNA3Ops.v_cndmask_b16, src=(b,a,pred))),
  (UPat(GroupOp.Binary|GroupOp.Unary, name="x"), lambda x: x.ins(OP_INS[x.op][x.dtype])),
  (UPat(Ops.WMMA, name="wmma"), render_wmma),
  # --- casting ---
  (UPat.var("y", dtypes.ints).cast(dtypes.ints).named("x"), lambda y,x: y.bitcast(x.dtype)
    if y.dtype.itemsize >= x.dtype.itemsize else
    x.ins(RDNA3Ops.v_bfe_i32 if y.dtype in dtypes.sints else RDNA3Ops.v_bfe_u32, src=(y,const(0),const(y.dtype.itemsize*8)))),
  # NOTE: dont realize weak casts
  (UPat.var("y", dtype=dtypes.ints+dtypes.floats).cast(name="x"),
    lambda y,x: x.ins(getattr(RDNA3Ops, f"v_cvt_{dt_to_isa[x.dtype]}_{dt_to_isa[y.dtype]}_e64"))),
  # --- mem ops ---
  (UPat.var("idx").store(UPat.var("val"), allow_any_len=True).named("x"), lambda ctx,x,idx,val:
    store(ctx,x,idx,val) if idx.addrspace is not AddrSpace.REG else
    x.replace(tag=(ctx.ren.vreg(ctx.gp_vgprs, width=(idx.dtype.itemsize+3)//4),)) if x.tag is None else None),
  (UPat.var("idx").load(name="x", allow_any_len=True), lambda ctx,x,idx:
    load(ctx,x,idx) if idx.addrspace is not AddrSpace.REG else None),
  # --- other ---
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), abi),
  (UPat((Ops.INS, Ops.STACK), name="x"), alloc_vregs),
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  (UPat(Ops.STACK, name="x"), lambda ctx,x: stack2regs(ctx, x) if len(x.src) and x.dtype.itemsize < 4 else None),
  # NOTE: commutative ALU that lands in a VOP3 encoding (v_add_nc_i32, v_mul_lo_u32) needs same legalization
  (UPat(Ops.INS, name="x"), lambda x: lvop2(x) if x.arg[0].func in {RDNA3Ops.VOP2, RDNA3Ops.VOP2_LIT} else
    lvop2(x, swap_only=True) if x.arg[0] in commutative_ins and len(x.src) == 2 else None),
  (UPat(Ops.INS, name="x"), lambda x: lvop3(x) if x.arg[0].func in {RDNA3Ops.VOP3, RDNA3Ops.VOP3SD, RDNA3Ops.VOPC, RDNA3Ops.VOP3P} else None),
])

# NOTE: could also match these by tag tuples (all valid load/store instructions) instead of using more ctx
pre_regalloc_matcher = PatternMatcher([
  # Lower a gated load as one adjacent sequence after linearization so the alt initialization cannot escape its CFG block.
  (UPat(Ops.INS, name="x"), lambda ctx,x: lower_gated_load(ctx,x) if ctx.ren.semantic_op.get(x.arg[0],x.op)
    is Ops.LOAD and x.src[-1].dtype is dtypes.bool and rafter(x.src[-1]).op is not Ops.BUFFER else None),
  (UPat(Ops.INS, name="x"), lambda ctx,x: lower_gated_store(ctx,x) if ctx.ren.semantic_op.get(x.arg[0],x.op)
    is Ops.STORE and x.src[-1].dtype is dtypes.bool else None),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  # TODO: figure out what Ops stay in graph this long and remove them?
  # NOTE: hacky, forces do_assemble in codegen but might hide incomplete lowering
  (UPat(GroupOp.All - {Ops.INS}, name="x"), lambda x: (x, [])),
])

def encode(x:UOp):
  def encfield(x:UOp):
    x = rafter(x)
    if is_const(x): return const_val(rafter(x))
    r, rs = rdef(x), rdefs(x)
    assert isinstance(r, Register), f"expect Register to encode, got {r} from {x}"
    for i,g in enumerate(rs[1:]): assert g.index == rs[i].index+1, "wide registers must be contiguous"
    dmap = { "vcc":dsl.VCC, "exec_lo":dsl.EXEC_LO, "v":dsl.v, "s":dsl.s }
    base = next(v for k,v in dmap.items() if k in r.name)
    return base[r.index] if len(rs) == 1 else base[r.index:r.index+len(rs)-1]

  match (group := x.arg[0].func):
    case RDNA3Ops.SCRATCH:
      fields = dict(offset=encfield(x.src[0]), sve=0)
      if rdef(x) is None: fields["data"] = encfield(x.src[1])
      else: fields["vdst"] = encfield(x)
    case RDNA3Ops.GLOBAL:
      fields = dict(addr=encfield(x.src[0]))
      if is_const(x.src[1]): fields["offset"] = encfield(x.src[1])
      else: fields["saddr"] = encfield(x.src[1])
      if rdef(x) is None: fields["data"]=encfield(x.src[2])
      else: fields["vdst"]=encfield(x)
    case RDNA3Ops.DS:
      offs = encfield(x.src[1])
      fields = dict(addr=encfield(x.src[0]), offset0=offs&0xFF, offset1=offs>>8)
      if rdef(x) is None: fields["data0"]=encfield(x.src[3])
      else: fields["vdst"]=encfield(x)
    case RDNA3Ops.SMEM: fields = dict(sdata=encfield(x), sbase=encfield(x.src[0]), offset=encfield(x.src[1]))
    case RDNA3Ops.SOPK: fields = dict(sdst=dsl.NULL, simm16=x.src[0].src[0].val)
    case RDNA3Ops.SOPP: fields = dict(simm16=(x.src[0].val if len(x.src) > 0 and x.src[0].op is Ops.CONST else 0))
    case RDNA3Ops.VOPC: fields = dict(src0=encfield(x.src[0]), vsrc1=encfield(x.src[1]))
    case RDNA3Ops.VOP3SD: fields = dict(sdst=encfield(vccop), vdst=encfield(x), **{f"src{i}":encfield(u) for i,u in enumerate(x.src[:3])})
    case RDNA3Ops.VOP3P:
      def _signed(dt:DType): return not (dtypes.is_unsigned(dt) or dtypes.is_float(dt))
      neg = _signed(x.src[0].dtype) | (_signed(x.src[1].dtype) << 1)
      fields = dict(vdst=encfield(x), **{f"src{i}":encfield(x.src[i]) for i in range(3)}, neg=neg)
    case RDNA3Ops.VOP2: fields = dict(vdst=encfield(x), src0=encfield(x.src[0]), vsrc1=encfield(x.src[1]))
    case RDNA3Ops.VOP3 | RDNA3Ops.VOP1 | RDNA3Ops.VOP3_SDST:
      fields = dict(vdst=encfield(x), **{f"src{i}":encfield(o) for i,o in enumerate(x.src)})
    case RDNA3Ops.SOP1 | RDNA3Ops.SOP2:
      fields = dict(sdst=encfield(x), **{f"ssrc{i}":encfield(o) for i,o in enumerate(x.src)})
    case _: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={x.arg[0].args[0].name.lower()}")
  return x.replace(arg=(x.arg[0](**fields), x.dtype))

class CntType(Enum):
  DS_CNT = auto(); LOAD_CNT = auto(); STORE_CNT = auto()

  def get(u:UOp):
    op = u.arg[0]
    if op.func in { RDNA3Ops.GLOBAL, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }:
      return CntType.STORE_CNT if u.dtype is dtypes.void else CntType.LOAD_CNT
    if op.func in { RDNA3Ops.SMEM, RDNA3Ops.DS }: return CntType.DS_CNT
    return None

@dataclass
class RDNA3LinearCtx:
  loop_label: dict[UOp, str] = field(default_factory=dict)

class RDNA3IselContext(PreRegallocContext):
  def __init__(self, sink:UOp, ren:RDNA3Renderer, info:ProgramInfo):
    super().__init__(sink, ren, info)
    # NOTE: entire kernel must fit on single CU? (WGP?)
    # 1536 vgprs per SIMD, 2 SIMD per CU
    # constrain waves*vgpr_limit <<< 1536*2, prevent dispatch hang
    waves = math.ceil(math.prod(info.local_size or (1,)) / 32)
    max_per_thread = min(1536*2 // waves, 256)
    # reserve VGPRs ahead of time to be excluded from normal allocation
    # 12x32 (wave 32) -> 384 spillable SGPR lanes
    n_spill_vgprs = 12
    self.gp_vgprs = VGPRS[1:max_per_thread-n_spill_vgprs]
    # TODO: how to make this per-kernel but accessible in renderer?
    self.ren.spill_vgprs: dict[Register, int] = {r:0 for r in VGPRS[max_per_thread-n_spill_vgprs:max_per_thread]}

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  extra_matcher = extra_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.SHR, Ops.SHL, Ops.MAX, Ops.MULACC)}
  pre_regalloc_ctx_type = RDNA3IselContext
  def __init__(self, target:Target):
    super().__init__(target)
    self.shared_max = HIPRenderer.shared_max
    self.tensor_cores = tc.get_amd(target.arch)
    self.post_regalloc_ctx = RDNA3LinearCtx()
    self.semantic_op = {}

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s}
  def is_two_address(self, x:UOp) -> bool: return False
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    return '\n'.join(str(encode(u).arg[0]) for u in uops)

  def assign_spill_slot(self, v:VRegister, vdef:UOp) -> tuple[int, int]:
    if v.cons[0].name[0] == 'v':
      sz = v.cons[0].size * v.width
      offset = self.spill_size + (sz - self.spill_size % sz) % sz
      return (offset, offset + sz)
    else:
      vgpr,lane = next(((r,l) for r,l in self.spill_vgprs.items() if 32 - l >= v.width), (None, None))
      assert vgpr is not None, "ran out of reserved SGPR spill lanes"
      self.spill_vgprs[vgpr] += v.width
      return ((vgpr,lane), self.spill_size)

  def spill(self, spill_offset:any, x:UOp, sub_idx:int|None=None) -> list[UOp]:
    regs = rdefs(x)
    if regs[0].name[0] == 'v':
      if sub_idx is not None:
        return [UOp(Ops.INS, arg=(RDNA3Ops.scratch_store_b32, dtypes.void),
          src=(const(spill_offset+sub_idx*4), def_reg(x.dtype, regs)))]
      batches = [regs[i*4:(i+1)*4] for i in range((len(regs)+3)//4)]
      return [UOp(Ops.INS, arg=(getattr(RDNA3Ops, f"scratch_store_b{len(b)*32}"), dtypes.void), \
        src=(const(spill_offset+j*16), def_reg(x.dtype, b))) for j,b in enumerate(batches)]
    else:
      vgpr,lane = spill_offset
      return [UOp(Ops.INS, arg=(RDNA3Ops.v_writelane_b32, dtypes.void), src=(def_reg(x.dtype, r),
        const(lane+i+(sub_idx or 0))), tag=vgpr) for i,r in enumerate(regs)]

  def fill(self, spill_offset:any, sub_idx:int|None, x:UOp, regs:tuple[Register,...]) -> tuple[UOp, list[UOp]]:
    if regs[0].name[0] == 'v':
      if sub_idx is not None:
        return (ld := UOp(Ops.INS, src=(const(spill_offset+sub_idx*4),), arg=(RDNA3Ops.scratch_load_b32, x.dtype), tag=regs)), [ld]
      batches = [regs[i*4:(i+1)*4] for i in range((len(regs)+3)//4)]
      ops = [UOp(Ops.INS, src=(const(spill_offset+j*16),), \
        arg=(getattr(RDNA3Ops, f"scratch_load_b{len(b)*32}"), x.dtype), tag=b) for j,b in enumerate(batches)]
      return UOp(Ops.STACK, src=tuple(ops), tag=regs), ops
    else:
      vgpr,lane = spill_offset
      movs = [UOp(Ops.INS, src=(def_reg(x.dtype, vgpr),
        const(lane+i+(sub_idx or 0))), arg=(RDNA3Ops.v_readlane_b32, x.dtype), tag=(r,)) for i,r in enumerate(regs)]
      return UOp(Ops.STACK, src=tuple(movs), tag=regs), movs

  def vcopy(self, u:UOp, vr:VRegister) -> tuple[UOp, list[UOp]]:
    if vr.width == 1: return (mov := vmov(u,vr)), [mov]
    # NOTE: the STACK gets one src per subregister, the geps are only emitted as operands
    movs, ins = [], []
    if u.op is Ops.STACK: ins = movs = [vmov(s, vr.sub(i)) for i,s in enumerate(u.src)]
    else:
      for i in range(vr.width):
        ins.extend([g := gep(u,i), mov := vmov(g, vr.sub(i))])
        movs.append(mov)
    grp = UOp(Ops.STACK, src=tuple(movs), tag=(vr,))
    return grp, ins + [grp]

  def copy(self, u:UOp, regs:tuple[Register,...]) -> list[UOp]:
    return [vmov(def_reg(u.dtype, rs),rd) for rs,rd in zip(rdefs(u), regs)]

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    deps: set[Register] = set()
    nuops, pending_scratch, pending_lds = [], False, False
    self.spill_vgprs: dict[Register, int] = {r:0 for r in self.spill_vgprs.keys()} # reset?

    # data dependency resolution (s_waitcnt)
    def waitcnt(): return UOp(Ops.INS, src=(const(0),), arg=(RDNA3Ops.s_waitcnt, dtypes.void))
    def wait_vscnt(): return UOp(Ops.INS, src=(const(0),), arg=(RDNA3Ops.s_waitcnt_vscnt, dtypes.void))
    for u in lin.src:
      if u.arg[0] is RDNA3Ops.s_barrier:
        # flush before barrier
        nuops.append(waitcnt())
        deps.clear()
        pending_lds = False
      elif isinstance(u.tag, str) and u.arg[0].func is RDNA3Ops.SOPP:
        # flush at loop backedge
        if deps: nuops.append(waitcnt()); deps.clear()
        if pending_scratch: nuops.append(wait_vscnt()); pending_scratch = False
      elif any(r in deps for s in u.src for r in rdefs(s)) or any(r in deps for r in rdefs(u)):
        nuops.append(waitcnt())
        deps.clear()
      if (tp := CntType.get(u)) is not None:
        if tp in [CntType.DS_CNT, CntType.LOAD_CNT]:
          # realize outstanding stores before reading
          if u.arg[0].func is RDNA3Ops.SCRATCH and pending_scratch:
            nuops.append(wait_vscnt()); pending_scratch = False
          if tp is CntType.DS_CNT:
            if u.arg[1] is dtypes.void: pending_lds = True
            elif pending_lds:
              nuops.append(waitcnt())
              pending_lds = False
          deps.update(rdefs(u))
          if u.arg[1] is dtypes.void: # protect address registers?
            for s in u.src: deps.update(rdefs(s))
        elif u.arg[0].func is RDNA3Ops.SCRATCH: pending_scratch = True
      nuops.append(u)

    pc = 0
    targets: dict[str, int] = {}
    upc: dict[UOp, int] = {}
    uops = nuops.copy()
    nuops = []
    for u in uops:
      if u.arg[0] is RDNA3Ops.s_nop and isinstance(u.tag, str): targets[u.tag] = pc
      else:
        upc[u] = pc = pc + (u := encode(u)).arg[0].size()
        nuops.append(u)

    lin = lin.replace(src=tuple([u if not isinstance(u.tag, str) else \
      u.replace(arg=(RDNA3Ops.SOPP(u.arg[0].op, (targets[u.tag] - upc[u]) // 4), u.dtype)) for u in nuops]))
    return assemble_linear(prg, lin, self.target.arch, scratch_size=self.spill_size)

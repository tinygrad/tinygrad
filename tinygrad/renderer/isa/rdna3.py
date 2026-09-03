from tinygrad.dtype import dtypes, AddrSpace, truncate, DType, InvalidType, to_storage_scalar, ConstFloat
from typing import Any
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg, range_str, GroupOp, graph_rewrite, ProgramInfo
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef, copy_dst, PreLinearKernelCtx
from tinygrad.renderer.cstyle import create_non_native_float_pats, pm_manual_bf16_cast
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
from tinygrad.codegen.decomp.op import fast_idiv
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.cstyle import HIPRenderer
import tinygrad.renderer.amd.dsl as dsl
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
import itertools, functools, struct, math
from enum import Enum, auto

# ---- (UOp, dtype) -> Instruction tables ----
dt_to_isa = { dtypes.int32:"i32", dtypes.uint32:"u32", dtypes.float32:"f32", dtypes.float64:"f64", dtypes.float16:"f16", dtypes.int16:"i16", dtypes.uint16:"u16", dtypes.uint64:"u64", dtypes.int64:"i64", dtypes.bfloat16:"bf16", dtypes.uint8:"u8", dtypes.int8:"i8" }
isa_to_dt = { v:k for k,v in dt_to_isa.items() }

# (uop, prefix, opcodes, support 32 and 64 bit encoding (e32/e64 branches with keys))
insdefs = [
  (Ops.MAX, "v_max", ["f32_e32", "i32_e32", "u32_e32", "f64", "f16_e32"], False),
  (Ops.ADD, "v_add", ["f16_e32", "f32_e32", "f64", "nc_i32", "nc_u32_e32", "nc_u16", "nc_i16"], False),
  (Ops.SUB, "v_sub", ["f16_e32", "f32_e32", "nc_i32", "nc_i16", "nc_u16", "nc_u32_e32"], False),
  (Ops.MUL, "v_mul", ["f16_e32", "f32_e32", "f64", "lo_u32", "lo_u16"], False),
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
V_LSHL = { 2:RDNA3Ops.v_lshlrev_b16, 4:RDNA3Ops.v_lshlrev_b32_e32, 8:RDNA3Ops.v_lshlrev_b64 }
V_LSHR = { 2:RDNA3Ops.v_lshrrev_b16, 4:RDNA3Ops.v_lshrrev_b32_e32, 8:RDNA3Ops.v_lshrrev_b64 }
V_ASHR = { dtypes.int16:RDNA3Ops.v_ashrrev_i16, dtypes.int32:RDNA3Ops.v_ashrrev_i32_e32, dtypes.int64:RDNA3Ops.v_ashrrev_i64 }
V_LDEXP = { dtypes.float16:RDNA3Ops.v_ldexp_f16_e32, dtypes.float32:RDNA3Ops.v_ldexp_f32, dtypes.float64:RDNA3Ops.v_ldexp_f64 }
S_CMP = { Ops.CMPNE:RDNA3Ops.s_xor_b32, Ops.XOR:RDNA3Ops.s_xor_b32, Ops.OR: RDNA3Ops.s_or_b32, Ops.AND:RDNA3Ops.s_and_b32, Ops.CMPLT: RDNA3Ops.s_and_not1_b32, Ops.CMPEQ:RDNA3Ops.s_xnor_b32 }

# ---- helpers ----
lane_ctr = itertools.count(-1, -1)
def def_reg(dt:DType, defs:Any): return UOp.placeholder((1,), dt, next(lane_ctr), AddrSpace.REG).replace(tag=defs if isinstance(defs, tuple) else (defs,))
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
  assert x.dtype.itemsize <= 4, x.op
  nx = x.ins(RDNA3Ops.v_mov_b16_e64 if x.dtype is dtypes.half else RDNA3Ops.v_mov_b32_e32, src=(x,))
  return nx.rtag() if r is None else nx.replace(tag=(r,))
def rafter(x:UOp, bitcast=False) -> UOp:
  return rafter(x.src[0]) if x.op in ({Ops.AFTER, Ops.BITCAST} if bitcast else {Ops.AFTER}) else x
# stack of dword registers/register producing instructions grouped by order to be assigned contiguous register slice
def multireg(*src, dtype: DType, vr:VRegister|None=None) -> UOp:
  return UOp(Ops.STACK, src=tuple(s.bitcast(dtypes.uint32) for s in src), tag=(vr,) if vr else None).bitcast(dtype)
def const64(x:UOp, c:UOp):
  v = c.val.bits if dtypes.is_float(x.dtype) else c.val
  return multireg(vmov(const(v)), vmov(const(v >> 32)), dtype=x.dtype)

# ---- register classes/ABI regs ---
VGPRS = tuple(Register(f"v{i}", i, size=4) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i, size=4) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
VCC, EXEC = Register("vcc", 0, size=4), Register("exec_lo", 0, size=4)

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
  if isinstance(v, float): return not isinstance(v, ConstFloat) and v in dsl.SrcField._FLOAT_ENC
  if isinstance(v, int): return 0 <= v <= 64 or -16 <= v < 0
  return False

def lvop3(x:UOp):
  lits = [s for s in x.src if is_const(s) and not can_fold_lit(s)]
  return None if len(lits) == 1 else x.replace(src=tuple([vmov(s) if s in lits[1:] else s for s in x.src]))

rev_op_order = { RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshlrev_b16, RDNA3Ops.v_lshlrev_b64, RDNA3Ops.v_lshrrev_b32_e32, RDNA3Ops.v_lshrrev_b16, RDNA3Ops.v_lshrrev_b64, RDNA3Ops.v_ashrrev_i32_e32, RDNA3Ops.v_ashrrev_i64 }
commutative_ins = {i for op in (Ops.ADD, Ops.MUL, Ops.MAX) for i in OP_INS[op].values()}
def lvop2(x:UOp, swap_only=False):
  if not is_const(x.src[1]): return None
  rest = x.src[2:] if len(x.src) > 2 else ()
  non_commutative = x.arg[0] in set(OP_INS[Ops.SUB].values()) | rev_op_order
  if not non_commutative and not is_const(x.src[0]): return x.replace(src=(x.src[1], x.src[0]) + rest)
  # VOP3 encodes a const in src1 fine, it only ever wants the commutative swap above, never the vmov
  return None if swap_only else x.replace(src=(x.src[0], vmov(x.src[1])) + rest)

def defines_sgpr(enc) -> bool:
  dst = next((getattr(enc, n) for n in ("vdst", "sdst", "sdata") if hasattr(enc, n)), None)
  return isinstance(dst, (dsl.SGPRField, dsl.SSrcField, dsl.AlignedSGPRField))

def alloc_vregs(ctx, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], (Register, VRegister)): return None

  width, alignment = 1, 1
  is_sdst = x.op is Ops.INS and defines_sgpr(x.arg[0].func)
  if is_sdst:
    cons, width = ctx.gp_sgprs, (x.dtype.itemsize+3) // 4
    if width == 2: alignment = 2
  else: cons, width = ctx.gp_vgprs, ((x.dtype.itemsize+3) // 4) * (len(x.src) if x.op is Ops.STACK else 1)
  return x.replace(tag=(ctx.vreg(cons, width=width, alignment=alignment),))

def abi(ctx, x:UOp) -> UOp|None:
  if x.tag is True: return None
  if x.op is Ops.SPECIAL:
    if x.arg[0] == 'g': return vmov(ctx.reserved(WGIDS[int(x.arg[-1])], dtypes.uint32)).after(x.rtag())
    src = (ctx.reserved(WIIDS, dtypes.uint32), const(10*int(x.arg[-1])), const(10))
    return x.ins(RDNA3Ops.v_bfe_u32, dtype=dtypes.uint32, src=src)
  else:
    offs = const(sum(8 if u.op == Ops.PARAM else u.dtype.itemsize for u in ctx.func_args[:ctx.func_args.index(x)]))
    psrc = (ctx.reserved(KERNARG_PTR, dtypes.uint64), offs)
    if x.addrspace is AddrSpace.ALU: out = vmov(UOp(Ops.INS, src=psrc, arg=(RDNA3Ops.s_load_b32, x.dtype)))
    else: out = UOp(Ops.INS, src=psrc, arg=(RDNA3Ops.s_load_b64, dtypes.ulong))
  return out.after(x.rtag()) # preserve PARAM scheduling

# ----- memory access ----
# global address = saddr (base sgpr pair) + 32 bit unsigned vgpr byte offset
def fold_global(base:UOp, idx:UOp):
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(scale.bit_length() - 1, dtypes.int32)
  vaddr = to_vgpr(idx)
  if shft.src[0].val > 0: vaddr <<= shft
  return (vaddr, base)

def fold_lds(base:UOp, idx:UOp):
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(scale.bit_length() - 1, dtypes.int32)
  # offset0/offset1 form one 16 bit unsigned immediate, anything outside that has to stay in the addr vgpr
  def foldable(v:int) -> bool: return 0 <= v < (1 << 16)
  if is_const(idx) and foldable(v := const_val(idx) * scale): return (vmov(const(0)), const(v, dtypes.uint16), base)
  if idx.op is Ops.ADD and is_const(idx.src[1]) and foldable(v := const_val(idx.src[1]) * scale):
    return ((idx.src[0] << shft if scale > 1 else idx.src[0]).cast(dtypes.uint32), const(v, dtypes.uint16), base)
  return (idx << shft, const(0, dtypes.uint16), base)

def fold_address(x:UOp): return fold_lds(*x.src[:2]) if x.addrspace is AddrSpace.LOCAL else fold_global(*x.src[:2])

def batch_scratch(store:bool, base:int, dt:DType, regs:VRegister|tuple[Register,...]) -> list[UOp]:
  batches = []
  # batch registers into groups of 4 dwords per copy in/out
  if isinstance(regs, VRegister):
    batches = [regs] if regs.width <= 4 else [regs[i*4:(i+1)*4] for i in range(regs.width//4)]
  else: batches = [regs[i*4:(i+1)*4] for i in range((len(regs)+3)//4)]
  ops = []
  for j,b in enumerate(batches):
    mop = "store" if store else "load"
    n = b.width if isinstance(b, VRegister) else len(b)
    opc = getattr(RDNA3Ops, f"scratch_{mop}_b{n*32}")
    u = UOp(Ops.INS, arg=(opc, dtypes.void if store else dt), src=(const(base+j*16),))
    if store: u = u.replace(src=(*u.src, def_reg(dt, b)))
    else: u = u.replace(tag=b)
    ops.append(u)
  return ops

def load(ctx, x:UOp, idx:UOp):
  if idx.addrspace is AddrSpace.REG:
    if (buf := rafter(idx.src[0])).arg in ctx.overflows:
      slot = ctx.overflows[buf.arg][idx.src[1].src[0].val]
      vr = ctx.vreg(ctx.gp_vgprs, width=(buf.dtype.itemsize+3)//4)
      op = batch_scratch(False, slot, buf.dtype, vr)[0]
      return op.replace(src=(op.src[0].after(idx),))
    else:
      return ctx.bufreg(idx).after(idx)

  n = idx.src[-1].src[0].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.src[0].dtype.itemsize
  suffix = "b" if sz > 2 else "i" if x.dtype in dtypes.sints else "u"
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_load_{suffix}{sz*8}")
  return x.ins(opc, src=fold_address(rafter(idx, True))+x.src[1:], tag=(ctx.vreg(ctx.gp_vgprs, width=(sz+3)//4),))

def store(ctx, x:UOp, idx:UOp, val:UOp):
  if idx.addrspace is AddrSpace.REG:
    if (buf := rafter(idx.src[0])).arg in ctx.overflows:
      slot = ctx.overflows[buf.arg][idx.src[1].src[0].val]
      assert buf.dtype.itemsize == 4, "single dword BUFFER spill items only"
      return UOp(Ops.INS, arg=(RDNA3Ops.scratch_store_b32, dtypes.void), src=(const(slot), to_vgpr(val).after(idx)))
    else:
      return ctx.ren.copy(val.after(idx), rdefs(ctx.bufreg(idx)))[0]

  n = idx.src[-1].src[0].val if idx.op is Ops.SHRINK else 1
  sz = n * idx.dtype.itemsize
  prefix = "global" if idx.addrspace is AddrSpace.GLOBAL else "ds"
  opc = getattr(RDNA3Ops, f"{prefix}_store_b{sz*8}")
  return x.ins(opc, src=fold_address(rafter(idx, True))+(to_vgpr(val),*x.src[2:]))

# ------ ALU ------
def arith64(ctx, x:UOp, a:UOp, b:UOp):
  ins_lo = RDNA3Ops.v_add_co_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_u32
  ins_hi = RDNA3Ops.v_add_co_ci_u32 if x.op is Ops.ADD else RDNA3Ops.v_sub_co_ci_u32
  lo = UOp(Ops.INS, src=(gep(a,0), gep(b,0)), arg=(ins_lo, dtypes.uint32))
  hi = UOp(Ops.INS, src=(gep(a,1), gep(b,1), ctx.vccop, lo), arg=(ins_hi, dtypes.uint32))
  return multireg(lo, hi.after(lo), dtype=x.dtype)

def _mad(a:UOp, b:UOp, c:UOp=const(0, dtypes.uint64)):
  return UOp(Ops.INS, arg=(RDNA3Ops.v_mad_u64_u32, dtypes.uint64), src=(a, b, c))
def mul64(ctx, x:UOp, a:UOp, b:UOp):
  p1 = _mad(gep(a,1), gep(b,0)).bitcast(dtypes.uint64) << 32
  p2 = _mad(gep(a,0), gep(b,1)).bitcast(dtypes.uint64) << 32
  return _mad(gep(a,0), gep(b,0), p1 + p2).bitcast(x.dtype)

def mulhi32(a:UOp, b:UOp) -> UOp: return ((a.cast(dtypes.uint64) * b.cast(dtypes.uint64)) >> 32).cast(dtypes.uint32)
def mulhi64(a:UOp, b:UOp) -> UOp:
  def mul32(a:UOp, b:UOp) -> UOp: return multireg(a*b, mulhi32(a,b), dtype=dtypes.uint64)
  a0, a1, b0, b1 = gep(a,0), gep(a,1), gep(b,0), gep(b,1)
  t = mul32(a1,b0) + mulhi32(a0,b0).cast(dtypes.uint64)
  return mul32(a1,b1) + (t >> 32) + ((mul32(a0,b1) + t.cast(dtypes.uint32).cast(dtypes.uint64)) >> 32)

def idiv(x:UOp, a:UOp, b:UOp) -> UOp:
  # truncated integer division: estimate z ~= 2**w/b in float32, refine with newton steps, q = a*z >> w, fix up with two correction steps
  w, udt, sdt = (64, dtypes.uint64, dtypes.int64) if x.dtype in dtypes.int64s else (32, dtypes.uint32, dtypes.int32)
  mulhi = mulhi64 if w == 64 else mulhi32
  def sub(p:UOp, q:UOp) -> UOp: return p.alu(Ops.SUB, q)  # UOp.__sub__ lowers to p + q*-1, which is a mul
  def flip(v:UOp, s:UOp) -> UOp: return sub(v ^ s, s)  # -v if s is all ones, v if s is 0
  if (signed := not dtypes.is_unsigned(x.dtype)):
    # sign extend to w bits first (abs of a narrower type overflows at its min), then abs as (v ^ sign) - sign
    sa, sb = [v.cast(sdt) >> (w-1) for v in (a, b)]
    a, b, s = flip(a.cast(sdt), sa), flip(b.cast(sdt), sb), (sa ^ sb).bitcast(udt)
  a, b = a.cast(udt), b.cast(udt)
  if w == 32: z = (b.float().reciprocal() * (2**32 - 2**10)).cast(udt)
  else:
    # no f32 <-> u64 conversions on rdna3, go through the 32 bit halves
    m = (gep(b,1).float() * 2**32 + gep(b,0).float()).reciprocal() * (2**64 - 2**42)
    t = (m * 2**-32).trunc()
    z = (t.cast(dtypes.uint32).cast(udt) << 32) | (t * -2**32 + m).cast(dtypes.uint32).cast(udt)
  # newton: z += z * (2**w - b*z) / 2**w, the relative error squares each step so 32 bit needs one step and 64 bit needs two
  for _ in range(w // 32):
    z = z + mulhi(z, sub(UOp.const(0, sdt), (b * z).bitcast(sdt)).bitcast(udt))
  q = mulhi(a, z)
  r = sub(a, q * b)
  for _ in range(2): q, r = (r < b).where(q, q + 1), (r < b).where(r, sub(r, b))
  return (flip(q, s) if signed else q).cast(x.dtype)

def ldexp(x:UOp, c:UOp):
  n = c.val
  if isinstance(n, (ConstFloat, float)) and not n.is_integer(): return None
  if (n := int(n)) & (n-1) != 0 or abs(n) < 2: return None
  return x.ins(V_LDEXP[x.dtype], src=(x, const(n.bit_length()-1)))

def render_wmma(ctx, a:UOp, b:UOp, acc:UOp, wmma:UOp):
  srcdt = dt_to_isa[wmma.arg[1]]
  if wmma.arg[1] in dtypes.int8s: srcdt = "iu8"
  ins = getattr(RDNA3Ops, f"v_wmma_{dt_to_isa[wmma.dtype]}_16x16x16_{srcdt}")
  return UOp(Ops.INS, src=(a,b,acc), arg=(ins, wmma.dtype), tag=(ctx.vreg(ctx.gp_vgprs, width=8),))

# ---- casting utilities -----
def int_to_int64(y:UOp, tdt:DType):
  if dtypes.is_unsigned(y.dtype): hi = vmov(const(0))
  else: hi = to_vgpr(y).bitcast(dtypes.int32) >> max(y.dtype.itemsize*8, 32)-1
  return multireg(vmov(y), hi, dtype=tdt)

def f64_to_i64(y:UOp, tdt:DType):
  hi_dt = smux(tdt, dtypes.int32, dtypes.uint32)
  hi_f = (tr := y.trunc()) * const(2**-32, dtypes.double)
  hi_f = UOp(Ops.INS, src=(hi_f,), arg=(RDNA3Ops.v_floor_f64_e32, dtypes.float64))
  lo_f = hi_f * const(2**32, dtypes.double)
  lo_f = tr + lo_f * const(-1., dtypes.float64)
  return multireg(lo_f.cast(dtypes.uint32), hi_f.cast(hi_dt), dtype=tdt)

def i64_to_f64(x:UOp):
  hi_dt = smux(x.dtype, dtypes.int32, dtypes.uint32)
  return gep(x,0).double() + (gep(x,1).bitcast(hi_dt).double() * const(2**32, dtypes.double))

# ---- control flow ----
def restoreexec(ctx, mask:UOp) -> UOp: return UOp(Ops.INS, src=(ctx.execop,mask), arg=(RDNA3Ops.s_or_b32, dtypes.void), tag=(EXEC,))
def saveexec(ctx, gate:UOp) -> UOp: return UOp(Ops.INS, src=(gate,), arg=(RDNA3Ops.s_and_saveexec_b32, dtypes.uint32), tag=(ctx.vreg(ctx.gp_sgprs),))
def label(ctx, name:str) -> UOp: return UOp(Ops.INS, arg=(RDNA3Ops.s_nop, dtypes.void), tag=name)

def lower_gated_load(ctx, x:UOp):
  alt, gate = x.src[-2:]
  return x, ctx.ren.copy(alt, rdef(x))[1] + [(mask := saveexec(ctx, gate)), x, restoreexec(ctx, mask)]

def lower_range(ctx, x:UOp):
  if x.src[0].op is Ops.NOOP: return x, [label(ctx, f".LOOP_BODY_{range_str(x)}")]
  acc = x.ins(RDNA3Ops.v_mov_b32_e32, src=(const(0),))
  ctx.loop_label[acc] = range_str(x)
  return acc, [acc, label(ctx, f".LOOP_BODY_{range_str(x)}")]

def lower_end(ctx, x:UOp):
  if x.src[-3].src[0].op is Ops.NOOP: # loop
    rng, pred, mask = x.src[-3:]
    jmp = UOp(Ops.INS, arg=(RDNA3Ops.s_cbranch_execnz, dtypes.void), tag=f".LOOP_BODY_{range_str(rng)}")
    pred = UOp(Ops.INS, src=(pred,), arg=(RDNA3Ops.s_mov_b32, dtypes.void), tag=(EXEC,))
    return pred, [pred, jmp, restoreexec(ctx, mask)]
  else:
    acc,bnd,mask = x.src[-3:]
    loop_end = label(ctx, f".LOOP_END_{ctx.loop_label[acc]}")
    inc = acc.ins(RDNA3Ops.v_add_nc_u32_e32, src=(const(1), acc), tag=acc.tag)
    jmp = UOp(Ops.INS, arg=(RDNA3Ops.s_cbranch_execnz, dtypes.void), tag=f".LOOP_BODY_{ctx.loop_label[acc]}")
    pred = UOp(Ops.INS, src=(acc,bnd), arg=(RDNA3Ops.v_cmpx_lt_u32_e64, dtypes.void), tag=(EXEC,))
    return inc, [inc, pred, jmp, loop_end, restoreexec(ctx, mask)]

# ---- lowering passes ----
int1regs = dtypes.int8s + dtypes.int16s + dtypes.int32s
from tinygrad.renderer.tc import pm_validate_wmma_rdna3
extra_matcher = PatternMatcher([
  (UPat.cvar("c", dtypes.bfloat16), lambda c: UOp.const(c.val if isinstance(c.val, InvalidType) else
    to_storage_scalar(c.val, dtypes.bfloat16), dtypes.uint16).bitcast(dtypes.bfloat16)),
  (UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar("d"))),
    lambda ctx,x,d: fast_idiv(ctx, x, d.val) if x.vmin >= 0 or x.dtype in dtypes.uints else None),
  (UPat(Ops.CMOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - b * a.alu(Ops.CDIV, b)),
  (UPat(Ops.CDIV, dtypes.ints, (UPat.var("a"), UPat.var("b")), name="x"), idiv),
  (UPat(Ops.EXP2, dtypes.double, src=(UPat.var("d"),)), xexp2),
  (UPat(Ops.LOG2, dtypes.double, src=(UPat.var("d"),)), xlog2),
]) + pm_manual_bf16_cast + create_non_native_float_pats((dtypes.bfloat16,)) + pm_validate_wmma_rdna3

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

def recast(u:UOp, dt:DType):
  src = tuple(s.cast(dt) for s in u.src)
  return src[0].alu(u.op, *src[1:]).bitcast(u.dtype)

pre_isel_matcher = PatternMatcher([
  # --- bools are lane masks ---
  (UPat(Ops.STORE, src=(UPat.var("buf"), UPat.var("val", dtype=dtypes.bool)), allow_any_len=True, name="x"), \
    lambda buf,val,x: x.replace(src=(buf,val.cast(dtypes.uint32)) + x.src[2:])),
  (UPat(Ops.LOAD, dtypes.bool, src=(UPat.var("buf"),), allow_any_len=True, name="x"),
    lambda buf,x: x.replace(src=(buf.bitcast(dtypes.uchar),) +
      (() if len(x.src) == 1 else (x.src[1].cast(dtypes.uchar), x.src[2]))).cast(dtypes.bool)),
  (UPat.cvar("x").cast(dtypes.bool), lambda x: x.ins(RDNA3Ops.s_mov_b32, src=(const((1 << 32) - 1 if x.val else 0),))),
  (UPat.var("x").cast(dtypes.bool), lambda x: x.alu(Ops.CMPEQ, const(1, x.dtype))),
  (UPat.var("y", dtypes.bool).cast(name="x"), lambda y,x: y.where(const(1, x.dtype), const(0, x.dtype))),
  # --- int8 alu is int16 for now ---
  (UPat(GroupOp.ALU-{Ops.WHERE}, dtypes.int8s, name="x"), lambda x: recast(x, smux(x.dtype, dtypes.int16, dtypes.uint16))),
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.int8s), UPat()), name="x"),
    lambda x: x.replace(src=tuple(s.cast(smux(x.dtype, dtypes.int16, dtypes.uint16)) for s in x.src))),
  # -- int -> int casts ---
  (UPat.var("y", dtypes.int64s).cast(int1regs, name="x"), lambda y,x: gep(y, 0).bitcast(x.dtype)),
  (UPat.var("y", int1regs).cast(dtypes.int64s, name="x"), lambda y,x: int_to_int64(y, x.dtype)),
  (UPat.var("y", dtypes.double).cast(dtypes.half), lambda y: y.float().half()),
  # --- other ---
  (UPat((Ops.SHR, Ops.SHL), src=(UPat.var("val"), UPat.var("n", dtypes.int64s+(dtypes.float64,))), name="x"),
    lambda val,x,n: x.replace(src=(val, n.cast(dtypes.uint32)))),
  (UPat(Ops.INDEX, (dtypes.half,dtypes.bfloat16)+dtypes.int8s+dtypes.int16s,
    src=(UPat.var("buf"), UPat.cvar("c").cast()), name="idx"), unpack),
  (UPat(Ops.STACK, name="x"), lambda x: x.replace(src=tuple(vmov(s) if is_const(s) and s.dtype.itemsize < 8 else s for s in x.src))
    if any(is_const(s) for s in x.src) else None),
  (UPat(Ops.MUL, (dtypes.int16,dtypes.int32), name="x"), lambda x: recast(x, dtypes.uint32)),
  (UPat(Ops.MAX, dtypes.int16s, name="x"), lambda x: recast(x, smux(x.dtype, dtypes.int32, dtypes.uint32))),
  (UPat(Ops.MAX, dtypes.int64s, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: (a < b).where(b, a)),
  (UPat(Ops.MULACC, dtypes.ints, src=(UPat.var("a"), UPat.var("b"), UPat.var("c"))), lambda a,b,c: a*b + c),
  # --- non-native 64 bit alu expansions ---
  (UPat(Ops.WHERE, src=(UPat.var("pred"), UPat.var("a", dtype=dtypes.int64s+(dtypes.float64,)), UPat.var("b"))),
    lambda pred,a,b: multireg(pred.where(gep(a,0),gep(b,0)), pred.where(gep(a,1), gep(b,1)), dtype=a.dtype)),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), dtypes.int64s+(dtypes.float64,), src=(UPat.var("a"), UPat.var("b")), name="x"),
    lambda x,a,b: multireg(gep(a,0).alu(x.op, gep(b,0)), gep(a,1).alu(x.op, gep(b,1)), dtype=x.dtype)),
  (UPat.cvar("c").cast((dtypes.float64,)+dtypes.int64s, name="x"), const64),
]) + pm_float_to_int + pm_int_to_float

isel_matcher = PatternMatcher([
  # --- control flow ---
  (UPat(Ops.RANGE, name="rng"), lambda ctx,rng:
    rng.replace(src=rng.src + (UOp(Ops.INS, arg=(RDNA3Ops.s_mov_b32, dtypes.uint32), src=(ctx.execop,)),),
    tag=ctx.vreg(ctx.gp_vgprs)) if rng.tag is None else None),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng"), UPat()), name="x"),
    lambda x,rng: x.replace(src=(x.src[0],rng,x.src[-1],rng.src[-1])) if rng.tag is not None else None),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng")), name="x"), \
    lambda x,rng: x.replace(src=(x.src[0],rng,rng.src[0],rng.src[-1])) if rng.tag is not None else None),
  # --- double precision alu ---
  (UPat(Ops.MUL, dtypes.int64s, name="x", src=(UPat.var("a"), UPat.var("b"))), mul64),
  (UPat((Ops.ADD, Ops.SUB), dtypes.int64s, src=(UPat.var("a"), UPat.var("b")), name="x"), arith64),
  # --- operator fusion ---
  ((UPat.var("x", dtype=dtypes.floats) * UPat.cvar("c").cast()), ldexp),
  (UPat().sqrt().named("x").reciprocal(), lambda x: x.ins(V_RSQ[x.dtype]) if x.dtype in V_RSQ else None),
  (UPat(Ops.MULACC, dtypes.floats, name="x"), lambda x: x.ins(V_FMA[x.dtype])),
  (UPat(Ops.ADD, dtypes.uint32, src=(UPat(Ops.ADD, name="y"), UPat.var("b")), name="x"),
    lambda ctx,x,y,b: x.ins(RDNA3Ops.v_add3_u32, src=y.src + (b,))),
  # --- general alu ---
  (UPat(Ops.SHR, dtypes.uints, name="x"), lambda x: x.ins(V_LSHR[max(2, x.dtype.itemsize)], src=x.src[2::-1])),
  (UPat(Ops.SHR, dtypes.sints, name="x"), lambda x: x.ins(V_ASHR[x.dtype], src=x.src[2::-1])),
  (UPat(Ops.SHL, name="x"), lambda x: x.ins(V_LSHL[max(2, x.dtype.itemsize)], src=x.src[2::-1])),
  (UPat(GroupOp.Comparison|{Ops.XOR, Ops.AND, Ops.OR}, dtypes.bool, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
    lambda a,b,x: x.ins(S_CMP[x.op], src=(b,a) if x.op is Ops.CMPLT else (a,b))),
  (UPat(GroupOp.Comparison, dtypes.bool, name="x"), lambda x: x.ins(OP_INS[x.op][64][rafter(x.src[0]).dtype])),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), name="x"), lambda ctx,x: x.ins(getattr(RDNA3Ops, f"v_{x.op.name.lower()}_b32_e32"))),
  (UPat(Ops.WHERE, dtypes.bool, src=(UPat.var("m"), UPat.var("a"), UPat.var("b")), name="x"), lambda m,a,b,x: (m & a) | (~m & b)),
  (UPat.var("pred").where(UPat.var("a"), UPat.var("b")).named("x"), lambda pred,a,b,x:
    x.ins(RDNA3Ops.v_cndmask_b32_e64 if x.dtype.itemsize >= 4 else RDNA3Ops.v_cndmask_b16, src=(b,a,pred))),
  (UPat(GroupOp.Binary|GroupOp.Unary, name="x"), lambda x: x.ins(OP_INS[x.op][x.dtype])),
  (UPat(Ops.WMMA, src=(UPat.var("a"), UPat.var("b"), UPat.var("acc")), name="wmma"), render_wmma),
  # --- casting ---
  (UPat.var("y", dtypes.ints).cast(dtypes.ints).named("x"), lambda y,x: y.bitcast(x.dtype)
    if y.dtype.itemsize >= x.dtype.itemsize else
    x.ins(RDNA3Ops.v_bfe_i32 if y.dtype in dtypes.sints else
    RDNA3Ops.v_bfe_u32, src=(y,const(0),const(y.dtype.itemsize*8)))),
  (UPat.var("y", dtype=dtypes.ints+dtypes.floats).cast(name="x"),
    lambda y,x: x.ins(getattr(RDNA3Ops, f"v_cvt_{dt_to_isa[x.dtype]}_{dt_to_isa[y.dtype]}_e64"))),
  # --- mem ops ---
  (UPat.var("idx").store(UPat.var("val"), allow_any_len=True).named("x"), store),
  (UPat.var("idx").load(name="x", allow_any_len=True), load),
  # --- operand legalization ---
  (UPat(Ops.INS, name="x"), lambda x: lvop2(x) if x.arg[0].func in {RDNA3Ops.VOP2, RDNA3Ops.VOP2_LIT} else
    lvop2(x, swap_only=True) if x.arg[0] in commutative_ins and len(x.src) == 2 else None),
  (UPat(Ops.INS, name="x"), lambda x: lvop3(x) if x.arg[0].func in {RDNA3Ops.VOP3, RDNA3Ops.VOP3SD, RDNA3Ops.VOPC, RDNA3Ops.VOP3P} else None),
  # --- other ---
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), abi),
  (UPat((Ops.INS, Ops.STACK), name="x"), alloc_vregs),
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  (UPat(Ops.STACK, name="x"), lambda ctx,x: stack2regs(ctx, x) if len(x.src) and x.dtype.itemsize < 4 else None),
])

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), lambda ctx,x: lower_gated_load(ctx,x) if ctx.ins_schedule.get(x.arg[0],x.op)
    is Ops.LOAD and x.src[-1].dtype is dtypes.bool and rafter(x.src[-1]).op is not Ops.BUFFER else None),
  (UPat(Ops.INS, name="x"), lambda ctx,x: (x, [(mask := saveexec(ctx, x.src[-1])), x, restoreexec(ctx, mask)])
    if ctx.ins_schedule.get(x.arg[0],x.op) is Ops.STORE and x.src[-1].dtype is dtypes.bool else None),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  # NOTE: forces do_assemble in codegen but might hide incomplete lowering
  (UPat(GroupOp.All - {Ops.INS}, name="x"), lambda x: (x, [])),
])

def encode(x:UOp):
  def encfield(x:UOp):
    x = rafter(x)
    if is_const(x): return const_val(rafter(x))
    r, rs = rdef(x), rdefs(x)
    assert isinstance(r, Register), f"expect Register to encode, got {r} from {x}"
    for i,g in enumerate(rs[1:]):
      assert isinstance(g, Register) and isinstance((nxt := rs[i]), Register) and g.index == nxt.index+1, "wide registers must be contiguous"
    dmap = { "vcc":dsl.VCC, "exec_lo":dsl.EXEC_LO, "v":dsl.v, "s":dsl.s }
    base = next(v for k,v in dmap.items() if k in r.name)
    return base[r.index] if len(rs) == 1 else base[r.index:r.index+len(rs)-1]

  match (group := x.arg[0].func):
    case RDNA3Ops.SCRATCH:
      fields = dict(offset=encfield(x.src[0]), sve=0)
      if rdef(x) is None: fields["data"] = encfield(x.src[1])
      else: fields["vdst"] = encfield(x)
    case RDNA3Ops.GLOBAL:
      fields = dict(addr=encfield(x.src[0]), saddr=encfield(x.src[1]))
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
    case RDNA3Ops.VOP3SD: fields = dict(sdst=dsl.VCC, vdst=encfield(x), **{f"src{i}":encfield(u) for i,u in enumerate(x.src[:3])})
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

  @staticmethod
  def get(u:UOp):
    op = u.arg[0]
    if op.func in { RDNA3Ops.GLOBAL, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }:
      return CntType.STORE_CNT if u.dtype is dtypes.void else CntType.LOAD_CNT
    if op.func in { RDNA3Ops.SMEM, RDNA3Ops.DS }: return CntType.DS_CNT
    return None

class RDNA3PreLinearKernelCtx(PreLinearKernelCtx):
  def __init__(self, sink:UOp, ren:RDNA3Renderer, info:ProgramInfo):
    super().__init__(sink, ren, info)
    # NOTE: entire kernel must fit on single WGP
    # 1536 vgprs per SIMD, 2 SIMD per CU
    # constrain waves*vgpr_limit <<< 1536*2, prevent dispatch hang
    waves = math.ceil(math.prod(info.local_size or (1,)) / 32)
    max_per_thread = min(1536*2 // waves, 256)
    # reserve VGPRs ahead of time to be excluded from normal allocation
    # 12x32 (wave 32) -> 384 spillable SGPR lanes
    n_spill_vgprs = 12
    self.gp_vgprs, self.gp_sgprs = VGPRS[1:max_per_thread-n_spill_vgprs], SGPRS[5:]
    self.spill_vgprs: dict[Register, int] = {r:0 for r in VGPRS[max_per_thread-n_spill_vgprs:max_per_thread]}
    self.execop, self.vccop = self.reserved(EXEC, dtypes.uint32), self.reserved(VCC, dtypes.uint32)

    self.bufregs: dict[tuple[ParamArg, int], UOp] = {}
    self.n_reserved = 0

    # detect buffer overflows, pre-allocate scratch space
    rbufs: dict[int, UOp] = {u.arg.size*u.dtype.itemsize:u for u in sink.toposort() if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG}
    sizes = list(sorted(rbufs.keys(), reverse=True))
    spill_before = next((i for i,sz in enumerate(sizes) if sum(sizes[i:]) < len(self.gp_vgprs)*4), len(sizes))
    self.overflows: dict[UOp, Any] = {}

    for sz in sizes[:spill_before]:
      buf = rbufs[sz]
      vrs = [self.vreg(self.gp_vgprs, width=(buf.dtype.itemsize+3)//4) for i in range(buf.arg.size)]
      self.overflows[buf.arg] = [self.assign_spill_slot(vr, buf) for vr in vrs]

  def bufreg(self, idx:UOp) -> UOp:
    buf, idx = rafter(idx, True).src
    while buf.op is not Ops.BUFFER: buf=buf.src[0]
    ptr = (buf.arg, idx.src[0].val)
    if ptr not in self.bufregs:
      i, width = self.n_reserved, (buf.dtype.itemsize+3)//4
      r = self.gp_vgprs[i:i+width]
      self.n_reserved += width
      self.bufregs[ptr] = self.reserved(r, buf.dtype)
    return self.bufregs[ptr]

  def assign_spill_slot(self, v:VRegister, vdef:UOp) -> int|tuple[Register, int]:
    if v.cons[0] in VGPRS:
      sz = v.cons[0].size * v.width
      offset = self.spill_size + (sz - self.spill_size % sz) % sz
      self.spill_size = offset + sz
      return offset
    else:
      vgpr,lane = next(((r,l) for r,l in self.spill_vgprs.items() if 32 - l >= v.width), (None, None))
      assert vgpr is not None and lane is not None, "ran out of reserved SGPR spill lanes"
      self.spill_vgprs[vgpr] += v.width
      return (vgpr,lane)

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  extra_matcher = extra_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.SHR, Ops.SHL, Ops.MAX, Ops.MULACC)}
  kernel_ctx_type = RDNA3PreLinearKernelCtx
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.renderer.tc import get_amd
    self.shared_max, self.tensor_cores = HIPRenderer.shared_max, get_amd(target.arch)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s}
  def is_two_address(self, x:UOp) -> bool: return False
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    return '\n'.join(str(encode(u).arg[0]) for u in uops)

  def spill(self, spill_offset:Any, x:UOp, sub_idx:int|None=None) -> list[UOp]:
    regs = tuple(r for r in rdefs(x) if isinstance(r, Register))
    if regs[0].name[0] == 'v':
      if sub_idx is not None: spill_offset += sub_idx*4
      return batch_scratch(True, spill_offset, x.dtype, regs)
    else:
      vgpr,lane = spill_offset
      return [UOp(Ops.INS, arg=(RDNA3Ops.v_writelane_b32, dtypes.void), src=(def_reg(x.dtype, r),
        const(lane+i+(sub_idx or 0))), tag=vgpr) for i,r in enumerate(regs)]

  def fill(self, spill_offset:Any, x:UOp, dst:tuple[Register,...], sub_idx:int|None=None) -> tuple[UOp, list[UOp]]:
    if dst[0].name[0] == 'v':
      if sub_idx is not None: spill_offset += sub_idx*4
      ops = batch_scratch(False, spill_offset, x.dtype, dst)
      return UOp(Ops.STACK, src=tuple(ops), tag=dst) if len(ops) > 1 else ops[0], ops
    else:
      vgpr,lane = spill_offset
      movs = [UOp(Ops.INS, src=(def_reg(x.dtype, vgpr),
        const(lane+i+(sub_idx or 0))), arg=(RDNA3Ops.v_readlane_b32, x.dtype), tag=(r,)) for i,r in enumerate(dst)]
      return UOp(Ops.STACK, src=tuple(movs), tag=dst), movs

  def copy(self, u:UOp, dst:VRegister|Register|tuple[Register,...]) -> tuple[UOp, list[UOp]]:
    slots, tag = copy_dst(dst)
    if len(slots) == 1: return (mov := vmov(u, slots[0])), [mov]
    # NOTE: the STACK gets one src per slot, the geps are only emitted as operands
    movs, ins = [], []
    if u.op is Ops.STACK:
      ins = movs = [vmov(s, r) for s,r in zip(u.src, slots)]
    else:
      for i,r in enumerate(slots):
        ins.extend([g := gep(u,i), mov := vmov(g, r)])
        movs.append(mov)
    grp = UOp(Ops.STACK, src=tuple(movs), tag=tag)
    return grp, ins + [grp]

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    deps: set[Register] = set()
    nuops, pending_scratch, pending_lds = [], False, False

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
          deps.update([r for r in rdefs(u) if isinstance(r, Register)])
          if u.arg[1] is dtypes.void: # protect address registers?
            for s in u.src: deps.update([r for r in rdefs(u) if isinstance(r, Register)])
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
    return assemble_linear(prg, lin, self.target.arch, scratch_size=lin.arg or 0)

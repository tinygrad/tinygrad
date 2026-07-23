from tinygrad.dtype import dtypes, AddrSpace, truncate, DType
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, VRegister, rdefs, rdef
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
from dataclasses import dataclass, field
import itertools

# ---- (UOp, dtype) -> Instruction tables ----
dt_to_isa = { dtypes.int32:"i32", dtypes.uint32:"u32", dtypes.float32:"f32", dtypes.float64:"f64", dtypes.float16:"f16", dtypes.int16:"i16", dtypes.uint16:"u16", dtypes.uint64:"u64", dtypes.int64:"i64" }
isa_to_dt = { v:k for k,v in dt_to_isa.items() }

# (uop, prefix, opcodes, support 32 and 64 bit encoding (e32/e64 branches with keys))
insdefs = [
  (Ops.ADD, "v_add", ["f16_e32", "f32_e32", "f64", "nc_i32", "nc_u32_e32", "nc_u16", "nc_i16"], False),
  (Ops.SUB, "v_sub", ["f16_e32", "f32_e32", "nc_i32", "nc_i16", "nc_u16", "nc_u32_e32"], False),
  (Ops.MUL, "v_mul", ["f16_e32", "f32_e32", "f64", "lo_u32", "lo_u16"], False), # TODO: mul i16?
  (Ops.SQRT, "v_sqrt", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.LOG2, "v_log", ["f16_e32", "f32_e32"], False),
  (Ops.SIN, "v_sin", ["f32_e32"], False),
  (Ops.EXP2, "v_exp", ["f16_e32", "f32_e32"], False),
  (Ops.RECIPROCAL, "v_rcp", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.MAX, "v_max", ["f16_e32", "f32_e32", "u16", "i16", "u32_e32", "i32_e32"], False),
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
V_FMA =   { dtypes.float16:RDNA3Ops.v_fma_f16,      dtypes.float32:RDNA3Ops.v_fma_f32,      dtypes.float64:RDNA3Ops.v_fma_f64       }

def _cvt_ins(dtin, dtout):
  _valid_casts = {
      dtypes.float64    : (dtypes.int32, dtypes.float32, dtypes.uint32),
      dtypes.int32      : (dtypes.float64, dtypes.float32),
      dtypes.uint32     : (dtypes.float32, dtypes.float64),
      dtypes.float32    : (dtypes.float64, dtypes.uint32, dtypes.int32, dtypes.float64, dtypes.float16),
      dtypes.float16    : (dtypes.float32, dtypes.uint16, dtypes.int16),
      dtypes.int16      : (dtypes.int32, dtypes.float16),
      dtypes.uint16     : (dtypes.uint32, dtypes.float16)
  }
  assert dtin in _valid_casts and dtout in _valid_casts[dtin], f"cannot natively cast from {dtin} -> {dtout}"
  return getattr(RDNA3Ops, f"v_cvt_{dt_to_isa[dtout]}_{dt_to_isa[dtin]}_e32")

# ---- helpers ----
def reg(u:UOp): return rs[0] if len((rs := rdefs(u))) >= 1 else None
def def_reg(dt, reg:Register|tuple[Register,...]): return UOp.placeholder((1,), dt, next(lane_ctr), AddrSpace.REG).replace(tag=(reg,) if isinstance(reg,Register) else reg)
def const(dt, v:int) -> UOp: return UOp.const(dt,truncate[dt](v)).rtag()
def make_vgpr(ctx, width:int=1) -> Register: return ctx.vreg(GP_VGPRS, width=width)
def vmov(x:UOp) -> UOp: return x.ins(RDNA3Ops.v_mov_b16_e32 if x.dtype.itemsize == 2 else RDNA3Ops.v_mov_b32_e32, src=(x,))
def is_const(x:UOp): return is_const(x.src[0]) if x.op in {Ops.CAST, Ops.BITCAST, Ops.AFTER} else x.op is Ops.CONST
def to_vgpr(ctx, x:UOp) -> UOp: return vmov(x) if is_const(x) else x
def const_vgpr(ctx, dt, v:int) -> UOp: return to_vgpr(ctx, const(dt, v))
# NOTE: call this buildvector like LLVM?
def multireg(*args, dtype:DType): return UOp.group(*args).replace(dtype=dtype)
def getsign(u:UOp, nbits):
  if nbits < 32: u = UOp(Ops.SHL, dtypes.uint32, src=(u, const(dtypes.uint16, 32 - nbits)))
  return _aluhint(UOp(Ops.SHR, dtypes.uint32 if nbits <= 32 else dtypes.uint64, src=(u, const(dtypes.uint16, 31 if nbits <= 32 else 63))), RDNA3Ops.v_ashrrev_i32_e32 if nbits <= 32 else RDNA3Ops.v_ashrrev_i64)

# ---- register classes/kernel init state ----
VGPRS = tuple(Register(f"v{i}", i, size=4) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i, size=4) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
GP_SGPRS, GP_VGPRS = tuple(SGPRS[5:]), tuple(VGPRS[1:])
VCC, EXEC = Register("vcc", 0, size=4), Register("exec_lo", 0, size=4)
FLAT_SCRATCH_LO, FLAT_SCRATCH_HI = Register("flat_scratch_lo", 0, size=4), Register("flat_scratch_hi", 0, size=4)
lane_ctr = itertools.count()

kernarg_ptr = (def_reg(dtypes.uint32, KERNARG_PTR[0]), def_reg(dtypes.uint32, KERNARG_PTR[1]))
execop, lidop, vccop = def_reg(dtypes.uint32, EXEC), def_reg(dtypes.uint32, WIIDS[0]), def_reg(dtypes.uint32, VCC)
flat_scratch_ptr = (def_reg(dtypes.uint32, FLAT_SCRATCH_LO), def_reg(dtypes.uint32, FLAT_SCRATCH_HI))

# ---- register movement helpers ----
def packb16(ctx, lo:UOp, hi:UOp):
  if dtypes.is_float(lo.dtype): return UOp(Ops.INS, arg=RDNA3Ops.v_pack_b32_f16, src=(lo,hi))
  lo = lo & const(dtypes.uint32, 0xFFFF) # mask off upper half
  return _vop3(ctx, UOp(Ops.INS, arg=RDNA3Ops.v_lshl_or_b32, src=(hi, const(dtypes.int, 16), lo)))

def stack2regs(ctx, x:UOp, vreg:VRegister|None=None):
  nregs, mvs = ((len(x.src) * x.dtype.itemsize) + 3) // 4, []
  for i in range(nregs):
    if x.dtype.itemsize == 2:
      if i*2+1 < len(x.src): mvs.append(packb16(ctx, x.src[i*2], x.src[i*2+1]))
      else: mvs.append(vmov(x.src[i*2]))
    else: mvs.append(vmov(x.src[i]))
  nx = multireg(*mvs, dtype=x.dtype)
  if vreg is not None: nx = nx.replace(src=tuple(s.replace(tag=(vreg.sub(i),)) for i,s in enumerate(x.src)), tag=(vreg,))
  return nx

# ---- operand legalization wrappers ----
def _vop3(ctx, x:UOp):
  lits = [i for i,s in enumerate(x.src) if s.op is Ops.CONST]
  if len(lits) <= 1: return x
  new = list(x.src)
  for i in lits[1:]: new[i]=to_vgpr(ctx, new[i])
  return x.replace(src=tuple(new))

# TODO: pass in original op to use GroupOp.COMMUTATIVE?
rev_op_order = { RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshlrev_b16, RDNA3Ops.v_lshlrev_b64, RDNA3Ops.v_lshrrev_b32_e32, RDNA3Ops.v_lshrrev_b16, RDNA3Ops.v_lshrrev_b64, RDNA3Ops.v_ashrrev_i32_e32, RDNA3Ops.v_ashrrev_i64 }
def _vop2(ctx, x:UOp):
  assert x.op is Ops.INS, f"should only legalize INS ops: {x.op}"
  if x.arg in rev_op_order: x = x.replace(src=x.src[2::-1] + x.src[2:])
  # def _isvgpr(u:UOp): return (r := reg(u)) is not None and isinstance(r, Register) and r.cons[0].name[0] == "v"
  if not is_const(x.src[1]): return x
  rest = x.src[2:] if len(x.src) > 2 else ()
  non_commutative = x.arg in (RDNA3Ops.v_ashrrev_i32_e32, RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshrrev_b32_e32) # NOTE: add more
  if not non_commutative and not is_const(x.src[0]): return x.replace(src=(x.src[1], x.src[0]) + rest)
  return x.replace(src=(x.src[0], vmov(x.src[1])) + rest)

# TODO: allocate vgpr / sgpr based on op group (x.arg.func)
# - should almost never need to manually call ctx.vreg, control flow allocations should also be handled here?
def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  if x.op is Ops.GROUP and (len(x.src) == 0 or x.src[0].op is not Ops.INS): return None
  if x.dtype is dtypes.void: return None
  if x.op is Ops.BUFFER and (x.addrspace is not AddrSpace.REG or x.max_numel() > 1): return None
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], VRegister): return None
  if isinstance(x.tag, tuple): assert x.tag, f"got empty tuple for op: {x.op}, {x.arg}"

  if x.op is Ops.GROUP:
    sgpr = x.src[0].arg.func.__name__[0] == 'S'
    vreg = ctx.vreg(GP_SGPRS if sgpr else GP_VGPRS, width=len(x.src))
    return x.replace(tag=(vreg,), src=tuple(s.replace(tag=(vreg.sub(i),)) for i,s in enumerate(x.src)))

  defs = []
  if isinstance(x.tag, tuple):
    cons, width = x.tag, 1
    if isinstance(x.tag[0], tuple): cons, width = x.tag
    defs = [ctx.vreg(x.tag, width=width)]
  else:
    # NOTE: reg buffer doesn't actually need contiguous invariant
    n = max(x.dtype.itemsize // 4, 1)
    defs = [make_vgpr(ctx, width=n)]
  return x.replace(tag=tuple(defs))

# TODO: batch param loading? ex. s_load_b128
# https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  i = ctx.func_args.index(x)
  if x.op is Ops.SPECIAL: # maintain src edge?
    dim = int(x.arg[-1])
    if x.arg[0] == 'g': return UOp(Ops.INS, dtype=dtypes.uint32, arg=RDNA3Ops.v_mov_b32_e32, src=(def_reg(dtypes.uint32, WGIDS[dim]),))
    else: return x.ins(RDNA3Ops.v_bfe_u32, dtype=dtypes.uint32, src=(lidop, const(dtypes.uint32, 10 * dim), const(dtypes.uint32, 10)))
  offs = sum(8 if u.op == Ops.PARAM else 4 for u in ctx.func_args[:i])
  # if AddrSpace is ALU auto load into vgpr
  if x.addrspace is AddrSpace.ALU:
    val = x.ins(RDNA3Ops.s_load_b32, src=kernarg_ptr + (const(dtypes.uint32, offs),), tag=(ctx.vreg(GP_SGPRS),))
    return UOp(Ops.INS, arg=RDNA3Ops.v_mov_b32_e32, dtype=x.dtype, src=(val,))
  return x.ins(RDNA3Ops.s_load_b64, dtype=dtypes.ulong, src=kernarg_ptr + (const(dtypes.uint32, offs),), tag=(ctx.vreg(GP_SGPRS, width=2, alignment=2),))

# ----- memory access ----
# GLOBAL_ADDR = VADDR_U64 + IMMOFFS_u16
def fold_global(ctx, base:UOp, idx:UOp): # (saddr, voff, ioffs)
  disp_scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = const(dtypes.int, disp_scale.bit_length() - 1)
  vaddr, offs = idx, const(dtypes.int16, 0)
  if idx.op is Ops.CONST: vaddr = idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.int, idx.arg),))
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST and \
    -(1 << 12) <= (_offs := idx.src[1].arg * disp_scale) < (1 << 12):
    vaddr, offs = idx.src[0], const(dtypes.int16, _offs)
  vaddr = UOp(Ops.SHL, dtype=dtypes.uint64, src=(castint64(ctx, vaddr, dtypes.uint64), shft))
  return (UOp(Ops.ADD, dtype=dtypes.uint64, src=(vaddr, base.bitcast(dtype=dtypes.uint64))), offs)

# LDS_ADDR = VGPR_ADDR_u32 + imm_byte_offset_u16
# NOTE: keep base in src to maintain graph dependencies?
def fold_lds(ctx, base:UOp, idx:UOp): # (vaddr, ioffs)
  # TODO: actually calculate lds offset per seperate BUFFER, need some way to know what # this is and
  # the size of the other ones. Use isel ctx?
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32,0),)), const(dtypes.uint16, idx.arg * scale), base)
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0].cast(dtypes.uint32), const(dtypes.uint16, idx.src[1].arg * scale), base)
  # NOTE: manual SHL construction to avoid none shape error mixing with Ops.INS? fix this somehow
  shft = const(dtypes.uint32, scale.bit_length() - 1)
  offs = UOp(Ops.SHL, dtypes.uint32, src=(idx,shft))
  return (offs, const(dtypes.uint16, 0), base)

def fold_address(ctx, x:UOp): return fold_lds(ctx, *x.src[:2]) if x.addrspace is AddrSpace.LOCAL else fold_global(ctx, *x.src[:2])
def _insspace(gl,x): return gl[1] if x.addrspace is AddrSpace.LOCAL else gl[0]

def load(ctx, addr:UOp, x:UOp, gate:UOp|None = None, alt:UOp|None = None):
  alt, gate = x.src[1:] if len(x.src) > 1 else (None,None)
  base, idx = addr.src[:2]
  # NOTE: the problem with indexing into base with b64 dtypes is that it will interpret it as accessing a subreg??
  if base.addrspace is AddrSpace.REG: 
    # TODO: gated reg load
    assert idx.op is Ops.CONST and gate is None, "gated load on reg BUFFER"
    if base.dtype.itemsize <= 4: return base.index(idx)
    else: return multireg(vmov(base.index(0)), vmov(base.index(1)), dtype=base.dtype)
  # NOTE: load_i* automatically sign extends, this messes up some of the tests currently ex. i8 bitcast_alt
  imap = {
    1 : [(RDNA3Ops.global_load_u8,RDNA3Ops.ds_load_u8), (RDNA3Ops.global_load_i8,RDNA3Ops.ds_load_i8)],
    2 : [(RDNA3Ops.global_load_u16,RDNA3Ops.ds_load_u16), (RDNA3Ops.global_load_i16,RDNA3Ops.ds_load_i16)],
    4 : (RDNA3Ops.global_load_b32,RDNA3Ops.ds_load_b32),
    8 : (RDNA3Ops.global_load_b64,RDNA3Ops.ds_load_b64),
    16 : (RDNA3Ops.global_load_b128,RDNA3Ops.ds_load_b128),
  }
  n = addr.src[-1].arg if addr.op is Ops.SHRINK else 1
  nregs = (n * base.dtype.itemsize+3)//4
  vreg = make_vgpr(ctx, width=nregs)
  nbytes = n * base.dtype.itemsize
  tupins = imap[nbytes] if nbytes > 2 else imap[nbytes][not (dtypes.is_unsigned(x.dtype) or dtypes.is_float(x.dtype))]
  folded = fold_address(ctx, addr)
  nx = x.ins(_insspace(tupins, base), src=folded, tag=(vreg,))
  if gate is not None:
    if alt.op is Ops.GROUP: packed = alt.replace(src=tuple(s.replace(src=s.src + folded, tag=(vreg.sub(i),)) for i,s in enumerate(alt.src)), tag=(vreg,))
    else: packed = vmov(alt).replace(src=(alt,) + folded, tag=(vreg,))
    return nx.replace(src=(packed,) + nx.src  + (gate,def_reg(dtypes.uint32,GP_SGPRS)))
  return nx

def store(ctx, addr:UOp, x:UOp):
  val = x.src[1]
  gate = x.src[2] if len(x.src) > 2 else None
  base, idx = addr.src[:2]
  if base.addrspace is AddrSpace.REG:
    if len(rdefs(base)) == 0: return None
    # keep addr as a control dep so reduce-identity stores re-run inside their ranges
    if base.dtype.itemsize <= 4:
      mov = vmov(val).replace(tag=rdefs(base))
      return mov.replace(src=mov.src+(addr,))
    else:
      buf = base.src[0] if base.op is Ops.BUFFER else base.src[0].src[0]
      assert buf.arg == 1, f"reg buf of multiple ({buf.arg}) 2 reg values"
      vreg = rdef(base)
      ms = [vmov(val.index(i)).replace(tag=(vreg.sub(i),)) for i in range(vreg.width)]
      return UOp.group(*[m.replace(src=m.src+(addr,)) for m in ms])

  def _gate(o:UOp): return o.replace(src=o.src + (gate,def_reg(dtypes.uint32,GP_SGPRS))) if gate is not None else o
  n = addr.src[-1].arg if addr.op is Ops.SHRINK else 1
  nregs = (n*addr.dtype.itemsize+3)//4
  imap = {
    1:(RDNA3Ops.global_store_b8,RDNA3Ops.ds_store_b8),
    2:(RDNA3Ops.global_store_b16,RDNA3Ops.ds_store_b16),
    4:(RDNA3Ops.global_store_b32,RDNA3Ops.ds_store_b32),
    8:(RDNA3Ops.global_store_b64,RDNA3Ops.ds_store_b64),
    16:(RDNA3Ops.global_store_b128,RDNA3Ops.ds_store_b128)
  }
  return _gate(UOp(Ops.INS, arg=_insspace(imap[n * addr.dtype.itemsize],base), dtype=dtypes.void, src=fold_address(ctx, addr) + (to_vgpr(ctx,val),)))

# ------ ALU ------
def cvt(ctx, y:UOp, x:UOp): # TODO: b64 -> b64
  def _needcast(x:DType, y:DType): return not (dt_to_isa[x][0] == dt_to_isa[y][0])
  if x.dtype in (dtypes.uint64, dtypes.int64) and y.dtype.itemsize == 4: # b32 -> b64
    targ = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
    lo = y.ins(_cvt_ins(y.dtype, targ)) if _needcast(y.dtype, targ) else y
    return to_vgpr(ctx, UOp(Ops.STACK, src=(lo, const(targ, 0))))
  elif y.dtype.itemsize == 8 and x.dtype.itemsize == 4 and y.dtype is not dtypes.float64: # b64 -> b32
    src = dtypes.uint32 if dtypes.is_unsigned(y.dtype) else dtypes.int32
    if _needcast(src, x.dtype): return x.ins(_cvt_ins(src, x.dtype), src=(y.index(0),))
    else: return y.index(0)
  return x.ins(_cvt_ins(y.dtype,x.dtype))

def cmp(ctx, x:UOp):
  _mask_cmp = { Ops.CMPNE:RDNA3Ops.s_xor_b32, Ops.XOR:RDNA3Ops.s_xor_b32, Ops.OR: RDNA3Ops.s_or_b32, Ops.AND:RDNA3Ops.s_and_b32, Ops.CMPLT: RDNA3Ops.s_and_not1_b32, Ops.CMPEQ:RDNA3Ops.s_xnor_b32 }
  scmp = x.src[0].dtype is dtypes.bool and x.src[1].dtype is dtypes.bool
  ins = _mask_cmp[x.op] if scmp else OP_INS[x.op][64][x.src[0].dtype]
  if scmp and x.op is Ops.CMPLT: x=x.replace(src=(x.src[1], x.src[0]))
  x = x.ins(ins, tag=GP_SGPRS)
  return x if scmp else _vop3(ctx, x)

def arith64(ctx, x:UOp, add:bool):
  if dtypes.is_float(x.dtype):
    assert add
    return x.ins(RDNA3Ops.v_add_f64)
  a, b = x.src
  ins_lo = RDNA3Ops.v_add_co_u32 if add else RDNA3Ops.v_sub_co_u32
  ins_hi = RDNA3Ops.v_add_co_ci_u32 if add else RDNA3Ops.v_sub_co_ci_u32
  narrow = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  vreg = make_vgpr(ctx, width=2) # NOTE: after causes a problem for auto allocating group reg
  lo = UOp(Ops.INS, dtype=dtypes.uint32, arg=ins_lo, src=(a.index(0), b.index(0)), tag=(vreg.sub(0),))
  hi = UOp(Ops.INS, dtype=narrow, arg=ins_hi, src=(a.index(1), b.index(1), vccop, lo), tag=(vreg.sub(1),)).after(lo)
  return multireg(lo, hi, dtype=x.dtype).replace(tag=(vreg,))

# a64 * b64 = (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo) =  a_hi * 2^32 * b_lo + b_hi * 2^32 * a_hi + a_lo * b_lo
def mul64(ctx, x:UOp):
  if dtypes.is_float(x.dtype): return x.ins(RDNA3Ops.v_mul_f64)
  def _mad(a:UOp, b:UOp, c:UOp=const(x.dtype,0)): return UOp(Ops.INS, x.dtype, arg=RDNA3Ops.v_mad_u64_u32, src=(a,b,c))
  def _up(x:UOp): return x.ins(RDNA3Ops.v_lshlrev_b64, src=(const(dtypes.int,32),x))
  a, b = x.src
  sign = not dtypes.is_unsigned(x.dtype)
  shup = const(dtypes.int, 32)
  p1 = _up(_mad(a.index(1), b.index(0)))
  p2 = _up(_mad(a.index(0), b.index(1)))
  p3 = arith64(ctx, UOp(Ops.ADD, x.dtype, src=(p1,p2)), add=True)
  return _mad(a.index(0), b.index(0), p3)

def bitwise64(ctx, x:UOp, ins):
  a, b = x.src
  lo = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.index(0), b.index(0)))
  hi = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.index(1), b.index(1)))
  return multireg(lo, hi, dtype=x.dtype)

# Allows embedding special alu instructions ex. mul_hi without introducing
# Ops.INS which have None shape and cause alu() _broadcast to error
def _aluhint(x:UOp, hint:InsOp): return x.replace(arg=hint)

# https://arxiv.org/pdf/2207.08420
def idiv(ctx, x:UOp):
  signed = not dtypes.is_unsigned(x.dtype)
  dt = dtypes.uint32 if x.dtype.itemsize <= 4 else dtypes.uint64
  a, b = x.src[0].cast(dt), x.src[1].cast(dt)
  if signed:
    nbits = x.dtype.itemsize*8
    sa, sb = getsign(a, nbits), getsign(b, nbits)
    a, b = (a + sa) ^ sa, (b + sb) ^ sb
    sign = sa ^ sb
  bs = b.cast(dtypes.float)
  ad, bd = a.cast(dtypes.double), b.cast(dtypes.double)
  invbs0  = bs.reciprocal()
  invbd0 = invbs0.cast(dtypes.double)
  alpha = -bd * invbd0 + const(dtypes.double, 1.0)
  invbd = alpha * invbd0 + invbd0
  qd = ad * invbd
  q1 = _aluhint(qd.trunc(), RDNA3Ops.v_rndne_f64_e32).cast(dtype=dtypes.uint64) # todo: this is hacky, not trunc
  r1 = UOp(Ops.SUB, dtypes.int64, src=(a.cast(dtypes.int64), b.cast(dtypes.int64) * q1.cast(dtypes.int64)))
  if x.dtype.itemsize <= 4:
    q = (r1 < const(dtypes.int64, 0)).where(UOp(Ops.SUB, dtypes.ulong, src=(q1, const(dtypes.ulong, 1))), q1).cast(dtypes.uint32)
  else:
    q3d = r1.cast(dtypes.double) * invbd
    q3 = _aluhint(q3d.trunc(), RDNA3Ops.v_rndne_f64_e32).cast(dtypes.int64)
    r3 = UOp(Ops.SUB, dtypes.int64, src=(r1, b.cast(dtypes.int64) * q3))
    q2 = (r3 < const(dtypes.int64, 0)).where(UOp(Ops.SUB, dtypes.int64, src=(q3, const(dtypes.int64, 1))), q3)
    q0 = q1 + q2.cast(dtypes.uint64)
    is_big = b.cast(dtypes.int64) < const(dtypes.int64, 0) # b >= 2^63
    is_one = b <= const(dtypes.ulong, 1)
    if_big = (a >= b).cast(dtypes.uint64)
    special = is_big.where(if_big, a)
    q = (is_one | is_big).where(special, q0)
  return (q ^ sign) + -sign if signed else q

_lshl = { 2:RDNA3Ops.v_lshlrev_b16, 4:RDNA3Ops.v_lshlrev_b32_e32, 8:RDNA3Ops.v_lshlrev_b64 }
_lshr = { 2:RDNA3Ops.v_lshrrev_b16, 4:RDNA3Ops.v_lshrrev_b32_e32, 8:RDNA3Ops.v_lshrrev_b64 }
def alu(ctx, x:UOp):
  dpreciz = x.dtype.itemsize == 8
  if dpreciz and x.op is Ops.ADD: return arith64(ctx, x, add=True)
  if dpreciz and x.op is Ops.SUB: return arith64(ctx, x, add=False)
  if dpreciz and x.op is Ops.MUL: return mul64(ctx, x)

  # NOTE: ignore b16 instructions for now
  def _bitwise(sins:InsOp, hins:InsOp): return bitwise64(ctx, x, sins) if dpreciz else _vop2(ctx, x.ins(sins))
  if x.op is Ops.AND: return _bitwise(RDNA3Ops.v_and_b32_e32, RDNA3Ops.v_and_b16)
  elif x.op is Ops.OR: return _bitwise(RDNA3Ops.v_or_b32_e32, RDNA3Ops.v_or_b16)
  elif x.op is Ops.XOR: return _bitwise(RDNA3Ops.v_xor_b32_e32, RDNA3Ops.v_xor_b16)

  if x.op is Ops.SHL:
    ins = _lshl[max(2, x.dtype.itemsize)]
  elif x.op is Ops.SHR:
    if x.dtype is dtypes.int32: ins = RDNA3Ops.v_ashrrev_i32_e32 # TODO: handle 64 bit
    else: ins = _lshr[max(2,x.dtype.itemsize)]

  if isinstance(x.arg, InsOp): ins = x.arg # used for instruction overrides, ex. mul_hi for cdiv
  elif x.op in OP_INS:
    dt = x.dtype
    if x.op is Ops.MUL and x.dtype is dtypes.int: dt = dtypes.uint32
    if dt in OP_INS[x.op]: ins = OP_INS[x.op][dt]
  elif x.op not in {Ops.SHL, Ops.SHR}: raise NotImplementedError(f"alu optype not implemented. op={x.op}, dtype={x.dtype}")
  return x.ins(ins) if len(x.src) == 1 else _vop2(ctx, x.ins(ins))

# ---- casting utilities -----
# NOTE: make this a pm?
def intcast(y:UOp, x:UOp):
  # NOTE: use v_bfe instead of hand rolled masking
  if y.dtype.itemsize == x.dtype.itemsize: return y  # same size noop
  if x.dtype.itemsize > y.dtype.itemsize:
    if y.dtype.itemsize == 1: return y.bitcast(x.dtype)
    if x.dtype.itemsize == 2: return (y & const(dtypes.uint32, 0xFFFF)).bitcast(x.dtype)
    return (y & const(y.dtype, 0xFFFFFFFF)).bitcast(x.dtype)
  if y.dtype.itemsize <= 4 and x.dtype.itemsize < y.dtype.itemsize: # masked narrow
    if x.dtype.itemsize == 2: return (y & const(y.dtype, 0xFFFF)).bitcast(x.dtype)
    return (y & const(y.dtype, 0xFF)).bitcast(x.dtype)

# TODO: move this into pattern matcher
# NOTE: this needs work, maybe cleaner to define 2 reg buffer and just .store()
def castint64(ctx, y:UOp, tdt:DType):
  hi_dt = dtypes.uint32 if dtypes.is_unsigned(tdt) else dtypes.int32
  if y.dtype in dtypes.ints:
    do_sext = not dtypes.is_unsigned(y.dtype)
    if do_sext:
      nbits = y.dtype.itemsize*8
      hi = getsign(vmov(y), nbits)
      # extend sign to upper part of low
      lo = vmov(y) if y.dtype.itemsize >= 4 else UOp(Ops.OR, dtypes.uint32, src=(vmov(y), UOp(Ops.AND, dtypes.uint32, src=(hi, const(dtypes.uint32, ~((1 << nbits) - 1)))))) # TODO: cleanup manual constr.
    else: lo, hi = vmov(y), vmov(const(dtypes.uint32, 0))
    return multireg(lo, hi, dtype=tdt)
  elif y.dtype is dtypes.float64: # https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp#L3691
    tr = UOp(Ops.TRUNC, dtypes.float64, src=(y,))
    hi_f = tr.ins(RDNA3Ops.v_ldexp_f64, src=(tr,const(dtypes.int16, -32)))
    hi_f = UOp(Ops.INS, dtypes.float64, arg=RDNA3Ops.v_floor_f64_e32, src=(hi_f,))
    lo_f = hi_f.ins(RDNA3Ops.v_ldexp_f64, src=(hi_f, const(dtypes.int16, 32))) # tr - hi_f * 2 ^ 32
    lo_f = UOp(Ops.ADD, dtypes.float64, src=(tr, UOp(Ops.MUL, dtypes.float64, src=(lo_f, const(dtypes.float64, -1.)))))
    return multireg(lo_f.cast(dtypes.uint32), hi_f.cast(hi_dt), dtype=tdt)
  raise NotImplementedError()

# TODO: currently only 53 bit precision (f64 mantissa), could do better
def long2double(x:UOp):
  lo = x.index(0).replace(dtype=dtypes.uint32).cast(dtypes.float64)
  hi = x.index(1).replace(dtype=dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32).cast(dtypes.float64)
  hi = hi.ins(RDNA3Ops.v_ldexp_f64, src=(hi,const(dtypes.int16, 32)))
  return UOp(Ops.ADD, dtype=dtypes.float64, src=(lo,hi))

# casting between long/ulong and floats is more complicated, may belong in isel?
def const64(x:UOp):
  v = x.arg.bits if dtypes.is_float(x.dtype) else x.arg
  hi_dt = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  return multireg(vmov(const(dtypes.uint32,v)), vmov(const(hi_dt, v >> 32)), dtype=x.dtype)

# ---- control flow ----
def restoreexec(mask:UOp) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_or_b32, src=(execop,mask), tag=(EXEC,))
def label(ctx, name:str) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_nop, tag=name)
memgroups = { RDNA3Ops.GLOBAL, RDNA3Ops.SMEM, RDNA3Ops.DS, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }

def lower_gated(ctx, x:UOp):
  if x.arg.func not in memgroups: return None
  store = x.dtype is dtypes.void
  gated_store, gated_load = store and len(x.src) > 4, not store and len(x.src) > 3
  if not (gated_store or gated_load): return None
  skip_label = "_".join(str(i) for i in x.src)
  lbl = label(ctx, f".EXIT_{skip_label}")
  skip = UOp(Ops.INS, arg=RDNA3Ops.s_cbranch_execz, tag=f".EXIT_{skip_label}")
  save = x.src[-1].ins(RDNA3Ops.s_and_saveexec_b32, src=(x.src[-2],))
  nsrc = x.src[:-2] if gated_store else x.src[1:-2]
  nx = x.replace(src=nsrc)
  line = [save, skip, nx, lbl, restoreexec(x.src[-1])]
  return nx, line

def prep_range(ctx, bnd:UOp, x:UOp):
  if x.dtype is dtypes.uint32: return None # this is a shit predicate, maybe utilize ctx
  mask = def_reg(dtypes.uint32, GP_SGPRS)
  # keep control-flow edges from pm_add_control_flow (src[1:]) so nest/sibling order is preserved through linearize
  return x.replace(src=(bnd,)+x.src[1:]+(mask,)).replace(dtype=dtypes.uint32)

def prep_end(ctx, x:UOp, rng:UOp):
  if not (len(x.src) == 2 and rng.dtype is dtypes.uint32): return None
  one = const_vgpr(ctx,dtypes.uint32,1)
  mask, bnd = rng.src[-1], to_vgpr(ctx, rng.src[0])
  return x.replace(src=x.src + (bnd,one,mask))

# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SILowerControlFlow.cpp#L423
def lower_range(ctx, x:UOp):
  loop_label = "_".join(str(i) for i in x.arg[:-1])
  acc = x.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32,0),))
  mask = x.src[-1].ins(RDNA3Ops.s_mov_b32, src=(execop,))
  ctx.loop_label[acc] = loop_label
  lbl = label(ctx, f".LOOP_{loop_label}")
  return acc, [acc, mask, lbl]

def lower_end(ctx, x:UOp):
  gate = UOp(Ops.INS, arg=RDNA3Ops.v_cmpx_lt_u32_e64, src=(x.src[1], x.src[-3]), tag=(EXEC,))
  inc = x.src[1].ins(RDNA3Ops.v_add_nc_u32_e32, src=(x.src[1], x.src[-2])) # TODO: one fold to imm
  loop = UOp(Ops.INS, arg=RDNA3Ops.s_cbranch_execnz, tag=f".LOOP_{ctx.loop_label[x.src[1]]}")
  return inc, [inc, gate, loop, restoreexec(x.src[-1])]

# --- other stuff ---
# NOTE: this should just be triggered in to_vgpr????
def gethalf(x:UOp, buf:UOp, idx:UOp):
  i = idx.arg
  b32 = buf.index(UOp.const(dtypes.int, i // 2)).replace(dtype=dtypes.uint32)
  if i % 2 != 0: return (b32 >> const(dtypes.uint32,16)).bitcast(x.dtype)
  else: return x.ins(RDNA3Ops.v_mov_b16_e32, src=(b32,))

# NOTE: handle 64 bit where??, should be 2 32 bit cndmasks
def where(ctx, pred:UOp, a:UOp, b:UOp, x:UOp):
  if x.dtype is dtypes.bool: return (pred & a) | (~pred & b)
  ins = RDNA3Ops.v_cndmask_b32_e64 if x.dtype.itemsize >= 4 else RDNA3Ops.v_cndmask_b16
  return _vop3(ctx, x.ins(ins, src=(b,a,pred)))

def bufreg(ctx, x:UOp):
  # we need to rewrite the buffer reference to a scalar buffer
  # NOTE: for now assume 1 layer deep ex. AFTER
  buf = x.src[0] if x.src[0].op is Ops.BUFFER else x.src[0].src[0]
  if buf.max_numel() == 1: return None
  sbuf = UOp.placeholder((1,), x.dtype, ctx.buf_slots.setdefault((buf, x.src[1].arg), next(lane_ctr)), AddrSpace.REG).replace(tag=GP_VGPRS)
  nbase = sbuf if x.src[0].op is Ops.BUFFER else x.src[0].replace(src=(sbuf,) + x.src[0].src[1:])
  nx = nbase.index(0)
  return nx.replace(src=nx.src + x.src[2:])

wmma_op = {
  (dtypes.float32, dtypes.float16) : RDNA3Ops.v_wmma_f32_16x16x16_f16,
  (dtypes.float32, dtypes.bfloat16) : RDNA3Ops.v_wmma_f32_16x16x16_bf16,
  (dtypes.float16, dtypes.float16) : RDNA3Ops.v_wmma_f16_16x16x16_f16,
  (dtypes.bfloat16, dtypes.bfloat16) : RDNA3Ops.v_wmma_bf16_16x16x16_bf16,
}
def render_wmma(ctx, x:UOp):
  a,b,acc = x.src
  ins = getattr(RDNA3Ops, f"v_wmma_{dt_to_isa[a.dtype]}_16x16x16_{dt_to_isa[b.dtype]}")
  return UOp(Ops.INS, arg=ins, src=(a,b,acc), tag=(make_vgpr(ctx, width=8),))

# ---- lowering passes ----
from tinygrad.renderer.cstyle import create_non_native_float_pats, pm_manual_bf16_cast
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
from tinygrad.dtype import to_storage_scalar
extra_matcher = PatternMatcher([
  (UPat.cvar("x", dtype=dtypes.bfloat16), lambda x: const(dtypes.uint16, to_storage_scalar(x.arg, dtypes.bfloat16)).bitcast(dtypes.bfloat16)),
  (UPat(Ops.EXP2, dtypes.double, src=(UPat.var("d"),)), xexp2),
  (UPat(Ops.LOG2, dtypes.double, src=(UPat.var("d"),)), xlog2),
  (UPat(Ops.CMOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - b * a.alu(Ops.CDIV, b)), # hack from x86
  # prevent 64 bit immediate from being realized into 2 regs for shift
  (UPat((Ops.SHR, Ops.SHL), dtypes.int64s+(dtypes.float64,), src=(UPat(), UPat.cvar("y")), name="x"), lambda y,x: x.replace(src=(x.src[0], y.replace(dtype=dtypes.uint32)))),
]) + pm_manual_bf16_cast + create_non_native_float_pats((dtypes.bfloat16,))

# TODO: Ops.NEG should be 0 - x, 64 bit consts should also be folded into imm field when possible
def _smux(dt:DType, sdt:DType, udt:DType): return udt if dtypes.is_unsigned(dt) else sdt
# cast i8 -> i16/i32 = bfe
# NOTE: down casting float to int should round first then reduce precision
pre_isel_matcher = PatternMatcher([
  # NOTE: does src order matter for upat?, maybe thats an arg
  # (UPat(Ops.ADD, src=[UPat(Ops.MUL, src=(UPat.var("a"), UPat.cvar("c"))), UPat.var("b")], name="x"), lambda ctx,x,a,b,c: x.replace(op=Ops.SUB, src=(b, a)) if c.arg == -1.0 else None),
  # realize bool const as sgpr mask
  (UPat.cvar("x", dtypes.bool), lambda x: x.ins(RDNA3Ops.s_mov_b32, src=(const(dtypes.uint32, (1 << 32) - 1 if x.arg else 0),), tag=GP_SGPRS)),
  (UPat((Ops.CAST, Ops.BITCAST), dtypes.uchar, src=(UPat.var("y", dtype=dtypes.int8),)), lambda y: (y & const(dtypes.uint8, (1 << 8) - 1)).replace(dtype=dtypes.uint8)),
  (UPat((Ops.CAST, Ops.BITCAST), dtypes.ushort, src=(UPat.var("y", dtype=dtypes.int16),)), lambda y: (y & const(dtypes.uint16, (1 << 16) - 1)).replace(dtype=dtypes.uint16)),
  # NOTE: does this not work for int8 alu? Do we need to sign extend or something..
  (UPat(GroupOp.ALU, dtypes.int8s, name="x"), lambda x: x.replace(dtype=_smux(x.dtype, dtypes.int16, dtypes.uint16))),
  (UPat(GroupOp.Comparison, src=(UPat.var("y", dtype=dtypes.int8s), UPat()), name="x"), lambda x,y: x.replace(src=(y.bitcast(_smux(y.dtype, dtypes.int16, dtypes.uint16)), x.src[1]))),
  (UPat(Ops.STACK, name="x"), stack2regs),
  (UPat(Ops.CDIV, name="x"), idiv),
  # TODO: handle gated bool load/store
  # NOTE: booleans get passed around as sgpr masks in between loads and stores, but are converted / realized at mem ops to u8
  (UPat(Ops.STORE, src=(UPat.var("buf"), UPat.var("val", dtype=dtypes.bool)), allow_any_len=True, name="x"), lambda buf,val,x: x.replace(src=(buf,val.cast(dtypes.uint8)))),
  (UPat(Ops.LOAD, dtypes.bool, allow_any_len=True, name="x"), lambda x: x.replace(dtype=dtypes.uint32) != const(dtypes.uint32, 0)),
  (UPat(Ops.BUFFER, dtypes.bool, name="x"), lambda x: x.replace(dtype=dtypes.uint8) if x.addrspace is AddrSpace.REG else None),
  # TODO: use bfe/bi to unpack/pack once we have batched loads/stores
  # NOTE: int8s also have to be converted at memory boundary, native alu is in b16
  (UPat.var("y", dtypes.bool).cast(name="x"), lambda y,x: y.where(const(x.dtype, 1), const(x.dtype, 0))),
  # TODO: what cases does this fail?
  (UPat().cast().named("x").bitcast(), lambda x: x),
  # --- casting rewrites ---
  # float -> int
  (UPat.var("y", dtypes.half).cast((dtypes.double,)+dtypes.int32s+dtypes.int64s, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", dtypes.half).cast(dtypes.int8s, name="x"), lambda y,x: y.cast(_smux(x.dtype, dtypes.int16, dtypes.uint16)).bitcast(x.dtype)),
  (UPat.var("y", dtypes.float32).cast(dtypes.int16s+dtypes.int8s, name="x"), lambda y,x: y.cast(_smux(x.dtype, dtypes.int32, dtypes.uint32))),
  (UPat.var("y", dtypes.float32).cast(dtypes.int64s, name="x"), lambda y,x: y.cast(_smux(x.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  (UPat.var("y", dtypes.double).cast((dtypes.half,)+dtypes.int16s+dtypes.int8s, name="x"), lambda y,x: y.float().cast(dtypes.half).cast(x.dtype)),
  (UPat.var("y", dtypes.int16s+dtypes.int32s+dtypes.int8s).cast(dtypes.int16s+dtypes.int32s+dtypes.int8s, name="x"), intcast),
  # int -> float
  (UPat.var("y", dtypes.int32s).cast(dtypes.half), lambda y: y.float().cast(dtypes.half)),
  (UPat.var("y", dtypes.int8s).cast(dtypes.half), lambda y: y.cast(_smux(y.dtype, dtypes.int16, dtypes.uint16)).cast(dtypes.half)),
  (UPat.var("y", dtypes.int8s+dtypes.int16s).cast((dtypes.float,dtypes.double), name="x"), lambda y,x: y.cast(_smux(y.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  # other
  (UPat.var("y", dtypes.int64s).cast(dtypes.int64s), lambda y: y),
  (UPat.var("x", dtype=(dtypes.ulong, dtypes.long)).cast(dtypes.float64), long2double),
  (UPat.var("y", dtype=dtypes.int32s+dtypes.int16s+dtypes.int8s+dtypes.floats).cast((dtypes.ulong, dtypes.long), name="x"), lambda ctx,x,y: castint64(ctx, y, x.dtype)),
  # narrowing long goes through b32
  (UPat.var("y", dtypes.int64s).cast((dtypes.float, dtypes.half), name="x"), lambda y,x: long2double(y).cast(dtypes.float).cast(x.dtype)),
  (UPat.var("y", dtypes.int64s).cast(dtypes.int16s+dtypes.int8s+dtypes.int32s, name="x"),
    lambda y,x: y.index(0).replace(dtype=_smux(y.dtype, dtypes.int32, dtypes.uint32)).cast(x.dtype)),
  # NOTE: this only works because we assume upper half is right, widen cast is noop
  (UPat(Ops.MUL, dtypes.int16, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a.cast(dtypes.int32) * b.cast(dtypes.int32)),
  (UPat(Ops.CONST, (dtypes.float64, dtypes.long, dtypes.ulong), name="x"), const64),
  # expand 64 bit where, 2 cndmasks
  (UPat(Ops.WHERE, src=(UPat.var("pred"), UPat.var("a", dtype=(dtypes.ulong,dtypes.long,dtypes.float64)), UPat.var("b"))), lambda pred,a,b: 
    multireg(pred.where(a.index(0),b.index(0)), pred.where(a.index(1), b.index(1)), dtype=a.dtype) if a.op is not Ops.INDEX else None),
  # --- perf/folding ---
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("y"),UPat.cvar("x"))), lambda x,y: y if x.arg == 0 else None),
])

# NOTE: maybe add the range exec mask to end src in pre-regalloc?
# TODO: u64/i64 -> f64?
isel_matcher = PatternMatcher([
  (UPat(name="x").bitcast(), lambda x: x),
  (UPat(Ops.WMMA, name="x"), render_wmma),
  # control flow
  (UPat(Ops.RANGE, src=(UPat.var("bnd"),), allow_any_len=True, name="x"), prep_range),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng")), name="x"), prep_end),
  # noop
  (UPat.var("a").cast(name="x"), lambda a,x: a if a.dtype == x.dtype else None),
  # rtag every const, masks tag type as non Register to ensure it doesn't get treated as one
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),
  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM), name="x"), abi),
  # TODO: add fma/mad fuse detection to alu()
  # fused multiply add, use FMAC in the future?
  ((UPat(Ops.MUL, dtype=dtypes.floats, name="a") + UPat.var("b")).named("x"), lambda ctx,a,b,x: _vop3(ctx, x.ins(V_FMA[a.dtype], src=a.src + (b,)))),
  (UPat(Ops.ADD, dtype=dtypes.uint32, src=(UPat(Ops.ADD, name="y"), UPat.var("b")), name="x"), lambda ctx,x,y,b: _vop3(ctx, x.ins(RDNA3Ops.v_add3_u32, src=y.src + (b,)))),
  # cast
  (UPat.var("y", dtypes.int).cast(dtypes.uint, name="x"), lambda y,x: y), # noop?
  (UPat.var("y").cast(name="x"), cvt),
  # note: *_e64 cmp and cndmask encoding allows for storage/usage of VCC as SGPR
  (UPat.var("pred").where(UPat.var("a"), UPat.var("b")).named("x"), where),
  # perf: cmp shouldn't always be materialized to sgpr, only for where
  (UPat(GroupOp.Comparison|{Ops.XOR, Ops.AND, Ops.OR}, dtypes.bool, name="x"), cmp),
  # mem ops
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(allow_any_len=True, name="x"), store),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(allow_any_len=True, name="x"), load),
  # 16 bit indexes get expanded into extract moves/shifts, this only works for const indexes (everything but load/store?)
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.cvar("idx")), name="x", dtype=(dtypes.half,dtypes.int16,dtypes.uint16)), gethalf),
  # unified alu experiment
  (UPat(GroupOp.Binary|GroupOp.Unary, name="x"), alu),
  # barrier
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: None if x.addrspace is not AddrSpace.REG else bufreg(ctx, x)),
  # allocate virtual registers
  (UPat((Ops.INS, Ops.GROUP, Ops.RANGE, Ops.BUFFER), name="x"), alloc_vregs),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.INS, name="x"), lambda x: (x,[]) if x.arg is RDNA3Ops.s_nop else None),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  (UPat(Ops.INS, name="x"), lower_gated), # NOTE: find cleaner way to do this?
])

def encode(ctx, x:UOp):
  import tinygrad.renderer.amd.dsl as dsl
  if x.arg in [RDNA3Ops.s_nop, RDNA3Ops.s_endpgm]: return x.replace(arg=x.arg())
  dmap = { "vcc" : dsl.VCC, "exec_lo" : dsl.EXEC_LO, "v" : dsl.v, "s" : dsl.s  }
  def _route(r:Register): return dmap[r.name] if r.name in dmap else dmap[r.name[0]]
  def _immorreg(x:UOp):
    return x.arg if x.op is Ops.CONST else _fuse(rdefs(x))
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc, oprs = x.arg, x.arg.func, x.arg.opc, x.src

  # NOTE: hacky fixes, find cleaner way to conform to isa
  kw = args = None
  if group is RDNA3Ops.SMEM: kw = dict(sdata=_fuse(rdefs(x)), sbase=_fuse(tuple(u.tag[0] for u in oprs[:-1])), soffset=dsl.NULL, offset=oprs[-1].arg)
  elif group is RDNA3Ops.SOPK: args = [dsl.NULL, oprs[0].arg]
  elif group is RDNA3Ops.GLOBAL:
    kw = dict(addr=_immorreg(oprs[0]),  offset=_immorreg(oprs[1]))
    if reg(x) is None: kw["data"]=_fuse(rdefs(oprs[2]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.DS:
    kw = dict(addr=_immorreg(oprs[0]), offset1=_immorreg(oprs[1]))
    if reg(x) is None: kw["data0"]=_fuse(rdefs(oprs[3]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.VOP3SD: kw = dict(sdst=_immorreg(vccop), vdst=_fuse(rdefs(x)), **{f"src{i}":_immorreg(u) for i,u in enumerate(oprs[:3])})
  elif group is RDNA3Ops.VOPC: args = [_immorreg(u) for u in oprs]
  elif group in [RDNA3Ops.VOP3, RDNA3Ops.VOP2, RDNA3Ops.VOP1, RDNA3Ops.SOP1, RDNA3Ops.SOP2, RDNA3Ops.VOP3_SDST, RDNA3Ops.VOP3P]: # alu
    if group in [RDNA3Ops.VOP1, RDNA3Ops.SOP1]: oprs = oprs[:1]
    if group in [RDNA3Ops.VOP2, RDNA3Ops.SOP2]: oprs = oprs[:2]
    if group in [RDNA3Ops.VOP3, RDNA3Ops.VOP3P]: oprs = oprs[:3]
    args = [_fuse(rdefs(x))] + [_immorreg(u) for u in oprs]
  elif group is RDNA3Ops.SOPP: args = (0,)
  elif group is RDNA3Ops.VOPD:
    y = x.src[0]
    kw = dict(opy=y.arg.args[0], vdstx=_fuse(rdefs(x)), vdsty=_fuse(rdefs(y)), srcx0=_immorreg(x.src[1]), srcy0=_immorreg(y.src[0]))
    dual_binary = { RDNA3Ops.v_dual_mul_f32, RDNA3Ops.v_dual_add_f32 }
    if x.arg in dual_binary: kw["vsrcx1"] = _immorreg(x.src[2])
    if y.arg in dual_binary: kw["vsrcy1"] = _immorreg(y.src[1])
  else: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={opc}")

  ret = enc(**kw) if kw is not None else enc(*args)
  return x.replace(arg=ret)

# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L250
from enum import Enum, auto
class CntType(Enum):
  DS_CNT = auto(); LOAD_CNT = auto(); STORE_CNT = auto()

def ctp(x:UOp) -> CntType|None:
  if x.arg.func in { RDNA3Ops.GLOBAL, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }: return CntType.STORE_CNT if x.dtype is dtypes.void else CntType.LOAD_CNT
  if x.arg.func in { RDNA3Ops.SMEM, RDNA3Ops.DS }: return CntType.DS_CNT
  return None

def insertwaitcnts(uops:list[UOp]) -> list[UOp]:
  nuops = []
  for u in uops:
    nuops.append(u)
    if (tp := ctp(u)) is not None:
      nuops.append(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt, src=(const(dtypes.int16, 0),)))
  return nuops

# basic ones to start
def _dual_ops():
  dual_op_srcs = { "mov_b32", "mul_f32", "add_f32", "fmac_f32" }
  return { getattr(RDNA3Ops, f"v_{opc}_e32") : getattr(RDNA3Ops, f"v_dual_{opc}") for opc in dual_op_srcs }
dual_ops = _dual_ops()

def dual_alu(uops:list[UOp]):
  nuops = []
  for x,y in zip(uops[::2], uops[1::2]):
    indp = all(r not in rdefs(x) for s in y.src for r in rdefs(s))
    # ensure they dont have different literals
    _consts = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0}
    _consts.update(range(64))
    lits = set([u.arg for u in x.src+y.src if u.op is Ops.CONST and u.arg not in _consts])
    if x.arg in dual_ops and y.arg in dual_ops and indp and (rdef(x).index + rdef(y).index) % 2 != 0 and len(lits) <= 1:
      dx, dy = x.replace(arg=dual_ops[x.arg]), y.replace(arg=dual_ops[y.arg])
      nuops.append(dx.replace(src=(dy,) + dx.src))
    else: nuops.extend([x,y])
  if len(uops) % 2 != 0: nuops.append(uops[-1])
  return nuops

@dataclass
class RDNA3LinearCtx:
  loop_label: dict[UOp, str] = field(default_factory=dict)

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  extra_matcher = extra_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.SHR, Ops.SHL)}
  post_regalloc_ctx = RDNA3LinearCtx()
  def __init__(self, target:Target):
    from tinygrad.codegen.opt import tc
    super().__init__(target)
    self.tensor_cores = tc.get_amd(target.arch)

  def is_two_address(self, x:UOp) -> bool: return False
  def asm_str(self, uops:list[UOp], function_name:str) -> str: return ""

  # NOTE; FLAT_SCRATCH base implicit, since this is only used for spill/fill just fold ioffs
  def fill(self, spill_offset:int, x:UOp) -> UOp:
    bufsz = sum([r.size for r in rdefs(x)])
    _insmap = {4:RDNA3Ops.scratch_load_b32,8:RDNA3Ops.scratch_load_b64,16:RDNA3Ops.scratch_load_b128}
    return UOp(Ops.INS, arg=_insmap[bufsz], src=(const(dtypes.uint32, spill_offset),), tag=rdefs(x))

  def spill(self, spill_offset:int, x:UOp) -> UOp:
    bufsz = sum([r.size for r in rdefs(x)])
    _insmap = {4:RDNA3Ops.scratch_store_b32,8:RDNA3Ops.scratch_store_b64,16:RDNA3Ops.scratch_store_b128}
    return UOp(Ops.INS, arg=_insmap[bufsz], src=(const(dtypes.uint32, spill_offset),x))

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # uops = lin.src
    uops = dual_alu(lin.src)
    nuops = insertwaitcnts(uops)

    # labels + encode
    pc = 0
    targets: dict[str, int] = {}
    _asm: list[tuple[UOp,int]] = []
    for u in nuops:
      if u.arg is RDNA3Ops.s_nop:
        if isinstance(u.tag, str):
          targets[u.tag] = pc
        continue
      l = encode(self,u)
      pc += l.arg.size()
      _asm.append((l,pc))

    # if (cond) PC = PC + (SIMM16 *4) +4
    def _reslv(u:UOp,upc:int):
      if not isinstance(u.tag, str): return u
      simm = (targets[u.tag] - upc) // 4
      return u.replace(arg=RDNA3Ops.SOPP(u.arg.op, simm))
    lin = lin.replace(src=tuple([_reslv(u,p) for u,p in _asm]))

    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch, scratch_size=256)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s}

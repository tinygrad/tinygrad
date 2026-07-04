from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate, DType
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, VRegister, VSubRegister, rdefs
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
from dataclasses import dataclass, field
import itertools


# ---- (UOp, dtype) -> Instruction tables ----
dt_to_isa = { dtypes.int32:"i32", dtypes.uint32:"u32", dtypes.float32:"f32", dtypes.float64:"f64", dtypes.float16:"f16", dtypes.int16:"i16", dtypes.uint16:"u16", dtypes.uint64:"u64", }
isa_to_dt = { v:k for k,v in dt_to_isa.items() }

# (uop, prefix, opcodes, support 32 and 64 bit encoding (e32/e64 branches with keys))
insdefs = [
  (Ops.ADD, "v_add", ["f16_e32", "f32_e32", "f64", "nc_i32", "nc_u32_e32", "nc_u16", "nc_i16"], False),
  (Ops.SUB, "v_sub", ["f16_e32", "f32_e32", "nc_i32", "nc_i16", "nc_u16", "nc_u32_e32"], False),
  (Ops.MUL, "v_mul", ["f16_e32", "f32_e32", "f64", "i32_i24_e32", "lo_u32", "lo_u16"], False), # TODO: mul i16?
  (Ops.SQRT, "v_sqrt", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.LOG2, "v_log", ["f16_e32", "f32_e32"], False),
  (Ops.EXP2, "v_exp", ["f16_e32", "f32_e32"], False),
  (Ops.RECIPROCAL, "v_rcp", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.MAX, "v_max", ["f16_e32", "f32_e32", "u16", "i16", "u32_e32", "i32_e32"], False),
  (Ops.TRUNC, "v_trunc", ["f16_e32", "f32_e32", "f64_e32"], False),
  (Ops.CMPLT, "v_cmp_lt", ["f16", "f32", "f64", "u32", "i32"], True),
  (Ops.CMPNE, "v_cmp", ["neq_f16", "neq_f32", "neq_f64", "ne_u32", "ne_i32", "ne_i16", "ne_u16"], True),
  (Ops.CMPEQ, "v_cmp_eq", ["f16", "f32", "f64", "u16", "u32", "i16", "i32"], True)
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

# ---- everything else ----

def reg(u:UOp): return rs[0] if len((rs := rdefs(u))) >= 1 else None
def def_reg(dt, reg:Register|tuple[Register,...]): return UOp.placeholder((1,), dt, next(lane_ctr), AddrSpace.REG).replace(tag=(reg,) if isinstance(reg,Register) else reg)
def const(dt, v:int) -> UOp: return UOp.const(dt,truncate[dt](v)).rtag()
def make_vgpr(ctx, width:int=1) -> Register: return ctx.vreg(GP_VGPRS, width=width)
def vmov(x:UOp) -> UOp: return x.ins(RDNA3Ops.v_mov_b16_e32 if x.dtype.itemsize == 2 else RDNA3Ops.v_mov_b32_e32, src=(x,))
def multireg(*args, dtype:DType): return UOp.group(*args).replace(dtype=dtype)

VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],) 
GP_SGPRS, GP_VGPRS = tuple(SGPRS[5:]), tuple(VGPRS[1:])
VCC, EXEC = Register("vcc", 0), Register("exec_lo", 0)
lane_ctr = itertools.count()

kernarg_ptr = (def_reg(dtypes.uint32, KERNARG_PTR[0]), def_reg(dtypes.uint32, KERNARG_PTR[1]))
execop = def_reg(dtypes.uint32, EXEC)
lidop = def_reg(dtypes.uint32, WIIDS[0])
vccop = def_reg(dtypes.uint32, VCC)

# ---- register movement helpers ----

def packb16(ctx, lo:UOp, hi:UOp):
  if dtypes.is_float(lo.dtype): return UOp(Ops.INS, arg=RDNA3Ops.v_pack_b32_f16, src=(lo,hi))
  lo = lo & const(dtypes.uint32, 0xFFFF) # mask off upper half
  return _vop3(ctx, UOp(Ops.INS, arg=RDNA3Ops.v_lshl_or_b32, src=(hi, const(dtypes.int, 16), lo)))

def stack2regs(ctx, x:UOp, vreg:VRegister|None=None):
  nregs, mvs = ((len(x.src) * x.dtype.itemsize) + 3) // 4, []
  for i in range(nregs):
    if x.dtype.itemsize == 2: mvs.append(packb16(ctx, x.src[i*2], x.src[i*2+1]))
    else: mvs.append(vmov(x.src[i]))
  nx = multireg(*mvs, dtype=x.src[0].dtype)
  if vreg is not None: nx = nx.replace(src=tuple(s.replace(tag=(vreg.sub(i),)) for i,s in enumerate(x.src)), tag=(vreg,))
  return nx

def to_vgpr(ctx, x:UOp) -> UOp: return vmov(x) if x.op is Ops.CONST else x
def const_vgpr(ctx, dt, v:int) -> UOp: return to_vgpr(ctx, const(dt, v))

# ---- operand legalization wrappers ----

def _vop3(ctx, x:UOp):
  lits = [i for i,s in enumerate(x.src) if s.op is Ops.CONST]
  if len(lits) <= 1: return x
  new = list(x.src)
  for i in lits[1:]: new[i]=to_vgpr(ctx, new[i])
  return x.replace(src=tuple(new))

# TODO: pass in original op to use GroupOp.COMMUTATIVE?
def _vop2(ctx, x:UOp):
  # def _isvgpr(u:UOp): return (r := reg(u)) is not None and isinstance(r, Register) and r.cons[0].name[0] == "v"
  def _isconst(u:UOp): return u.op is Ops.CONST
  if not _isconst(x.src[1]): return x
  rest = x.src[2:] if len(x.src) > 2 else ()
  non_commutative = x.arg in (RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshrrev_b32_e32) # NOTE: add more
  if not non_commutative and not _isconst(x.src[0]): 
    return x.replace(src=(x.src[1], x.src[0]) + rest)
  return x.replace(src=(x.src[0], to_vgpr(ctx, x.src[1])) + rest)

# TODO: allocate vgpr / sgpr based on op group (x.arg.func)
# - should almost never need to manually call ctx.vreg, control flow allocations should also be handled here?
def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  if x.op is Ops.GROUP and x.src[0].op is not Ops.INS: return None
  if x.op is Ops.BUFFER and x.addrspace is not AddrSpace.REG: return None
  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], VRegister|VSubRegister): return None
  if isinstance(x.tag, tuple): assert x.tag, f"got empty tuple for op: {x.op}, {x.arg}"

  if x.op is Ops.GROUP:
    sgpr = x.src[0].arg.func.__name__[0] == 'S'
    vreg = ctx.vreg(GP_SGPRS if sgpr else GP_VGPRS, width=len(x.src))
    return x.replace(tag=(vreg,), src=tuple(s.replace(tag=(vreg.sub(i),)) for i,s in enumerate(x.src)))

  defs = []
  if isinstance(x.tag, tuple):
    cons, width = x.tag, 1
    if isinstance(x.tag[0], tuple):
      cons, width = x.tag
    defs = [ctx.vreg(x.tag, width=width)]
  else:
    if x.op is Ops.BUFFER: n = (x.dtype.itemsize // 4) * x.src[0].arg
    else: n = max(x.dtype.itemsize // 4, 1)
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
  return x.ins(RDNA3Ops.s_load_b64, src=kernarg_ptr + (const(dtypes.uint32, offs),), tag=(ctx.vreg(GP_SGPRS, width=2),))


# ----- memory access ----

# GLOBAL_ADDR = SGPR_u64 + VGPR_OFFS_U32 + IMMOFFS_u16
def fold_global(ctx, base:UOp, idx:UOp): # (saddr, voff, ioffs)
  # TODO: handle offseting cleanly, ensure 13 bit imoff doesnt overflow
  disp_scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = to_vgpr(ctx, const(dtypes.int, disp_scale.bit_length() - 1))
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.int, idx.arg * disp_scale),)), base, const(dtypes.int16, 0))
  return (idx << shft, base, const(dtypes.int16, 0))

# LDS_ADDR = VGPR_ADDR_u32 + imm_byte_offset_u16
# NOTE: keep base in src to maintain graph dependencies?
def fold_lds(ctx, base:UOp, idx:UOp): # (vaddr, ioffs) 
  scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32,0),)), const(dtypes.uint16, idx.arg * scale), base)
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0].cast(dtypes.uint32), const(dtypes.uint16, idx.src[1].arg * scale), base)
  shft = to_vgpr(ctx, const(dtypes.uint32, scale.bit_length() - 1))
  return (idx.cast(dtypes.uint32) << shft, const(dtypes.uint16, 0), base)

# TODO: handle 16 bit loads?

def fold_address(ctx, x:UOp): return fold_lds(ctx, *x.src[:2]) if x.addrspace is AddrSpace.LOCAL else fold_global(ctx, *x.src[:2])
def _insspace(gl,x): return gl[1] if x.addrspace is AddrSpace.LOCAL else gl[0]

def load(ctx, addr:UOp, x:UOp, gate:UOp|None = None, alt:UOp|None = None):
  alt, gate = x.src[1:] if len(x.src) > 1 else (None,None)
  base, idx = addr.src[:2]
  if base.addrspace is AddrSpace.REG: # handle loading/storing multiple registers
    assert idx.op is Ops.CONST and base.dtype.itemsize == 4 and gate is None
    # TODO: gated reg load
    return x.ins(RDNA3Ops.v_mov_b32_e32, dtype=base.dtype, src=(base.index(idx),))
  imap = {
    2 : (RDNA3Ops.global_load_u16,RDNA3Ops.ds_load_u16),
    4 : (RDNA3Ops.global_load_b32,RDNA3Ops.ds_load_b32),
    8 : (RDNA3Ops.global_load_b64,RDNA3Ops.ds_load_b64),
    16 : (RDNA3Ops.global_load_b128,RDNA3Ops.ds_load_b128),
  }
  n = addr.src[-1].arg if addr.op is Ops.SHRINK else 1
  nregs = (n * x.dtype.itemsize+3)//4
  vreg = make_vgpr(ctx, width=nregs)
  # the alt realize has to load them into the same vreg/subregs?
  def _gate(o:UOp): return o.replace(src=(stack2regs(ctx, alt, vreg),) + o.src  + (gate,def_reg(dtypes.uint32,GP_SGPRS))) if gate is not None else o
  return _gate(x.ins(_insspace(imap[n * x.dtype.itemsize],base), src=fold_address(ctx, addr), tag=(vreg,)))

def store(ctx, addr:UOp, x:UOp):
  val = x.src[1]
  gate = x.src[2] if len(x.src) > 2 else None
  base, idx = addr.src[:2]
  # NOTE: reg buf stores have to happen pre-regalloc after buffer is already given a vreg
  if base.addrspace is AddrSpace.REG: return None

  def _gate(o:UOp): return o.replace(src=o.src + (gate,def_reg(dtypes.uint32,GP_SGPRS))) if gate is not None else o
  n = addr.src[-1].arg if addr.op is Ops.SHRINK else 1
  nregs = (n*addr.dtype.itemsize+3)//4
  imap = {
    2:(RDNA3Ops.global_store_b16,RDNA3Ops.ds_store_b16),
    4:(RDNA3Ops.global_store_b32,RDNA3Ops.ds_store_b32),
    8:(RDNA3Ops.global_store_b64,RDNA3Ops.ds_store_b64),
    16:(RDNA3Ops.global_store_b128,RDNA3Ops.ds_store_b128)
  }
  return _gate(UOp(Ops.INS, arg=_insspace(imap[n * addr.dtype.itemsize],base), dtype=dtypes.void, src=fold_address(ctx, addr) + (to_vgpr(ctx,val),)))

def regstore(addr:UOp, x:UOp):
  base, idx = addr.src
  nx = x.ins(RDNA3Ops.v_mov_b32_e32, src=(x.src[1],), tag=(rdefs(base)[0].sub(idx.arg),))
  return nx, [nx]

# ------ ALU ------

def cvt(ctx, y:UOp, x:UOp): # TODO: b64 -> b64
  def _needcast(x:DType, y:DType): return not (dt_to_isa[x][0] == dt_to_isa[y][0])
  if x.dtype in (dtypes.uint64, dtypes.int64) and y.dtype.itemsize == 4: # b32 -> b64
    targ = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
    lo = y.ins(_cvt_ins(y.dtype, targ)) if _needcast(y.dtype, targ) else y
    return to_vgpr(ctx, UOp(Ops.STACK, src=(y, const(targ, 0))))
  elif y.dtype.itemsize == 8 and x.dtype.itemsize == 4 and y.dtype is not dtypes.float64: # b64 -> b32
    src = dtypes.uint32 if dtypes.is_unsigned(y.dtype) else dtypes.int32
    if _needcast(src, x.dtype): return x.ins(_cvt_ins(src, x.dtype), src=(y.gep(0),))
    else: return y.gep(0)
  return x.ins(_cvt_ins(y.dtype,x.dtype))

# NOTE: maybe add this functionality to to_vgpr? (bool const + cmp output handling, realize as vgpr per lane result)
# NOTE: also maybe add this to pre_isel?
def cmp(ctx, x:UOp):
  rlz = []
  if x.op in GroupOp.Comparison:
    for u in x.src: # convert cmp outputs and const bools to vgpr, NOTE: just add this to isel matching somehow?
      if u.dtype is dtypes.bool:
        if u.op is Ops.CONST: rlz.append(const(dtypes.uint32, 1) if u.arg else const(dtypes.uint32,0))
        else: rlz.append(u.where(const(dtypes.uint32,1), const(dtypes.uint32,0)))
      else: rlz.append(u)
    x = x.replace(src=tuple(rlz))
    dt = x.src[0].dtype.scalar()
    # NOTE: maybe easier to just always write to new sgpr, can be optimized to use e32 later but there will be lots of VCC spills in regalloc
    ins = OP_INS[x.op][64][dt]
  else:
    if x.op is Ops.AND: ins = RDNA3Ops.s_and_b32

  x = x.ins(ins, tag=GP_SGPRS)
  if x.op is Ops.AND: return x
  return _vop3(ctx, x)

def add64(ctx, x:UOp):
  if dtypes.is_float(x.dtype): return x.ins(RDNA3Ops.v_add_f64)
  a, b = x.src
  narrow = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  # after causes a problem for auto allocating group reg
  vreg = make_vgpr(ctx, width=2)
  lo = UOp(Ops.INS, dtype=dtypes.uint32, arg=RDNA3Ops.v_add_co_u32, src=(a.gep(0), b.gep(0)), tag=(vreg.sub(0),))
  hi = UOp(Ops.INS, dtype=narrow, arg=RDNA3Ops.v_add_co_ci_u32, src=(a.gep(1), b.gep(1), lo), tag=(vreg.sub(1),)).after(lo)
  return UOp.group(lo, hi).replace(dtype=x.dtype, tag=(vreg,))

# TODO: signed
def mul64(ctx, x:UOp):
  if dtypes.is_float(x.dtype): return x.ins(RDNA3Ops.v_mul_f64)
  a, b = x.src
  p1 = UOp(Ops.INS, arg=RDNA3Ops.v_mad_u64_u32, dtype=dtypes.uint64, src=(a.gep(0), b.gep(0), const(dtypes.uint64,0)))
  p2 = UOp(Ops.INS, arg=RDNA3Ops.v_mad_u64_u32, dtype=dtypes.uint64, src=(a.gep(1), b.gep(0), p1))
  return UOp(Ops.INS, arg=RDNA3Ops.v_mad_u64_u32, dtype=dtypes.uint64, src=(a.gep(0), b.gep(1), p2))

def bitwise64(ctx, x:UOp, ins):
  a, b = x.src
  lo = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.gep(0), b.gep(0)))
  hi = UOp(Ops.INS, dtypes.uint32, arg=ins, src=(a.gep(1), b.gep(1)))
  return UOp.group(lo,hi).replace(dtype=x.dtype)

# cdiv of int types needs to be converted to float then cast back after ceil op
# NOTE: we need to cast to float version of this bitwidth..., start with just 32
# https://github.com/llvm-mirror/llvm/blob/master/lib/Transforms/Utils/IntegerDivision.cpp
def cdiv(x:UOp):
  if x.dtype.itemsize != 4:
    raise NotImplementedError(f"cdiv expansion not implemented for dtype: {x.dtype}")
  c = x.src[0].float() / x.src[1].float()
  return c.ins(RDNA3Ops.v_ceil_f32).cast(x.dtype)

# NOTE: booleans should be natively represented as vcc/scc
# TODO: handle 16/64 bit semantics
def alu(ctx, x:UOp):
  dpreciz = x.dtype.itemsize == 8
  if dpreciz and x.op is Ops.ADD: return add64(ctx, x)
  if dpreciz and x.op is Ops.MUL: return mul64(ctx, x)

  ins = None
  def _bitwise(ins): return bitwise64(ctx, x, ins) if dpreciz else _vop2(ctx, x.ins(ins))
  if x.op is Ops.AND: return _bitwise(RDNA3Ops.v_and_b32_e32)
  elif x.op is Ops.OR: return _bitwise(RDNA3Ops.v_or_b32_e32)
  elif x.op is Ops.XOR: return _bitwise(RDNA3Ops.v_xor_b32_e32)

  def _bmux(sins, dins): return dins if dpreciz else sins
  if x.op is Ops.SHL: return _vop2(ctx, x.replace(src=x.src[::-1]).ins(_bmux(RDNA3Ops.v_lshlrev_b32_e32, RDNA3Ops.v_lshlrev_b64)))
  elif x.op is Ops.SHR: return _vop2(ctx, x.replace(src=x.src[::-1]).ins(_bmux(RDNA3Ops.v_lshrrev_b32_e32, RDNA3Ops.v_lshrrev_b64)))

  if ins is None:
    if x.op in OP_INS: ins = OP_INS[x.op][x.dtype]
    else: raise NotImplementedError(f"alu optype not implemented. op={x.op}, is_unary={len(x.src)==1}")
  return x.ins(ins) if len(x.src) == 1 else _vop2(ctx, x.ins(ins))

# ---- casting utilities -----

def widenshort(y:UOp, x:UOp):
  mid = dtypes.int32 if y.dtype is dtypes.int16 else dtypes.uint32
  y = y.cast(mid)
  if x.dtype is dtypes.float32: y = y.cast(dtypes.float32)
  if x.dtype is dtypes.float64: return y.cast(dtypes.float64)
  return y

# NOTE: this needs work, maybe cleaner to define 2 reg buffer and just .store()
def castint64(ctx, y:UOp, x:UOp):
  # if src (y) is an int just allocate reg buffer and store?
  hi_dt = dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32
  if y.dtype in (dtypes.int, dtypes.short, dtypes.uint, dtypes.ushort):
    lo = y.ins(RDNA3Ops.v_mov_b32_e32, src=(y,))
    hi = UOp(Ops.INS, hi_dt, arg=RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32,0),))
    return multireg(lo, hi, dtype=x.dtype)
  elif y.dtype is dtypes.float64:
    # https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp#L3691
    tr = y.trunc()
    hi_f = tr.ins(RDNA3Ops.v_ldexp_f64, src=(tr,const(dtypes.int16, -32))).trunc() # hi_f = trunc(x * 2^-32)
    lo_f = UOp(Ops.INS, dtypes.float64, arg=RDNA3Ops.v_fma_f64, src=(hi_f, const(dtypes.float64, -(2 ** 32)), tr)) # tr - hi_f * 2 ^ 32
    return multireg(lo_f.cast(dtypes.uint32), hi_f.cast(hi_dt), dtype=x.dtype)
  else:
    lo = y.float().cast(dtypes.uint32)
    hi = vmov(const(hi_dt, 0))
    return multireg(lo, hi, dtype=x.dtype)

  print(y.dtype, x.dtype)
  raise NotImplementedError()

def long2double(x:UOp):
  lo = x.gep(0).replace(dtype=dtypes.uint32).cast(dtypes.float64)
  hi = x.gep(1).replace(dtype=dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32).cast(dtypes.float64)
  hi = hi.ins(RDNA3Ops.v_ldexp_f64, src=(hi,const(dtypes.int16, 32)))
  return lo + hi

# casting between long/ulong and floats is more complicated, may belong in isel?
def const64(x:UOp):
  v = x.arg.bits if dtypes.is_float(x.dtype) else x.arg
  return UOp.group(vmov(const(dtypes.uint32,v)), vmov(const(dtypes.uint32, v >> 32))).replace(dtype=x.dtype)

# ---- control flow ----
def restoreexec(mask:UOp) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_or_b32, src=(execop,mask), tag=(EXEC,))
def label(ctx, name:str) -> UOp: return UOp(Ops.INS, arg=RDNA3Ops.s_nop, tag=name)
memgroups = { RDNA3Ops.GLOBAL, RDNA3Ops.SMEM, RDNA3Ops.DS, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }

def lower_gated(x:UOp):
  if x.arg.func not in memgroups: return None
  store = x.dtype is dtypes.void

  if store and len(x.src) > 4:
    branch = x.src[-1].ins(RDNA3Ops.s_and_saveexec_b32, src=(x.src[-2],))
    return branch, [branch, x.replace(src=x.src[:-2]), restoreexec(x.src[-1])]

  if not store and len(x.src) > 3:
    save = x.src[-1].ins(RDNA3Ops.s_and_saveexec_b32, src=(x.src[-2],))
    return x.src[0], [x.src[0], save, x.replace(src=x.src[1:-2]), restoreexec(x.src[-1])]

def prep_range(ctx, bnd:UOp, x:UOp):
  if x.dtype is dtypes.uint32: return None # this is a shit predicate, maybe utilize ctx
  mask = def_reg(dtypes.uint32, GP_SGPRS)
  return x.replace(src=(bnd,mask)).replace(dtype=dtypes.uint32)

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
  inc = x.src[1].ins(RDNA3Ops.v_add_nc_u32_e32, src=(x.src[1], x.src[-2]))
  loop = UOp(Ops.INS, arg=RDNA3Ops.s_cbranch_execnz, tag=f".LOOP_{ctx.loop_label[x.src[1]]}")
  return inc, [inc, gate, loop, restoreexec(x.src[-1])]

# --- other stuff ---
# TODO: get rid of these hacky dtype replaces, just done to avoid triggering recursive rewrite
# NOTE: this should just be triggered in to_vgpr????
def gethalf(x:UOp, buf:UOp, idx:UOp):
  i = idx.arg
  b32 = buf.index(UOp.const(dtypes.int, i // 2)).replace(dtype=dtypes.uint32)
  if i % 2 != 0: return (b32 >> 16).replace(dtype=x.dtype)
  else: return x.ins(RDNA3Ops.v_mov_b16_e32, src=(b32,))

def where(ctx, pred:UOp, a:UOp, b:UOp, x:UOp):
  ins = RDNA3Ops.v_cndmask_b32_e64 if x.dtype.itemsize ==  4 else RDNA3Ops.v_cndmask_b16
  return _vop3(ctx, x.ins(ins, src=(b,a,cmp(ctx,pred))))

# ---- lowering passes ----
# TODO: simplify these cast rules, maybe just make a legalize cast function?
pre_isel_matcher = PatternMatcher([
  # bitcast is noop? 
  (UPat.var("y").bitcast().named("x"), lambda y,x: y),
  # NOTE: casting comparison output to float should be treated as a where pred ? 0.0 : 1.0
  (UPat.var("y", dtype=dtypes.bool).cast(name="x"), lambda y,x: y.where(const(x.dtype, 1), const(x.dtype, 0))),
  # cast noops
  (UPat.var("y", dtype=(dtypes.uint32,dtypes.int32)).cast((dtypes.uint32,dtypes.int32)), lambda y: y), # same size int b32
  (UPat.var("y", dtype=(dtypes.uint32,dtypes.int32)).cast((dtypes.int16, dtypes.uint16), name="x"), lambda y,x: y.replace(dtype=x.dtype)), # narrow int
  (UPat.var("y", dtype=(dtypes.int16,dtypes.uint16)).cast((dtypes.uint16,dtypes.int16)), lambda y: y), # same size int b16
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None), # cast to ptr
  # cast rewrites
  # u64/i64 -> f32 = take lo then cast
  (UPat.var("y", dtype=(dtypes.long, dtypes.ulong)).cast((dtypes.float, dtypes.half, dtypes.short, dtypes.ushort, dtypes.int, dtypes.uint), name="x"),
    lambda y,x: y.gep(0).replace(dtype=dtypes.uint32 if dtypes.is_unsigned(y.dtype) else dtypes.int32).cast(x.dtype)),
  (UPat.var("y", dtype=(dtypes.int16,dtypes.uint16)).cast(name="x", dtype=(dtypes.uint32, dtypes.int32, dtypes.float32,dtypes.float64)), widenshort),
  # (f16 -> f64/i32/u32 ) to (f16 -> f32 -> f64/i32/u32)
  (UPat.var("y", dtype=dtypes.half).cast(name="x", dtype=(dtypes.double, dtypes.int32, dtypes.uint32)), lambda y,x: y.float().cast(x.dtype)),
  (UPat.var("y", dtype=(dtypes.half, dtypes.float)).cast((dtypes.ulong,dtypes.long), name="x"),
    lambda y,x: y.cast(dtypes.uint32 if dtypes.is_unsigned(x.dtype) else dtypes.int32).cast(x.dtype)),
  # (u32/i32 -> f16) to (-> f32 -> f16)
  (UPat.var("y", dtype=(dtypes.uint32, dtypes.int32)).cast(dtypes.half), lambda y: y.cast(dtypes.float32).cast(dtypes.half)), 
  # (f64 -> f16/i16/u16) to (f64 -> f64 ?-> i16/u16)
  (UPat.var("y", dtype=dtypes.double).cast((dtypes.half, dtypes.int16, dtypes.uint16), name="x"), lambda y,x: y.cast(dtypes.float32).cast(dtypes.half).cast(x.dtype)),
  # (f32 -> u16/i16) to (f32 -> u32/i32)
  (UPat.var("y", dtype=dtypes.float32).cast((dtypes.uint16,dtypes.int16), name="x"), lambda y,x: y.cast(dtypes.uint32 if x.dtype is dtypes.uint16 else dtypes.int32)),
  # this only works because we assume upper half is right, widen cast is noop
  (UPat(Ops.MUL, src=(UPat.var("a"), UPat.var("b")), dtype=dtypes.int16), lambda a,b: a.cast(dtypes.int32) * b.cast(dtypes.int32)),
  (UPat(Ops.CONST, (dtypes.float64, dtypes.long, dtypes.ulong), name="x"), const64),
])

# NOTE: maybe add the range exec mask to end src in pre-regalloc?
# TODO: u64/i64 -> f64?
isel_matcher = PatternMatcher([
  (UPat(Ops.STACK, name="x"), stack2regs),
  (UPat.var("x", dtype=(dtypes.ulong, dtypes.long)).cast(dtypes.float64), long2double),
  (UPat.var("y", dtype=dtypes.ints+dtypes.uints+dtypes.floats).cast((dtypes.ulong, dtypes.long), name="x"), castint64),
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
  # cast
  (UPat.var("y", dtypes.int).cast(dtypes.uint, name="x"), lambda y,x: y), # noop?
  (UPat.var("y").cast(name="x"), cvt),
  # note: *_e64 cmp and cndmask encoding allows for storage/usage of VCC as SGPR
  (UPat.var("pred").where(UPat.var("a"), UPat.var("b")).named("x"), where),
  # perf: cmp shouldn't always be materialized to sgpr, only for where
  (UPat(GroupOp.Comparison|{Ops.AND, Ops.OR}, dtypes.bool, name="x"), cmp),
  # mem ops
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(allow_any_len=True, name="x"), store),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(allow_any_len=True, name="x"), load),
  # 16 bit indexes get expanded into extract moves/shifts, this only works for const indexes (everything but load/store?)
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.cvar("idx")), name="x", dtype=(dtypes.half,dtypes.int16,dtypes.uint16)), gethalf), 
  # unified alu experiment
  (UPat(Ops.CDIV, name="x"), cdiv),
  (UPat(Ops.CMOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - b * a.alu(Ops.CDIV, b)), # hack from x86
  (UPat(GroupOp.Binary|GroupOp.Unary, name="x"), alu),
  # barrier
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  # allocate virtual registers
  (UPat((Ops.INS, Ops.GROUP, Ops.BUFFER, Ops.RANGE), name="x"), alloc_vregs),
])

pre_regalloc_matcher = PatternMatcher([
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(allow_any_len=True, name="x"), regstore),
])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.INS, name="x"), lambda x: (x,[]) if x.arg is RDNA3Ops.s_nop else None),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  (UPat(Ops.INS, name="x"), lower_gated),
])

def encode(ctx, x:UOp):
  if x.arg in [RDNA3Ops.s_nop, RDNA3Ops.s_endpgm]: return x.replace(arg=x.arg())
  import tinygrad.renderer.amd.dsl as dsl
  def _route(r:Register):
    dmap = { "vcc" : dsl.VCC, "exec_lo" : dsl.EXEC_LO, "v" : dsl.v, "s" : dsl.s  } 
    return dmap[r.name] if r.name in dmap else dmap[r.name[0]]
  def _immorreg(x:UOp): return x.arg if x.op == Ops.CONST else _fuse(rdefs(x))
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc, oprs = x.arg, x.arg.func, x.arg.opc, x.src

  # NOTE: hacky fixes, find cleaner way to conform to isa
  kw = args = None
  if group is RDNA3Ops.SMEM:
    kw = dict(sdata=_fuse(rdefs(x)), sbase=_fuse(tuple(u.tag[0] for u in oprs[:-1])), soffset=dsl.NULL, offset=oprs[-1].arg)
  elif group is RDNA3Ops.GLOBAL:
    kw = dict(addr=_immorreg(oprs[0]), saddr=_fuse(rdefs(oprs[1])), offset=_immorreg(oprs[2]))
    if reg(x) is None: kw["data"]=_fuse(rdefs(oprs[3]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.DS:
    kw = dict(addr=_immorreg(oprs[0]), offset1=_immorreg(oprs[1]))
    if reg(x) is None: kw["data0"]=_fuse(rdefs(oprs[3]))
    else: kw["vdst"]=_fuse(rdefs(x))
  elif group is RDNA3Ops.SOPK: args = [dsl.NULL, oprs[0].arg]
  elif group is RDNA3Ops.VOP3SD:
    oprs = oprs[:3]
    kw = dict(sdst=_immorreg(vccop), vdst=_fuse(rdefs(x)))
    for i,u in enumerate(oprs): kw[f"src{i}"]=_immorreg(u)
  elif group is RDNA3Ops.VOPC:
    args = [_immorreg(u) for u in oprs]
  elif group in [RDNA3Ops.VOP3, RDNA3Ops.VOP2, RDNA3Ops.VOP1, RDNA3Ops.SOP1, RDNA3Ops.SOP2, RDNA3Ops.VOP3_SDST]: # alu
    if group is RDNA3Ops.VOP2: oprs = oprs[:2]
    if group is RDNA3Ops.VOP3: oprs = oprs[:3]
    args = [_fuse(rdefs(x))] + [_immorreg(u) for u in oprs]
  elif group is RDNA3Ops.SOPP: args = (0,)
  else: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={opc}")


  ret = enc(**kw) if kw is not None else enc(*args)
  return x.replace(arg=ret)

# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L250
from enum import Enum, auto
class CntType(Enum):
  DS_CNT = RDNA3Ops.s_waitcnt_lgkmcnt
  LOAD_CNT = RDNA3Ops.s_waitcnt_vmcnt
  STORE_CNT = RDNA3Ops.s_waitcnt_vscnt

def ctp(x:UOp) -> CntType|None:
  if x.arg.func in { RDNA3Ops.GLOBAL, RDNA3Ops.FLAT, RDNA3Ops.SCRATCH }: return CntType.STORE_CNT if x.dtype is dtypes.void else CntType.LOAD_CNT
  if x.arg.func in { RDNA3Ops.SMEM }: return CntType.DS_CNT
  return None
  
def insertwaitcnts(uops:list[UOp]) -> list[UOp]:
  total: dict[CntType, int] = { CntType.DS_CNT: 0, CntType.LOAD_CNT: 0, CntType.STORE_CNT: 0 }
  outstnd: dict[CntType, int] = { CntType.DS_CNT: 0, CntType.LOAD_CNT: 0, CntType.STORE_CNT: 0 }
  ddeps: dict[Register, tuple[CntType, int]] = {}
  nuops = []

  for u in uops: 
    oprs = u.src
    if (tp := ctp(u)):
      if u.dtype is dtypes.void: oprs = oprs[:-1]
      rs = rdefs(u.src[-1]) if u.dtype is dtypes.void else rdefs(u) 
      for r in rs: ddeps[r] = (tp,outstnd[tp])
      outstnd[tp] += 1
      total[tp] += 1

    tosync: dict[CntType, int] = {}
    for tp,n in [ddeps[r] for s in oprs for r in rdefs(s) if r in ddeps]:
      tosync[tp] = min(tosync.setdefault(tp, float('inf')), outstnd[tp] - n - 1)
    for tp,cnt in tosync.items():
      nuops.append(UOp(Ops.INS, arg=tp.value, src=(const(dtypes.int16, cnt),)))
    nuops.append(u)
  return nuops

@dataclass
class RDNA3LinearCtx:
  loop_label: dict[UOp, str] = field(default_factory=dict)

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR)}
  post_regalloc_ctx = RDNA3LinearCtx()
  def __init__(self, target:Target):
    super().__init__(target)

  def is_two_address(self, x:UOp) -> bool: return False

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    nuops = insertwaitcnts(lin.src)

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

    print(prg.arg)
    for u in lin.src: print(u.arg)
     
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}

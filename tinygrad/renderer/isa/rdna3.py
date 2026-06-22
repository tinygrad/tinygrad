from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, ParamArg
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, regs, reg
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
from dataclasses import dataclass, field

VCC, EXEC_LO = Register("vcc", 0), Register("exec_lo", 0) # hack: special regs
VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],) # reserved for abi
GP_SGPRS, GP_VGPRS = tuple(SGPRS[5:]), tuple(VGPRS[1:])

def def_reg(dt, reg:Register): return UOp(Ops.INS, arg=RDNA3Ops.s_nop, dtype=dt, tag=(reg,))
vccop, execop = def_reg(dtypes.uint32, VCC), def_reg(dtypes.uint32, EXEC_LO)

def const(dt, v:int) -> UOp: return UOp.const(dt,truncate[dt](v)).rtag()
# TODO: impl with trunc and casting?
def imm16(dt, v): return const(dt, v)
def is_vgpr(x:UOp) -> bool: return x.tag is not None and x.tag != True and x.tag != GP_SGPRS and x.tag[0].cons[0].name[0] == "v"
def to_vgpr(x:UOp) -> UOp: return x.ins(RDNA3Ops.v_mov_b32_e32, src=(x,)) if x.op is Ops.CONST else x

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # real registers
  if x.op is Ops.BUFFER and x.tag is not None: return None
  # no register definition
  if x.dtype is dtypes.void: return None
  # already allocated vregs
  if isinstance(x.tag, tuple) and isinstance(x.tag[0], Register) and x.tag[0]._cons: return None
  # allocate vreg definitions
  defs = []
  # don't generally allocate to SGPRS, only works wave uniform possible future optim
  if isinstance(x.tag, tuple):
    vr = ctx.vreg(x.tag)
    defs = [vr] if isinstance(vr, Register) else [*vr]
  else: defs = [ctx.vreg(GP_VGPRS)]
  return x.replace(tag=tuple(defs))

# https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  i = ctx.func_args.index(x)
  if x.op is Ops.SPECIAL:
    dim = int(x.arg[-1])
    if x.arg[0] == 'g': return def_reg(dtypes.int, WGIDS[dim])
    else: return x.ins(RDNA3Ops.v_bfe_u32, src=(def_reg(dtypes.uint32, WIIDS[0]), const(dtypes.uint32, 10 * dim), const(dtypes.uint32, 10)))
  offs = sum(8 if u.op == Ops.PARAM else 4 for u in ctx.func_args[:i])
  kernarg_ptr = (def_reg(dtypes.uint32, KERNARG_PTR[0]), def_reg(dtypes.uint32, KERNARG_PTR[1]))
  return x.ins(RDNA3Ops.s_load_b64, src=kernarg_ptr + (UOp.const(dtypes.uint32, offs).rtag(),), tag=ctx.vreg((GP_SGPRS, 2)))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

V_ADD =   { dtypes.float16:RDNA3Ops.v_add_f16_e32,  dtypes.float32:RDNA3Ops.v_add_f32_e32,  dtypes.float64:RDNA3Ops.v_add_f64,   dtypes.int32:RDNA3Ops.v_add_nc_i32,       dtypes.uint32:RDNA3Ops.v_add_nc_u32_e32,  }
V_SUB =   { dtypes.float16:RDNA3Ops.v_sub_f16_e32,  dtypes.float32:RDNA3Ops.v_sub_f32_e32,  dtypes.int32:RDNA3Ops.v_sub_nc_i32,  dtypes.uint32:RDNA3Ops.v_sub_nc_u32_e32,  }
V_MUL =   { dtypes.float16:RDNA3Ops.v_mul_f16_e32,  dtypes.float32:RDNA3Ops.v_mul_f32_e32,  dtypes.float64:RDNA3Ops.v_mul_f64,   dtypes.int32:RDNA3Ops.v_mul_i32_i24_e32,  dtypes.uint32:RDNA3Ops.v_mul_u32_u24_e32, }
V_SQRT =  { dtypes.float16:RDNA3Ops.v_sqrt_f16_e32, dtypes.float32:RDNA3Ops.v_sqrt_f32_e32, dtypes.float64:RDNA3Ops.v_sqrt_f64_e32  }
V_LOG =   { dtypes.float16:RDNA3Ops.v_log_f16_e32,  dtypes.float32:RDNA3Ops.v_log_f32_e32 }
V_EXP =   { dtypes.float16:RDNA3Ops.v_exp_f16_e32,  dtypes.float32:RDNA3Ops.v_exp_f32_e32 }
V_MAX =   { dtypes.float16:RDNA3Ops.v_max_f16_e32,  dtypes.float32:RDNA3Ops.v_max_f32_e32,  dtypes.uint16:RDNA3Ops.v_max_u16,     dtypes.int16:RDNA3Ops.v_max_i16,          dtypes.uint32:RDNA3Ops.v_max_u32_e32,
              dtypes.int32:RDNA3Ops.v_max_i32_e32}
V_RCP =   { dtypes.float16:RDNA3Ops.v_rcp_f16_e32,  dtypes.float32:RDNA3Ops.v_rcp_f32_e32,  dtypes.float64:RDNA3Ops.v_rcp_f64_e32   }
V_SIN =   { dtypes.float16:RDNA3Ops.v_sin_f16_e32,  dtypes.float32:RDNA3Ops.v_sin_f32_e32 }
V_TRUNC = { dtypes.float16:RDNA3Ops.v_trunc_f16_e32,dtypes.float32:RDNA3Ops.v_trunc_f32_e32,dtypes.float64:RDNA3Ops.v_trunc_f64_e32 }
V_FMA =   { dtypes.float16:RDNA3Ops.v_fma_f16,      dtypes.float32:RDNA3Ops.v_fma_f32,      dtypes.float64:RDNA3Ops.v_fma_f64       }
V_CMPLT = { dtypes.float16:RDNA3Ops.v_cmp_lt_f16_e64, dtypes.float32:RDNA3Ops.v_cmp_lt_f32_e64, dtypes.float64:RDNA3Ops.v_cmp_lt_f64_e64, dtypes.uint32:RDNA3Ops.v_cmp_lt_u32_e64,
  dtypes.int32:RDNA3Ops.v_cmp_lt_i32_e64, dtypes.int16:RDNA3Ops.v_cmp_lt_i16_e64, dtypes.uint16:RDNA3Ops.v_cmp_lt_u16_e64 }
V_CMPGT = { dtypes.float16:RDNA3Ops.v_cmp_gt_f16_e64, dtypes.float32:RDNA3Ops.v_cmp_gt_f32_e64, dtypes.float64:RDNA3Ops.v_cmp_gt_f64_e64, dtypes.uint32:RDNA3Ops.v_cmp_gt_u32_e64,
  dtypes.int32:RDNA3Ops.v_cmp_gt_i32_e64, dtypes.int16:RDNA3Ops.v_cmp_gt_i16_e64, dtypes.uint16:RDNA3Ops.v_cmp_gt_u16_e64 }
V_CMPEQ = { dtypes.float16:RDNA3Ops.v_cmp_eq_f16_e32, dtypes.float32:RDNA3Ops.v_cmp_eq_f32_e32, dtypes.float64:RDNA3Ops.v_cmp_eq_f64_e32, dtypes.uint16:RDNA3Ops.v_cmp_eq_u16_e32, dtypes.uint32:RDNA3Ops.v_cmp_eq_u32_e32, dtypes.int16:RDNA3Ops.v_cmp_eq_i16_e32, dtypes.int32:RDNA3Ops.v_cmp_eq_i32_e32 }
V_CMPNE = { dtypes.float16:RDNA3Ops.v_cmp_neq_f16_e64,dtypes.float32:RDNA3Ops.v_cmp_neq_f32_e64,dtypes.float64:RDNA3Ops.v_cmp_neq_f64_e64, dtypes.uint32:RDNA3Ops.v_cmp_ne_u32_e64,
  dtypes.int32:RDNA3Ops.v_cmp_ne_i32_e64, dtypes.int16:RDNA3Ops.v_cmp_ne_i16_e64, dtypes.uint16:RDNA3Ops.v_cmp_ne_u16_e64 }

def legalize_operands(x:UOp):
  group, opc = x.arg.func, x.arg.opc
  if group in [RDNA3Ops.VOP2, RDNA3Ops.VOPC]:
    if any(s.tag is None for s in x.src[:2]): return None
    suffix = x.src[2:] if len(x.src) > 2 else ()
    a, b = x.src[:2]
    if is_vgpr(b): return None
    if is_vgpr(a): return x.replace(src=(b,a) + suffix)
    return x.replace(src=((a, to_vgpr(b))) + suffix)
  elif group in [RDNA3Ops.GLOBAL] and "store" in opc: 
    if is_vgpr(x.src[-1]): return None
    return x.replace(src=x.src[:-1] + (to_vgpr(x.src[-1]),))
  elif group is RDNA3Ops.VOP3:
    lits = [i for i,s in enumerate(x.src) if s.op is Ops.CONST]
    if len(lits) > 1:
      new = list(x.src)
      for i in lits[1:]: new[i]=to_vgpr(new[i])
      return x.replace(src=tuple(new))
  return None

def cmp(x:UOp):
  dt = x.src[0].dtype
  if x.op is Ops.CMPLT: ins = V_CMPLT[dt]
  elif x.op is Ops.CMPEQ: ins = V_CMPEQ[dt]
  elif x.op is Ops.CMPNE: ins = V_CMPNE[dt]
  else: ins = V_CMPGT[dt]
  # else: raise NotImplementedError("comparison type instruction dne")
  return x.ins(ins, tag=GP_SGPRS)

def stack(ctx:IselContext, x:UOp):
  return UOp.group(*[u.ins(RDNA3Ops.v_mov_b32_e32, tag=(vr,), src=(u,))
                     for vr, u in zip(ctx.vreg((GP_VGPRS,len(x.src))), x.src)])

# GLOBAL_ADDR = SGPR_u64 + VGPR_OFFS_U32 + IMMOFFS_u16
def _offs(v:int) -> UOp: return UOp.const(dtypes.int16, ((1 << 13) - 1) & v).rtag() # TODO: handle overflow
def fold_global(ctx, base:UOp, idx:UOp): # (saddr, voff, ioffs)
  disp_scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  shft = to_vgpr(const(dtypes.int, disp_scale // 2))
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32, 0),)), base, _offs(idx.arg * disp_scale))
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0] << shft, base, _offs(idx.src[1].arg * disp_scale))
  return (idx << shft, base, _offs(0))

def local_addr(ctx, x:UOp):
  if x.addrspace is not AddrSpace.LOCAL: return None
  ptr = ctx.lds_size
  ctx.lds_size += x.dtype.itemsize # wrong
  return x.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32, ptr),))

# LDS_ADDR = VGPR_ADDR_u32 + imm_byte_offset_u16
def fold_lds(ctx, base:UOp, idx:UOp): # (vaddr, ioffs)
  # scale = base.dtype.itemsize if base.op in {Ops.PARAM, Ops.BUFFER, Ops.AFTER} else 1
  scale = 1 
  print(base.op, idx.op)
  if idx.op is Ops.CONST: return (base, idx.arg * scale)
  shft = to_vgpr(const(dtypes.int, scale // 2))
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST:
    return (base + (idx.src[0] << shft), idx.src[1].arg * scale)
  raise NotImplementedError(f"cant fold lds, idx.op={idx.op}")

# def fold_address(ctx, x:UOp): return x.src[:2]
def fold_address(ctx, x:UOp): return fold_global(ctx, *x.src[:2]) if x.addrspace is AddrSpace.GLOBAL else fold_lds(ctx, *x.src[:2])

# todo: handle 16 bit loads?
def _insspace(gl,x): return gl[0] if x.addrspace is AddrSpace.GLOBAL else gl[1]
def load(ctx, addr:UOp, x:UOp):
  base, idx = addr.src[:2]
  if base.addrspace is AddrSpace.REG:
    return x.ins(RDNA3Ops.v_mov_b32_e32, dtype=addr.src[0].dtype, src=(addr.src[0],))
  imap = {
    1 : (RDNA3Ops.global_load_b32,RDNA3Ops.ds_load_b32),
    2 : (RDNA3Ops.global_load_b64,RDNA3Ops.ds_load_b64),
    4 : (RDNA3Ops.global_load_b128,RDNA3Ops.ds_load_b128),
  }
  n = addr.src[-1].arg if addr.op is Ops.SHRINK else 1
  nregs = (n * x.dtype.itemsize+3)//4
  return x.ins(_insspace(imap[nregs],base), src=fold_address(ctx, addr), tag=GP_VGPRS if nregs == 1 else (GP_VGPRS, nregs))

def store(ctx, addr:UOp, val:UOp, x:UOp):
  base, idx = addr.src[:2]
  n = len(val.src) if val.op is Ops.STACK else 1
  if base.addrspace is AddrSpace.REG:
    mvs = [UOp(Ops.INS, dtype=val.dtype.scalar(), arg=RDNA3Ops.v_mov_b32_e32, src=(val.gep(i),), tag=tg)
        for i, tg in zip(range(n), [GP_VGPRS] if n == 1 else ctx.vreg((GP_VGPRS,n)))]
    return UOp.group(*mvs) if len(mvs) > 1 else mvs[0]
  nregs = (n*val.dtype.itemsize+3)//4
  imap = {
    1:(RDNA3Ops.global_store_b32,RDNA3Ops.ds_store_b32),
    2:(RDNA3Ops.global_store_b64,RDNA3Ops.ds_store_b64),
    4:(RDNA3Ops.global_store_b128,RDNA3Ops.ds_store_b128)
  }
  return UOp(Ops.INS, arg=_insspace(imap[nregs],base), dtype=dtypes.void, src=fold_address(ctx, addr) + (val,))

def prepare_range(ctx, x:UOp, bnd:UOp):
  if x.src[-1].op is Ops.INS and x.src[-1].arg is RDNA3Ops.s_nop: return None # already processed
  # mask = UOp(Ops.DEFINE_REG, dtypes.uint32, tag=(ctx.vreg(GP_SGPRS),))
  mask = def_reg(dtypes.uint32, ctx.vreg(GP_SGPRS))
  return x.replace(src=x.src + (mask,), tag=(ctx.vreg(GP_VGPRS),))

def gated_load(ctx, addr:UOp, gate:UOp, alt:UOp, x:UOp):
  if alt.op is Ops.CONST: alt = UOp(Ops.INS, arg=RDNA3Ops.v_mov_b32_e32, src=(alt,))
  mask_exec = UOp(Ops.INS, arg=RDNA3Ops.s_and_saveexec_b32, src=(gate,), tag=GP_SGPRS)
  loaded = addr.load().after(mask_exec, alt)
  saved = mask_exec.after(loaded)
  # restore depends on saved mask exec which passes through after load op
  restore = UOp(Ops.INS, arg=RDNA3Ops.s_or_b32, src=(execop, saved), tag=(EXEC_LO,)) 
  return loaded.after(restore) # pass loaded through, but restore actually happens first

pre_isel_matcher = PatternMatcher([
  # cast to ptr is noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(UPat.var("alt"), UPat.var("gate"), name="x"), gated_load),
])

# TODO: control flow, gated load/store and RANGE/END
# TODO: 64 bit integer mul with MAD_u64/i64 and MUL_lo_u32, 32x32 + 64 -> 64
# TODO: handle unsupported dtypes in pre_isel_matcher by casting?
isel_matcher = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), local_addr),
  # noop
  (UPat.var("a").cast(name="x"), lambda a,x: a if a.dtype == x.dtype else None),
  # rtag every const, masks tag type as non Register to ensure it doesn't get treated as one
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),
  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM), name="x"), abi),
  # Range and end gets lowered after regalloc
  (UPat(Ops.RANGE, src=(UPat.cvar("bnd"),), allow_any_len=True, name="x"), prepare_range),
  (UPat(Ops.END, src=(UPat(), UPat.var("rng")), name="x"), # wire sgpr execmask into src to model reg dependency
    lambda x,rng: x.replace(src=x.src + (rng.src[-1],)) if len(x.src) == 2 else None), 
  # unary alu ops
  (UPat.var("y").log2().named("x"), lambda y,x: x.ins(V_LOG[y.dtype])),
  (UPat.var("y").exp2().named("x"), lambda y,x: x.ins(V_EXP[y.dtype])),
  (UPat.var("y").sqrt().named("x"), lambda y,x: x.ins(V_SQRT[y.dtype])),
  (UPat.var("y").sin().named("x"), lambda y,x: x.ins(V_SIN[y.dtype])),
  (UPat.var("y").trunc().named("x"), lambda y,x: x.ins(V_TRUNC[y.dtype])),
  (UPat(Ops.RECIPROCAL, name="x", src=(UPat.var("y"),)), lambda y,x: x.ins(V_RCP[y.dtype])),
  # fused multiply add, use FMAC in the future?
  ((UPat(Ops.MUL, dtype=dtypes.floats, name="a") + UPat.var("b")).named("x"),
    lambda a,b,x: x.ins(V_FMA[a.dtype], src=a.src + (b,))),
  # binary alu ops
  ((UPat() + UPat()).named("x"), lambda x: x.ins(V_ADD[x.dtype])),
  ((UPat() * UPat()).named("x"), lambda x: x.ins(V_MUL[x.dtype])),
  (UPat(Ops.SUB, name="x"), lambda x: x.ins(V_SUB[x.dtype])),
  (UPat(Ops.MAX, name="x"), lambda x: x.ins(V_MAX[x.dtype])),
  (UPat(Ops.XOR, dtype=dt_32bit, name="x"), lambda x: x.ins(RDNA3Ops.v_xor_b32_e32)),
  # note: *_e64 cmp and cndmask encoding allows for storage/usage of VCC as SGPR
  (UPat.var("m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"),
    lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e64, src=(b,a,cmp(m)))),
  # cmp, materialize mask -> 0/1 in VGPR
  (UPat(GroupOp.Comparison, dtypes.bool, name="x"),
   lambda x: x.ins(RDNA3Ops.v_cndmask_b32_e64, dtype=dtypes.uint32, src=(const(dtypes.uint32,0), const(dtypes.uint32,1), cmp(x)))),
  # mem ops
  (UPat.var("addr").store(UPat.var("val"), name="x"), store),
  (UPat(Ops.LOAD, name="x", src=(UPat.var("addr"))), load),
  # bit shifts
  # ((UPat(name="a", dtype=dt_16bit) << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b16, src=(b,a))),
  ((UPat(name="a") << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b32_e32, src=(b,a))),
  (UPat(Ops.STACK, name="x"), stack),
  (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(RDNA3Ops.s_barrier)),
  # allocate virtual registers
  (UPat((Ops.INS, Ops.BUFFER), name="x"), alloc_vregs),
  # normalize and satisfy operand orders/reg types, this should probably be handled per pattern?
  (UPat(Ops.INS, name="x"), legalize_operands),
])

def encode(x:UOp):
  if x.arg in [RDNA3Ops.s_nop, RDNA3Ops.s_endpgm]: return x.replace(arg=x.arg())
  from tinygrad.renderer.amd.dsl import v as dsl_v, s as dsl_s, NULL as dsl_null, VCC as dsl_vcc
  def _route(r:Register): return dsl_vcc if r.name == "vcc" else dsl_v if r.name[0] == "v" else dsl_s
  def _immorreg(x:UOp): return x.arg if x.op == Ops.CONST else _fuse(regs(x))
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc, oprs = x.arg, x.arg.func, x.arg.opc, x.src

  # hacky fixes, find cleaner way to conform to isa
  kw = args = None
  if group is RDNA3Ops.SMEM:
    kw = dict(sdata=_fuse(regs(x)), sbase=_fuse(tuple(u.tag[0] for u in oprs[:-1])), soffset=dsl_null, offset=oprs[-1].arg)
  elif group is RDNA3Ops.GLOBAL:
    kw = dict(addr=_immorreg(oprs[0]), saddr=_fuse(regs(oprs[1])), offset=_immorreg(oprs[2]))
    if reg(x) is None: kw["data"]=_fuse(regs(oprs[3]))
    else: kw["vdst"]=_fuse(regs(x))
  elif group is RDNA3Ops.SOPK: args = [dsl_null, oprs[0].arg]
  elif group in [RDNA3Ops.VOP3, RDNA3Ops.VOP2, RDNA3Ops.VOP1, RDNA3Ops.VOPC, RDNA3Ops.SOP1, RDNA3Ops.SOP2]: # alu
    args = [_fuse(regs(x))] + [_immorreg(u) for u in x.src]
  else: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={opc}")

  ret = enc(**kw) if kw is not None else enc(*args)
  nx = x.replace(arg=ret)
  return nx

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm)])),
  (UPat(Ops.INS, name="x"), lambda x: (x,[]) if x.arg is RDNA3Ops.s_nop else None),
  # strip everything but Ops.INS to bypass render rewrite
  (UPat((Ops.NOOP, Ops.BUFFER, Ops.CONST, Ops.GROUP, Ops.STACK, Ops.INDEX, Ops.GEP, Ops.AFTER, Ops.RANGE, Ops.END), name="x"), lambda ctx,x: (x,[])),
])

# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L250
from enum import Enum, auto
class CounterType(Enum):
  DS_CNT = auto(); LOAD_CNT = auto(); STORE_CNT = auto()

def _counter(x:UOp):
  if x.arg.func in [RDNA3Ops.DS, RDNA3Ops.SMEM]: return CounterType.DS_CNT
  elif "load" in x.arg.opc: return CounterType.LOAD_CNT
  else: return CounterType.STORE_CNT

@dataclass
class RDNA3LinearCtx:
  loop_label: dict[UOp, str] = field(default_factory=dict)

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.SIN, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR)}
  post_regalloc_ctx = RDNA3LinearCtx()
  def __init__(self, target:Target):
    super().__init__(target)

  # hack for now, should be removed from ISARenderer (should be CPU/GPU agnostic)
  def stack_pointer(self) -> UOp: return def_reg(dtypes.uint32, GP_SGPRS[-2])
  def spill(self, disp:UOp, x:UOp) -> UOp: return x
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: return x

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    bld_cntstate: dict[CounterType, int] = {CounterType.DS_CNT:0, CounterType.LOAD_CNT:0, CounterType.STORE_CNT:0}
    fill_cntstate: dict[CounterType, int] = {CounterType.DS_CNT:0, CounterType.LOAD_CNT:0, CounterType.STORE_CNT:0}
    # maps op that consumes sync dependent register to cnt requirement + pc from which this cnt is required
    tosync: dict[Register, tuple[CounterType, list[tuple[int, int]]]] = {}
    nuops = []
   
    waitins = { CounterType.DS_CNT : RDNA3Ops.s_waitcnt_lgkmcnt, CounterType.LOAD_CNT : RDNA3Ops.s_waitcnt_vmcnt, CounterType.STORE_CNT : RDNA3Ops.s_waitcnt_vscnt }
    for i, u in enumerate(lin.src):
      if u.arg.func not in [RDNA3Ops.GLOBAL, RDNA3Ops.SMEM, RDNA3Ops.DS]: continue
      ctp = _counter(u)
      if reg(u) is not None:
        for r in regs(u): tosync.setdefault(r,(ctp,[]))[1].append((i,bld_cntstate[ctp]))
      bld_cntstate[ctp] += 1

    for i, u in enumerate(lin.src):
      if u.arg.func in [RDNA3Ops.GLOBAL, RDNA3Ops.SMEM, RDNA3Ops.DS]: fill_cntstate[(ctp:=_counter(u))] += 1
      deps = [r for s in u.src for r in regs(s) if r in tosync]
      waits = {}
      for r in deps:
        tp, pts = tosync[r]
        cnt = next((n for j,n in reversed(pts) if i > j), None)
        if (cnt is not None) and fill_cntstate[tp] > cnt:
          fill_cntstate[tp] = waits[tp] = min(waits.setdefault(tp, float('inf')), cnt)
      nuops.extend([UOp(Ops.INS, arg=waitins[tp], src=(const(dtypes.uint16,n),)) for tp,n in waits.items()])
      nuops.append(u)

    lin = lin.replace(src=tuple(encode(u) for u in nuops if u.arg is not RDNA3Ops.s_nop))

    print(prg.arg)
    for u in lin.src: print(u.arg)
     
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

  # def asm_str(self, uops:list[UOp], function_name:str) -> str:

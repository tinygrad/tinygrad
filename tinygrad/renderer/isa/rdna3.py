from tinygrad.dtype import PtrDType, dtypes, AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, regs, reg
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

EXEC_LO = Register("exec_lo", 0)
VCC = Register("vcc", 0)
VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
# contiguous multi-register constraint tuples, ex. v[4:5], v[6:9]
GP_SGPRS, GP_VGPRS = tuple(SGPRS[5:]), tuple(VGPRS[1:])
# VGPR_PAIRS = tuple((GP_VGPRS[i],GP_VGPRS[i+1]) for i in range(0, 254, 2))
# VGPR_QUADS = tuple(VGPR_PAIRS[i] + VGPR_PAIRS[i+1] for i in range(0, len(VGPR_PAIRS)//2, 2))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)

# def geopc(x:UOp): return "" if not isinstance(x.arg, InsOp) else x.arg.args[0].name.lower()
def const(dt, v:int) -> UOp: return UOp.const(dt,v)

# def map_addrspace(x:UOp, local_ins, global_ins) -> UOp|None: return local_ins if x.addrspace == AddrSpace.LOCAL else global_ins if x.addrspace == AddrSpace.GLOBAL else None
def def_reg(dt, reg:Register): return UOp(Ops.DEFINE_REG, dt, tag=(reg,))
def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # real registers
  if x.op is Ops.DEFINE_REG and x.tag is not None: return None
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
  # pin kernarg ptr and contiguous gp sgprs for load
  return x.ins(RDNA3Ops.s_load_b64,
               src=(def_reg(dtypes.uint32, KERNARG_PTR[0]), def_reg(dtypes.uint32, KERNARG_PTR[1]), UOp.const(dtypes.uint32, offs).rtag()),
               tag=(ctx.vreg(GP_SGPRS[2*i]), ctx.vreg(GP_SGPRS[2*i+1])))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

def is_vgpr(x:UOp) -> bool: return x.tag is not None and x.tag != True and x.tag != GP_SGPRS and x.tag[0].cons[0].name[0] == "v"
def to_vgpr(x:UOp) -> UOp: return x.ins(RDNA3Ops.v_mov_b32_e32, src=(x,)) if x.op is Ops.CONST else x

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]: # returns addr, data, saddr (offset=0x0)
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), UOp.const(dtypes.int16, 0))
  def _offs(v:int) -> UOp: return UOp.const(dtypes.int16, ((1 << 13) - 1) & v).rtag() # TODO: handle overflow
  # def _const(v:int) -> UOp: return UOp.const(dtypes.int32, v)
  base, idx = x.src
  # TODO: handle multi-register index ex. 64 bit SGPR pair
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  # really should get stored in sgpr
  shft = to_vgpr(const(dtypes.int, disp_scale // 2))
  if idx.op is Ops.CONST: return (idx.ins(RDNA3Ops.v_mov_b32_e32, src=(const(dtypes.uint32, 0),)), base, _offs(idx.arg * disp_scale))
  # NOTE: dont cast for now so I dont need to impl cast alu
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0] << shft, base, _offs(idx.src[1].arg * disp_scale))
  # For now dont use immediate offset (set to 0x0)
  # lane relative offsets need to be stored in vgpr
  return (idx << shft, base, _offs(0))

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
V_CMPEQ = { dtypes.float16:RDNA3Ops.v_cmp_nlg_f16_e64,dtypes.float32:RDNA3Ops.v_cmp_nlg_f32_e64,dtypes.float64:RDNA3Ops.v_cmp_nlg_f64_e64 }
V_CMPNE = { dtypes.float16:RDNA3Ops.v_cmp_neq_f16_e64,dtypes.float32:RDNA3Ops.v_cmp_neq_f32_e64,dtypes.float64:RDNA3Ops.v_cmp_neq_f64_e64, dtypes.uint32:RDNA3Ops.v_cmp_ne_u32_e64,
  dtypes.int32:RDNA3Ops.v_cmp_ne_i32_e64, dtypes.int16:RDNA3Ops.v_cmp_ne_i16_e64, dtypes.uint16:RDNA3Ops.v_cmp_ne_u16_e64 }
V_CVT = {
  dtypes.uint16:  { dtypes.float16:RDNA3Ops.v_cvt_u16_f16_e32 },
  dtypes.float16: { dtypes.float32:RDNA3Ops.v_cvt_f16_f32_e32,  dtypes.uint16:RDNA3Ops.v_cvt_f16_u16_e32, dtypes.int16:RDNA3Ops.v_cvt_f16_i16_e32   },
  dtypes.float32: { dtypes.uint32:RDNA3Ops.v_cvt_f32_u32_e32,   dtypes.int32:RDNA3Ops.v_cvt_f32_i32_e32,  dtypes.float16:RDNA3Ops.v_cvt_f32_f16_e32,
    dtypes.float64:RDNA3Ops.v_cvt_f32_f64_e32 },
  dtypes.uint32:  { dtypes.float32:RDNA3Ops.v_cvt_u32_f32_e32,  dtypes.float64:RDNA3Ops.v_cvt_u32_f64_e32,dtypes.uint16:RDNA3Ops.v_cvt_u32_u16_e32  },
  dtypes.int32:   { dtypes.float32:RDNA3Ops.v_cvt_i32_f32_e32,  dtypes.float64:RDNA3Ops.v_cvt_i32_f64_e32,dtypes.int16:RDNA3Ops.v_cvt_i32_i16_e32   },
}

def legalize_operands(x:UOp):
  group, opc = x.arg.func, x.arg.opc
  # group, opc = x.arg.func, geopc(x)
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
  return None

# IDEA: casting utility function, auto converts between compatible/unsupported hardware dtypes
# if mapping dt_a -> dt_b does not exist automatically search for cvt chain path, else raise exception
def cvt(a:UOp, x:UOp):
  implct = {
    (dtypes.uint16, dtypes.uint8, dtypes.bool): dtypes.uint16,
    (dtypes.uint, dtypes.uint32) : dtypes.uint32,
    (dtypes.float, dtypes.float32) : dtypes.float32,
    (dtypes.float16,) : dtypes.float16,
    (dtypes.int16,) : dtypes.int16
  }
  try:
    for match, dt in implct.items():
      if a.dtype in match: return x.ins(V_CVT[dt][x.dtype])
  except:
    raise NotImplementedError(f"no cast instruction for target dtype mapping: {a.dtype} -> {x.dtype}")

def cmp(x:UOp):
  dt = x.src[0].dtype
  if x.op is Ops.CMPLT: ins = V_CMPLT[dt]
  elif x.op is Ops.CMPEQ: ins = V_CMPEQ[dt]
  elif x.op is Ops.CMPNE: ins = V_CMPNE[dt]
  else: ins = V_CMPGT[dt]

  # else: raise NotImplementedError("comparison type instruction dne")
  return x.ins(ins, tag=GP_SGPRS)

"""
def gated_load(ctx, base:UOp, idx:UOp, cast:UOp, alt:UOp, gate:UOp, x:UOp):
  local = UOp(Ops.DEFINE_LOCAL, base.dtype.base.ptr(x.dtype.count, AddrSpace.LOCAL), arg=next(ctx)).rtag()
  local_idx = local.index(const(dtypes.int32, 0), ptr=True)
  ptr = gate.where(base.index(idx, ptr=True), local_idx).after((local_idx if x.dtype.count == 1 else local).store(alt))
  return ptr.cast(cast.dtype).load(dtype=x.dtype)
"""

pre_isel_matcher = PatternMatcher([
  # (UPat.var("base").index(UPat.var("idx")).or_casted(name="cast").load(UPat.var("alt"), UPat.var("gate"), name="x"), gated_load),
  # cast to ptr is noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
])

# def _contiguous_groups(n, base_set): return tuple(tuple(base_set[i*n:(i+1)*n]) for i in range(len(base_set) // n))
def stack(ctx:IselContext, x:UOp):
  _slice = Register.contiguous(ctx, GP_VGPRS, len(x.src))
  movs = [u.ins(RDNA3Ops.v_mov_b32_e32, tag=(vr,), src=(u,)) for vr, u in zip(_slice, x.src)]
  return UOp.group(*movs)

# all we need to do is prune the group op emitted by stack through an ins pattern that 
# searches children for pseudo ops

# TODO: handle unsupported dtypes in pre_isel_matcher by casting?
# TODO: cleanup!! maybe make pseudops in regalloc dependent on arch/renderer
# TODO: check for uniformity for SALU usage instead, like x86 is_foldable?
isel_matcher = PatternMatcher([
  # (UPat(src=UPat(Ops.STACK, name="x"), name="y"), lambda x,y: None),
  # noop
  (UPat.var("a").cast(name="x"), lambda a,x: a if a.dtype == x.dtype else None),

  # rtag every const, masks tag type as non Register to ensure it doesn't get treated as one
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),

  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM, Ops.DEFINE_VAR), name="x"), abi),

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

  # note: *_e64 cmp and cndmask encoding allows for storage/usage of VCC as SGPR
  (UPat.var("m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"),
    lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e64, src=(b,a,cmp(m)))),

  # comparisons
  # ((UPat(dtype=dt_32bit) > UPat()).named("x"), lambda x: cmp(x,True)),
  # (UPat(Ops.CMPEQ, name="x", src=(UPat(dtype=dt_32bit), UPat())), lambda x: cmp(x,True)),
  # (UPat(Ops.CMPNE, name="x", src=(UPat(dtype=dt_32bit), UPat())), lambda x: cmp(x,True)),
  
  # materialize mask -> 0/1 in VGPR
  (UPat(GroupOp.Comparison, dtypes.bool, name="x"),
   lambda x: x.ins(RDNA3Ops.v_cndmask_b32_e64, dtype=dtypes.uint32, src=(const(dtypes.uint32,0), const(dtypes.uint32,1), cmp(x)))),

  # casts
  # (UPat(dtype=(dtypes.uint8, dtypes.bool, dtypes.uint16)).cast(name="x"), lambda x: x.ins(V_CVT[dtypes.uint16][x.dtype])),
  # (UPat.var("a").cast(name="x"), lambda a,x: x.ins(V_CVT[a.dtype][x.dtype])),

  # mem ops
  (UPat(Ops.LOAD, dt_128bit, name="x", src=(UPat(name="idx"))),
    lambda ctx,x,idx: x.ins(RDNA3Ops.global_load_b128, src=fold_address(idx), tag=(GP_VGPRS, 4))),
  (UPat.var("a").store(UPat.var("b", dtype=dt_128bit), name="x"),
    lambda a,b,x: x.ins(RDNA3Ops.global_store_b128, dtype=dtypes.void, src=fold_address(a) + (b,))),

  (UPat(Ops.LOAD, dt_32bit, name="x", src=(UPat(name="idx"))),
    lambda x,idx: x.ins(RDNA3Ops.global_load_b32, src=fold_address(idx))),
  (UPat.var("a").store(UPat.var("b", dtype=dt_32bit), name="x"),
    lambda a,b,x: x.ins(RDNA3Ops.global_store_b32, dtype=dtypes.void, src=fold_address(a) + (b,))),

  # bit shifts
  # ((UPat(name="a", dtype=dt_16bit) << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b16, src=(b,a))),
  ((UPat(name="a") << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b32_e32, src=(b,a))),

  (UPat(Ops.STACK, name="x"), stack),

  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs),

  # normalize and satisfy operand orders/reg types
  (UPat(Ops.INS, name="x"), legalize_operands),
])

# wait maybe this is unecessary, src already models the relationship 
pre_regalloc_matcher = PatternMatcher([
])

def encode(x:UOp):
  from tinygrad.renderer.amd.dsl import v as dsl_v, s as dsl_s, NULL as dsl_null, VCC as dsl_vcc
  def _route(r:Register): return dsl_vcc if r.name == "vcc" else dsl_v if r.name[0] == "v" else dsl_s
  def _immorreg(x:UOp): return x.arg if x.op == Ops.CONST else _fuse(regs(x))
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc, oprs, suffix = x.arg, x.arg.func.__name__, x.arg.opc, x.src, []

  # hacky fixes, find cleaner way to conform to isa
  kw = args = None
  if group == "SMEM": kw = dict(sdata=_fuse(regs(x)), sbase=_fuse(tuple(u.tag[0] for u in oprs[:-1])), soffset=dsl_null, offset=oprs[-1].arg)
  elif group == "GLOBAL":
    kw = dict(addr=_immorreg(oprs[0]), saddr=_fuse(regs(oprs[1])), offset=_immorreg(oprs[2]))
    if x.tag is None: kw["data"]=_fuse(regs(oprs[3]))
    else: kw["vdst"]=_fuse(regs(x))
  elif group == "SOPK": args = [dsl_null, oprs[0].arg]
  elif group[:3] in ["VOP", "SOP"]: args = [_fuse(regs(x))] + [_immorreg(u) for u in x.src]
  else: raise NotImplementedError(f"instruction type encoding unsupported, ins group={group}, opcode={opc}")

  # sync loads across wave
  if "load" in opc: suffix.append(encode(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt_lgkmcnt if group == "SMEM" else RDNA3Ops.s_waitcnt_vmcnt, src=(UOp.const(dtypes.uint16, 0),)))[0])

  # print(opc, [regs(u) for u in x.src], args)
  ret = enc(**kw) if kw is not None else enc(*args)
  nx = x.replace(arg=ret)
  return nx, [nx] + suffix

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm())])),

  # strip everything but Ops.INS to bypass render rewrite
  (UPat((Ops.DEFINE_REG, Ops.CONST, Ops.GROUP, Ops.STACK, Ops.GEP, Ops.AFTER), name="x"), lambda ctx,x: (x,[])),

  # final operand legalization and then filling of dsl Inst class partials from autogen
  (UPat(Ops.INS, name="x"), encode),
])

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.SIN, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE)}
  def __init__(self, target:Target):
    super().__init__(target)

  # hack for now
  def stack_pointer(self) -> UOp: return def_reg(dtypes.uint32, GP_SGPRS[-2])
  def spill(self, disp:UOp, x:UOp) -> UOp: return x
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: return x

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    # expects arg of every op in lin to be filled Inst class
    print(prg.arg)
    for u in lin.src: print(u.arg)
    return assemble_linear(prg, lin, self.target.arch)

  """
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    asm = [f"{function_name}:"]
    uops = inswaits(uops)
    for u in uops:
      if u.op is not Ops.INS: continue
      asm.append(str(encode(u)[0]))
    return "\n\t".join(asm)
  """

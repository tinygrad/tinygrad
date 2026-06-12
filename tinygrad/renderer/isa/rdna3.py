from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

VCC = Register("vcc", 0)
VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
GP_SGPRS, GP_VGPRS = tuple(SGPRS[4:]), tuple(VGPRS[1:])

def geopc(x:UOp): return "" if not isinstance(x.arg, InsOp) else x.arg.args[0].name.lower()
def const(dt, v:int) -> UOp: return UOp.const(dt,v)

# def map_addrspace(x:UOp, local_ins,global_ins) -> UOp|None: return local_ins if x.addrspace == AddrSpace.LOCAL else global_ins if x.addrspace == AddrSpace.GLOBAL else None
def def_reg(dt, reg:Register): return UOp(Ops.DEFINE_REG, dt, tag=(reg,))
def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # real registers
  if x.op is Ops.DEFINE_REG and x.tag is not None: return None
  # no register definition
  if x.dtype is dtypes.void: return None
  # already allocated vregs
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None
  # allocate vreg definitions
  defs = []
  # don't generally allocate to SGPRS, only works wave uniform possible future optim
  if isinstance(x.tag, tuple): defs = [ctx.vreg(x.tag)]
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
               src=(def_reg(dtypes.uint32, KERNARG_PTR[0]), def_reg(dtypes.uint32, KERNARG_PTR[1]),
                    UOp.const(dtypes.uint32, offs).rtag()),
               tag=(ctx.vreg(GP_SGPRS[2*i]), ctx.vreg(GP_SGPRS[2*i+1])))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2 and dt not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4 and dt not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8 and dt not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

def is_vgpr(x:UOp) -> bool: return x.tag is not None and x.tag != True and x.tag[0].cons[0].name[0] == "v"
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
  if idx.op is Ops.CONST: return (UOp(Ops.NOOP), base, _offs(idx.arg * disp_scale))
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
V_RCP =   { dtypes.float16:RDNA3Ops.v_rcp_f16_e32,  dtypes.float32:RDNA3Ops.v_rcp_f32_e32,  dtypes.float64:RDNA3Ops.v_rcp_f64_e32   }
V_SIN =   { dtypes.float16:RDNA3Ops.v_sin_f16_e32,  dtypes.float32:RDNA3Ops.v_sin_f32_e32 }
V_TRUNC = { dtypes.float16:RDNA3Ops.v_trunc_f16_e32,dtypes.float32:RDNA3Ops.v_trunc_f32_e32,dtypes.float64:RDNA3Ops.v_trunc_f64_e32 }
V_FMA =   { dtypes.float16:RDNA3Ops.v_fma_f16,      dtypes.float32:RDNA3Ops.v_fma_f32,      dtypes.float64:RDNA3Ops.v_fma_f64       }
V_CMPLT = { dtypes.float16:RDNA3Ops.v_cmp_lt_f16_e32, dtypes.float32:RDNA3Ops.v_cmp_lt_f32_e32, dtypes.float64:RDNA3Ops.v_cmp_lt_f64_e32, dtypes.uint32:RDNA3Ops.v_cmp_lt_u32_e32,
  dtypes.int32:RDNA3Ops.v_cmp_lt_i32_e32, dtypes.int16:RDNA3Ops.v_cmp_lt_i16_e32, dtypes.uint16:RDNA3Ops.v_cmp_lt_u16_e32 }
V_CMPGT = { dtypes.float16:RDNA3Ops.v_cmp_gt_f16_e32, dtypes.float32:RDNA3Ops.v_cmp_gt_f32_e32, dtypes.float64:RDNA3Ops.v_cmp_gt_f64_e32, dtypes.uint32:RDNA3Ops.v_cmp_gt_u32_e32,
  dtypes.int32:RDNA3Ops.v_cmp_gt_i32_e32, dtypes.int16:RDNA3Ops.v_cmp_gt_i16_e32, dtypes.uint16:RDNA3Ops.v_cmp_gt_u16_e32 }
V_CMPEQ = { dtypes.float16:RDNA3Ops.v_cmp_nlg_f16_e32,dtypes.float32:RDNA3Ops.v_cmp_nlg_f32_e32,dtypes.float64:RDNA3Ops.v_cmp_nlg_f64_e32 }
V_CMPNE = { dtypes.float16:RDNA3Ops.v_cmp_neq_f16_e32,dtypes.float32:RDNA3Ops.v_cmp_neq_f32_e32,dtypes.float64:RDNA3Ops.v_cmp_neq_f64_e32, dtypes.uint32:RDNA3Ops.v_cmp_ne_u32_e32,
  dtypes.int32:RDNA3Ops.v_cmp_ne_i32_e32, dtypes.int16:RDNA3Ops.v_cmp_ne_i16_e32, dtypes.uint16:RDNA3Ops.v_cmp_ne_u16_e32 }

def legalize_operands(x:UOp):
  group, opc = x.arg.func, geopc(x)
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

def cmp(x:UOp,vcc:bool=False):
  dt = x.src[0].dtype
  if x.op is Ops.CMPLT: ins = V_CMPLT[dt]
  elif x.op is Ops.CMPEQ: ins = V_CMPEQ[dt]
  elif x.op is Ops.CMPNE: ins = V_CMPNE[dt]
  else: ins = V_CMPGT[dt]
  # else: raise NotImplementedError("comparison type instruction dne")
  return x.ins(ins, tag=(VCC,) if vcc else True)

# TODO: handle unsupported dtypes in pre_isel_matcher by casting?
# TODO: check for uniformity for SALU usage instead, like x86 is_foldable?
isel_matcher = PatternMatcher([
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

  # TODO: How to handle bool cast?
  # ex. (a < b, dtype=dtypes.bool).cast(dtypes.uint) ...
  # OH this is actually a case where I need to load from vcc cause that was the output of cmp
  # need to somehow check if the output of cmp is used, in that case use cmpx ins??

  # conditional moves, VCC used immediately after, doesn't need to be stored in register
  # - folds cmp instruction into src to preserve before discarding in encode
  (UPat(Ops.CMPLT, name="m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"), lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e32, src=(b,a,cmp(m)))),
  (UPat(Ops.CMPEQ, name="m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"), lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e32, src=(b,a,cmp(m)))),
  (UPat(Ops.CMPNE, name="m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"), lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e32, src=(b,a,cmp(m)))),
  ((UPat() > UPat()).named("m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"), lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e32, src=(b,a,cmp(m)))),

  # comparisons
  ((UPat(dtype=dt_32bit) > UPat()).named("x"), cmp),
  ((UPat(dtype=dt_32bit) < UPat()).named("x"), cmp),
  (UPat(Ops.CMPEQ, name="x", src=(UPat(dtype=dt_32bit), UPat())), cmp),
  (UPat(Ops.CMPNE, name="x", src=(UPat(dtype=dt_32bit), UPat())), cmp),

  # mem ops
  (UPat(Ops.LOAD, dt_32bit, name="x", src=(UPat(name="idx"))),
    lambda x,idx: x.ins(RDNA3Ops.global_load_b32, src=fold_address(idx))),
  (UPat.var("a").store(UPat.var("b", dtype=dt_32bit), name="x"),
    lambda a,b,x: x.ins(RDNA3Ops.global_store_b32, dtype=dtypes.void, src=fold_address(a) + (b,))),

  # bit shifts
  # ((UPat(name="a", dtype=dt_16bit) << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b16, src=(b,a))),
  ((UPat(name="a") << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b32_e32, src=(b,a))),

  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs),

  # normalize and satisfy operand orders/reg types
  (UPat(Ops.INS, name="x"), legalize_operands),
])

def fillinsts(x:UOp):
  from tinygrad.renderer.amd.dsl import v as dsl_v, s as dsl_s, NULL as dsl_null, VCC as dsl_vcc
  def _route(r:Register): return dsl_vcc if r.name == "vcc" else dsl_v if r.name[0] == "v" else dsl_s
  def _immorreg(x:UOp): return x.arg if x.op == Ops.CONST else _fuse(x.tag)
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    return r[rr[0].index:rr[0].index+len(rr)-1] if len(rr) > 1 else r[rr[0].index]
  enc, group, opc = x.arg, x.arg.func.__name__, geopc(x)
  oprs, suffix = x.src, []

  # hacky fixes, find cleaner way to conform to isa
  if "cndmask" in opc: oprs = oprs[:-1]
  if "load" in opc:
      suffix.append(fillinsts(UOp(Ops.INS,
                                  arg=RDNA3Ops.s_waitcnt_lgkmcnt if group == "SMEM" else RDNA3Ops.s_waitcnt_vmcnt,
                                  src=(UOp.const(dtypes.uint16, 0),)))[0])
  kw = args = None
  if group == "SMEM": kw = dict(sdata=_fuse(x.tag), sbase=_fuse(tuple(u.tag[0] for u in oprs[:-1])), soffset=dsl_null, offset=oprs[-1].arg)
  elif group == "GLOBAL":
    kw = dict(addr=_immorreg(oprs[0]), saddr=_fuse(oprs[1].tag), offset=_immorreg(oprs[2]))
    if x.tag is None: kw["data"]=_fuse(oprs[3].tag)
    else: kw["vdst"]=_fuse(x.tag)
  elif "VOP" in group: args = ([_fuse(x.tag)] if group != "VOPC" else []) + [_immorreg(u) for u in x.src]
  elif group == "SOPK": args = [dsl_null, oprs[0].arg]
  else: raise NotImplementedError("instruction type encoding unsupported")
  ret = enc(**kw) if kw is not None else enc(*args)
  nx = x.replace(arg=ret)
  return nx, [nx] + suffix

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(RDNA3Ops.s_endpgm())])),

  # strip everything but Ops.INS to bypass render rewrite
  (UPat((Ops.DEFINE_REG, Ops.CONST, Ops.GROUP), name="x"), lambda ctx,x: (x,[])),

  # final operand legalization and then filling of dsl Inst class partials from autogen
  (UPat(Ops.INS, name="x"), fillinsts),
])

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = PatternMatcher([])
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.LOG2, Ops.EXP2, Ops.SUB, Ops.RECIPROCAL, Ops.SIN, Ops.TRUNC, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE)}
  def __init__(self, target:Target):
    super().__init__(target)

  # hack for now
  def stack_pointer(self) -> UOp: return def_reg(dtypes.uint32, GP_SGPRS[-1])
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

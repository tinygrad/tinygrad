from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate
from tinygrad.helpers import Target
from tinygrad.renderer.amd.dsl import InsOp
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

# Goals in progress:
# (1) impl and verify cmp semantics, cmp vs cmpx depending on if result is used
#   - .where() move is a weird case
# (2) get casts working
# (3) clean up / unify encoding pass in post_regalloc
# (4) optimize s_waitcnt insertions?
# (5) unify/clean up to_vgpr*s? 
# *try not to add too many instructions without being able to verify them

VCC = Register("vcc", 0)
VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
KERNARG_PTR, WGIDS, WIIDS = tuple(SGPRS[:2]), tuple(SGPRS[2:5]), (VGPRS[0],)
GP_SGPRS, GP_VGPRS = tuple(SGPRS[4:]), tuple(VGPRS[1:])

def _geopc(x:UOp):
  return "" if not isinstance(x.arg, InsOp) else x.arg.args[0].name.lower()
def _const(dt, v:int) -> UOp: return UOp.const(dt,v)
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
    else: return x.ins(RDNA3Ops.v_bfe_u32, src=(def_reg(dtypes.uint32, WIIDS[0]), _const(dtypes.uint32, 10 * dim), _const(dtypes.uint32, 10)))
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

# def is_sgpr(x:UOp) -> bool: return x.tag[0].cons[0].name[0] == "s"
def is_vgpr(x:UOp) -> bool: return x.tag is not None and x.tag != True and x.tag[0].cons[0].name[0] == "v"
def to_vgpr(x:UOp):
  if x.op is Ops.CONST: return x.ins(RDNA3Ops.v_mov_b32_e32, src=(x,))
  return x

# idx may need to be added then shifted?
def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]: # returns addr, data, saddr (offset=0x0)
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), UOp.const(dtypes.int16, 0))
  def _offs(v:int) -> UOp: return UOp.const(dtypes.int16, ((1 << 13) - 1) & v).rtag() # TODO: handle overflow
  # def _const(v:int) -> UOp: return UOp.const(dtypes.int32, v)
  base, idx = x.src
  # TODO: handle multi-register index ex. 64 bit SGPR pair
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  # really should get stored in sgpr
  shft = to_vgpr(_const(dtypes.int, disp_scale // 2))
  if idx.op is Ops.CONST: return (UOp(Ops.NOOP), base, _offs(idx.arg * disp_scale))
  # NOTE: dont cast for now so I dont need to impl cast alu
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0] << shft, base, _offs(idx.src[1].arg * disp_scale))
  # For now dont use immediate offset (set to 0x0)
  # lane relative offsets need to be stored in vgpr
  return (idx << shft, base, _offs(0))

V_ADD =   { dtypes.float16:RDNA3Ops.v_add_f16_e32,  dtypes.float32:RDNA3Ops.v_add_f32_e32, dtypes.float64:RDNA3Ops.v_add_f64,   dtypes.int32:RDNA3Ops.v_add_nc_i32,       dtypes.uint32:RDNA3Ops.v_add_nc_u32_e32,  }
V_SUB =   { dtypes.float16:RDNA3Ops.v_sub_f16_e32,  dtypes.float32:RDNA3Ops.v_sub_f32_e32, dtypes.int32:RDNA3Ops.v_sub_nc_i32,  dtypes.uint32:RDNA3Ops.v_sub_nc_u32_e32,  }
V_MUL =   { dtypes.float16:RDNA3Ops.v_mul_f16_e32,  dtypes.float32:RDNA3Ops.v_mul_f32_e32, dtypes.float64:RDNA3Ops.v_mul_f64,   dtypes.int32:RDNA3Ops.v_mul_i32_i24_e32,  dtypes.uint32:RDNA3Ops.v_mul_u32_u24_e32, }
V_SQRT =  { dtypes.float16:RDNA3Ops.v_sqrt_f16_e32, dtypes.float32:RDNA3Ops.v_sqrt_f32_e32, dtypes.float64:RDNA3Ops.v_sqrt_f64_e32 }
V_CMPLT = { dtypes.float16:RDNA3Ops.v_cmp_lt_f16_e32, dtypes.float32:RDNA3Ops.v_cmp_lt_f32_e32, dtypes.float64:RDNA3Ops.v_cmp_lt_f64_e32, dtypes.uint32:RDNA3Ops.v_cmp_lt_u32_e32,
  dtypes.int32:RDNA3Ops.v_cmp_lt_i32_e32, dtypes.int16:RDNA3Ops.v_cmp_lt_i16_e32, dtypes.uint16:RDNA3Ops.v_cmp_lt_u16_e32 }
V_CVT = {
  dtypes.float32 : {
    dtypes.float16 : RDNA3Ops.v_cvt_f32_f16_e32,
    dtypes.uint32 : RDNA3Ops.v_cvt_f32_u32_e32,
    dtypes.int32  : RDNA3Ops.v_cvt_f32_i32_e32,
    dtypes.float64 : RDNA3Ops.v_cvt_f32_f64_e32,
  },
  dtypes.uint32 : {
    dtypes.float32 : RDNA3Ops.v_cvt_u32_f32_e32,
    dtypes.float64 : RDNA3Ops.v_cvt_u32_f64_e32,
    dtypes.uint16 : RDNA3Ops.v_cvt_u32_u16_e32
  },
}

def legalize_operands(x:UOp):
  group, opc = x.arg.func, _geopc(x)
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

isel_matcher = PatternMatcher([
  # rtag every const, masks tag type as non Register to ensure it doesn't get treated as one
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),

  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM, Ops.DEFINE_VAR), name="x"), abi),

  # unary alu ops
  # (UPat.var(name="y").sqrt().named("x"), lambda y,x: x.ins(V_SQRT[x.dtype], src=(y,))),

  # binary alu ops
  # TODO: handle unsupported dtypes in pre_isel_matcher by casting?
  ((UPat() + UPat()).named("x"), lambda x: x.ins(V_ADD[x.dtype])),
  ((UPat() * UPat()).named("x"), lambda x: x.ins(V_MUL[x.dtype])),

  # TODO: How to handle bool cast?
  # ex. (a < b, dtype=dtypes.bool).cast(dtypes.uint) ...
  # OH this is actually a case where I need to load from vcc cause that was the output of cmp
  # need to somehow check if the output of cmp is used, in that case use cmpx ins??

  # (UPat(name="a", dtype=dtypes.bool).cast(name="x", dtype=dtypes.uints), lambda a,x: _const(x.dtype, int(a.arg))),
  (UPat.var("a").cast(name="x"), lambda a,x: x.ins(V_CVT[a.dtype][x.dtype])),
  (UPat(Ops.SUB, name="x"), lambda x: x.ins(V_SUB[x.dtype])),


  # conditional moves
  ((UPat.var("a", dtype=dt_32bit) < UPat.var("b")).named("m"),
    lambda a,b,m: m.ins(V_CMPLT[a.dtype], src=(a,b), dtype=dtypes.uint32, tag=(VCC,))),
  (UPat(Ops.INS, name="m").where(UPat.var("a", dtype=dt_32bit), UPat().var("b")).named("x"),
    lambda m,a,b,x: x.ins(RDNA3Ops.v_cndmask_b32_e32, src=(a,b,m))),

  # mem ops
  (UPat(Ops.LOAD, dt_32bit, name="x", src=(UPat(name="idx"))),
    lambda x,idx: x.ins(RDNA3Ops.global_load_b32, src=fold_address(idx))),
  (UPat.var("a").store(UPat.var("b", dtype=dt_32bit), name="x"),
    lambda a,b,x: x.ins(RDNA3Ops.global_store_b32, dtype=dtypes.void, src=fold_address(a) + (b,))),

  # hack
  ((UPat(name="a") << UPat(name="b")).named("x"), lambda a,b,x: x.ins(RDNA3Ops.v_lshlrev_b32_e32, src=(b,a))),

  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs),

  # normalize and satisfy operand orders/reg types
  (UPat(Ops.INS, name="x"), legalize_operands),
])

# TODO: clean up this slop, use **kw for all op types?
def fillinsts(x:UOp):
  from tinygrad.renderer.amd.dsl import v as dsl_v, s as dsl_s, NULL as dsl_null, VCC as dsl_vcc
  enc, group, opc = x.arg, x.arg.func.__name__, x.arg.args[0].name.lower()
  def _route(r:Register): return dsl_vcc if r.name == "vcc" else dsl_v if r.name[0] == "v" else dsl_s
  def _immorreg(x:UOp):
    if x.op == Ops.CONST: return x.arg
    else: return _route(x.tag[0])[x.tag[0].index]
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    if len(rr) > 1: return r[rr[0].index:rr[0].index+len(rr)-1]
    else: return r[rr[0].index]
  oprs, fields = x.src, [_fuse(x.tag)] if x.tag is not None else []
  suffix = []
  if group == "SMEM":
    assert oprs[-1].op == Ops.CONST
    sbase = oprs[0].tag if len(oprs) == 2 else oprs[0].tag + oprs[1].tag
    fields.extend([_fuse(sbase), dsl_null, oprs[-1].arg])
    ret = enc(*fields)
    if "load" in opc:
      suffix.append(fillinsts(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt_lgkmcnt, src=(UOp.const(dtypes.uint16, 0),)))[0])
  elif group == "GLOBAL":
    vaddr, base, offs = oprs[0], oprs[1], oprs[2]
    kw = dict(addr=_immorreg(vaddr), saddr=_fuse(base.tag), offset=_immorreg(offs))
    if x.tag is None: kw["data"]=_fuse(oprs[3].tag)
    else:
      suffix.append(fillinsts(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt_vmcnt, src=(UOp.const(dtypes.uint16, 0),)))[0])
      kw["vdst"]=_fuse(x.tag)
    ret = enc(**kw)
  elif "VOP" in group:
    # remove stale cmps? find better way to do this
    if "cndmask" in opc: oprs = oprs[:-1]
    if "cmp" in opc: oprs = oprs[1:]
    fields.extend([_immorreg(u) for u in oprs])
    ret = enc(*fields)
  elif group == "SOPK":
    fields.extend([dsl_null, oprs[0].arg])
    ret = enc(*fields)
  else:
    raise NotImplementedError("instruction type encoding unsupported")
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

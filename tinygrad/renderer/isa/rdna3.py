from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops

# Baseline functionality: add 2 tensors (a + b)
# Requirements:
# - abi/kernargs
# - register allocation
# - ds ops, load/store lowering and address folding
# - alu ins
# - ins encoding
# 
# Notes:
# - AMD memory/SALU are async. Need s_waitcnt after global_load
#   - implement as flush pass after isel matching
# - Wave front size is set to 32 in amd/elf.py
# - look at amd/elf.py  for kernel launching semantics/metadata
# - exec masking??

"""
First principles thinking:

- regalloc + renderer stubs handles loading register values
- need widened vregs for instructions like global_load_b64/b128 etc.. v[4:8]
- start with pinned pairs

- goal, ensure UOp src operands come in order of instruction encoding to satisfy isa, makes encoding easier
  - look at vop2()
  - explicitly emit register moves if necessary
- where should register fusion happen? at encoding step probably not in izsel_matcher

Target program input AST:

Tensor.empty(3,3) + Tensor.empty(3,3)

c0 = UOp(Ops.PARAM, dtypes.float.ptr(9), (), ParamArg(0))
c2 = UOp.special(3, 'lidx0', dtype=dtypes.int)
c3 = UOp.special(3, 'gidx0', dtype=dtypes.int)
c5 = c2+c3*UOp.const(dtypes.int, 3)
c7 = UOp(Ops.PARAM, dtypes.float.ptr(9), (), ParamArg(1))
c9 = c7.index(c5, ptr=True).load()
c10 = UOp(Ops.PARAM, dtypes.float.ptr(9), (), ParamArg(2))
c12 = c10.index(c5, ptr=True).load()
c13 = c9+c12
c15 = c0.index(c5, ptr=True).store(c13).end(c3, c2)

handwritten asm:

kernel:
"""

VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(106))
# Unless the Target Properties column of AMDGPU Processors specifies otherwise, a separate VGPR register is used per work-item ID.
WGIDS, WIIDS = tuple(SGPRS[2:5]), tuple(VGPRS[:3])
# s[0:1] reserved for kernarg ptr
GP_SGPRS, GP_VGPRS = tuple(SGPRS[4:]), tuple(VGPRS[3:])

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
    return def_reg(dtypes.int, WGIDS[dim] if x.arg[0] == 'g' else WIIDS[dim])
  offs = sum(8 if u.op == Ops.PARAM else 4 for u in ctx.func_args[:i])
  # pin kernarg ptr and contiguous gp sgprsfor load
  return x.ins(RDNA3Ops.s_load_b64,
               src=(def_reg(dtypes.uint32, SGPRS[0]), def_reg(dtypes.uint32, SGPRS[1]),
                    UOp.const(dtypes.uint32, offs).rtag()),
               tag=(ctx.vreg(GP_SGPRS[2*i]), ctx.vreg(GP_SGPRS[2*i+1])))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2 and dt not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4 and dt not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8 and dt not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

def is_sgpr(x:UOp) -> bool: return x.tag[0].cons[0].name[0] == "s"
def is_imm(x:UOp) -> bool: return x.tag == True
def is_vgpr(x:UOp) -> bool: return x.tag is not None and (not is_imm(x)) and x.tag[0].cons[0].name[0] == "v"
def to_vgpr(x:UOp):
  if is_vgpr(x): return x
  # TODO: different move instruction based on dtype size?
  return x.ins(RDNA3Ops.v_mov_b32_e32, src=(x,), tag=None) # tag=None forces vreg GP_VGPR alloc

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]: # returns addr, data, saddr (offset=0x0)
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), UOp.const(dtypes.int16, 0))
  def _offs(v:int) -> UOp: return UOp.const(dtypes.int16, ((1 << 13) - 1) & v).rtag() # TODO: handle overflow
  # def _const(v:int) -> UOp: return UOp.const(dtypes.int32, v)
  base, idx = x.src
  # TODO: handle multi-register index ex. 64 bit SGPR pair
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.CONST: return (UOp(Ops.NOOP), base, _offs(idx.arg * disp_scale))
  # NOTE: dont cast for now so I dont need to impl cast alu
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (idx.src[0], base, _offs(idx.src[1].arg * disp_scale))

  # For now dont use immediate offset (set to 0x0)
  # lane relative offsets need to be stored in vgpr
  return (idx, base, _offs(0))

V_ADD = {
  dtypes.float16 : RDNA3Ops.v_add_f16_e32,  dtypes.float32 : RDNA3Ops.v_add_f32_e32,
  dtypes.float64 : RDNA3Ops.v_add_f64,      dtypes.int32 : RDNA3Ops.v_add_nc_i32,
  dtypes.uint32 : RDNA3Ops.v_add_nc_u32_e32,
}

V_MUL = {
  dtypes.float16 : RDNA3Ops.v_mul_f16_e32,  dtypes.float32 : RDNA3Ops.v_mul_f32_e32,
  dtypes.float64 : RDNA3Ops.v_mul_f64,      dtypes.int32 : RDNA3Ops.v_mul_i32_i24_e32,
  dtypes.uint32 : RDNA3Ops.v_mul_u32_u24_e32,
}

# ensures vsrc1 is a vgpr operand
def legalize_vop2(x:UOp):
  if x.arg.func is not RDNA3Ops.VOP2: return None
  if any(s.tag is None for s in x.src): return None
  a, b = x.src
  if is_vgpr(b): return None
  if is_vgpr(a): return x.replace(src=(b,a))
  return x.replace(src=((a, to_vgpr(b))))

isel_matcher = PatternMatcher([
  # rtag every const, masks tag type as non Register to ensure it doesn't get treated as one
  (UPat.cvar("x"), lambda x: x.rtag() if not x.tag else None),

  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM, Ops.DEFINE_VAR), name="x"), abi),

  # alu ops
  ((UPat() + UPat()).named("x"), lambda x: x.ins(V_ADD[x.dtype])),
  ((UPat() * UPat()).named("x"), lambda x: x.ins(V_MUL[x.dtype])),

  # mem ops
  (UPat(Ops.LOAD, dt_32bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_load_b32, src=fold_address(idx))),
  # store value b in addr a?
  (UPat.var("a").store(UPat.var("b", dtype=dt_32bit), name="x"), lambda a,b,x: x.ins(RDNA3Ops.global_store_b32, src=fold_address(a) + (b,))),

  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs),

  # normalize and satisfy operand orders/reg types
  (UPat(Ops.INS, name="x"), legalize_vop2),
])

from tinygrad.renderer.amd.dsl import Inst
def encode(x:UOp) -> tuple[Inst, bytes]:
  from tinygrad.renderer.amd.dsl import v as dsl_v, s as dsl_s, NULL as dsl_null
  enc, group, opc = x.arg, x.arg.func.__name__, x.arg.args[0].name.lower()
  def _route(r:Register): return dsl_v if r.name[0] == "v" else dsl_s
  def _immorreg(x:UOp):
    if x.op == Ops.CONST: return x.arg
    else: return _route(x.tag[0])[x.tag[0].index]
  def _fuse(rr:tuple[Register,...]):
    r = _route(rr[0])
    if len(rr) > 1: return r[rr[0].index:rr[0].index+len(rr)-1]
    else: return r[rr[0].index]
  oprs, fields = x.src, [_fuse(x.tag)] if x.tag is not None else []
  if group == "SMEM":
    assert oprs[-1].op == Ops.CONST
    sbase = oprs[0].tag if len(oprs) == 2 else oprs[0].tag + oprs[1].tag
    fields.extend([_fuse(sbase), dsl_null, oprs[-1].arg])
    ret = enc(*fields)
  elif group == "GLOBAL":
    vaddr, base, offs = oprs[0], oprs[1], oprs[2]
    kw = dict(addr=_immorreg(vaddr), saddr=_fuse(base.tag), offset=_immorreg(offs))
    if x.tag is None: kw["data"]=_fuse(oprs[3].tag)
    else: kw["vdst"]=_fuse(x.tag)
    ret = enc(**kw)
  elif "VOP" in group:
    fields.extend([_immorreg(u) for u in oprs])
    ret = enc(*fields)
  elif group == "SOPK":
    fields.extend([dsl_null, oprs[0].arg])
    ret = enc(*fields)
  else:
    raise NotImplementedError("instruction type encoding unsupported")
  return ret, ret.to_bytes()

# insert s_waitcnt where necessary to sync async ops across wave ex. global load
def inswaits(uops:list[UOp]):
  # v1 naive just insert s_waitcnt vmcnt(0) after every s/global load
  # v2 use s_waitcnt vmcnt(N) on usage boundaries, detect when loaded register is used again
  nuops = []
  for u in uops:
    nuops.append(u)
    if u.op is not Ops.INS: continue
    opc = u.arg.args[0].name.lower()
    if "load" in opc:
      nuops.append(UOp(Ops.INS, arg=RDNA3Ops.s_waitcnt_vmcnt, src=(UOp.const(dtypes.uint16, 0),)))
  return nuops

class RDNA3Renderer(ISARenderer):
  device = "AMD"
  pre_isel_matcher = PatternMatcher([])
  isel_matcher = isel_matcher
  post_regalloc_matcher = PatternMatcher([])
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    self.compiler = AMDLLVMCompiler(arch="rdna3")

  # hack for now
  def stack_pointer(self) -> UOp: return def_reg(dtypes.uint32, GP_SGPRS[-1])
  def spill(self, disp:UOp, x:UOp) -> UOp: return x
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: return x
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # TODO: global load flush pass, insert s_waitcnt_vmcnt(0) before first ALU op that consumes result
    # TODO: before s_load use s_waitcnt lgkmnct(0)
    pass

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    asm = [f"{function_name}:"]
    uops = inswaits(uops)
    for u in uops:
      if u.op is not Ops.INS: continue
      asm.append(str(encode(u)[0]))
    return "\n\t".join(asm)

  def render(self, uops:list[UOp]) -> str:
    binary = bytearray()
    for u in uops:
      if u.op is not Ops.INS: continue
      binary.extend(encode(u)[1])
    return binary.hex()

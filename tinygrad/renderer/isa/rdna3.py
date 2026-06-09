from tinygrad.dtype import PtrDType, dtypes, AddrSpace, truncate
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
import tinygrad.runtime.autogen.amd.rdna3.ins as RDNA3Ops
from tinygrad.renderer.amd.elf import assemble_linear

# Baseline functionality: add 2 tensors (a + b)
# Requirements:
# - abi/kernargs
# - register allocation
# - ds ops, load/store lowering and address folding
# - alu ins
# - ins encoding
#
# Implement:
# - abi(), make kernel parameters/variable definitions match abi
# - fold_address(), extract rdna3 style addressing information out of load/store op?
# - alloc_vregs(), allocate virtual registers according to ops
#   - hardware register allocation is performed via codegen/late/regalloc.py, linear scan
# 
# Notes:
# - AMD memory/SALU are async. Need s_waitcnt after global_load
#   - implement as flush pass after isel matching
# - Wave front size is set to 32 in amd/elf.py
# - look at amd/elf.py  for kernel launching semantics/metadata
# - exec masking??

# TODO: need some way to represent Register ranges in constraint mechanisms/regalloc
# ex. kernarg ptr is s[0:1]
# - maybe its just done via tag constraints and Op.DEFINE_REG args specific to this isa
VGPRS = tuple(Register(f"v{i}", i) for i in range(256))
SGPRS = tuple(Register(f"s{i}", i) for i in range(256))
WGIDS = tuple(Register(f"s{i}", i) for i in (2, 3, 4)))

def map_addrspace(x:UOp, local_ins,global_ins) -> UOp|None:
  return local_ins if x.addrspace == AddrSpace.LOCAL else global_ins if x.addrspace == AddrSpace.GLOBAL else None

def _to_sgpr(v:UOp) -> UOp: return v.ins(RDNA3Ops.s_mov_b32, src=(v,), tag=SGPR_PAIRS)
def _to_vgpr(v:UOp) -> UOp: return v.ins(RDNA3Ops.v_mov_b32_e32, src=(v,), tag=VGPRS)

# https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
def def_reg(dt, reg:Register|tuple[Register,...], nregs:int=1): return UOp(Ops.DEFINE_REG, dt, args=nregs, tag=reg if isinstance(reg, tuple) else (reg,))
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  i = ctx.func_args.index(x)
  if x.op is Ops.SPECIAL:
    dim = int(x.arg[-1])
    if x.arg[0] == 'g': return x.ins(RDNA3Ops.s_mov_b32, src=(def_reg(dtypes.int, WGIDS[dim])), tag=SGPRS)
    else: return x.ins(RDNA3Ops.v_mov_b32_e32, src=(def_reg(dtypes.int, VGPRS[dim])))
def _size(x:UOp): return 8 if x.op == Ops.PARAM else 4
  offs = sum(_size(u) for u in ctx.func_args[:i])
# return x.ins(RDNA3Ops.s_load_b64, src=(def_reg(dtypes.uint64, KERNARG_PTR),UOp.const(dtypes.uint32, offs)), tag=SGPR_PAIRS)

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), UOp.const(dtypes.int16, 0))
  def _offs(v:int) -> UOp:
    # TODO: handle overflow
    return UOp.const(dtypes.int16, ((1 << 13) - 1) & v)
  # Either 64 bit address in VGPR or 32 bit VGOR offset and 32 bit SGPR base
  # - start with saddr mode, implement full 64 bit addresing by emitting vaddr add op?
  # assume base is wave unfiorm, use saddr mode
  base, idx = x.src
  # TODO: saddr should be representing as adjacent SGPRS?? ex. s[5:6]
  base = _to_sgpr(base.cast(dtypes.uint64)) # 64 bit SGPR
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.CONST: return (base, UOp(Ops.NOOP), _offs(idx.arg * disp_scale))
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (base, _to_vgpr(idx.src[0].cast(dtypes.uint32)), _offs(idx.src[1].arg * disp_scale))
  return (base, _to_vgpr(idx.cast(dtypes.uint32)), _offs(0))

# VGPR only allocation baseline, SGPRs only needed for abi, special cases and eventually wave uniformity optimization?
def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # real registers
  if x.op is Ops.DEFINE_REG and x.tag is not None: return None
  # no register definition
  if x.dtype is dtypes.void: return None
  # already allocated vregs
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None
  # allocate vreg definitions
  defs = []
  if isinstance(x.tag, tuple): defs = [ctx.vreg(x.tag)]
  else: defs = [ctx.vreg(VGPRS)]
  return x.replace(tag=tuple(defs))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2 and dt not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4 and dt not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8 and dt not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

isel_matcher = PatternMatcher([
  # function abi
  (UPat((Ops.SPECIAL, Ops.PARAM, Ops.DEFINE_VAR), name="x"), abi),

  # alu ops
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(RDNA3Ops.v_add_f32_e32)),
  ((UPat(dtype=dtypes.int32) + UPat().named("x")), lambda x: x.ins(RDNA3Ops.v_add_nc_i32)),

  # mem ops
  (UPat(Ops.LOAD, dt_32bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_load_b32, src=fold_address(idx))),
  (UPat(Ops.LOAD, dt_64bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_load_b64, src=fold_address(idx))),
  (UPat(Ops.LOAD, dt_128bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_load_b128, src=fold_address(idx))),
  (UPat(Ops.STORE, dt_32bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_store_b32, src=fold_address(idx))),
  (UPat(Ops.STORE, dt_64bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_store_b64, src=fold_address(idx))),
  (UPat(Ops.STORE, dt_128bit, name="x", src=(UPat(name="idx"))), lambda x,idx: x.ins(RDNA3Ops.global_store_b128, src=fold_address(idx))),

  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs)
])

class RDNA3Renderer(ISARenderer):
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    self.compiler = AMDLLVMCompiler(arch="rdna3")

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # TODO: global load flush pass, insert s_waitcnt_vmcnt(0) before first ALU op that consumes result
    # TODO: before s_load use s_waitcnt lgkmnct(0)
    pass
    # return assemble_linear(prg, lin, "rdna3")

  def render(self, uops:list[UOp]) -> str:
    pass

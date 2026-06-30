from __future__ import annotations

from tinygrad.device import CompileError
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import Target
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.renderer.amd.dsl import v
from tinygrad.runtime.autogen.amd.rdna3 import ins as r3
from tinygrad.runtime.autogen.amd.rdna3.isa_renderer import (
  AMDOps, AMD_ATOMIC_ADD, SGPR, VGPR, LID, WGID, KERNARG_REG, TMP_VDATA, TMP_VADDR, TMP_EXEC, TMP_SDATA0, TMP_SDATA1,
  make_isel_matcher, isel_matcher, pre_isel_matcher, pre_regalloc_matcher, post_regalloc_matcher,
  insts_for_uop, insts_from_linear, _lower_reg_store, _parallel_vmov,
)
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp

class AMDRenderer(ISARenderer):
  device = "AMD"
  has_local = True
  has_shared = True
  supports_float4 = True
  float4_dtypes = (dtypes.float32,)
  wide_regalloc = True
  preferred_reduce_group = 16
  global_max = (0x8fffffff, 0x8fffffff, 0x8fffffff)
  local_max = (1024, 1, 1)
  local_prod_max = 1024
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  _code_ops = (Ops.ADD, Ops.SUB, Ops.MUL, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN, Ops.MAX,
               Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ)
  code_for_op = {op: (lambda: None) for op in _code_ops}

  def __init__(self, target:Target):
    if not target.arch.startswith("gfx11"): raise RuntimeError(f"AMDRenderer is RDNA3/gfx11 only, got {target.arch}")
    super().__init__(target)

  def stack_pointer(self) -> UOp: return UOp(Ops.INS, dtypes.uint32, arg=AMDOps.SCRATCH_BASE)
  def register_slots(self, x:UOp, vreg:Register|None=None) -> int:
    if vreg is None or x.dtype.count == 1 or not all(c.index >= 256 for c in vreg.cons): return 1
    return max(1, (x.dtype.itemsize + 3) // 4)
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), AMDOps.MOV, (reg,))
  def spill(self, disp:UOp, x:UOp) -> UOp:
    if x.reg.index < 256: raise CompileError("AMDRenderer does not support SGPR spills yet")
    return UOp(Ops.INS, dtypes.void, (disp, x), AMDOps.SPILL)
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    if reg.index < 256: raise CompileError("AMDRenderer does not support SGPR fills yet")
    return UOp(Ops.INS, x.dtype, (disp,), AMDOps.FILL, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    ret = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is AMDOps.LABEL: ret.append(f"{u.tag}:")
      elif u.arg in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1): ret.append(f"  {u.arg.name.lower()} {u.tag}")
      else: ret.append(f"  {u.arg.name.lower()} " + ", ".join(str(s.reg or s.arg) for s in u.src))
    return "\n".join(ret)

  def render(self, uops:list[UOp]) -> str: return self.asm_str(uops, "kernel")
  def _insts_for_uop(self, u:UOp): return insts_for_uop(u)
  def _insts_from_linear(self, lin:UOp): return insts_from_linear(lin)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    insts = self._insts_from_linear(lin)
    insts.append(r3.s_endpgm())
    nlin = lin.replace(src=tuple(UOp(Ops.INS, arg=i) for i in insts))
    return assemble_linear(prg, nlin, self.target.arch)

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.float16, dtypes.float32}

from __future__ import annotations
from typing import List, Set, Optional, Tuple, Any, Dict
from tinygrad.helpers import DEBUG, flatten
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.shape.symbolic import Variable, sint
from enum import Enum, auto
from dataclasses import dataclass

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); IF = auto(); ENDLOOP = auto(); ENDIF = auto(); SPECIAL = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); CONST = auto(); BARRIER = auto(); PHI = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto(); GEP = auto() # noqa: E702

@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType]
  vin: Tuple[UOp, ...]
  arg: Any
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"

def get_recursive_children(uops:List[UOp], x:UOp) -> Set[UOp]:
  deps = set([x])
  ssize = 0
  while ssize != len(deps):
    ssize = len(deps)
    for u in uops:
      if len(deps.intersection([x for x in u.vin if x.uop != UOps.PHI])):
        deps.add(u)
  return deps

UOPS_W_SIDE_EFFECTS = {UOps.STORE, UOps.BARRIER, UOps.DEFINE_GLOBAL}
def remove_childless_uops(uops:List[UOp]) -> List[UOp]:
  # NOTE: DEFINE_GLOBAL should be removable, but we'd have to propagate that
  while 1:
    has_child: Set[UOp] = set()
    for ru in uops:
      for vu in ru.vin:
        has_child.add(vu)
    nu: List[UOp] = [x for x in uops if x in has_child or x.uop in UOPS_W_SIDE_EFFECTS]
    if len(nu) == len(uops): break
    if DEBUG >= 4: print(f"reduced UOp count from {len(uops)} to {len(nu)}")
    uops = nu
    del nu
  return uops

def fix_loop_scope(get_recursive_parents, uops:List[UOp]) -> List[UOp]:
  loop_stack: List[List[UOp]] = [[]]
  # push uops upward out of loop if it does not depend on the loop
  for u in uops:
    if not loop_stack[-1]: loop_stack[-1].append(u)
    elif u.uop == UOps.LOOP: loop_stack.append([u])
    elif u.uop not in [UOps.CONST, UOps.ALU, UOps.CAST, UOps.LOAD]: loop_stack[-1].append(u)
    else:
      parents = get_recursive_parents(u, with_phi=True)
      # don't push any local buffer because there might have STORE and BARRIER (not considered as parent) between DEFINE_LOCAL and here
      if any(u.uop == UOps.DEFINE_LOCAL for u in parents): loop_stack[-1].append(u)
      else:
        for i in reversed(range(len(loop_stack))):
          # check backwards and put the uop in the first encounter with some dependency
          if any(x in parents for x in loop_stack[i]) or i == 0:
            loop_stack[i].append(u)
            break
  return flatten(loop_stack)

# optional
def uops_type_verify(uops:List[UOp]):
  for u in uops:
    uop, arg, vin, dtype = u.uop, u.arg, u.vin, u.dtype
    if uop == UOps.ALU:
      if arg in UnaryOps:
        assert dtype == vin[0].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=}"
      elif arg in (BinaryOps.CMPLT, BinaryOps.CMPEQ):
        assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
        assert vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
      elif arg in BinaryOps:
        assert dtype == vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
      elif arg == TernaryOps.WHERE:
        assert vin[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {vin[0].dtype=} != {dtypes.bool}"
        assert dtype == vin[1].dtype == vin[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {vin[1].dtype=} != {vin[2].dtype=}"

def uops_alu_resolve(u:UOp, vars:Dict[str, Variable]) -> sint:
  if u.uop == UOps.CONST: return u.arg
  elif u.uop == UOps.DEFINE_GLOBAL: return vars[u.arg]
  elif u.uop == UOps.ALU and u.arg == BinaryOps.MUL:
    return uops_alu_resolve(u.vin[0], vars) * uops_alu_resolve(u.vin[1], vars)
  elif u.uop == UOps.ALU and u.arg == BinaryOps.ADD:
    return uops_alu_resolve(u.vin[0], vars) + uops_alu_resolve(u.vin[1], vars)
  else:
    raise RuntimeError(f"ALU resolve fail @ {u.uop}")

def uops_flops_mem(uops:List[UOp], vars:Dict[str, Variable]) -> Tuple[sint, sint]:
  flops: sint = 0
  mem: sint = 0
  mults: sint = 1
  mult_stack = []
  for u in uops:
    if u.uop is UOps.LOOP:
      mult_stack.append(mults)
      mults *= uops_alu_resolve(u.vin[1], vars)
    if u.uop is UOps.ENDLOOP:
      mults = mult_stack.pop(-1)
    if u.uop is UOps.ALU:
      flops += (2 if u.arg is TernaryOps.MULACC else 1) * mults
    if u.uop is UOps.LOAD:
      assert u.dtype is not None
      mem += u.dtype.itemsize * mults
    if u.uop is UOps.STORE:
      assert u.vin[2].dtype is not None
      mem += u.vin[2].dtype.itemsize * mults
    if u.uop is UOps.WMMA:
      if u.arg.startswith("__metal_wmma"): flops += 2*(8*8*8)//32 * mults
      elif u.arg == "__hip_wmma_f16_f16" or u.arg == "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32": flops += 2*(16*16*16)//32 * mults
      else: raise Exception("not implemented")
  return flops, mem

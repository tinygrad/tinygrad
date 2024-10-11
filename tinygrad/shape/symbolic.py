from __future__ import annotations
from typing import Union, Optional, Dict, Literal
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, exec_alu, ConstType

sint = Union[int, UOp]

def NumNode(val:int): return UOp.const(dtypes.int, val)
def Variable(expr:str, nmin:ConstType, nmax:ConstType): return UOp.define_var(expr, dtypes.int, nmin, nmax)

def sym_infer(uop: Union[UOp, int], var_vals: Optional[Dict[UOp[Literal[UOps.DEFINE_VAR]], int]]) -> int:
  if isinstance(uop, (int, float)): return uop   # TODO: ugh, the float is a hack for qcom
  if uop.op == UOps.CONST: return uop.arg
  if uop.op == UOps.DEFINE_VAR and var_vals is not None: return var_vals[uop]
  if uop.op == UOps.BIND: return uop.src[1].arg  # bound variable returns bound value
  if uop.op == UOps.ALU:
    src_values = [sym_infer(src, var_vals) for src in uop.src]
    return exec_alu(uop.arg, uop.dtype, src_values)
  raise NotImplementedError(f"Unsupported UOp {uop.op}")

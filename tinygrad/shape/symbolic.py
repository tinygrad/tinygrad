from __future__ import annotations
from typing import Tuple, Union, Optional, Dict
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, ConstType, UOps, exec_alu

sint = Union[int, UOp]

# broken
Node = UOp
MulNode = UOp
SumNode = UOp
DivNode = UOp
ModNode = UOp
LtNode = UOp
AndNode = UOp

def NumNode(val:int): return UOp.const(dtypes.pyint, val)
def Variable(expr:str, nmin:int, nmax:int): return UOp.define_var(expr, dtypes.pyint, nmin, nmax)

def sym_infer(uop: Union[UOp, int], var_vals: Optional[Dict[UOp, int]]) -> int:
  if isinstance(uop, int): return uop
  if uop.op == UOps.CONST: return uop.arg
  if uop.op == UOps.DEFINE_VAR and var_vals is not None: return var_vals[uop]
  if uop.op == UOps.ALU:
    src_values = [sym_infer(src, var_vals) for src in uop.src]
    return exec_alu(uop.arg, uop.dtype, src_values)
  raise NotImplementedError(f"Unsupported UOp {uop.op}")
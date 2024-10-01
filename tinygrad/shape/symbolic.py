from typing import Union, Dict, Optional
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, exec_alu

def Variable(expr:str, nmin:int, nmax:int):
  return UOp.define_var(expr, dtypes.pyint, nmin, nmax)

# broken
Node = UOp
MulNode = UOp
SumNode = UOp
DivNode = UOp
ModNode = UOp
LtNode = UOp
AndNode = UOp
NumNode = UOp

sint = Union[int, UOp]

def sym_infer(uop: Union[UOp, int], var_vals: Optional[Dict[UOp, int]]) -> int:
  if isinstance(uop, int): return uop
  if uop.op == UOps.CONST:
    return uop.arg
  elif uop.op == UOps.DEFINE_VAR:
    return var_vals[uop]
  elif uop.op == UOps.ALU:
    src_values = [sym_infer(src, var_vals) for src in uop.src]
    return exec_alu(uop.arg, uop.dtype, src_values)
  else:
    raise NotImplementedError(f"Unsupported UOp {uop.op}")

from __future__ import annotations
from typing import Union, Optional, Dict, cast
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, exec_alu, ConstType

sint = Union[int, UOp]

# broken
Node = UOp
MulNode = UOp
SumNode = UOp
DivNode = UOp
ModNode = UOp
LtNode = UOp
AndNode = UOp
def NumNode(val:int): return UOp.const(dtypes.int, val)

class Variable(UOp):
  def __new__(cls, expr:str, nmin:ConstType, nmax:ConstType):  # pylint: disable=signature-differs
    return super().__new__(cls, UOps.DEFINE_VAR, dtypes.int, arg=(expr, nmin, nmax))
  def __init__(self, expr:str, nmin:ConstType, nmax:ConstType):
    super().__init__(UOps.DEFINE_VAR, dtypes.int, arg=(expr, nmin, nmax))
  def bind(self, val:int):
    assert self.op is UOps.DEFINE_VAR, f"op is {self.op}"
    return UOp(UOps.ASSIGN, self.dtype, (self, self.const_like(val)))
  @property
  def expr(self): return self.arg[0]

def sym_infer(uop: Union[UOp, int], var_vals: Optional[Dict[Variable, int]]) -> int:
  if isinstance(uop, int): return uop
  if uop.op == UOps.CONST: return uop.arg
  if uop.op == UOps.DEFINE_VAR and var_vals is not None: return var_vals[cast(Variable, uop)]
  if uop.op == UOps.ASSIGN: return uop.src[1].arg  # bound variable returns bound value
  if uop.op == UOps.ALU:
    src_values = [sym_infer(src, var_vals) for src in uop.src]
    return exec_alu(uop.arg, uop.dtype, src_values)
  raise NotImplementedError(f"Unsupported UOp {uop.op}")

from __future__ import annotations
from typing import Tuple, Union, Optional, Dict, cast
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

vcache: Dict[str, UOp] = {}
class Variable(UOp):
  def __new__(cls, expr:str, nmin:int, nmax:int, bound:Optional[ConstType]=None):
    if expr in vcache and bound is None: return vcache[expr]
    return super().__new__(cls)
  def __init__(self, expr:str, nmin:int, nmax:int, bound:Optional[ConstType]=None):
    if bound is None: vcache[expr] = self
    super().__init__(UOps.DEFINE_VAR, dtypes.pyint, arg=(expr, nmin, nmax, bound) if bound is not None else (expr, nmin, nmax))
  def bind(self, val:ConstType):
    assert self.op is UOps.DEFINE_VAR and len(self.arg) == 3
    return Variable(self.arg[0], self.arg[1], self.arg[2], val)
  def unbind(self) -> Tuple[Variable, int]:
    assert self.op is UOps.DEFINE_VAR and len(self.arg) == 4
    return Variable(self.arg[0], self.arg[1], self.arg[2]), self.arg[3]
  @property
  def expr(self): return self.arg[0]
  @property
  def val(self):
    assert self.op is UOps.DEFINE_VAR and len(self.arg) == 4, f"no val for {self}"
    return self.arg[3]
  @staticmethod
  def sum(args): return sum(args)

def sym_infer(uop: Union[UOp, int], var_vals: Optional[Dict[Variable, int]]) -> int:
  if isinstance(uop, int): return uop
  if uop.op == UOps.CONST: return uop.arg
  if uop.op == UOps.DEFINE_VAR and var_vals is not None: return var_vals[cast(Variable, uop)]
  if uop.op == UOps.ALU:
    src_values = [sym_infer(src, var_vals) for src in uop.src]
    return exec_alu(uop.arg, uop.dtype, src_values)
  raise NotImplementedError(f"Unsupported UOp {uop.op}")

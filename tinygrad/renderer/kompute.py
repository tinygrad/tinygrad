import math
from tinygrad.codegen.uops import UOpGraph
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.renderer.cstyle import CStyleLanguage
from typing import Dict, List, Union
from typing import Tuple

type_map = {dtypes.float64: "double", dtypes.float: "float", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool"}

def cmplt(a, b, dtype):
  # quick fix to fix "bool < bool", get rid of this...
  if dtype == dtypes.bool: return f"(float({a})<float({b}))"
  return f"({a}<{b})"

class KomputeLanguage(CStyleLanguage):
  gid = [f"int(gl_WorkGroupID.{'xyz'[x]})" for x in range(3)]
  lid = [f"int(gl_LocalInvocationID.{'xyz'[x]})" for x in range(3)]
  xid = [f"int(gl_GlobalInvocationID.{'xyz'[x]})" for x in range(3)]
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype is dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b,dtype: f"({a}&&{b})" if dtype == dtypes.bool else f"({a}*{b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: cmplt(a, b, dtype),
    BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:UOpGraph, prefix=None) -> str:
    local_size = [1] # local_size[::-1] if local_size else [1]
    prg = "#version 450\n\n"
    local_sizes = [f"layout (local_size_{'xyz'[i]} = {i + 999}) in;" for i in range(len(local_size))]
    prg += "\n".join(local_sizes) + "\n"
    def r(t, n): return "{ " + f"{type_map[t if isinstance(t, DType) else t[0]]} {n}[];" + " }"
    prg += f"\nlayout (set = 0, binding = 0) writeonly buffer buf_data0 {r(bufs[0][1], 'data0')};\n"
    prg += "\n".join([f"layout (set = 0, binding = {i}) readonly buffer buf_{name} {r(dtype, name)};"
                      for i, (name,dtype) in enumerate(bufs) if name != "data0"])
    return prg + "\n\nvoid main() {\n" + "\n".join(kernel) + "\n}"

  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    assert not bitcast, "bitcast not supported"
    if type_map[var_dtype]:
      return f"{type_map[var_dtype]}({x[0]})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if math.isnan(x): val = "(0.0/0.0)"
    elif math.isinf(x): val = "(" + ("-" if x < 0 else "") + "1.0/0.0)"
    elif var_dtype == dtypes.float64: val = f"{float(x)}"
    else: val = f"{float(x)}f" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return (self.render_cast([val]*var_dtype.count, var_dtype)
      if var_dtype.count > 1 or var_dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

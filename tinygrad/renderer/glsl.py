from tinygrad.helpers import dtypes, DType
from tinygrad.renderer.cstyle import CStyleLanguage
from typing import List, Union
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
import math
from typing import Tuple, Dict

type_map = {dtypes.float: "float", dtypes.half: "float16", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool"}
sampler_prefix = {dtypes.float: "", dtypes.half: "", dtypes.int32: "i", dtypes.uint32: "u",dtypes.int8: "i", dtypes.uint8: "u", dtypes.bool: "i"}
fragment_center_offset = 0.5

class GLSLLanguage(CStyleLanguage):
  xid = ["int(gl_FragCoord.y-0.5) * w + int(gl_FragCoord.x-0.5)"]
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x: f"(-{x})",
    UnaryOps.EXP2: lambda x: f"exp2({x})",
    UnaryOps.LOG2: lambda x: f"log2({x})",
    UnaryOps.SIN: lambda x: f"sin({x})",
    UnaryOps.SQRT: lambda x: f"sqrt({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.MAX: lambda a,b: f"max({a},{b})", BinaryOps.MOD: lambda a,b: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b: f"float({a}<{b})", TernaryOps.MULACC: lambda a,b,c: f"(({a}*{b})+{c})",
    TernaryOps.WHERE: lambda a,b,c: f"(float({a})!=0.0?{b}:{c})"
  }

  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "intBitsToFloat(int(0xFFC00000u))"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "(1./0.)"
    else: val = "({:.1f})".format(x) if x == int(x) and dtypes.is_float(var_dtype) else f"({x})"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"bool({cond})?({x}):{y}"
  
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    local_size = local_size[::-1] if local_size else [1]
    prg = "#version 330\nin vec2 uv;\nuniform int w;\n"
    prg += "\n".join([f"uniform {sampler_prefix[dtype]}sampler2D {name};" for name,dtype in bufs if name != "data0"])
    dummy_line = "float dummy = float(texture(data1, vec2(0.0f,0.0f)).r);\n" if ("sampler2D data1" in prg) else ""
    dummy_line += "\nint xyz = w;\n"
    prg += f"\nout {sampler_prefix[bufs[0][1]]}vec4 out_data;\n"
    return prg + f"\nvoid main() {{\n{dummy_line}" + "\n".join(kernel) + "\n}"

  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    if type_map[var_dtype]: 
      return f"{type_map[var_dtype]}({x[0]})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    x_calc = f"({idx})%textureSize({buf_name}, 0).x"
    y_calc = f"({idx})/textureSize({buf_name}, 0).x"
    out_val = f"texture({buf_name}, vec2(float({x_calc} + 0.5)/textureSize({buf_name}, 0).x, float({y_calc} + 0.5)/textureSize({buf_name}, 0).y)).r"
    return f"{self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val}"

  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx, local=False) -> str:
    return f"out_data = {sampler_prefix[buf_dtype]}vec4({var_name}, 0.0, 0.0, 0.0);"
  
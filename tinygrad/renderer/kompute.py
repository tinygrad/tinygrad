from tinygrad.codegen.uops import UOpGraph
from tinygrad.dtype import dtypes, DType
from tinygrad.renderer.cstyle import CStyleLanguage
from typing import List
from typing import Tuple

type_map = {dtypes.float64: "double", dtypes.float: "float", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool"}

class KomputeLanguage(CStyleLanguage):
  gid = [f"int(gl_WorkGroupID.{'xyz'[x]})" for x in range(3)]
  lid = [f"int(gl_LocalInvocationID.{'xyz'[x]})" for x in range(3)]
  xid = [f"int(gl_GlobalInvocationID.{'xyz'[x]})" for x in range(3)]

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:UOpGraph, prefix=None) -> str:
    local_size = [1] # local_size[::-1] if local_size else [1]
    prg = "#version 450\n\n"
    local_sizes = [f"layout (local_size_{'xyz'[i]} = {i + 999}) in;" for i in range(len(local_size))]
    prg += "\n".join(local_sizes) + "\n"
    r = lambda t, n: "{ " + f"{type_map[t if isinstance(t, DType) else t[0]]} {n}[];" + " }"
    prg += f"\nlayout (set = 0, binding = 0) writeonly buffer buf_data0 {r(bufs[0][1], 'data0')};\n"
    prg += "\n".join([f"layout (set = 0, binding = {i}) readonly buffer buf_{name} {r(dtype, name)};" for i, (name,dtype) in enumerate(bufs) if name != "data0"])
    return prg + "\n\nvoid main() {\n" + "\n".join(kernel) + "\n}"

  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    assert not bitcast, "bitcast not supported"
    if type_map[var_dtype]:
      return f"{type_map[var_dtype]}({x[0]})"
    raise NotImplementedError(f"no cast for {var_dtype}")

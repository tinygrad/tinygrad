from typing import Optional, List, Tuple, Dict
import functools
from dataclasses import dataclass
from tinygrad.helpers import to_function_name
from tinygrad.codegen.uops import UOpGraph
from tinygrad.shape.symbolic import sym_infer, sint, Variable

@dataclass(frozen=True)
class Program:
  name:str
  src:str
  dname:str
  global_size:Optional[List[int]]=None
  local_size:Optional[List[int]]=None
  uops:Optional[UOpGraph]=None
  op_estimate:sint=0
  mem_estimate:sint=0

  @functools.cached_property
  def vars(self) -> List[Variable]: return [] if self.uops is None else self.uops.vars()

  @functools.cached_property
  def globals(self) -> List[Tuple[int, bool]]: return [] if self.uops is None else self.uops.globals()

  @functools.cached_property
  def outcount(self) -> int: return sum(x[1] for x in self.globals)

  @functools.cached_property
  def function_name(self) -> str: return to_function_name(self.name)

  def launch_dims(self, var_vals:Dict[Variable, int]):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else None
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else None
    return global_size, local_size

class Renderer:
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_shared: bool = True
  has_tensor_cores: bool = False
  # NOTE: these two should be in z,y,x(reversed) order for cstyle backends, they are flipped when kernel is rendered
  global_max: Optional[List[int]] = None
  local_max: Optional[List[int]] = None
  shared_max: int = 32768

  def render(self, name:str, uops:UOpGraph) -> str: raise NotImplementedError("needs a renderer")

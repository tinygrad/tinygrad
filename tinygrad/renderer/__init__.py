from typing import Optional, List, Tuple, Dict, Any
import functools
from dataclasses import dataclass
from tinygrad.helpers import to_function_name, dedup
from tinygrad.codegen.uops import UOps, UOp, flops_mem
from tinygrad.shape.symbolic import sym_infer, sint, Variable
from tinygrad.dtype import DType

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: Tuple[int,int,int] # N, M, K
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  threads: List[Tuple[int,int]] # list of (TC dim,amt) that construct the warp thread structure
  def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])

@dataclass(frozen=True)
class Program:
  name:str
  src:str
  dname:str
  global_size:Optional[List[int]]=None
  local_size:Optional[List[int]]=None
  uops:Optional[List[UOp]]=None
  mem_estimate:sint=0  # TODO: get this from the load/store uops once min/max are good

  @property
  def op_estimate(self) -> sint: return self._ops_lds[0]
  @property
  def lds_estimate(self) -> sint: return self._ops_lds[1]
  @functools.cached_property
  def _ops_lds(self) -> Tuple[sint, sint]: return (0,0) if self.uops is None else flops_mem(self.uops, ignore_indexing=True)

  @functools.cached_property
  def vars(self) -> List[Variable]:
    return [] if self.uops is None else sorted([x.arg for x in self.uops if x.op is UOps.DEFINE_VAR], key=lambda v: v.expr)
  @functools.cached_property
  def globals(self) -> List[int]: return [] if self.uops is None else [x.arg for x in self.uops if x.op is UOps.DEFINE_GLOBAL]

  @functools.cached_property
  def outcount(self) -> int: return 1 if self.uops is None else len(dedup([x.src[0] for x in self.uops if x.op is UOps.STORE]))

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
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now
  local_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: List[TensorCore] = []
  extra_matcher: Any = None

  def render(self, name:str, uops:List[UOp]) -> str: raise NotImplementedError("needs a renderer")

from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts
from tinygrad.ops import RawBuffer, Device, ASTRunner
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

@dataclass(frozen=True)
class JitItem:
  prg: ASTRunner
  rawbufs: List[Optional[RawBuffer]]

class BatchExecutor:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int]):
    self.jit_cache: List[JitItem] = jit_cache
    self.input_replace: Dict[Tuple[int, int], Union[int, str]] = {}
    for j,ji in enumerate(jit_cache):
      for i,a in enumerate(ji.rawbufs):
        if a in [v for v in input_rawbuffers.values()]:
          self.input_replace[(j,i)] = [k for k,v in input_rawbuffers.items() if v == a][0]
    assert set(self.input_replace.values()) == set(input_rawbuffers.keys()), "some input tensors not found"
    self.clear_jit_inputs()
  
  def __call__(self, input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int]):
    for (j,i),input_name in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name]
    for ji in self.jit_cache: ji.prg(cast(List[RawBuffer], ji.rawbufs), {v:var_vals[v] for v in getattr(ji.prg,"vars",[])}, jit=True)
    self.clear_jit_inputs()

  def clear_jit_inputs(self):
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.jit_fxn: Optional[BatchExecutor] = None
    self.cnt: int = 0
    self.ret: Any = None
    self.expected_vals: Optional[Tuple[Variable, ...]] = None
    self.expected_sts_dtype: Optional[Tuple[Tuple[ShapeTracker, DType], ...]] = None
  
  @property
  def jit_cache(self) -> List[JitItem]: return self.jit_fxn.jit_cache if self.jit_fxn else []

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT.split(":")[0] not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device

    # all inputs are realized
    input_tensors: Dict[Union[int, str], Tensor] = {cast(Union[int, str], k):v.realize() for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    expected_sts_dtype = tuple([(v.lazydata.st.unbind(), v.dtype) for v in input_tensors.values()])

    # get rawbuffers
    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {k:cast(RawBuffer, v.lazydata.realized) for k,v in input_tensors.items()}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.lazydata.st.var_vals for arg in input_tensors.values()] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])
    expected_vals = tuple(var_vals.keys())

    if self.cnt >= 2:
      assert self.expected_vals == expected_vals, "mismatch of var_vals"
      assert self.expected_sts_dtype == expected_sts_dtype, "mismatch of sts"
      assert self.jit_fxn, "didn't get jitted?"
      self.jit_fxn(input_rawbuffers, var_vals)
    elif self.cnt == 1:
      self.expected_vals, self.expected_sts_dtype = expected_vals, expected_sts_dtype

      CacheCollector.start(var_vals)
      self.ret = self.fxn(*args, **kwargs)
      jit_cache = CacheCollector.finish()
      assert len(jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      alt_batch_exec = Device[Device.DEFAULT].batch_executor
      self.jit_fxn = (BatchExecutor if alt_batch_exec is None else alt_batch_exec)(jit_cache, input_rawbuffers, var_vals)
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)

    self.cnt += 1
    return self.ret

class _CacheCollector:
  def __init__(self):
    self.cache: Optional[List[JitItem]] = None
  def start(self, var_vals:Optional[Dict[Variable, int]]=None):
    self.cache = []
    self.var_vals = var_vals if var_vals is not None else {}
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    for k,v in var_vals.items(): assert k in self.var_vals and self.var_vals[k] == v, f"var_vals {k} mismatch {v} != {self.var_vals.get(k)}"
    self.cache.append(JitItem(prg, rawbufs))
  def finish(self) -> List[JitItem]:
    if self.cache is None: return []
    ret = self.cache
    self.cache = None
    return ret
CacheCollector = _CacheCollector()

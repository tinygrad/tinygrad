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

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[JitItem] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]] = {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT.split(":")[0] not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device

    # all inputs are realized
    input_tensors: Dict[Union[int, str], Tensor] = {cast(Union[int, str], k):v.realize() for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}

    # get rawbuffers
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {k:(cast(RawBuffer, v.lazydata.realized), v.lazydata.st) for k,v in input_tensors.items()}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.lazydata.st.var_vals for arg in input_tensors.values()] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])

    if self.cnt >= 2:
      # check validity and assign the inputs
      for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
        assert input_rawbuffers[input_name][1].unbind() == expected_st, f"ShapeTracker mismatch in JIT, {input_rawbuffers[input_name][1].unbind()} != {expected_st}"
        self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name][0]
      for ji in self.jit_cache: ji.prg(cast(List[RawBuffer], ji.rawbufs), {v:var_vals[v] for v in getattr(ji.prg,"vars",[])}, jit=True)
    elif self.cnt == 1:
      CacheCollector.start(var_vals)
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")
      # get the inputs for replacement
      for j,ji in enumerate(self.jit_cache):
        for i,a in enumerate(ji.rawbufs):
          if a in [v[0] for v in input_rawbuffers.values()]:
            self.input_replace[(j,i)] = [(k, v[1].unbind(), v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)

    # clear the inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None
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

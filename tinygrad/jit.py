from __future__ import annotations
from typing import Callable, List, Tuple, Dict, cast, Union, Optional, TypeVar, Generic
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts
from tinygrad.ops import RawBuffer, Device, ASTRunner, BatchExecutor, JitItem
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable
from weakref import ref, WeakKeyDictionary

ReturnType = TypeVar('ReturnType')

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def reset(self):
    self.jit_fxn: Optional[BatchExecutor] = None
    self.cnt: int = 0
    self.ret: Optional[ReturnType] = None
    self.expected_vals: Optional[Tuple[Variable, ...]] = None
    self.expected_sts_dtype: Optional[Tuple[Tuple[ShapeTracker, DType], ...]] = None

  @property
  def jit_cache(self) -> List[JitItem]: return self.jit_fxn.jit_cache if self.jit_fxn else []
  @property
  def input_replace(self) -> Dict[Tuple[int, int], Union[int, str]]: return self.jit_fxn.input_replace if self.jit_fxn else {}

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> ReturnType:
    # all inputs are realized
    input_tensors: Dict[Union[int, str], Tensor] = {cast(Union[int, str], k):v.realize() for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    expected_sts_dtype = tuple([(v.lazydata.st.unbind(), v.dtype) for v in input_tensors.values()])

    # get rawbuffers
    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {k:cast(RawBuffer, v.lazydata.realized) for k,v in input_tensors.items()}
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.lazydata.st.var_vals for arg in input_tensors.values()] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])
    expected_vals = tuple(var_vals.keys())

    if self.cnt >= 2:
      assert self.expected_vals == expected_vals, "mismatch of var_vals"
      assert self.expected_sts_dtype == expected_sts_dtype, f"mismatch of sts, expected {self.expected_sts_dtype} got {expected_sts_dtype}"
      assert self.jit_fxn, "didn't get jitted?"
      self.jit_fxn(input_rawbuffers, var_vals, DEBUG>=2)
    elif self.cnt == 1:
      self.expected_vals, self.expected_sts_dtype = expected_vals, expected_sts_dtype

      CacheCollector.start(var_vals)
      self.ret = self.fxn(*args, **kwargs)
      jit_cache = CacheCollector.finish()
      assert len(jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      self.jit_fxn = Device[Device.DEFAULT].batch_executor(jit_cache, input_rawbuffers, var_vals)
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)

    self.cnt += 1
    return cast(ReturnType, self.ret)

class PlaceHolder:
  def __init__(self, buf:RawBuffer): self.size, self.dtype, self._device, self.ref, self.buftype, self.bufid = buf.size, buf.dtype, getattr(buf, '_device', None), ref(buf), type(buf), id(buf._buf)
  def to_tuple(self): return (self.size, self.dtype, self._device, self.buftype, self.bufid)
  def __hash__(self): return hash(self.to_tuple())
  def __eq__(self, x): return isinstance(x, PlaceHolder) and self.to_tuple() == x.to_tuple()
  def alloc_if_needed(self, buffer_cache: Dict[PlaceHolder, RawBuffer]) -> RawBuffer:
    ret = self.ref()
    if ret: return ret
    if self not in buffer_cache: buffer_cache[self] = self.buftype(self.size, self.dtype, **({'device':self._device} if self._device is not None else dict()))
    return buffer_cache[self]

class _CacheCollector:
  def __init__(self):
    self.cache: Optional[List[Tuple[ASTRunner, List[Union[RawBuffer, PlaceHolder]]]]] = None

  def start(self, var_vals:Optional[Dict[Variable, int]]=None):
    self.cache = []
    self.placeholders: WeakKeyDictionary[RawBuffer, PlaceHolder] = WeakKeyDictionary()
    self.var_vals = var_vals if var_vals is not None else {}

  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    for k,v in var_vals.items(): assert k in self.var_vals and self.var_vals[k] == v, f"var_vals {k} mismatch {v} != {self.var_vals.get(k)}"
    self.placeholders[rawbufs[0]] = PlaceHolder(rawbufs[0])    # NOTE: this is making an assumption that 0 is special
    self.cache.append((prg, [self.placeholders.get(x, x) if isinstance(x, RawBuffer) else x for x in rawbufs]))

  def finish(self) -> List[JitItem]:
    if self.cache is None: return []
    buffer_cache: Dict[PlaceHolder, RawBuffer] = {}
    saved_cache, self.cache = self.cache, None
    return [JitItem(prg, [x.alloc_if_needed(buffer_cache) if isinstance(x, PlaceHolder) else x for x in pl]) for prg, pl in saved_cache]
CacheCollector = _CacheCollector()

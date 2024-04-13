from __future__ import annotations
from typing import TypeVar, Generic, Callable, List, Tuple, Union, Dict, cast
import functools, itertools
from dataclasses import dataclass
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, GRAPH, BEAM, getenv
from tinygrad.device import Buffer, Runner
from tinygrad.dtype import DType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem
from tinygrad.nn.state import get_parameters
from weakref import ref, WeakKeyDictionary

def get_input_replace(jit_cache: List[ExecItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.rawbufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace

class PlaceHolder:
  placeholders: WeakKeyDictionary[Buffer, PlaceHolder] = WeakKeyDictionary()
  def __init__(self, buf:Buffer):
    self.size, self.dtype, self.device, self.ref, self.bufid, self.options = buf.size, buf.dtype, buf.device, ref(buf), id(buf._buf), buf.options
  def to_tuple(self): return (self.size, self.dtype, self.device, self.bufid, self.options)
  def __hash__(self): return hash(self.to_tuple())
  def __eq__(self, x): return isinstance(x, PlaceHolder) and self.to_tuple() == x.to_tuple()
  @staticmethod
  def create_if_needed(buf:Buffer) -> Union[PlaceHolder, Buffer]:
    if found:=PlaceHolder.placeholders.get(buf, None): return found
    if hasattr(buf, '_buf'): return buf
    PlaceHolder.placeholders[buf] = ret = PlaceHolder(buf.ensure_allocated())  # TODO: do I need to allocate here?
    return ret

  def alloc_if_needed(self, buffer_cache: Dict[PlaceHolder, Buffer]) -> Buffer:
    ret = self.ref()
    if ret: return ret
    if self not in buffer_cache: buffer_cache[self] = Buffer(self.device, self.size, self.dtype, options=self.options).allocate()
    return buffer_cache[self]

@dataclass(frozen=True)
class WeakExecItem:
  prg: Runner
  rawbufs: List[Union[PlaceHolder, Buffer]]

capturing: List[TinyJit] = []

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def add(self, ei:ExecItem):
    self._cc.append(WeakExecItem(ei.prg, [PlaceHolder.create_if_needed(buf) for buf in ei.rawbufs if buf is not None]))

  def reset(self):
    self._cc: List[WeakExecItem] = []
    self.jit_cache: List[ExecItem] = []
    self.input_replace: Dict[Tuple[int, int], int] = {}
    self.cnt: int = 0

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: List[Tuple[Union[int, str], Tensor]] = \
      [(cast(Union[int, str], k),v) for k,v in itertools.chain(enumerate(args), sorted(kwargs.items())) if v.__class__ is Tensor]
    Tensor.corealize([x[1] for x in input_tensors])
    lbs: List[LazyBuffer] = flatten([v.lazydata.lbs for _,v in input_tensors])
    expected_sts_var_dtype_device = [(*x.st.unbind(), x.dtype, x.device) for x in lbs]
    input_rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
    var_vals: Dict[Variable, int] = merge_dicts([x[1] for x in expected_sts_var_dtype_device])

    expected_names, expected_lbs = [x[0] for x in input_tensors], [(x[0], tuple(x[1].keys()), x[2], x[3]) for x in expected_sts_var_dtype_device]
    if self.cnt >= 2:
      # jit exec
      assert self.expected_names == expected_names and self.expected_lbs == expected_lbs, "args mismatch in JIT"
      for (j,i),input_idx in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_idx]
      if DEBUG >= 1: print(f"jit execs {len(self.jit_cache)} kernels")
      for ei in self.jit_cache: ei.run(var_vals, jit=True)
    elif self.cnt == 1:
      # jit capture
      self.expected_names: List[Union[int, str]] = expected_names
      self.expected_lbs: List[Tuple[ShapeTracker, Tuple[Variable, ...], DType, str]] = expected_lbs
      with Context(GRAPH=getenv("JITGRAPH", GRAPH.value), BEAM=getenv("JITBEAM", BEAM.value)):
        capturing.append(self)
        self.ret = self.fxn(*args, **kwargs)
        Tensor.corealize(get_parameters(self.ret))
        capturing.clear()
      buffer_cache: Dict[PlaceHolder, Buffer] = {}
      self.jit_cache = \
        [ExecItem(ei.prg, [x.alloc_if_needed(buffer_cache) if isinstance(x, PlaceHolder) else x for x in ei.rawbufs]) for ei in self._cc]
      del self._cc
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")
      self.input_replace = get_input_replace(self.jit_cache, input_rawbuffers)
    elif self.cnt == 0:
      # jit ignore
      self.ret = self.fxn(*args, **kwargs)
      Tensor.corealize(get_parameters(self.ret))

    # clear jit inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None

    self.cnt += 1
    return self.ret

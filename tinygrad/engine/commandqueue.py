# NOTE: this will replace jit.py, realize.py, and a lot of the boilerplate in each graph executor
from __future__ import annotations
from typing import List, Dict, TypeVar, Generic, Callable, Optional
import functools
from tinygrad.helpers import getenv, all_same
from tinygrad.ops import ScheduleItem, LoadOps
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import Variable
from tinygrad.device import Buffer
#from tinygrad.tensor import Tensor
#from tinygrad.nn.state import get_parameters

"""
ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  active_jit: Optional[TinyJit] = None
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def reset(self):
    self.cqs: List[CommandQueue] = []
    self.cnt: int = 0

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> ReturnType:
    # this setup for the jit will stay similar
    # but _CacheCollector and PlaceHolder will be removed, replaced with capturing CommandQueues called in the jit context

    # default run
    if self.cnt < 2:
      assert TinyJit.active_jit is None, "can't nest jits"
      if self.cnt == 1: TinyJit.active_jit = self
      self.ret = self.fxn(*args, **kwargs)
      Tensor.corealize(get_parameters(self.ret))
      TinyJit.active_jit = None
      self.cnt += 1
      return

    # all inputs (except const) are realized
    input_tensors: Dict[Union[int, str], Union[LazyBuffer, MultiLazyBuffer]] = { cast(Union[int, str], k):v.realize().lazydata for k,v in itertools.chain(enumerate(args), sorted(kwargs.items())) if v.__class__ is Tensor }  # noqa: E501
    expected_name_sts_dtype_device = tuple([(k, v.st.unbind()[0] if isinstance(v, LazyBuffer) else ShapeTracker.from_shape(v.shape), v.dtype, v.device) for k,v in input_tensors.items()]) #noqa: E501

    # get rawbuffers
    lbs: List[LazyBuffer] = [v for v in input_tensors.values() if isinstance(v, LazyBuffer)] + flatten([mlb.lbs for mlb in input_tensors.values() if isinstance(mlb, MultiLazyBuffer)]) #noqa: E501
    input_rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
    assert len(set(input_rawbuffers)) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.st.var_vals for arg in lbs] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])  # noqa: E501
    expected_vals = tuple(var_vals.keys())
"""

# this will interface with HWCommandQueue to replace Graph
logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
class CommandQueue:
  def __init__(self, schedule:List[ScheduleItem], outs:List[LazyBuffer]):
    # loop through the schedule, find (real) inputs, add assign outputs, and split into different devices
    schedule_per_device = {}
    while len(schedule):
      si = schedule.pop(0)
      assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or (si.ast[0].op in {LoadOps.COPY, LoadOps.WAIT} and len(si.outputs) == 1)
      if (device:=si.outputs[0].device) not in schedule_per_device: schedule_per_device[device] = []
      schedule_per_device[device].append(si)

    # do memory planning and allocations across the devices
    for k,v in schedule_per_device.items():
      print("*****", k)
      for si in v:
        print(si)




    #while len(schedule):
    #  si = schedule.pop(0)
    #  if logops and si.ast[0].op not in LoadOps and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")


    pass
    # loop through the ScheduleItem
    # NOTE: that list will be emptied
    # all LazyBuffers are either alloced or not alloced
    # non alloced LazyBuffers are free for memory scheduling, in addition to ones not in outs and not assigned to
    # if a buffer is assigned to or is in outs, it's guarenteed to be valid after exiting this function

    # CommandQueue will contain 1 or 2 queues per device, depending on if we split copy
    # Pairs of devices either support fast syncing or they don't

    # inputs will be extracted from the schedule, tensors that are read from but not written to

  def replace_inputs_and_var_vals(self, input_rawbuffers:List[Buffer], var_vals: Dict[Variable, int]):
    # if it's the same graph run on different stuff
    pass

  def __call__(self):
    # this should be callable if we discover a full lazy graph has the same hash
    #if TinyJit.active_jit is not None: TinyJit.active_jit.cqs.append(self)

    pass




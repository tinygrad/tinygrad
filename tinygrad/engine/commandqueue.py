# NOTE: this will replace jit.py, realize.py, and a lot of the boilerplate in each graph executor
from __future__ import annotations
from typing import List, Dict, TypeVar, Generic, Callable, Optional, Union, Tuple, DefaultDict, cast
from collections import defaultdict
import functools
from tinygrad.helpers import getenv, all_same
from tinygrad.features.graph import realized_lazybuffer
from tinygrad.ops import ScheduleItem, LoadOps, BufferOps, LazyOp, GlobalCounters
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import Variable
from tinygrad.device import Buffer, JITRunner, Device, BufferXfer, BufferRead, BufferCopy, update_stats
from dataclasses import dataclass
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG
from tinygrad.engine.realize import lower_schedule_item

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

@dataclass(frozen=True)
class SyncItem:
  sid: int

@dataclass(frozen=True)
class WaitItem:
  sid: int

class FutureJITRunner:
  def __init__(self, device:str, ast_tuple:Tuple[LazyOp, ...], optional_input_device:Optional[str]=None):
    self.device, self.ast_tuple = device, ast_tuple
    if self.ast_tuple[0].op is LoadOps.COPY:
      assert optional_input_device is not None
      if hasattr(Device[device].allocator, 'transfer') and device.split(":")[0] == optional_input_device.split(":")[0]: self.runner = BufferXfer()
      elif optional_input_device.startswith("DISK"): self.runner = BufferRead()
      else: self.runner = BufferCopy()
    else:
      # TODO: add parallel compiler dispatch here
      self.runner = None
  def __call__(self, *args, **kwargs):
    if self.runner is None: self.runner = Device[self.device].get_runner(*self.ast_tuple)
    return self.runner(*args, **kwargs)

# this will interface with HWCommandQueue to replace Graph
#logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
class CommandQueue:
  def __init__(self, schedule:List[ScheduleItem], outs:List[LazyBuffer]):
    sync_sid = 0

    # loop through the schedule, find (real) inputs, add assign outputs, and split into different devices
    self.q: DefaultDict[str, Union[ScheduleItem, SyncItem, WaitItem]] = defaultdict(list)
    while len(schedule):
      si = schedule.pop(0)
      assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or (si.ast[0].op is LoadOps.COPY and len(si.outputs) == 1)
      device = si.outputs[0].device
      if si.ast[0].op is LoadOps.COPY and si.inputs[0].device not in {"EXT", "DISK"}:
        # add sync between the devices
        if not len(self.q[si.inputs[0].device]) or not isinstance(self.q[si.inputs[0].device][-1], SyncItem):
          self.q[si.inputs[0].device].append(SyncItem(sync_sid))
          sync_sid += 1
        self.q[device].append(WaitItem(self.q[si.inputs[0].device][-1].sid))
      self.q[device].append(si)

  def __call__(self):
    # this should be callable if we discover a full lazy graph has the same hash
    active_queues = list(self.q.keys())
    waiting_queues = defaultdict(list)
    seen_sids = set()
    while len(active_queues):
      device = active_queues.pop(0)
      if not len(self.q[device]): continue
      si = self.q[device].pop(0)
      if isinstance(si, SyncItem):
        Device[device].synchronize()
        et = cpu_time_execution(Device[device].synchronize, enable=DEBUG>=2)
        update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=device)
        if si.sid in waiting_queues:
          active_queues += waiting_queues[si.sid]
          waiting_queues[si.sid].clear()
        seen_sids.add(si.sid)
      elif isinstance(si, WaitItem):
        if si.sid not in seen_sids:
          waiting_queues[si.sid].append(device)
          continue
      else:
        prg = lower_schedule_item(si)

        for out in si.outputs:
          # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
          if out.size > 0:
            if out.op is LoadOps.ASSIGN and out.srcs[1].base.realized is not None:
              # if the buffer isn't realized, it might be a const or something. this is fine
              out.realized = out.srcs[1].base.realized
            else:
              out.realized = Buffer(out.device, out.size, out.dtype, "PLACEHOLDER" if getattr(prg, "skip_allocation", False) else None)
            del out.srcs

        # run the function (put it in JIT)
        real_buffers = [x.realized for x in si.outputs+si.inputs if x.size != 0]
        assert all(x is not None for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
        if prg: prg.exec(cast(List[Buffer], real_buffers), si.var_vals)
        elif (out:=si.outputs[0]).size > 0: update_stats(colored(f"empty {out.st.size:10d} {out.dtype}", "yellow"), 0, 0, {}, None, 1, device=out.device)
        if GRAPH:
          for out in si.outputs: realized_lazybuffer(out, GlobalCounters.kernel_count)

      active_queues.append(device)







    #if TinyJit.active_jit is not None: TinyJit.active_jit.cqs.append(self)
    #for k,v in self.q.items():
    #  print("***", k)
    #  for si in v:
    #    print(str(si)[0:80])




from typing import List, Dict, Optional, Set
from tinygrad.ops import LoadOps, ScheduleItem, LazyOp, BufferOps, MemBuffer, ConstBuffer
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import print_tree, log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import prod, GlobalCounters
from tinygrad.shape.symbolic import Variable
from tinygrad.lazy import LazyBuffer

# *** schedule running ***

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.COPY, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"  # noqa: E501
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.COPY: return BufferCopy
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  while len(schedule):
    si = schedule.pop(0)
    assert all(x.realized for x in si.inputs), f"can't run schedule, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"

    # get the program
    prg = lower_schedule_item(si)

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    assert si.out._base is None, "no setting of non-base LazyBuffers"
    si.out._realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

    # run the function (put it in JIT)
    if prg: prg.exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)
    realized_lazybuffer(si.out, GlobalCounters.kernel_count)

# *** schedule creation ***

def _recursive_get_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], first=True) -> LazyOp:
  log_lazybuffer(buf)
  if buf.base.op == LoadOps.CONST:
    # const is never a buffer
    return LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.base.arg), buf.dtype, buf.st.simplify().unbind()))
  if buf.op == LoadOps.CONTIGUOUS and first:
    # include one contiguous
    return _recursive_get_lazyop(buf.srcs[0], inputs, True)
  if buf.op == LoadOps.COPY and first:
    inputs.append(buf.srcs[0].base)
    return LazyOp(LoadOps.COPY, (), buf.srcs[0].base)
  if buf.op == LoadOps.CUSTOM and first:
    inputs += buf.srcs
    return LazyOp(LoadOps.CUSTOM, (), buf.arg)
  if buf.op == LoadOps.EMPTY and first:
    return LazyOp(LoadOps.EMPTY)
  if buf.realized or buf != buf.base or buf.base.op in LoadOps or (len(buf.base.children) > 1 and not first):
    # have to do a load
    if buf.base not in inputs: inputs.append(buf.base)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf.base)+1, buf.dtype, buf.st.simplify().unbind()))
  # convert to a lazyop
  assert buf.op is not None
  return LazyOp(buf.op, tuple(_recursive_get_lazyop(x, inputs, False) for x in buf.srcs), buf.arg)

def create_schedule(out:LazyBuffer, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()
  if out in seen or out.realized or out.is_unrealized_const(): return []
  seen.add(out)
  log_lazybuffer(out)
  if out.base is not out: return create_schedule(out.base, seen)

  inputs: List[LazyBuffer] = []
  op = _recursive_get_lazyop(out, inputs)
  if op.op not in LoadOps: op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, out.st))
  ret: List[ScheduleItem] = []
  for x in inputs:
    assert x.base == x, f"all inputs must be base, {x} isn't"
    ret += create_schedule(x, seen)

  # check if we can reuse the output buffer
  # if it's aliased, don't use it
  if out.output_buffer is not None:
    for i,a in enumerate(inputs):
      # TODO: if this is contiguous it's fine
      if a.realized == out.output_buffer:
        if any(not x.arg.st.contiguous for x in op.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
          out.output_buffer = None
          break

  #from tinygrad.graph import print_tree
  #print_tree(op)
  return ret + [ScheduleItem(op, out, tuple(inputs), {})]

from typing import List, Dict, Optional, Set
from tinygrad.ops import LoadOps, ScheduleItem, LazyOp, BufferOps, MemBuffer, ConstBuffer, vars_from_ast, ReduceOps, BinaryOps, UnaryOps, TernaryOps
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import print_tree, log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import prod, GlobalCounters, merge_dicts, DEBUG
from tinygrad.shape.symbolic import Variable
from tinygrad.shape.shapetracker import ShapeTracker
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

def _find_reducebuf(buf:LazyBuffer) -> Optional[LazyBuffer]:
  if buf.base != buf: return _find_reducebuf(buf.base) if buf.st.contiguous and buf.st.size() == buf.base.st.size() else None
  assert buf.base == buf
  if buf.op in ReduceOps: return buf
  if isinstance(buf.op, (UnaryOps, BinaryOps, TernaryOps)):
    for x in buf.srcs:
      if (rb := _find_reducebuf(x)): return rb
  return None

def _recursive_get_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], st:ShapeTracker, reduce:Optional[LazyBuffer]=None, first=True) -> LazyOp:
  log_lazybuffer(buf)
  if buf.base.op == LoadOps.CONST:
    # const is never a buffer
    return LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.base.arg), buf.dtype, (buf.st+st).simplify().unbind()))
  if buf.op == LoadOps.CONTIGUOUS and first:
    # include one contiguous
    return _recursive_get_lazyop(buf.srcs[0], inputs, st, reduce, True)
  if buf.op == LoadOps.COPY and first:
    inputs.append(buf.srcs[0].base)
    return LazyOp(LoadOps.COPY, (), buf.srcs[0].base)
  if buf.op == LoadOps.CUSTOM and first:
    inputs += buf.srcs
    return LazyOp(LoadOps.CUSTOM, (), buf.arg)
  if buf.op == LoadOps.EMPTY and first:
    return LazyOp(LoadOps.EMPTY)

  # we can merge this as a LazyOp
  if not buf.realized and buf.base.op not in LoadOps and buf.base.op not in BufferOps: # and (len(buf.base.children) == 1 or first): # and buf == buf.base:
    if buf.base == reduce:
      st = ShapeTracker.from_shape(buf.base.srcs[0].shape)
      return LazyOp(buf.base.op, tuple(_recursive_get_lazyop(x, inputs, st, reduce, False) for x in buf.base.srcs), buf.base.arg)
    elif isinstance(buf.base.op, (UnaryOps, BinaryOps, TernaryOps)):
      st = (buf.st+st).simplify()
      return LazyOp(buf.base.op, tuple(_recursive_get_lazyop(x, inputs, st, reduce, False) for x in buf.base.srcs), buf.base.arg)

  # have to do a load
  if buf.base not in inputs: inputs.append(buf.base)
  return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf.base)+1, buf.dtype, (buf.st+st).simplify().unbind()))

def _create_schedule(out:LazyBuffer, seen:Set[LazyBuffer]) -> List[ScheduleItem]:
  if out in seen or out.realized or out.is_unrealized_const(): return []
  seen.add(out)
  log_lazybuffer(out)
  if out.base is not out: return _create_schedule(out.base, seen)
  assert out.base == out and out.op is not None
  reduce = _find_reducebuf(out)

  inputs: List[LazyBuffer] = []
  st = ShapeTracker.from_shape(reduce.shape if reduce else out.shape)
  op = _recursive_get_lazyop(out, inputs, st, reduce)
  if op.op not in LoadOps: op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, st.simplify().unbind()))
  ret: List[ScheduleItem] = []
  for x in inputs:
    assert x.base == x, f"all inputs must be base, {x} isn't"
    ret += _create_schedule(x, seen)

  # check if we can reuse the output buffer
  # if it's aliased, don't use it
  if out.output_buffer is not None:
    for i,a in enumerate(inputs):
      # TODO: if this is contiguous it's fine
      if a.realized == out.output_buffer:
        if any(not x.arg.st.contiguous for x in op.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
          out.output_buffer = None
          break

  if DEBUG >= 5:
    from tinygrad.graph import print_tree
    print_tree(op)

  var_vals = merge_dicts([out.st.var_vals] + [buf.st.var_vals for buf in inputs])
  return ret + [ScheduleItem(op, out, tuple(inputs), {k:var_vals[k] for k in vars_from_ast(op)})]

def create_schedule(out:LazyBuffer, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()
  log_lazybuffer(out, scheduled=True)
  return _create_schedule(out, seen)

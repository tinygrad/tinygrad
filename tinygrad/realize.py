from typing import List, Dict, Optional, Set, Tuple, Any
from tinygrad.ops import LoadOps, ScheduleItem, LazyOp, BufferOps, MemBuffer, ConstBuffer, vars_from_ast, get_lazyop_info
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import log_schedule_item, print_tree, log_lazybuffer
from tinygrad.helpers import prod, dedup, merge_dicts, ImageDType, DEBUG
from tinygrad.shape.symbolic import Variable
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.lazy import LazyBuffer

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
    #if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), f"can't run schedule, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"

    # get the program
    prg = lower_schedule_item(si)

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    assert si.out._base is None, "no setting of non-base LazyBuffers"
    si.out._realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

    # run the function (put it in JIT)
    if prg: prg.exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)

def _recursive_get_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], first=True) -> LazyOp:
  log_lazybuffer(buf)
  if buf.base.op == LoadOps.CONST:
    # const is never a buffer
    op = LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.base.arg), buf.dtype, buf.st.simplify().unbind()))
  elif buf.op == LoadOps.CONTIGUOUS and first:
    # include one contiguous
    return _recursive_get_lazyop(buf.srcs[0], inputs, True)
  elif buf.op == LoadOps.COPY and first:
    if buf.srcs[0].base.is_unrealized_const():
      # CONSTs don't actually have to be copied
      op = LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.srcs[0].base.arg), buf.dtype, buf.st.simplify().unbind()))
    else:
      inputs.append(buf.srcs[0].base)
      return LazyOp(LoadOps.COPY, (), buf.srcs[0].base)
  elif buf.realized or buf != buf.base or buf.base.op in LoadOps:
    # have to do a load
    if buf.base not in inputs: inputs.append(buf.base)
    op = LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf.base)+1, buf.dtype, buf.st.simplify().unbind()))
  else:
    op = LazyOp(buf.op, tuple(_recursive_get_lazyop(x, inputs, False) for x in buf.srcs), buf.arg)
  return LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, buf.dtype, buf.st)) if first else op

#def _recursive_get_lazyop(op:LazyOp, srcs:Tuple[LazyBuffer], arg:Any) -> LazyOp:
#  if src.realized: return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(x.base)+1, x.dtype, st))
#  if not src.realized:
#  return LazyOp(op, )

def _schedule_one(out:LazyBuffer, seen:Optional[Set[LazyBuffer]]) -> List[ScheduleItem]:
  if out in seen or out.realized or out.is_unrealized_const(): return []
  seen.add(out)
  if out.base is not out:
    log_lazybuffer(out)
    return _schedule_one(out.base, seen)

  inputs: List[LazyBuffer] = []
  op = _recursive_get_lazyop(out, inputs)
  ret: List[ScheduleItem] = []
  for x in inputs:
    assert x.base == x, f"all inputs must be base, {x} isn't"
    ret += _schedule_one(x, seen)

  #from tinygrad.graph import print_tree
  #print_tree(op)
  return ret + [ScheduleItem(op, out, tuple(inputs), {})]


  print(op)


  # schedule the past
  ret:List[ScheduleItem] = []
  inputs: List[LazyBuffer] = dedup([x.base for x in out.src if not x.is_unrealized_const()])
  for x in inputs: ret += _schedule_one(x, seen)

  # replace srcs
  srcs_op: List[LazyOp] = []
  for x in out.src:
    st = x.st.simplify().unbind()
    if x.base in inputs:
      srcs_op.append(LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(x.base)+1, x.dtype, st)))
    elif not x.realized and x.base.op.op == LoadOps.CONST:
      srcs_op.append(LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.base.op.arg), x.dtype, st)))
    else:
      raise NotImplementedError(f"not handled {x}")
  op = LazyOp(out.op, tuple(srcs_op), out.arg)

  # check if we can reuse the output buffer
  # if it's aliased, don't use it
  # TODO: this is pretty wrong actually, who knows where else this buffer is used?
  # TODO: what if an assign is required? this silently is wrong
  # NOTE: this has been moved to schedule, as this is only an issue if buffers are already realized
  """
  if out.output_buffer is not None:
    for i,a in enumerate(inputs):
      # TODO: if this is contiguous it's fine
      if a.realized == out.output_buffer:
        if any(not x.arg.st.contiguous for x in op.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
          out.output_buffer = None
          break
  """

  if op.op not in LoadOps:
    # add the store
    info = get_lazyop_info(op)
    assert info.dtype == out.dtype or isinstance(out.dtype, ImageDType), f"dtype mismatch {info.dtype=} != {out.dtype=}"

    #if isinstance(self.dtype, ImageDType) and (prod(self.shape) != prod(self.dtype.shape) or not any(self.shape[x]%4 == 0 for x in self.st.unit_stride_axes())):
    #  if DEBUG >= 3: print(f"forcing image {self.dtype} to float32")
    #  self.dtype = dtypes.float32  # NOTE; this is what makes the dtype above not match
    #  op = LazyOp(UnaryOps.CAST, (op, ), (dtypes.float32, False))

    # TODO: why doesn't this match?
    #assert info.shape == self.shape, f"shape mismatch {info.shape=} != {self.shape=}"
    op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, ShapeTracker.from_shape(info.shape)))
  else:
    # check loadop validity of bufferops
    for i,s in enumerate(op.src): assert isinstance(s, LazyOp) and s.op == BufferOps.LOAD and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"

  # return scheduleitem
  var_vals = merge_dicts([out.st.var_vals] + [buf.st.var_vals for buf in inputs])
  ret.append(ScheduleItem(op, out, tuple(inputs), {k:var_vals[k] for k in vars_from_ast(op)}))
  return ret

def create_schedule(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  ret = []
  seen = set()
  for out in outs: ret += _schedule_one(out, seen)
  return ret

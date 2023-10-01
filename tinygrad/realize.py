from typing import List, Tuple, cast, Dict, Callable, Optional, Union
import numpy as np
import sys
from weakref import ref
from collections import defaultdict
from tinygrad.ops import ASTRunner, LazyOp, BufferOps, LoadOps, Device, Compiled
from tinygrad.graph import log_schedule_item
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, getenv, ImageDType

from tinygrad.runtime.lib import RawBuffer, RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_disk import RawDiskBuffer

P2P = getenv("P2P", 0)

# *** realization (unrelated to lazy) ***

def __get_output_buffer(ast, output, inputs) -> Optional[RawBuffer]:
  for i,a in enumerate(inputs):
    # TODO: if this is contiguous it's fine
    if a == output.realized and any(not x.arg.st.contiguous for x in ast.get_lazyops() if x.op == BufferOps.MEM and x.arg.idx == i+1):
      return None
  return output.output_buffer

def run_prebuilt(schedule:List[Tuple[LazyOp, LazyBuffer, Tuple[LazyBuffer, ...]]]):
  prebuilt_info: List[Tuple[LazyBuffer, Union[ASTRunner, LazyOp], Optional[List[LazyBuffer]]]] = []
  while len(schedule):
    op,out,buffers = schedule.pop(0)
    log_schedule_item(op, out, buffers)
    if DEBUG >= 3:
      from extra.utils import print_tree   # type: ignore
      print_tree(op)
    # TODO: why can't we delete these LoadOps?
    if op.op in LoadOps: prebuilt_info.append((out, op, []))
    else:
      if isinstance(Compiled, Device[out.device]):
        if out.output_buffer is not None: out.realized = __get_output_buffer(op, out, [x.realized for x in buffers])
        op = Device[out.device].compile_ast(op, args_info=[(x.shape, x.dtype) for x in ([out]+list(buffers))], var_vals=out.var_vals, **out._device_extra_args())
      prebuilt_info.append((out, op, buffers))
      del out.op
      for v in out.views: del v.op
  return prebuilt_info

def allocate_buffers(prebuilt_info: List[Tuple[LazyBuffer, Union[ASTRunner, LazyOp], Optional[List[LazyBuffer]]]]):
  last_use, cnt = {}, defaultdict(int)
  for j,(out,prog,buffers) in enumerate(prebuilt_info):
    for buf in buffers: last_use[ref(buf)], cnt[ref(buf)] = j, cnt[ref(buf)] + 1

  query_list = []
  for j,(out,prog,buffers) in enumerate(prebuilt_info):
    if isinstance(prog, LazyOp) or out.realized is not None: continue
    if sys.getrefcount(out)-cnt[ref(out)] == 3:
      query_list.append((prod((s if isinstance(s, int) else s.max for s in out.shape))*out.dtype.itemsize, j, last_use[ref(out)], ref(out)))

  def _can_use_rawbuf(out:LazyBuffer, with_buf:RawBuffer):
    out_size = prod((s if isinstance(s, int) else s.max for s in out.shape))
    return (out_size*out.dtype.itemsize<=with_buf.size*with_buf.dtype.itemsize if not isinstance(out.dtype, ImageDType) and not isinstance(with_buf.dtype, ImageDType) else out_size==with_buf.size and buf.dtype==with_buf.dtype and buf.dtype.shape==with_buf.dtype.shape)

  # The query list contains a query for every placeholder that should be replaced with the actual rawbuffer. Queries are served from the largest to the smallest.
  # For each query, find any rawbuffer that is free within the query timeframe or allocate a new one.
  rawbuf_pool: Dict[str, List[Tuple[RawBuffer, List[Tuple[int, int]]]]] = defaultdict(list)
  query_list = sorted(query_list, key=lambda x: x[0], reverse=True)
  for _, start, end, buf in query_list:
    pool_idx = next((i for i,(with_buf, usages) in enumerate(rawbuf_pool[buf().device]) if _can_use_rawbuf(buf(), with_buf) and all(en < start or end < st for st, en in usages)), -1)
    if pool_idx == -1:
      new_buf = Device[buf().device].buffer(prod((s if isinstance(s, int) else s.max for s in buf().shape)), buf().dtype, **out._device_extra_args())
      rawbuf_pool[buf().device].insert(0, (new_buf, []))
      pool_idx = 0
    buf().realized = rawbuf_pool[buf().device][pool_idx][0]
    rawbuf_pool[buf().device][pool_idx][1].append((start, end))

def run_schedule(schedule:List[Tuple[LazyOp, LazyBuffer, Tuple[LazyBuffer, ...]]]):
  prebuilt_info = run_prebuilt(schedule)
  allocate_buffers(prebuilt_info)

  while len(prebuilt_info):
    out,prog,buffers = prebuilt_info.pop(0)
    if isinstance(prog, LazyOp):
      if prog.op in LoadOps: LOAD_OPS_DISPATCHER[cast(LoadOps, prog.op)](out)
      else: out.realized = Device[out.device].exec_ast(prog, output=out, inputs=[x.realized for x in buffers], var_vals=out.var_vals, **out._device_extra_args())
    else:
      if out.realized is None: out.realized = Device[out.device].buffer(prod((s if isinstance(s, int) else s.max for s in out.shape)), out.dtype, **out._device_extra_args())
      assert out.realized and isinstance(out.realized, Device[out.device].buffer), f"device mismatch on realized got {type(out.realized)} expected {out.device}"
      prog.exec([out.realized]+[x.realized for x in buffers], var_vals=out.var_vals)

def _realize_contiguous(buffer: LazyBuffer) -> None:
  # this is just a copy now, if it's not a copy schedule will handle it
  src = cast(LazyBuffer, buffer.op.src[0])
  buffer.realized = src.realized
  assert buffer.dtype == src.dtype, f"contiguous dtype mismatch, expecting {buffer.dtype}, got {src.dtype}"

def _realize_custom(buffer: LazyBuffer) -> None:
  # this needs to immediately realize
  buffer.realized = buffer.op.arg(buffer, *[x.realize() for x in buffer.op.src])

def _realize_from(buffer: LazyBuffer) -> None:
  rawbuf = buffer.op.src[0].realize()
  assert rawbuf.realized, "realize failed?"
  if DEBUG >= 3: print(f"*** copy {buffer.device} <- {rawbuf.device} size {rawbuf.realized.size} dtype {rawbuf.realized.dtype}")
  # TODO: make this generic
  if isinstance(rawbuf.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    rawbuf.prepare_transfer().readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(rawbuf.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and P2P >= 1:
    buffer.realized = cast(RawBufferTransfer, Device[buffer.device].buffer).transfer(rawbuf.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(rawbuf.toCPU(), **buffer._device_extra_args())

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=buffer.shape, dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args()) # type: ignore

def _realize_const(buffer: LazyBuffer) -> None:
  buffer.realized = Device[buffer.device].buffer.fromCPU(np.array(buffer.op.arg, dtype=buffer.dtype.np), **buffer._device_extra_args())

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.CONTIGUOUS: _realize_contiguous,
  LoadOps.CUSTOM: _realize_custom,
  LoadOps.FROM: _realize_from,
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.CONST: _realize_const,
}

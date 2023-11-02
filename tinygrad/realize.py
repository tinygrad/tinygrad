from typing import List, cast, Dict, Callable, Tuple
import functools
import numpy as np
from tinygrad.ops import ScheduleItem, LazyOp, LoadOps, Device, BufferOps, Interpreted, Compiled
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, getenv, IMAGE

from tinygrad.features.image import fix_schedule_for_images

# *** compile ast to python function ***

@functools.lru_cache(None)   # this is the method cache for interpreted
def interpret_ast(device:Interpreted, ast:LazyOp) -> Callable:
  raise NotImplementedError("write this to return a function that applies the AST")

@functools.lru_cache(None)   # this is the new method cache
def compile_ast(device:Compiled, ast:LazyOp) -> Callable:
  # get linearizer
  from tinygrad.codegen.linearizer import Linearizer
  lin = Linearizer(ast, device.linearizer_opts)

  # TODO: search optimizations
  lin.hand_coded_optimizations()

  # generate uops from the AST
  lin.linearize()

  # render the source code
  # TODO: move global_size and local_size to runtime_args
  src, runtime_args = device.renderer(lin.function_name, lin.uops)
  if DEBUG >= 4: print(src)

  # compile the source code. TODO: pass in device identifier
  lib: bytes = device.compiler(src)

  # get the function
  return device.runtime(lin.function_name, lib, lin.global_size, lin.local_size, **runtime_args)

# *** main schedule runner ***

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  # HACK: images can be not usable due to shape
  if IMAGE >= 2: schedule = fix_schedule_for_images(schedule)

  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"
    if DEBUG >= 3: print_tree(si.ast)
    device = Device[si.out.device]
    if si.ast.op in LoadOps:
      # confirm the LoadOps are contiguous and in order
      for i,s in enumerate(si.ast.src): assert isinstance(s, LazyOp) and s.op == BufferOps.MEM and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
      LOAD_OPS_DISPATCHER[cast(LoadOps, si.ast.op)](si.out, *si.inputs)
    elif isinstance(device, Interpreted):
      # compile AST to interpreted function for speed
      si.out.realized = interpret_ast(device, si.ast)(*[x.realized for x in si.inputs])
    else:
      # allocate empty output buffer
      # TODO: memory scheduling
      # TODO: before or after compile?
      si.out.realized = device.buffer(si.out.st.size(), si.out.dtype)

      # run the program
      compile_ast(device, si.ast)(si.out.realized, *[x.realized for x in si.inputs])

      # TODO: do the rest of the ASTRunner stuff here, print, GlobalCounters etc...
    del si.out.op
    for v in si.out.views: del v.op
    assert si.out.realized and isinstance(si.out.realized, Device[si.out.device].buffer), f"device mismatch on realized got {type(si.out.realized)} expected {si.out.device}"
    assert si.out.realized.dtype == si.out.dtype, "realized dtype is incorrect"

# *** zero op LoadOps ***

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***     empty {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***      rand {buffer.device}    seed {buffer.op.arg:<10d}  shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=prod(buffer.shape), dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args())

# *** one op LoadOps ***

from tinygrad.runtime.lib import RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_disk import RawDiskBuffer
def _realize_from(buffer: LazyBuffer, src: LazyBuffer) -> None:
  assert src.realized.size == buffer.st.size(), f"size mismatch on FROM {src.realized.size} != {buffer.st.size()}"
  assert src.st.contiguous and buffer.st.contiguous, "all must be contiguous for from"
  if DEBUG >= 2: print(f"***      copy {buffer.device} <- {src.device} size {src.realized.size:<16d} shape {str(buffer.shape):23s} dtype {src.realized.dtype}")
  # TODO: make this generic
  if isinstance(src.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    src.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(src.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and getenv("P2P", 0) >= 1:
    buffer.realized = cast(RawBufferTransfer, Device[buffer.device].buffer).transfer(src.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())
  else:
    # TODO: schedule this as FROM to go to CPU, and a FROM to go to device
    buffer.realized = Device[buffer.device].buffer.fromCPU(src.realized.toCPU(), **buffer._device_extra_args())

# *** n op LoadOps ***

def _realize_custom(buffer: LazyBuffer, *inputs: LazyBuffer) -> None:
  if DEBUG >= 2: print(f"***    custom {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = buffer.op.arg(buffer, *inputs)

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.FROM: _realize_from,
  LoadOps.CUSTOM: _realize_custom,
}

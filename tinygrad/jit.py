from typing import TypeVar, Callable, Generic, overload
from tinygrad.function import _function
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG, ALLOW_DEVICE_USAGE
from tinygrad.engine.jit import CapturedJit, _prepare_jit_inputs, get_input_replace, JitError
from tinygrad.engine.realize import capturing, ExecItem

ReturnType = TypeVar('ReturnType')
class _jit(_function[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool=False):
    super().__init__(fxn, precompile=precompile)
    self._prev_key:bytes|None = None
    self._captured:CapturedJit|None = None
    self._cnt:int = 0

  @property
  def jit_cache(self) -> list[ExecItem]: return self._captured._jit_cache if self._captured is not None else []

  def reset(self):
    self._prev_key = None
    self._captured = None
    self._cnt = 0

  def __call__(self, *args, **kwargs) -> ReturnType:
    # when nested inside another @function/@jit, just act as @function (no capture/replay)
    if not ALLOW_DEVICE_USAGE: return super().__call__(*args, **kwargs)

    if self._captured is not None:
      input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
      if self._captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self._captured.expected_names=} != {names}")
      if self._captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self._captured.expected_input_info=} != {expected_input_info=}")
      return self._captured(input_buffers, var_vals)

    # build CALL UOp via @function, then realize
    ret = super().__call__(*args, **kwargs)
    call_key = ret.uop.src[0].key
    if call_key == self._prev_key:
      self._do_capture(ret, args, kwargs)
    else:
      self._prev_key = call_key
      ret.realize()

    self._cnt += 1
    if self._cnt >= 5 and self._captured is None and DEBUG >= 1: print("WARNING: jit not captured after 5 calls")
    return ret

  def _do_capture(self, ret:Tensor, args, kwargs):
    input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    jit_cache:list[ExecItem] = []

    class _capture_ctx:
      def add(_, ei:ExecItem): jit_cache.append(ei)

    capturing.append(_capture_ctx())
    try: ret.realize()
    finally: capturing.clear()

    if not jit_cache: raise JitError("didn't JIT anything!")
    if DEBUG >= 1: print(f"JIT captured {len(jit_cache)} kernels with {len(input_buffers)} inputs")

    # track input views
    extra_view_inputs = []
    for item in jit_cache:
      for b in item.bufs:
        if b is not None and b._base is not None and b._base in input_buffers:
          input_buffers.append(b)
          extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

    input_replace = get_input_replace(jit_cache, input_buffers)
    self._captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, expected_input_info)

@overload
def jit(fxn:Callable[..., ReturnType], *, precompile:bool=False) -> _jit[ReturnType]: ...
@overload
def jit(fxn:None=None, *, precompile:bool=False) -> Callable[[Callable[..., ReturnType]], _jit[ReturnType]]: ...
def jit(fxn=None, *, precompile:bool=False):
  if fxn is None: return lambda f: _jit(f, precompile=precompile)
  return _jit(fxn, precompile=precompile)

TinyJit = _jit  # noqa: F841  # NEW_JIT=1 swaps engine.jit.TinyJit with this

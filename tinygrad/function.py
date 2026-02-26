import functools
from typing import Generic, TypeVar, Callable, cast
from tinygrad.helpers import Context, dedup, getenv
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import Tensor

def _srcs(u:UOp) -> tuple[UOp, ...]:
  """Get sources of a UOp, skipping src[0] of CALL nodes (other functions' bodies with their own PARAMs)."""
  return u.src[1:] if u.op is Ops.CALL else u.src

ReturnType = TypeVar('ReturnType')
class function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_uops: list[UOp] = [(t.uop if isinstance(t, Tensor) else t)
                             for name,t in list(enumerate(args))+sorted(kwargs.items()) if isinstance(t, (Tensor, UOp))]

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup(input_uops)

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(call_uops):
      # TODO: this can be better
      if x.op is Ops.BIND: subs[x] = UOp.param(i, x.dtype, x._shape, x._device, x._min_max, x.src[0].arg[0])
      else: subs[x] = UOp.param(i, x.dtype, x._shape, x._device)
    uret = ret.uop.substitute(subs)

    # the BUFFERs that are left are the implicit inputs
    subs = {}
    for x in uret.toposort():
      if x.op is Ops.BUFFER:
        subs[x] = UOp.param(len(call_uops), x.dtype, x._shape, x._device)
        call_uops.append(x)
    uret = uret.substitute(subs)

    pbuffer = UOp.param(len(call_uops), uret.dtype, uret._shape, uret._device)
    assigned = pbuffer.assign(uret).sink()

    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__

    buffer = UOp.new_buffer(pbuffer.device, pbuffer.size, pbuffer.dtype).reshape(uret.shape)
    call = assigned.call(*call_uops, buffer, name=name)
    ret = buffer.after(call)

    return cast(ReturnType, Tensor(ret, device=ret.device))


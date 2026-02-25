import functools
from typing import Generic, TypeVar, Callable, cast
from dataclasses import dataclass, field
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite
from tinygrad.tensor import Tensor

@dataclass
class _ImplicitBufCtx:
  offset: int
  bufs: list[UOp] = field(default_factory=list)

def _replace_implicit_buffer(ctx:_ImplicitBufCtx, b:UOp):
  if b not in ctx.bufs: ctx.bufs.append(b)
  return UOp.param(ctx.offset + ctx.bufs.index(b), b.dtype, b.shape, b._device)

pm_implicit = PatternMatcher([(UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), _replace_implicit_buffer)])

ReturnType = TypeVar('ReturnType')
class function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_uops: list[UOp] = [(t.uop if isinstance(t, Tensor) else t).multibase
                             for name,t in list(enumerate(args))+sorted(kwargs.items()) if isinstance(t, (Tensor, UOp))]

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=0):
      ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"

    # replace the known inputs with params
    subs = {}
    for i,x in enumerate(input_uops):
      # TODO: this can be better
      if x.op is Ops.BIND: subs[x] = UOp.param(i, x.dtype, x._shape, x._device, x._min_max)
      else: subs[x] = UOp.param(i, x.dtype, x._shape, x._device)
    uret = ret.uop.substitute(subs)

    # replace the implicit BUFFER inputs with params using graph_rewrite
    ctx = _ImplicitBufCtx(offset=len(input_uops))
    uret = graph_rewrite(uret, pm_implicit, ctx=ctx)

    return cast(ReturnType, Tensor(uret.call(*input_uops, *ctx.bufs, name=self.fxn.__name__), device=ret.device))

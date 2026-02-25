from typing import Generic, TypeVar, Callable
from dataclasses import dataclass, field
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite

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

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
    input_uops = [x[1].uop.multibase for x in input_tensors]

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=0):
      ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"

    # replace the known inputs with params
    uret = ret.uop.substitute({x:UOp.param(i, x.dtype, x.shape, x.device) for i,x in enumerate(input_uops)})

    # replace the implicit BUFFER inputs with params using graph_rewrite
    ctx = _ImplicitBufCtx(offset=len(input_uops))
    uret = graph_rewrite(uret, pm_implicit, ctx=ctx)

    return Tensor(uret.call(*[x.contiguous() for x in input_uops], *ctx.bufs, name=self.fxn.__name__), device=ret.device)


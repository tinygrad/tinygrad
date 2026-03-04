import functools
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv
from tinygrad.uop.ops import UOp, Ops, GroupOp, graph_rewrite, PatternMatcher, UPat
from tinygrad.tensor import Tensor

def add_to_ctx(ctx, x:UOp):
  ret = x.param_like(len(ctx))
  ctx.append(x)
  return ret

pm_ctx = PatternMatcher([
  (UPat((Ops.BUFFER, Ops.BIND), name="x"), add_to_ctx),
  (UPat((Ops.ASSIGN, Ops.CONTIGUOUS), name="x"),
   lambda ctx,x: add_to_ctx(ctx,x) if not x.op_in_backward_slice_with_self(Ops.PARAM) else None),
])
pm_make_pure = PatternMatcher([
  (UPat(Ops.ASSIGN, src=(UPat((*GroupOp.Movement, Ops.CONTIGUOUS, Ops.CAST), name="dst"), UPat(name="rhs")), name="x"),
   lambda dst, rhs, x: rhs if dst.op_in_backward_slice_with_self(Ops.PARAM) and dst in rhs.backward_slice_with_self else None),
])

ReturnType = TypeVar('ReturnType')
class _function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool=False):
    self.fxn = fxn
    self.precompile = precompile

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_uops: list[UOp] = [(t.uop if isinstance(t, Tensor) else t)
                             for name,t in list(enumerate(args))+sorted(kwargs.items()) if isinstance(t, (Tensor, UOp))]

    # use the base
    #input_uops = [x.multibase for x in input_uops]

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup(input_uops)

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(call_uops): subs[x] = x.param_like(i)
    uret = ret.uop.substitute(subs)
    uret = graph_rewrite(uret, pm_make_pure, bottom_up=True, name="impure_to_pure")

    # add contiguous to call_uops
    #call_uops = [x.contiguous() for x in call_uops]

    # the BUFFERs that are left are the implicit inputs
    uret = graph_rewrite(uret, pm_ctx, call_uops, bottom_up=True, name="get_implicit_inputs")
    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__

    # assign output
    #pbuffer = uret.param_like(len(call_uops))
    #assigned = pbuffer.assign(uret).sink()
    #buffer = UOp.new_buffer(pbuffer.device, pbuffer.size, pbuffer.dtype).reshape(uret.shape)
    #call = assigned.call(*call_uops, buffer, name=name)
    #ret = buffer.after(call)

    ret = uret.call(*call_uops, name=name, precompile=self.precompile)
    return cast(ReturnType, Tensor(ret, device=ret.device))

# overload signatures support both @function and @function(precompile=True) syntax
@overload
def function(fxn:Callable[..., ReturnType], *, precompile:bool=False) -> _function[ReturnType]: ...
@overload
def function(fxn:None=None, *, precompile:bool=False) -> Callable[[Callable[..., ReturnType]], _function[ReturnType]]: ...
def function(fxn=None, *, precompile:bool=False):
  if fxn is None: return lambda f: _function(f, precompile=precompile)
  return _function(fxn, precompile=precompile)

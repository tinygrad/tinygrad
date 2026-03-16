import functools
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat
from tinygrad.gradient import compute_gradient
from tinygrad.tensor import Tensor

def add_to_ctx(ctx, x:UOp):
  ret = x.param_like(len(ctx))
  ctx.append(x)
  return ret

pm_ctx = PatternMatcher([
  (UPat((Ops.BUFFER, Ops.BIND), name="x"), add_to_ctx),
  (UPat((Ops.ASSIGN, Ops.CONTIGUOUS), name="x"),
   lambda ctx,x: add_to_ctx(ctx,x) if not x.op_in_backward_slice_with_self(Ops.PARAM) else None),
  # strip UNIQUE from unique consts — they don't need buffer identity inside function bodies
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="x"), lambda ctx,x: x.replace(src=(x.src[1],))),
])

ReturnType = TypeVar('ReturnType')
class _function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool=False, precompile_backward:bool=False):
    self.fxn = fxn
    self.precompile = precompile
    self.precompile_backward = precompile_backward

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
    if isinstance(ret, Tensor):
      uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
      uret = UOp.maketuple(*[x.uop for x in ret])
    else:
      raise RuntimeError(f"function return type {type(ret)} not supported")

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(call_uops): subs[x] = x.param_like(i)
    uret = uret.substitute(subs)

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

    # precompute the backward: determine which inputs need gradients and build the backward CALL body
    grad_fxn = None
    inputs = [t for t in list(args)+[kwargs[k] for k in sorted(kwargs)] if isinstance(t, (Tensor, UOp))]
    grad_params = {x.arg:x for x in uret.toposort(enter_calls=False) if x.op == Ops.PARAM}
    # find which param slots correspond to requires_grad inputs
    need_grad = {i for i, t in enumerate(inputs) if isinstance(t, Tensor) and t.requires_grad}
    target_params = {grad_params[i] for i in need_grad if i in grad_params}
    if target_params:
      grad_fxn = self._make_grad_fxn(uret, len(call_uops), target_params, need_grad, name, self.precompile_backward)

    fret = uret.call(*call_uops, name=name, precompile=self.precompile, precompile_backward=self.precompile_backward, grad_fxn=grad_fxn)
    if isinstance(ret, tuple):
      return cast(ReturnType, tuple(Tensor(fret.gettuple(i), device=fret.device) for i in range(len(ret))))
    else:
      return cast(ReturnType, Tensor(fret, device=fret.device))

  @staticmethod
  def _make_grad_fxn(uret:UOp, num_args:int, target_params:set[UOp], need_grad:set[int], name:str, precompile_backward:bool):
    def grad_fxn(ctx, k):
      fxn, args = k.src[0], k.src[1:]
      params = {x.arg:x for x in fxn.toposort(enter_calls=False) if x.op == Ops.PARAM}
      # compute gradients only for needed params
      if isinstance(ctx, dict):
        all_grads: dict[UOp, UOp] = {}
        for idx, grad_out in ctx.items():
          elem_grads = compute_gradient(fxn.src[idx], grad_out.param_like(len(args) + idx), target_params)
          for p, g in elem_grads.items():
            if p in all_grads: all_grads[p] = all_grads[p] + g
            else: all_grads[p] = g
        grads = all_grads
        grad_ctx_inputs = tuple(ctx.get(i, fxn.src[i].const_like(0)) for i in range(len(fxn.src)))
      else:
        grads = compute_gradient(fxn, ctx.param_like(len(args)), target_params)
        grad_ctx_inputs = (ctx,)
      # collect gradients for needed params only
      grad_indices: list[int] = []
      grad_uops: list[UOp] = []
      for i in range(len(args)):
        if i in need_grad and (p:=params.get(i)) is not None and p in grads:
          grad_indices.append(i)
          grad_uops.append(grads[p])
      if len(grad_uops) == 0: return (None,) * len(args)
      # replace forward output references with PARAMs to avoid recomputation
      if fxn.op is Ops.TUPLE:
        fwd_subs = {elem: elem.param_like(len(args) + len(grad_ctx_inputs) + i) for i, elem in enumerate(fxn.src)}
        fwd_inputs = tuple(k.gettuple(i) for i in range(len(fxn.src)))
      else:
        fwd_subs = {fxn: fxn.param_like(len(args) + 1)}
        fwd_inputs = (k,)
      grad_uops = [g.substitute(fwd_subs) for g in grad_uops]
      # build a single backward CALL returning a TUPLE of all gradients
      bwd_body = UOp.maketuple(*grad_uops)
      bwd_call = bwd_body.call(*args, *grad_ctx_inputs, *fwd_inputs, name=name+"_backward", precompile=precompile_backward)
      # extract each gradient via GETTUPLE
      ret: list[UOp|None] = []
      gi = 0
      for i in range(len(args)):
        if gi < len(grad_indices) and grad_indices[gi] == i:
          ret.append(bwd_call.gettuple(gi))
          gi += 1
        else:
          ret.append(None)
      return tuple(ret)
    return grad_fxn

# overload signatures support both @function and @function(precompile=True) syntax
@overload
def function(fxn:Callable[..., ReturnType], *, precompile:bool=False, precompile_backward:bool=False) -> _function[ReturnType]: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False) -> \
  Callable[[Callable[..., ReturnType]], _function[ReturnType]]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False):
  if fxn is None: return lambda f: _function(f, precompile=precompile, precompile_backward=precompile_backward)
  return _function(fxn, precompile=precompile, precompile_backward=precompile_backward)

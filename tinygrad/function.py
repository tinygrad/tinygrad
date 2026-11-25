import functools, time
from dataclasses import replace
from typing import TypeVar, Callable, overload
from tinygrad.helpers import Context, dedup, getenv, DEBUG
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

def _invalid_outputs(topo:dict[UOp, None]) -> set[UOp]:
  # invalids() returns fresh write-only scratch: a clone storing CONST(Invalid)
  return {u.src[0].buf_uop for u in topo if u.op is Ops.STORE and u.src[1].base.is_invalid and not u.src[0].buf_uop.is_realized}

def _canonicalize_allocs(nodes, invalid_bufs:set[UOp]) -> dict[UOp, UOp]:
  return {b: b.replace(op=Ops.ALLOC, arg=replace(b.arg, slot=-1-i, buffer=None, bind_on_realize=b.arg.bind_on_realize and b not in invalid_bufs))
          for i, b in enumerate(x for x in nodes if x in invalid_bufs or (x.op is Ops.ALLOC and x.arg.slot >= 0))}

def renumber_invalid_outputs(uret:UOp) -> UOp:
  topo = uret.toposort()
  return uret.substitute(_canonicalize_allocs(topo, _invalid_outputs(topo)), walk=True)

F = TypeVar('F', bound=Callable)
@overload
def function(fxn:F, *, precompile:bool=False, precompile_backward:bool=False, allow_implicit:bool=False, grad_fxn:Callable|None=None) -> F: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> Callable[[F], F]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False, allow_implicit:bool=False, grad_fxn:Callable|None=None):
  if fxn is None:
    return functools.partial(function, precompile=precompile, precompile_backward=precompile_backward,
                             allow_implicit=allow_implicit, grad_fxn=grad_fxn)

  name = getattr(fxn, '__qualname__', None) or type(fxn).__qualname__
  @functools.wraps(fxn)
  def call(*args, **kwargs):
    st = time.perf_counter()

    params = get_state_dict((args, kwargs), tensor_type=(Tensor, UOp)).values()

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup([u for t in params if (u:=t._uop).device is not None])

    alloc_start = next(UOp.unique_num)
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      ret = fxn(*args, **kwargs)
    if isinstance(ret, Tensor): uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret): uret = UOp.sink(*[x.uop for x in ret])
    else: raise RuntimeError(f"function return type {type(ret)} not supported")

    # replace known inputs with params (deduplicated slots), capture caller-owned implicit storage, and canonicalize local ALLOCs/invalids in one pass
    HAS_PARAM, HAS_IMPLICIT, IS_IMPLICIT = 1, 2, 4
    num_explicit, topo, flags = len(call_uops), uret.toposort(), dict[UOp, int]()
    explicit, implicit_subs = {x: x.param_like(i) for i, x in enumerate(call_uops)}, dict[UOp, UOp]()
    invalid_bufs = _invalid_outputs(topo)
    for u in topo:
      if u in explicit: flags[u] = HAS_PARAM
      else:
        # Variables are caller-provided values (slot -1); capture caller-owned storage, not allocations created while tracing this function
        is_implicit = (u.is_variable and u.arg.slot == -1) or u.op is Ops.BUFFER or \
          (u.op is Ops.ALLOC and u.arg.bind_on_realize and u.arg.slot < alloc_start)
        flags[u] = functools.reduce(lambda a, s: a | (flags[s] & (HAS_PARAM | HAS_IMPLICIT)), u.src_without_body,
                                    (HAS_PARAM if u.op is Ops.PARAM and u.arg.slot >= 0 else 0) | (HAS_IMPLICIT | IS_IMPLICIT if is_implicit else 0))
    # caller-owned storage left in the graph becomes implicit inputs (don't capture fresh write-only invalid buffers)
    seen, stack = set[UOp](), [uret]
    while stack:
      if (u:=stack.pop()) in seen: continue
      seen.add(u)
      if u in explicit or u in implicit_subs: continue
      is_imp_leaf = u.op in GroupOp.Defines and bool(flags[u] & IS_IMPLICIT)
      if (is_imp_leaf or (u.op in {Ops.AFTER, Ops.STAGE} and flags[u] == HAS_IMPLICIT)) and u.buf_uop not in invalid_bufs:
        implicit_subs[u] = u.param_like(len(call_uops))
        call_uops.append(u)
      else: stack.extend(reversed(u.src_without_body))
    for v in (v for p in (*explicit.values(), *implicit_subs.values()) for v in p.toposort() if v.is_variable and v.arg.slot == -1):
      if v not in implicit_subs:
        implicit_subs[v] = v.param_like(len(call_uops))
        call_uops.append(v)
    subs = ({k: v.substitute(implicit_subs, walk=True) for k, v in (explicit | implicit_subs).items()} if implicit_subs else explicit) | \
      _canonicalize_allocs((x for x in topo if x in seen and x not in explicit and x not in implicit_subs), invalid_bufs)
    uret = uret.substitute(subs, walk=True)
    if not allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op in {Ops.BUFFER, Ops.ALLOC}]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.max_numel()}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    outs = UOp.call_with_outputs(uret.src if isinstance(ret, tuple) else (uret,), *call_uops, grad_fxn=grad_fxn,
                                 precompile=precompile, precompile_backward=precompile_backward, lift=False)

    if DEBUG >= 2:
      print(f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}")

    return tuple(Tensor(o) for o in outs) if isinstance(ret, tuple) else Tensor(outs[0])
  return call

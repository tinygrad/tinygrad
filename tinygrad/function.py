import functools
from typing import Generic, TypeVar, Callable, cast
from tinygrad.helpers import Context, dedup, getenv
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import Tensor

def _srcs(u:UOp) -> tuple[UOp, ...]:
  """Get sources of a UOp, skipping src[0] of CALL nodes (other functions' bodies with their own PARAMs)."""
  return u.src[1:] if u.op is Ops.CALL else u.src

def _find_implicit_inputs(uret:UOp) -> list[UOp]:
  """Find implicit inputs by starting at remaining BUFFERs and walking up to the branching point where PARAM-derived nodes meet."""
  all_nodes = list(uret.toposort())
  # build parent map, gating on src[0] of CALL nodes
  parents_of: dict[UOp, set[UOp]] = {}
  for u in all_nodes:
    for s in _srcs(u):
      parents_of.setdefault(s, set()).add(u)
  # mark which nodes have a PARAM in their subtree (bottom-up, toposort is already bottom-up)
  has_param: dict[UOp, bool] = {}
  for u in all_nodes:
    if u.op is Ops.PARAM: has_param[u] = True
    else: has_param[u] = any(has_param.get(s, False) for s in _srcs(u))
  # for each remaining BUFFER, walk up until we hit a node whose parent has PARAM in its subtree
  implicit_inputs: list[UOp] = []
  for buf in all_nodes:
    if buf.op is not Ops.BUFFER: continue
    cur = buf
    while True:
      ps = parents_of.get(cur, set())
      if not ps or any(has_param.get(p, False) for p in ps):
        implicit_inputs.append(cur)
        break
      cur = next(iter(ps))
  return dedup(implicit_inputs)

ReturnType = TypeVar('ReturnType')
class function(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_uops: list[UOp] = [(t.uop if isinstance(t, Tensor) else t)
                             for name,t in list(enumerate(args))+sorted(kwargs.items()) if isinstance(t, (Tensor, UOp))]

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    unique_uops: list[UOp] = dedup(input_uops)

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      ret = self.fxn(*args, **kwargs)
    assert isinstance(ret, Tensor), "only supports one tensor return for now"

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(unique_uops):
      # TODO: this can be better
      if x.op is Ops.BIND: subs[x] = UOp.param(i, x.dtype, x._shape, x._device, x._min_max)
      else: subs[x] = UOp.param(i, x.dtype, x._shape, x._device)
    uret = ret.uop.substitute(subs)

    # find implicit inputs by walking up from remaining BUFFERs to branching points
    implicit = _find_implicit_inputs(uret)
    for i,imp in enumerate(implicit):
      subs[imp] = UOp.param(len(unique_uops) + i, imp.dtype, imp._shape, imp._device)
    uret = ret.uop.substitute(subs)

    return cast(ReturnType, Tensor(uret.call(*unique_uops, *implicit, name=self.fxn.__name__), device=ret.device))

# the elementwise inputs of custom kernels computed in them: a custom kernel's input that is elementwise ops (and movement ops) of buffers isn't
# materialized by a kernel of its own, the custom kernel's loads of it load the buffers and compute it (at the indices it loads)
from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp
from tinygrad.helpers import getenv, prod
from tinygrad.dtype import Invalid, dtypes
from tinygrad.schedule.indexing import apply_movement_op, broadcast_rngs, _stack_select

INLINE_MOVEMENT = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK}
INLINE_OPS = GroupOp.ALU | {Ops.CAST, Ops.STACK}

def _expr_ok(s:UOp, limit:int=64) -> bool:
  # elementwise and movement ops (not sharded) above the leaves, at least one of them (a buffer by itself is passed as it is). the leaves:
  # arrays of a concrete dtype (no scalars, variables or weak dtypes: they aren't kernel args a load reads)
  seen:set[UOp] = set()
  inner, stack = 0, [s]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if u.op is Ops.CONST: continue
    if u.dtype in dtypes.weaks or u._shape is None: return False
    if u.op in INLINE_OPS | INLINE_MOVEMENT:
      inner += u.op in INLINE_OPS
      if u.axis is not None or (isinstance(u.device, tuple) and u.shape != u.max_shard_shape): return False
      # (a movement op's other srcs are its arguments)
      stack.extend(u.src if u.op in INLINE_OPS else u.src[:1])
    elif u.shape == () or u.device is None or u.is_variable: return False
  return inner > 0 and len(seen) <= limit

def _chain_to(u:UOp, p:UOp) -> list[UOp]|None:
  # the movement ops from u down to the param p (u first), None if u isn't a view of p
  chain = []
  while u is not p:
    if u.op not in INLINE_MOVEMENT: return None
    chain.append(u)
    u = u.src[0]
  return chain

def inline_custom_kernel_srcs(c:UOp) -> UOp|None:
  body, srcs = c.src[0], c.src[1:]
  if body.op is not Ops.SINK or not getenv("INLINE_CUSTOM_SRCS", 1): return None
  nodes = body.toposort()
  params = {u.arg.slot: u for u in nodes if u.op is Ops.PARAM and u.arg.slot >= 0}
  consumers: dict[UOp, list[UOp]] = {}
  for u in nodes:
    for s in u.src: consumers.setdefault(s, []).append(u)
  # (a param is the flat storage of its arg, the kernel's view of it a reshape)
  inline = [i for i, s in enumerate(srcs) if i in params and params[i]._shape is not None and s._shape is not None and
            (params[i].shape == s.shape or params[i].shape == (prod(s.shape),)) and _expr_ok(s)]
  if not inline: return None
  repl: dict[UOp, UOp] = {}
  leaves: dict[UOp, tuple[int, UOp]] = {}  # leaf -> the slot of its new param, the kernel's view of it
  done: list[int] = []
  for i in inline:
    p, local = params[i], {}
    new_leaves: dict[UOp, tuple[int, UOp]] = {}
    def leaf_param(x:UOp) -> UOp:
      if x in leaves: return leaves[x][1]
      if x not in new_leaves: new_leaves[x] = (slot:=len(srcs) + len(leaves) + len(new_leaves), UOp.placeholder_like(x, slot=slot))
      return new_leaves[x][1]
    def ev(x:UOp, idx:tuple[UOp, ...]) -> UOp:
      if x.op in INLINE_MOVEMENT: return ev(x.src[0], apply_movement_op(x.op, x.src[0].shape, x.marg, idx))
      if x.op is Ops.CONST: return x.replace(src=())
      # (a STACK: the source of the leading index, a select as rangeify's)
      if x.op is Ops.STACK: return _stack_select(idx[0], [ev(s, idx[1:]) for s in x.src])
      if x.op in INLINE_OPS: return x.replace(src=tuple(ev(s, broadcast_rngs(x, s, idx)) for s in x.src))
      return leaf_param(x).index(*idx).load()
    # every use of the param: views of it indexed with all their dims, the index loaded (or used as the value), not stored to
    def uses(u:UOp) -> bool:
      for cu in consumers.get(u, []):
        if cu.op in INLINE_MOVEMENT and cu.src[0] is u:
          if not uses(cu): return False
        elif cu.op is Ops.INDEX and cu.src[0] is u:
          idx = cu.src[1:]
          if len(idx) != len(u.shape) or any(y.op is Ops.CONST and y.arg is Invalid for v in idx for y in v.toposort()) or \
             any(o.op is Ops.STORE and o.src[0] is cu for o in consumers.get(cu, [])): return False
          pidx = tuple(idx)
          for m in (_chain_to(u, p) or []): pidx = apply_movement_op(m.op, m.src[0].shape, m.marg, pidx)
          if p.shape != srcs[i].shape: pidx = apply_movement_op(Ops.RESHAPE, srcs[i].shape, p.shape, pidx)
          local[cu] = ev(srcs[i], pidx)
          for o in consumers.get(cu, []):
            if o.op is Ops.LOAD: local[o] = local[cu]
        else: return False
      return True
    if uses(p) and local:
      repl.update(local)
      leaves.update(new_leaves)
      done.append(i)
  if not done: return None
  # the new body: the inlined params gone (the other slots renumbered), the leaves' params after them
  keep = [i for i in range(len(srcs)) if i not in done]
  slot_map = {old: new for new, old in enumerate(keep)}
  slot_map.update({slot: len(keep) + j for j, (slot, _) in enumerate(leaves.values())})
  new_body = body.substitute(repl, walk=True)
  renum = {u: u.replace(arg=replace(u.arg, slot=slot_map[u.arg.slot])) for u in new_body.toposort() if u.op is Ops.PARAM and u.arg.slot >= 0
           and u.arg.slot in slot_map and slot_map[u.arg.slot] != u.arg.slot}
  new_body = new_body.substitute(renum, walk=True)
  return c.replace(src=(new_body, *[srcs[i] for i in keep], *leaves.keys()))

pm_inline_custom_srcs = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), allow_any_len=True, name="c"), inline_custom_kernel_srcs)])

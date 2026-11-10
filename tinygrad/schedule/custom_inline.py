# the elementwise inputs of custom kernels computed in them: a custom kernel's input that is elementwise ops (and movement ops) of buffers isn't
# materialized by a kernel of its own, the custom kernel's loads of it load the buffers and compute it (at the indices it loads)
import functools
from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp
from tinygrad.helpers import prod, ContextVar
from tinygrad.dtype import Invalid, dtypes
from tinygrad.schedule.indexing import apply_movement_op, broadcast_rngs, _stack_select

PARAM_VIEWS = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK}
INLINE_MOVEMENT = PARAM_VIEWS | {Ops.PAD}
INLINE_OPS = GroupOp.ALU | {Ops.CAST, Ops.STACK}
# a sum over a leading axis of at most this many (k-split partials) is computed in the loads too, in order: ((p0 + p1) + p2) + ...
INLINE_REDUCE, INLINE_REDUCE_READS = ContextVar("INLINE_REDUCE", 0), ContextVar("INLINE_REDUCE_READS", 4)

def _small_reduce(u:UOp) -> bool:
  return u.op is Ops.REDUCE and u.arg[0] is Ops.ADD and u.arg[1] == 1 and isinstance(k:=u.src[0].shape[0], int) and 1 < k <= INLINE_REDUCE.value

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
    if u.op in INLINE_OPS | INLINE_MOVEMENT or _small_reduce(u):
      # (a pad of a buffer isn't a buffer: a copy kernel, unless loaded through it)
      inner += u.op in INLINE_OPS or u.op is Ops.PAD or _small_reduce(u)
      if u.axis is not None or (isinstance(u.device, tuple) and u.shape != u.max_shard_shape): return False
      # (a movement op's other srcs are its arguments)
      stack.extend(u.src if u.op in INLINE_OPS else u.src[:1])
    elif u.shape == () or u.device is None or u.is_variable: return False
  return inner > 0 and len(seen) <= limit

def _leaves(s:UOp) -> list[UOp]:
  # the leaves of an inlinable expression (_expr_ok's)
  seen: set[UOp] = set()
  stack, ret = [s], []
  while stack:
    if (u:=stack.pop()) in seen or u.op is Ops.CONST: continue
    seen.add(u)
    if u.op in INLINE_OPS | INLINE_MOVEMENT or _small_reduce(u): stack.extend(u.src if u.op in INLINE_OPS else u.src[:1])
    else: ret.append(u)
  return ret

class InlineCtx:
  # the number of states (AFTERs) of every buffer in the graph: a leaf whose buffer has states after the leaf's may be a value read before a
  # write that the custom kernel runs after (a copy kernel can run before the write, the custom kernel's loads can't)
  def __init__(self, sink:UOp):
    # (a STORE made a dependency of another buffer's state writes its target without a state of it: any such target is written)
    self.states: dict[UOp, int] = {}
    self.stored: set[UOp] = set()
    for u in sink.toposort():
      if u.op is not Ops.AFTER: continue
      self.states[u.storage_base] = self.states.get(u.storage_base, 0) + 1
      self.stored.update(x.src[0].storage_base for x in u.src[1:] if x.op is Ops.STORE and x.src[0].storage_base is not u.storage_base)
  def written(self, leaf:UOp) -> bool:
    n, x = 0, leaf
    while x.op is Ops.AFTER or x.op in INLINE_MOVEMENT:
      n += x.op is Ops.AFTER
      x = x.src[0]
    return leaf.storage_base in self.stored or self.states.get(leaf.storage_base, 0) > n

def _chain_to(u:UOp, p:UOp) -> list[UOp]|None:
  # the movement ops from u down to the param p (u first), None if u isn't a view of p
  chain = []
  while u is not p:
    if u.op not in PARAM_VIEWS: return None
    chain.append(u)
    u = u.src[0]
  return chain

def _safe(ctx:InlineCtx, c:UOp, i:int, s:UOp) -> bool:
  # the leaves aren't the buffers of the kernel's other args (it may write them in place) nor written after the state read
  others = {x.storage_base for j, x in enumerate(c.src[1:]) if j != i and x._shape is not None}
  return not any(lf.storage_base in others or ctx.written(lf) for lf in _leaves(s))

def inline_custom_kernel_srcs(ctx:InlineCtx|None, c:UOp) -> UOp|None:
  body, srcs = c.src[0], c.src[1:]
  if body.op is not Ops.SINK: return None
  nodes = body.toposort()
  params = {u.arg.slot: u for u in nodes if u.op is Ops.PARAM and u.arg.slot >= 0}
  consumers: dict[UOp, list[UOp]] = {}
  for u in nodes:
    for s in u.src: consumers.setdefault(s, []).append(u)
  # (a param is the flat storage of its arg, the kernel's view of it a reshape)
  inline = [i for i, s in enumerate(srcs) if i in params and params[i]._shape is not None and s._shape is not None and
            (params[i].shape == s.shape or params[i].shape == (prod(s.shape),)) and _expr_ok(s) and
            (ctx is None or _safe(ctx, c, i, s))]
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
      if x.op is Ops.PAD:
        # (the loads of the source gated to its range, the value outside of it zero)
        pidx = apply_movement_op(x.op, x.src[0].shape, x.marg, idx)
        valid = functools.reduce(lambda a, b: a & b, [i.get_valid() for i in pidx])
        return valid.where(ev(x.src[0], pidx), UOp.const(0, x.dtype))
      if x.op in INLINE_MOVEMENT: return ev(x.src[0], apply_movement_op(x.op, x.src[0].shape, x.marg, idx))
      if x.op is Ops.CONST: return x.replace(src=())
      # (a STACK: the source of the leading index, a select as rangeify's)
      if x.op is Ops.STACK: return _stack_select(idx[0], [ev(s, idx[1:]) for s in x.src])
      if x.op in INLINE_OPS: return x.replace(src=tuple(ev(s, broadcast_rngs(x, s, idx)) for s in x.src))
      if _small_reduce(x):
        parts = [ev(x.src[0], (UOp.const(j, idx[0].dtype if idx else dtypes.weakint),) + idx) for j in range(x.src[0].shape[0])]
        return functools.reduce(lambda a, b: a + b, parts)
      return leaf_param(x).index(*idx).load()
    def indexed(u:UOp, cu:UOp, prefix:tuple[UOp, ...]) -> bool:
      # an index of the view u (cu, after the indices prefix of the INDEXes it indexes): all its dims indexed (an INDEX of fewer is indexed by
      # INDEXes of the rest), loaded (or used as the value), not stored to
      idx = prefix + cu.src[1:]
      if any(y.op is Ops.CONST and y.arg is Invalid for v in cu.src[1:] for y in v.toposort()) or \
         any(o.op is Ops.STORE and o.src[0] is cu for o in consumers.get(cu, [])): return False
      if len(idx) < len(u.shape):
        return len(cs:=consumers.get(cu, [])) > 0 and all(o.op is Ops.INDEX and o.src[0] is cu and indexed(u, o, idx) for o in cs)
      if len(idx) != len(u.shape): return False
      pidx = tuple(idx)
      for m in (_chain_to(u, p) or []): pidx = apply_movement_op(m.op, m.src[0].shape, m.marg, pidx)
      if p.shape != srcs[i].shape: pidx = apply_movement_op(Ops.RESHAPE, srcs[i].shape, p.shape, pidx)
      local[cu] = ev(srcs[i], pidx)
      for o in consumers.get(cu, []):
        if o.op is Ops.LOAD: local[o] = local[cu]
      return True
    # every use of the param: views of it indexed
    def uses(u:UOp) -> bool:
      for cu in consumers.get(u, []):
        if cu.op in PARAM_VIEWS and cu.src[0] is u:
          if not uses(cu): return False
        elif cu.op is Ops.INDEX and cu.src[0] is u:
          if not indexed(u, cu, ()): return False
        else: return False
      return True
    if uses(p) and local:
      # (a small reduce in it: its partials read at every load. not if the kernel loads an element more than INLINE_REDUCE_READS times, a bound:
      # the threads of all its ranges, every one at every index of it)
      if any(_small_reduce(u) for u in srcs[i].toposort()):
        threads = prod(int(r.vmax)+1 for r in nodes if r.op is Ops.RANGE)
        if threads * sum(1 for u in local if u.op is Ops.INDEX) > INLINE_REDUCE_READS.value * prod(p.shape): continue
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

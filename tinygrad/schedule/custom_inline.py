# the inputs of custom kernels computed in them: an input that is elementwise ops and views of buffers isn't materialized by a kernel of its own,
# the custom kernel's loads of it load the buffers and compute it (at the indices it loads)
import functools
from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, GroupOp, KernelInfo
from tinygrad.helpers import prod
from tinygrad.dtype import Invalid, dtypes
from tinygrad.schedule.indexing import apply_movement_op, broadcast_rngs, _stack_select

PARAM_VIEWS = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK}
INLINE_MOVEMENT = PARAM_VIEWS | {Ops.PAD}

def _inlined(u:UOp) -> bool:
  # elementwise ops (a bitcast of the same width), stacks and views: computed at the index loaded
  if u.op is Ops.BITCAST: return u.dtype.itemsize == u.src[0].dtype.itemsize
  return u.op in GroupOp.ALU | {Ops.CAST, Ops.STACK} | INLINE_MOVEMENT

def _expr_srcs(u:UOp) -> tuple[UOp, ...]: return u.src[:1] if u.op in INLINE_MOVEMENT else u.src

def _leaves(s:UOp, limit:int=64) -> list[UOp]|None:
  # the arrays an input is computed from (not sharded, of a concrete dtype: no scalars, variables or weak dtypes, they aren't kernel args a load
  # reads), None if it isn't such an expression. a buffer (or a reshape of one) is passed as it is
  top = s
  while top.op is Ops.RESHAPE: top = top.src[0]
  if not _inlined(top): return None
  seen: set[UOp] = set()
  stack, leaves = [s], []
  while stack:
    if (u:=stack.pop()) in seen or u.op is Ops.CONST: continue
    seen.add(u)
    if u.dtype in dtypes.weaks or u._shape is None: return None
    if _inlined(u):
      if u.axis is not None or (isinstance(u.device, tuple) and u.shape != u.max_shard_shape): return None
      stack.extend(_expr_srcs(u))
    elif u.shape == () or u.device is None or u.is_variable: return None
    else: leaves.append(u)
  return leaves if len(seen) <= limit else None

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

def _loaded(u:UOp, consumers:dict[UOp, list[UOp]]) -> bool:
  # every use of a param (u, a view or an INDEX of it) a load: its INDEXes loaded or values of ALU ops and stores, never addresses
  for c in consumers.get(u, []):
    if c.op in PARAM_VIEWS | {Ops.INDEX} and c.src[0] is u:
      if not _loaded(c, consumers): return False
    elif u.op is not Ops.INDEX or c.op not in GroupOp.ALU | {Ops.CAST, Ops.BITCAST, Ops.LOAD, Ops.STORE, Ops.INDEX} or \
         (c.op is Ops.STORE and c.src[0] is u): return False
  return True

def _chain_to(u:UOp, p:UOp) -> list[UOp]:
  # the movement ops from the view u down to the param p (u first)
  chain = []
  while u is not p:
    chain.append(u)
    u = u.src[0]
  return chain

def inline_custom_kernel_srcs(ctx:InlineCtx, c:UOp) -> UOp|None:
  body, srcs = c.src[0], c.src[1:]
  if not isinstance(body.arg, KernelInfo): return None
  nodes = body.toposort()
  params = {u.arg.slot: u for u in nodes if u.op is Ops.PARAM and u.arg.slot >= 0}
  consumers: dict[UOp, list[UOp]] = {}
  for u in nodes:
    for s in u.src: consumers.setdefault(s, []).append(u)
  # the leaves aren't the buffers of args the kernel writes (in place, or through an address) nor written after the state read
  written = {srcs[j].storage_base for j, p in params.items() if j < len(srcs) and not _loaded(p, consumers)}
  # (a param is the flat storage of its arg, the kernel's view of it a reshape)
  inline = [i for i, s in enumerate(srcs) if i in params and params[i]._shape is not None and s._shape is not None and
            params[i].shape in (s.shape, (prod(s.shape),)) and (lv:=_leaves(s)) is not None and
            not any(lf.storage_base in written or ctx.written(lf) for lf in lv)]
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
      if _inlined(x): return x.replace(src=tuple(ev(s, broadcast_rngs(x, s, idx)) for s in x.src))
      return leaf_param(x).index(*idx).load()
    def indexed(u:UOp, cu:UOp, prefix:tuple[UOp, ...]) -> bool:
      # an index of the view u (cu, after the indices prefix of the INDEXes it indexes): all its dims indexed (an INDEX of fewer is indexed by
      # INDEXes of the rest), loaded (or used as the value)
      idx = prefix + cu.src[1:]
      if any(y.op is Ops.CONST and y.arg is Invalid for v in cu.src[1:] for y in v.toposort()): return False
      if len(idx) < len(u.shape):
        return len(cs:=consumers.get(cu, [])) > 0 and all(o.op is Ops.INDEX and o.src[0] is cu and indexed(u, o, idx) for o in cs)
      if len(idx) != len(u.shape) or not _loaded(cu, consumers): return False
      pidx = tuple(idx)
      for m in _chain_to(u, p): pidx = apply_movement_op(m.op, m.src[0].shape, m.marg, pidx)
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

# e-graph (equality saturation) for UOp rewriting
# instead of greedy first-match rewriting, we explore ALL equivalent forms and extract the cheapest
from __future__ import annotations
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, graph_rewrite

# *** union-find (keyed by UOp identity) ***

def uf_find(parent:dict[UOp, UOp], x:UOp) -> UOp:
  while parent[x] is not x:
    parent[x] = parent[parent[x]]
    x = parent[x]  # path compression
  return x

def uf_union(parent:dict[UOp, UOp], size:dict[UOp, int], a:UOp, b:UOp) -> UOp:
  a, b = uf_find(parent, a), uf_find(parent, b)
  if a is b: return a
  if size[a] < size[b]: a, b = b, a  # merge smaller into larger
  parent[b] = a
  size[a] += size[b]
  return a

# *** e-graph core ***

def rewrite_all(pm:PatternMatcher, uop:UOp, ctx=None) -> list[UOp]:
  """Apply ALL matching rewrite rules to uop, returning every distinct result."""
  results: list[UOp] = []
  seen: dict[UOp, None] = {}
  for _, match, early_reject in pm.pdict.get(uop.op, []):
    if not early_reject.issubset({u.op for u in uop.src}): continue
    try: ret = match(uop, ctx)
    except Exception: continue  # skip rules that crash on this node (e.g. division by zero in divmod folding)
    if ret is not None and ret is not uop and ret not in seen:
      results.append(ret)
      seen[ret] = None
  return results

class EGraph:
  """E-graph with full equality saturation (including rebuilding)."""
  __slots__ = ("parent", "size", "eclass", "eclass_uses", "all_nodes")
  def __init__(self, root:UOp):
    nodes = list(root.toposort())
    self.parent: dict[UOp, UOp] = {u: u for u in nodes}
    self.size: dict[UOp, int] = {u: 1 for u in nodes}
    self.eclass: dict[UOp, dict[UOp, None]] = {u: {u: None} for u in nodes}  # canonical -> members
    # canonical eclass representative -> dict of nodes that USE this eclass as a child
    self.eclass_uses: dict[UOp, dict[UOp, None]] = {u: {} for u in nodes}
    self.all_nodes: dict[UOp, None] = dict.fromkeys(nodes)
    # build initial parent-child uses
    for u in nodes:
      for s in u.src:
        canon = uf_find(self.parent, s)
        self.eclass_uses.setdefault(canon, {})[u] = None

  def _add_node(self, u:UOp):
    """Register a new UOp (and its subtree) in the e-graph."""
    for sub in u.toposort():
      if sub in self.parent: continue
      self.parent[sub] = sub
      self.size[sub] = 1
      self.eclass[sub] = {sub: None}
      self.all_nodes[sub] = None
      self.eclass_uses[sub] = {}
      for s in sub.src:
        canon = uf_find(self.parent, s)
        self.eclass_uses.setdefault(canon, {})[sub] = None

  def _merge(self, a:UOp, b:UOp) -> UOp|None:
    """Merge two e-classes. Returns the winner, or None if already merged."""
    ra, rb = uf_find(self.parent, a), uf_find(self.parent, b)
    if ra is rb: return None
    winner = uf_union(self.parent, self.size, ra, rb)
    loser = rb if winner is ra else ra
    self.eclass[winner] = {**self.eclass[winner], **self.eclass[loser]}
    # merge uses
    winner_uses = self.eclass_uses.setdefault(winner, {})
    winner_uses.update(self.eclass_uses.pop(loser, {}))
    del self.eclass[loser]
    return winner

  def _canonical(self, u:UOp) -> UOp:
    """Rebuild node with canonical representative for each child's eclass."""
    if not u.src: return u
    new_src = []
    for s in u.src:
      canon = uf_find(self.parent, s)
      members = self.eclass.get(canon)
      if members is not None:
        best = min(members, key=lambda m: (len(m.src), m.op.value, m.arg if isinstance(m.arg, (int, float, str)) else 0))
        new_src.append(best)
      else:
        new_src.append(s)
    new_src_tuple = tuple(new_src)
    if new_src_tuple == u.src: return u
    return UOp(u.op, u.dtype, new_src_tuple, u.arg, u.tag)

  def _rebuild(self, dirty:dict[UOp, None], pm:PatternMatcher, ctx=None) -> list[tuple[UOp, UOp]]:
    """Rebuild parents of dirty eclasses, creating canonical versions and matching rules."""
    new_equalities: list[tuple[UOp, UOp]] = []
    affected: dict[UOp, None] = {}
    for d in dirty:
      canon = uf_find(self.parent, d)
      affected.update(self.eclass_uses.get(canon, {}))
    for u in affected:
      rebuilt = self._canonical(u)
      if rebuilt is not u:
        if rebuilt in self.parent and uf_find(self.parent, rebuilt) is uf_find(self.parent, u): continue
        self._add_node(rebuilt)
        new_equalities.append((u, rebuilt))
        for new in rewrite_all(pm, rebuilt, ctx):
          if new in self.parent and uf_find(self.parent, new) is uf_find(self.parent, rebuilt): continue
          self._add_node(new)
          new_equalities.append((rebuilt, new))
    return new_equalities

def egraph_saturate(root:UOp, pm:PatternMatcher, max_iters:int=10, ctx=None) -> dict[UOp, dict[UOp, None]]:
  """Build an e-graph with full equality saturation (with rebuilding). Returns eclass map."""
  eg = EGraph(root)
  for _ in range(max_iters):
    # phase 1: match all rules on all known nodes, skip if already in same eclass
    new_equalities: list[tuple[UOp, UOp]] = []
    for u in list(eg.all_nodes):
      for new in rewrite_all(pm, u, ctx):
        if new in eg.parent and uf_find(eg.parent, new) is uf_find(eg.parent, u): continue
        eg._add_node(new)
        new_equalities.append((u, new))
    if not new_equalities: break

    # phase 2: merge and rebuild until no new merges
    while new_equalities:
      dirty: dict[UOp, None] = {}
      for a, b in new_equalities:
        merged = eg._merge(a, b)
        if merged is not None: dirty[merged] = None
      if not dirty: break
      # phase 3: rebuild parents of dirty eclasses
      new_equalities = eg._rebuild(dirty, pm, ctx)

  return eg.eclass

# *** cost model ***

OP_COST: dict[Ops, int] = {
  Ops.CONST: 0, Ops.VCONST: 0, Ops.DEFINE_VAR: 0,
  Ops.ADD: 1, Ops.MUL: 2, Ops.SUB: 1, Ops.NEG: 1,
  Ops.IDIV: 5, Ops.MOD: 5, Ops.FDIV: 3,
  Ops.SHL: 1, Ops.SHR: 1,
  Ops.AND: 1, Ops.OR: 1, Ops.XOR: 1,
  Ops.MAX: 1, Ops.CMPLT: 1, Ops.CMPNE: 1, Ops.CMPEQ: 1,
  Ops.CAST: 1, Ops.BITCAST: 1,
  Ops.WHERE: 2, Ops.MULACC: 2,
  Ops.EXP2: 8, Ops.LOG2: 8, Ops.SIN: 8, Ops.SQRT: 4, Ops.RECIPROCAL: 3,
  Ops.POW: 10, Ops.TRUNC: 1,
}

def node_cost(u:UOp) -> int:
  c = OP_COST.get(u.op, 3)
  # tiebreaker: penalize non-canonical operand order (consts should be on the right for commutative ops)
  if len(u.src) == 2 and u.src[0].op is Ops.CONST and u.src[1].op is not Ops.CONST: c += 1
  return c

# *** extraction ***

def egraph_extract(root:UOp, pm:PatternMatcher, max_iters:int=10, ctx=None) -> UOp:
  """Run equality saturation on root, then extract the cheapest equivalent expression."""
  eclass = egraph_saturate(root, pm, max_iters, ctx)

  # build eclass lookup: node -> canonical eclass representative
  eclass_of: dict[UOp, UOp] = {}
  for canon, members in eclass.items():
    for u in members: eclass_of[u] = canon

  all_nodes: list[UOp] = [u for members in eclass.values() for u in members]

  # bottom-up DP: for each eclass, find the cheapest representative
  cost_of: dict[UOp, tuple[int, UOp]] = {}  # eclass_canon -> (cost, best_uop)

  depth_cache: dict[UOp, int] = {}
  def _depth(u:UOp) -> int:
    if u in depth_cache: return depth_cache[u]
    depth_cache[u] = 0  # break cycles
    depth_cache[u] = (1 + max((_depth(s) for s in u.src), default=0)) if u.src else 0
    return depth_cache[u]

  for u in sorted(all_nodes, key=_depth):
    canon = eclass_of[u]
    child_cost = 0
    for s in u.src:
      if (s_canon := eclass_of.get(s)) is not None and s_canon in cost_of: child_cost += cost_of[s_canon][0]
      else: child_cost += node_cost(s)
    total = node_cost(u) + child_cost
    if canon not in cost_of or total < cost_of[canon][0]:
      cost_of[canon] = (total, u)

  root_canon = eclass_of.get(root)
  if root_canon is not None and root_canon in cost_of: return _rebuild_tree(cost_of[root_canon][1], eclass_of, cost_of)
  return root

def _rebuild_tree(u:UOp, eclass_of:dict[UOp, UOp], cost_of:dict[UOp, tuple[int, UOp]]) -> UOp:
  """Recursively rebuild a UOp tree, picking the cheapest representative for each child's eclass."""
  if not u.src: return u
  new_src = []
  for s in u.src:
    s_canon = eclass_of.get(s)
    if s_canon is not None and s_canon in cost_of:
      new_src.append(_rebuild_tree(cost_of[s_canon][1], eclass_of, cost_of))
    else:
      new_src.append(_rebuild_tree(s, eclass_of, cost_of))
  new_src_tuple = tuple(new_src)
  return u if new_src_tuple == u.src else UOp(u.op, u.dtype, new_src_tuple, u.arg, u.tag)

# *** graph-level rewrite: drop-in replacement for graph_rewrite when EGRAPH is set ***

def egraph_rewrite(sink:UOp, sym_pm:PatternMatcher, extra_pm:PatternMatcher|None=None, ctx=None, name:str|None=None) -> UOp:
  """Replace graph_rewrite(sink, sym+extra, ctx) with e-graph extraction for sym, then greedy for the rest."""
  sink = egraph_extract(sink, sym_pm)
  # run greedy with the full combined matcher to catch enabling transformations the e-graph skipped
  combined = sym_pm+extra_pm if extra_pm is not None else sym_pm
  sink = graph_rewrite(sink, combined, ctx=ctx, name=name)
  return sink

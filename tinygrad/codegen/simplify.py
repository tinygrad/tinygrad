from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, graph_rewrite, _substitute
from tinygrad.uop.symbolic import symbolic_flat

def flatten_range(r:UOp):
  off = 2 if r.op is Ops.STORE else 1
  rngs = r.src[off:]
  if not len(rngs): return None
  new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
  return r.replace(src=r.src[:off]+tuple(new_rngs))

pm_flatten_range = PatternMatcher([
  # real ranges only
  (UPat((Ops.REDUCE, Ops.STORE), name="r"), flatten_range),
])

def count_divmod(x:UOp): return len([u for u in x.toposort() if u.op in {Ops.IDIV, Ops.MOD}])
def simplify_merge_adjacent(u:UOp) -> UOp|None:
  i = 2 if u.op is Ops.STORE else 1
  while i < len(u.src)-1:
    r0, r1 = u.src[i], u.src[i+1]
    # check same type
    if r0.arg[1] == r1.arg[1]:
      s0, s1 = r0.src[0], r1.src[0]
      # do the merge
      new_range = r0.replace(src=(s0*s1,))
      nidx = graph_rewrite(u, _substitute+symbolic_flat+pm_flatten_range, ctx={r0:new_range//s1, r1:new_range%s1}, name=f"check_merge_{i}_{i+1}")
      # check if it simplifies
      if count_divmod(nidx) <= count_divmod(u):
        u = nidx
        continue
    i += 1
  return u

pm_simplify_ranges = PatternMatcher([
  (UPat((Ops.STORE, Ops.REDUCE), name="u"), simplify_merge_adjacent),
])
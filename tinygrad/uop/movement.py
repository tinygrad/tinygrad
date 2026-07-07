from tinygrad.uop.ops import PatternMatcher, UPat, Ops

# TODO: pm_mops from rangeify belongs here. this is all pattern matchers that strictly clean up movement ops

mop_cleanup = PatternMatcher([
  # remove noop RESHAPE/PERMUTE/INDEX
  (UPat(Ops.RESHAPE, src=(UPat(name="x2"), UPat()), name="x"), lambda x,x2: x2 if x2._shape is not None and x2.shape == x.shape else None),
  (UPat(Ops.PERMUTE, name="x"), lambda x: x.src[0] if list(x.arg) == list(range(len(x.arg))) else None),
  (UPat(Ops.INDEX, src=(UPat.var('x'),)), lambda x: x),
  # merge adjacent RESHAPES
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE, name="x2"), UPat()), name="x"), lambda x,x2: x.replace(src=(x2.src[0], x.src[1]))),
  # merge PERMUTEs
  (UPat(Ops.PERMUTE, src=(UPat(Ops.PERMUTE, name="x2"),), name="x"), lambda x,x2: x2.replace(arg=tuple(x2.arg[i] for i in x.arg))),
  # STACK on INDEX CONST
  (UPat(Ops.STACK, src=UPat(Ops.INDEX, src=(UPat.var("src"), UPat(Ops.CONST))), name="stk"),
   lambda src,stk: src if stk.shape == src.shape and list(range(len(stk.src))) == [x.src[1].arg for x in stk.src] else None),
  # INDEX on STACK (simple)
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="stk"), UPat(Ops.CONST, name="c"))), lambda stk,c: stk.src[c.arg]),
])

from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, dtypes, GroupOp, AddrSpace

def do_split_ends(e:UOp):
  ret, backedge = e.src[0], tuple(x for x in e.src[1:] if x.dtype in (dtypes.void, dtypes.bool))
  for r in sorted(UOp.sink(*[x for x in e.src[1:] if x not in backedge]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret.end(*backedge) if len(backedge) else ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])

def lower_gated_load(ctx:dict[UOp, dict[UOp, None]], load:UOp, cond:UOp):
  branch = UOp(Ops.IF, dtypes.void, (cond,))
  gated_loads = [x for x in ctx[cond] if x.op is Ops.LOAD]
  in_if_then = tuple(x.replace(src=x.src[:1] + (branch,)) for x in gated_loads)
  in_if_else = tuple(x.src[1] for x in gated_loads)
  if_then = UOp(Ops.THEN, dtypes.void, src=(branch,) + in_if_then)
  if_else = UOp(Ops.ELSE, dtypes.void, src=(branch,) + in_if_else)
  merge = UOp(Ops.ENDIF, dtypes.void, src=(if_then, if_else))
  phi = UOp(Ops.GETTUPLE, load.dtype, (merge,), arg=next(i for i,l in enumerate(gated_loads) if l is load))
  return phi

def lower_gated_store(ctx:dict[UOp, dict[UOp, None]], cond:UOp):
  branch = UOp(Ops.IF, dtypes.void, (cond,))
  gated_stores = [x for x in ctx[cond] if x.op is Ops.STORE and len(x.src) == 3 and x.src[2] is cond]
  in_if_then = tuple(x.replace(src=x.src[:2] + (branch,)) for x in gated_stores)
  if_then = UOp(Ops.THEN, dtypes.void, src=(branch,) + in_if_then)
  if_else = UOp(Ops.ELSE, dtypes.void, src=(branch,))
  merge = UOp(Ops.ENDIF, dtypes.void, src=(if_then, if_else))
  return merge

pm_lower_gated_load_store = PatternMatcher([
  (UPat((Ops.INDEX, Ops.SHRINK)).load(UPat(), UPat.var("cond", dtypes.bool), name="load"), lower_gated_load),
  (UPat((Ops.INDEX, Ops.SHRINK)).store(UPat(), UPat.var("cond", dtypes.bool)), lower_gated_store),
])

# there are 3 relationships between ranges:
# nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
# dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
# independent, endrange y is not a dependency of endrange x
# everything is nested inside the sink
class CFGContext2:
  def __init__(self, sink:UOp):
    topo = sink.toposort()
    params = tuple(sorted([u for u in topo if u.op is Ops.PARAM], key=lambda x: x.arg.slot))
    func_name = sink.arg.function_name if sink.arg is not None else "test"
    self.start = start = UOp(Ops.START, dtypes.void, params, func_name)

    def _entry(x:UOp) -> UOp:
      if x.op is Ops.END: return x.src[1]
      if x.op is Ops.ENDIF: return x.src[0].src[0]
      if x.op is Ops.SINK: return start

    deps: dict[UOp, dict[UOp, None]] = {start: {start: None}}
    nesting: dict[UOp, UOp] = {}
    for u in topo:
      # get the deps from the src
      deps[u] = {start: None}
      for s in u.src: deps[u] |= deps[s]

      if u.op in (Ops.END, Ops.ENDIF, Ops.SINK): nesting |= {x:u for x in deps[u] if _entry(u) in deps[x] and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF): deps[u][u] = None

    self.idom: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # ranges that have dependencies on other siblings need to be scheduled after them
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      for x,y in zip(order, order[1:] + [k]): self.idom[y if y is k else _entry(y)] = x

def lower_range(ctx:CFGContext2, x:UOp) -> UOp|None:
  idom = ctx.idom.get(x)
  if idom is None: return None
  arg = UOp(Ops.CONST, dtypes.int32, (), 0)
  # if the loop might not run at all we must gate it
  if x.src[0].vmin < 1:
    cond = UOp(Ops.CMPLT, dtypes.bool, (arg, x.src[0]))
    gate = UOp(Ops.IF, dtypes.void, (idom, cond))
    idom = UOp(Ops.THEN, dtypes.void, (gate,))
  return UOp(Ops.RANGE, dtypes.void, (idom, arg), x.arg)

def lower_end(ctx:CFGContext2, x:UOp) -> UOp|None:
  idom = ctx.idom.get(x)
  if idom is None: return None
  rng = x.src[1]
  inc = UOp(Ops.ADD, rng.dtype, (rng, UOp(Ops.CONST, rng.dtype, arg=1)))
  cond = UOp(Ops.CMPLT, dtypes.bool, (inc, rng.src[0]))
  rest = () if x.src[0].op is Ops.END else x.src[0].src if x.src[0].op is Ops.GROUP else (x.src[0],)
  end = UOp(Ops.END, dtypes.void, (idom, rng, cond, inc) + rest)
  # if the loop might not run at all we must gate it
  if rng.src[0].vmin < 1:
    cond = UOp(Ops.CMPLT, dtypes.bool, (UOp(Ops.CONST, dtypes.int32, (), 0), rng.src[0]))
    gate = UOp(Ops.IF, dtypes.void, (ctx.idom[rng], cond))
    if_else = UOp(Ops.ELSE, dtypes.void, (gate,))
    return UOp(Ops.ENDIF, dtypes.void, (end, if_else))
  return end

pm_add_control_flow2 = PatternMatcher([
  (UPat(Ops.END, name="x"), lower_end),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat((Ops.IF, Ops.SINK), name="x"), lambda ctx,x: x.replace(src=(y,)+x.src) if (y:=ctx.idom.get(x)) is not None else None),
  # the uses of RANGE that aren't control are really using the iterator
  (UPat(GroupOp.All - GroupOp.Control - {Ops.GETTUPLE, Ops.AFTER}, name="x"), lambda x:
   x.replace(src=tuple(s if s.op is not Ops.RANGE else UOp(Ops.GETTUPLE, s.dtype, (s,), 0) for s in x.src))),
   # the uses of PARAM now use its projection
  (UPat(GroupOp.All - {Ops.START, Ops.GETTUPLE}, name="x"), lambda ctx,x:
   x.replace(src=tuple(s if s.op is not Ops.PARAM else UOp(Ops.GETTUPLE, s.dtype, (ctx.start,), s.arg.slot) for s in x.src))),
])

class LowerRegBufferContext:
  def __init__(self, sink:UOp):
    self.regbuf_init: dict[tuple[UOp, UOp], UOp] = {}
    self.rng_args: dict[UOp, dict[tuple[UOp, UOp], UOp]] = {}
    self.end_args: dict[UOp, dict[tuple[UOp, UOp], UOp]] = {}

    def after_rng(u:UOp): return u.op is Ops.AFTER and u.src[0].op is Ops.BUFFER and u.src[0].addrspace is AddrSpace.REG and u.src[1].op is Ops.STORE

    for u in sink.toposort():
      if after_rng(u):
        for s in u.src:
          if s.op is Ops.STORE: self.regbuf_init[(u.src[0], s.src[0].src[1])] = s.src[1]

      if u.op is Ops.STORE and after_rng(u.src[0].src[0]):
        after = u.src[0].src[0]
        buf = after.src[0]
        arg = self.regbuf_init[(buf, u.src[0].src[1])]
        idx = (buf, u.src[0].src[1])
        rngs = [s for s in after.src if s.op is Ops.RANGE]
        for i,rng in enumerate(rngs):
          if i == 0: self.rng_args.setdefault(rng, {})[idx] = arg
          else: self.rng_args.setdefault(rng, {})[idx] = UOp(Ops.GETTUPLE, arg.dtype, (rngs[i-1],), len(self.rng_args[rngs[i-1]]))
          self.end_args.setdefault(rng, {})[idx] = u.src[1]

def lower_end_args(ctx:LowerRegBufferContext, end:UOp) -> UOp|None:
  if end.src[1] not in ctx.end_args: return None
  return end.replace(src=end.src[:4] + tuple(ctx.end_args[end.src[1]].values()))

def lower_range_args(ctx:LowerRegBufferContext, rng:UOp) -> UOp|None:
  if rng not in ctx.rng_args: return None
  return rng.replace(src=rng.src + tuple(ctx.rng_args[rng].values()))

# the after/index/load sequence after the END is removed and the updated value is accessed directly
def lower_load_after(ctx:LowerRegBufferContext, buf:UOp, end:UOp, ofst:UOp) -> UOp|None:
  if buf.addrspace is not AddrSpace.REG: return None
  val = ctx.end_args[end.src[1]].get((buf, ofst))
  # if value isn't updated we access the init directly, no block arg is created
  return ctx.regbuf_init[(buf, ofst)] if val is None else val

# the buffer/index/store/after/index/load sequence is removed and the block argument is accessed directly
def lower_load_update(ctx:LowerRegBufferContext, load:UOp, ofst:UOp, buf:UOp, after:UOp) -> UOp|None:
  if buf.addrspace != AddrSpace.REG: return None
  inner_rng = after.src[-1]
  return UOp(Ops.GETTUPLE, load.dtype, (inner_rng,), 1+next(i for i,idx in enumerate(ctx.rng_args[inner_rng]) if idx == (buf, ofst)))

after_end = UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf"), UPat(Ops.END, name="end")))
after_endif = UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf"), UPat(Ops.ENDIF, src=(UPat(Ops.END, name="end"),), allow_any_len=True)))
after_range = UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf"),), allow_any_len=True, name="after")

pm_lower_reg_buffer = PatternMatcher([
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_end, UPat.cvar("ofst"))).load(), lower_load_after),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_endif, UPat.cvar("ofst"))).load(), lower_load_after),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_range, UPat.cvar("ofst"))).load(name="load"), lower_load_update),
  (UPat(Ops.RANGE, name="rng"), lower_range_args),
  (UPat(Ops.END, name="end"), lower_end_args),
])

# TODO: must handle gated range, in that case the endif must merge the updated arg inside END and the init arg in ELSE

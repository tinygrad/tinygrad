from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, dtypes, GroupOp, AddrSpace, identity_element

def do_split_ends(e:UOp):
  ret = e.src[0]
  for r in sorted(UOp.sink(*e.src[1:]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])

# TODO: load/store need to be pinned to the branch because the scheduler isn't free to pull them out, use Ops.AFTER
def lower_gated_load(load:UOp, addr:UOp, alt:UOp, cond:UOp):
  load = load.replace(src=(addr,))
  branch = UOp(Ops.IF, cond.dtype, (cond,))
  true_block = UOp(Ops.BLOCKEND, dtypes.void, src=(branch, load), arg=0)
  false_block = UOp(Ops.BLOCKEND, dtypes.void, src=(branch, alt), arg=1)
  merge = UOp(Ops.ENDIF, dtypes.void, src=(true_block, false_block))
  out_proj = UOp(Ops.GETTUPLE, load.dtype, (merge,), arg=0)
  return out_proj

def lower_gated_store(store:UOp, addr:UOp, val:UOp, cond:UOp):
  store = store.replace(src=(addr, val))
  branch = UOp(Ops.IF, cond.dtype, (cond,))
  true_block = UOp(Ops.BLOCKEND, dtypes.void, src=(branch, store), arg=0)
  false_block = UOp(Ops.BLOCKEND, dtypes.void, src=(branch,), arg=1)
  merge = UOp(Ops.ENDIF, dtypes.void, src=(true_block, false_block))
  return merge

pm_lower_gated_load_store = PatternMatcher([
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(UPat.var("alt"), UPat.var("cond", dtypes.bool), name="load"), lower_gated_load),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(UPat.var("val"), UPat.var("cond", dtypes.bool), name="store"), lower_gated_store),
])

# there are 3 relationships between ranges:
# nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
# dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
# independent, endrange y is not a dependency of endrange x
# everything is nested inside the sink
class CFGContext:
  def __init__(self, sink:UOp):
    topo = sink.toposort()
    params = tuple(u for u in topo if u.op is Ops.PARAM)
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
      #print(k.op, [(o.op, o.arg) for o in order])
      for x,y in zip(order, order[1:] + [k]): self.idom[y if y is k else _entry(y)] = x
      # TODO: this can happen! it causes infinite loop in shufflenet
      #assert y.src[1] not in x.backward_slice_with_self

def lower_range(ctx:CFGContext, x:UOp) -> UOp|None:
  if x not in ctx.idom: return None
  it = UOp(Ops.CONST, dtypes.int32, (), 0)
  rng = UOp(Ops.RANGE, dtypes.void, (ctx.idom[x], it), x.arg)
  return rng

def lower_end(ctx:CFGContext, x:UOp) -> UOp|None:
  if x not in ctx.idom: return None
  rng = x.src[1]
  inc = UOp(Ops.ADD, rng.dtype, (rng, UOp(Ops.CONST, rng.dtype, arg=1)))
  cond = UOp(Ops.CMPLT, dtypes.bool, (inc, rng.src[0]))
  end = UOp(Ops.END, dtypes.void, (ctx.idom[x], rng, cond, inc, x.src[0]))
  return end

pm_add_control_flow = PatternMatcher([
  # the uses of RANGE that aren't control flow are really using the iterator
  (UPat(GroupOp.All - GroupOp.ControlFlow - {Ops.GETTUPLE, Ops.AFTER}, name="x"), lambda x:
   x.replace(src=tuple(s if s.op is not Ops.RANGE else UOp(Ops.GETTUPLE, s.dtype, (s,), 0) for s in x.src))),
  (UPat(GroupOp.All - {Ops.START, Ops.GETTUPLE}, name="x"), lambda ctx,x:
   x.replace(src=tuple(s if s.op is not Ops.PARAM else UOp(Ops.GETTUPLE, s.dtype, (ctx.start,), s.arg.slot) for s in x.src))),
  (UPat(Ops.END, name="x"), lower_end),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat((Ops.IF, Ops.SINK), name="x"), lambda ctx,x: x.replace(src=(y,)+x.src) if (y:=ctx.idom.get(x)) is not None else None),
])

# the after/index/load sequence after the END is removed and the updated value is accessed directly
def lower_load_after(buf:UOp, end:UOp) -> UOp|None:
  if buf.addrspace != AddrSpace.REG: return None
  return end.src[4].src[1]

# the store inside END that updates buf is removed
def lower_store_update(update:UOp, buf:UOp, rng:UOp) -> UOp|None:
  if buf.addrspace != AddrSpace.REG: return None
  return update

# the buffer/index/store/after/index/load sequence is removed and the block argument is accessed directly
def lower_load_update(ctx:dict[UOp, list[UOp]], load:UOp, buf:UOp, rng:UOp) -> UOp|None:
  if buf.addrspace != AddrSpace.REG: return None
  arg_init = load.src[0].src[0].src[1].src[1]
  if rng not in ctx: ctx[rng] = []
  arg_pos = len(rng.src[1:]) + len(ctx[rng])
  ctx[rng].append(arg_init)
  proj = UOp(Ops.GETTUPLE, load.dtype, (rng,), arg_pos)
  return proj

def lower_range_args(ctx:dict[UOp, list[UOp]], rng:UOp) -> UOp|None:
  if rng not in ctx: return None
  return rng.replace(src=rng.src + tuple(ctx[rng]))

after_end = UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf"), UPat(Ops.END, name="end")))
after_range = UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, name="buf"), UPat(), UPat(Ops.RANGE, name="rng")))

pm_lower_reg_buffer = PatternMatcher([
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_end, UPat())).load(), lower_load_after),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_range, UPat())).store(UPat.var("update")), lower_store_update),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(after_range, UPat())).load(name="load"), lower_load_update),
  (UPat(Ops.RANGE, name="rng"), lower_range_args),
])

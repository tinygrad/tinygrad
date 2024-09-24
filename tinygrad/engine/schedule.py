import sys, pickle, atexit, importlib, contextlib
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Optional, DefaultDict, cast, get_args
from tinygrad.ops import REDUCE_ALU, MetaOps, ReduceOps, UNSAFE_PAD_OPS, UnaryOps, UOp, UOps, PatternMatcher, UPat, graph_rewrite
from tinygrad.engine.graph import log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, MULTIOUTPUT, SAVE_SCHEDULE, FUSE_CONV_BW, FUSE_ARANGE, AST_REWRITE, \
                             GlobalCounters, all_same, colored, prod, dedup, all_int, merge_dicts, getenv, Metadata, unwrap
from tinygrad.shape.symbolic import Variable, sint
from tinygrad.dtype import ConstType, ImageDType, PtrDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.shape.view import View, strides_for_shape

# creation can recurse a lot
sys.setrecursionlimit(10000)

# *** ScheduleItem return type ***

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[Tuple[Metadata, ...]] = None
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is UOps.SINK else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is UOps.SINK else self.bufs[1:]

@dataclass(frozen=True)
class LBScheduleItem:
  ast: UOp
  bufs: Tuple[LazyBuffer, ...]
  metadata: Optional[Tuple[Metadata, ...]] = None
  @property
  def outputs(self) -> Tuple[LazyBuffer, ...]: return self.bufs[:len(self.ast.src)] if self.ast.op is UOps.SINK else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[LazyBuffer, ...]: return self.bufs[len(self.ast.src):] if self.ast.op is UOps.SINK else self.bufs[1:]

# *** UOp with SWIZZLE (movementops) rewriting to UOp we can index ***

# ** helpers for doing movementops on uops

def st_fixup(u:UOp, apply_to_st:Callable[[ShapeTracker], ShapeTracker], cache:Dict[UOp, UOp]) -> UOp:
  if (n:=cache.get(u)): return n
  if u.op is UOps.SHAPETRACKER:
    new_st = apply_to_st(u.arg)
    return u if u.arg == new_st else UOp(UOps.SHAPETRACKER, dtypes.void, (), new_st)
  if len(u.src) == 0 or (u.st is not None and u.st == apply_to_st(u.st)): return u
  new_srcs = tuple(st_fixup(x, apply_to_st, cache) for x in u.src)
  cache[u] = ret = u if new_srcs == u.src else UOp(u.op, u.dtype, new_srcs, u.arg)
  return ret

def permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis)+axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

# ** reduceop fusor

def push_swizzle_up_through_reduce(swizzle:UOp, reduceop:UOp) -> Optional[UOp]:
  if (swizzle_st:=unwrap(swizzle.st)).contiguous: return None
  rsrc = reduceop.src[0]
  tmp, rshape = permute_reduce(ShapeTracker.from_shape(unwrap(rsrc.st).shape), reduceop.arg[1])
  prshape = prod(rshape)
  strides = strides_for_shape(rshape)
  nv: List[View] = []
  for v in swizzle_st.views:
    nv.append(View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                          v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None))
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  _, new_rshape = permute_reduce(new_input_st, reduceop.arg[1])
  new_axis = tuple(range(len(new_input_st.shape)-len(new_rshape), len(new_input_st.shape)))
  return UOp(UOps.REDUCE_AXIS, reduceop.dtype, (st_fixup(rsrc, lambda st:st+new_input_st, {}),),
             (reduceop.arg[0], new_axis)).swizzle(ShapeTracker.from_shape(swizzle_st.shape))

def push_swizzle_down_through_reduce(root:UOp, swizzle:UOp) -> UOp:
  swizzle_st = unwrap(swizzle.st)
  assert swizzle_st.contiguous, "can't push a non contiguous SWIZZLE down to STORE"
  assert prod(swizzle_st.shape) == prod(unwrap(swizzle.src[0].st).shape), "can't push expands down to STORE"
  return UOp(UOps.REDUCE_AXIS, root.dtype, swizzle.src, root.arg).swizzle(ShapeTracker.from_shape(swizzle_st.reduce(root.arg[1])))

def push_swizzle_down_through_elementwise(root:UOp) -> Optional[UOp]:
  swizzles = [x for x in root.src if x.op is UOps.SWIZZLE]
  if len(swizzles) == 0: return None
  swizzle_shapes = [(unwrap(x.st).shape, unwrap(x.src[0].st).shape) for x in swizzles]
  assert all_same([(x, prod(x), prod(y)) for x,y in swizzle_shapes]), f"swizzles must have the same size {swizzle_shapes}"
  new_shape, new_input_shape = swizzle_shapes[0]
  fixup_cache: Dict[UOp, UOp] = {}
  new_srcs = [x.src[0] if x.op is UOps.SWIZZLE else st_fixup(x, lambda st:st.reshape(new_input_shape), fixup_cache) for x in root.src]
  ret = UOp(root.op, root.dtype, tuple(new_srcs), root.arg)
  return ret if ret.op is UOps.STORE else ret.swizzle(ShapeTracker.from_shape(new_shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is UOps.REDUCE_AXIS for x in first_reduce.parents), "can't merge more than two reduceops at a time"
  new_axis: Tuple[int, ...] = root.arg[1]+first_reduce.arg[1]
  return UOp(UOps.REDUCE_AXIS, first_reduce.dtype, first_reduce.src, (first_reduce.arg[0], new_axis))

reduceop_fusor = PatternMatcher([
  # push a SWIZZLE up to LOAD, through a reduce (eg. expands)
  (UPat(UOps.SWIZZLE, src=(UPat(UOps.REDUCE_AXIS, name="reduceop"),), name="swizzle"), push_swizzle_up_through_reduce),
  # push a SWIZZLE down to STORE, through a reduce (ONLY reshapes)
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.SWIZZLE, name="swizzle"),), name="root"), push_swizzle_down_through_reduce),
  # push SWIZZLE(s) down to STORE, through an elementwise op (ONLY reshapes)
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.STORE), name="root"), push_swizzle_down_through_elementwise),
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

def full_ast_rewrite(sink:UOp) -> UOp:
  if not AST_REWRITE: return sink
  return graph_rewrite(sink, reduceop_fusor)

# *** List[LazyBuffer] lowering to ScheduleItem ***

def _recursive_uop(buf:LazyBuffer, st:ShapeTracker, outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], inputs:List[LazyBuffer],
                      realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                      cache:Dict[Tuple[LazyBuffer, ShapeTracker], UOp]) -> UOp:
  """recursively create a UOp"""
  if buf is not buf.base: st, buf = buf.st+st, buf.base
  if (buf, st) in cache: return cache[(buf, st)]
  assert buf.op is not None, "base must be a base itself"
  dtype = buf.dtype.base if isinstance(buf.dtype, ImageDType) else buf.dtype

  # buffer ops define ShapeTracker
  if buf.realized is not None or (buf in realizes and buf not in outputs):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    # if it's a const, we generate it
    if buf.op is MetaOps.CONST:
      if isinstance(val:=buf.arg, Variable):
        val, var_val = val.unbind()
        var_vals[val] = var_val
      else: assert isinstance(val, get_args(ConstType)), f"cannot create ConstBuffer with value {val}"
      return UOp(UOps.VALID, dtypes.bool, (unbound_st.to_uop(),)).where(UOp.const(dtype, val), UOp.const(dtype, 0))
    # otherwise, it's a load and we add it to the inputs
    if buf in assign_targets and not (unbound_st.contiguous or (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and \
        ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask))):
      # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                           +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
    if buf not in assign_targets and buf not in inputs: inputs.append(buf)
    ubuf = UOp(UOps.DEFINE_GLOBAL, buf.dtype if isinstance(buf.dtype, ImageDType) else PtrDType(buf.dtype), (),
               outputs.index(assign_targets[buf]) if buf in assign_targets else len(outputs)+inputs.index(buf))
    return UOp(UOps.LOAD, dtype, (ubuf, unbound_st.to_uop()))

  # reduce ops change ShapeTracker
  if buf.op in ReduceOps:
    rsrc = _recursive_uop(buf.srcs[0], ShapeTracker.from_shape(buf.srcs[0].shape), outputs, var_vals, inputs, realizes, assign_targets, cache)
    return cache.setdefault((buf, st), UOp(UOps.REDUCE_AXIS, dtype, (rsrc,), (REDUCE_ALU[cast(ReduceOps, buf.op)], buf.arg)).swizzle(st))

  # elementwise ops pass shapetracker
  in_uops = tuple(_recursive_uop(x, st, outputs, var_vals, inputs, realizes, assign_targets, cache) for x in buf.srcs)
  if buf.op is MetaOps.CONTIGUOUS:
    assert buf in outputs, f"{buf.op} must be writable"
    return in_uops[0]
  if buf.op is MetaOps.ASSIGN: return cache.setdefault((buf, st), UOp(UOps.ASSIGN, dtype, (in_uops[1].src[0], in_uops[0])))
  if buf.op is UnaryOps.CAST: return cache.setdefault((buf, st), UOp(UOps.CAST, dtype, in_uops))
  if buf.op is UnaryOps.BITCAST: return cache.setdefault((buf, st), UOp(UOps.BITCAST, dtype, in_uops))
  return cache.setdefault((buf, st), UOp(UOps.ALU, dtype, in_uops, buf.op))

def _lower_lazybuffer(outs:List[LazyBuffer], realizes:Dict[LazyBuffer, None]) -> Tuple[LBScheduleItem, Dict[Variable, int]]:
  """describe the computation for a LazyBuffer with UOp + inputs + var_vals"""
  if (out:=outs[0]).op in {MetaOps.CUSTOM, MetaOps.COPY, MetaOps.EMPTY, MetaOps.VIEW}:
    return LBScheduleItem(UOp(UOps.EXT, out.dtype, (), (out.op, out.arg)), (out,)+tuple(x.base for x in out.srcs)), {}
  # create the stores
  var_vals = merge_dicts([out.st.var_vals.copy() for out in outs])
  assign_targets = {x.srcs[1]:x for x in outs if x.op is MetaOps.ASSIGN}
  cache: Dict[Tuple[LazyBuffer, ShapeTracker], UOp] = {}
  ast: List[UOp] = []
  inputs: List[LazyBuffer] = []
  for i, out in enumerate(outs):
    src = _recursive_uop(out, output_st:=ShapeTracker.from_shape(out.shape), tuple(outs), var_vals, inputs, realizes, assign_targets, cache=cache)
    if out.op is MetaOps.ASSIGN and out.arg:
      assert out.arg[0].shape == out.shape, f"ASSIGN must not override output shape {out.arg[0].shape} != {out.shape}"
      output_st = out.arg[0]
    output_st, vv = output_st.simplify().unbind()
    var_vals.update(vv)
    ubuf = UOp(UOps.DEFINE_GLOBAL, out.dtype if isinstance(out.dtype, ImageDType) else PtrDType(out.dtype), (), i)
    ast.append(UOp(UOps.STORE, dtypes.void, (ubuf, output_st.to_uop(), src)))
  sink = full_ast_rewrite(ast[0].sink(*ast[1:]))
  return LBScheduleItem(sink, tuple(outs+inputs), tuple(dedup([x[0].metadata for x in cache if x[0].metadata and x[0] not in inputs]))), var_vals

# *** DAG creation: decide which LazyBuffers should realize ***

def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],
                children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
  """recursively search the entire graph for all LazyBuffers, insert realizes after expands"""
  if buf in allbufs or buf.base.realized is not None: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  # check if we need to realize views
  if buf is not buf.base:
    # fuse some pads
    if len(buf.st.views) == 1 and buf.st.views[-1].mask is not None and all_int(buf.base.st.shape) and \
        prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
      simple_pads[buf.base] = None
    # realize all expands
    elif prod(buf.base.st.shape) < prod(buf.st.shape):
      # this was causing "test_lil_model" to fail
      if buf.base.op is UnaryOps.CAST and isinstance(buf.base.srcs[0].dtype, ImageDType) and isinstance(buf.base.arg, ImageDType):
        simple_pads[buf.base] = None # don't realize image to image casts. this is part of a larger problem
      else: realizes[buf.base] = None
    # check all other pads for safe fusion
    elif any(v.mask is not None for v in buf.st.views): simple_pads[buf.base] = None
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)
  if buf.op in ReduceOps and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None
  allbufs[buf] = None
  if buf.forced_realize or buf.op in MetaOps: realizes[buf] = None
  if buf.op is MetaOps.ASSIGN:
    assert buf.srcs[1].base is buf.srcs[1], f"assign must be to base {buf.srcs[1]}"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    assign_targets[buf.srcs[1]] = buf
  if buf.op is MetaOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes[buf.srcs[0].base] = None
  if buf.op is MetaOps.VIEW: realizes[buf.srcs[0].base] = None
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)

def _is_padding_okay(buf:LazyBuffer, realizes:Dict[LazyBuffer, None]) -> bool:
  if buf in realizes or buf.realized is not None: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def _recursive_group(tr:LazyBuffer, st:ShapeTracker, r:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],
                     realizes:Dict[LazyBuffer, None], reduce_for_op:Dict[LazyBuffer, LazyBuffer], group:Dict[LazyBuffer, None],
                     cache:Dict[Tuple[LazyBuffer, ShapeTracker], None]) -> None:
  """recursively search the LazyBuffer for groupable children, realize the LazyBuffer if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != r.st.size or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if tr_next.op in ReduceOps: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next.srcs if s.base == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+st_childs[0].st, r, children, realizes, reduce_for_op, group, cache)

def _get_isolated_children(r:LazyBuffer, reduce_for_op:Dict[LazyBuffer, LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],\
    realizes:Dict[LazyBuffer, None], group:Dict[LazyBuffer, None]) -> Dict[LazyBuffer, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op in ReduceOps: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[LazyBuffer, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def _get_output_groups(outs:List[LazyBuffer]) -> \
  Tuple[DefaultDict[LazyBuffer, List[LazyBuffer]],  # these are the output groups
        Dict[LazyBuffer, None],                     # these are all the realizes in the graph
        Dict[LazyBuffer, LazyBuffer]]:              # these are the buffers we ASSIGN to in this schedule
  """find all the realizes in the graph, group the output LazyBuffers into kernels."""
  # start by just realizing the buffers passed in
  realizes: Dict[LazyBuffer, None] = {x.base:None for x in outs if x.base.realized is None}
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Dict[LazyBuffer, None] = {}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  double_reduces: Dict[LazyBuffer, None] = {}
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes[p] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  reduce_of_const: List[LazyBuffer] = []
  for r in allbufs:
    if r.op not in ReduceOps or r in realizes: continue

    group: Dict[LazyBuffer, None] = {}
    _recursive_group(r, r.st, r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).realized or p in realizes:
          if p in assign_targets and assign_targets[p] not in group: forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(s for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg.itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes[tr] = None
    else: reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r.op is ReduceOps.SUM and r.srcs[0].base.op is MetaOps.CONST: reduce_of_const.append(r)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = reduceop.base.srcs[0].base
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for r in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is r}
    if DEBUG_ARANGE:=(getenv("DEBUG_ARANGE")): print(f"checking {r} {group=}")
    if any(tr.forced_realize for tr in group) or any(x.base in group for x in outs): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {MetaOps.COPY, MetaOps.VIEW}}
    if len(kernel_children) == 0: continue
    if DEBUG_ARANGE: print(colored(f"folding {r}", "green"))
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  for buf in realizes:
    if buf.realized is not None or buf.op is MetaOps.CONST: continue
    output_groups[reduce_for_op[buf] if buf in reduce_for_op and MULTIOUTPUT else buf].append(buf)

    # make things that can't be images not images
    if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                              not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
      if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
      buf.dtype = dtypes.float32
      # hack the underlying buffer too
      if buf.base is buf:
        assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
        buf.buffer.dtype = dtypes.float32
        buf.buffer.options = None
  return output_groups, realizes, assign_targets

SCHEDULES: List[Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]], DefaultDict[LBScheduleItem, int]]] = []
def _graph_schedule(outs:List[LazyBuffer]) -> \
  Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]],  # this is the graph
        DefaultDict[LBScheduleItem, int], # this is the in-degree of the graph
        Dict[Variable, int]]: # this has all the var values of the schedule
  """create a graph for realizing the outputs"""
  output_groups, realizes, assign_targets = _get_output_groups(outs)
  # preschedule all buffers in realizes
  prescheduled: List[LBScheduleItem] = []
  var_vals: Dict[Variable, int] = {}
  for group in output_groups.values():
    prescheduled.append((ret:=_lower_lazybuffer(group, realizes))[0])
    var_vals = merge_dicts([var_vals, ret[1]])
  schedule_targets = {out:lsi for lsi in prescheduled for out in lsi.outputs}

  graph: DefaultDict[LBScheduleItem, List[LBScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[LBScheduleItem, int] = defaultdict(int)
  for lsi in prescheduled:
    if lsi not in in_degree: in_degree[lsi] = 0
    # realize outputs after all parents are realized
    scheduled_parents = dedup(schedule_targets[x] for x in lsi.inputs if x in schedule_targets)
    for x in scheduled_parents:
      graph[x].append(lsi)
      in_degree[lsi] += 1
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(schedule_targets[assign_targets[x]] for x in lsi.inputs if x in assign_targets)
    for assign in parents_assigns:
      graph[lsi].append(assign)
      in_degree[assign] += 1

  if SAVE_SCHEDULE:
    def _save():
      print(f"saving {len(SCHEDULES)} schedule graphs to", fp:=getenv("SAVE_SCHEDULE_PATH", "schedule.pkl"))
      with open(fp, "wb") as f: pickle.dump(SCHEDULES, f)
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.append((graph, in_degree))
  return graph, in_degree, var_vals

# *** DAG ordering: breadth first search ***

def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  graph, in_degree, var_vals = _graph_schedule(outs)
  if getenv("RUN_PROCESS_REPLAY") and getenv("COMPARE_SCHEDULE", 1):
    # NOTE: process relpay needs PYTHONPATH=., remove this once it just pickles LazyBuffers
    with contextlib.suppress(Exception): importlib.import_module("test.external.process_replay.diff_schedule").process_replay(outs, graph, in_degree)

  queue = deque(lsi for lsi,deg in in_degree.items() if deg == 0)
  schedule: List[ScheduleItem] = []
  kernel_number = GlobalCounters.kernel_count
  while queue:
    lsi = queue.popleft()
    if GRAPH:
      kernel_number += 1
      for out in lsi.outputs: realized_lazybuffer(out, kernel_number)
    for out in lsi.outputs: del out.srcs  # can only schedule once
    schedule.append(ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.bufs if x.size != 0), lsi.metadata))
    for x in graph[lsi]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if any(degree != 0 for degree in in_degree.values()) or len(in_degree) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(in_degree)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, var_vals

def create_schedule(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule

# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Optional, Callable
from tinygrad.helpers import merge_dicts, getenv, prod, all_int, flatten
from tinygrad.shape.view import View, strides_for_shape, unravel, _reshape_mask
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops, graph_rewrite, split_uop, symbolic_flat, Variable, sint, uop_given_valid, simplify_valid, sint_to_uop, Context
from tinygrad.codegen.rewriter import sym
import numpy as np
import highspy

highs_inst = highspy.Highs()

def st_to_highs(st, idxs) -> tuple:
  # add necessary auxiliary variables to highs_inst and return:
  # (linear expression for idx, [constrs for valid], [constrs whose disjunction gives invalid])
  # i.e. all the constrs in valid must be true to be valid, and one of the constrs in invalid must be true to be invalid
  mask = st.views[-1].mask or tuple((0,sh) for sh in st.views[-1].shape)
  valid = [mask[k][0] <= idxs[k] <= mask[k][1]-1 for k in range(len(idxs))]
  invalid = sum(([idxs[k] <= mask[k][0]-1, idxs[k] >= mask[k][1]] for k in range(len(idxs))), start=[])
  for j in range(len(st.views)-1, 0, -1):
    mask = st.views[j-1].mask or tuple((0,sh) for sh in st.views[j-1].shape)
    A = np.array([unravel(st.views[j-1].shape, st.views[j].strides[k]) for k in range(len(st.views[j].strides))]).T
    K = np.diag(st.views[j-1].shape) + np.diag([-1]*(len(st.views[j-1].shape)-1), 1)
    u = [highs_inst.addVariable(lb=-np.inf, ub=np.inf, type=highspy.HighsVarType.kInteger) for k in range(len(st.views[j-1].shape))]
    idxs = A@idxs + K@u + unravel(st.views[j-1].shape, st.views[j].offset)
    for k in range(len(idxs)):
      highs_inst.addConstr(0 <= idxs[k] <= st.views[j-1].shape[k]-1)
      valid.append(mask[k][0] <= idxs[k] <= mask[k][1]-1)
      invalid += [idxs[k] <= mask[k][0]-1, idxs[k] >= mask[k][1]]
  idx = np.array(st.views[0].strides)@idxs + st.views[0].offset
  return idx, valid, invalid

def solve_highs(constrs, min_obj) -> Optional[int]:
  # None if the constaints are infeasible, otherwise returns the value of the first variable (x)
  cs = [highs_inst.addConstr(c) for c in constrs]
  highs_inst.minimize(min_obj)
  if highs_inst.getModelStatus() == highspy.HighsModelStatus.kTimeLimit: raise RuntimeError("highs timeout")
  assert highs_inst.getModelStatus() in {highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kInfeasible}, \
    "highs returned something other than Infeasible or Optimal"
  if highs_inst.getModelStatus() == highspy.HighsModelStatus.kInfeasible: ret = None
  else: ret = round(highs_inst.getSolution().col_value[0])
  for _ in range(len(cs)): highs_inst.removeConstr(cs[0].index)  # weird constr indexing behaviour
  return ret

@functools.lru_cache(maxsize=None)
def collapse_st(st:ShapeTracker, keep_shape:bool=False) -> Optional[View]:
  try:
    return _collapse_st(st, keep_shape)
  except RuntimeError:
    print(f"highs timeout with {st}, {keep_shape=}")
    return None

def _collapse_st(st:ShapeTracker, keep_shape:bool=False) -> Optional[View]:
  # Look for a view whose action is equiv. to st; this is complete
  if len(st.views) == 0: raise ValueError()
  if len(st.views) == 1: return st.views[0]
  # TODO work symbolic offset into this
  if not all_int(flatten(((*v.shape, *v.strides, v.offset, *flatten(v.mask or ((0,),))) for v in st.views))):
#    import pdb; pdb.set_trace()
    return None
  highs_inst.clear(); highs_inst.setOptionValue("time_limit", 1); highs_inst.setOptionValue("log_to_console", False)
  # presolve sometimes returns "infeasible" incorrectly
  highs_inst.setOptionValue("presolve", "off")
  highs_inst.setOptionValue("mip_feasibility_tolerance", 1e-10)

#  import pdb; pdb.set_trace()
#  print(st)

  x = highs_inst.addVariable(0, prod(st.shape)-1, type=highspy.HighsVarType.kInteger)
  in_idxs = [highs_inst.addVariable(0, st.shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(st.shape))]
  highs_inst.addConstr(sum(in_idxs[k]*prod(st.shape[k+1:]) for k in range(len(st.shape))) == x)
  idx, valid, invalid = st_to_highs(st, in_idxs)
  valid_start = solve_highs(valid, x)
  if valid_start is None: return View.create(st.shape, (0,)*len(st.shape), 0, ((0,0),)*len(st.shape))

  # find mask, TODO skip this if all points are valid
  shape, mask, denom = [], [], 1
  x_step = highs_inst.addVariable(-np.inf, np.inf, type=highspy.HighsVarType.kInteger)
  while denom < prod(st.shape):
    a = min(solve_highs([constr] + [x==valid_start+denom*x_step, x>=valid_start], x) or prod(st.shape) for constr in invalid)
    b = solve_highs(valid + [x==valid_start+denom*x_step, x>=a], x)
    shape.append((b-valid_start)//denom if b is not None else prod(st.shape)//denom)
    mask.append([valid_start//denom%shape[-1], a//denom%shape[-1]])
    if mask[-1][1] == 0: mask[-1][1] = shape[-1]
    denom *= shape[-1]
    if prod(st.shape) % denom != 0: return None
  shape, mask = shape[::-1], mask[::-1]
  # new indexes based on the found shape
  new_in_idxs = [highs_inst.addVariable(0, shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(shape))]
  new_in_constr = [sum(new_in_idxs[k]*prod(shape[k+1:]) for k in range(len(shape))) == x]
  new_valid = [mask[k][0] <= new_in_idxs[k] <= mask[k][1]-1 for k in range(len(shape))]
  new_invalid = sum(([new_in_idxs[k] <= mask[k][0]-1, new_in_idxs[k] >= mask[k][1]] for k in range(len(shape))), start=[])
  # check all points in new mask are valid, and all valid points are in new mask
  if any(solve_highs(new_in_constr + new_valid + [constr], x) is not None for constr in invalid): return None
  if any(solve_highs(new_in_constr + valid + [constr], x) is not None for constr in new_invalid): return None
  
  # find strides
#  import pdb; pdb.set_trace()
  while True:
    if len(shape) > 20:  # TODO remove
      import pdb; pdb.set_trace()
    # get strides for current shape
    strides = []
    out_idx_at_valid_start = st.to_indexed_uops(unravel(st.shape, valid_start))[0].simplify().arg
    running_product = [prod(shape[j+1:]) for j in range(len(shape))]
    for p in running_product:
      strides.append(st.to_indexed_uops(unravel(st.shape, valid_start+p))[0].simplify().arg - out_idx_at_valid_start)
    offset = out_idx_at_valid_start - sum(strides[j]*mask[j][0] for j in range(len(mask)))
    # look for necessary extra axes
    new_in_idxs = [highs_inst.addVariable(0, shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(shape))]
    new_in_constr = [sum(new_in_idxs[k]*prod(shape[k+1:]) for k in range(len(shape))) == x]
    test_idx = offset + sum(new_in_idxs[j]*strides[j] for j in range(len(shape)))
    x_gt, x_lt = solve_highs(new_in_constr + valid + [idx>=test_idx+1], x), solve_highs(new_in_constr + valid + [idx<=test_idx-1], x)
    if x_gt is None and x_lt is None: break
    diff = [unravel(shape, min(x_gt or np.inf, x_lt or np.inf))[j] - unravel(shape,valid_start)[j] for j in range(len(shape))]
    if sum(d != 0 for d in diff) != 1: return None
    ax, newsh = [d != 0 for d in diff].index(True), sum(diff)
    if shape[ax] % newsh != 0: return None
    oldshape = [s for s in shape]
    shape[ax] //= newsh
    shape.insert(ax+1, newsh)
    if (mask := _reshape_mask(tuple((a,b) for a,b in mask), tuple(oldshape), tuple(shape))) is None: return None

  final =  View.create(tuple(shape), tuple(strides), offset, tuple((a,b) for a,b in mask))
  return final if not keep_shape else final.reshape(st.shape)

def overflow(u: UOp): return u.vmax > dtypes.max(dtypes.int) or u.vmin < dtypes.min(dtypes.int)

# If a node overflow, its srcs need to be checked to see if this overflow is the result of an ALU operation,
# or that the node simply inherits the dtype from srcs. Upcast is either `Ops.CAST`+`replace` or just `replace`.
def upcast(u: UOp):
  srcs = tuple(upcast(_src) for _src in u.src)
  if u.dtype.scalar() is dtypes.int:
    dtype = dtypes.int64.vec(u.dtype.count) if u.dtype.count > 1 else dtypes.int64
    upcasted = u.replace(dtype=dtype, src=tuple([_src.cast(dtype) for _src in srcs]))
    if overflow(u): return upcasted
    # Check the original src, new srcs has Ops.CAST whose vmin, vmax change the real bounds
    # Cast back is required because if the node is in range, siblings would never be upcasted
    if any((overflow(src) for src in u.src)): return upcasted.cast(u.dtype)
  return u.replace(src=tuple(srcs))

# pooling op may overflow before folding causing unnecessary upcast
def folded_upcast(u: UOp):
  with Context(TRACK_MATCH_STATS=0):
    return upcast(graph_rewrite(u, sym, {}))

@functools.lru_cache(None)
def views_to_indexed_uops(views: tuple[View, ...], _idxs:Optional[tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
  idx, valid = views[-1].to_indexed_uops(_idxs)
  for view in reversed(views[0:-1]):
    view = view.minify()
    idx, valid = view.to_indexed_uops([sint_to_uop(i) for i in unravel(view.shape, idx)], valid)
  return idx, valid

@functools.lru_cache(None)
def views_to_real_strides(views: tuple[View, ...], ignore_valid=False) -> tuple[Optional[sint], ...]:
  # NOTE: if a stride is not always valid, it will be None
  if len(views) == 1 and views[-1].mask is None: return views[-1].strides
  ret: list[Optional[sint]] = [None] * len(views[-1].shape)
  idx, valid = (graph_rewrite(u, symbolic_flat) for u in views_to_indexed_uops(views))
  # TODO: always apply these in to_indexed_uops?
  if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
  if (newidx:=uop_given_valid(valid, idx)) is not None: idx = graph_rewrite(newidx, symbolic_flat)
  for c in split_uop(idx, Ops.ADD):
    if c.op is Ops.RANGE: ret[c.arg] = 1
    if c.op is Ops.MUL and c.src[0].op is Ops.RANGE and c.src[1].op is Ops.CONST: ret[c.src[0].arg] = c.src[1].arg
    if c.op is Ops.MUL and c.src[1].op is Ops.RANGE and c.src[0].op is Ops.CONST: ret[c.src[1].arg] = c.src[0].arg
  used_ranges = [x.arg for x in idx.toposort if x.op is Ops.RANGE]
  ret = [x if i in used_ranges else 0 for i,x in enumerate(ret)]
  if not ignore_valid:
    for masked_axis in [x.arg for x in valid.toposort if x.op is Ops.RANGE]: ret[masked_axis] = None
  return tuple(ret)

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    if len(st.views) == 0: return self
    if len(self.views) == 0: return st
    new_views = (merged,) if (merged := self.views[-1]+st.views[0]) is not None else (self.views[-1], st.views[0])
    return ShapeTracker(self.views[:-1] + new_views) + ShapeTracker(st.views[1:])
#    ret = self
#    for v in st.views: ret = ShapeTracker(ret.views[:-1])
#    return ret

  def invert(self, out_shape:tuple[sint, ...]) -> Optional[ShapeTracker]:
    inverted_views:list[View] = []
    for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
      if (inverted:= v.invert(s)) is None: return None
      inverted_views.append(inverted)
    return ShapeTracker(tuple(inverted_views)).reshape(out_shape)

  @staticmethod
  def from_shape(shape:tuple[sint, ...]) -> ShapeTracker: return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def consecutive(self) -> bool: return len(self.views) == 1 and (v:=self.views[0]).mask is None and v.strides == strides_for_shape(v.shape)

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def reduce(self, axis:tuple[int, ...]) -> tuple[sint, ...]: return tuple(1 if i in axis else s for i,s in enumerate(self.shape))

  def to_uop(self) -> UOp: return UOp(Ops.VIEW, dtypes.void, (), self)
  def to_indexed_uops(self, _idxs:Optional[list[UOp]|tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
    idx, valid = views_to_indexed_uops(self.views, tuple(_idxs) if _idxs is not None else None)
    return folded_upcast(idx), folded_upcast(valid)

  # upper bound on buffer size required to fit this shapetracker
  def real_size(self) -> int:
    if 0 in self.shape: return 0
    view = (v.shrink(v.mask) if (v:=self.views[0]).mask else v)
    idx, _ = views_to_indexed_uops((view,))
    assert idx.vmax < 1e12, f"real_size broken for {self}"
    return int(idx.vmax + 1)

  def vars(self) -> set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> tuple[ShapeTracker, dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  def real_strides(self, ignore_valid=False) -> tuple[Optional[sint], ...]: return views_to_real_strides(self.views, ignore_valid)
  def unit_stride_axes(self, ignore_valid=False) -> list[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.to_indexed_uops()
    return axis in [x.arg for x in graph_rewrite(valid, symbolic_flat).toposort if x.op is Ops.RANGE]

  def simplify(self, *, keep_shape=True) -> ShapeTracker:
    if len(self.views) == 0: return self
    for j in range(len(self.views)):
      if (new := collapse_st(ShapeTracker(self.views[j:]), keep_shape=keep_shape)) is not None:
        return ShapeTracker(self.views[:j]).simplify(keep_shape=False) + ShapeTracker((new,))  # guaranteed to return when j=-1
    raise Exception()  # TODO get rid of this after testing that this works
#    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
#      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
#    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

  def mop(self, op, arg): return mops[op](self, arg)

mops: dict[Ops, Callable] = {Ops.RESHAPE: ShapeTracker.reshape, Ops.PERMUTE: ShapeTracker.permute, Ops.EXPAND: ShapeTracker.expand,
                             Ops.SHRINK: ShapeTracker.shrink, Ops.STRIDE: ShapeTracker.stride, Ops.PAD: ShapeTracker.pad}

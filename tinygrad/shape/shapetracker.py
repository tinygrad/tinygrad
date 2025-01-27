# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Optional, Callable
from tinygrad.helpers import merge_dicts, getenv, prod
from tinygrad.shape.view import View, strides_for_shape, unravel, _reshape_mask
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops, graph_rewrite, split_uop, symbolic_flat, Variable, sint, uop_given_valid, simplify_valid, sint_to_uop, Context
from tinygrad.ops import highs_pre, highs_reduce, highs_inst
from tinygrad.codegen.rewriter import sym
import numpy as np
import highspy

def solve_highs(constrs, min_obj) -> Optional[int]:
  # None if the constaints are infeasible
  cs = [highs_inst.addConstr(c) for c in constrs]
  highs_inst.minimize(min_obj)
  assert highs_inst.getModelStatus() in {highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kInfeasible}, "highs returned something other than Infeasible or Optimal"
  if highs_inst.getModelStatus() == highspy.HighsModelStatus.kInfeasible: ret = None
  else: ret = round(highs_inst.getSolution().col_value[0])
  for _ in range(len(cs)): highs_inst.removeConstr(cs[0].index)  # weird constr indexing behaviour
  return ret

def collapse_st(st:ShapeTracker, keep_shape:bool=False) -> Optional[View]:
  # presolve sometimes returns "infeasible" incorrectly
  highs_inst.clear(); highs_inst.setOptionValue("presolve", "off"); highs_inst.setOptionValue("log_to_console", False)
#  import pdb; pdb.set_trace()

  hx = highs_inst.addVariable(0, prod(st.shape)-1, type=highspy.HighsVarType.kInteger)
  x = UOp(Ops.HIGHS_EXPR, arg=(hx, 0, prod(st.shape)-1), dtype=dtypes.int)
  idx, valid = st.to_indexed_uops(unravel(st.shape, x))
  idx, invalid = graph_rewrite(graph_rewrite(idx, highs_pre), highs_reduce), graph_rewrite(graph_rewrite(valid!=True, highs_pre), highs_reduce)
  assert idx.op == Ops.HIGHS_EXPR and invalid.op == Ops.HIGHS_EXPR, "failed rewrite to highs program"
  hidx, hinvalid = idx.arg[0], invalid.arg[0]
  if solve_highs([hinvalid==0], hx) is None: return View.create(st.shape, (0,)*len(st.shape), 0, ((0,0),)*len(st.shape))

  # find mask  TODO check all-valid to skip this for speed
  vshape, mask, denom, valid_start = [], [], 1, 0
  while denom < prod(st.shape):
    x_mod_d = graph_rewrite(x%denom, highs_reduce).arg[0]  # factor out parts of the mask we already found
    a = solve_highs([hinvalid==0, x_mod_d==valid_start], hx)  # first valid section
    b = solve_highs([hinvalid>=1, x_mod_d==valid_start, hx>=a+1], hx) or prod(st.shape)  # second (or first) invalid section
    c = solve_highs([hinvalid==0, x_mod_d==valid_start, hx>=b+1], hx)  # second valid section
    vshape.append(c-a if c is not None else prod(st.shape)//denom)
    mask.append((a//denom%vshape[-1], (b-1)//denom%vshape[-1]+1))
    denom *= vshape[-1]
    if prod(st.shape) % denom != 0: return None
    valid_start = a % denom
  vshape, mask = vshape[::-1], mask[::-1]
  vidxs = [graph_rewrite(graph_rewrite(i, highs_pre), highs_reduce).arg[0] for i in unravel(vshape,x)]
  # check there are no invalid points in the new mask
  if solve_highs([hinvalid>=1] + [mask[j][0] <= vidxs[j] <= mask[j][1]-1 for j in range(len(vidxs))], hx) is not None: return None
  
  # find strides
#  import pdb; pdb.set_trace()
  while True:
    strides = []
    out_idx_at_valid_start = st.to_indexed_uops(unravel(st.shape, valid_start))[0].simplify().arg
    running_product = [prod(vshape[j+1:]) for j in range(len(vshape))]
    for p in running_product:
      strides.append(st.to_indexed_uops(unravel(st.shape, valid_start+p))[0].simplify().arg - out_idx_at_valid_start)
    offset = out_idx_at_valid_start - sum(strides[j]*mask[j][0] for j in range(len(mask)))
    unravelled = unravel(vshape, x)
    test_idx = graph_rewrite(graph_rewrite(offset+sum(strides[j]*unravelled[j] for j in range(len(vshape))), highs_pre), highs_reduce)
    test_hidx = test_idx.arg[0] if test_idx.op is Ops.HIGHS_EXPR else test_idx.arg  # can cancel to 0 here
    x_gt, x_lt = solve_highs([hinvalid==0, test_hidx>=(hidx+1)], hx), solve_highs([hinvalid==0, test_hidx<=(hidx-1)], hx)
    if x_gt is None and x_lt is None: break
    diff = [unravel(vshape, min(x_gt or np.inf, x_lt or np.inf))[j] - unravel(vshape,valid_start)[j] for j in range(len(vshape))]
    if sum(d != 0 for d in diff) != 1: return None
    ax, newsh = [d != 0 for d in diff].index(True), sum(diff)
    if vshape[ax] % newsh != 0: return None
    oldshape = [s for s in vshape]
    vshape[ax] //= newsh
    vshape.insert(ax+1, newsh)
    if (mask := _reshape_mask(tuple(mask), tuple(oldshape), tuple(vshape))) is None: return None

  final =  View.create(tuple(vshape), tuple(strides), offset, mask)
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
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

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

  def simplify(self) -> ShapeTracker:
    return ShapeTracker((collapse_st(self),))
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

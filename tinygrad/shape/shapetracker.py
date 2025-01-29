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
# time limit in seconds for each call of solver
highs_inst.setOptionValue("time_limit", 10); highs_inst.setOptionValue("log_to_console", False)
# presolve sometimes returns "infeasible" incorrectly
highs_inst.setOptionValue("presolve", "off"); highs_inst.setOptionValue("mip_feasibility_tolerance", 1e-10)

def st_to_highs(st, idxs) -> tuple:
  # add necessary auxiliary variables to highs_inst and return:
  # (highs_linear_expression for idx, [constrs whose conjunction when points are valid], [constrs whose disjunction when invalid])
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

def solve_highs(constrs, var, which: Literal["min", "max"]) -> Optional[int]:
  # None if the constaints are infeasible, otherwise returns the value of var; remove constrs from model when done
  cs = [highs_inst.addConstr(c) for c in constrs]
  if which == "min": highs_inst.minimize(var)
  elif which == "max": highs_inst.maximize(var)
  status, sol = highs_inst.getModelStatus(), round(highs_inst.getSolution().col_value[var.index])
  for _ in range(len(cs)): highs_inst.removeConstr(cs[0].index)  # weird constr indexing behaviour

  if status == highspy.HighsModelStatus.kTimeLimit:
#    import pdb; pdb.set_trace()
    raise RuntimeError("highs timeout")
  elif status == highspy.HighsModelStatus.kInfeasible: return None
  elif status == highspy.HighsModelStatus.kOptimal: return sol
  raise Exception(f"highs returned {status}, expected Infeasible or Optimal or TimeLimit")

@functools.lru_cache(maxsize=None)
def collapse_st(st:ShapeTracker, keep_shape:bool=True) -> Optional[View]:
  try:
    return _collapse_st_keep_shape(st) if keep_shape else _collapse_st_no_keep_shape(st)
  except RuntimeError:
    print(f"highs timeout with {st}, {keep_shape=}")
    return None

def _collapse_st_keep_shape(st):
  assert len(st.views) > 0
  if len(st.views) == 1: return st.views[0]

  ret = st.views[0]
  for v in st.views[1:]:
    ret = ret + v
    if ret is None: break
  if ret is not None: return ret

  if not all_int(flatten(((*v.shape, *v.strides, v.offset, *flatten(v.mask or ((0,),))) for v in st.views))): return None
  highs_inst.clearModel();

  in_idxs = [highs_inst.addVariable(0, st.shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(st.shape))]
  idx, valid, invalid = st_to_highs(st, in_idxs)
  def feasible(constrs): return True if solve_highs(constrs, in_idxs[0], "min") is not None else False
  if not feasible(valid): return View.create(st.shape, (0,)*len(st.shape), 0, ((0,0),)*len(st.shape))

  # find mask
  mask = []
  for j in range(len(in_idxs)):
    mask.append([solve_highs(valid, in_idxs[j], "min"), solve_highs(valid, in_idxs[j], "max")+1])

  new_valid = [mask[j][0] <= in_idxs[j] <= mask[j][1]-1 for j in range(len(st.shape))]
  new_invalid = sum(([in_idxs[j] <= mask[j][0]-1, in_idxs[j] >= mask[j][1]] for j in range(len(st.shape))), start=[])
  # check all points in new mask are valid, and all valid points are in new mask
  if any(feasible(new_valid + [constr]) for constr in invalid): return None
  if any(feasible(valid + [constr]) for constr in new_invalid): return None
  
  # find strides
  origin = st.to_indexed_uops([m[0] for m in mask])[0].simplify().arg
  strides = []
  for j in range(len(st.shape)):
    strides.append(st.to_indexed_uops([m[0]+1 if j==k else m[0] for k,m in enumerate(mask)])[0].simplify().arg - origin)
  offset = origin - sum(strides[j]*mask[j][0] for j in range(len(mask)))
  test_idx = offset + sum(i*st for i,st in zip(in_idxs, strides))
  gt, lt = feasible(valid + [test_idx >= idx+1]), feasible(valid + [test_idx <= idx-1])
  if gt or lt: return None

  final = View.create(tuple(st.shape), tuple(strides), offset, tuple((a,b) for a,b in mask))
  print(f"found reduction {st} --> {final}")
  return final

def _collapse_st_no_keep_shape(st:ShapeTracker) -> Optional[View]:
#  print("NO KEEP SHAPE")
  # Look for a view whose action is equiv. to st; this is complete
  assert len(st.views) > 0, "can't collapse empty st"
  if len(st.views) == 1: return st.views[0]
  # View.__add__ is faster, try that first
  ret = st.views[0]
  for v in st.views[1:]:
    ret = ret + v
    if ret is None: break
  if ret is not None: return ret

  # TODO symbolic mask and offset might be possible with this?
  if not all_int(flatten(((*v.shape, *v.strides, v.offset, *flatten(v.mask or ((0,),))) for v in st.views))): return None
  highs_inst.clearModel();

  # cutoff for speed; most hard cases are small shape anyway
#  if prod(st.shape) > 1e6: return None

#  import pdb; pdb.set_trace()
#  print(st)

  x = highs_inst.addVariable(0, prod(st.shape)-1, type=highspy.HighsVarType.kInteger)
  def feasible(constrs): return True if solve_highs(constrs, x, "min") is not None else False
  in_idxs = [highs_inst.addVariable(0, st.shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(st.shape))]
  highs_inst.addConstr(sum(in_idxs[k]*prod(st.shape[k+1:]) for k in range(len(st.shape))) == x)
  idx, valid, invalid = st_to_highs(st, in_idxs)
  valid_start = solve_highs(valid, x, "min")
  if valid_start is None: return View.create(st.shape, (0,)*len(st.shape), 0, ((0,0),)*len(st.shape))

  if not any(feasible([constr]) is not None for constr in invalid):
    # all points valid
    shape, mask = [prod(st.shape)], [[0, prod(st.shape)]]
  else:
    # find mask
    shape, mask, denom = [], [], 1
    x_step = highs_inst.addVariable(0, np.inf, type=highspy.HighsVarType.kInteger)
    while denom < prod(st.shape):
      a = min(solve_highs([constr] + [x==valid_start+denom*x_step, x>=valid_start], x, "min") or prod(st.shape) for constr in invalid)
      b = solve_highs(valid + [x==valid_start+denom*x_step, x>=a], x, "min")
      shape.append((b-valid_start)//denom if b is not None else prod(st.shape)//denom)
      mask.append([valid_start//denom%shape[-1], a//denom%shape[-1]])
      if mask[-1][1] == 0: mask[-1][1] = shape[-1]
      denom *= shape[-1]
      if prod(st.shape) % denom != 0: return None
    highs_inst.deleteVariable(x_step)

  shape, mask = shape[::-1], mask[::-1]
  # new indexes based on the found shape
  new_in_idxs = [highs_inst.addVariable(0, shape[k]-1, type=highspy.HighsVarType.kInteger) for k in range(len(shape))]
  new_in_constr = [sum(new_in_idxs[k]*prod(shape[k+1:]) for k in range(len(shape))) == x]
  new_valid = [mask[k][0] <= new_in_idxs[k] <= mask[k][1]-1 for k in range(len(shape))]
  new_invalid = sum(([new_in_idxs[k] <= mask[k][0]-1, new_in_idxs[k] >= mask[k][1]] for k in range(len(shape))), start=[])
  # check all points in new mask are valid, and all valid points are in new mask
  if any(feasible(new_in_constr + new_valid + [constr]) for constr in invalid): return None
  if any(feasible(new_in_constr + valid + [constr]) for constr in new_invalid): return None
  for _ in range(len(new_in_idxs)): highs_inst.deleteVariable(new_in_idxs[0].index)
  
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
    x_gt, x_lt = solve_highs(new_in_constr + valid + [idx>=test_idx+1], x, "min"), solve_highs(new_in_constr + valid + [idx<=test_idx-1], x, "min")
    if x_gt is None and x_lt is None: break
    diff = [unravel(shape, min(x_gt or np.inf, x_lt or np.inf))[j] - unravel(shape,valid_start)[j] for j in range(len(shape))]
    if sum(d != 0 for d in diff) != 1: return None
    ax, newsh = [d != 0 for d in diff].index(True), sum(diff)
    if shape[ax] % newsh != 0: return None
    oldshape = [s for s in shape]
    shape[ax] //= newsh
    shape.insert(ax+1, newsh)
    for _ in range(len(new_in_idxs)): highs_inst.deleteVariable(new_in_idxs[0].index)
    if (mask := _reshape_mask(tuple((a,b) for a,b in mask), tuple(oldshape), tuple(shape))) is None: return None

  final = View.create(tuple(shape), tuple(strides), offset, tuple((a,b) for a,b in mask))
  print(f"found reduction {st} --> {final}")
  return final

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

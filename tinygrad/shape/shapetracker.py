# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, Iterable, cast, Any
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, DivNode, ModNode, LtNode, AndNode, sint
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, graph_rewrite
from tinygrad.codegen.uopgraph import constant_folder

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker, only ints and UOps
def variable_to_uop(x, ctx=None) -> UOp: return UOp.const(dtypes.pyint, x) if isinstance(x, int) else x.render(render_ops, ctx)
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.pyint, self.b),
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*variable_to_uop(self.b, ctx),
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//variable_to_uop(self.b, ctx),
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%variable_to_uop(self.b, ctx),
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(variable_to_uop(self.b, ctx)),
  Variable: lambda self,ops,ctx: ctx[self] if ctx is not None and self in ctx else \
    UOp(UOps.DEFINE_VAR, dtypes.int, (UOp.const(dtypes.int, self.min), UOp.const(dtypes.int, self.max)), self),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

def _uop_view(view:View, idxs:List[UOp], vexpr:UOp) -> Tuple[UOp, UOp]:
  # TODO: dtypes.realint
  iexpr = variable_to_uop(view.offset)
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr = iexpr + idx*variable_to_uop(st)
    if m is not None:
      if m[0] != 0: vexpr = vexpr * idx.ge(variable_to_uop(m[0]))
      if m[1] != sh: vexpr = vexpr * idx.lt(variable_to_uop(m[1]))
  return iexpr, vexpr

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    inverted_views:List[View] = []
    for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
      if (inverted:= v.invert(s)) is None: return None
      inverted_views.append(inverted)
    return ShapeTracker(tuple(inverted_views)).reshape(out_shape)

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def consecutive(self) -> bool: return len(self.views) == 1 and (v:=self.views[0]).mask is None and v.strides == strides_for_shape(v.shape)

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def reduce(self, axis:Tuple[int, ...]) -> Tuple[sint, ...]: return tuple(1 if i in axis else s for i,s in enumerate(self.shape))

  def to_uop(self) -> UOp: return UOp(UOps.SHAPETRACKER, None, (), self)

  def to_indexed_uops(self, _idxs:Optional[List[UOp]]=None) -> Tuple[UOp, UOp]:
    idxs = [UOp(UOps.RANGE, dtypes.pyint, (UOp.const(dtypes.pyint, 0), variable_to_uop(s)), i) for i,s in enumerate(self.shape)] \
      if _idxs is None else _idxs
    idx, valid = _uop_view(self.views[-1], idxs, UOp.const(dtypes.bool, True))
    for view in reversed(self.views[0:-1]):
      view = view.minify()
      acc, idxs = 1, []
      for _d in reversed(view.shape):
        d = variable_to_uop(_d)
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _uop_view(view, idxs[::-1], valid)
    return idx, valid

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    idx, valid = self.to_indexed_uops()
    if not valid.vmax.arg: return 0
    assert idx.vmax.arg < 1e12, f"real_size broken for {self}"
    return idx.vmax.arg+1

  def vars(self) -> Set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = cast(sint, stride_maybe)
      except ValueError: bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)
    idx, valid = self.views[-1].expr(idxs)
    for view in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = view.expr(idxs[::-1], valid)
    assert not isinstance(idx.min, int) or idx.min >= -2**31, f"idx.min too small. {idx=}, {idx.min=}"
    assert not isinstance(idx.max, int) or idx.max < 2**31, f"idx.max too big. {idx=}, {idx.max=}"
    return idx, valid

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.to_indexed_uops()
    return axis in [x.arg for x in graph_rewrite(valid, constant_folder).sparents if x.op is UOps.RANGE]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

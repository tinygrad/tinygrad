# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, Any
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.symbolic import Variable, MulNode, SumNode, NumNode, DivNode, ModNode, LtNode, AndNode, sint
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, BinaryOps, graph_rewrite, resolve
from tinygrad.codegen.uopgraph import sym, _get_chain

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker, only ints and UOps
def variable_to_uop(x, ctx=None) -> UOp: return UOp.const(dtypes.pyint, x) if isinstance(x, int) else x.render(render_ops, ctx)
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.pyint, self.b),
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*variable_to_uop(self.b, ctx),
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//variable_to_uop(self.b, ctx),
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%variable_to_uop(self.b, ctx),
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(variable_to_uop(self.b, ctx)),
  Variable: lambda self,ops,ctx: ctx[self] if ctx is not None and self in ctx else UOp.define_var(self.expr, dtypes.int, self.min, self.max),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

def _uop_view(view:View, idxs:List[UOp], vexpr:UOp) -> Tuple[UOp, UOp]:
  # TODO: dtypes.realint
  iexpr = variable_to_uop(view.offset)
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if resolve(sh != 1) and resolve(st != 0): iexpr = iexpr + idx*variable_to_uop(st)
    if m is not None:
      if resolve(m[0] != 0): vexpr = vexpr * idx.ge(variable_to_uop(m[0]))
      if resolve(m[1] != sh): vexpr = vexpr * idx.lt(variable_to_uop(m[1]))
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

  def to_uop(self) -> UOp: return UOp(UOps.SHAPETRACKER, dtypes.void, (), self)

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
    if not valid.vmax: return 0
    assert idx.vmax < 1e12, f"real_size broken for {self}"
    return int(idx.vmax+1)

  def vars(self) -> Set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    ret: List[Optional[sint]] = [None] * len(self.shape)
    idx, valid = self.to_indexed_uops()
    idx = graph_rewrite(idx, pm=sym)
    for c in _get_chain(idx, BinaryOps.ADD):
      if c.op is UOps.RANGE: ret[c.arg] = 1
      if c.op is UOps.ALU and c.arg is BinaryOps.MUL and c.src[0].op is UOps.RANGE and c.src[1].op is UOps.CONST: ret[c.src[0].arg] = c.src[1].arg
      if c.op is UOps.ALU and c.arg is BinaryOps.MUL and c.src[1].op is UOps.RANGE and c.src[0].op is UOps.CONST: ret[c.src[1].arg] = c.src[0].arg
    used_ranges = [x.arg for x in graph_rewrite(idx, pm=sym).sparents if x.op is UOps.RANGE]
    ret = [x if i in used_ranges else 0 for i,x in enumerate(ret)]
    if not ignore_valid:
      masked_axis = [x.arg for x in graph_rewrite(valid, pm=sym).sparents if x.op is UOps.RANGE]
      ret = [None if i in masked_axis else x for i,x in enumerate(ret)]
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.to_indexed_uops()
    return axis in [x.arg for x in graph_rewrite(valid, sym).sparents if x.op is UOps.RANGE]

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

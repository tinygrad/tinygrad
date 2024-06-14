from __future__ import annotations
from typing import List, Tuple, Iterable
import math
from tinygrad.codegen.kernel import Kernel
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.ops import BufferOps, LazyOp, TernaryOps, ReduceOps, BinaryOps
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.renderer import Program
from tinygrad.helpers import to_function_name, colored, DEBUG, getenv

def _uop_view(view:View, idxs:List[UOp], vexpr:UOp=UOp.const(dtypes.bool, True)) -> Tuple[UOp, UOp]:
  # TODO: dtypes.realint
  iexpr = UOp.const(dtypes.int32, view.offset)
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr = iexpr + idx*st
    if m is not None:
      vexpr = vexpr * idx.ge(m[0]) * idx.lt(m[1])
  return iexpr, vexpr

def st_to_uops(st:ShapeTracker, idxs:Iterable[UOp]) -> Tuple[UOp, UOp]:
  idx, valid = _uop_view(st.views[-1], idxs)
  for view in reversed(st.views[0:-1]):
    view = view.minify()
    acc, idxs = 1, []
    for d in reversed(view.shape):
      idxs.append((idx//acc)%d)
      acc *= d
    idx, valid = _uop_view(view, idxs[::-1], valid)
  return idx, valid

def get_reduce_acc(reduceop:LazyOp):
  if reduceop.op is ReduceOps.SUM: return 0.0 if dtypes.is_float(reduceop.dtype) else 0
  if reduceop.op is ReduceOps.MAX:
    if dtypes.is_int(reduceop.dtype): return 0 if dtypes.is_unsigned(reduceop.dtype) else -2**(reduceop.dtype.itemsize*8-1)
    return -math.inf if dtypes.is_float(reduceop.dtype) else False

class Lowerer(Kernel):
  def to_uop(self, x:LazyOp) -> UOp:
    print(x.op)
    if x.op in BufferOps:
      idx, valid = st_to_uops(self.sts[self.bufs.index(x.arg)], self.idxs)
      if x.op is BufferOps.CONST:
        return UOp.alu(TernaryOps.WHERE, valid, UOp.const(x.arg.dtype, x.arg.val), UOp.const(x.arg.dtype, 0))
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(x.arg.dtype), (), (x.arg.idx, any(x.arg.idx == y.idx for y in self.outbufs)))
        if x.op is BufferOps.LOAD:
          return UOp(UOps.LOAD, x.arg.dtype, (buf, idx, valid))
        else:
          return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0]), valid))
    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op in ReduceOps:
      loops = [self.idxs[i] for i in range(len(self.full_shape)) if self.full_shape[i] != self.sts[0].shape[i]]
      # TODO: DEFINE_ACC should have a const input
      op = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[x.op]
      return UOp(UOps.REDUCE, x.dtype, (in_uops[0], UOp.const(x.dtype, get_reduce_acc(x))) + tuple(loops), op)
    return UOp.alu(x.op, *in_uops)

  def linearize(self) -> Lowerer:
    # kernel name (before late upcast)
    self.name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.lazyops) else "E")) + \
                 (f"{len(self.outbufs)}_" if len(self.outbufs) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])
    self.idxs = []

    # for clang
    """
    for i,g in enumerate(self.full_shape):
      self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, g)), (i,0)))
    self.global_size, self.local_size = None, None
    """

    # TODO: why is this middle name arg here?
    for i,g in enumerate(self.full_shape[:self.global_dims]):
      self.idxs.append(UOp(UOps.SPECIAL, dtypes.int32, (), (self.global_dims-1-i, f"gidx{i}", g)))
    for i,g in enumerate(self.full_shape[self.global_dims:self.global_dims+self.local_dims]):
      self.idxs.append(UOp(UOps.SPECIAL, dtypes.int32, (), (self.local_dims-1-i, f"lidx{i}", g)))
    for i,g in enumerate(self.full_shape[self.first_reduce:]):
      self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, g)), (i,0)))

    self.global_size = list(self.full_shape[:self.global_dims][::-1])
    self.local_size = list(self.full_shape[self.global_dims:self.global_dims+self.local_dims][::-1])
    self.global_size += [1]*(3-len(self.global_size))
    self.local_size += [1]*(3-len(self.local_size))

    self.uops:UOpGraph = UOpGraph([self.to_uop(x) for x in self.ast])

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"): self.uops.graph()

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(to_function_name(self.name), self.uops)
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size, self.uops, *self.uops.flops_mem())
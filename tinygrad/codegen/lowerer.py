from __future__ import annotations
from typing import List, Tuple, Iterable, cast
import math
from dataclasses import replace
from tinygrad.codegen.kernel import LocalBuffer, Kernel
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.ops import BufferOps, LazyOp, TernaryOps, ReduceOps, BinaryOps, UnaryOps
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.renderer import Program
from tinygrad.helpers import to_function_name, colored, DEBUG, getenv, prod
from tinygrad.engine.graph import print_tree

def _uop_view(view:View, idxs:List[UOp], vexpr:UOp) -> Tuple[UOp, UOp]:
  # TODO: dtypes.realint
  iexpr = UOp.const(dtypes.int32, view.offset)
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr = iexpr + idx*st
    if m is not None:
      if m[0] != 0: vexpr = vexpr * idx.ge(m[0])
      if m[1] != sh: vexpr = vexpr * idx.lt(m[1])
  return iexpr, vexpr

def st_to_uops(st:ShapeTracker, idxs:Iterable[UOp]) -> Tuple[UOp, UOp]:
  idx, valid = _uop_view(st.views[-1], idxs, UOp.const(dtypes.bool, True))
  for view in reversed(st.views[0:-1]):
    view = view.minify()
    acc, idxs = 1, []
    for d in reversed(view.shape):
      idxs.append((idx//acc)%d)
      acc *= d
    idx, valid = _uop_view(view, idxs[::-1], valid)
  return idx, valid

def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0):
  local_idxs = loop_local_idxs = [UOp(UOps.SPECIAL, dtypes.int32, (), (i, f"{prefix}{start_dim+i}", s)) for i,s in enumerate((prod(local_dims[:-(maxdim-1)]),) + local_dims[-(maxdim-1):] if len(local_dims) > maxdim else local_dims)]  # noqa: E501
  if maxdim != 0 and len(local_dims) > maxdim:
    dd = local_idxs[0]
    nli = []
    for s in local_dims[:-(maxdim-1)]:
      nli.append(dd % s)
      dd //= s
    local_idxs = nli + local_idxs[-(maxdim-1):]
  return local_idxs, loop_local_idxs

def get_reduce_acc(reduceop:LazyOp):
  if reduceop.op is ReduceOps.SUM: return 0.0 if dtypes.is_float(reduceop.dtype) else 0
  if reduceop.op is ReduceOps.MAX:
    if dtypes.is_int(reduceop.dtype): return 0 if dtypes.is_unsigned(reduceop.dtype) else -2**(reduceop.dtype.itemsize*8-1)
    return -math.inf if dtypes.is_float(reduceop.dtype) else False

uop_graphed = False
class Lowerer(Kernel):
  def to_uop(self, x:LazyOp) -> UOp:
    #print(x.op)
    if x.op in BufferOps:
      idx, valid = st_to_uops(x.arg.st, self.idxs)
      # TODO: check has_valid in UPat, not here
      has_valid = valid.uop is not UOps.CONST or valid.arg is not True
      if x.op is BufferOps.CONST:
        return UOp.alu(TernaryOps.WHERE, valid, UOp.const(x.arg.dtype, x.arg.val), UOp.const(x.arg.dtype, 0))
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(x.arg.dtype), (), (x.arg.idx, any(x.arg.idx == y.idx for y in self.outbufs)))
        if x.op is BufferOps.LOAD:
          return UOp(UOps.LOAD, x.arg.dtype, (buf, idx) + ((valid, UOp.const(x.arg.dtype, 0)) if has_valid else ()))
        else:
          return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid) if has_valid else ()))
    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is UnaryOps.CAST:
      return UOp(UOps.CAST, x.arg, in_uops)
    if x.op in ReduceOps:
      op = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[x.op]
      return UOp(UOps.REDUCE, x.dtype, (in_uops[0], UOp.const(x.dtype, get_reduce_acc(x))) + tuple(self.idxs[i] for i in x.arg), op)
    return UOp.alu(x.op, *in_uops)

  def linearize(self) -> Lowerer:
    global uop_graphed
    # kernel name (before late upcast)
    self.name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.lazyops) else "E")) + \
                 (f"{len(self.outbufs)}_" if len(self.outbufs) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])
    if DEBUG >= 4: print(self.name)
    self.idxs = []

    # set the shapetrackers to the optimized ones, fixup reduceop
    # transformed to the final LazyOp
    def fixup_ast(op):
      if op.op in BufferOps:
        arg = replace(op.arg, st=self.sts[self.bufs.index(op.arg)])
      elif op.op in ReduceOps:
        arg = tuple(i for i in range(len(self.full_shape)) if self.full_shape[i] != self.sts[0].shape[i])
        print(op.arg, arg)
      else:
        arg = op.arg
      return LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), arg)
    self.ast = [fixup_ast(x) for x in self.ast]
    print_tree(self.ast[0])

    # add a local buffer for multistage reduce. # TODO: use local alias
    """
    if self.group_for_reduces:
      for i in range(len(self.reduceops)):
        # TODO: the strides of this can be controlled
        self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces]) + [1] * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        temp_dtype = cast(LazyOp, self.reduceop).dtype
        self.bufs.append(LocalBuffer(name:=f"temp{i if len(self.reduceops) > 1 else ''}", buf_size:=self.sts[-1].size, temp_dtype))
        print_tree(self.ast[0])
        print(self.bufs[-1], self.sts[-1])
    """

    # define indexes
    global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, self.full_shape[:self.global_dims], 3 if self.opts.has_local else 0)
    local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+self.group_for_reduces], 3 if self.opts.has_local else 0)  # noqa: E501
    self.idxs = global_idxs + local_idxs

    for i,g in enumerate(self.full_shape[self.first_reduce:]):
      unrolled = (self.first_reduce+i) >= (self.shape_len-self.upcasted)
      self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, g)), (i,unrolled)))

    self.global_size = [x.arg[2] for x in loop_global_idxs]
    self.local_size = [x.arg[2] for x in loop_local_idxs]
    self.global_size += [1]*(3-len(self.global_size))
    self.local_size += [1]*(3-len(self.local_size))

    self.uops:UOpGraph = UOpGraph([self.to_uop(x) for x in self.ast])

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS") and not uop_graphed:
      self.uops.graph()
      uop_graphed = True
    return self

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(to_function_name(self.name), self.uops)
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size, self.uops, *self.uops.flops_mem())
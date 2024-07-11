from __future__ import annotations
from typing import List, Tuple, cast, Optional, Any, Dict
import functools
from tinygrad.codegen.kernel import Kernel
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType
from tinygrad.ops import BufferOps, LazyOp, TernaryOps, ReduceOps, UnaryOps, get_lazyop_info
from tinygrad.codegen.uops import UOp, flops_mem, UOps
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.renderer import Program
from tinygrad.helpers import to_function_name, DEBUG, getenv, prod, diskcache_put, ContextVar

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker, only ints and UOps
from tinygrad.shape.symbolic import Variable, NumNode, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode
def variable_to_uop(x, ctx=None) -> UOp: return UOp.const(dtypes.bigint, x) if isinstance(x, int) else x.render(render_ops, ctx)
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.bigint, self.b),
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*variable_to_uop(self.b, ctx),
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//variable_to_uop(self.b, ctx),
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%variable_to_uop(self.b, ctx),
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(variable_to_uop(self.b, ctx)),
  Variable: lambda self,ops,ctx: ctx[self] if ctx is not None and self in ctx else UOp(UOps.DEFINE_VAR, dtypes.int32, (), self),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

if getenv("UOP_IS_SYMBOLIC"):
  # TODO: change this once UOps is ready to replace symbolic
  def _uop_view(view:View, idxs:List[UOp], vexpr:UOp) -> Tuple[UOp, UOp]:
    # TODO: dtypes.realint
    iexpr = variable_to_uop(view.offset)
    for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
      if sh != 1 and st != 0: iexpr = iexpr + idx*variable_to_uop(st)
      if m is not None:
        if m[0] != 0: vexpr = vexpr * idx.ge(variable_to_uop(m[0]))
        if m[1] != sh: vexpr = vexpr * idx.lt(variable_to_uop(m[1]))
    return iexpr, vexpr

  def st_to_uops(st:ShapeTracker, idxs:List[UOp]) -> Tuple[UOp, UOp]:
    idx, valid = _uop_view(st.views[-1], idxs, UOp.const(dtypes.bool, True))
    for view in reversed(st.views[0:-1]):
      view = view.minify()
      acc, idxs = 1, []
      for _d in reversed(view.shape):
        d = variable_to_uop(_d)
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _uop_view(view, idxs[::-1], valid)
    return idx, valid
else:
  def st_to_uops(st:ShapeTracker, idxs:List[UOp]) -> Tuple[UOp, UOp]:
    fake_idxs = [Variable(f"__idx{i}", 0, s-1) for i,s in enumerate(st.shape)]
    idx, valid = st.expr_idxs(fake_idxs)
    ctx = dict(zip(fake_idxs, idxs))
    uidx, uvalid = idx.render(render_ops, ctx), valid.render(render_ops, ctx)
    if uvalid.op is UOps.CONST: uvalid = UOp.const(dtypes.bool, uvalid.arg)
    assert uvalid.dtype == dtypes.bool
    return uidx, uvalid

def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0) -> Tuple[List[UOp], List[UOp]]:
  local_idxs = loop_local_idxs = [UOp(UOps.SPECIAL, dtypes.bigint, (), (i, f"{prefix}{start_dim+i}", s)) for i,s in enumerate((prod(local_dims[:-(maxdim-1)]),) + local_dims[-(maxdim-1):] if len(local_dims) > maxdim else local_dims)]  # noqa: E501
  if maxdim != 0 and len(local_dims) > maxdim:
    dd = local_idxs[0]
    nli = []
    for s in local_dims[:-(maxdim-1)]:
      nli.append(dd % s)
      dd //= s
    local_idxs = nli + local_idxs[-(maxdim-1):]
  return local_idxs, loop_local_idxs

class Lowerer(Kernel):
  def to_uop(self, x:LazyOp) -> UOp:
    if uop:=self.uop_cache.get(x, None): return uop
    ret = self._to_uop(x)
    self.uop_cache[x] = ret
    return ret

  def _to_uop(self, x:LazyOp) -> UOp:
    if x.op in BufferOps:
      idx, valid = st_to_uops(x.arg.st, self.ridxs if x.op is BufferOps.LOAD and x.arg.idx == -1 else self.idxs)
      # TODO: check has_valid in UPat, not here
      has_valid = valid.op is not UOps.CONST or valid.arg is not True
      if x.op is BufferOps.CONST:
        dtype = x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype
        return UOp.alu(TernaryOps.WHERE, valid, UOp.const(dtype, x.arg.val), UOp.const(dtype, 0))
      if x.arg.idx == -1:
        buf = UOp(UOps.DEFINE_LOCAL, PtrDType(x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype), (), ("temp2", x.arg.st.size))
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (),
                  (x.arg.idx, any(x.arg.idx == y.arg.idx for y in self.ast)))
      if x.op is BufferOps.LOAD:
        barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()
        return UOp(UOps.LOAD, x.arg.dtype.scalar(), (buf, idx) + ((valid, UOp.const(x.arg.dtype.scalar(), 0)) if has_valid else ()) + barrier)
      # NOTE: only store the local reduceop in the first thread
      if self.group_for_reduces > 0 and x.arg.idx != -1: valid, has_valid = valid * self.idxs[self.first_reduce].eq(0), True
      return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid,) if has_valid else ()))

    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is UnaryOps.CAST: return UOp(UOps.CAST, x.arg.scalar(), in_uops)
    if x.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, x.arg.scalar(), in_uops)
    if x.op in ReduceOps:
      dtype = x.dtype.base if isinstance(x.dtype, ImageDType) else x.dtype
      if x.op is ReduceOps.WMMA:
        wmma_sz, upcast_axis = x.arg[4], x.arg[6]
        ret = UOp(UOps.WMMA, dtype=dtype.vec(wmma_sz[2]), src=(
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[0].dtype).vec(wmma_sz[0]), src=(in_uops[0],), arg=(upcast_axis[0],)),
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[1].dtype).vec(wmma_sz[1]), src=(in_uops[1],), arg=(upcast_axis[1],)),
          UOp.const(dtype.vec(wmma_sz[2]), 0.0)), arg=x.arg)
        return UOp(UOps.EXPAND, dtype, tuple(UOp(UOps.GEP, dtype, (ret,), i) for i in range(wmma_sz[2])), arg=upcast_axis[2])
      # NOTE: always using ridxs is fine here
      return UOp(UOps.REDUCE, dtype, (in_uops[0],) + tuple(self.ridxs[i] for i in x.arg), x.op)
    return UOp.alu(x.op, *in_uops)

  def linearize(self) -> Lowerer:
    modified_ast, ki = self.get_optimized_ast()
    if DEBUG >= 3:
      from tinygrad.engine.graph import print_tree
      for mast in modified_ast: print_tree(mast)

    if self.opts.has_local:
      # define indexes
      global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, ki.full_shape[:ki.global_dims], 3)
      local_idxs, loop_local_idxs = get_grouped_dims("lidx", ki.global_dims, ki.full_shape[ki.global_dims:ki.first_reduce+ki.group_for_reduces], 3)
      self.idxs = global_idxs + local_idxs

      # define sizes
      self.global_size: Optional[List[int]] = [x.arg[2] for x in loop_global_idxs]
      self.local_size: Optional[List[int]] = [x.arg[2] for x in loop_local_idxs]
      self.global_size += [1]*(3-len(self.global_size))
      self.local_size += [1]*(3-len(self.local_size))
    else:
      # all loops are RANGES
      self.idxs = [UOp(UOps.RANGE, dtypes.bigint, (UOp.const(dtypes.bigint, 0), variable_to_uop(g)), (i, False))
                   for i,g in enumerate(ki.full_shape[:ki.first_reduce])]
      self.global_size, self.local_size = None, None

    # reduce loops
    self.idxs += [UOp(UOps.RANGE, dtypes.bigint, (UOp.const(dtypes.bigint, 0), variable_to_uop(g)), (i, True))
      for i,g in enumerate(ki.full_shape[ki.first_reduce+ki.group_for_reduces:ki.shape_len-ki.upcasted], start=ki.first_reduce+ki.group_for_reduces)]

    # upcast loops
    for i,g in enumerate(ki.full_shape[ki.shape_len-ki.upcasted:], start=ki.shape_len-ki.upcasted):
      assert isinstance(g, int), "needs to be int to upcast/unroll"
      self.idxs.append(UOp(UOps.EXPAND, dtypes.bigint, tuple(UOp.const(dtypes.bigint, j) for j in range(0, g)), i))

    # late indexes (group for reduce)
    self.ridxs = self.idxs[:]
    for a in range(ki.first_reduce, ki.first_reduce+ki.group_for_reduces):
      self.ridxs[a] = UOp(UOps.RANGE, dtypes.bigint, (UOp.const(dtypes.bigint, 0), variable_to_uop(ki.full_shape[a])), (1000+a, True))

    self.uop_cache: Dict[LazyOp, UOp] = {}
    self.uops:UOpGraph = UOpGraph([self.to_uop(x) for x in modified_ast], self.opts)

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"):
      self.uops.graph()
      if getenv("GRAPHUOPS") == 2: exit(0)
    return self

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(name:=to_function_name(self.name), self.uops)
    if getenv("RUN_PROCESS_REPLAY"):
      diskcache_put("process_replay", id(self), (self.ast, self.opts, self.applied_opts, name, src, {k:v.value for k,v in ContextVar._cache.items()}))
    info = get_lazyop_info(self.ast[0])
    ops, mem = flops_mem(self.uops.uops)
    run_count = prod((self.global_size or []) + (self.local_size or []))
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size,
                   self.uops, min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))

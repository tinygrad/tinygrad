from __future__ import annotations
from typing import List, Tuple, cast, Optional, Any, Dict
import functools
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.shape.symbolic import sint
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType
from tinygrad.ops import BufferOps, LazyOp, ReduceOps, UnaryOps, MetaOps, KernelInfo, MemBuffer, BinaryOps
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.renderer import Renderer
from tinygrad.helpers import getenv, all_int, get_contraction, prod, partition, flatten

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker, only ints and UOps
from tinygrad.shape.symbolic import Variable, NumNode, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode
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

# TODO: change this once UOps is ready to replace symbolic
def st_to_uops_graph(st:ShapeTracker, idxs:List[UOp], dtype:DType) -> Tuple[UOp, UOp]:
  idx, valid = _uop_view(st.views[-1], idxs, UOp.const(dtypes.bool, True))
  for view in reversed(st.views[0:-1]):
    view = view.minify()
    acc, idxs = 1, []
    for _d in reversed(view.shape):
      d = variable_to_uop(_d)
      idxs.append((idx//acc)%d)
      acc *= d
    idx, valid = _uop_view(view, idxs[::-1], valid)
  if isinstance(dtype, ImageDType):
    idx = UOp(UOps.VECTORIZE, dtypes.int.vec(3), ((idx // 4) % dtype.shape[1], (idx // (4 * dtype.shape[1])), idx % 4))
  return idx, valid

# TODO: this is the old one, delete when ready
def st_to_uops_symbolic(st:ShapeTracker, idxs:List[UOp], dtype:DType) -> Tuple[UOp, UOp]:
  fake_idxs = [Variable(f"__idx{i}", 0, s-1) for i,s in enumerate(st.shape)]
  idx, valid = st.expr_idxs(fake_idxs)
  ctx = dict(zip(fake_idxs, idxs))
  uvalid = valid.render(render_ops, ctx)
  if isinstance(dtype, ImageDType):
    image_idxs = (idx // 4) % dtype.shape[1], (idx // (4 * dtype.shape[1])), idx % 4
    uidx = UOp(UOps.VECTORIZE, dtypes.int.vec(3), tuple(x.render(render_ops, ctx) for x in image_idxs))
  else:
    uidx = idx.render(render_ops, ctx)
  if uvalid.op is UOps.CONST: uvalid = UOp.const(dtypes.bool, uvalid.arg)
  assert uvalid.dtype == dtypes.bool
  return uidx, uvalid

def st_to_uops(st:ShapeTracker, idxs:List[UOp], dtype:DType) -> Tuple[UOp, UOp]:
  if getenv("SYMBOLIC_DIFF"):
    symbolic_idx, symbolic_valid = st_to_uops_symbolic(st, idxs, dtype)
    graph_idx, graph_valid = st_to_uops_graph(st, idxs, dtype)
    import ocdiff
    from tinygrad.codegen.uopgraph import UOpGraph
    from tinygrad.renderer.cstyle import OpenCLRenderer

    def render(s1, s2):
      glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg="idxs")
      st = tuple(UOp(UOps.STORE, None, (glbl, UOp.const(dtypes.int, i), s)) for i,s in enumerate([s1,s2]))
      return OpenCLRenderer().render("indexing", UOpGraph(UOp(UOps.SINK, None, st)).linearize(skip_check=True).uops)

    cmp_symbolic, cmp_graph = render(symbolic_idx, symbolic_valid), render(graph_idx, graph_valid)
    if cmp_symbolic != cmp_graph: print(ocdiff.console_diff(f"SYMBOLIC {len(cmp_symbolic)}\n"+cmp_symbolic, f"GRAPH {len(cmp_graph)}\n"+cmp_graph))
  return st_to_uops_graph(st, idxs, dtype) if getenv("UOP_IS_SYMBOLIC") else st_to_uops_symbolic(st, idxs, dtype)

def _limit_dims(dims:Tuple[sint, ...], max_sizes:Tuple[int, ...]):
  # TODO: symbolic shape
  if not all_int(dims): return dims
  while len(dims) > len(max_sizes) or any(d > m for d,m in zip(dims, max_sizes)):
    for i,m in enumerate(max_sizes):
      if dims[i] * dims[i+1] <= m:
        dims = dims[:i] + (dims[i]*dims[i+1],) + dims[i+2:]
        break
    else: raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
  return dims

def get_grouped_dims(prefix, dims:Tuple[sint, ...], max_sizes:Optional[Tuple[int, ...]], reverse=False) -> List[UOp]:
  if reverse: dims = dims[::-1]
  limited = _limit_dims(dims, max_sizes) if max_sizes is not None else dims
  ret = raw_idxs = [UOp(UOps.SPECIAL, dtypes.pyint, (), (f"{prefix}{i}", s)) for i,s in enumerate(limited)]
  if limited != dims:
    ret = []
    # cast for mypy, get_contraction won't be None
    for idx, contraction in zip(raw_idxs, cast(List[List[int]], get_contraction(dims, limited))):
      if len(contraction) == 1: ret.append(idx)
      else:
        for c in contraction:
          ret.append(idx % dims[c])
          idx //= dims[c]
  return ret[::-1] if reverse else ret

class IndependentLowerer:
  def lower(self, ast:LazyOp, opts:Renderer) -> UOp:
    self.output_count = len(ast.src)

    ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
    # NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
    full_shape = ast.full_shape
    first_upcasted = len(full_shape)-ki.upcasted
    # if there's no reduce, this is first_upcasted
    first_reduce = [x!=y for x,y in zip(ast.src[0].arg.st.shape[:first_upcasted]+(0,), full_shape[:first_upcasted]+(1,))].index(True)
    local_loads = [x for x in ast.lazyops if x.op is BufferOps.LOAD and x.arg.idx == -1]
    # NOTE: this is taking the first one...there may be subtlelies here with multireduces
    group_for_reduces = sum([x!=y for x,y in zip(
      local_loads[0].arg.st.shape[first_reduce:first_upcasted], ast.src[0].arg.st.shape[first_reduce:first_upcasted])]) if local_loads else 0
    global_dims = first_reduce-ki.local_dims

    if opts.has_local:
      if ki.dont_use_locals:
        assert ki.local_dims == 0, "can't use locals if there's no local dims"
        self.idxs = get_grouped_dims("idx", full_shape[:global_dims], opts.global_max, reverse=True)
      else:
        # define indexes for GPU-like execution
        self.idxs = get_grouped_dims("gidx", full_shape[:global_dims], opts.global_max, reverse=True) + \
                    get_grouped_dims("lidx", full_shape[global_dims:first_reduce+group_for_reduces], opts.local_max)
    else:
      # all loops are RANGES
      self.idxs = [UOp(UOps.RANGE, dtypes.pyint, (UOp.const(dtypes.pyint, 0), variable_to_uop(g)), (i, False))
                   for i,g in enumerate(full_shape[:first_reduce])]

    # reduce loops
    self.idxs += [UOp(UOps.RANGE, dtypes.pyint, (UOp.const(dtypes.pyint, 0), variable_to_uop(g)), (i, True))
      for i,g in enumerate(full_shape[first_reduce+group_for_reduces:first_upcasted], start=first_reduce+group_for_reduces)]

    # upcast loops
    for i,g in enumerate(full_shape[first_upcasted:], start=first_upcasted):
      assert isinstance(g, int), "needs to be int to upcast/unroll"
      self.idxs.append(UOp(UOps.EXPAND, dtypes.pyint, tuple(UOp.const(dtypes.pyint, j) for j in range(0, g)), ((i,g),)))

    # late indexes (group for reduce)
    self.ridxs = self.idxs[:]
    for a in range(first_reduce, first_reduce+group_for_reduces):
      self.ridxs[a] = UOp(UOps.RANGE, dtypes.pyint, (UOp.const(dtypes.pyint, 0), variable_to_uop(full_shape[a])), (1000+a, True))

    self.uop_cache: Dict[LazyOp, UOp] = {}
    return self.to_uop(ast)

  def to_uop(self, x:LazyOp) -> UOp:
    if uop:=self.uop_cache.get(x, None): return uop
    ret = self._to_uop(x)
    self.uop_cache[x] = ret
    return ret

  def _to_uop(self, x:LazyOp) -> UOp:
    if x.op in BufferOps:
      idx, valid = st_to_uops(x.arg.st, self.ridxs if x.op is BufferOps.LOAD and x.arg.idx == -1 else self.idxs,
        x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) and (not isinstance(x.arg, MemBuffer) or x.arg.idx == -1) else x.arg.dtype)
      # TODO: check has_valid in UPat, not here
      has_valid = valid.op is not UOps.CONST or valid.arg is not True
      if x.op is BufferOps.CONST:
        dtype = x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype
        return valid.where(UOp.const(dtype, x.arg.val), UOp.const(dtype, 0))
      if x.arg.idx < 0:
        buf = UOp(UOps.DEFINE_LOCAL, PtrDType(x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype),
                  arg=(f"temp{-x.arg.idx}", x.arg.st.real_size()))
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (), x.arg.idx)
      if x.op is BufferOps.LOAD:
        barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()
        load_dtype = x.arg.dtype.scalar()
        if idx.dtype == dtypes.int.vec(3):
          # this should all simplify if there's consts for id4. if not, w/e
          idx, id4 = UOp(UOps.VECTORIZE, dtypes.int.vec(2), (idx.src[0], idx.src[1])), idx.src[2]
          vec_load = UOp(UOps.LOAD, load_dtype.vec(4), (buf, idx) + ((UOp.const(load_dtype.vec(4), 0), valid) if has_valid else ()) + barrier)
          return functools.reduce(lambda ret, i: id4.ne(i).where(ret, UOp(UOps.GEP, load_dtype, (vec_load,), i)),
                                  range(4), UOp.const(load_dtype, float('nan')))
        return UOp(UOps.LOAD, load_dtype, (buf, idx) + ((UOp.const(load_dtype, 0), valid) if has_valid else ()) + barrier)
      # NOTE: only store the local reduceop in the first thread (this is wrong for non group for reduces!)
      if x.arg.idx >= 0:
        for oidx, ridx in zip(self.idxs, self.ridxs):
          if oidx != ridx: valid = valid * oidx.eq(0)
        has_valid = valid.op is not UOps.CONST or valid.arg is not True
      return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid,) if has_valid else ()))

    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is MetaOps.KERNEL: return UOp(UOps.SINK, src=in_uops)
    if x.op is UnaryOps.CAST: return UOp(UOps.CAST, x.arg.scalar(), in_uops)
    if x.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, x.arg.scalar(), in_uops)
    if x.op in ReduceOps:
      dtype = x.dtype.base if isinstance(x.dtype, ImageDType) else x.dtype
      if x.op is ReduceOps.WMMA:
        upcast_axes = x.arg[-2]
        wmma_sz = [prod(x[1] for x in l) for l in upcast_axes]
        ret = UOp(UOps.WMMA, dtype=dtype.vec(wmma_sz[2]), src=(
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[0].dtype).vec(wmma_sz[0]), src=(in_uops[0],), arg=upcast_axes[0]),
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[1].dtype).vec(wmma_sz[1]), src=(in_uops[1],), arg=upcast_axes[1]),
          UOp.const(dtype.vec(wmma_sz[2]), 0.0)), arg=x.arg)
        return UOp(UOps.EXPAND, dtype, tuple(UOp(UOps.GEP, dtype, (ret,), i) for i in range(wmma_sz[2])), arg=upcast_axes[2])
      # NOTE: always using ridxs is fine here
      reduce_range, reduce_expand = partition([self.ridxs[i] for i in x.arg], lambda y: y.op is UOps.RANGE)
      alu_op = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, x.op)]
      ret = in_uops[0]
      if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
        ret = UOp(UOps.CONTRACT, dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))
        ret = functools.reduce(lambda x,y: x.alu(alu_op, y), [ret.gep(i) for i in range(cast(DType, ret.dtype).count)])
      return UOp(UOps.REDUCE, dtype, (ret,) + tuple(reduce_range), alu_op) if len(reduce_range) else ret
    return in_uops[0].alu(x.op, *in_uops[1:])

def lazyop_to_uop(ast:LazyOp, opts:Renderer) -> UOp: return IndependentLowerer().lower(ast, opts)

from __future__ import annotations
from typing import List, Tuple, cast, Optional, Any, Dict, Final, DefaultDict
import functools
from dataclasses import replace
from collections import defaultdict
from tinygrad.codegen.kernel import LocalBuffer, Kernel
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType
from tinygrad.ops import BufferOps, LazyOp, TernaryOps, ReduceOps, UnaryOps, MemBuffer, BinaryOps, get_lazyop_info
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.renderer import Program
from tinygrad.helpers import to_function_name, colored, DEBUG, getenv, prod, flatten

# TODO: this needs to be replaced, there shouldn't be variables in the shapetracker
def variable_to_uop(x, ctx=None) -> UOp:
  if isinstance(x, int): return UOp.const(dtypes.int32, x)
  return x.render(render_ops, ctx)

from tinygrad.shape.symbolic import Variable, NumNode, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode
render_ops: Any = { NumNode: lambda self, ops, ctx: UOp.const(dtypes.int, self.b),
                    MulNode: lambda self, ops, ctx: self.a.render(ops, ctx)*variable_to_uop(self.b, ctx),
                    DivNode: lambda self, ops, ctx: self.a.render(ops, ctx)//variable_to_uop(self.b, ctx),
                    ModNode: lambda self, ops, ctx: self.a.render(ops, ctx)%variable_to_uop(self.b, ctx),
                    LtNode: lambda self, ops, ctx: self.a.render(ops, ctx).lt(variable_to_uop(self.b, ctx)),
  Variable: lambda self,ops,ctx: ctx[self] if ctx is not None and self in ctx else UOp(UOps.DEFINE_VAR, dtypes.int32, (), self),
  SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a+b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)),
  AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: a*b.render(ops, ctx), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

# TODO: change this once UOps is ready to replace symbolic
def st_to_uops(st:ShapeTracker, idxs:List[UOp]) -> Tuple[UOp, UOp]:
  fake_idxs = [Variable(f"__idx{i}", 0, s-1) for i,s in enumerate(st.shape)]
  idx, valid = st.expr_idxs(fake_idxs)
  ctx = dict(zip(fake_idxs, idxs))
  return idx.render(render_ops, ctx), valid.render(render_ops, ctx).cast(dtypes.bool)

def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0) -> Tuple[List[UOp], List[UOp]]:
  local_idxs = loop_local_idxs = [UOp(UOps.SPECIAL, dtypes.int32, (), (i, f"{prefix}{start_dim+i}", s)) for i,s in enumerate((prod(local_dims[:-(maxdim-1)]),) + local_dims[-(maxdim-1):] if len(local_dims) > maxdim else local_dims)]  # noqa: E501
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
      has_valid = valid.op is not UOps.CONST or (valid.arg is not True and valid.arg != 1)
      if x.op is BufferOps.CONST:
        dtype = x.arg.dtype.base if isinstance(x.arg.dtype, ImageDType) else x.arg.dtype
        return UOp.alu(TernaryOps.WHERE, valid, UOp.const(dtype, x.arg.val), UOp.const(dtype, 0))
      if isinstance(self.bufs[x.arg.idx], LocalBuffer):
        # TODO: this should come from somewhere else
        lb = self.bufs[x.arg.idx]
        buf = UOp(UOps.DEFINE_LOCAL, PtrDType(lb.dtype), (), (lb.name, lb.size))
      else:
        buf = UOp(UOps.DEFINE_GLOBAL, x.arg.dtype if isinstance(x.arg.dtype, ImageDType) else PtrDType(x.arg.dtype), (),
                  (x.arg.idx, any(x.arg.idx == y.idx for y in self.outbufs)))
      if x.op is BufferOps.LOAD:
        barrier = (UOp(UOps.BARRIER, None, (self.to_uop(x.src[0]),)),) if len(x.src) else ()
        return UOp(UOps.LOAD, x.arg.dtype.scalar(), (buf, idx) + ((valid, UOp.const(x.arg.dtype.scalar(), 0)) if has_valid else ()) + barrier)
      if self.group_for_reduces > 0 and x.arg.idx != -1: valid, has_valid = valid * self.idxs[self.first_reduce].eq(0), True
      return UOp(UOps.STORE, None, (buf, idx, self.to_uop(x.src[0])) + ((valid,) if has_valid else ()))

    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is UnaryOps.CAST: return UOp(UOps.CAST, x.arg.scalar(), in_uops)
    if x.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, x.arg.scalar(), in_uops)
    if x.op in ReduceOps:
      # NOTE: always using ridxs is fine here
      #arg = x.op
      dtype = x.dtype.base if isinstance(x.dtype, ImageDType) else x.dtype
      if x.op is ReduceOps.WMMA:
        upcast_axis = self.shape_len-self.upcasted+1
        ret = UOp(UOps.WMMA, dtype=dtype.vec(2), src=(
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[0].dtype).vec(2), src=(in_uops[0],), arg=(upcast_axis, 2)),
          UOp(UOps.CONTRACT, dtype=cast(DType, in_uops[1].dtype).vec(2), src=(in_uops[1],), arg=(upcast_axis, 2)),
          UOp.const(dtype.vec(2), 0.0)), arg=x.arg)
        return UOp(UOps.EXPAND, dtype, tuple(UOp(UOps.GEP, dtype, (ret,), i) for i in range(2)), arg=upcast_axis)
      src = (in_uops[0],) + tuple(self.ridxs[i] for i in x.arg)
      return UOp(UOps.REDUCE, dtype, src, x.op)
    return UOp.alu(x.op, *in_uops)

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self) -> Lowerer:
    sts_backup, bufs_backup = self.sts, self.bufs

    self.uop_cache: Dict[LazyOp, UOp] = {}

    # late alias the tensor core buffers
    if (tc:=self.tensor_core) and self.tensor_core_opts is not None:
      # NOTE: the 4 is required if using real locals
      alias_pattern = [0]*(self.global_dims) + [2]*(len(tc.threads)) + [4]*(self.local_dims-len(tc.threads)) + [0]*(self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3]*(self.upcasted-2)  # noqa: E501
      for op, tc_bufs in self.bufs_for_tensor_core.items():
        for tc_buf in tc_bufs: self.alias_buffer(op, tc_buf, alias_pattern)

    # kernel name (before late upcast)
    self.name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.lazyops) else "E")) + \
                 (f"{len(self.outbufs)}_" if len(self.outbufs) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])
    if DEBUG >= 4: print(self.name)

    # name the function something unique
    Lowerer.kernel_cnt[(function_name := to_function_name(self.name))] += 1
    suffix = f"{'n'+str(Lowerer.kernel_cnt[function_name]-1)}" if Lowerer.kernel_cnt[function_name] > 1 else ""
    self.name = self.name+colored(suffix, 'BLACK')

    self.idxs = []
    # add a local buffer for multistage reduce.
    if self.group_for_reduces:
      for i in range(len(self.reduceops)):
        # TODO: the strides of this can be controlled
        self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces]) + [1] * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        temp_dtype = cast(LazyOp, self.reduceop).dtype
        self.bufs.append(LocalBuffer(f"temp{i if len(self.reduceops) > 1 else ''}", self.sts[-1].size,
                                     temp_dtype.base if isinstance(temp_dtype, ImageDType) else temp_dtype))

    # set the shapetrackers to the optimized ones, fixup reduceop
    # transformed to the final LazyOp
    @functools.lru_cache(None)
    def fixup_ast(op:LazyOp) -> LazyOp:
      if op.op in BufferOps:
        idx = self.bufs.index(op.arg)
        arg = replace(op.arg, st=self.sts[idx])
        for top, v in self.local_alias.items():
          if idx in v and (tc:=self.tensor_core):
            lbuf = v[idx]
            lidx = self.bufs.index(lbuf)
            assert arg.st.real_size() == self.sts[idx].real_size()

            # two shapetrackers
            st1:ShapeTracker = arg.st
            st2 = self.sts[lidx]

            shape_szs = {i+1:k for i,(_,k) in enumerate(tc.threads)}
            shape_szs[-1] = tc.thread_local_sizes[-1][0]

            tc_buf_num = self.bufs_for_tensor_core[top].index(idx)
            tla = tc.thread_local_aliases[tc_buf_num]

            # very hacky shapetracker fixup
            new_shape = tuple(st1.shape[:self.shape_len-self.upcasted])
            mtla = tla[len(tc.threads):]
            for i,t in enumerate(mtla):
              if len(t) == 1:
                new_shape += (st1.shape[self.shape_len-self.upcasted+i],)
              else:
                new_shape += tuple(shape_szs[tt] for tt in t[::-1])
            new_shape += tuple(st1.shape[self.shape_len-self.upcasted+len(mtla):])
            permaxis = list(range(0, len(new_shape)))
            fmtla = flatten([x[::-1] for x in mtla])

            for i,a in enumerate(fmtla):
              tidx = self.shape_len-self.upcasted+i
              if a == -1: swap = self.shape_len-self.upcasted+len(fmtla)-1
              elif a > 0: swap = self.global_dims+a-1
              else: continue
              permaxis[swap], permaxis[tidx] = permaxis[tidx], permaxis[swap]

            def fix_st(st, new_shape, permaxis):
              old_shape = st.shape
              st = st.reshape(old_shape[:self.shape_len-self.upcasted] + new_shape[self.shape_len-self.upcasted:])
              st = st.permute(tuple(permaxis))
              return st.reshape(old_shape)

            st1 = fix_st(st1, new_shape, permaxis)
            st2 = fix_st(st2, new_shape, permaxis)

            start = LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), MemBuffer(arg.idx, arg.dtype, st1))
            return start
            #local_store = LazyOp(BufferOps.STORE, (start,), MemBuffer(lidx, start.dtype, st2))
            #local_load = LazyOp(BufferOps.LOAD, (local_store,), MemBuffer(lidx, start.dtype, self.sts[lidx]))
            #return local_load
      elif op.op in ReduceOps:
        arg = tuple(i for i in range(self.first_reduce+self.group_for_reduces, self.shape_len) if self.full_shape[i] != self.sts[0].shape[i])
        if op in self.bufs_for_tensor_core and (tc := self.tensor_core):
          rsrc = op.src[0]
          if rsrc.op is UnaryOps.CAST: rsrc = rsrc.src[0]
          assert rsrc.op is BinaryOps.MUL
          wmma_sz = [prod(l) for l in tc.thread_local_sizes]
          wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, tuple(wmma_sz), self.opts.device)
          reduce_axis = self.shape_len-self.upcasted
          ret = LazyOp(ReduceOps.WMMA, tuple(fixup_ast(x) for x in rsrc.src), wmma_arg)
          return LazyOp(op.op, (ret,), tuple(i for i in arg if i != reduce_axis)) if len(arg) > 1 else ret
        if self.group_for_reduces:
          start = LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), arg)
          local_buffer = MemBuffer(-1, start.dtype, self.sts[-1])
          local_store = LazyOp(BufferOps.STORE, (start,), local_buffer)
          local_load = LazyOp(BufferOps.LOAD, (local_store,), local_buffer)
          return LazyOp(op.op, (local_load,), tuple(range(self.first_reduce, self.first_reduce+self.group_for_reduces)))
      else:
        arg = op.arg
      return LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), arg)
    modified_ast = tuple(fixup_ast(x) for x in self.ast)

    if DEBUG >= 4:
      from tinygrad.engine.graph import print_tree
      for mast in modified_ast: print_tree(mast)

    if self.opts.has_local:
      # define indexes
      global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, self.full_shape[:self.global_dims], 3 if self.opts.has_local else 0)
      local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+self.group_for_reduces], 3 if self.opts.has_local else 0)  # noqa: E501
      self.idxs = global_idxs + local_idxs

      # define sizes
      self.global_size: Optional[List[int]] = [x.arg[2] for x in loop_global_idxs]
      self.local_size: Optional[List[int]] = [x.arg[2] for x in loop_local_idxs]
      self.global_size += [1]*(3-len(self.global_size))
      self.local_size += [1]*(3-len(self.local_size))
    else:
      # all loops
      self.idxs = []
      for i,g in enumerate(self.full_shape[:self.first_reduce]):
        self.idxs.append(UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(g)), (i, False)))
      self.global_size, self.local_size = None, None

    # reduce loops
    for i,g in enumerate(self.full_shape[self.first_reduce+self.group_for_reduces:], start=self.first_reduce+self.group_for_reduces):
      unrolled, is_reduce = i >= (self.shape_len-self.upcasted), self.full_shape[i] != self.output_shape[i]
      if unrolled:
        assert isinstance(g, int), "needs to be int to unroll"
        uop = UOp(UOps.EXPAND, dtypes.int32, tuple(UOp.const(dtypes.int32, j) for j in range(0, g)), i)
      else:
        uop = UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(g)), (i, is_reduce))
      self.idxs.append(uop)

    # late indexes
    self.ridxs = self.idxs[:]
    for a in range(self.first_reduce, self.first_reduce+self.group_for_reduces):
      self.ridxs[a] = UOp(UOps.RANGE, dtypes.int32, (UOp.const(dtypes.int32, 0), variable_to_uop(self.full_shape[a])), (1000+a, True))

    self.uops:UOpGraph = UOpGraph([self.to_uop(x) for x in modified_ast], self.opts)

    self.sts, self.bufs = sts_backup, bufs_backup

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"):
      self.uops.graph()
      if getenv("GRAPHUOPS") == 2: exit(0)
    return self

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(to_function_name(self.name), self.uops)
    info = get_lazyop_info(self.ast[0])
    ops, mem = self.uops.flops_mem()
    run_count = prod((self.global_size or []) + (self.local_size or []))
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size,
                   self.uops, min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))

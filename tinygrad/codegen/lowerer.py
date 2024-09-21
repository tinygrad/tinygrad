from __future__ import annotations
import functools
from typing import List, Tuple, cast, Optional, Dict
from tinygrad.shape.shapetracker import ShapeTracker, variable_to_uop
from tinygrad.shape.symbolic import sint
from tinygrad.dtype import dtypes
from tinygrad.ops import KernelInfo, BinaryOps, BUFFER_UOPS, UOp, UOps
from tinygrad.renderer import Renderer
from tinygrad.helpers import all_int, get_contraction, prod, partition, flatten

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
  def lower(self, ast:UOp, opts:Renderer) -> UOp:
    self.output_count = len(ast.src)

    ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
    # NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
    full_shape = ast.full_shape
    first_upcasted = len(full_shape)-ki.upcasted
    first_output_st: ShapeTracker = ast.src[0].st_arg
    # if there's no reduce, this is first_upcasted
    first_reduce = [x!=y for x,y in zip(first_output_st.shape[:first_upcasted]+(0,), full_shape[:first_upcasted]+(1,))].index(True)
    local_loads = [x for x in ast.parents if x.op is UOps.LOAD and x.src[0].op is UOps.DEFINE_LOCAL]
    # NOTE: sum up the reduced axes looking across all local loads, yields the number of grouped reduces
    group_for_reduces = sum([any(j!=y for j in x) for x,y in zip(
      [[l.st_arg.shape[i] for l in local_loads] for i in range(first_reduce,first_upcasted)],
      first_output_st.shape[first_reduce:first_upcasted])]) if local_loads else 0
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
      self.idxs.append(UOp(UOps.EXPAND, dtypes.pyint, (UOp.const(dtypes.pyint.vec(g), tuple(range(g))),), ((i,g),)))

    # late indexes (group for reduce)
    self.ridxs = self.idxs[:]
    for a in range(first_reduce, first_reduce+group_for_reduces):
      self.ridxs[a] = UOp(UOps.RANGE, dtypes.pyint, (UOp.const(dtypes.pyint, 0), variable_to_uop(full_shape[a])), (1000+a, True))

    self.uop_cache: Dict[UOp, UOp] = {}
    return self.to_uop(ast)

  def to_uop(self, x:UOp) -> UOp:
    if uop:=self.uop_cache.get(x, None): return uop
    ret = self._to_uop(x)
    self.uop_cache[x] = ret
    return ret

  def _to_uop(self, x:UOp) -> UOp:
    if x.op in BUFFER_UOPS:
      idx, valid = x.st_arg.to_indexed_uops(self.ridxs if x.op is UOps.LOAD and x.src[0].op is UOps.DEFINE_LOCAL else self.idxs)
      if x.op is UOps.VALID: return valid
      # TODO: check has_valid in UPat, not here
      has_valid = valid.op is not UOps.CONST or valid.arg is not True
      buf = x.src[0]
      if x.op is UOps.LOAD:
        barrier = (UOp(UOps.BARRIER, dtypes.void, (self.to_uop(x.src[2]),)),) if x.src[0].op is UOps.DEFINE_LOCAL else ()
        return UOp(UOps.LOAD, x.dtype, (buf, idx) + ((x.const_like(0), valid) if has_valid else ()) + barrier)
      # NOTE: only store the local reduceop in the threads that are actually doing the reduce
      store_back = x.src[0].op is UOps.DEFINE_LOCAL and x.src[2].op is UOps.REDUCE_AXIS and \
        x.src[2].src[0].op is UOps.LOAD and x.src[2].src[0].src[0].op is UOps.DEFINE_LOCAL
      # NOTE: If we're storing the reduced value back into each thread, need to zero-out the reduced axes
      if store_back: idx, _ = x.st_arg.to_indexed_uops([u.const_like(0) if i in x.src[2].arg[1] else u for i,u in enumerate(self.idxs)])
      if x.src[0].op is UOps.DEFINE_GLOBAL or store_back:
        for oidx, ridx in zip(self.idxs, self.ridxs):
          if oidx != ridx: valid = valid * oidx.eq(0)
        has_valid = valid.op is not UOps.CONST or valid.arg is not True
      return UOp(UOps.STORE, dtypes.void, (buf, idx, self.to_uop(x.src[2])) + ((valid,) if has_valid else ()))

    in_uops = tuple(self.to_uop(y) for y in x.src)
    if x.op is UOps.WMMA:
      upcast_axes = x.arg[-2]
      wmma_sz = [prod(x[1] for x in l) for l in upcast_axes]
      ret = UOp(UOps.WMMA, dtype=x.dtype.vec(wmma_sz[2]), src=(
        UOp(UOps.CONTRACT, dtype=in_uops[0].dtype.vec(wmma_sz[0]), src=(in_uops[0],), arg=upcast_axes[0]),
        UOp(UOps.CONTRACT, dtype=in_uops[1].dtype.vec(wmma_sz[1]), src=(in_uops[1],), arg=upcast_axes[1]),
        UOp.const(x.dtype.vec(wmma_sz[2]), 0.0)), arg=x.arg)
      return UOp(UOps.EXPAND, x.dtype, (ret,), arg=upcast_axes[2])
    if x.op is UOps.REDUCE_AXIS:
      # NOTE: always using ridxs is fine here
      reduce_range, reduce_expand = partition([self.ridxs[i] for i in x.arg[1]], lambda y: y.op is UOps.RANGE)
      alu_op: BinaryOps = x.arg[0]
      ret = in_uops[0]
      if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
        ret = UOp(UOps.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))
        ret = functools.reduce(lambda x,y: x.alu(alu_op, y), [ret.gep(i) for i in range(ret.dtype.count)])
      return UOp(UOps.REDUCE, x.dtype, (ret,) + tuple(reduce_range), alu_op) if len(reduce_range) else ret
    return x if x.src == in_uops else UOp(x.op, x.dtype, in_uops, x.arg)

def ast_to_uop(ast:UOp, opts:Renderer) -> UOp: return IndependentLowerer().lower(ast, opts)

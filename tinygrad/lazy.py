from __future__ import annotations
import math
from typing import Union, Optional, Any, Tuple, List
from tinygrad.dtype import dtypes, DType, ConstType
from tinygrad.helpers import prod, getenv, all_int, all_same, DEBUG
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op, exec_alu, python_alu
from tinygrad.shape.symbolic import sint, Variable
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from weakref import ref, ReferenceType, WeakValueDictionary

lazycache: WeakValueDictionary[Any, LazyBuffer] = WeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None, enable_cache=bool(getenv("LAZYCACHE", 1))):
  if st.size == 0: op, arg, srcs, base = LoadOps.CONST, 0, (), None
  if op is LoadOps.CONST: arg, enable_cache = dtypes.as_const(arg, dtype) if not isinstance(arg, Variable) else arg, True

  cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))
  if enable_cache and (rret := lazycache.get(cache_key, None)): return rret

  ret = LazyBuffer(device, st, dtype, op, arg, srcs, base=base)
  if enable_cache: lazycache[cache_key] = ret
  return ret

view_supported_devices = {"LLVM", "CLANG", "CUDA", "DISK"}
class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None):
    self.device, self.st, self.dtype, self.shape, self.size = device, st, dtype, st.shape, st.size
    self._base: Optional[LazyBuffer] = None
    if base is None:
      # properties on base
      self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
      assert self.op is not LoadOps.ASSIGN or srcs[1].base.realized is not None, "assign target must be realized"

      if (self.op is LoadOps.CONTIGUOUS or (self.op is UnaryOps.CAST and self.arg[1] is True)) and srcs[0].st.consecutive and \
          not srcs[0].is_unrealized_const() and device.split(":")[0] in view_supported_devices:
        # some LazyBuffers can be processed with only a view, no AST required
        self.buffer: Buffer = srcs[0].base.buffer.view(st.size, dtype, srcs[0].st.views[0].offset * srcs[0].dtype.itemsize)
        self.op = LoadOps.VIEW
      else:
        self.buffer = srcs[1].base.buffer if self.op is LoadOps.ASSIGN else Buffer(device, self.size, dtype)
      self.buffer.ref(1)
      self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None
      self.forced_realize = False
    else:
      # properties on view
      assert base.base == base, "base must be a base itself"
      self._base = base

  def __del__(self):
    if hasattr(self, 'buffer'): self.buffer.ref(-1)

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} {str(self.dtype)[7:]} {self.st if self.base != self else (self.op, self.realized)}>"

  @property
  def realized(self) -> Optional[Buffer]:
    # NOTE: we check for a lack of srcs instead of an allocated buffer to make unrealized assigns return None here
    return self.buffer if self._base is None and not hasattr(self, 'srcs') else None

  # NOTE: this has to be a function to prevent self reference
  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  # same API as multi
  @property
  def lbs(self) -> List[LazyBuffer]: return [self]

  @staticmethod
  def loadop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Tuple[LazyBuffer, ...]=(), enable_cache=False) -> LazyBuffer:
    assert isinstance(src, tuple)
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, src, enable_cache=enable_cache)

  def const(self, val:ConstType, shape:Optional[Tuple[sint,...]]=None) -> LazyBuffer:
    shape = self.shape if shape is None else shape
    return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=val).reshape((1,)*len(shape)).expand(shape)

  def is_realized(self) -> bool: return self.base.realized is not None

  def assign(self, x:LazyBuffer) -> LazyBuffer:
    assert x.size == self.size, f"assign target must have same size {self.size=} != {x.size=}"
    return LazyBuffer.loadop(LoadOps.ASSIGN, self.shape, self.dtype, self.device, arg=() if self.st.contiguous else (self.st,), src=(x, self.base))

  def contiguous(self):
    if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():
      ret = self.e(LoadOps.CONTIGUOUS)
      if (sti := self.st.invert(self.base.shape)) is not None: self.base.contiguous_child = ref(ret), sti
      return ret
    self.base.forced_realize = True
    return self

  def cast(self, dtype:DType, bitcast:bool=False):
    if self.dtype == dtype: return self
    if self.device.startswith("DISK") and not bitcast: raise RuntimeError("attempted to cast disk buffer (bitcast only)")
    if self.is_unrealized_unmasked_const() and not bitcast:
      return create_lazybuffer(self.device, self.st, dtype, LoadOps.CONST, dtypes.as_const(self.base.arg, dtype))
    # TODO: applying this makes gpt2 slower
    if getenv("CAST_BEFORE_VIEW", 1) and dtype.itemsize <= self.dtype.itemsize and self != self.base:
      return self.base.cast(dtype, bitcast)._view(self.st)
    new_shape = self.shape
    if bitcast and self.dtype.itemsize != dtype.itemsize:
      if not self.device.startswith("DISK"): raise RuntimeError("shape changing bitcast only supported on DISK right now")
      if not all_int(new_shape): raise RuntimeError("shape changing bitcast with symbolic shape isn't supported yet")
      # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
      if not (new_shape[-1]*self.dtype.itemsize) % dtype.itemsize == 0: raise RuntimeError("unsupported size in bitcast")
      new_shape = new_shape[:-1] + ((new_shape[-1]*self.dtype.itemsize) // dtype.itemsize,)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), dtype, UnaryOps.CAST, (dtype, bitcast), (self,))

  def is_unrealized_const(self): return self.base.realized is None and self.base.op is LoadOps.CONST and not isinstance(self.base.arg, Variable)
  def is_unrealized_unmasked_const(self): return self.is_unrealized_const() and all(v.mask is None for v in self.st.views)

  def _copy(self, device:str) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, self.buffer.nbytes, (self,), enable_cache=False)

  def copy_to_device(self, device:str, force: bool = False) -> LazyBuffer:
    # no COPY
    if self.device == device: return self

    # double COPY = one COPY
    if not force and self.st.contiguous and self.size == self.base.size and not self.base.realized and self.base.op is LoadOps.COPY:
      return self.base.srcs[0].copy_to_device(device).reshape(self.st.shape)

    # const doesn't have to be copied (issues with disk tensor)
    if self.is_unrealized_const():
      return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, device, arg=self.base.arg)._view(self.st)

    # if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if prod(self.st.shape) < prod(self.base.st.shape): return self.contiguous()._copy(device)

    # copy the base and apply the shapetracker on the new device
    return self.base._copy(device)._view(self.st)

  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs: List[LazyBuffer] = []
    for s in (self,)+in_srcs:
      if s == s.base and s.base.contiguous_child and (root:=s.base.contiguous_child[0]()) is not None:
        srcs.append(root._view(s.base.contiguous_child[1]))
      else:
        srcs.append(s)
    assert all_same(dts:=[x.dtype.scalar() for x in (srcs[1:] if op is TernaryOps.WHERE else srcs)]), f"all dtypes must match {dts} on {op}"
    assert all_same([x.shape for x in srcs]), f"all shapes must be the same {[x.shape for x in srcs]}"
    if op is TernaryOps.WHERE: assert srcs[0].dtype == dtypes.bool, "TernaryOps.WHERE must have the first arg be bool"
    if op is UnaryOps.NEG: assert srcs[0].dtype != dtypes.bool, "UnaryOps.NEG does not accept dtype bool"

    out_dtype = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else srcs[-1].dtype

    # const folding
    if op in python_alu and all(s.is_unrealized_unmasked_const() for s in srcs):
      return self.cast(out_dtype).const(exec_alu(op, out_dtype, [s.base.arg for s in srcs]))
    if op is UnaryOps.NEG and self.base.op is UnaryOps.NEG: return self.base.srcs[0]
    if op in BinaryOps: x, y = self, in_srcs[0]
    if op is BinaryOps.ADD:
      if y.is_unrealized_unmasked_const() and y.base.arg == 0: return x
      if x.is_unrealized_unmasked_const() and x.base.arg == 0: return y
    if op is BinaryOps.SUB and y.is_unrealized_unmasked_const() and y.base.arg == 0: return x
    if op is BinaryOps.MUL:
      if x.is_unrealized_unmasked_const() and (val := x.base.arg) in (1, 0, -1):
        return y if val == 1 else y.const(0) if val == 0 else y.e(UnaryOps.NEG)
      if y.is_unrealized_unmasked_const() and (val := float(y.base.arg)) in (1, 0, -1):
        return x if val == 1 else x.const(0) if val == 0 else x.e(UnaryOps.NEG)
    if op is BinaryOps.DIV and dtypes.is_float(x.dtype) and y.is_unrealized_unmasked_const() and y.base.arg != 0:
      return x.e(BinaryOps.MUL, x.const(1 / y.base.arg))

    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, arg, tuple(srcs))

  # *** reduce ops ***

  def _reduce_op(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    assert all(0 <= x < len(self.shape) for x in axis), f"axis args {axis} out of range for shape {self.shape}"
    axis = tuple(x for x in axis if self.shape[x] != 1)
    if len(axis) == 0: return self
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, axis, (self,))

  def r(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    # TODO: this logic should move to the scheduler
    if self.size == 0 and 0 not in new_shape: return self.const({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[op], new_shape)

    # const folding
    if self.is_unrealized_unmasked_const():
      return self.const(self.base.arg * {ReduceOps.SUM: prod(self.shape[i] for i in axis), ReduceOps.MAX: 1}[op], new_shape)

    # TODO: can we split symbolic shape if the reduce axis is not symbolic?
    if not getenv("SPLIT_REDUCEOP", 1) or not all_int(self.shape) or (0 in self.shape) or \
      prod(self.shape) // prod(new_shape) < getenv("REDUCEOP_SPLIT_THRESHOLD", 32768):
      return self._reduce_op(op, axis)

    # if there are few globals, make some reduces into globals by splitting into two kernels
    # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
    #   ~2**10 should be enough if GROUP is used
    # 256 split maximum should be "negligible reduce" for low prod(new_shape), 8 split minimum.
    # split is moved to the end to provide maximum locality for the second phase reduce.
    self_real_strides = self.st.real_strides(ignore_valid=True)
    split_candidates = [(i, x) for i in axis for x in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(new_shape)),8-1,-1)
                        if self.shape[i] % x == 0 and self_real_strides[i] != 0]
    if not split_candidates: return self._reduce_op(op, axis)
    dim_to_split, divisor = split_candidates[0]
    splitted_shape = self.shape[:dim_to_split] + (divisor,) + (self.shape[dim_to_split]//divisor,) + self.shape[dim_to_split+1:]
    splitted = self.reshape(splitted_shape).permute(tuple([x for x in range(len(splitted_shape)) if x != dim_to_split]+[dim_to_split]))
    if DEBUG >= 3: print(f"split {divisor}: {self.shape} -> {splitted.shape} -> {new_shape}")
    return splitted._reduce_op(op, axis)._reduce_op(op, (len(new_shape),)).reshape(new_shape)  # reduce original axes, then split

  # *** movement ops ***

  def _view(self, new_st:ShapeTracker) -> LazyBuffer:
    if self.st.size == 0 or (new_st.views[-1].mask is not None and any((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return self.const(0, new_st.shape)
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self._view(self.st.permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self._view(self.st.stride(arg))

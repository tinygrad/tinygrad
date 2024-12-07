from __future__ import annotations
from typing import Optional, Any, Tuple, List, get_args
from tinygrad.dtype import dtypes, DType, ConstType, to_dtype, ImageDType
from tinygrad.helpers import prod, getenv, all_int, all_same, DEBUG, _METADATA, Metadata, SPLIT_REDUCEOP, LAZYCACHE
from tinygrad.ops import exec_alu, python_alu
from tinygrad.ops import identity_element, MathTrait, resolve, UOp, sint, GroupOp, Ops, view_supported_devices
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from weakref import ref, ReferenceType, WeakValueDictionary

lazycache: WeakValueDictionary[Any, LazyBuffer] = WeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType, op:Optional[Ops]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None, enable_cache=bool(LAZYCACHE)):
  dtype = to_dtype(dtype)
  if op is Ops.CONST: arg, enable_cache = dtypes.as_const(arg, dtype) if not isinstance(arg, UOp) else arg, True

  cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))
  if enable_cache and (rret := lazycache.get(cache_key, None)) is not None: return rret

  ret = LazyBuffer(device, st, dtype, op, arg, srcs, base=base, metadata=_METADATA.get())
  if enable_cache: lazycache[cache_key] = ret
  return ret

class LazyBuffer(MathTrait):
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Ops]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None, metadata:Optional[Metadata]=None):
    self.device, self.st, self.dtype, self.shape, self.size, self.metadata = device, st, to_dtype(dtype), st.shape, st.size, metadata
    self._base: Optional[LazyBuffer] = None
    if base is None:
      # properties on base
      self.op, self.arg, self.srcs = op, arg, srcs  # this is a UOp, except the src is LazyBuffers and not UOps
      assert self.op is not Ops.ASSIGN or srcs[0].base.realized is not None, "assign target must be realized"
      assert all_same([x.st.shape for x in self.srcs]), f"src shape mismatch! {self.srcs}"

      if self.op is Ops.BUFFER_VIEW:
        # some LazyBuffers can be processed with only a view, no AST required
        self.buffer: Buffer = srcs[0].base.buffer.view(st.size, self.dtype, srcs[0].st.views[0].offset * srcs[0].dtype.itemsize)
      else:
        self.buffer = srcs[0].base.buffer if self.op is Ops.ASSIGN else Buffer(device, self.size, self.dtype)
      self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None
      self.forced_realize = False
    else:
      # properties on view
      assert base.base == base, "base must be a base itself"
      self._base = base

  def __del__(self):
    if hasattr(self, 'buffer'): self.buffer.ref(-1)

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} {str(self.dtype)[7:]} {self.st if self.base is not self else (self.op, self.realized)}>"

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
  def metaop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Tuple[LazyBuffer, ...]=(), enable_cache=False) -> LazyBuffer:
    assert isinstance(src, tuple)
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, src, enable_cache=enable_cache)

  def const_like(self, b): return self.const_with_shape(b, self.shape)
  def const_with_shape(self, val:ConstType, shape:Tuple[sint,...]) -> LazyBuffer:
    assert isinstance(val, get_args(ConstType)), f"{val=} has {type(val)=}, not a ConstType"
    return LazyBuffer.metaop(Ops.CONST, tuple(), self.dtype, self.device, arg=val).reshape((1,)*len(shape)).expand(shape)

  @property
  def is_realized(self) -> bool: return self.base.realized is not None

  def assign(self, x:LazyBuffer) -> LazyBuffer:
    assert x.size == self.size, f"assign target must have same size {self.size=} != {x.size=}"
    assert self.is_realized, f"assign target must be realized {self}"
    return LazyBuffer.metaop(Ops.ASSIGN, self.shape, self.dtype, self.device, arg=None if self.st.contiguous else self.st,
                             src=(self, x), enable_cache=True) # NOTE: assign to VIEW is fine

  def can_view(self):
    return (self.st.consecutive and not self.is_unrealized_const() and not isinstance(self.dtype, ImageDType) and
            self.device.split(":")[0] in view_supported_devices)

  def contiguous(self, allow_buffer_view=True):
    if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():
      ret = self.alu(Ops.BUFFER_VIEW) if allow_buffer_view and self.can_view() else self.alu(Ops.CONTIGUOUS)
      if (sti := self.st.invert(self.base.shape)) is not None: self.base.contiguous_child = ref(ret), sti
      return ret
    self.base.forced_realize = True
    return self

  def bitcast(self, dtype:DType) -> LazyBuffer: return self.cast(dtype, bitcast=True)
  def cast(self, dtype:DType, bitcast:bool=False, allow_buffer_view=True) -> LazyBuffer:
    if self.dtype == dtype: return self
    if self.device.startswith("DISK") and not bitcast: raise RuntimeError("attempted to cast disk buffer (bitcast only)")
    if self.is_unrealized_unmasked_const() and not bitcast:
      return create_lazybuffer(self.device, self.st, dtype, Ops.CONST, dtypes.as_const(self.base.arg, dtype))
    new_shape = self.shape
    if bitcast and self.dtype.itemsize != dtype.itemsize:
      if not self.device.startswith("DISK"): raise RuntimeError("shape changing bitcast only supported on DISK right now")
      if not all_int(new_shape): raise RuntimeError("shape changing bitcast with symbolic shape isn't supported yet")
      # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
      if not (new_shape[-1]*self.dtype.itemsize) % dtype.itemsize == 0: raise RuntimeError("unsupported size in bitcast")
      new_shape = new_shape[:-1] + ((new_shape[-1]*self.dtype.itemsize) // dtype.itemsize,)
    elif getenv("CAST_BEFORE_VIEW", 1) and dtype.itemsize <= self.dtype.itemsize and self is not self.base:
      # TODO: applying this makes gpt2 slower
      return self.base.cast(dtype, bitcast)._view(self.st)
    cast_op: Ops = (Ops.BUFFER_VIEW if self.can_view() and allow_buffer_view else Ops.BITCAST) if bitcast else Ops.CAST
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), dtype, cast_op, None, (self,))

  def is_unrealized_const(self): return self.base.realized is None and self.base.op is Ops.CONST and not isinstance(self.base.arg, UOp)
  def is_unrealized_unmasked_const(self): return self.is_unrealized_const() and all(v.mask is None for v in self.st.views)

  def _copy(self, device:str) -> LazyBuffer:
    assert self.st.contiguous and self.size == self.base.size, f"can only copy contig {self} {self.base}"
    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, Ops.COPY, self.buffer.nbytes, (self,), enable_cache=False)

  def copy_to_device(self, device:str, force:bool=False, clone:bool=False) -> LazyBuffer: return self.base._copy(device)._view(self.st)

  def clone(self) -> LazyBuffer: return self.copy_to_device(self.device, clone=True)

  def alu(self, op:Ops, *in_srcs:LazyBuffer) -> LazyBuffer:
    srcs: List[LazyBuffer] = [self, *in_srcs]
    out_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else srcs[-1].dtype
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, None, tuple(srcs))

  # *** reduce ops ***

  def _reduce_op(self, op:Ops, axis:Tuple[int, ...]) -> LazyBuffer:
    assert all(0 <= x < len(self.shape) for x in axis), f"axis args {axis} out of range for shape {self.shape}"
    axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.st.reduce(axis)), self.dtype, Ops.REDUCE_AXIS, (op, axis), (self,))

  def r(self, op:Ops, axis:Tuple[int, ...]) -> LazyBuffer: return self._reduce_op(op, axis)

  # *** movement ops ***

  def _view(self, new_st:ShapeTracker) -> LazyBuffer:
    if self.st.size == 0 or (new_st.views[-1].mask is not None and any((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return self.const_with_shape(0, new_st.shape)
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self._view(self.st.permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self._view(self.st.stride(arg))

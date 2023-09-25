from __future__ import annotations
from typing import Optional, Union, cast, Tuple, Any, List, Dict, Mapping, Callable
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.ops import LazyOp, LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
ElementwiseOps = {*UnaryOps, *BinaryOps, *TernaryOps}

from tinygrad.ops import BufferOps, ConstBuffer, MemBuffer, Device
from tinygrad.helpers import DType, dtypes, all_int, dedup, DEBUG, getenv, prod
from tinygrad.runtime.lib import RawConst, RawBuffer, buf_is_kernel_arg
from tinygrad.runtime.ops_cpu import RawNumpyBuffer
from tinygrad.shape.symbolic import sint
from weakref import WeakSet
import numpy as np

OPT = getenv("OPT", 2)
MERGE_ELEMENTWISE_INTO_REDUCE = OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2


def _ast_reduceops(op:LazyOp) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = op.src[0]
  if not src.realized:
    assert isinstance(src.op, LazyOp), "if not src.realized, then src.op must be a LazyOp"
    if MERGE_ELEMENTWISE_INTO_REDUCE and src.op.op in ElementwiseOps and len(src.children) <= 1:
      src = src.op
  return LazyOp(op.op, (src,), op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(op:LazyBuffer, output_shape:Tuple[sint, ...]) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Optional[Union[LazyOp, LazyBuffer]]] = {x:None for x in op.buffers}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs: List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x.alias if x.alias is not None and x.alias.st.contiguous else x) for k,x in zip(real_srcs.keys(), real_srcs.keys()) if not x.realized and x.op.op in ReduceOps and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape: Tuple[sint, ...] = output_shape
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE and psrcs:
    psrc = psrcs[0] # NOTE: right now we can't handle multiple, as we'd have to check for loop
    if psrc[1].op.op in ReduceOps:
      top = _ast_reduceops(psrc[1].op)
    real_srcs[psrc[0]] = top
    real_srcs.update({x:x for x in top.buffers})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrc[0].shape != psrc[1].shape:
      intermediate_shape = psrc[1].shape
      assert psrc[0].shape == output_shape, f"shape mismatch {psrc[0].shape} != {output_shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None: real_srcs[x] = x.reshape(intermediate_shape)

  # NOTE: cast the type to remove the Optional
  return op.map_buffers(cast(Dict[LazyBuffer, Union[LazyOp, LazyBuffer]], real_srcs))

def _replace_bufferops(op:LazyOp) -> Tuple[LazyOp, List[LazyBuffer]]:
  replacements:Dict[LazyBuffer, LazyOp] = {}
  realized_bufs = dedup([x.realized for x in op.buffers if buf_is_kernel_arg(x)])
  for x in op.buffers:
    assert x.realized, "buffer isn't realized"
    if isinstance(x.realized, RawConst):
      replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(x.realized._buf, x.realized.dtype, x.st.simplify()))
    elif x.realized in realized_bufs:
      replacements[x] = LazyOp(BufferOps.MEM, (), MemBuffer(realized_bufs.index(x.realized)+1, x.realized.dtype, x.st.simplify()))
    else:
      raise NotImplementedError(f"not handled {x}")
  return op.map_buffers(replacements), realized_bufs

class LazyBuffer:
  def __init__(self, op:Optional[LazyOp], st:ShapeTracker, dtype:DType, device:str, src:Optional[RawBuffer]=None, alias:Optional[LazyBuffer]=None):
    self.st: ShapeTracker = st
    self.shape, self.dtype, self.device = self.st.shape, dtype, device
    self.output_buffer: Optional[RawBuffer] = None
    if alias:
      if alias.alias: alias = alias.alias  # no recurse
      self.alias: Optional[LazyBuffer] = alias
      alias.children.add(self)
    else:
      self.alias = None
      self._realized: Optional[RawBuffer] = src
      self._children: WeakSet = WeakSet()
      if op:
        self._op: LazyOp = op
        for x in op.buffers: x.children.add(self)

  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}

  # handle alias
  @property
  def op(self): return self.alias._op if self.alias else self._op
  @property
  def realized(self): return self.alias._realized if self.alias else self._realized
  @realized.setter
  def realized(self, val): self._realized = val
  @property
  def children(self): return self.alias._children if self.alias else self._children

  def contiguous(self) -> LazyBuffer:
    if self.st.contiguous: return self
    return LazyBuffer(LazyOp(UnaryOps.NOOP, (self,)), ShapeTracker.from_shape(self.shape), self.dtype, self.device)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    return LazyBuffer(LazyOp(op, tuple() if src is None else (src,), arg), ShapeTracker.from_shape(tuple(shape)), dtype, device)

  def const(self, val:Union[float, int]) -> LazyBuffer:
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer(None, ShapeTracker.from_shape(x.shape), dtypes.from_np(x.dtype), "CPU", src=RawNumpyBuffer.fromCPU(x))

  def toCPU(self) -> np.ndarray:
    assert self.dtype.np, f"{self.dtype} is not supported in toCPU"
    self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    realized = self_casted.contiguous().realize().realized
    assert all_int(self.shape), f"no toCPU if shape is symbolic, {self.shape=}"
    return cast(RawBuffer, realized).toCPU().reshape(self.shape)

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs = (self,)+srcs
    out_dtype = max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]
    return LazyBuffer(LazyOp(op, srcs, arg), ShapeTracker.from_shape(self.shape), out_dtype, self.device)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    return LazyBuffer(LazyOp(op, (self,), new_shape), ShapeTracker.from_shape(new_shape), self.dtype, self.device)

  def reshape(self, arg) -> LazyBuffer:
    return LazyBuffer(None, self.st.reshape(arg), self.dtype, self.device, alias=self)

  def expand(self, arg) -> LazyBuffer:
    return LazyBuffer(None, self.st.expand(arg), self.dtype, self.device, alias=self)

  def permute(self, arg) -> LazyBuffer:
    return LazyBuffer(None, self.st.permute(arg), self.dtype, self.device, alias=self)

  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      if self.alias: self.alias.realize()
      elif self.op.op in LoadOps: LOAD_OPS_DISPATCHER[cast(LoadOps, self.op.op)](self)
      else:
        op = self.op
        if op.op in ElementwiseOps: op = _ast_binaryops(op, self.shape)
        elif op.op in ReduceOps: op = _ast_reduceops(op)

        for x in self.op.buffers: x.realize()
        op, realized_bufs = _replace_bufferops(op)
        self.realized = Device[self.device].exec_ast(op, output=self, inputs=realized_bufs, var_vals={}, **self._device_extra_args())

        #from extra.utils import print_tree
        #print_tree(op)

    return self

def _realize_from(buffer: LazyBuffer) -> None:
  rawbuf = buffer.op.src[0].realize()
  assert rawbuf.realized, "realize failed?"
  if DEBUG >= 3: print(f"*** copy {buffer.device} <- {rawbuf.device} size {rawbuf.realized.size} dtype {rawbuf.realized.dtype}")
  buffer.realized = Device[buffer.device].buffer.fromCPU(rawbuf.toCPU(), **buffer._device_extra_args())

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.FROM: _realize_from,
}
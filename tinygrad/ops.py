from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Union, Type, NamedTuple, Tuple, Any, List, ClassVar, Optional, Callable, Dict
import functools, operator
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape import ShapeTracker

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
class UnaryOps(Enum): NOOP = auto(); NEG = auto(); EXP = auto(); LOG = auto(); NOT = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); FLIP = auto(); PAD = auto(); SHRINK = auto() # noqa: E702
class FusedOps(Enum): MULACC = auto() # noqa: E702
class LoadOps(Enum): FROMCPU = auto(); CONTIGUOUS = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[FusedOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])
def map_buffers(real_srcs, x:LazyOp) -> LazyOp:
  if x in real_srcs: return map_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple((map_buffers(real_srcs, y) if isinstance(y, LazyOp) else real_srcs[y]) for y in x.src), x.arg)

# a placeholder class to extend by the exec classes
class DeviceBuffer:
  _buf: Any                # underlying buffer
  shape: Tuple[int, ...]
  @staticmethod
  def fromCPU(x:np.ndarray) -> DeviceBuffer: raise NotImplementedError("must be implemented")
  def toCPU(self:DeviceBuffer) -> np.ndarray: raise NotImplementedError("must be implemented")
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer=None): raise NotImplementedError("must be implemented")

# this is a quick "buffer" class for flop tracking
class GenericShape(NamedTuple):
  shape : Tuple[int, ...]
  flops : int = 0
shape_fxn_for_op : Dict[Op, Callable] = {
  **{op:lambda self: GenericShape(self.shape, self.flops + prod(self.shape)) for op in UnaryOps},
  **{op:lambda self,y: GenericShape(self.shape, self.flops + y.flops + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: GenericShape(new_shape, self.flops + prod(self.shape)) for op in ReduceOps},
  **{op:functools.partial(lambda mop,self,arg: GenericShape(ShapeTracker(self.shape).movement_op(mop, arg).shape, self.flops), op) for op in MovementOps}}

# used in CPUBuffer and TorchBuffer
class GenericExecAST(DeviceBuffer):  # pylint: disable=abstract-method
  fxn_for_op : ClassVar = shape_fxn_for_op
  # TODO: use generic types here to remove __init__ in specialized classes
  def __init__(self, lbuf:Any): self.buf, self.shape = lbuf, tuple(lbuf.shape)
  def contiguous(self): return type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg=None): return type(self)(self.fxn_for_op[op](self.buf, arg)) if op in self.fxn_for_op else type(self)(getattr(self.buf, op.name.lower())(arg))
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[GenericExecAST]=None):
    if FusedOps.MULACC in cls.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, ast.src[0].src, ast.arg)
    srcs = [cls.exec_ast(x) if isinstance(x, LazyOp) else x for x in ast.src]
    if DEBUG >= 4: print("exec_ast", ast.op, [x.shape for x in srcs], ast.arg)
    if ast.op in BinaryOps: assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
    if ast.op in ReduceOps: assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
    if ast.op in MovementOps: ret = srcs[0].movement_op(ast.op, ast.arg)
    else: ret = cls(cls.fxn_for_op[ast.op](*([x.buf for x in srcs] + ([ast.arg] if ast.arg else []))))
    if output_buffer is not None:
      assert output_buffer.shape == ret.shape
      output_buffer.buf = ret.buf
      return output_buffer
    else:
      return ret
def get_lazyop_info(ast:LazyOp): return GenericExecAST.exec_ast(map_buffers({x:GenericExecAST(GenericShape(x.shape)) for x in get_buffers(ast)}, ast)).buf

# assumes you are using ShapeTracker
# used in GPUBuffer and LLVMBuffer
class ExplicitExecAST(DeviceBuffer):  # pylint: disable=abstract-method
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None, backing:Optional[np.ndarray]=None, force_create=False):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._buf : Any = hostbuf._buf if hostbuf is not None else None
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if (self._backing is not None and self._backing.shape != (1,)) or force_create:
      self._load()

  def create(self, base_shape: Tuple[int, ...]) -> Any: raise NotImplementedError("must be implemented")
  def copyin(self, source: np.ndarray): raise NotImplementedError("must be implemented")
  def copyout(self, cl_buf: ExplicitExecAST, dest: np.ndarray): raise NotImplementedError("must be implemented")

  def _load(self) -> Any:
    if self._buf is None:
      self._buf = self.create(self._base_shape)
    assert self._buf is not None
    if self._backing is not None:
      self.copyin(self._backing)
      self._backing = None
    return self._buf

  def toCPU(self) -> np.ndarray:
    buf_contiguous = self.contiguous()
    buf_contiguous._load() # force buffer creation, happens if it's a backed buffer that hasn't been created yet
    data = np.empty(self.shape, dtype=np.float32)
    self.copyout(buf_contiguous, data)
    return data

  # universal for shape tracked
  def contiguous(self): return self if self.st.contiguous and prod(self._base_shape) == prod(self.shape) else type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)

class GlobalCounters:
  global_ops : ClassVar[int] = 0
  global_mem : ClassVar[int] = 0
  time_sum : ClassVar[int] = 0
  kernel_count : ClassVar[int] = 0
  mem_used : ClassVar[int] = 0   # NOTE: this is not reset
  cache : ClassVar[Optional[list]] = None
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0,0,None
  @staticmethod
  def log_kernel(op_estimate:int, mem_estimate:int):
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += mem_estimate
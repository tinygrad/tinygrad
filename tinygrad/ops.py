from __future__ import annotations
import functools, itertools, operator, random
import numpy as np
from enum import Enum, auto
from typing import Union, Type, NamedTuple, Tuple, Any, List, ClassVar, Optional, Callable, Dict, TypeVar, Set, Final
from tinygrad.helpers import prod, DEBUG, getenv, DType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker, MovementOps

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
class UnaryOps(Enum): NOOP = auto(); NEG = auto(); EXP = auto(); LOG = auto(); NOT = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class FusedOps(Enum): MULACC = auto() # noqa: E702
class LoadOps(Enum): FROMCPU = auto(); CONTIGUOUS = auto(); TOCPU = auto(); CUSTOM = auto() # noqa: E702

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
def map_buffers(real_srcs:Dict[Any, Any], x:Any) -> LazyOp:
  if x in real_srcs: return map_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple((map_buffers(real_srcs, y) if isinstance(y, LazyOp) else real_srcs[y]) for y in x.src), x.arg)

_T = TypeVar("_T")
class Copyable:
  @classmethod
  def fromCPU(cls:Type[_T], x:np.ndarray) -> _T: raise NotImplementedError("must be implemented")
  def toCPU(self:Copyable) -> np.ndarray: raise NotImplementedError("must be implemented")

class RawBuffer(Copyable):  # pylint: disable=abstract-method
  def __init__(self, size:int, dtype:DType):
    self.size: int = size
    self.dtype: DType = dtype
    self._memsz: int = size*dtype.itemsize
    GlobalCounters.mem_used += self._memsz
  def __del__(self): GlobalCounters.mem_used -= self._memsz

class RawBufferCopyIn(RawBuffer):
  def copyin(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  @classmethod
  def fromCPU(cls, x:np.ndarray):
    ret = cls(prod(x.shape), dtypes.from_np(x))
    ret.copyin(x)
    return ret

class RawBufferMapped(RawBufferCopyIn):
  def _buffer(self) -> memoryview: raise NotImplementedError("must be implemented")
  def toCPU(self) -> np.ndarray: return np.frombuffer(self._buffer(), dtype=self.dtype.np)
  def copyin(self, x:np.ndarray) -> None: np.copyto(self.toCPU(), x.reshape(-1))

class RawBufferCopyInOut(RawBufferCopyIn):
  def copyout(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  def toCPU(self) -> np.ndarray:
    x: np.ndarray = np.empty(self.size, dtype=self.dtype.np)
    self.copyout(x)
    return x

# a placeholder class to extend by the exec classes
class DeviceBuffer(Copyable):
  _buf: Any                # underlying buffer
  shape: Tuple[int, ...]
  dtype: DType
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer=None): raise NotImplementedError("must be implemented")

# this is a quick "buffer" class for flop tracking and getting the output shape
class GenericShape:
  def __init__(self, shape:Tuple[int, ...], dtype:DType=dtypes.float32, flops:int=0): self.shape, self.dtype, self.flops = shape, dtype, flops
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
shape_fxn_for_op: Dict[Op, Callable] = {
  **{op:lambda self: GenericShape(self.shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in UnaryOps},
  **{op:lambda self,y: GenericShape(self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: GenericShape(new_shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  **{op:functools.partial(lambda mop,self,arg: GenericShape(ShapeTracker(self.shape).movement_op(mop, arg).shape, self.dtype, self.consume_flops()), op) for op in MovementOps}}
def get_lazyop_info(ast:LazyOp) -> GenericShape: return InterpretedBuffer.exec_ast(map_buffers({x:InterpretedBuffer(GenericShape(x.shape, x.dtype)) for x in get_buffers(ast)}, ast))._buf

# used in CPUBuffer and TorchBuffer
class InterpretedBuffer(DeviceBuffer):  # pylint: disable=abstract-method
  fxn_for_op: ClassVar = shape_fxn_for_op
  def __init__(self, lbuf:Any):
    self._buf: Any = lbuf
    self.shape: Tuple[int, ...] = tuple(lbuf.shape)
    self.dtype: DType = self.to_tinygrad_dtype() if hasattr(self, 'to_tinygrad_dtype') else lbuf.dtype
    # NOTE: this is overcounting the memory used, as reshapes and stuff are aliases
    self._memsz = (prod(self.shape) * self.dtype.itemsize) if not isinstance(lbuf, GenericShape) else 0
    GlobalCounters.mem_used += self._memsz
  def __del__(self): GlobalCounters.mem_used -= self._memsz
  def contiguous(self): return type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg=None): return type(self)(self.fxn_for_op[op](self._buf, arg)) if op in self.fxn_for_op else type(self)(getattr(self._buf, op.name.lower())(arg))
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[InterpretedBuffer]=None, context=None):
    if FusedOps.MULACC in cls.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, ast.src[0].src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if ast in context: return context[ast]
    srcs = [cls.exec_ast(x, context=context) if isinstance(x, LazyOp) else x for x in ast.src]
    if ast.op in BinaryOps: assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
    if ast.op in ReduceOps: assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
    if ast.op in MovementOps: ret = srcs[0].movement_op(ast.op, ast.arg)
    else: ret = cls(cls.fxn_for_op[ast.op](*([x._buf for x in srcs] + ([ast.arg] if ast.arg else []))))
    if DEBUG >= 4 or (not isinstance(srcs[0]._buf, GenericShape) and DEBUG >= 3):
      print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB op: {ast.op:20s} out({ret.dtype.name}): {str(ret.shape):30s} in({len(srcs)}):", list(set(x.shape for x in srcs)), ast.arg if ast.arg is not None else "")
    context[ast] = ret
    if output_buffer is not None:
      assert output_buffer.shape == ret.shape, output_buffer.dtype == ret.dtype
      output_buffer._buf = ret._buf
      return output_buffer
    else:
      return ret

class ASTRunner:
  def __init__(self, name, prg, bufs_to_delete:Optional[Set[int]]=None, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0):
    if DEBUG >= 4: print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.bufs_to_delete, self.op_estimate, self.mem_estimate = name, prg, global_size, local_size, bufs_to_delete if bufs_to_delete else set(), op_estimate, mem_estimate

  def build(self, runtime):
    self.clprg = runtime(self.name, self.prg)
    return self

  def exec(self, bufs:List[Optional[CompiledBuffer]]) -> Optional[float]:
    rawbufs = [x.raw() for i,x in enumerate(bufs) if x is not None and i not in self.bufs_to_delete]
    if getenv("OPTLOCAL") and self.global_size is not None and self.local_size is None: self.local_size = self.optimize_local_size(rawbufs)
    if GlobalCounters.cache is not None: GlobalCounters.cache.append((self, rawbufs))
    return self(rawbufs)

  def __call__(self, rawbufs:List[RawBuffer]) -> Optional[float]:
    if et := self.clprg(self.global_size, self.local_size, *rawbufs, wait=DEBUG>=2): GlobalCounters.time_sum_s += et
    if DEBUG >= 1:
      print(f"*** {GlobalCounters.kernel_count:4d} {self.name:20s} arg {len(rawbufs):3d} sz {str(self.global_size):18s} {str(self.local_size):12s} OPs {self.op_estimate/1e6:7.1f}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({self.op_estimate/(et*1e9):8.2f} GFLOPS, {self.mem_estimate/(et*1e9):6.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += self.op_estimate
    GlobalCounters.global_mem += self.mem_estimate
    if getenv("EARLY_STOPPING") and GlobalCounters.kernel_count == getenv("EARLY_STOPPING"): exit(0)
    return et

  def timeit(self, rawbufs:List[RawBuffer], local_override=None) -> float:
    try: return self.clprg(self.global_size, local_override if local_override is not None else self.local_size, *rawbufs, wait=True)
    except Exception: return float('inf')

  def optimize_local_size(self, rawbufs:List[RawBuffer]) -> List[int]:
    assert self.global_size is not None, "needs a global size to optimize local size"
    if any(x == rawbufs[0] for x in rawbufs[1:]):  # this is an assignment, replace the output buffer
      output_replacement = type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype)
      rawbufs = [output_replacement if x == rawbufs[0] else x for x in rawbufs]
    MAX_WORKGROUP = self.clprg.max_work_group_size() if hasattr(self.clprg, 'max_work_group_size') else 1024
    local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in self.global_size]
    local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
    return min([(self.timeit(rawbufs, local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])[1]

from tinygrad.codegen.ast import ASTKernel
class Specialized(NamedTuple):
  raw_buffer: Type[RawBuffer]
  codegen: Type[ASTKernel]
  runtime: Type

# assumes you are using ShapeTracker
# used in GPUBuffer and LLVMBuffer
class CompiledBuffer(DeviceBuffer):  # pylint: disable=abstract-method
  spec: ClassVar[Specialized]

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[CompiledBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False, dtype:DType=dtypes.float32):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self.dtype = dtype
    assert hostbuf is None or hostbuf.dtype == dtype, f"hostbuf dtype {hostbuf.dtype} != {dtype}"
    self._base_shape: Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._buf = hostbuf._buf if hostbuf is not None else None
    self._backing: Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    assert self._backing is None or dtypes.from_np(self._backing) == dtype, f"backing dtype {dtypes.from_np(self._backing)} != {dtype}"
    if (self._backing is not None and self._backing.shape != (1,)) or force_create: self.raw()

  def __repr__(self): return f"{type(self).__name__}(shape={self.st}, hostbuf={type(self).__name__}(shape={self._base_shape}" + (f", backing=np.array({self._backing}, dtype=np.{self.dtype.np.__name__}), dtype={self.dtype}), dtype={self.dtype})" if self._backing is not None else f", force_create=True, dtype={self.dtype}), dtype={self.dtype})")

  def create_raw_buffer(self, shape:Tuple[int, ...], backing:Optional[np.ndarray], dtype:DType) -> RawBuffer:
    assert backing is None or prod(shape) == prod(backing.shape), "backing has the wrong shape"
    assert backing is None or GlobalCounters.cache is None, f"can't copy in {backing.shape} while caching"
    if DEBUG >= 4: print(f"create raw buffer {shape} {dtype} backed:{backing is not None}")
    return self.spec.raw_buffer(prod(shape), dtype) if backing is None else self.spec.raw_buffer.fromCPU(backing)

  def raw(self) -> RawBuffer:
    if self._buf is None:
      if DEBUG >= 4 and self._backing is not None: print(f"**** copy in {self._backing.shape} to {type(self)}")
      self._buf = self.create_raw_buffer(self._base_shape, self._backing, self.dtype)
      self._backing = None
    return self._buf

  @classmethod
  def fromCPU(cls, x:np.ndarray) -> CompiledBuffer: return cls(x.shape, backing=x.ravel(), dtype=dtypes.from_np(x))
  def toCPU(self) -> np.ndarray:
    assert GlobalCounters.cache is None, f"can't copy out {self} while caching"
    if DEBUG >= 3: print(f"**** copy out {self.shape}")
    return self.contiguous().raw().toCPU().reshape(self.shape)

  method_cache: Final[Dict[str, ASTRunner]] = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[CompiledBuffer]=None):
    k = cls.spec.codegen(ast, output_buffer)
    if getenv("ENABLE_METHOD_CACHE", 1):  # this is the default now
      if k.key not in cls.method_cache: cls.method_cache[k.key] = k.codegen().build(cls.spec.runtime)
      elif DEBUG >= 4: print(f"method cache hit : {k.key}")
      prg = cls.method_cache[k.key]
    else:
      prg = k.codegen().build(cls.spec.runtime)
    if getenv("PRINT_AST", "") == prg.name or getenv("PRINT_AST", "") == "1":
      k.print()
      print(prg.prg)
    prg.exec(k.bufs)
    return k.ret

  # universal for shape tracked
  def contiguous(self): return self if self.st.contiguous and prod(self._base_shape) == prod(self.shape) else type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), hostbuf=self, dtype=self.dtype)

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  cache: ClassVar[Optional[List[Tuple[Callable, Any]]]] = None
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0.0,0,None

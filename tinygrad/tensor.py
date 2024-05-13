# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools
from contextlib import ContextDecorator
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, DefaultDict, cast, get_args, Set
from collections import defaultdict
import numpy as np

from tinygrad.dtype import DType, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype
from tinygrad.helpers import argfix, make_pair, flatten, prod, all_int, round_up, merge_dicts, fully_flatten, argsort, IMAGE, DEBUG, WINO, THREEFRY
from tinygrad.helpers import getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.device import Buffer, BufferOptions
from tinygrad.device import Device
from tinygrad.shape.symbolic import sint, Variable, MulNode, Node
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars

# **** start with two base classes, Tensor and Function ****

class Function:
  def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor.__new__(Tensor)
    ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

import tinygrad.function as F

def _loadop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Tuple[LazyBuffer, ...]=()):
  if isinstance(device, str): return LazyBuffer.loadop(op, shape, dtype, device, arg, src)
  return MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype, d, arg, src) for d in device], None)

def _fromcpu(x: np.ndarray) -> LazyBuffer:
  ret = LazyBuffer.loadop(LoadOps.EMPTY, x.shape, dtypes.from_np(x.dtype), "NPY")
  # fake realize
  ret.buffer.allocate(x)
  del ret.srcs
  return ret

def _get_winograd_matcols(mat, dims:int, shp:Tuple[sint, ...], device:Union[str, Tuple[str, ...]]) -> List[List[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

def _pad_left(*shps:Tuple[sint, ...], v=1): return tuple((v,) * (max(len(i_) for i_ in shps) - len(i)) + i for i in shps)
def broadcast_shape(*shps:Tuple[sint, ...]): return tuple(0 if any(sh_ == 0 for sh_ in sh) else max(sh) for sh in zip(*_pad_left(*shps)))

class Tensor:
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor
  ```
  """
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  class train(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  no_grad: ClassVar[bool] = False
  class inference_mode(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.no_grad = Tensor.no_grad, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.no_grad = self.prev
  def __init__(self, data:Union[None, ConstType, List, Tuple, LazyBuffer, np.ndarray, bytes, MultiLazyBuffer, Variable],
               device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
    assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    # tensors have gradients, buffers do not
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx: Optional[Function] = None
    if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
    elif isinstance(data, get_args(ConstType)): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, Variable): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data.unbind()[1]), device, data)
    elif isinstance(data, bytes): data = _fromcpu(np.frombuffer(data, np.uint8))
    elif data is None: data = _loadop(LoadOps.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, list):
      if dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float
      if dtype == dtypes.bfloat16: data = Tensor(_fromcpu(np.array(data, np.float32)), device=device).cast(dtypes.bfloat16).lazydata
      else: data = _fromcpu(np.array(data, dtype.np))
    elif isinstance(data, np.ndarray):
      if data.shape == (): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
      else: data = _fromcpu(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

    # data is a LazyBuffer, but it might be on the wrong device
    if not isinstance(data, (LazyBuffer, MultiLazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
    if isinstance(device, tuple):
      # TODO: what if it's a MultiLazyBuffer on other devices?
      self.lazydata: Union[LazyBuffer, MultiLazyBuffer] = MultiLazyBuffer.from_sharded(data, device, None) if isinstance(data, LazyBuffer) else data
    else:
      self.lazydata = data if data.device == device else data.copy_to_device(device)

  def __repr__(self): return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self): return self.shape[0] if len(self.shape) else 1

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  def schedule_with_vars(self, *lst:Tensor, seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
    """Create the schedule needed to realize these Tensor(s), with Variables."""
    if getenv("FUZZ_SCHEDULE"):
      from test.external.fuzz_schedule import fuzz_schedule
      fuzz_schedule(flatten([x.lazydata.lbs for x in (self,)+lst]))
    schedule, var_vals = create_schedule_with_vars(flatten([x.lazydata.lbs for x in (self,)+lst]), seen)
    return memory_planner(schedule), var_vals

  def schedule(self, *lst:Tensor, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
    """Create the schedule needed to realize these Tensor(s)."""
    schedule, var_vals = self.schedule_with_vars(*lst, seen=seen)
    assert len(var_vals) == 0
    return schedule

  def realize(self, *lst:Tensor) -> Tensor:
    """Trigger the computation needed to create these Tensor(s)."""
    run_schedule(*self.schedule_with_vars(*lst))
    return self

  def replace(self, x:Tensor) -> Tensor:
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert not x.requires_grad and getattr(self, '_ctx', None) is None
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.lazydata = x.lazydata
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="NPY", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x.numpy().data)
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata is x.lazydata: return self  # a self assign is a NOOP
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
    assert not isinstance(self.lazydata, MultiLazyBuffer) or self.lazydata.axis == x.lazydata.axis, "axis must match on MultiLazyBuffer"
    assert not x.requires_grad  # self requires_grad is okay?
    if not self.lazydata.is_realized(): return self.replace(x)
    self.lazydata = self.lazydata.assign(x.lazydata)
    return self
  def detach(self) -> Tensor: return Tensor(self.lazydata, device=self.device, requires_grad=False)

  def _data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0))
    # NOTE: this realizes on the object from as_buffer being a Python object
    cpu = self.cast(self.dtype.scalar()).contiguous().to("CLANG").realize()
    buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)
    if self.device != "CLANG": buf.options = BufferOptions(nolru=True)
    return buf.as_buffer(allow_zero_copy=True if self.device != "CLANG" else False)

  def data(self) -> memoryview:
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._data().cast(self.dtype.fmt, self.shape)
  def item(self) -> ConstType:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert self.numel() == 1, "must have one element for item"
    return self._data().cast(self.dtype.fmt)[0]
  # TODO: should be Tensor.tolist() -> Union[List[ConstType], ConstType]. The List is Sequence because mypy expects memoryview.tolist() -> list[int]
  # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
  def tolist(self) -> Union[Sequence[ConstType], ConstType]:
    """
    Returns the value of this tensor as a nested list.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    """
    return self.data().tolist()
  def numpy(self) -> np.ndarray:
    """
    Returns the value of this tensor as a numpy array.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.numpy())
    ```
    """
    if self.dtype == dtypes.bfloat16: return self.float().numpy()
    assert self.dtype.np is not None, f"no np dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return np.frombuffer(self._data(), dtype=self.dtype.np).reshape(self.shape)

  def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.lazydata, device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret

  def to_(self, device:Optional[Union[str, Tuple[str, ...]]]):
    real = self.to(device)
    # TODO: is this assign?
    if self.grad is not None and real.grad is not None: self.grad.lazydata = real.grad.lazydata
    self.lazydata = real.lazydata

  def shard(self, devices:Tuple[str, ...], axis:Optional[int]=None) -> Tensor:
    assert isinstance(self.lazydata, LazyBuffer), "can't shard a MultiLazyBuffer"
    canonical_devices = tuple(Device.canonicalize(x) for x in devices)
    if axis is not None and axis < 0: axis += len(self.shape)
    return Tensor(MultiLazyBuffer.from_sharded(self.lazydata, canonical_devices, axis), device=canonical_devices, requires_grad=self.requires_grad)

  def shard_(self, devices:Tuple[str, ...], axis:Optional[int]=None):
    self.lazydata = self.shard(devices, axis).lazydata
    return self

  @staticmethod
  def from_node(y:Node, **kwargs) -> Tensor:
    if isinstance(y, MulNode): return Tensor.from_node(y.a, **kwargs) * y.b
    if isinstance(y, Variable): return Tensor(y, **kwargs, requires_grad=False)
    raise RuntimeError(f"unhandled Node {y}")

  # ***** creation llop entrypoint *****

  @staticmethod
  def _loadop(op, shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
    if isinstance(device, tuple):
      return Tensor(MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(d), arg) \
                                      for d in device], None), device, dtype, **kwargs)
    return Tensor(LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(device), arg), device, dtype, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs):
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor._loadop(LoadOps.EMPTY, argfix(*shape), **kwargs)

  _seed: int = int(time.time())
  _rng_counter: Optional[Tensor] = None
  @staticmethod
  def manual_seed(seed=0):
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor._seed)
    ```
    """
    Tensor._seed, Tensor._rng_counter = seed, Tensor([0], dtype=dtypes.uint32, requires_grad=False)

  @staticmethod
  def rand(*shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DType]=None, **kwargs):
    """
    Creates a tensor with the given shape, filled with random values between the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    if Tensor._rng_counter is None: Tensor._rng_counter = Tensor([0], dtype=dtypes.uint32, requires_grad=False)
    if not THREEFRY.value:
      # for bfloat16, numpy rand passes buffer in float
      if (dtype or dtypes.default_float) == dtypes.bfloat16:
        return Tensor.rand(*shape, **kwargs, device=device, dtype=dtypes.float).cast(dtypes.bfloat16)
      return Tensor._loadop(LoadOps.CUSTOM, argfix(*shape), arg=custom_random, device=device, dtype=dtype, **kwargs)

    # threefry
    if (num := prod((shape:=argfix(*shape)))) == 0: return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
    counts = (Tensor.arange(num, device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._rng_counter.to(device)).realize().pad(((0,num%2),))
    Tensor._rng_counter.assign(Tensor._rng_counter + num).realize()

    rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
    ks = [0x0, Tensor._seed ^ 0x0 ^ 0x1BD11BDA, Tensor._seed]

    x = [(c := counts.chunk(2))[0] + ks[-1], c[1] + ks[0]]
    for i in range(5):
      for r in rotations[i % 2]: x[0], x[1] = (x0 := x[0] + x[1]), x0 ^ ((x[1] * (2 ** r)) + (x[1].div(2 ** (32 - r), upcast=False)))
      x = [(x[0] + ks[i % 3]), (x[1] + ks[(i + 1) % 3] + i + 1)]
    out = x[0].cat(x[1])[:num].div(2 ** 8, upcast=False).cast(dtypes.float32).div(2 ** 24)
    out = out.reshape(shape).cast(dtypes.default_float if dtype is None else dtype)
    out.requires_grad = kwargs.get("requires_grad")
    return out.contiguous()

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:ConstType, **kwargs):
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.full((2, 3), 42)
    print(t.numpy())
    ```
    """
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs):
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(t.numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs):
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs):
    """
    If `stop` is not specified, creates a tensor with the given shape, filled with values from `0` to `start` with the given step size.

    If `stop` is specified, creates a tensor with the given shape, filled with values from `start` to `stop` with the given step size.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(5)
    print(t.numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(5, 10)
    print(t.numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(5, 10, 2)
    print(t.numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    assert all(isinstance(s, (int, float)) for s in (start, stop, step)), f"symbolic arange not supported {start=}, {stop=}, {step=}"
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs)._cumsum() + (start - step)).cast(dtype)

  @staticmethod
  def eye(dim:int, **kwargs):
    """
    Creates an identity matrix of the given dimension.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.eye(3)
    print(t.numpy())
    ```
    """
    return Tensor.ones((dim,1),**kwargs).pad((None,(0,dim))).flatten().shrink(((0,dim*dim),)).reshape(dim, dim)

  def full_like(self, fill_value:ConstType, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with the given value.
    If `dtype` is not specified, the dtype of `tensor` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    ot = Tensor.ones(2, 3)
    t = Tensor.full_like(ot, 42)
    print(t.numpy())
    ```
    """
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  def zeros_like(self, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    ot = Tensor.ones(2, 3)
    t = Tensor.zeros_like(ot)
    print(t.numpy())
    ```
    """
    return self.full_like(0, **kwargs)
  def ones_like(self, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    ot = Tensor.zeros(2, 3)
    t = Tensor.ones_like(ot)
    print(t.numpy())
    ```
    """
    return self.full_like(1, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    """
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randint(2, 3, low=5, high=10)
    print(t.numpy())
    """
    assert dtypes.is_int(dtype := kwargs.pop("dtype", dtypes.int32)), f"Unsupported dtype {dtype} for randint"
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given mean and standard deviation.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=10, std=2)
    print(t.numpy())
    ```
    """
    return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution with the given lower and upper bounds.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.uniform(2, 3, low=2, high=10)
    print(t.numpy())
    ```
    """
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values
    from a uniform distribution with a mean of zero and a standard deviation of `(prod(shape)**-0.5`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.scaled_uniform(2, 3)
    print(t.numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.glorot_uniform(2, 3)
    print(t.numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.kaiming_uniform(2, 3)
    print(t.numpy())
    ```
    """
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.kaiming_normal(2, 3)
    print(t.numpy())
    ```
    """
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1, device=self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: yield from _deepwalk(i, visited)
        yield node
    return list(_deepwalk(self, set()))

  def backward(self) -> Tensor:
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      if t0.grad is None: raise RuntimeError(f"tensor {t0} has no grad")
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
    return self

  # ***** movement mlops *****

  def view(self, *shape) -> Tensor: return self.reshape(shape)  # in tinygrad, view and reshape are the same thing
  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)])
    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self
  def expand(self, shape, *args) -> Tensor:
    return self._broadcast_to(tuple(sh if s==-1 or s is None else s for s, sh in zip(*(_pad_left(argfix(shape, *args), self.shape)))))
  def permute(self, order, *args) -> Tensor: return F.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args) -> Tensor: return F.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    if all(x is None or x == (0,s) for x,s in zip(arg, self.shape)): return self
    return F.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape)))
  def pad(self, arg:Tuple[Optional[Tuple[sint, sint]], ...], value:float=0.0) -> Tensor:
    if all(x is None or x == (0,0) for x in arg): return self
    ret = F.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + F.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

  # ***** movement hlops *****

  # Supported Indexing Implementations:
  #   1. Int indexing (no copy)
  #     - for all dims where there's int, shrink -> reshape
  #     - negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  #     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  #     - Then we reshape (collapse) the int dim away such that for X: (1,1,9) -> (9,)
  #   2. Slice indexing (no copy)
  #     - for all dims where slice is start:end:stride, shrink -> Optional[flip] -> pad -> reshape -> shrink
  #     - first shrink the Tensor to X.shrink(((start, end),))
  #     - then we apply stride through Optional[flip] -> pad -> reshape -> shrink
  #       - flip where dim value is negative
  #       - pad 0's on dims such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] is possible
  #       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  #       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] and now you have your stride
  #   3. None indexing (no copy)
  #     - reshape (inject) a dim at the dim where there's None
  #   4. Tensor indexing (copy)
  #     - use Tensor.arange == tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  #     - combine masks together with mul
  #     - apply mask to self by mask * self
  #     - sum reduce away the extra dims added from creating masks
  # Tiny Things:
  #   1. Supported indices: Union[int, slice, Tensor, None, List, Tuple, Ellipsis]
  #     - for any list, List[Union[List, Tuple, int]], must have homogeneous shape
  #     - for any tuple, Tuple[Union[List, Tuple, int]], must have homogeneous shape
  #   2. Bool indexing is not supported
  #   3. Out of bounds Tensor indexing results in 0
  #     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 and 3 are OOB
  def __getitem__(self, indices) -> Tensor:
    # 1. indices normalization and validation
    # treat internal tuples and lists as Tensors and standardize indices to list type
    if isinstance(indices, list) and all_int(indices): indices = [Tensor(indices, self.device, requires_grad=False)]
    elif isinstance(indices, (tuple, list)):
      indices = [Tensor(list(i), self.device, requires_grad=False) if isinstance(i, (tuple, list)) else i for i in indices]
    else: indices = [indices]

    # turn scalar Tensors into const val for int indexing if possible
    indices = [self._to_const_val(i) if isinstance(i, Tensor) and i.shape == () else i for i in indices]
    # move Tensor indices to the same device as self
    indices = [i.to(self.device) if isinstance(i, Tensor) else i for i in indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    ellipsis_idx = [dim for dim, i in enumerate(indices) if i is Ellipsis]
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    indices[fill_idx:fill_idx+1] = [slice(None)] * (len(self.shape) - num_indices)

    # use Dict[type, List[dimension]] to track elements in indices
    type_dim: DefaultDict[Union[type, None], List[int]] = defaultdict(list)

    # record None for dimension injection later and filter None and record rest of indices
    type_dim[None] = [dim for dim, i in enumerate(indices) if i is None]
    indices_filtered = [v for v in indices if v is not None]
    for dim,i in enumerate(indices_filtered): type_dim[type(i)].append(dim)

    for index_type in type_dim:
      if index_type not in [None, int, slice, Tensor]: raise IndexError(f"{index_type=} not supported")
    if len(ellipsis_idx) > 1: raise IndexError("indices can only have a single ellipsis ('...')")
    if num_indices > self.ndim: raise IndexError(f"too many {num_indices=} for {self.ndim=}")

    # 2. basic indexing, uses only movement ops (no copy)
    # currently indices_filtered: Tuple[Union[slice, int, Tensor], ...]
    # turn indices in indices_filtered to Tuple[shrink_arg, strides]
    for dim in type_dim[int]:
      if (index := indices_filtered[dim]) >= (size := self.shape[dim]) or index < -size:
        raise IndexError(f"{index=} is out of bounds on {dim=} with {size=}")
      indices_filtered[dim] = ((index, index+1), 1) if index >= 0 else ((size+index, size+index+1), 1)
    for dim in type_dim[slice]:
      if (index := indices_filtered[dim]).step == 0: raise ValueError(f"{index=} on {dim=} cannot have 0 as step")
      s, e, st = index.indices(self.shape[dim])
      indices_filtered[dim] = ((0, 0) if (st * (e - s)) < 0 else (s, e) if st > 0 else (e+1, s+1), st)
    # record tensors and skip all Tensor dims for basic indexing
    tensor_index: List[Tensor] = []
    for dim in type_dim[Tensor]:
      tensor_index.append(index := indices_filtered[dim])
      if not dtypes.is_int(index.dtype): raise IndexError(f"{index.dtype=} on {dim=} is not supported, only int tensor indexing is supported")
      indices_filtered[dim] = ((0, self.shape[dim]), 1)

    new_slice, strides = ((),()) if not indices_filtered else zip(*indices_filtered)
    ret = self.shrink(new_slice).flip(tuple(i for i, s in enumerate(strides) if s < 0))
    if any(abs(s) != 1 for s in strides):
      strides = tuple(abs(s) for s in strides)
      ret = ret.pad(tuple((0, round_up(sh, s) - sh) for s, sh in zip(strides, ret.shape)))
      ret = ret.reshape(tuple(flatten((sh // s, s) for s, sh in zip(strides, ret.shape))))
      ret = ret.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in ret.shape[::2]))).reshape(ret.shape[::2])

    # inject 1 for dim where it's None and collapse dim for int
    new_shape = list(ret.shape)
    for dim in type_dim[None]: new_shape.insert(dim, 1)
    for dim in (dims_collapsed := tuple(dim + sum(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int]))): new_shape.pop(dim)

    ret = ret.reshape(new_shape)
    assert all_int(ret.shape), f"does not support symbolic shape {ret.shape}"

    # 3. advanced indexing (copy)
    if type_dim[Tensor]:
      # calculate dim of current ret by subtracting dims collapsed and adding dims injected up until tensor_dim
      def calc_dim(tensor_dim:int) -> int:
        return tensor_dim - sum(1 for d in dims_collapsed if tensor_dim >= d) + sum(1 for d in type_dim[None] if tensor_dim >= d)

      # track tensor_dim and tensor_index using a dict
      # calc_dim to get dim and use that to normalize the negative tensor indices
      idx: Dict[int,Tensor] = {(dim := calc_dim(td)):(tensor<0).where(ret.shape[dim],0) + tensor for td,tensor in zip(type_dim[Tensor], tensor_index)}

      masks, first_dim, last_dim = [], min(idx.keys()), max(idx.keys())
      pre_reduce_shape = ret.shape[:first_dim] + (big_shape := broadcast_shape(*(t.shape for t in idx.values()))) + ret.shape[first_dim:]

      # create masks
      for dim, i in idx.items():
        try: i = i.reshape(i.shape + (1,)*(ret.ndim - first_dim)).expand(pre_reduce_shape)
        except ValueError as e: raise IndexError(f"cannot broadcast indices: {e}") from e
        a = Tensor.arange(ret.shape[dim], device=self.device, requires_grad=False).reshape((ret.shape[dim],) + (1,)*(ret.ndim - dim - 1))
        masks.append(i == a)

      # reduce masks to 1 mask
      mask: Tensor = functools.reduce(lambda x,y: x.mul(y), masks)

      # inject 1's for the extra dims added in create masks
      sh = ret.shape[:first_dim] + (1,) * len(big_shape) + ret.shape[first_dim:]
      # sum reduce the extra dims introduced in create masks
      ret = (ret.reshape(sh) * mask).sum(tuple(i + len(big_shape) for i in idx.keys()), acc_dtype=ret.dtype)

      # special permute case
      if first_dim != 0 and len(idx) != 1 and tuple(idx.keys()) != tuple(range(first_dim, last_dim+1)):
        ret = ret.permute(*range(first_dim, first_dim+len(big_shape)), *range(0, first_dim), *range(first_dim+len(big_shape), ret.ndim))
    return ret

  def __setitem__(self, indices, v:Union[Tensor, ConstType]) -> None:
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      self.__getitem__(indices).assign(v)
      return
    # NOTE: check that setitem target is valid first
    assert all(lb.st.contiguous for lb in self.lazydata.lbs), "setitem target needs to be contiguous"
    if not isinstance(v, (Tensor, float, int, bool)): raise TypeError(f"can't set a {type(v).__name__} to a Tensor")
    if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
    if self.requires_grad or v.requires_grad: raise NotImplementedError("setitem with requires_grad is not supported")

    assign_to = self.realize().__getitem__(indices)
    # NOTE: contiguous to prevent const folding.
    v = v.cast(assign_to.dtype)._broadcast_to(broadcast_shape(assign_to.shape, v.shape)).contiguous()
    assign_to.assign(v).realize()

  # NOTE: using slice is discouraged and things should migrate to pad and shrink
  def slice(self, arg:Sequence[Optional[Tuple[int, sint]]], value:float=0) -> Tensor:
    arg_ = tuple(a if a is not None else (0, s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -l), max(0, r-s)) for s,(l,r) in zip(self.shape, arg_))
    return self.pad(padding, value=value).shrink(tuple((l + pl, r + pl) for (l,r),(pl,_) in zip(arg_, padding)))

  def gather(self:Tensor, idx:Tensor, dim:int) -> Tensor:
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"
    dim = self._resolve_dim(dim)
    idx = idx.to(self.device).transpose(0, dim).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(self.shape[dim], requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(
      tuple([*[(0,sh) for sh in idx.shape[1:-1]], None])).unsqueeze(0)).sum(-1, acc_dtype=self.dtype).transpose(0, dim)

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    dim = self._resolve_dim(dim)
    assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    cat_dims = [s.shape[dim] for s in catargs]
    cat_dim_cumsum = [0, *itertools.accumulate(cat_dims)]
    slc:List[List[Optional[Tuple[sint, sint]]]] = [[None for _ in self.shape] for _ in catargs]
    for d,k,s in zip(cat_dims, cat_dim_cumsum[:-1], slc): s[dim] = (k, cat_dim_cumsum[-1] - k - d)
    return functools.reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])

  @staticmethod
  def stack(tensors:Sequence[Tensor], dim:int=0) -> Tensor:
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors]
    # checks for shapes and number of dimensions delegated to cat
    return unsqueezed_tensors[0].cat(*unsqueezed_tensors[1:], dim=dim)

  def repeat(self, repeats:Sequence[int]) -> Tensor:
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

  def _resolve_dim(self, dim:int, *, outer:bool=False) -> int:
    if not -max(1, self.ndim+outer) <= dim < max(1, self.ndim+outer):
      raise IndexError(f"{dim=} out of range {[-max(1, self.ndim+outer), max(1, self.ndim+outer)-1]}")
    return dim + self.ndim+outer if dim < 0 else dim

  def split(self, sizes:Union[int, List[int]], dim:int=0) -> Tuple[Tensor, ...]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = self._resolve_dim(dim)
    if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
    assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
    return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])

  def chunk(self, num:int, dim:int=0) -> List[Tensor]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    assert num > 0, f"expect num to be greater than 0, got: {num}"
    dim = self._resolve_dim(dim)
    return list(self.split(math.ceil(self.shape[dim]/num) if self.shape[dim] else [0]*num, dim=dim))

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1:])

  def unsqueeze(self, dim:int) -> Tensor:
    dim = self._resolve_dim(dim, outer=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Sequence[int], value:float=0) -> Tensor:
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self.slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

  @property
  def T(self) -> Tensor: return self.transpose()
  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(self.ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)
  def flatten(self, start_dim=0, end_dim=-1):
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])
  def unflatten(self, dim:int, sizes:Tuple[int,...]):
    dim = self._resolve_dim(dim)
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim+1:])

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
    if self.ndim == 0:
      if axis is not None and axis not in [-1, 0]: raise IndexError(f"{axis=} out of range of [-1, 0]")
      axis = None
    axis_: Tuple[int, ...] = tuple(range(len(self.shape))) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
    axis_ = tuple(self._resolve_dim(x) for x in axis_)
    ret = fxn.apply(self, axis=axis_)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis_))

  def sum(self, axis=None, keepdim=False, acc_dtype:Optional[DType]=None):
    ret = self.cast(acc_dtype or sum_acc_dtype(self.dtype))._reduce(F.Sum, axis, keepdim)
    return ret.cast(self.dtype) if self.dtype in {dtypes.float16, dtypes.bfloat16} else ret
  def max(self, axis=None, keepdim=False): return self._reduce(F.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    return numerator.div(prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if si != so])).cast(output_dtype)
  def var(self, axis=None, keepdim=False, correction=1):
    assert all_int(self.shape), "does not support symbolic shape"
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(max(0, prod(self.shape)/prod(square_sum.shape)-correction))
  def std(self, axis=None, keepdim=False, correction=1): return self.var(axis, keepdim, correction).sqrt()

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  def logsumexp(self, axis=None, keepdim=False):
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + m.squeeze(axis)

  def argmax(self, axis=None, keepdim=False):
    # NOTE: return the first index if there are multiple occurrences of the maximum values
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape)
      return (prod(self.shape) - idx.max() - 1).cast(dtypes.int32)
    axis = self._resolve_dim(axis)
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1).cast(dtypes.int32)
  def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)

  @staticmethod
  def einsum(formula:str, *raw_xs, acc_dtype:Optional[DType]=None) -> Tensor:
    xs:Tuple[Tensor] = argfix(*raw_xs)
    formula = formula.replace(" ", "")
    inputs_str, output = formula.split("->") if "->" in formula else (formula, sorted(formula))
    inputs = [x for x in cast(str,inputs_str).split(',')]
    assert len(xs) == len(inputs), f"number of inputs doesn't match number of operands in formula, expected {len(inputs)}, got {len(xs)}"

    # map the value of each letter in the formula
    letter_val = sorted(merge_dicts([{letter:dim for letter, dim in zip(letters, tensor.shape)} for letters, tensor in zip(inputs, xs)]).items())

    xs_:List[Tensor] = []
    lhs = [sorted(enumerate(s), key=lambda e:e[1]) for s in inputs]
    for x,(order,letters) in zip(xs, [list(zip(*l)) for l in lhs]):
      # permute to the sorted letter order, then reshape/expand to create dimensions for the missing letters
      xs_.append(x.permute(order).reshape([val if letter in letters else 1 for letter,val in letter_val]).expand([val for _,val in letter_val]))

    # determine the inverse permutation to revert back to original order
    rhs_letter_order = argsort(list(output))
    rhs_order = argsort(rhs_letter_order)

    # sum over all axes that's not in the output, then permute to the output order
    return functools.reduce(lambda a,b:a*b, xs_) \
      .sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in output],acc_dtype=acc_dtype).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop_, i_ = [None] * len(self.shape[:-len(k_)]), self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      # repeats such that we don't need padding
      xup = self.repeat([1]*len(noop_) + [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)])
      # slice by dilation
      xup = xup.shrink(tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])).reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      # handle stride
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))
      # permute to move reduce to the end
      return xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = self.pad(tuple(noop_ + [(0, max(0,o*s-i)) for i,o,s in zip(i_, o_, s_)])).shrink(tuple(noop_ + [(0,o*s) for o,s in zip(o_, s_)]))
    xup = xup.reshape(noop_ + flatten(((o,s) for o,s in zip(o_, s_))))
    xup = xup.shrink(noop_ + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(noop_)), *[len(noop_)+i*2 for i in range(len(i_))], *[len(noop_)+i*2+1 for i in range(len(i_))])

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape)+1))
    x, w = self, weight.unflatten(0, (groups, -1)).permute(0,2,1,*trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s>1 for s in stride):
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-p,(k-1)*d-p+op) for k,d,p,op in reversed(list(
      zip(HW, make_pair(dilation, len(HW)), make_pair(padding, len(HW)), make_pair(output_padding, len(HW)))))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype:Optional[DType]=None) -> Tensor:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])  # noqa: E501

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    d = self.pad2d(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def dot(self, w:Tensor, acc_dtype:Optional[DType]=None) -> Tensor:
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype))

  def matmul(self, x:Tensor, reverse=False, acc_dtype:Optional[DType]=None) -> Tensor:
    return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)

  def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor:
    pl_sz = self.shape[axis] - int(not _first_zero and self.shape[axis] != 0)
    return self.transpose(axis,-1).pad2d((pl_sz,0))._pool((self.shape[axis] or 1,)).sum(-1).transpose(axis,-1)
  def cumsum(self, axis:int=0) -> Tensor:
    # TODO: someday the optimizer will find this on it's own
    # for now this is a two stage cumsum
    SPLIT = 256
    if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
    ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
    ret = ret.unflatten(-1, (-1, SPLIT))._cumsum(-1)
    base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
    base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
    def fix(x:Tensor): return x.flatten(start_dim=-2)[..., -self.shape[axis]:].transpose(axis,-1)
    return fix(ret) + fix(base_add)

  @staticmethod
  def _tri(r:sint, c:sint, k:int=0, **kwargs) -> Tensor:
    assert all_int((r,c)), "does not support symbolic"
    if r == 0: return Tensor.zeros((r, c), **kwargs)
    return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r,c) <= Tensor.arange(-k, c-k, **kwargs).unsqueeze(0).expand(r,c)
  def triu(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k, device=self.device).where(self, 0)
  def tril(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k+1, device=self.device).where(0, self)

  # ***** mlops (unary) *****

  def logical_not(self): return F.Eq.apply(*self._broadcasted(False))
  def neg(self): return F.Neg.apply(self) if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self): return F.Contiguous.apply(self)
  def contiguous_backward(self): return F.ContiguousBackward.apply(self)
  def log(self): return F.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self): return self.log()/math.log(2)
  def exp(self): return F.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self): return F.Exp.apply(self*math.log(2))
  #def log2(self): return F.Log2.apply(self.cast(least_upper_float(self.dtype)))
  #def log(self): return self.log2() / math.log2(math.e)
  #def exp2(self): return F.Exp2.apply(self.cast(least_upper_float(self.dtype)))
  #def exp(self): return (self * math.log2(math.e)).exp2()
  def relu(self): return F.Relu.apply(self)
  def sigmoid(self): return F.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def sin(self): return F.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def sqrt(self): return F.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self): return self.reciprocal().sqrt()
  def cos(self): return ((math.pi/2)-self).sin()
  def tan(self): return self.sin() / self.cos()

  # ***** math functions (unary) *****

  def trunc(self: Tensor) -> Tensor: return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor: return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor: return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())
  def lerp(self, end: Tensor, weight: Union[Tensor, float]) -> Tensor: return self + (end - self) * weight
  def square(self): return self*self
  def clip(self, min_, max_): return self.maximum(min_).minimum(max_)
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return ((self.float()) / (self.float().abs() + 1e-12)).cast(self.dtype)
  def reciprocal(self): return F.Reciprocal.apply(self.cast(least_upper_float(self.dtype)))

  # ***** activation functions (unary) *****

  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def celu(self, alpha=1.0): return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish F.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sinh(self): return (self.exp() - self.neg().exp()) / 2
  def cosh(self): return (self.exp() + self.neg().exp()) / 2
  def atanh(self): return ((1 + self)/(1 - self)).log() / 2
  def asinh(self): return (self + (self.square() + 1).sqrt()).log()
  def acosh(self): return (self + (self.square() - 1).sqrt()).log()
  def hardtanh(self, min_val=-1, max_val=1): return self.clip(min_val, max_val)
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
  def mish(self): return self * self.softplus().tanh()
  def softplus(self, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()
  def softsign(self): return self / (1 + self.abs())

  # ***** broadcasted elementwise mlops *****
  def _broadcast_to(self, shape:Tuple[sint, ...]):
    reshape_arg, _ = _pad_left(self.shape, shape)
    if self.ndim > len(shape) or not all(sh in {s,1} or (s==0 and sh==1) for sh,s in zip(reshape_arg, shape)):
      raise ValueError(f"cannot broadcast tensor with shape={self.shape} to {shape=}")
    return F.Expand.apply(self.reshape(reshape_arg), shape=shape) if shape != self.shape else self

  def _broadcasted(self, y:Union[Tensor, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (float, int, bool, Node)), f"{type(y)=}, {y=}"
      if isinstance(self.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      else: y_dtype = dtypes.from_py(y)
      if isinstance(y, Node): y = Tensor.from_node(y, device=self.device)
      else: y = Tensor(dtypes.as_const(y, y_dtype), self.device, y_dtype, requires_grad=False)

    if match_dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # broadcast
    out_shape = broadcast_shape(x.shape, y.shape)
    return x._broadcast_to(out_shape), y._broadcast_to(out_shape)

  def _to_const_val(self, x:Union[Tensor, ConstType]) -> Union[Tensor, ConstType]:
    # TODO: update with multi
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_unmasked_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor: return F.Add.apply(*self._broadcasted(x, reverse))
  def sub(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor: return F.Sub.apply(*self._broadcasted(x, reverse))
  def mul(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor: return F.Mul.apply(*self._broadcasted(x, reverse))
  def div(self, x:Union[Tensor, ConstType], reverse=False, upcast=True) -> Tensor:
    numerator, denominator = self._broadcasted(x, reverse)
    if upcast: numerator, denominator = numerator.cast(least_upper_float(numerator.dtype)), denominator.cast(least_upper_float(denominator.dtype))
    return F.Div.apply(numerator, denominator)
  def xor(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor: return F.Xor.apply(*self._broadcasted(x, reverse))

  def pow(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x == 0: return 1 + self * 0
      if x in [3,2,1]: return functools.reduce(lambda acc,_: acc * self, range(int(x)-1), self)
      if x == 0.5: return self.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()
    ar = self.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else self.mul(math.log(abs(x))).exp()
    # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (self * math.pi).cos()
    # we only need to correct the sign if the base is negative
    base_sign = ((self.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
    # we need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (1.5 * (1 - (self.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x))))))
    # inject nan if the base is negative and the power is not an integer
    to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else \
              int(bool(x - int(x))) if not reverse else ((self - self.trunc()) * 1e10).abs().clip(0, 1)) * base_sign
    inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)

  def maximum(self, x:Union[Tensor, ConstType]) -> Tensor:
    return (self<x).detach().where(x, (self==x).detach().where(((self * 0.5 + x * 0.5).cast(self.dtype)), self))
  def minimum(self, x:Union[Tensor, ConstType]) -> Tensor: return -((-self).maximum(-x))

  def where(self:Tensor, input_:Union[Tensor, ConstType], other:Union[Tensor, ConstType]):
    if isinstance(input_, Tensor): input_, other = input_._broadcasted(other)
    elif isinstance(other, Tensor): other, input_ = other._broadcasted(input_)
    x_,y = self._broadcasted(input_, match_dtype=False)
    x,z = x_._broadcasted(other, match_dtype=False)
    return F.Where.apply(x.cast(dtypes.bool), *y._broadcasted(z))

  def masked_fill(self:Tensor, mask:Tensor, value:Union[Tensor, ConstType]): return mask.where(value, self)

  # ***** op wrappers (wasted lines to make the typechecker happy) *****

  def __neg__(self) -> Tensor: return self.neg()

  def __add__(self, x) -> Tensor: return self.add(x)
  def __sub__(self, x) -> Tensor: return self.sub(x)
  def __mul__(self, x) -> Tensor: return self.mul(x)
  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __truediv__(self, x) -> Tensor: return self.div(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)
  def __xor__(self, x) -> Tensor: return self.xor(x)

  def __radd__(self, x) -> Tensor: return self.add(x, True)
  def __rsub__(self, x) -> Tensor: return self.sub(x, True)
  def __rmul__(self, x) -> Tensor: return self.mul(x, True)
  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)
  def __rxor__(self, x) -> Tensor: return self.xor(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))

  def __lt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, True))
  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __eq__(self, x) -> Tensor: return F.Eq.apply(*self._broadcasted(x, True))       # type: ignore[override]
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()                       # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  def dropout(self, p=0.5) -> Tensor:
    if not Tensor.training or p == 0: return self
    return self * (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p) * (1/(1.0 - p))

  def one_hot(self, num_classes:int) -> Tensor:
    return (self[..., None] == Tensor.arange(num_classes, requires_grad=False, device=self.device)).where(1, 0)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None,
                                   dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
    # NOTE: it works if key, value have symbolic shape
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
    if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
    qk = self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])
    return ((qk+attn_mask) if attn_mask is not None else qk).softmax(-1).dropout(dropout_p) @ value

  def binary_crossentropy(self, y:Tensor) -> Tensor:
    return (-y*self.log() - (1-y)*(1-self).log()).mean()

  def binary_crossentropy_logits(self, y:Tensor) -> Tensor:
    return (self.maximum(0) - y * self + (1 + self.abs().neg().exp()).log()).mean()

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index=-1, label_smoothing=0.0) -> Tensor:
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    # NOTE: self is a logits input
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index)
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask).sum()
    return -((1 - label_smoothing) * (log_probs * y).sum() + smoothing) / loss_mask.sum()

  # ***** cast ops *****

  def llvm_bf16_cast(self, dtype:DType):
    # hack for devices that don't support bfloat16
    assert self.dtype == dtypes.bfloat16
    return self.to("LLVM").bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)
  def cast(self, dtype:DType) -> Tensor: return self if self.dtype == dtype else F.Cast.apply(self, dtype=dtype)
  def bitcast(self, dtype:DType) -> Tensor:
    if self.requires_grad: raise RuntimeError("can't backprop through bitcast")
    return F.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
  def float(self) -> Tensor: return self.cast(dtypes.float32)
  def half(self) -> Tensor: return self.cast(dtypes.float16)

  # ***** convenience stuff *****

  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> sint: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
  def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)
  def size(self, dim=None) -> Union[sint, Tuple[sint, ...]]: return self.shape if dim is None else self.shape[dim]

# register functions to move between devices
for device in Device._devices: setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, device))

if IMAGE:
  # if IMAGE>0 we install these replacement functions in Tensor (hack!)
  from tinygrad.features.image import image_conv2d, image_dot
  setattr(Tensor, "conv2d", image_conv2d)
  setattr(Tensor, "dot", image_dot)

# TODO: eventually remove this
def custom_random(out:Buffer):
  Tensor._seed += 1
  rng = np.random.default_rng(Tensor._seed)
  if out.dtype == dtypes.half: rng_np_buffer = (rng.integers(low=0, high=2047, size=out.size) / 2048).astype(np.half, copy=False)
  else: rng_np_buffer = rng.random(size=out.size, dtype=np.float32).astype(dtype=out.dtype.np, copy=False)
  out.copyin(rng_np_buffer.data)

# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools, struct, sys, inspect, pathlib, hashlib, weakref
from contextlib import ContextDecorator
from typing import Any, Callable, ClassVar, Sequence, cast, get_args, Literal, ParamSpec, TypeVar, Generic, TYPE_CHECKING
if TYPE_CHECKING: import numpy
from tinygrad.dtype import DType, DTypeLike, dtypes, ConstType, least_upper_float, least_upper_dtype, to_dtype, truncate
from tinygrad.dtype import _from_np_dtype, _to_np_dtype, PyConst, Invalid
from tinygrad.helpers import argfix, flatten, prod, all_int, round_up, getenv, all_same, fully_flatten, ceildiv, fetch, flat_to_grouped
from tinygrad.helpers import resolve_pool_pads, IMAGE, FLOAT16, WINO, Metadata, TRACEMETA, is_numpy_ndarray, TracingKey, cpu_profile
from tinygrad.helpers import suppress_finalizing, disable_gc
from tinygrad.gradient import compute_gradient
from tinygrad.mixin import OpMixin
from tinygrad.uop.ops import UOp, Ops, sint, all_metadata, _index_to_concrete_int, Variable, _broadcast_shape
from tinygrad.schedule import create_linear_with_vars
from tinygrad.device import Buffer, canonicalize_device
from tinygrad.engine.realize import run_linear
from tinygrad.callify import transform_to_call

# *** all in scope Tensors are here. this gets relevant UOps ***

all_tensors: dict[weakref.ref[Tensor], None] = {}
def _apply_map_to_tensors(applied_map:dict[UOp, UOp], name:str, walk:bool=False) -> None:
  with cpu_profile(TracingKey(name), "TINY"):
    # get tensors in scope
    in_scope: dict[UOp, bool] = {}
    def visitor(node: UOp) -> bool: return True if node in applied_map else any(in_scope.get(s, False) for s in node.src)
    scope_tensors: list[Tensor] = [t for tref in list(all_tensors) if (t:=tref()) is not None and t.uop.topovisit(visitor, in_scope)]

    # get all Tensors and apply the map
    sink = UOp.sink(*[t.uop for t in scope_tensors])
    new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)

    # set the relevant uop to the realized UOps
    for t,s,ns in zip(scope_tensors, sink.src, new_sink.src):
      if s is ns: continue
      t.uop = ns

# **** Tensor helper functions ****

def _fromnp(x: 'numpy.ndarray') -> UOp:
  ret = UOp.new_buffer("NPY", x.size, _from_np_dtype(x.dtype))
  # fake realize
  ret.buffer.allocate(x)
  return ret.reshape(x.shape)

def get_shape(x) -> tuple[int, ...]:
  # NOTE: str is special because iterating it still yields strs
  if not hasattr(x, "__len__") or isinstance(x, str) or getattr(x, "shape", None) == (): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

def _frompy(x:list|tuple|bytes, dtype:DType, device:str|tuple[str,...]) -> UOp:
  if isinstance(x, bytes): ret, data = UOp.new_buffer("PYTHON", len(x)//dtype.itemsize, dtype), x
  else:
    ret = UOp.empty(shape:=get_shape(x), dtype, "PYTHON")
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    truncate_function = truncate[dtype]
    data = struct.pack(f"{prod(shape)}{dtype.fmt}", *[truncate_function(dtype.const(xi)) for xi in fully_flatten(x)])
  # fake realize. if target device is PYTHON it needs bytearray to be writable
  ret.buffer.allocate(memoryview(data if device != "PYTHON" else bytearray(data)))
  return ret

def _get_winograd_matcols(mat, dims:int, shp:tuple[sint, ...], device:str|tuple[str, ...], dtype:DType) -> list[list[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

class Tensor(OpMixin):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "uop", "requires_grad", "grad"
  training: ClassVar[bool] = False

  def __init__(self, data:ConstType|bytes|list|tuple|UOp|'numpy.ndarray'|pathlib.Path|None,
               device:str|tuple|list|None=None, dtype:DTypeLike|None=None, requires_grad:bool|None=None, _force_unique:bool=False):
    if device is None:
      if isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
      elif isinstance(data, UOp): device = data._device
    _dtype:DType|None = to_dtype(dtype) if dtype is not None else None
    _device:str|tuple[str, ...] = canonicalize_device(device)
    del device, dtype

    # tensors can have gradients if you have called .backward
    self.grad:Tensor|None = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad:bool|None = requires_grad

    # create a UOp from the different types of inputs
    if isinstance(data, UOp):
      assert _dtype is None or _dtype==data.dtype or data.dtype==dtypes.weakint, f"dtype mismatch: {_dtype} vs {data.dtype}"
      # if data is dtype.weakint that means that this is a symbolic int and we need to lower it to something we can make a Tensor out of
      if data.dtype == dtypes.weakint: data = Tensor.from_uop(data, device=_device).uop
    elif data is None:
      data = UOp.const(_dtype or dtypes.default_float, 0, _device)
    elif isinstance(data, get_args(ConstType)):
      if _force_unique or requires_grad: data = UOp.unique_const(data, _dtype, _device)
      else: data = UOp.const(_dtype or dtypes.from_py(data), data, _device)
    elif isinstance(data, bytes): data = _frompy(data, _dtype or dtypes.uint8, _device)
    elif isinstance(data, (list, tuple)):
      if _dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): _dtype = dtypes.bool
        else: _dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float  # NOTE: this works because all_int([True, False]) is True
      if _dtype in [dtypes.bfloat16, *dtypes.fp8s]: data = _frompy(data, dtypes.float32, _device).cast(_dtype)
      else: data = _frompy(data, _dtype, _device)
    elif is_numpy_ndarray(data):
      import numpy as np
      assert isinstance(data, np.ndarray), f"expected np.ndarray, got {data}"
      if data.shape == ():
        data = UOp.const(_dtype or _from_np_dtype(data.dtype), data.item(), _device)
      else:
        data = _fromnp(data.astype(npdtype) if _dtype is not None and (npdtype:=_to_np_dtype(_dtype)) is not None else data)
    elif isinstance(data, pathlib.Path):
      _dtype = _dtype or dtypes.uint8
      data = UOp.new_buffer(f"DISK:{data.resolve()}", data.stat().st_size // _dtype.itemsize, _dtype)

    # by this point, it has to be a UOp
    if not isinstance(data, UOp): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    self.uop:UOp = data if data.device == _device else data.copy_to_device(_device)

    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(self)] = None

  @suppress_finalizing
  def __del__(self): all_tensors.pop(weakref.ref(self), None)

  def _apply_uop(self, fxn:Callable[..., UOp], *x:Tensor, extra_args=(), **kwargs) -> Tensor:
    srcs = (self,)+x
    new_uop: UOp = fxn(*[t.uop for t in srcs], *extra_args, **kwargs)
    if TRACEMETA >= 1 and (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
    needs_input_grad = [t.requires_grad for t in srcs]
    # directly create the Tensor
    ret = Tensor.__new__(Tensor)
    ret.uop, ret.grad = new_uop, None
    ret.requires_grad = True if any(needs_input_grad) else None if None in needs_input_grad else False
    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(ret)] = None
    return ret

  # alu and const_like are used by the mixins
  def alu(self, op: Ops, *src: Tensor) -> Tensor: return self._apply_uop(lambda *u: u[0].alu(op, *u[1:]), *src)
  def const_like(self, b:ConstType) -> Tensor: return Tensor(self.uop.const_like(b), requires_grad=False)
  @staticmethod
  def unique_const(fill_value:ConstType|UOp, **kwargs) -> Tensor: return Tensor(fill_value, _force_unique=True, **kwargs)

  def requires_grad_(self, requires_grad=True) -> Tensor:
    # make the UOp unique if it's a CONST to prevent gradient accumulation bugs with cached const UOps
    if requires_grad and self.uop.op is Ops.CONST: self.replace(Tensor(self.uop.arg, device=self.device, dtype=self.dtype, requires_grad=True))
    self.requires_grad = requires_grad
    return self

  class train(ContextDecorator):
    def __init__(self, mode:bool = True): self.mode = mode
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.mode
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  def __repr__(self):
    ld = self.uop
    ld_repr = f"<UOp {ld.device} {ld.shape} {str(ld.dtype)[7:]}>"
    return f"<Tensor {ld_repr} on {self.device} with grad {(self.grad.uop if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self):
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  @property
  def device(self) -> str|tuple[str, ...]: return self.uop.device

  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape

  @property
  def dtype(self) -> DType: return self.uop.dtype

  # ***** data handlers ****

  def as_param(self, slot:int):
    if self.uop.axis is not None:
      param = UOp.param(slot, self.dtype, self.uop.shard_shape, self.device).multi(self.uop.axis)
    else:
      param = UOp.param(slot, self.dtype, self.shape, self.device)
    return Tensor(param)
  def call(self, *lst:Tensor, fxn:Tensor|UOp, grad_fxn:Callable|None=None) -> Tensor:
    fret = (fxn.uop if isinstance(fxn, Tensor) else fxn).call(*[t.uop for t in (self,)+lst], grad_fxn=grad_fxn)
    return Tensor(fret.gettuple(0))

  def custom_kernel(self, *lst:Tensor, fxn:Callable, grad_fxn:Callable|None=None) -> list[Tensor]:
    """
    Call into a custom kernel written in UOps. Returns the Tensors after the Kernel has been applied.

    This API is alpha and may change.
    """
    return [Tensor(u) for u in UOp.custom_kernel(*[t.uop for t in (self,)+lst], fxn=fxn, grad_fxn=grad_fxn)]

  def callify(self, *lst:Tensor) -> Tensor:
    big_sink = UOp.sink(*[x.uop for x in (self,)+lst])
    big_sink, buffer_map = transform_to_call(big_sink)
    _apply_map_to_tensors({x:y.after(big_sink) for x,y in buffer_map.items()}, name="callify")
    return self

  def linear_with_vars(self, *lst:Tensor) -> tuple[UOp, dict[str, int]]:
    """Creates the LINEAR UOp needed to realize these Tensor(s), with Variables."""
    big_sink, becomes_map = transform_to_call(UOp.sink(*[x.uop for x in (self,)+lst]))
    _apply_map_to_tensors(becomes_map, name="buffers")
    return create_linear_with_vars(big_sink)

  def schedule_linear(self, *lst:Tensor) -> UOp:
    """Creates the schedule needed to realize these Tensor(s)."""
    linear, var_vals = self.linear_with_vars(*lst)
    assert len(var_vals) == 0
    return linear

  @disable_gc()
  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Triggers the computation needed to create these Tensor(s)."""
    if len(to_realize:=[x for x in (self,)+lst if not x.uop.has_buffer_identity()]):
      run_linear(*Tensor.linear_with_vars(*to_realize), do_update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor) -> Tensor:
    """
    Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.uop = x.uop
    return self

  def assign(self, x:Tensor|PyConst|list|tuple) -> Tensor:
    is_disk = isinstance(self.device, str) and self.device.startswith("DISK")
    if not isinstance(x, Tensor): x = Tensor(x, device="CPU" if is_disk else self.device, dtype=self.dtype)
    if self.uop is x.uop: return self  # a self assign is a NOOP
    # broadcast x (shape only, dtype must match)
    if self.shape != x.shape: x = x._broadcast_to(self.shape)
    if self.shape != x.shape: raise RuntimeError(f"assign shape mismatch {self.shape} != {x.shape}")
    if not is_disk and self.device != x.device: raise RuntimeError(f"assign device mismatch {self.device} != {x.device}")
    if not is_disk and self.dtype != x.dtype: raise RuntimeError(f"assign dtype mismatch {self.dtype} != {x.dtype}")
    if isinstance(self.device, tuple) and self.uop.axis != x.uop.axis: raise RuntimeError(f"multi axis mismatch {self.uop.axis} != {x.uop.axis}")

    # TODO: this is a hack for writing to DISK. remove with working assign
    if is_disk:
      self._buffer().copyin(x._data())
      return self
    # STORE+AFTER: STORE is the write effect (void), AFTER wraps the view for correct shape/ranging
    assign = self.uop.after(self.uop.store(x.uop))
    if (base := self.uop.base).op in {Ops.BUFFER, Ops.AFTER} and self.uop is not base and not self.uop.has_buffer_identity():
      # view assign: replace at the buffer-identity level (e.g. RESHAPE(BUFFER)) so @function's substitution catches it
      ib = self.uop
      while not ib.has_buffer_identity() and ib is not base: ib = ib.src[0]
      assigned_ib = ib.after(assign)
      _apply_map_to_tensors({ib: assigned_ib}, name="Embed View Assign", walk=True)
    else:
      # simple assign
      self.uop = assign
    return self

  def detach(self) -> Tensor:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return Tensor(self.uop.detach(), requires_grad=False)

  def _buffer(self) -> Buffer:
    from tinygrad.engine.realize import capturing
    if capturing and not getenv("UNSAFE_ALLOW_JIT_BUFFER"):
      from tinygrad.engine.jit import JitError
      raise JitError("cannot access tensor data during JIT capture, the value will be baked in")
    x = self.cast(self.dtype.base).contiguous()
    if isinstance(self.device, tuple): x = x.to("CPU")
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()
  def _data(self) -> memoryview: return self._buffer().as_memoryview()

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.base.fmt)
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    assert self.dtype.base.fmt is not None, f"no fmt dtype for {self.dtype.base}"
    assert self.dtype.base.fmt != "e" or sys.version_info >= (3, 12)
    return self._buffer().as_memoryview().cast(self.dtype.base.fmt, self.shape)

  def item(self) -> PyConst:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

  # NOTE: list[Any] because return type is recursive (list[list[...]] for higher dimensions)
  def tolist(self) -> PyConst|list[Any]:
    """
    Returns the value of this tensor as a nested list.
    Returns single value for const tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(5)
    print(t.tolist())
    ```
    """
    # TODO: remove half once minimum python supports it
    if self.dtype in (dtypes.half, dtypes.bfloat16, *dtypes.fp8s): return self.cast(dtypes.float32).tolist()
    return self.data().tolist()

  def numpy(self) -> 'numpy.ndarray':
    """
    Returns the value of this tensor as a `numpy.ndarray`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(repr(t.numpy()))
    ```
    """
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    import numpy as np
    if self.dtype.base in { dtypes.bfloat16, *dtypes.fp8s }: return self.float().numpy()
    if 0 in self.shape: return np.empty(self.shape, dtype=_to_np_dtype(self.dtype.base))
    return self._buffer().numpy().reshape(self.shape)

  def clone(self) -> Tensor:
    """
    Creates a clone of this tensor allocating a separate buffer for the data.
    """
    ret = self.empty_like()
    if self.grad is not None: ret.grad = self.grad.clone()
    return ret.assign(self)

  def to(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    if (device:=canonicalize_device(device)) == self.device: return self
    ret = Tensor(self.uop.copy_to_device(device), requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    return ret

  def to_(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device in place.
    """
    real = self.to(device)
    if self.grad is not None and real.grad is not None: self.grad.replace(real.grad)
    return self.replace(real)

  def shard(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices. Optionally specify which axis to shard on.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 4)
    print(t.shard((t.device, t.device), axis=1).uop)
    ```
    """
    if not isinstance(self.device, str): raise RuntimeError("can't shard a multi-device tensor")
    if len(devices) == 1: return self.to(devices[0])
    devices = cast(tuple[str, ...], canonicalize_device(devices))
    uop = self.uop.shard(devices, self._resolve_dim(axis)) if axis is not None else self.uop.copy_to_device(devices)
    return Tensor(uop, requires_grad=self.requires_grad)

  def shard_(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices in place.
    """
    return self.replace(self.shard(devices, axis))

  def shard_like(self, y:Tensor) -> Tensor:
    """
    Shards the tensor the same way as `y` (same devices and axis).
    """
    if isinstance(y.device, str): return self.to(y.device)
    return self if isinstance(self.device, tuple) and (y.device, y.uop.axis) == (self.device, self.uop.axis) else self.shard(y.device, y.uop.axis)

  CHUNK_SIZE = 2**20
  def fs_load(self, size:int) -> Tensor:
    """
    Load a tensor from storage.

    self should be a tensor of the hash to load
    """
    # TODO: this should work locally as well
    assert self.dtype == dtypes.uint8, "hash is expected to be uint8"
    h = self.contiguous().flatten()
    assert h.shape[0] == 16, "expected hash"

    base_chunks = math.ceil(size / Tensor.CHUNK_SIZE)
    tree_depth = math.ceil(math.log(base_chunks, Tensor.CHUNK_SIZE // 16))
    data, level_chunks = h, 0
    for i in reversed(range(tree_depth + 1)):
      data = data.to("tinyfs:load")

      # if not last level, its still hashes
      if i > 0 or tree_depth == 0:
        level_chunks = max(1, math.ceil(base_chunks / (Tensor.CHUNK_SIZE // 16)**(i-1)))
        pad_amt = 16 * level_chunks
      else: pad_amt = Tensor.CHUNK_SIZE * level_chunks
      if (tsize := data.shape[0]) < pad_amt: data = data.pad((0, pad_amt - tsize))
      data = data[:pad_amt].contiguous()
      if i != 0: data = data.to(self.device)

    return data[:size]

  def fs_store(self) -> Tensor:
    """
    Store a tensor to storage.
    """
    # TODO: this should work locally as well
    data = self.contiguous().flatten().bitcast(dtypes.uint8)

    # pad to a multiple of 1mb
    if (tsize := data.shape[0]) % Tensor.CHUNK_SIZE != 0: data = data.pad((0, Tensor.CHUNK_SIZE - tsize % Tensor.CHUNK_SIZE))
    size = data.shape[0]

    base_chunks = math.ceil(size / Tensor.CHUNK_SIZE)
    tree_depth = math.ceil(math.log(base_chunks, Tensor.CHUNK_SIZE // 16))

    to_device = "CPU" if isinstance(self.device, str) and self.device.startswith("DISK") else self.device

    level_chunks = base_chunks
    for _ in range(tree_depth + 1):
      data = data.to("tinyfs:store")[:level_chunks * 16].contiguous().to(to_device)
      if (tsize := data.shape[0]) % Tensor.CHUNK_SIZE != 0: data = data.pad((0, Tensor.CHUNK_SIZE - tsize % Tensor.CHUNK_SIZE))
      level_chunks = math.ceil(data.shape[0] / Tensor.CHUNK_SIZE)

    return data[:16].contiguous()

  @staticmethod
  def from_uop(y:UOp, **kwargs) -> Tensor:
    # TODO: remove this and stay in weakint
    if y.dtype == dtypes.weakint: y = _index_to_concrete_int(y)
    if y.op is Ops.BIND:
      var, val = y.unbind()
      _device = canonicalize_device(kwargs.get("device"))
      const = UOp.const(var.dtype, val, _device, ())
      return Tensor(y.replace(src=(var.replace(src=const.src), const)), **kwargs, requires_grad=False)
    if y.op is Ops.CONST: return Tensor(y.arg, **kwargs, requires_grad=False)
    if y.op is Ops.MUL: return Tensor.from_uop(y.src[0]) * Tensor.from_uop(y.src[1])
    if y.op is Ops.ADD: return Tensor.from_uop(y.src[0]) + Tensor.from_uop(y.src[1])
    raise RuntimeError(f"unhandled UOp {y}")

  # ***** creation entrypoint *****

  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor(UOp.empty(argfix(*shape), dtype, device), **kwargs)

  def empty_like(self, dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return Tensor(self.uop.empty_like(dtype, device), **kwargs)

  @staticmethod
  def from_blob(ptr:int, shape:tuple[int, ...], **kwargs) -> Tensor:
    """
    Exposes the pointer as a Tensor without taking ownership of the original data.
    The pointer must remain valid for the entire lifetime of the created Tensor.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.
    """
    r = Tensor.empty(*shape, **kwargs)
    assert isinstance(r.device, str)
    cast(Buffer, r.uop.buffer).allocate(external_ptr=ptr)
    return r

  @staticmethod
  def from_url(url:str, gunzip:bool=False, **kwargs) -> Tensor:
    """
    Creates a Tensor from a URL.

    This is the preferred way to access Internet resources.
    It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
    This also will soon become lazy (when possible) and not print progress without DEBUG.

    The `gunzip` flag will gzip extract the resource and return an extracted Tensor.
    """
    return Tensor(fetch(url, gunzip=gunzip), **kwargs)

  _seed: int = int(time.time())
  _device_seeds: dict[str, Tensor] = {}
  _device_rng_counters: dict[str, Tensor] = {}
  @staticmethod
  def manual_seed(seed=0) -> None:
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)  # reset to the same seed
    print(Tensor.rand(5).numpy())
    print(Tensor.rand(5).numpy())
    ```
    """
    Tensor._seed, Tensor._device_seeds, Tensor._device_rng_counters = seed, {}, {}

  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    dt = to_dtype(dtype or dtypes.default_float)
    if not dtypes.is_float(dt): raise ValueError(f"rand only supports float dtypes, got {dt}")
    if not all_int(shape:=argfix(*shape)) or not all(s >= 0 for s in shape): raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str): raise ValueError(f"rand only supports single device, got {device=}")
    device = cast(str, canonicalize_device(device))

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0: return Tensor.zeros(shape, device=device, dtype=dt, **kwargs)
    num = ceildiv(numel * dt.itemsize, 4)

    # generate per device seeds and rng counter if we haven't seen this device yet
    if device not in Tensor._device_seeds:
      Tensor._device_seeds[device] = Tensor(
        [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed],
        device=device, dtype=dtypes.uint32, requires_grad=False)
      Tensor._device_rng_counters[device] = Tensor([0, 0], device=device, dtype=dtypes.uint32, requires_grad=False).contiguous()

    # increment rng counter for devices
    new_low = Tensor._device_rng_counters[device][0:1] + (num & 0xffffffff)
    new_high = Tensor._device_rng_counters[device][1:2] + (num >> 32) + (new_low < Tensor._device_rng_counters[device][0]).cast(dtypes.uint32)
    Tensor._device_rng_counters[device].assign(new_low.cat(new_high))

    low = Tensor._device_rng_counters[device][0:1] - (num & 0xffffffff)
    high = Tensor._device_rng_counters[device][1:2] - (num >> 32) - (Tensor._device_rng_counters[device][0] < (num & 0xffffffff)).cast(dtypes.uint32)
    bits = Tensor.random_bits(Tensor._device_seeds[device], low.cat(high), num)

    # bitcast to uint with same number of bits
    _, nmant = dtypes.finfo(dt)
    uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dt.itemsize]
    bits = bits.bitcast(uint_dtype)
    # only randomize the mantissa bits and set the exponent to 1
    one = Tensor.ones_like(bits, device=bits.device, dtype=dt).bitcast(uint_dtype)
    bits = bits.rshift(dt.bitsize - nmant).bitwise_or(one)
    # bitcast back to the original dtype and reshape
    out = bits.bitcast(dt)[:numel].sub(1).reshape(shape).requires_grad_(kwargs.get("requires_grad"))
    return out.contiguous() if contiguous else out

  # ***** creation helper functions *****

  @classmethod
  def eye(cls, n:int, m:int|None=None, dtype=None, device=None, requires_grad:bool|None=None) -> Tensor:
    """
    Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(2, 4).numpy())
    ```
    """
    return super().eye(n, m, dtype, device).requires_grad_(requires_grad)

  def _multi_like(self, fxn, *args, **kwargs) -> Tensor:
    dtype = kwargs.pop("dtype", self.dtype)
    if kwargs.get("device") is not None: raise RuntimeError("cannot specify `device` on `*_like` of a multi device tensor")
    if self.uop.axis is None: return fxn(self.shape, *args, dtype=dtype, **kwargs).shard(self.device)
    stacked = UOp.mstack(*[fxn(self.uop.shard_shape, *args, device=d, dtype=dtype, **kwargs).uop for d in self.device])
    return Tensor(stacked.multi(self.uop.axis))

  def full_like(self, fill_value:ConstType, dtype=None, device=None, requires_grad=None) -> Tensor:
    """
    Creates a tensor with the same shape as `self`, filled with the given value.
    If `dtype` is not specified, the dtype of `self` is used.

    You can pass in the `device` keyword argument to control device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    if device is not None:
      if isinstance(self.device, tuple): raise RuntimeError("cannot specify `device` on `full_like` of a multi device tensor")
      return Tensor.full(self.shape, fill_value, dtype=dtype or self.dtype, device=device).requires_grad_(requires_grad)
    if requires_grad:
      return Tensor.full(self.shape, fill_value, dtype=dtype or self.dtype, device=self.device).requires_grad_(requires_grad)
    return super().full_like(fill_value, dtype)

  def rand_like(self, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.rand_like(t).numpy())
    ```
    """
    if isinstance(self.device, tuple): return self._multi_like(Tensor.rand, **kwargs)
    return Tensor.rand(*self.shape, device=kwargs.pop("device", self.device), dtype=kwargs.pop("dtype", self.dtype), **kwargs)

  # ***** random functions *****

  def randn_like(self, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the same shape and sharding as `self`, filled with random values from a normal distribution with mean 0 and variance 1.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.randn_like(t).numpy())
    ```
    """
    src = self.stack(self).rand_like(**{**kwargs, "dtype": dtypes.float32})
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    return (src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or self.dtype)).requires_grad_(requires_grad)

  @staticmethod
  def randn(*shape, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randn(2, 3).numpy())
    ```
    """
    return Tensor.empty(*shape, **kwargs).randn_like(dtype=dtype, requires_grad=requires_grad)

  @staticmethod
  def randint(*shape, low=0, high=10, dtype=dtypes.int32, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    ```
    """
    if not all_int([low, high]): raise TypeError(f"{low=} and {high=} must be integers")
    if not dtypes.is_int(dtype := to_dtype(dtype)): raise TypeError(f"{dtype=} must be int")
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    return (std * Tensor.randn(*shape, **kwargs) + mean).requires_grad_(requires_grad)

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, dtype:DTypeLike|None=None, requires_grad:bool|None=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    return (((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype or dtypes.default_float) + low).requires_grad_(requires_grad)

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution
    over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.scaled_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    bound = (6 / (argfix(*shape)[0]+prod(argfix(*shape)[1:]))) ** 0.5
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_uniform(2, 3).numpy())
    ```
    """
    bound = (6 / (1 + a ** 2) / prod(argfix(*shape)[1:])) ** 0.5
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_normal(2, 3).numpy())
    ```
    """
    std = (2 / (1 + a ** 2) / prod(argfix(*shape)[1:])) ** 0.5
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  @staticmethod
  def randperm(n:int, device=None, dtype=dtypes.int32, **kwargs) -> Tensor:
    """
    Returns a tensor with a random permutation of integers from `0` to `n-1`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randperm(6).numpy())
    ```
    """
    return Tensor.rand(n, device=device, **kwargs).argsort().cast(dtype)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    """
    Returns a tensor with `num_samples` indices sampled from a multinomial distribution weighted by `self`.

    NOTE: `replacement=False` for `num_samples > 1` is not supported yet.
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor([1, 2, 3, 4])
    print(t.multinomial(20, replacement=True).numpy())
    ```
    """
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1).to(self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  # ***** toposort and backward pass *****

  def gradient(self, *targets:Tensor, gradient:Tensor|None=None) -> list[Tensor]:
    """
    Computes the gradient of the targets with respect to self.

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)

    print(dx.tolist())  # dz/dx
    print(dy.tolist())  # dz/dy
    ```
    """
    assert gradient is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in targets)): raise RuntimeError("only float Tensors have gradient")
    if gradient is None: gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)
    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    ret:list[Tensor] = []
    for x in target_uops:
      if (y:=grads.get(x)) is None: y = x.const_like(0)
      ret.append(Tensor(y))
    return ret

  def backward(self, gradient:Tensor|None=None) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    tensors_need_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.requires_grad]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      if t.grad is None: t.grad = g
      else: t.grad.assign(t.grad + g.to(t.grad.device))
    return self

  # ***** movement ops *****

  def _mop(self, op:Ops, arg) -> Tensor: return self._apply_uop(UOp._mop, extra_args=(op,), arg=arg)
  def _rop(self, op:Ops, axis:tuple[int, ...]) -> Tensor: return self._apply_uop(UOp._rop, op=op, axis=axis)

  def _getitem(self, indices, v: Tensor|None = None) -> Tensor:
    # view-only indexing (no Tensor/list indices, no setitem) is handled by MovementMixin.__getitem__
    if v is None and not any(isinstance(i, (Tensor, list, tuple)) for i in (indices if isinstance(indices, tuple) else (indices,))):
      return super().__getitem__(indices)
    # wrap single index into a list
    if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)): indices = [indices]
    indices_parsed, dim = [], 0
    for index in self._normalize_indices(list(indices)):
      size = 1 if index is None else self.shape[dim]
      parsed = {"size":size, "boundary":(0, size), "stride":1, "collapse_dim":False}
      match index:
        case Tensor():
          if not dtypes.is_int(index.dtype): raise IndexError(f"index dtype {index.dtype} is not supported")
          if index.device != self.device: raise RuntimeError(f"expected index and self on the same device, {index.device=}, {self.device=}")
          assert isinstance(size, int), "size must be an int"
          index = (index < 0).where(index+size, index)  # treat negative index values
        case list() | tuple():
          if not dtypes.is_int((ti:=Tensor(index)).dtype): raise IndexError(f"{index=} contains non-int element")
          index = Tensor([i+size if i<0 else i for i in fully_flatten(index)], self.device, requires_grad=False).reshape(ti.shape)
        case _: parsed = self._parse_view_index(index, size)
      indices_parsed.append({**parsed, "index":index})
      if index is not None: dim += 1

    # apply view ops then dim injection (None) and collapse (int)
    x = self._apply_view_ops(mops) if (mops := [p for p in indices_parsed if p["index"] is not None]) else self
    x_dims = [p for p in indices_parsed if not p["collapse_dim"]]
    x = x.reshape(tuple(p["size"] for p in x_dims))

    # tensor indexing
    if tops := [(d, p) for d, p in enumerate(x_dims) if isinstance(p['index'], Tensor)]:
      dims, tensors, masks = [d for d, _ in tops], cast(list[Tensor], [p['index'] for _, p in tops]), []
      big_shape = _broadcast_shape(*(t.shape for t in tensors))

      # consecutive tensor indices with int shapes: use linear indexing instead of one-hot masks
      consecutive = dims == list(range(dims[0], dims[0] + len(dims)))
      if v is None and len(dims) > 1 and consecutive and all_int(ishp := tuple(x.shape[d] for d in dims)):
        strides = tuple(prod(ishp[i+1:]) for i in range(len(dims)))
        try: linear_idx = Tensor.usum(*[t._broadcast_to(big_shape) * s for t, s in zip(tensors, strides)])
        except ValueError as err: raise IndexError(f"cannot broadcast indices: {err}") from err
        valid = Tensor.uprod(*[(t >= 0) & (t < s) for t, s in zip(tensors, ishp)])
        pre, post = x.shape[:dims[0]], x.shape[dims[-1]+1:]
        x = x.reshape(pre + (prod(ishp),) + post)[tuple([slice(None)] * len(pre)) + (valid.where(linear_idx, 0),)]
        return valid.reshape((1,) * len(pre) + big_shape + (1,) * len(post)).where(x, 0)

      pre_reduce_shape = x.shape[:dims[0]] + big_shape + x.shape[dims[0]:]

      # create index masks
      for dim, tensor in zip(dims, tensors):
        try: i = tensor.reshape(tensor.shape + (1,)*(x.ndim - dims[0])).expand(pre_reduce_shape)
        except ValueError as err: raise IndexError(f"cannot broadcast indices: {err}") from err
        masks.append(i._one_hot_along_dim(num_classes=x.shape[dim], dim=(dim - x.ndim)))

      # reduce masks to 1 mask
      mask: Tensor = Tensor.uprod(*masks)

      # inject 1's for the extra dims added in create masks
      reshape_arg = x.shape[:dims[0]] + (1,) * len(big_shape) + x.shape[dims[0]:]
      # sum reduce the extra dims introduced in create masks
      x_pre = x  # save collapsed shape for advanced setitem
      x = (mask.where(x.reshape(reshape_arg), 0)).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), dtype=x.dtype)

      # special permute case
      if (permuted := dims[0] != 0 and len(dims) != 1 and tuple(dims) != tuple(range(dims[0], dims[-1]+1))):
        mask, x = (y.permute(*range(dims[0], dims[0]+len(big_shape)), *range(0, dims[0]), *range(dims[0]+len(big_shape), y.ndim)) for y in (mask, x))

      if v is None: return x  # advanced getitem
      # advanced setitem: resolve tensor dims in collapsed space, then fall through to basic setitem path
      vb = v.cast(self.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
      for dim in sum_axis: vb = vb.unsqueeze(dim)  # add back reduced dims from sum
      start = dims[0] if not permuted else 0
      vb = x_pre._masked_merge(vb, mask, tuple(range(start, start + len(big_shape))))
    elif v is None: return x  # basic getitem
    # basic setitem: broadcast v, reshape to self.ndim (unsqueeze int dims, squeeze None dims)
    else: vb = v.cast(self.dtype)._broadcast_to(x.shape)
    vb = vb.reshape(tuple(1 if isinstance(p['index'], sint) else p['size'] for p in indices_parsed if p['index'] is not None))
    per_dim = []
    for d, m in enumerate(mops):
      (s, e), st = m['boundary'], abs(m['stride'])
      if st != 1 and vb.shape[d] > 1:  # un-stride: interleave with zeros
        vb = vb.unsqueeze(d+1)
        vb = vb.pad_to(tuple(st if j == d+1 else None for j in range(vb.ndim)))
        vb = vb.reshape(vb.shape[:d] + (vb.shape[d]*vb.shape[d+1],) + vb.shape[d+2:])
        vb = vb.shrink_to(tuple(e-s if j == d else None for j in range(self.ndim)))
      idx = Tensor.arange(self.shape[d], device=self.device).reshape([1]*d + [self.shape[d]] + [1]*(self.ndim - d - 1))
      per_dim.append((idx >= s) & (idx < e) & (((e-1-idx) if m['stride'] < 0 else (idx-s)) % st == 0))
    vb = vb.flip(tuple(d for d, m in enumerate(mops) if m['stride'] < 0))
    vb = vb.pad(tuple((m['boundary'][0], self.shape[d] - m['boundary'][1]) for d, m in enumerate(mops)))
    return (Tensor.uprod(*per_dim) if per_dim else Tensor(True, dtype=dtypes.bool, device=self.device)).where(vb, self)

  def __getitem__(self, indices) -> Tensor:
    """
    Retrieves a sub-tensor using indexing.

    Supported Index Types: `int | slice | Tensor | None | list | tuple | Ellipsis`

    Examples:
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(12).reshape(3, 4)
    print(t.numpy())
    ```

    - Int Indexing: Select an element or sub-tensor using integers for each dimension.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[1, 2].numpy())
      ```

    - Slice Indexing: Select a range of elements using slice notation (`start:end:stride`).
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[0:2, ::2].numpy())
      ```

    - Tensor Indexing: Use another tensor as indices for advanced indexing. Using `tuple` or `list` here also works.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[Tensor([2, 0, 1]), Tensor([1, 2, 3])].numpy())
      ```

    - `None` Indexing: Add a new dimension to the tensor.
      ```python exec="true" source="above" session="tensor" result="python"
      print(t[:, None].shape)
      ```

    NOTE: Out-of-bounds indexing results in a value of `0`.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t[Tensor([4, 3, 2])].numpy())
    ```
    """
    return self._getitem(indices)

  def __setitem__(self, indices, v:Tensor|PyConst|list|tuple) -> None:
    if isinstance(v, Tensor) and v.dtype != self.dtype: raise RuntimeError(f"setitem dtype mismatch: {self.dtype=} != {v.dtype=}")
    if self.requires_grad or (isinstance(v, Tensor) and v.requires_grad):
      # for +=/-=, v's graph references self.uop through the view — exclude those from the stale-use check
      v_uop, v_bw = (v.uop, v.uop.backward_slice) if isinstance(v, Tensor) else (None, {})
      if any(self.uop in t.uop.backward_slice for tref in all_tensors
             if (t:=tref()) is not None and t is not self and t.uop is not v_uop and t.uop not in v_bw):
        raise RuntimeError("can't setitem on a tensor that already has other uses and requires grad")
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      # __iadd__/__isub__ creates AFTER(view, STORE(view, computed)); unwrap to get the computed value
      if v.uop.op is Ops.AFTER and any(s.op is Ops.STORE for s in v.uop.src[1:]): v = v._apply_uop(lambda x: x.src[1].src[1])
      self.replace(self._getitem(indices, v))
      return
    idx = [indices] if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)) else list(indices)
    is_disk = isinstance(self.device, str) and self.device.startswith("DISK")
    if any(isinstance(i, (Tensor, list, tuple)) for i in idx): # advanced setitem
      if is_disk: raise RuntimeError("advanced setitem is not supported for DISK tensors")
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      self.assign(self._getitem(indices, v))
    elif is_disk or self.uop.is_realized or self.uop.base.op is Ops.BUFFER or self.uop._base_buffer_is_realized(): # basic setitem
      view = self[indices]
      if isinstance(v, Tensor) and v.uop.op is Ops.AFTER and v.uop in view.uop.base.src: return
      view.assign(v)
    else: # basic setitem, self is not realized
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      # __iadd__/__isub__ creates AFTER(view, STORE(view, computed)); unwrap to get the computed value
      if v.uop.op is Ops.AFTER and any(s.op is Ops.STORE for s in v.uop.src[1:]): v = v._apply_uop(lambda x: x.src[1].src[1])
      self.replace(self._getitem(indices, v))

  def __delitem__(self, indices) -> None:
    raise TypeError("Tensor does not support deleting items")

  def masked_select(self, mask):
    """
    Selects elements from `self` based on the boolean `mask`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
    print(t.numpy())
    print(mask.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.masked_select(mask).numpy())
    ```
    """
    if not dtypes.is_bool(mask.dtype): raise RuntimeError(f"masked_select expects bool mask tensor, got {mask.dtype}")
    x, mask = self.flatten(), mask._broadcast_to(self.shape).flatten()
    mask_cumsum = mask.cumsum()
    counts = Tensor.zeros(mask_cumsum[-1].item(), dtype=dtypes.int32)
    idxs = counts.scatter(0, mask_cumsum, 1, reduce='add').cumsum()
    return x[idxs]

  def nonzero(self) -> Tensor:
    """
    Returns the indices of the elements that are non-zero.

    Returns a 2D tensor where each row is the index of a non-zero element.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 0, 2, 0, 3])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.nonzero().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0], [0, 2]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.nonzero().numpy())
    ```
    """
    mask = (self != 0).flatten()
    indices = Tensor.stack(*[Tensor.arange(s, device=self.device).reshape(*[1]*i, s, *[1]*(self.ndim-i-1)).expand(self.shape).flatten()
                             for i, s in enumerate(self.shape)], dim=-1)
    return indices.masked_select(mask.unsqueeze(-1).expand(*mask.shape, self.ndim)).reshape(-1, self.ndim)

  # ***** reduce ops *****

  def keccak(self, cfg:str|tuple[int, int]="sha3_256"):
    """
    Calculates a Keccak hash over the last dimension. Uses "sha3_256" by default.

    ```python exec="false" source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").keccak()
    print(t.data().hex())
    ```
    """

    # https://keccak.team/keccak_specs_summary.html

    def ctensor(l: Sequence[PyConst], dtype: DType = dtypes.uint64):
      # TODO: contiguous is here for compile speed
      return Tensor.stack(*(Tensor(v, dtype=dtype, device=self.device) for v in l)).contiguous()
    rot_offsets = [44, 43, 21, 14, 28, 20, 3, 45, 61, 1, 6, 25, 8, 18, 27, 36, 10, 15, 56, 62, 55, 39, 41, 2]
    rot_offsets_v0, rot_offsets_v1 =  ctensor([0] + [1 << v for v in rot_offsets]), ctensor([1] + [1 << (64 - v) for v in rot_offsets])

    # calculated from π step
    reorder_indexes = ctensor([0,6,12,18,24,3,9,10,16,22,1,7,13,19,20,4,5,11,17,23,2,8,14,15,21], dtype=dtypes.int32)
    rnd_const_masks = [ctensor([v]).pad((0, 24)) for v in (1, 0x8082, 0x800000000000808a, 0x8000000080008000, 0x808b, 0x80000001, 0x8000000080008081,
    0x8000000000008009, 0x8a, 0x88, 0x80008009, 0x8000000a, 0x8000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x80000001, 0x8000000080008008)]

    rate, dsbyte = {"sha3_224": (144, 6), "sha3_256": (136, 6), "shake_128": (168, 31)}[cfg] if isinstance(cfg, str) else cfg
    data, data_pad = self.bitcast(dtypes.uint8).reshape(prod(self.shape[:-1]), self.shape[-1]), rate - (self.shape[-1] * self.dtype.itemsize % rate)
    # pad batches then pad blocks
    data = data.pad((None, (0, data_pad))).reshape(bs := data.shape[0], -1, rate).pad((None, None, (0, 200 - rate)))

    # create pad mask
    lbe = prod(data.shape[1:]) + rate - data_pad - 200
    if data_pad == 1: mb = [(lbe, 0), (1, dsbyte ^ 0x80), (200 - rate, 0)]
    else: mb = [(lbe, 0), (1, dsbyte), (data_pad - 2, 0), (1, 0x80), (200 - rate, 0)]
    pad_mask = Tensor.cat(*(Tensor(v, dtype=dtypes.uint8, device=data.device).expand(l) for l, v in mb if l > 0)).unsqueeze(0)

    data = (data.flatten(1) ^ pad_mask).reshape(*data.shape[:2], 200).bitcast(dtypes.uint64)

    state = Tensor.zeros(bs, 25, device=self.device, dtype=dtypes.uint64)
    for k in range(int(data.shape[1])):
      state = state ^ data.shrink((None, (k, k+1), None)).squeeze(1)
      for i in range(24): # f1600
        # θ step
        p = state.reshape(bs, 5, 5).transpose(2, 1)
        t1 = (p[:,:,0] ^ p[:,:,1] ^ p[:,:,2] ^ p[:,:,3] ^ p[:,:,4]).roll(-1, 1) # xor reduce
        state = state ^ (t1.roll(2, 1).bitwise_xor((t1 << 1) ^ (t1 >> 63)).unsqueeze(2).expand(bs, 5, 5).transpose(2, 1).flatten(1))
        # ρ and π steps
        state = state[:, reorder_indexes]
        state = (state * rot_offsets_v0).bitwise_or(state // rot_offsets_v1).reshape(bs, 5, 5)
        # χ and ι step
        state = state.bitwise_xor(~state.roll(shifts=-1, dims=2) & state.roll(shifts=-2, dims=2))
        state = state.flatten(1) ^ rnd_const_masks[i]
      # NOTE: there was a kernelize here to prevent internal stack from growing propotional to data size, do we need something else?
    return state.bitcast(dtypes.uint8)[:,:(obytes:=(200 - rate) // 2)].reshape(*self.shape[:-1], obytes)

  def _hash_1mb(self) -> Tensor:
    assert self.dtype == dtypes.uint8, "only support uint8 tensors for hashing"
    assert self.ndim == 2, "only support batched 1d tensors"
    assert self.shape[1] == 1024 * 1024, "only support messages of 1mb"

    blocks = self.shape[0] * self.shape[1] // 4096
    data = self.reshape(blocks, 4096)
    block_hashes = data.keccak("shake_128").reshape(self.shape[0], 4096)
    return block_hashes.keccak("shake_128").reshape(self.shape[0], 16)

  def hash(self) -> Tensor:
    """
    Calculates a 16-byte hash of the tensor.
    ```python exec="false source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").hash()
    print(t.data().hex())
    ```
    """

    data = self.flatten().bitcast(dtypes.uint8)
    if (tsize := data.shape[0]) % 2**20 != 0: data = data.pad((0, 2**20 - tsize % 2**20))
    base_chunks = ceildiv(data.shape[0], 2**20)
    tree_depth = math.ceil(math.log(base_chunks, 65536)) if base_chunks > 1 else 0

    level_chunks = base_chunks
    for _ in range(tree_depth + 1):
      data = data.reshape(level_chunks, 2**20)._hash_1mb().flatten()
      if (tsize := data.shape[0]) % 2**20 != 0: data = data.pad((0, 2**20 - tsize % 2**20))
      level_chunks = ceildiv(data.shape[0], 2**20)

    return data[:16]

  # ***** processing ops *****

  # TODO: winograd can be a rewrite rule like split_reduceop
  def _conv2d_winograd(self, weight:Tensor, bias:Tensor|None, groups:int, padding:int|Sequence[int], dtype:DTypeLike|None) -> Tensor:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    padding_ = resolve_pool_pads(padding, len(HW))
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape),\
        f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    rcout, oyx = cout//groups, self.pad(padding_)._pool(HW, 1, 1).shape[2:-len(HW)]
    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # TODO: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    pads = [(pB, pA + (-(s + pB + pA - 2) % 4)) for (pB, pA), s in zip(flat_to_grouped(padding_), self.shape[-len(HW):])]
    d = self.pad(flatten(reversed(pads)))._pool(HWI, HWO)
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
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink_to(bs, cout, *oyx)

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def conv2d(self, weight:Tensor, bias:Tensor|None=None, groups=1, stride=1, dilation=1, padding:int|Sequence[int]=0,
             dtype:DTypeLike|None=None) -> Tensor:
    """
    Applies a convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    if IMAGE: return self.image_conv2d(weight, bias, groups, stride, dilation, padding, dtype)
    if WINO and all(x == 3 for x in weight.shape[2:]) and stride == dilation == 1: return self._conv2d_winograd(weight, bias, groups, padding, dtype)
    return super().conv2d(weight, bias, groups, stride, dilation, padding, dtype)

  def dot(self, w:Tensor, dtype:DTypeLike|None=None) -> Tensor:
    if IMAGE: return self.image_dot(w, dtype)
    return super().dot(w, dtype)

  # ***** unary ops *****

  def contiguous(self, *args, **kwargs) -> Tensor:
    """
    Returns a contiguous tensor.
    """
    return self._apply_uop(UOp.contiguous, extra_args=args, **kwargs)

  # ***** broadcasted elementwise ops *****

  def ufix(self, x) -> Tensor:
    # TODO: x:ConstType|UOp does not work because mixin only accepts Self | ConstType
    assert isinstance(x, (*get_args(ConstType), UOp)), f"{type(x)=}, {x=}"
    return Tensor(x, self.device, self.dtype if self._ufix_keep_dtype(x) else None, requires_grad=False)

  def div(self, x:Tensor|ConstType|UOp, reverse=False, rounding_mode:Literal["trunc", "floor"]|None=None) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self / x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
    `div` performs true division.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.div(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
    ```
    """
    if rounding_mode is None: return super().div(x, reverse)  # type: ignore[arg-type]
    numerator, denominator = self._broadcasted(x, reverse)
    if dtypes.is_int(dt:=least_upper_dtype(numerator.dtype, denominator.dtype)):
      numerator, denominator = numerator.cast(dt), denominator.cast(dt)
      if rounding_mode == "trunc": return numerator.idiv(denominator)
      if rounding_mode == "floor":
        truncate_div, truncate_mod = numerator.idiv(denominator), numerator._binop(Ops.MOD, denominator, False)
        opposite_sign = ((numerator>0)&(denominator<0)) | ((numerator<0)&(denominator>0))
        return (opposite_sign&(truncate_mod!=0)).where(truncate_div-1, truncate_div)
    d = numerator.cast(least_upper_float(numerator.dtype)) * denominator.cast(least_upper_float(denominator.dtype)).reciprocal()
    output_dtype = numerator.dtype if dtypes.is_int(numerator.dtype) else d.dtype
    if rounding_mode == "trunc": return d.trunc().cast(output_dtype)
    if rounding_mode == "floor": return d.floor().cast(output_dtype)
    raise RuntimeError(f"{rounding_mode=} is not supported")

  def mod(self, x:Tensor|ConstType, reverse=False) -> Tensor:
    """
    Mod `self` by `x`.
    Equivalent to `self % x`.
    Supports broadcasting to a common shape, type promotion, and integer inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
    ```
    """
    a, b = self._broadcasted(x, reverse)
    return a - a.div(b, rounding_mode="floor") * b

  def where(self:Tensor, x:Tensor|ConstType|sint, y:Tensor|ConstType|sint) -> Tensor:
    """
    Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
    `output_i = x_i if self_i else y_i`.

    ```python exec="true" source="above" session="tensor" result="python"
    cond = Tensor([[True, True, False], [True, False, False]])
    print(cond.where(1, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    cond = Tensor.randn(2, 3)
    print(cond.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((cond > 0).where(cond, -float("inf")).numpy())
    ```
    """
    if isinstance(x, Tensor): x, y = x._broadcasted(y)
    elif isinstance(y, Tensor): y, x = y._broadcasted(x)
    else: x, y = Tensor(x, self.device, requires_grad=False)._broadcasted(y)
    out_shape = _broadcast_shape(self.shape, x.shape)
    return self.cast(dtypes.bool)._broadcast_to(out_shape)._apply_uop(UOp.where, x._broadcast_to(out_shape), y._broadcast_to(out_shape))

  # ***** op wrappers *****

  # TODO: combine with UOps __floordiv__
  def __floordiv__(self, x): return self.div(x, rounding_mode="floor")
  def __rfloordiv__(self, x): return self.div(x, rounding_mode="floor", reverse=True)

  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.__floordiv__(x))

  # unlike Tensors, UOps are immutable, so these don't go in mixin
  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x)) # type: ignore[misc]
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x)) # type: ignore[misc]
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x)) # type: ignore[misc]
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x)) # type: ignore[misc]
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x)) # type: ignore[misc]
  def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x)) # type: ignore[misc]
  def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x)) # type: ignore[misc]
  def __ixor__(self, x) -> Tensor: return self.assign(self.bitwise_xor(x)) # type: ignore[misc]
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x)) # type: ignore[misc]
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x)) # type: ignore[misc]
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x)) # type: ignore[misc]

  def __eq__(self, x) -> Tensor: return self.eq(x)                      # type: ignore[override]

  # ***** encoding/decoding ops *****

  def decode_hevc_frame(self, frame_pos:Variable, shape:tuple[int,...], state:Tensor, ref_frames:list[Tensor]|None=None) -> Tensor:
    """
    Creates a Tensor by decoding an HEVC frame chunk.

    You must provide the output shape of the decoded data (`shape`), the HEVC context (`vstate`), and, if required by the chunk,
    the reference frames (`ref_frames`).
    """
    ref_frames = [x.contiguous() for x in ref_frames or []]
    assert frame_pos.op is Ops.BIND, "frame_pos must be a bound Variable"
    srcs = (out:=Tensor.empty(*shape, device=self.device, dtype=self.dtype), self.contiguous(), state.contiguous(), *ref_frames)
    fn = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(frame_pos.src[0], *[UOp.const(dtypes.int, s) for s in shape]), arg="encdec")
    return Tensor(out.uop.after(fn.call(*[s.uop for s in srcs], frame_pos)))

  # ***** functional nn ops *****

  def dropout(self, p=0.5) -> Tensor:
    """
    Applies dropout to `self`.

    NOTE: dropout is only applied when `Tensor.training` is `True`.

    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Tensor.train():
      print(t.dropout().numpy())
    ```
    """
    if not 0 <= p <= 1: raise ValueError(f"{p=} is out of range [0, 1]")
    if not Tensor.training or p == 0: return self
    if p == 1: return self.zeros_like()
    return (Tensor.rand_like(self, requires_grad=False, dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Tensor|None=None, dropout_p:float=0.0,
                                   is_causal:bool=False, enable_gqa:bool=False) -> Tensor:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

    - Paper: https://arxiv.org/abs/1706.03762v7

    ```python exec="true" source="above" session="tensor" result="python"
    q = Tensor.randn(2, 4, 8)
    k = Tensor.randn(2, 4, 8)
    v = Tensor.randn(2, 4, 8)
    print(q.scaled_dot_product_attention(k, v).numpy())
    ```
    """
    # GQA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if enable_gqa:
      key = key.repeat_interleave(int(self.shape[-3] // key.shape[-3]), dim=-3)
      value = value.repeat_interleave(int(self.shape[-3] // value.shape[-3]), dim=-3)

    q = self
    qk = q.matmul(key.transpose(-2,-1), dtype=least_upper_dtype(q.dtype, key.dtype, dtypes.float32)) / math.sqrt(q.shape[-1])
    # handle attention mask
    if is_causal:
      if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
      attn_mask = qk.ones_like(requires_grad=False, dtype=dtypes.bool).tril()
    if attn_mask is not None:
      if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
      qk = qk + attn_mask
    return qk.cast(self.dtype).softmax(-1).dropout(dropout_p) @ value

  def qr(self) -> tuple[Tensor, Tensor]:
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
    R = self.clone()
    Q = Tensor.eye(m, dtype=self.dtype).reshape((1,) * len(b_shape) + (m, m)).expand(b_shape + (m, m))
    for i in range(min(m, n)):
      x = R[..., i:m, i]
      norm = x.square().sum(-1).sqrt()
      s = (x[..., 0] != 0).where(-x[..., 0].sign(), -1)
      u1 = x[..., 0] - s * norm
      w = x.unsqueeze(-1) / (norm != 0).where(u1, 1).reshape(b_shape + (1, 1))
      w[..., 0, 0] = 1
      tau = (-s * u1 / (norm != 0).where(norm, 1)).reshape(b_shape + (1, 1))
      tau = (norm != 0).reshape(b_shape + (1, 1)).where(tau, 0)
      R[..., i:m, :] = R[..., i:m, :] - (w * tau) @ (w.transpose(-2, -1) @ R[..., i:m, :])
      Q[..., :, i:m] = Q[..., :, i:m] - (Q[..., :, i:m] @ w) @ (tau * w).transpose(-2, -1)
    return Q,R

  def svd(self, full_matrices = True) -> tuple[Tensor, Tensor, Tensor]:
    #partial implementation of https://www.netlib.org/lapack/lawnspdf/lawn169.pdf , pg 26
    assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
    b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
    #preprocess the matrix
    Q, R = (self.qr() if m >= n else self.transpose(-2, -1).qr())
    num, q_num = min(m, n), max(m, n)
    # TODO: codegen infinite loop without contiguous
    U = R.shrink(tuple([None] * len(b_shape) + [(0, num), (0, num)])).contiguous()
    V = Tensor.eye(num, dtype=self.dtype).reshape((1,) * len(b_shape) + (num, num)).expand(b_shape + (num, num)).contiguous()
    #prepare round robin pairing
    permute, inverse_permute = Tensor.arange(0, num, dtype=dtypes.int), Tensor.zeros(num, dtype=dtypes.int)
    permute[num//2:num] = permute[num//2:num].flip(0)
    inverse_permute[permute] = Tensor.arange(num, dtype=dtypes.int)
    def one_round_jacobi(U, V,permute,inverse_permute):
      #pair all the columns
      V_permuted, runoff_V = (V[..., permute].split(num - 1, -1)) if num % 2 == 1 else (V[..., permute], None)
      V_left, V_right = V_permuted.split(num//2, -1)
      U_permuted, runoff_U = (U[..., permute].split(num - 1, -1)) if num % 2 == 1 else (U[..., permute], None)
      U_left, U_right = U_permuted.split(num//2, -1)
      #compute the jacobi rotations for each pairing
      gamma = (U_left * U_right).sum(-2).reshape(b_shape + (1, num//2))
      alpha, beta = U_permuted.square().sum(-2).unsqueeze(-2).split(num//2, -1)
      rot = gamma != 0
      tau = (beta - alpha) / (2 * rot.where(gamma, 1))
      t = (tau != 0).where(tau.sign(), 1) / (tau.abs() + (1 + tau.square()).sqrt())
      t = rot.where(t, 0)
      c = 1 / (1 + t.square()).sqrt()
      s = c * t
      #apply the rotations
      U_left, U_right = c * U_left - s * U_right, s * U_left + c * U_right
      U = U_left.cat(U_right.cat(runoff_U, dim = -1) if num % 2 == 1 else U_right, dim = -1)[..., inverse_permute]
      V_left, V_right = c * V_left - s * V_right, s * V_left + c * V_right
      V = V_left.cat(V_right.cat(runoff_V, dim = -1) if num % 2 == 1 else V_right, dim = -1)[..., inverse_permute]
      #prepare the next round robin pairings
      if num % 2 == 1: permute = ((permute - 1) % num)
      else: permute = permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1)
      inverse_permute = inverse_permute.scatter(0,permute,Tensor.arange(num,dtype=dtypes.int32))
      return U, V, permute, inverse_permute
    max_iterations, iterations_per_round = 1, int(num * math.log2(num) * 2 + 2)#sorta heuristic, most use num*log2(num)
    for _ in range(max_iterations * iterations_per_round): U, V, permute, inverse_permute = one_round_jacobi(U, V, permute, inverse_permute)
    #extract singular values and sort. construct U from Q
    S, indices = U.square().sum(-2).sqrt().sort(dim = -1, descending=True)
    new_indices = indices.reshape(b_shape + (1, num)).expand(b_shape + (num, num))
    U = U.gather(-1, new_indices) / (S != 0).where(S, 1).unsqueeze(-2)
    V = V.gather(-1, new_indices)

    padded_u = Tensor.eye(q_num, dtype=U.dtype).reshape((1,) * len(b_shape) + (q_num, q_num)).expand(b_shape + (q_num, q_num))
    padded_u[..., 0:num, 0:num] = U
    U = Q @ padded_u
    if not full_matrices: U, V = U[..., 0:num], V[..., 0:num]
    return (U, S, V.transpose(-2,-1)) if m >= n else (V, S, U.transpose(-2, -1))

  # ***** cast ops *****

  def cast(self, dtype:DTypeLike) -> Tensor:
    """
    Casts `self` to the given `dtype`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.uint8)
    print(t.dtype, t.numpy())
    ```
    """
    return self if self.dtype == (dt:=to_dtype(dtype)) else self._apply_uop(UOp.cast, dtype=dt)

  def bitcast(self, dtype:DTypeLike) -> Tensor:
    """
    Bitcasts `self` to the given `dtype` of the same itemsize.

    `self` must not require a gradient.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
    if self.requires_grad: raise RuntimeError("can't backprop through bitcast")
    dt = to_dtype(dtype)
    if (ns:=dt.itemsize) != (os:=self.dtype.itemsize) and (self.shape[-1]*os) % ns != 0: raise RuntimeError("unsupported size in bitcast")
    if (not isinstance(self.device, str) or not self.device.startswith("DISK")) and ns != os:
      new_uint, old_uint = to_dtype(f"uint{8*ns}"), to_dtype(f"uint{8*os}")
      tmp = self.bitcast(old_uint)
      if ns > os:
        tmp = tmp.reshape(self.shape[:-1] + (self.shape[-1]//(rate := ns//os), rate))
        nones = (None,) * (tmp.ndim - 1)
        return Tensor.usum(*[tmp.shrink(nones + ((i, i+1),)).cast(new_uint)<<8*i*os for i in range(rate)]).squeeze(-1).bitcast(dtype)
      return Tensor.stack(*(tmp>>8*i*ns for i in range(os//ns)), dim=-1).flatten(-2).cast(new_uint).bitcast(dtype)
    return self._apply_uop(UOp.bitcast, dtype=dt) if self.dtype != dt else self

  # *** image Tensor function replacements ***

  def image_dot(self, w:Tensor, dtype:DTypeLike|None=None) -> Tensor:
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    if not (self.ndim > 0 and w.ndim > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {self.ndim=}, {w.ndim=}")
    if self.shape[-1] != w.shape[-min(w.ndim, 2)]: raise RuntimeError(f"cannot image_dot {self.shape} and {w.shape}")

    bs, groups, cin, cout = prod(self.shape[0:-2]), prod(w.shape[0:-2]), w.shape[-2], w.shape[-1]
    out_shape_t = self.shape[0:-2] + (cout,-1) if len(self.shape) > 1 else (cout,)

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(self.ndim-1, self.ndim-2).reshape(bs//groups, groups*cin, -1, 1)
    # groups*cout x cin x H, W
    cw = w.transpose(w.ndim-1, w.ndim-2).reshape(groups*cout, cin, 1, 1)
    return cx.image_conv2d(cw, groups=groups, dtype=dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

  def image_conv2d(self, weight:Tensor, bias:Tensor|None=None, groups=1, stride=1, dilation=1, padding=0, dtype=None) -> Tensor:
    dtsz = 2 if FLOAT16 else 4

    (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
    x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

    padding_neg, padding_pos = [min(0, p) for p in resolve_pool_pads(padding, 2)], [max(0, p) for p in resolve_pool_pads(padding, 2)]
    x = x.pad(padding_neg)
    iy, ix = x.shape[2:]

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
      added_input_channels = 4 - (cin % 4)
      cin = cin + added_input_channels
      w = w.pad_to(None, None, cin, None, None)
      x = x.pad_to(None, None, cin, None, None).reshape(bs, groups*cin, iy, ix)

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.pad_to(None, rcout, None, None, None)

    # packed (note: flipping bs and iy would make the auto-padding work)
    x = x.permute(0,2,3,1)
    cin_last = iy == 1 and ix == 1
    if cin == 1: w = w.reshape(cout//4,4,H,W).permute(0,2,3,1)
    elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1)

    # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    def is_pow2(v): return v > 0 and v & (v - 1) == 0
    # pad dimension i to amt with invalids
    def ipad(t, i, amt):
      shape = (None,)*i + (amt,) + (None,)*(t.ndim-i-1)
      return Tensor(True, device=t.device).expand(t.shape).pad_to(shape).where(t.pad_to(shape), Invalid) if amt != t.shape[i] else t

    # align a dimension, use at to specify the dimension to pad in, defaults to first
    def pad_align(t, dim, at=None, force=False):
      # align to 64 pixels when height is real, otherwise 64 bytes is sufficient
      align = (64 // dtsz) if prod(t.shape[:dim]) == 1 or prod(t.shape) < 16384 * 4 else 256
      return ipad(t, at:=at or dim, round_up(t.shape[at] + int(force), align // math.gcd(prod(t.shape[dim:]) // t.shape[at], align)))

    # bank conflicts
    if cin >= 8 and is_pow2(cin // 4):
      x, w = pad_align(x.reshape(bs, iy, ix, groups, cin // 4, 4), 2, at=4, force=True), pad_align(w, 1, at=2, force=True)
    else: x, w = pad_align(x, 2), pad_align(w, 1)

    if FLOAT16: x, w = x.cast(dtypes.half).contiguous().cast(dtypes.float), w.cast(dtypes.half).contiguous().cast(dtypes.float)
    else: x, w = x.contiguous(), w.contiguous()

    # undo alignment hacks
    if cin >= 8 and is_pow2(cin // 4): x, w = x[:, :, :ix, :, :cin // 4, :], w[:, :H, :cin // 4, ...]
    else: x, w = x[:, :, :ix, :], w[:, :H, ...]

    # expand out
    rcin_hi, rcin_lo = (cin//4, 4) if cin >= 4 else (1, 1)
    group_shape, rcout_expand = (groups//4, 4) if cin == 1 else (groups, 1), (rcout//4, 4) if rcout >= 4 else (1, 1)
    x = x.reshape(bs, iy, -1, groups, rcin_hi, rcin_lo)
    if cin_last: w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)
    else: w = w.reshape(cout//4, H, rcin_hi, W, rcin_lo, 4).permute(0,1,2,3,5,4)

    # prepare input
    x = x.permute(0,3,4,5,1,2).pad(padding_pos)._pool((H,W), stride, dilation)# -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *group_shape, 1, 1, rcin_hi, rcin_lo, H, W)

    # prepare weights
    w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *group_shape, *rcout_expand, rcin_hi, rcin_lo, H, W))

    added_ox = 0
    assert isinstance(ox, int) and isinstance(cout, int)
    if (ox * cout) % (64 // dtsz):
      added_ox = round_up(ox, 64 // (dtsz * math.gcd(cout, 64 // dtsz))) - ox
      ox = ox + added_ox
      x = x.pad_to(None, None, ox, None, None, None, None, None, None, None, None)

    # the conv!
    ret = (x*w).cast(dtypes.float32).sum((-4, -3, -2, -1), dtype=dtype)

    if added_ox:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :-added_ox, ...]
      ox = ox - added_ox

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      cout = groups * (rcout - added_output_channels)

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

P = ParamSpec("P")
T = TypeVar("T")

# this tracks the tensor.py METADATA, contextvars.ContextVar was switched to this due to thread safety issues
class _ContextVar(Generic[T]):
  def __init__(self, default:T): self.state:T = default
  def get(self) -> T: return self.state
  def set(self, x:T) -> T:
    ret, self.state = self.state, x
    return ret
_METADATA: _ContextVar[Metadata|None] = _ContextVar(default=None)

def _metadata_wrapper(fn: Callable[P, T]) -> Callable[P, T]:
  def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
    if TRACEMETA < 1 or _METADATA.get() is not None: return fn(*args, **kwargs)

    if TRACEMETA >= 2:
      caller_frame = sys._getframe(frame := 1)
      caller_module = caller_frame.f_globals.get("__name__", None)
      caller_func = caller_frame.f_code.co_name
      if caller_module is None: return fn(*args, **kwargs)

      # if its called from nn we want to step up frames until we are out of nn
      while caller_module.startswith("tinygrad.nn") and "optim" not in caller_module:
        caller_frame = sys._getframe(frame := frame + 1)
        caller_module = caller_frame.f_globals.get("__name__", None)
        if caller_module is None: return fn(*args, **kwargs)

      # if its called from a lambda in tinygrad we want to look two more frames up
      if caller_module.startswith("tinygrad") and caller_func == "<lambda>": caller_frame = sys._getframe(frame := frame + 2)
      caller_module = caller_frame.f_globals.get("__name__", None)
      if caller_module is None: return fn(*args, **kwargs)
      caller_func = caller_frame.f_code.co_name
      caller_lineno = caller_frame.f_lineno

      caller = f"{caller_module}:{caller_lineno}::{caller_func}"
    else: caller = ""

    token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))
    with cpu_profile(TracingKey(fn.__name__), "USER"):
      ret = fn(*args, **kwargs)
    _METADATA.set(token)
    return ret
  return _wrapper

if TRACEMETA >= 1:
  for name, fn in inspect.getmembers(Tensor, inspect.isfunction):
    if name in ["__class__", "__del__", "__init__", "__new__", "__repr__", "backward", "sequential", "gradient"]: continue
    setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))

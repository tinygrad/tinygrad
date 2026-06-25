# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools, sys, inspect, pathlib, hashlib, weakref
from typing import Any, Callable, Sequence, cast, get_args, ParamSpec, TypeVar, Generic, TYPE_CHECKING
if TYPE_CHECKING: import numpy
from tinygrad.dtype import DType, DTypeLike, dtypes, ConstType, to_dtype
from tinygrad.dtype import _from_np_dtype, _to_np_dtype, PyConst, Invalid
from tinygrad.helpers import argfix, flatten, prod, all_int, round_up, getenv, fully_flatten, ceildiv, fetch, flat_to_grouped
from tinygrad.helpers import resolve_pool_pads, IMAGE, FLOAT16, WINO, Metadata, TRACEMETA, is_numpy_ndarray, TracingKey, cpu_profile
from tinygrad.helpers import suppress_finalizing, disable_gc, TRAINING
from tinygrad.uop.ops import UOp, Ops, sint, all_metadata, _index_to_concrete_int, Variable, _broadcast_shape
from tinygrad.mixin.rand import RandMixin
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

def _get_winograd_matcols(mat, dims:int, shp:tuple[sint, ...], dtype:DType) -> list[list[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), dtype=dtype, buffer=False) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.dtype)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

# TODO: deprecate this, always use TRAINING
class TensorMeta(type):
  @property
  def training(cls) -> bool: return bool(TRAINING.value)
  @training.setter
  def training(cls, mode:bool): TRAINING.value = int(mode)

class Tensor(RandMixin, metaclass=TensorMeta):
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn, Context
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "uop", "is_param", "grad"

  def __init__(self, data:ConstType|bytes|list|tuple|UOp|'numpy.ndarray'|pathlib.Path|None,
               device:str|tuple|list|None=None, dtype:DTypeLike|None=None):
    if device is None:
      if isinstance(data, pathlib.Path): device = f"DISK:{data.resolve()}"  # keep it on the disk if device is None
      elif isinstance(data, UOp): device = data.device
    _dtype:DType|None = to_dtype(dtype) if dtype is not None else None
    _device:str|tuple[str, ...] = canonicalize_device(device)
    del device, dtype

    # tensors can have gradients if you have called .backward
    self.grad:Tensor|None = None

    self.is_param:bool = True

    # create a UOp from the different types of inputs
    if isinstance(data, UOp):
      assert _dtype is None or _dtype==data.dtype or data.dtype==dtypes.weakint, f"dtype mismatch: {_dtype} vs {data.dtype}"
      # if data is dtype.weakint that means that this is a symbolic int and we need to lower it to something we can make a Tensor out of
      if data.dtype == dtypes.weakint: data = _index_to_concrete_int(data)
    elif data is None:
      data = UOp.const(_dtype or dtypes.default_float, 0)
    elif isinstance(data, get_args(ConstType)):
      data = UOp.const(_dtype or dtypes.from_py(data), data)
    elif isinstance(data, bytes): data = UOp._frompy(data, _dtype or dtypes.uint8, _device)
    elif isinstance(data, (list, tuple)):
      if _dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): _dtype = dtypes.bool
        else: _dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float  # NOTE: this works because all_int([True, False]) is True
      data = UOp._frompy(data, _dtype, _device)
    elif is_numpy_ndarray(data):
      import numpy as np
      assert isinstance(data, np.ndarray), f"expected np.ndarray, got {data}"
      if data.shape == ():
        data = UOp.const(_dtype or _from_np_dtype(data.dtype), data.item())
      else:
        data = _fromnp(data.astype(npdtype) if _dtype is not None and (npdtype:=_to_np_dtype(_dtype)) is not None else data)
    elif isinstance(data, pathlib.Path):
      _dtype = _dtype or dtypes.uint8
      data = UOp.new_buffer(f"DISK:{data.resolve()}", data.stat().st_size // _dtype.itemsize, _dtype)

    # by this point, it has to be a UOp
    if not isinstance(data, UOp): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

    # data might be on a different device
    self.uop:UOp = data if data.device is None or data.device == _device else data.copy_to_device(_device)

    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(self)] = None

  @suppress_finalizing
  def __del__(self): all_tensors.pop(weakref.ref(self), None)

  def _apply_uop(self, fxn:Callable[..., UOp], *x:Tensor, **kwargs) -> Tensor:
    srcs = (self,)+x
    new_uop: UOp = fxn(*[t.uop for t in srcs], **kwargs)
    if TRACEMETA >= 1 and (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
    # directly create the Tensor
    ret = Tensor.__new__(Tensor)
    ret.uop, ret.grad, ret.is_param = new_uop, None, True
    # add to all_tensors after construction succeeds
    all_tensors[weakref.ref(ret)] = None
    return ret

  # alu, _uop, _wrap_uop and const are used by the mixins
  def alu(self, op: Ops, *src: Tensor) -> Tensor: return self._apply_uop(lambda *u: u[0].alu(op, *u[1:]), *src)
  @property
  def _uop(self) -> UOp: return self.uop
  def _wrap_uop(self, u:UOp) -> Tensor: return Tensor(u)
  @staticmethod
  def const(dtype:DType, b:ConstType|UOp) -> Tensor: return Tensor(UOp.const(dtype, b))

  def is_param_(self, is_param:bool=True) -> Tensor:
    self.is_param = is_param
    return self

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
  def device(self) -> str|tuple[str, ...]|None: return self.uop.device

  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape

  @property
  def dtype(self) -> DType: return self.uop.dtype

  # ***** data handlers ****

  def as_param(self, slot:int):
    return Tensor(UOp.param(slot, self.dtype, self.uop.shard_shape, self.device, axis=self.uop.axis))

  def call(self, *lst:Tensor, fxn:Tensor|UOp, grad_fxn:Callable|None=None) -> Tensor:
    fret = fxn._uop.call(*[t.uop for t in (self,)+lst], grad_fxn=grad_fxn)
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
    if len(to_realize:=[x for x in (self,)+lst if x.uop.device is not None and not x.uop.has_buffer_identity()]):
      run_linear(*Tensor.linear_with_vars(*to_realize), update_stats=do_update_stats)
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
    x = x._broadcast_to(self.shape)
    if not is_disk and x.uop.device is not None and self.device is not None and self.device != x.device:
      raise RuntimeError(f"assign device mismatch {self.device} != {x.device}")
    if not is_disk and self.dtype != x.dtype: raise RuntimeError(f"assign dtype mismatch {self.dtype} != {x.dtype}")
    if isinstance(self.device, tuple) and x.uop.device is not None and self.uop.axis != x.uop.axis:
      raise RuntimeError(f"multi axis mismatch {self.uop.axis} != {x.uop.axis}")

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

  def _buffer(self) -> Buffer:
    from tinygrad.engine.realize import capturing
    if capturing and not getenv("UNSAFE_ALLOW_JIT_BUFFER"):
      from tinygrad.engine.jit import JitError
      raise JitError("cannot access tensor data during JIT capture, the value will be baked in")
    x = self.cast(self.dtype.base).contiguous()
    if self.uop.device is None or isinstance(self.device, tuple): x = x.clone("CPU")
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
    return self._data().cast(self.dtype.base.fmt, self.shape)

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

  def clone(self, device:str|tuple[str, ...]|None=None) -> Tensor:
    """
    Creates a clone of this tensor allocating a separate buffer for the data.
    If `device` is specified, the clone is placed on that device.
    """
    ret = Tensor(self.uop.clone(device=device))
    if self.grad is not None: ret.grad = self.grad.clone(device=device)
    return ret.is_param_(self.is_param)

  def to(self, device:str|tuple[str, ...]|None) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    if self.uop.device is None: return self
    if (device:=canonicalize_device(device)) == self.device: return self
    ret = Tensor(self.uop.copy_to_device(device))
    if self.grad is not None: ret.grad = self.grad.to(device)
    return ret.is_param_(self.is_param)

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
    if self.uop.device is None: return self
    if not isinstance(self.device, str): raise RuntimeError("can't shard a multi-device tensor")
    if len(devices) == 1: return self.to(devices[0])
    devices = cast(tuple[str, ...], canonicalize_device(devices))
    uop = self.uop.shard(devices, None if axis is None else self._resolve_dim(axis))
    return Tensor(uop).is_param_(self.is_param)

  def shard_(self, devices:tuple[str, ...], axis:int|None=None) -> Tensor:
    """
    Shards the tensor across the given devices in place.
    """
    return self.replace(self.shard(devices, axis))

  def shard_like(self, y:Tensor) -> Tensor:
    """
    Shards the tensor the same way as `y` (same devices and axis).
    """
    if y.device is None: return self
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

  # ***** creation entrypoint *****

  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor(UOp.empty(argfix(*shape), dtype, device))

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
  def _next_counter(device:str, num:int) -> tuple[Tensor, Tensor]:
    if device not in Tensor._device_seeds:
      seed = [int.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed]
      Tensor._device_seeds[device] = Tensor(seed, device=device, dtype=dtypes.uint32)
      Tensor._device_rng_counters[device] = Tensor([0, 0], device=device, dtype=dtypes.uint32)
    counter = Tensor._device_rng_counters[device]
    new_low = counter[0:1] + (num & 0xffffffff)
    new_high = counter[1:2] + (num >> 32) + (new_low < counter[0])
    counter.assign(new_low.cat(new_high))
    low = counter[0:1] - (num & 0xffffffff)
    high = counter[1:2] - (num >> 32) - (counter[0] < (num & 0xffffffff))
    return Tensor._device_seeds[device], low.cat(high)

  # ***** toposort and backward pass *****

  def backward(self, gradient:Tensor|None=None) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    # backward fills .grad for every in-scope non-CONST float tensor
    tensors_need_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.is_floating_point() and t.uop.op is not Ops.CONST]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      if g.device is None and t.device is not None: g = g.clone(device=t.device)
      if t.grad is None: t.grad = g
      else: t.grad.assign(t.grad + g.to(t.grad.device))
    return self

  # ***** movement ops *****

  def _mop(self, op:Ops, arg) -> Tensor: return self._apply_uop(UOp._mop, op=op, arg=arg)
  def _rop(self, op:Ops, axis:tuple[int, ...]) -> Tensor: return self._apply_uop(UOp._rop, op=op, axis=axis)

  def __setitem__(self, indices, v:Tensor|PyConst|list|tuple) -> None:
    if isinstance(v, Tensor) and v.dtype != self.dtype: raise RuntimeError(f"setitem dtype mismatch: {self.dtype=} != {v.dtype=}")
    # raise if mutation would diverge from eager (allow only pure views of a realized buffer; exclude +=/-= RHS via v_uop/v_bw)
    v_uop, v_bw = (v.uop, v.uop.backward_slice) if isinstance(v, Tensor) else (None, {})
    if self.uop.op_in_backward_slice_with_self(Ops.BUFFER):
      shared = self.uop.base if self.uop.base.is_realized else None
      if any(self.uop in t.uop.backward_slice_with_self and t.uop.base is not shared for tref in all_tensors
             if (t:=tref()) is not None and t is not self and t.uop is not v_uop and t.uop not in v_bw):
        raise RuntimeError("can't setitem on a tensor with other uses")
    idx = [indices] if (isinstance(indices, list) and all_int(indices)) or not isinstance(indices, (tuple, list)) else list(indices)
    is_disk = isinstance(self.device, str) and self.device.startswith("DISK")
    advanced = any(isinstance(i, (Tensor, list, tuple)) for i in idx)
    realized = is_disk or self.uop.base.op is Ops.BUFFER or self.uop._base_buffer_is_realized()
    if (not self.uop.base.is_realized and self.is_floating_point()) or not (advanced or realized):
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      # __iadd__/__isub__ creates AFTER(view, STORE(view, computed)); unwrap to get the computed value
      if v.uop.op is Ops.AFTER and any(s.op is Ops.STORE for s in v.uop.src[1:]): v = v._apply_uop(lambda x: x.src[1].src[1])
      self.replace(self._getitem(indices, v))
    elif advanced: # advanced setitem
      if is_disk: raise RuntimeError("advanced setitem is not supported for DISK tensors")
      if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
      self.assign(self._getitem(indices, v))
    else: # basic setitem
      view = self[indices]
      if isinstance(v, Tensor) and v.uop.op is Ops.AFTER and v.uop in view.uop.base.src: return
      view.assign(v)

  def __delitem__(self, indices) -> None:
    raise TypeError("Tensor does not support deleting items")

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
    data = self.bitcast(dtypes.uint8).reshape(prod(self.shape[:-1]), self.shape[-1])
    data_pad = rate - data.shape[-1] % rate
    # pad batches then pad blocks
    data = data.pad((None, (0, data_pad))).reshape(bs := data.shape[0], -1, rate).pad_to(None, None, 200)

    # create pad mask
    lbe = (data.shape[1] - 1) * 200 + rate - data_pad
    if data_pad == 1: mb = [(lbe, 0), (1, dsbyte ^ 0x80), (200 - rate, 0)]
    else: mb = [(lbe, 0), (1, dsbyte), (data_pad - 2, 0), (1, 0x80), (200 - rate, 0)]
    pad_mask = Tensor.cat(*(Tensor(v, dtype=dtypes.uint8, device=data.device).expand(l) for l, v in mb if l > 0)).unsqueeze(0)

    data = (data.flatten(1) ^ pad_mask).reshape(*data.shape[:2], 200).bitcast(dtypes.uint64)

    state = Tensor.zeros(bs, 25, dtype=dtypes.uint64, buffer=False)
    for k in range(int(data.shape[1])):
      state = state ^ data[:, k]
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
    return self.reshape(-1, 4096).keccak("shake_128").reshape(self.shape[0], -1).keccak("shake_128")

  def hash(self) -> Tensor:
    """
    Calculates a 16-byte hash of the tensor.
    ```python exec="false source="above" session="tensor" result="python"
    t = Tensor(b"Hello World!").hash()
    print(t.data().hex())
    ```
    """
    data = self.flatten().bitcast(dtypes.uint8)
    n = data.shape[0]
    assert isinstance(n, int), "hash requires concrete shape"
    chunks = ceildiv(n, 2**20)
    while chunks > 1:
      data = data.pad_to(chunks * 2**20).reshape(chunks, 2**20)._hash_1mb().flatten()
      chunks = ceildiv(chunks, 65536)
    return data.pad_to(2**20).unsqueeze(0)._hash_1mb().flatten()[:16]

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

    # compute 6x6 winograd tiles: GgGt, BtdB. contiguous so the transforms are materialized once
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).contiguous().reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).contiguous().reshape(*HWI, bs, groups, 1, cin, *tyx)

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

  # ***** broadcasted elementwise ops *****

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
    else: x, y = self.ufix(x)._broadcasted(y)
    out_shape = _broadcast_shape(self.shape, x.shape)
    return self.cast(dtypes.bool)._broadcast_to(out_shape)._apply_uop(UOp.where, x._broadcast_to(out_shape), y._broadcast_to(out_shape))

  # ***** op wrappers *****

  # unlike Tensors, UOps are immutable, so these don't go in mixin
  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x)) # type: ignore[misc]
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x)) # type: ignore[misc]
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x)) # type: ignore[misc]
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x)) # type: ignore[misc]
  def __ifloordiv__(self, x) -> Tensor: return self.assign(self.__floordiv__(x)) # type: ignore[misc]
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

  # ***** cast ops *****

  def bitcast(self, dtype:DTypeLike) -> Tensor:
    """
    Bitcasts `self` to the given `dtype` of the same itemsize.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
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

    (bs,_,_,_), (cout,cin,H,W) = self.shape, weight.shape
    assert isinstance(cin, int) and isinstance(cout, int)
    x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

    padding_neg, padding_pos = [min(0, p) for p in resolve_pool_pads(padding, 2)], [max(0, p) for p in resolve_pool_pads(padding, 2)]
    x = x.pad(padding_neg)
    iy, ix = x.shape[2:]

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      new_cin = round_up(cin, 4)
      w = w.pad_to(None, None, new_cin, None, None)
      x = x.reshape(bs, groups, cin, iy, ix)
      x = x.pad_to(None, None, new_cin, None, None).reshape(bs, groups*new_cin, iy, ix)
      cin = new_cin

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

    def is_pow2(v): return v > 0 and v & (v - 1) == 0
    # pad dimension i to amt with invalids
    def ipad(t, i, amt):
      return t.pad(tuple(None if d != i else (0, amt-s) for d,s in enumerate(t.shape)), value=Invalid) if amt != t.shape[i] else t
    # align a dimension, use at to specify the dimension to pad in, defaults to first
    def pad_align(t, dim, at=None, force=False):
      # align to 64 pixels when height is real, otherwise 64 bytes is sufficient
      align = (64 // dtsz) if prod(t.shape[:dim]) == 1 or prod(t.shape) < 16384 * 4 else 256
      return ipad(t, at:=at or dim, round_up(t.shape[at] + int(force), align // math.gcd(prod(t.shape[dim:]) // t.shape[at], align)))

    # bank conflicts
    bank_conflict = cin >= 8 and is_pow2(cin // 4)
    if bank_conflict:
      x, w = pad_align(x.reshape(bs, iy, ix, groups, cin // 4, 4), 2, at=4, force=True), pad_align(w, 1, at=2, force=True)
    else: x, w = pad_align(x, 2), pad_align(w, 1)

    # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    if FLOAT16: x, w = x.cast(dtypes.half).contiguous().cast(dtypes.float), w.cast(dtypes.half).contiguous().cast(dtypes.float)
    else: x, w = x.contiguous(), w.contiguous()

    # undo alignment hacks
    if bank_conflict: x, w = x[:, :, :, :, :cin // 4, :], w[:, :, :cin // 4, ...]
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

    # the conv!
    ret = (x*w).cast(dtypes.float32).sum((-4, -3, -2, -1), dtype=dtype)

    ret = ret.reshape(bs, oy, ox, groups, rcout)
    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels: ret = ret[:, :, :, :, :-added_output_channels]
    # NCHW output
    ret = ret.reshape(bs, oy, ox, groups * (rcout - added_output_channels)).permute(0,3,1,2)
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

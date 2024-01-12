# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Iterable, DefaultDict, cast, get_args
from collections import defaultdict
from functools import partialmethod, reduce
import numpy as np

from tinygrad.dtype import DType, dtypes, ImageDType, least_upper_float, least_upper_dtype
from tinygrad.helpers import argfix, make_pair, getenv, IMAGE, DEBUG, flatten, prod, all_int, round_up, merge_dicts, fully_flatten
from tinygrad.lazy import LazyBuffer, create_schedule
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.device import Device, Buffer
from tinygrad.shape.symbolic import sint
from tinygrad.realize import run_schedule

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

import tinygrad.mlops as mlops

def _loadop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Optional[LazyBuffer]=None):
  if isinstance(device, str): return LazyBuffer.loadop(op, shape, dtype, device, arg, src)
  return MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype, d, arg, src) for d in device], None)

Scalar = Union[float, int, bool]

class Tensor:
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  class train:
    def __init__(self, val=True): self.val = val
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  no_grad: ClassVar[bool] = False
  def __init__(self, data:Union[None, Scalar, List, Tuple, LazyBuffer, np.ndarray, bytes, MultiLazyBuffer],
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
    elif isinstance(data, get_args(Scalar)): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, bytes): data = LazyBuffer.fromCPU(np.frombuffer(data, np.uint8))
    elif data is None: data = _loadop(LoadOps.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, list):
      if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtype or dtypes.bool
      elif d and all_int(d): dtype = dtype or dtypes.default_int
      else: dtype = dtype or dtypes.default_float
      # NOTE: cast at the end for the dtypes that do not have a numpy dtype
      data = LazyBuffer.fromCPU(np.array(data, dtype.np)).cast(dtype)
    elif isinstance(data, np.ndarray):
      if data.shape == (): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
      else: data = LazyBuffer.fromCPU(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

    # data is a LazyBuffer, but it might be on the wrong device
    if not isinstance(data, (LazyBuffer, MultiLazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
    if isinstance(device, tuple):
      # TODO: what if it's a MultiLazyBuffer on other devices?
      self.lazydata: Union[LazyBuffer, MultiLazyBuffer] = MultiLazyBuffer.from_sharded(data, device, None) if isinstance(data, LazyBuffer) else data
    else:
      self.lazydata = data if data.device == device else data.copy_to_device(device)

  def __repr__(self):
    return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  @staticmethod
  def corealize(lst:Iterable[Tensor]):
    return run_schedule(create_schedule(flatten([x.lazydata.lbs if isinstance(x.lazydata, MultiLazyBuffer) else [x.lazydata] for x in lst])))

  def realize(self) -> Tensor:
    run_schedule(self.lazydata.schedule())
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="CPU", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x.numpy().data)
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert not x.requires_grad  # self requires_grad is okay?
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.dtype == x.dtype and not getenv("DISALLOW_ASSIGN"):
      if isinstance(self.lazydata, MultiLazyBuffer):
        for d,s in zip(x.lazydata.lbs, self.lazydata.lbs): d.output_buffer = s.base.realized
      else:
        if self.lazydata.base.realized is not None: x.lazydata.output_buffer = self.lazydata.base.realized
    self.lazydata = x.lazydata
    return self
  def detach(self) -> Tensor: return Tensor(self.lazydata, device=self.device, requires_grad=False)

  # TODO: these are good places to start removing numpy
  def item(self) -> Scalar:
    assert self.numel() == 1, "must have one element for item"
    return cast(Buffer, self.contiguous().realize().lazydata.base.realized).toCPU().item()
  def data(self) -> memoryview: return self.numpy().data

  # TODO: this should import numpy and use .data() to construct the array
  def numpy(self) -> np.ndarray:
    assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}"
    assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
    if 0 in self.shape: return np.zeros(self.shape, dtype=self.dtype.np)
    t = self if isinstance(self.device, str) else self.to("CPU")
    return t.cast(self.dtype.scalar()).contiguous().realize().lazydata.base.realized.toCPU().astype(self.dtype.np, copy=True).reshape(self.shape)

  def to(self, device:Optional[str]) -> Tensor:
    if device is None or device == self.device: return self
    ret = Tensor(self.lazydata, device)
    if self.grad: ret.grad = self.grad.to(device)
    return ret

  def to_(self, device:Optional[str]):
    if device is None or device == self.device: return
    if self.grad: self.grad = self.grad.to_(device)
    _ret = Tensor(self.lazydata, device)
    self.lazydata = _ret.lazydata

  def shard(self, devices:Tuple[str, ...], axis:Optional[int]=None) -> Tensor:
    assert isinstance(self.lazydata, LazyBuffer), "can't shard a MultiLazyBuffer"
    canonical_devices = tuple(Device.canonicalize(x) for x in devices)
    return Tensor(MultiLazyBuffer.from_sharded(self.lazydata, canonical_devices, axis), device=canonical_devices)

  def shard_(self, devices:Tuple[str, ...], axis:Optional[int]=None):
    self.lazydata = self.shard(devices, axis).lazydata
    return self

  # ***** creation llop entrypoint *****

  @staticmethod
  def _loadop(op, shape, device:Optional[str]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
    return Tensor(LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs): return Tensor._loadop(LoadOps.EMPTY, argfix(*shape), **kwargs)

  _seed: int = int(time.time())
  @staticmethod
  def manual_seed(seed=0): Tensor._seed = seed

  @staticmethod
  def rand(*shape, **kwargs): return Tensor._loadop(LoadOps.CUSTOM, argfix(*shape), arg=custom_random, **kwargs)

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:Scalar, **kwargs):
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs): return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs): return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs):
    if stop is None: stop, start = start, 0
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs).cumsum() + (start - step)).cast(dtype)

  @staticmethod
  def eye(dim:int, **kwargs):
    return Tensor.ones((dim,1),**kwargs).pad((None,(0,dim))).flatten().shrink(((0,dim*dim),)).reshape(dim, dim)

  def full_like(self, fill_value:Scalar, **kwargs):
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  def zeros_like(self, **kwargs): return self.full_like(0, **kwargs)
  def ones_like(self, **kwargs): return self.full_like(1, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **kwargs)
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=low, high=high, dtype=dtypes.int32)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor: return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.default_int)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  def backward(self) -> Tensor:
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1.0, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      assert (t0.grad is not None)
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

  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)])
    return mlops.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self
  def expand(self, shape, *args) -> Tensor:
    new_shape = tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))])
    return mlops.Expand.apply(self, shape=new_shape) if new_shape != self.shape else self
  def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    if all(x is None or x == (0,s) for x,s in zip(arg, self.shape)): return self
    return mlops.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape)))
  def pad(self, arg:Tuple[Optional[Tuple[sint, sint]], ...], value:float=0.0) -> Tensor:
    if all(x is None or x == (0,0) for x in arg): return self
    ret = mlops.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

  # ***** movement hlops *****

  # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  # - A slice i:j returns the elements with indices in [i, j)
  #    - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
  #    - Negative values for i and j are taken relative to the end of the sequence
  #    - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
  # - Indexing with None on a given axis will add a new dimension of size one before that axis
  # - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
  # - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
  # - Strides > 1 and < 0 are now allowed!:
  #    - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
  #    - Idea of stride < 0 support:
  #        - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
  #    - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
  #        - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
  #        - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
  #          is possible.
  #        - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
  # - Fancy indexing and combined indexing is supported
  #    - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
  #    - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
  #        - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
  #    - There's a special case where a permute is needed at the end:
  #        - if first Tensor passed in (expand dims) is not at dim 0
  #        - and following Tensors does not follow consecutively to the end of fancy indexing's dims
  # TODO: boolean indices
  # TODO: figure out the exact acceptable types for indices, especially for internal list/tuple types
  # TODO: update docs
  def __getitem__(self, indices: Union[int, slice, Tensor, None, List, Tuple]) -> Tensor: # no ellipsis type...
    # 1. indices normalization and validation
    # treat internal tuples and lists as Tensors and standardize indices to list type
    if isinstance(indices, (tuple, list)):
      # special case <indices: List[int]>, a lil ugly
      if isinstance(indices, list) and all_int(indices): indices = [Tensor(indices, requires_grad=False, device=self.device)]
      else: indices = [Tensor(list(i), requires_grad=False, device=self.device) if isinstance(i, (tuple, list)) else i for i in indices]
    else: indices = [indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    ellipsis_idx = [dim for dim, i in enumerate(indices) if i is Ellipsis]
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_slices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    indices[fill_idx:fill_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

    # use Dict[type, List[dimension]] to track elements in indices
    type_dim: DefaultDict[Union[type, None], List[int]] = defaultdict(list)

    # record None for dimension injection later and filter None and record rest of indices
    type_dim[None] = [dim for dim, i in enumerate(indices) if i is None]
    indices_filtered = [v for v in indices if v is not None]
    for dim,i in enumerate(indices_filtered): type_dim[type(i)].append(dim)

    # validation! raise Errors
    if slice in type_dim and self.ndim == 0: raise IndexError("slice cannot be applied to a 0-dim tensor.")
    if len(ellipsis_idx) > 1: raise IndexError("an index can only have a single ellipsis ('...')")
    if float in type_dim: raise IndexError("float type is not valid index")
    if any(isinstance(i, slice) and i.step == 0 for i in indices): raise ValueError('slice step cannot be 0')
    if num_slices > len(self.shape): raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")

    # 2. basic indexing (no copy)
    # currently indices_filtered: Tuple[Union[slice, int, Tensor], ...]
    # turn indices in indices_filtered to Tuple[shrink_arg, strides]
    for dim in type_dim[int]:
      if (index := indices_filtered[dim]) >= (size := self.shape[dim]) or index < -size:
        raise IndexError(f"{index=} is out of bounds for dimension {dim} with {size=}")
      indices_filtered[dim] = ((index, index+1), 1) if index >= 0 else ((size+index, size+index+1), 1)
    for dim in type_dim[slice]:
      s, e, st = indices_filtered[dim].indices(self.shape[dim])
      indices_filtered[dim] = ((0, 0) if (st > 0 and e < s) or (st <= 0 and e > s) else (s, e) if st > 0 else (e+1, s+1), st)
    for dim in type_dim[Tensor]: indices_filtered[dim] = ((0, self.shape[dim]), 1)

    new_slice, strides = ((),()) if not indices_filtered else zip(*indices_filtered)
    ret = self.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
    # add strides by pad -> reshape -> shrink
    if any(abs(s) != 1 for s in strides):
      strides = tuple(abs(s) for s in strides)
      ret = ret.pad(tuple((0, round_up(sh, s) - sh) for s, sh in zip(strides, ret.shape)))
      ret = ret.reshape(flatten([sh // s, s] for s, sh in zip(strides, ret.shape)))
      ret = ret.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in ret.shape[::2]))).reshape(ret.shape[::2])

    # inject 1 for dim where it's None and collapse dim for int
    new_shape = list(ret.shape)
    for dim in type_dim[None]: new_shape.insert(dim, 1)
    for dim in (dims_collapsed := [dim + sum(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int])]): new_shape.pop(dim)
    assert all_int(new_shape), f"does not support symbolic shape {new_shape}"

    ret = ret.reshape(new_shape)

    # 3. advanced indexing (copy)
    if type_dim[Tensor]:

      # extract tensors and tensor dimensions
      idx, tdim = [], []
      for tensor_dim in type_dim[Tensor]:
        dims_collapsed_, dims_injected = sum(1 for d in dims_collapsed if tensor_dim >= d), sum(1 for d in type_dim[None] if tensor_dim >= d)
        tdim.append(td := tensor_dim - dims_collapsed_ + dims_injected)
        # normalize the negative tensor indices
        idx.append(((t := indices[tensor_dim + dims_injected]) < 0).where(ret.shape[td], 0) + t)
        # TODO uint8 and bool tensor indexing
        if not (dtypes.is_int(t.dtype) or t.dtype == dtypes.bool): raise IndexError("tensors used as indices must be int or bool tensors")

      # compute sum_dim, arange, and idx
      max_dim = max(i.ndim for i in idx)
      sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(tdim)]
      arange = [Tensor.arange(ret.shape[d], requires_grad=False, device=self.device).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, tdim))]   # noqa: E501
      first_idx = [idx[0].reshape(*[1]*tdim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - tdim[0] - 1))]
      rest_idx = [i.reshape(*[1]*tdim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - tdim[0] - n)) for n,i in enumerate(idx[1:], 1)]
      reshaped_idx = first_idx + rest_idx
      ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])

      # iteratively eq -> mul -> sum fancy index
      try:
        for a,i,sd in zip(arange, reshaped_idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
      except AssertionError as exc: raise IndexError(f"cannot broadcast with index shapes {', '.join(str(i.shape) for i in idx)}") from exc

      # special permute case
      if tdim[0] != 0 and len(tdim) != 1 and tdim != list(range(tdim[0], tdim[-1]+1)):
        ret_dims = list(range(ret.ndim))
        ret = ret.permute(ret_dims[tdim[0]:tdim[0]+max_dim] + ret_dims[:tdim[0]] + ret_dims[tdim[0]+max_dim:])
    return ret

  def __setitem__(self,indices,v): return self.__getitem__(indices).assign(v)

  # NOTE: using slice is discouraged and things should migrate to pad and shrink
  def slice(self, arg:Sequence[Optional[Tuple[int, sint]]], value:float=0) -> Tensor:
    arg_ = tuple(a if a is not None else (0, s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -l), max(0, r-s)) for s,(l,r) in zip(self.shape, arg_))
    return self.pad(padding, value=value).shrink(tuple((l + pl, r + pl) for (l,r),(pl,_) in zip(arg_, padding)))

  def gather(self:Tensor, idx:Tensor, dim:int) -> Tensor:
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"
    if dim < 0: dim += self.ndim
    idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(self.shape[dim], requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,self.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)  # noqa: E501

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    if dim < 0: dim += self.ndim
    assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    cat_dims = [s.shape[dim] for s in catargs]
    cat_dim_cumsum = [0, *itertools.accumulate(cat_dims)]
    slc:List[List[Optional[Tuple[sint, sint]]]] = [[None for _ in self.shape] for _ in catargs]
    for d,k,s in zip(cat_dims, cat_dim_cumsum[:-1], slc): s[dim] = (k, cat_dim_cumsum[-1] - k - d)
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])

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

  def split(self, sizes:Union[int, List[int]], dim:int=0) -> Tuple[Tensor, ...]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = dim + self.ndim if dim < 0 else dim
    if isinstance(sizes, int): return tuple(self.chunk(math.ceil(self.shape[dim]/sizes)))
    return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])

  def chunk(self, num:int, dim:int=0) -> List[Tensor]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim, step = dim + self.ndim if dim < 0 else dim, math.ceil(self.shape[dim]/num)
    slice_params = [[slice(None)]*dim + [slice(k, k + step)] for k in range(0, self.shape[dim], step)]
    return [self[tuple(sl)] for sl in slice_params]

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    if self.ndim == 0 and dim in [-1, 0]: return self  # this is to match torch behavior
    if not -self.ndim <= dim <= self.ndim-1: raise IndexError(f"{dim=} out of range {[-self.ndim, self.ndim-1] if self.ndim else [-1, 0]}")
    if dim < 0: dim += self.ndim
    return self if self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1:])

  def unsqueeze(self, dim:int) -> Tensor:
    if dim < 0: dim = self.ndim + dim + 1
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
    start_dim, end_dim = start_dim + self.ndim if start_dim < 0 else start_dim, end_dim + self.ndim if end_dim < 0 else end_dim
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])
  def unflatten(self, dim:int, sizes:Tuple[int,...]):
    if dim < 0: dim += self.ndim
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim+1:])

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
    axis_: List[int] = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    axis_ = [x if x >= 0 else x+len(self.shape) for x in axis_]
    shape = tuple(s for i,s in enumerate(self.shape) if i not in axis_)
    if 0 in self.shape and 0 not in shape:
      return Tensor.full(tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape, {mlops.Sum: 0.0, mlops.Max: -float("inf")}[fxn])
    ret = fxn.apply(self, new_shape=tuple([1 if i in axis_ else s for i,s in enumerate(self.shape)]))
    return ret if keepdim else ret.reshape(shape=shape)

  def sum(self, axis=None, keepdim=False):
    acc_dtype = least_upper_dtype(self.dtype, dtypes.uint) if dtypes.is_unsigned(self.dtype) else \
                least_upper_dtype(self.dtype, dtypes.int) if (dtypes.is_int(self.dtype) or self.dtype==dtypes.bool) else \
                least_upper_dtype(self.dtype, dtypes.float)
    # cast back to float16 or bfloat16 to match torch / jax behavior, but we use float for acc
    output_dtype = self.dtype if self.dtype in (dtypes.float16, dtypes.bfloat16) else acc_dtype
    return self.cast(acc_dtype)._reduce(mlops.Sum, axis, keepdim).cast(output_dtype)

  def max(self, axis=None, keepdim=False): return self._reduce(mlops.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    assert all_int(self.shape), "does not support symbolic shape"
    out = self.sum(axis=axis, keepdim=keepdim)
    return out.mul(prod(out.shape)/prod(self.shape)) if 0 not in self.shape else out
  def std(self, axis=None, keepdim=False, correction=1):
    assert all_int(self.shape), "does not support symbolic shape"
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(prod(self.shape)/prod(square_sum.shape)-correction).sqrt()
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

  def argmax(self, axis=None, keepdim=False):
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape)
      return (prod(self.shape) - idx.max() - 1).cast(dtypes.default_int)
    axis = axis + len(self.shape) if axis < 0 else axis
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1).cast(dtypes.default_int)
  def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)

  @staticmethod
  def einsum(formula:str, *raw_xs) -> Tensor:
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

    rhs_order, rhs_letters = tuple(zip(*sorted(enumerate(output), key=lambda e:e[1]))) or ([], [])
    # sum over all axes that's not in the output, then permute to the output order
    return reduce(lambda a,b:a*b, xs_).sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in rhs_letters]).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = [(0,x) for x in self.shape[0:-len(k_)]], self.shape[0:-len(k_)], self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      e_ = [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)]  # expands such that we don't need padding
      xup = self.reshape(*prefix, *flatten((1,i) for i in i_)).expand(*prefix, *flatten((e,i) for e,i in zip(e_, i_))).reshape(*prefix, *[e*i for e,i in zip(e_, i_)])  # noqa: E501
      # slide by dilation
      xup = xup.slice(slc_prefix + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])
      xup = xup.reshape(*prefix, *flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      xup = xup.slice(slc_prefix + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))
      # handle stride, and permute to move reduce to the end
      xup = xup.reshape(*prefix, *flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.slice(slc_prefix + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))
      xup = xup.reshape(*prefix, *flatten((k,o) for k,o in zip(k_, o_)))
      return xup.permute(*range(len(prefix)), *[len(prefix)+i*2+1 for i in range(len(k_))], *[len(prefix)+i*2 for i in range(len(k_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = self.slice(slc_prefix + [(0,o*s) for o,s in zip(o_, s_)])
    xup = xup.reshape(*prefix, *flatten(((o, s) for o,s in zip(o_, s_))))
    xup = xup.slice(slc_prefix + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(prefix)), *[len(prefix)+i*2 for i in range(len(k_))], *[len(prefix)+i*2+1 for i in range(len(k_))])

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))  # noqa: E501
  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))  # noqa: E501

  def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape)+1))
    x, w = self, weight.unflatten(0, (groups, -1)).permute(0,2,1,*trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s>1 for s in stride):
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-p,(k-1)*d-p+op) for k,d,p,op in reversed(list(zip(HW, make_pair(dilation, len(HW)), make_pair(padding, len(HW)), make_pair(output_padding, len(HW)))))))  # noqa: E501
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  wino = getenv("WINO", 0)
  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0) -> Tensor:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])  # noqa: E501

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not Tensor.wino:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
    def apply_matrix(mat, t, dim=0): return t if dim == len(HW) else Tensor.stack([apply_matrix(mat, sum(mm*t[j] for j,mm in enumerate(m) if mm), dim=dim+1) for m in mat])  # noqa: E501
    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    d = self.pad2d(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW))).contiguous_backward()
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = apply_matrix(winograd_G, g).contiguous().reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = apply_matrix(winograd_Bt, d).contiguous().reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = apply_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW)))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def dot(self, w:Tensor) -> Tensor:
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"  # noqa: E501
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(-1).cast(least_upper_dtype(x.dtype, w.dtype))

  def matmul(self, x:Tensor, reverse=False) -> Tensor: return x.dot(self) if reverse else self.dot(x)

  def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor:
    return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
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
    return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r,c) <= Tensor.arange(-k, c-k, **kwargs).unsqueeze(0).expand(r,c)
  def triu(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k, device=self.device).where(self, 0)
  def tril(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k+1, device=self.device).where(0, self)

  # ***** mlops (unary) *****

  def neg(self): return mlops.Neg.apply(self)
  def logical_not(self): return self.neg() if self.dtype == dtypes.bool else (1.0-self)
  def contiguous(self): return mlops.Contiguous.apply(self)
  def contiguous_backward(self): return mlops.ContiguousBackward.apply(self)
  def log(self): return mlops.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self): return self.log()/math.log(2)
  def exp(self): return mlops.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self): return mlops.Exp.apply(self*math.log(2))
  def relu(self): return mlops.Relu.apply(self)
  def sigmoid(self): return mlops.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def sin(self): return mlops.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def sqrt(self): return mlops.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self): return self.reciprocal().sqrt()
  def cos(self): return ((math.pi/2)-self).sin()
  def tan(self): return self.sin() / self.cos()

  # ***** math functions (unary) *****

  def trunc(self: Tensor) -> Tensor: return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor: return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor: return (self < (b := self.trunc())).where(b-1, b)

  def square(self): return self*self
  def clip(self, min_, max_): return self.maximum(min_).minimum(max_)
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return ((self.float()) / (self.float().abs() + 1e-12)).cast(self.dtype)
  def reciprocal(self): return 1.0/self

  # ***** activation functions (unary) *****

  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def celu(self, alpha=1.0): return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
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

  def _broadcasted(self, y:Union[Tensor, Scalar], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      if 0 in self.shape: return self, self.full_like(y)
      if isinstance(self.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      else: y_dtype = dtypes.from_py(y)
      y = Tensor(y, self.device, y_dtype, requires_grad=False)

    if match_dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # left pad shape with 1s
    if len(y.shape) < len(x.shape): y = y.reshape((1,) * (len(x.shape) - len(y.shape)) + y.shape)
    elif len(x.shape) < len(y.shape): x = x.reshape((1,) * (len(y.shape) - len(x.shape)) + x.shape)

    broadcasted_shape = tuple(0 if xi==0 or yi==0 else max(xi, yi) for xi, yi in zip(x.shape, y.shape))
    return x.expand(broadcasted_shape), y.expand(broadcasted_shape)

  def _to_const_val(self, x:Union[Tensor, Scalar]) -> Union[Tensor, Scalar]:
    # TODO: update with multi
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_contiguous_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    return mlops.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self
  def sub(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    return mlops.Sub.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-self if reverse else self)
  def mul(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if x.__class__ is not Tensor and x == 0.0: return mlops.Zero.apply(self)
    if x.__class__ is not Tensor and x == -1.0: return -self
    return mlops.Mul.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self
  def div(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    return mlops.Div.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(self.dtype) else self.mul(1/x)  # noqa: E501
  def pow(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x in [3,2,1,0]: return reduce(lambda acc,_: acc * self, range(int(x)), mlops.Zero.apply(self)+1)
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
    to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else int(bool(x - int(x))) if not reverse else ((self - self.trunc()) * 1e10).abs().clip(0, 1)) * base_sign  # noqa: E501
    inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)
  def xor(self, x:Tensor, reverse=False) -> Tensor: return mlops.Xor.apply(*self._broadcasted(x, reverse))

  # TODO: this implicitly changes dtype with /2
  def maximum(self, x:Union[Tensor, Scalar]) -> Tensor: return (self<x).detach().where(x, (self>x).detach().where(self, (self+x)/2))
  def minimum(self, x:Union[Tensor, Scalar]) -> Tensor: return -((-self).maximum(-x))

  def where(self:Tensor, input_:Union[Tensor, Scalar], other:Union[Tensor, Scalar]):
    x_,y = self._broadcasted(input_, match_dtype=False)
    x,z = x_._broadcasted(other, match_dtype=False)
    return mlops.Where.apply(x.cast(dtypes.bool), *y._broadcasted(z))

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

  def __lt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, True))
  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __eq__(self, x) -> Tensor: return mlops.Eq.apply(*self._broadcasted(x, True))   # type: ignore[override]
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()                       # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor) -> Tensor:
    x = (self - mean.reshape(shape=[1, -1, 1, 1]))
    if weight: x = x * weight.reshape(shape=[1, -1, 1, 1])
    ret = x.mul(invstd.reshape(shape=[1, -1, 1, 1]) if len(invstd.shape) == 1 else invstd)
    return (ret + bias.reshape(shape=[1, -1, 1, 1])) if bias else ret

  def dropout(self, p=0.5) -> Tensor:
    if not Tensor.training or p == 0: return self
    return self * (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p) * (1/(1.0 - p))

  def one_hot(self, num_classes:int, **kwargs) -> Tensor: return Tensor.where(self[..., None] == Tensor.arange(num_classes), 1, 0, **kwargs)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:  # noqa: E501
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

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index=-1) -> Tensor:
    # NOTE: self is a logits input
    loss_mask = (Y != ignore_index)
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

  # ***** cast ops *****

  def cast(self, dtype:DType) -> Tensor:
    if self.dtype == dtype: return self
    # hack for devices that don't support bfloat16
    if self.dtype == dtypes.bfloat16: return self.bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)
    return mlops.Cast.apply(self, dtype=dtype)
  def bitcast(self, dtype:DType) -> Tensor:
    assert self.dtype.itemsize == dtype.itemsize, "can't bitcast mismatched dtype itemsizes"
    return mlops.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
  def float(self) -> Tensor: return self.cast(dtypes.float32)
  def half(self) -> Tensor: return self.cast(dtypes.float16)

  # ***** convenience stuff *****

  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> sint: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
  def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)

# register functions to move between devices
for device in Device._devices: setattr(Tensor, f"{device.lower()}", partialmethod(Tensor.to, device))

if IMAGE:
  # if IMAGE>0 we install these replacement functions in Tensor (hack!)
  from tinygrad.features.image import image_conv2d, image_dot
  setattr(Tensor, "conv2d", image_conv2d)
  setattr(Tensor, "dot", image_dot)

# TODO: remove the custom op and replace with threefry
def custom_random(out:Buffer):
  Tensor._seed += 1
  if DEBUG >= 2: print(f"*** {out.device}   rand  seed {Tensor._seed} size {out.size:<15d} dtype {out.dtype}")
  rng = np.random.default_rng(Tensor._seed)
  rng_np_buffer = rng.random(size=out.size, dtype=np.float32).astype(dtype=out.dtype.np, copy=False)
  out.copyin(rng_np_buffer.data)

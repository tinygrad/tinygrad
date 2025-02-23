from tinygrad import Tensor, dtypes
import torch, contextlib
from torch.utils._python_dispatch import TorchDispatchMode
import numpy as np, os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
}

torch_into_uint = {
  torch.int32: dtypes.uint32,
}

def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if dtype is not None: return TTensor(Tensor.empty(*size, dtype=torch_to_tiny_dtype[dtype]))
  return TTensor(Tensor.empty(*size))

def to_copy_impl(input, dtype=None, device=None, layout=None, non_blocking=False, **kwargs):
  input_data = input.tiny.numpy()
  if dtype is not None: return TTensor(Tensor(input_data, dtype=torch_to_tiny_dtype[dtype]))
  return TTensor(Tensor(input_data))

def detach_impl(input):
  input_data = input.tiny.numpy()
  new_tensor = TTensor(Tensor(input_data))
  new_tensor.requires_grad = False
  return new_tensor

def uniform_impl(input, from_=0.0, to=1.0):
  random_values = torch.rand(input.shape, dtype=input.dtype, device=input.device)
  uniform_values = from_ + (to - from_) * random_values
  input.copy_(uniform_values)
  return input

def copy_impl(self, src):
  if type(src) == torch.Tensor: 
    src_data = src.numpy()
  elif type(src) == TTensor: 
    src_data = src.tiny.numpy()
  else:
    # TODO: Why? type inconsistency when tensors are initiated differently
    print(type(src))
  # src_data = src.numpy()
  self.tiny.numpy()[:] = src_data
  return self

def zeros_impl(size, dtype=None, layout=None, device=None, pin_memory=False):
  if dtype is None: return TTensor(Tensor.zeros(size), dtype=torch.float32)
  return TTensor(Tensor.zeros(size), dtype=dtype)

def ones_impl(size, dtype=None, layout=None, device=None, pin_memory=False):
  if dtype is None: return TTensor(Tensor.ones(size), dtype=torch.float32)
  return TTensor(Tensor.ones(size), dtype=dtype)

def zero_impl(input):
  if input.dim() == 0:
    input.tiny.numpy()[()] = 0
  else:
    input.tiny.numpy()[:] = 0
  return TTensor(input)

def fill_scalar_impl(input, value):
  input.tiny.numpy()[:] = value
  return TTensor(input)

tiny_backend = {
  "aten.add.Tensor": lambda x,y: TTensor(x.tiny + y.tiny),
  "aten.add_.Tensor": lambda x,y: x.copy_(x.tiny + y.tiny),
  "aten.sub.Tensor": lambda x,y: TTensor(x.tiny - y.tiny),
  "aten.sub_.Tensor": lambda x,y: x.copy_(x.tiny - y.tiny),
  "aten.div.Tensor": lambda x,y: TTensor(x.tiny / y.tiny),
  "aten.div_.Tensor": lambda x,y: x.copy_(x.tiny / y.tiny),
  "aten.gt.Tensor": lambda x,y: TTensor(x.tiny > y.tiny),
  "aten.ge.Tensor": lambda x,y: TTensor(x.tiny >= y.tiny),
  "aten.lt.Tensor": lambda x,y: TTensor(x.tiny < y.tiny),
  "aten.le.Tensor": lambda x,y: TTensor(x.tiny <= y.tiny),
  "aten.ne.Tensor": lambda x,y: TTensor(x.tiny != y.tiny),
  "aten.sum.default": lambda x: TTensor(x.tiny.sum()),
  "aten.mean.default": lambda x: TTensor(x.tiny.mean()),
  "aten.empty.memory_format": empty_memory_format,
  "aten.view.default": lambda x,sz: TTensor(x.tiny.reshape(sz)),
  "aten.abs.default": lambda x: TTensor(x.tiny.abs()),
  "aten.eq.Tensor": lambda x,y: TTensor(x.tiny == y.tiny),
  "aten.bitwise_and.Tensor": lambda x,y: TTensor(x.tiny & y.tiny),
  "aten.bitwise_and_.Tensor": lambda x,y: x.copy_(x.tiny & y.tiny),
  "aten.bitwise_or.Tensor": lambda x,y: TTensor(x.tiny | y.tiny),
  "aten.bitwise_or_.Tensor": lambda x,y: x.copy_(x.tiny | y.tiny),
  "aten.bitwise_xor.Tensor": lambda x, y: TTensor(x.tiny ^ y.tiny),
  "aten.bitwise_xor_.Tensor": lambda x, y: x.copy_(x.tiny ^ y.tiny),
  "aten.ne.Scalar": lambda x,y: TTensor(x.tiny != y),
  "aten.mul.Tensor": lambda x,y: TTensor(x.tiny * y.tiny),
  "aten.mul_.Tensor": lambda x, y: x.copy_(TTensor(x.tiny * y.tiny)),
  "aten.masked_select.default": lambda x,y: TTensor(Tensor(x.tiny.numpy()[y.tiny.numpy()])),
  "aten.lift_fresh.default": lambda x: TTensor(x, dtype=x.dtype),
  "aten._to_copy.default": to_copy_impl,
  "aten.detach.default": detach_impl,
  "aten.uniform_.default": uniform_impl,
  "aten.copy_.default": copy_impl,
  "aten.zeros.default": zeros_impl,
  "aten.ones.default": ones_impl,
  "aten.zero_.default": zero_impl,
  "aten.fill_.Scalar": fill_scalar_impl,
  "aten.neg.default": lambda x: TTensor(-x.tiny),
  "aten._local_scalar_dense.default": lambda x: TTensor(x.tiny.numpy(), dtype=x.dtype).item(),
  "aten.item.default": lambda x: x.tiny.item() if isinstance(x.tiny, (np.ndarray)) else TTensor(x.tiny.numpy(), dtype=x.dtype).item(),
  "aten.invert": lambda x: TTensor(~x.tiny),
  "aten.bitwise_not.default": lambda x: TTensor(x.tiny.bitwise_not()),
  "aten.pow_.Tensor": lambda x, y: x.copy_(TTensor(x.tiny.pow(y.tiny))),
  "aten.pow.Tensor_Tensor": lambda x, y: TTensor(x.tiny.pow(y.tiny)),
  "aten.dot.default": lambda x,w: TTensor(x.tiny.dot(w.tiny)),
  "aten.floor_divide.default": lambda x,y: TTensor(x.tiny // y.tiny),
  "aten.floor_divide_.Tensor": lambda x, y: x.copy_(x.tiny // y.tiny),
  "aten.remainder.Tensor": lambda x,y: TTensor(x.tiny.mod(y.tiny)),
  "aten.remainder_.Tensor": lambda x,y: x.copy_(x.tiny.mod(y.tiny)),
  "aten.__lshift__.Scalar": lambda x,y: TTensor(x.tiny.cast(torch_into_uint[x.dtype]) >> y),
  "aten.__ilshift__.Scalar": lambda x, y: x.copy_(x.tiny.cast(torch_into_uint[x.dtype]) >> y),
  "aten.__rshift__.Scalar": lambda x,y: TTensor(x.tiny.cast(torch_into_uint[x.dtype]) << y),
  "aten.__irshift__.Scalar": lambda x, y: x.copy_(x.tiny.cast(torch_into_uint[x.dtype]) << y),
  "aten.select.int": lambda x, dim, index: TTensor(x.tiny.interpolate(dim, index)),
}

class TTensor(torch.Tensor):
  tiny: Tensor
  context = contextlib.nullcontext

  @staticmethod
  def __new__(cls, tiny: Tensor, *args, **kwargs):
    dtype = kwargs.get('dtype', None)
    if dtype is None and len(args) > 0: dtype = args[0]
    if dtype is None: dtype = torch.float32
    out = torch.Tensor._make_wrapper_subclass(cls, tiny.shape, dtype=dtype)
    torch._C._set_throw_on_mutable_data_ptr(out)
    # TODO: why? torch.tensor gives Tinygrad tensor, but torch.ones/zeros gives torch tensor
    if type(tiny) == torch.Tensor: 
      out.tiny = tiny.to(dtype)
    elif type(tiny) == Tensor: 
      out.tiny = tiny.cast(torch_to_tiny_dtype[dtype])
    else:
      out.tiny = tiny
    return out
  def __repr__(self): return super().__repr__(tensor_contents=f"{self.tiny}")
  def __torch_dispatch__(cls, func, types, args, kwargs=None):
    print(f"Dispatch Log: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
    new_func = tiny_backend.get(str(func), None)
    if new_func is None: raise NotImplementedError(f"add support for {func}")
    return new_func(*args, **(kwargs or {}))

class Dispatcher(TorchDispatchMode): __torch_dispatch__ = TTensor.__torch_dispatch__
Dispatcher().__enter__()

if __name__ == "__main__":
  a = torch.tensor([1,2,3], dtype=torch.int32)
  b = torch.ones((4,), dtype=torch.int)
  # print(a)
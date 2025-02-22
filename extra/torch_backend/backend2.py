from tinygrad import Tensor, dtypes
import torch, contextlib
from torch.utils._python_dispatch import TorchDispatchMode

torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
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
    src_data = src.numpy()  
    self.tiny.numpy()[:] = src_data  
    return self 

tiny_backend = {
  "aten.empty.memory_format": empty_memory_format,
  "aten.view.default": lambda x,sz: TTensor(x.tiny.reshape(sz)),
  "aten.abs.default": lambda x: TTensor(x.tiny.abs()),
  "aten.eq.Tensor": lambda x,y: TTensor(x.tiny == y.tiny),
  "aten.bitwise_and.Tensor": lambda x,y: TTensor(x.tiny & y.tiny),
  "aten.ne.Scalar": lambda x,y: TTensor(x.tiny != y),
  "aten.mul.Tensor": lambda x,y: TTensor(x.tiny * y.tiny),
  "aten.masked_select.default": lambda x,y: TTensor(Tensor(x.tiny.numpy()[y.tiny.numpy()])),
  "aten.lift_fresh.default": lambda x: TTensor(x),
  "aten._to_copy.default": to_copy_impl,
  "aten.detach.default": detach_impl,
  "aten.uniform_.default": uniform_impl,
  "aten.copy_.default": copy_impl,
}

class TTensor(torch.Tensor):
  tiny: Tensor
  context = contextlib.nullcontext

  @staticmethod
  def __new__(cls, tiny, *args, **kwargs):
    out = torch.Tensor._make_wrapper_subclass(cls, tiny.shape)
    torch._C._set_throw_on_mutable_data_ptr(out)
    out.tiny = tiny
    return out
  def __repr__(self): return super().__repr__(tensor_contents=f"{self.tiny}")
  def __torch_dispatch__(cls, func, types, args, kwargs=None):
    print(f"Dispatch Log: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
    #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
    new_func = tiny_backend.get(str(func), None)
    if new_func is None: raise NotImplementedError(f"add support for {func}")
    return new_func(*args, **(kwargs or {}))

class Dispatcher(TorchDispatchMode): __torch_dispatch__ = TTensor.__torch_dispatch__
Dispatcher().__enter__()

if __name__ == "__main__":
  a = torch.empty((4,), dtype=torch.int)

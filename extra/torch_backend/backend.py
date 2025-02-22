from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG, getenv
import torch, pathlib

# https://pytorch.org/docs/stable/torch.compiler_ir.html

# TODO: don't replicate this in cpp
torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.uint8: dtypes.uint8,
  torch.int8: dtypes.int8,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
}

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
def wrap(x:Tensor) -> torch.Tensor: return mod.wrap(x)
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend: pass
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend)
torch.utils.generate_methods_for_privateuse1_backend()

@torch.library.impl("aten::zero_", "privateuseone")
def zero_(x):
  tt = unwrap(x)
  tt.replace(tt.zeros_like())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
def fill_scalar(x, y):
  tt = unwrap(x)
  tt.replace(tt.full_like(y))

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()

@torch.library.impl("aten::masked_select", "privateuseone")
def masked_select(self, mask):
  # err, bad
  return wrap(Tensor(self.cpu().numpy()[mask.cpu().numpy()]))

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor, size, stride, storage_offset=None):
  if size == [] and storage_offset is not None:
    # TODO: is this right?
    return wrap(unwrap(tensor).flatten()[storage_offset:storage_offset+1].reshape(()))
  print(tensor.shape, size, stride, storage_offset, "NOTE: this as_strided is wrong")
  return wrap(Tensor.zeros(*size))
  raise NotImplementedError("fix as_strided")

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout, device, pin_memory):
  if DEBUG >= 2: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=torch_to_tiny_dtype[dtype])
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if DEBUG >= 2: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=torch_to_tiny_dtype[dtype])
  return wrap(ret)

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  #print(input, weight, bias)
  print(f"{input.shape=} {weight.shape=} {bias.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  return wrap(unwrap(input).conv2d(unwrap(weight), unwrap(bias), groups=groups, stride=stride, dilation=dilation, padding=padding))
  #raise NotImplementedError("need convolution")

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src, dest):
  if str(src.device) == "tiny" and str(dest.device) == "tiny":
    unwrap(dest).replace(unwrap(src), allow_shape_mismatch=True)
  elif str(src.device) == "tiny" and str(dest.device) == "cpu":
    dest[:] = torch.from_numpy(unwrap(src).numpy())
  elif str(src.device) == "cpu" and str(dest.device) == "tiny":
    unwrap(dest).assign(Tensor(src.numpy()))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
def cat_out(tensors, out, dim=0): unwrap(out).replace(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim), allow_shape_mismatch=True)

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y): return wrap(unwrap(x)[y[0].tolist()])

tiny_backend = {
  "aten.bitwise_and.Tensor": lambda x,y: wrap(unwrap(x) & unwrap(y)),
  "aten.add.Tensor": lambda x,y: wrap(unwrap(x) + unwrap(y)),
  "aten.mul.Tensor": lambda x,y: wrap(unwrap(x) * unwrap(y)),
  "aten.div.Tensor": lambda x,y: wrap(unwrap(x) / unwrap(y)),
  "aten.eq.Tensor": lambda x,y: wrap(unwrap(x) == unwrap(y)),
  "aten.ne.Tensor": lambda x,y: wrap(unwrap(x) != unwrap(y)),
  "aten.eq.Scalar": lambda x,y: wrap(unwrap(x) == y),
  "aten.ne.Scalar": lambda x,y: wrap(unwrap(x) != y),
  "aten.lt.Scalar": lambda x,y: wrap(unwrap(x) < y),
  "aten.gt.Scalar": lambda x,y: wrap(unwrap(x) > y),
  "aten.add.out": lambda x,y,out: wrap(unwrap(out).replace(unwrap(x) + y, allow_shape_mismatch=True)), # unwrapping y?
  "aten.abs.out": lambda x,out: wrap(unwrap(out).replace(unwrap(x).abs(), allow_shape_mismatch=True)),
  "aten.exp2.out": lambda x,out: wrap(unwrap(out).replace(unwrap(x).exp2(), allow_shape_mismatch=True)),
  "aten.ceil.out": lambda x,out: wrap(unwrap(out).replace(unwrap(x).ceil(), allow_shape_mismatch=True)),
  "aten.view": lambda x,sz: wrap(unwrap(x).reshape(sz)),
  "aten.min": lambda x: wrap(unwrap(x).min()),
  "aten.max": lambda x: wrap(unwrap(x).max()),
  "aten.relu": lambda x: wrap(unwrap(x).relu()),
}

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(v)

if getenv("TORCH_DEBUG"):
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  DispatchLog().__enter__()

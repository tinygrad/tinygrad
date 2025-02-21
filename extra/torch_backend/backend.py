from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG
import torch, pathlib

# TODO: don't replicate this in cpp
torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
}

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[pathlib.Path(__file__).parent / "wrapped_tensor.cpp"])
wrap, unwrap = mod.wrap, mod.unwrap
class TinyBackend: pass
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend)
torch.utils.generate_methods_for_privateuse1_backend()

@torch.library.impl("aten::view", "privateuseone")
def view(x, sz): return mod.wrap(mod.unwrap(x).reshape(sz))

@torch.library.impl("aten::min", "privateuseone")
def min(x): return mod.wrap(mod.unwrap(x).min())

@torch.library.impl("aten::max", "privateuseone")
def max(x): return mod.wrap(mod.unwrap(x).max())

@torch.library.impl("aten::zero_", "privateuseone")
def zero_(x):
  tt = mod.unwrap(x)
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
  print(tensor.shape, size, stride, storage_offset)
  raise NotImplementedError

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
  print(input, weight, bias)
  raise NotImplementedError

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

@torch.library.impl("aten::exp2.out", "privateuseone")
def exp2_out(x, out): unwrap(out).replace(unwrap(x).exp2(), allow_shape_mismatch=True)

@torch.library.impl("aten::ceil.out", "privateuseone")
def ceil_out(x, out): unwrap(out).replace(unwrap(x).ceil(), allow_shape_mismatch=True)

@torch.library.impl("aten::abs.out", "privateuseone")
def abs_out(x, out): unwrap(out).replace(unwrap(x).abs(), allow_shape_mismatch=True)

@torch.library.impl("aten::bitwise_and.Tensor", "privateuseone")
def bitwise_and_tensor(x, y): return wrap(unwrap(x) & unwrap(y))

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(x, y): return wrap(unwrap(x) + unwrap(y))

@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul_tensor(x, y): return wrap(unwrap(x) * unwrap(y))

@torch.library.impl("aten::div.Tensor", "privateuseone")
def div_tensor(x, y): return wrap(unwrap(x) / unwrap(y))

@torch.library.impl("aten::eq.Tensor", "privateuseone")
def eq_tensor(x, y): return wrap(unwrap(x).eq(unwrap(y)))

@torch.library.impl("aten::ne.Tensor", "privateuseone")
def ne_tensor(x, y): return wrap(unwrap(x).ne(unwrap(y)))

@torch.library.impl("aten::ne.Scalar", "privateuseone")
def ne_scalar(x, y): return wrap(unwrap(x).ne(y))

@torch.library.impl("aten::gt.Scalar", "privateuseone")
def gt_scalar(x, y): return wrap(unwrap(x) > y)

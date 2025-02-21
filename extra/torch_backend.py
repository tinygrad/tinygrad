from tinygrad import Tensor
import torch

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(
  name="custom_device_extension",
  sources=["extra/torch_backend.cpp"],
  extra_include_paths=["cpp_extensions"],
  extra_cflags=["-g"],
  verbose=True)

@torch.library.impl("aten::zero_", "privateuseone")
def zero_(x):
  tt = mod.unwrap(x)
  tt.replace(tt.zeros_like())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
def fill_scalar(x, y):
  tt = mod.unwrap(x)
  tt.replace(tt.full_like(y))

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  ret = Tensor.empty(*size)
  return mod.wrap(ret)

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src, dest):
  dest[:] = torch.from_numpy(mod.unwrap(src).numpy())

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(x, y):
  return mod.wrap(mod.unwrap(x) + mod.unwrap(y))

# Module used for our backend
class _OpenRegMod: pass

if __name__ == "__main__":
  torch.utils.rename_privateuse1_backend("tiny")
  torch._register_device_module("tiny", _OpenRegMod())
  x = torch.ones(4, device="tiny")
  y = torch.ones(4, device="tiny")
  z = x+y
  print(z.cpu().numpy())

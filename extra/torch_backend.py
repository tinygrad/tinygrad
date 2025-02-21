from typing import cast
from tinygrad import Tensor
import traceback

import torch

#import pytorch_openreg._C  # noqa: F401  # usort: skip

import torch.utils.cpp_extension
module = torch.utils.cpp_extension.load(
  name="custom_device_extension",
  sources=["extra/torch_backend.cpp"],
  extra_include_paths=["cpp_extensions"],
  extra_cflags=["-g"],
  verbose=True)

# Module used for our backend
class _OpenRegMod: pass

if __name__ == "__main__":
  torch.utils.rename_privateuse1_backend("openreg")
  torch._register_device_module("openreg", _OpenRegMod())
  x = torch.tensor([[1], [2], [3]], device="openreg")
  #import pytorch_openreg
  #print(x)

exit(0)

#print(torch.__config__.show())

def generate_faked_module():
    class _OpenRegMod:
        pass

    return _OpenRegMod()


def generate_faked_module_methods():
    def device_count() -> int:
        return 1

    def get_rng_state(device= "openreg") -> torch.Tensor:
        # create a tensor using our custom device object.
        return torch.empty(4, 4, device="openreg")

    def set_rng_state(new_state: torch.Tensor, device = "openreg") -> None:
        pass

    def is_available():
        return True

    def current_device():
        return 0

    torch.openreg.device_count = device_count
    torch.openreg.get_rng_state = get_rng_state
    torch.openreg.set_rng_state = set_rng_state
    torch.openreg.is_available = is_available
    torch.openreg.current_device = current_device
    torch.openreg._lazy_init = lambda: None
    torch.openreg.is_initialized = lambda: True

module = torch.utils.cpp_extension.load(
  name="custom_device_extension",
  sources=[
      "/Users/light/build/pytorch/test/cpp_extensions/open_registration_extension.cpp",
  ],
  extra_include_paths=["cpp_extensions"],
  extra_cflags=["-g"],
  verbose=True)

#torch.utils.rename_privateuse1_backend("openreg")
#torch._register_device_module('openreg', generate_faked_module())
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
generate_faked_module_methods()
device = module.custom_device()
print(device)

#x = torch.empty(4, 4, device=device)
tensor = torch.tensor([1, 2, 3], device='privateuseone')

exit(0)

class MyPrivateUseOne:
  def __repr__(self): return f"MyPrivateUseOne()"
  @staticmethod
  def is_available(): return True
  @staticmethod
  def device_count(): return 1
  @staticmethod
  def current_device(): return 0

# register device
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module('tiny', MyPrivateUseOne)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

#storage = torch.UntypedStorage(4, device='privateuseone')

from torch.testing._internal.common_utils import is_privateuse1_backend_available
print(is_privateuse1_backend_available())

tensor = torch.tensor([1, 2, 3], device='privateuseone')
#test_tensor = torch.empty(4, 4, device='privateuseone')

exit(0)


def fallback_kernel(op, *args, **kwargs):
  print("FALLBACK", op)
#lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
#lib.fallback(fallback_kernel)

# implement functions
#@torch.library.impl("aten::fill_.Scalar", "PrivateUse1")
#def fill_scalar():
#  print("fill scalar")

class PrivateUseOneTensor(torch.Tensor):
  @property
  def device(self): return 'privateuseone:0'

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add(x, y):
  print("HERE")

"""
class TinyGradDLpackWrapper:
  def __init__(self, tinygrad_tensor):
    self.tg_t = tinygrad_tensor

  def __dlpack_device__(self):
    return (12, 0)

  def __dlpack__(self, stream=None):
    print("DLPACK")
    return None

"""
@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  print("Custom PrivateUse1 empty called!", size, dtype, layout, device, pin_memory, memory_format)
  # TODO: return Tensor with device privateuseone:0
  ret = Tensor.empty(*size)
  #return torch.from_dlpack(TinyGradDLpackWrapper(ret))

  import array
  a = array.array('i', [1, 2, 3])
  t = torch.frombuffer(a, dtype=dtype)
  t = t.to(device)
  print(t, t.device)
  return t

  raise Exception
  t = torch.tensor([0], device="privateuseone")
  print(t)

  #t = torch.empty(size, device="privateuseone")
  #t = t.as_subclass(PrivateUseOneTensor)
  #print(type(t), t.device, device)
  #assert t.device == device, "THIS FAILS"
  return t


dd = torch.device('privateuseone')
print(dd)

a = torch.ones(4, 4, device='privateuseone')
b = torch.ones(4, 4, device='privateuseone')
c = a+b
print(c.cpu().numpy())

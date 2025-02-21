from typing import cast
from tinygrad import Tensor
import traceback
import torch

def fallback_kernel(op, *args, **kwargs):
  print("FALLBACK", op)

# register device
lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
#lib.fallback(fallback_kernel)
#torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module('privateuseone', lib)

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
  #ret = Tensor.empty(*size)
  #return torch.from_dlpack(TinyGradDLpackWrapper(ret))
  #t = torch.empty(size, dtype=dtype, layout=layout, device="PrivateUse1")
  #t = t.as_subclass(PrivateUseOneTensor)
  #print(type(t), t.device, device)
  #assert t.device == device, "THIS FAILS"
  #return t


a = torch.ones(4, 4, device='privateuseone')
b = torch.ones(4, 4, device='privateuseone')
c = a+b
print(c.cpu().numpy())

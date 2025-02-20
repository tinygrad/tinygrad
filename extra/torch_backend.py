from tinygrad import Tensor
import torch

# register device
aten = torch._ops.ops.aten
lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
def impl(x):
  def _wrap(y): lib.impl(x, y)
  return _wrap
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module('privateuseone', lib)


# implement functions
@impl(aten.empty.memory_format)
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  print("Custom PrivateUse1 empty called!")
  print(size)
  return torch.empty(size, dtype=dtype, layout=layout, device="meta")
  #return Tensor.empty(*size)

a = torch.ones(4, 4, device='tiny:0')
b = torch.ones(4, 4, device='tiny:0')
c = a+b
print(c.cpu().numpy())

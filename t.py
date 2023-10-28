import os
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from tinygrad.ops import Device

Device.DEFAULT = "CUDA"
os.environ["CUDACPU"] = "1"
a = Tensor([[1,2],[3,2]], dtype=dtypes.half)@Tensor.eye(2, dtype=dtypes.float)
print(a.numpy())

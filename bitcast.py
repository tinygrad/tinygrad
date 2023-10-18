from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

a = Tensor([1,2,3,4], dtype=dtypes.float32).bitcast(dtypes.uint32)
print(a.numpy())

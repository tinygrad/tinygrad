from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

# X86=1 python test-tg.py

x = Tensor([1.1,2.2,3.3,4.4,5.5,6.6], requires_grad=True, dtype=dtypes.float32)
y = Tensor([1.1,2.2,3.3,4.4,5.5,6.6], requires_grad=True, dtype=dtypes.float32)
z = x + y

print(z.numpy())
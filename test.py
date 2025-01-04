import tinygrad as tg
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True, dtype=tg.dtypes.float8, device="cuda")
y = Tensor([[2.0,0,-2.0]], requires_grad=True, dtype=tg.dtypes.float8, device="cuda")
z = y.matmul(x).sum()
z.backward()

print(x.half().tolist())
print(y.half().tolist())
# print(x.grad.half().tolist())  # dz/dx
# print(y.grad.half().tolist())  # dz/dy
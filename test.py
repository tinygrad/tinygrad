from tinygrad import Tensor, dtypes
from icecream import install
install()

tensor_size, idx_size = 50257, 324
mask = Tensor.rand(1, tensor_size) > 0.5
idx1 = Tensor.zeros(idx_size, dtype=dtypes.int)
idx2 = Tensor.randint(idx_size, low=0, high=tensor_size, dtype=dtypes.int)
ic(mask, idx1, idx1.numpy(), idx2, idx2.numpy())

out = mask[idx1, idx2]
ic(out.numpy())

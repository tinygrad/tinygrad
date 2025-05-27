from tinygrad import Tensor, dtypes

t = Tensor.rand(4096, 4096)
idx_size = 324
idx1 = Tensor.zeros(idx_size, dtype=dtypes.int)
idx2 = Tensor.randint(idx_size, dtype=dtypes.int)

out = t[idx1, idx2]
print(out.numpy())

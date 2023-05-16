from tinygrad.tensor import Tensor
N = 1024
a, b = Tensor.randn(N, N), Tensor.randn(N, N)
c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).sum(axis=2)
print((c.numpy() - (a.numpy() @ b.numpy())).mean())

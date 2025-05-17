from tinygrad import Tensor


t0, t1, t2 = Tensor([[[1, 2]]]), Tensor([[3, 4]]), Tensor([[5, 6]])
print(t0.cat(t1, t2, dim=0).numpy())

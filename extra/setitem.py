from tinygrad import Tensor
# import torch
# t = torch.arange(42).reshape(6, 7)
# # # a = torch.rand(12).reshape(6, 2)  # somehow this does not work in torch
# a = torch.arange(12).reshape(6, 2)
# # t[:, 2:4] = a
# t.as_strided((6, 2), (7, 1), 2)[:] = a
# print(t)

# # materialize a buffer
# import torch
# a = torch.zeros(10, 10)
# b = a.permute(1, 0)
# b[3] = 1

a = Tensor.zeros(10, 10)
b = a.permute(1, 0)
# b is not contiguous
assert not b.lazydata.st.contiguous
assert b[3].lazydata.base is a.lazydata.base
# TODO: before updating the base, go through the views and make sure it's okay?
base = b[3].lazydata.base
print(b.lazydata.realized)
b.realize()
print(b.lazydata.realized)

# correct behavior is to continguous a, which becomes the view of b

# t = Tensor.arange(42).reshape(6, 7).realize()
# a = Tensor.rand(6, 2).realize()

# assert t[:].lazydata is t.lazydata
# assert t[:, 2:4].lazydata.base is t.lazydata.base

# t[:, 2:4] = a

# t.realize()

# print(t.numpy())
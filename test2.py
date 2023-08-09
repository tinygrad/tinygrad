from tinygrad.tensor import Tensor

a = Tensor([1.0]).to("LLVM").half().to('CPU').realize()
print(a.numpy())

from tinygrad import Tensor

if __name__ == "__main__":
  for d in ["gpu:0", "gpu:1"]:
    a = Tensor.rand(10, 10, device=d)
    b = Tensor.rand(10, 10, device=d)
    print((a+b).numpy())

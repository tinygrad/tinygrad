from tinygrad.tensor import Tensor

if __name__ == "__main__":
  Tensor.empty((100, 100), device="disk:/tmp/dt1")


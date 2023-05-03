#!/usr/bin/env python3
import numpy as np
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  a0 = Tensor.randn(4096, device="gpu:0").realize()
  b1 = Tensor.randn(4096, device="gpu:1").realize()

  # allreduce
  a1 = a0.to('gpu:1').realize()
  b0 = b1.to('gpu:0').realize()

  # sum
  ab0 = a0 + b0
  ab1 = a1 + b1

  # same
  np.testing.assert_allclose(ab0.numpy(), ab1.numpy())

  # devices
  print(ab0)
  print(ab1)

#!/usr/bin/env python
import numpy as np
from examples.llama import Transformer, onehot_encode
from tinygrad.tensor import Tensor

VOCAB_SIZE = 4
args_test = {"dim": 2, "multiple_of": 1, "n_heads": 1, "n_layers": 1, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE}

if __name__ == "__main__":
  Tensor.no_grad = True

  Tensor.manual_seed(1337)
  model = Transformer(**args_test)

  print("run a")
  outa_0 = model(onehot_encode([1], VOCAB_SIZE), 0).numpy()
  print(outa_0)
  outa_1 = model(onehot_encode([3], VOCAB_SIZE), 1).numpy()
  print(outa_1)

  print("run b")
  outb_0 = model(onehot_encode([1], VOCAB_SIZE), 0, False).numpy()
  print(outb_0)
  outb_1 = model(onehot_encode([3], VOCAB_SIZE), 1, False).numpy()
  print(outb_1)

  print("run c")
  outc_0 = model(onehot_encode([1], VOCAB_SIZE), 0).numpy()
  print(outc_0)
  outc_1 = model(onehot_encode([3], VOCAB_SIZE), 1).numpy()
  print(outc_1)

  # a and c are the same
  np.testing.assert_allclose(outa_0, outc_0)
  np.testing.assert_allclose(outa_1, outc_1)

  # b and c should the same
  np.testing.assert_allclose(outb_0, outc_0)
  print("FAILS")
  np.testing.assert_allclose(outb_1, outc_1)


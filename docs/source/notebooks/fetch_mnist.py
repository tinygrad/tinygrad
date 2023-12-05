import os, gzip, tarfile, pickle
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch

def fetch_mnist(tensors=False):
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"   # http://yann.lecun.com/exdb/mnist/ lacks https
  X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:]
  X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:]
  if tensors: return Tensor(X_train).reshape(-1, 1, 28, 28), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
  else: return X_train, Y_train, X_test, Y_test
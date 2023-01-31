import os
import io
import numpy as np
import gzip
import tarfile
import pickle
from extra.utils import fetch

def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse(os.path.dirname(__file__)+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(os.path.dirname(__file__)+"/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(os.path.dirname(__file__)+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(os.path.dirname(__file__)+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test

def fetch_cifar(train=True):
  tt = tarfile.open(fileobj=io.BytesIO(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')), mode='r:gz')
  if train:
    # TODO: data_batch 2-5
    db = [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)]
  else:
    db = [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")]
  X = np.concatenate([x[b'data'].reshape((-1, 3, 32, 32)) for x in db], axis=0)
  Y = np.concatenate([np.array(x[b'labels']) for x in db], axis=0)
  return X, Y

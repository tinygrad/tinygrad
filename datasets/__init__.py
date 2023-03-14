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
  cifar10_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618], dtype=np.float32).reshape(1,3,1,1)
  cifar10_std = np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628], dtype=np.float32).reshape(1,3,1,1)
  tt = tarfile.open(fileobj=io.BytesIO(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')), mode='r:gz')
  if train:
    # TODO: data_batch 2-5
    db = [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)]
  else:
    db = [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")]
  X = np.concatenate([x[b'data'].reshape((-1, 3, 32, 32)) for x in db], axis=0)
  Y = np.concatenate([np.array(x[b'labels']) for x in db], axis=0)
  X = (X - cifar10_mean) / cifar10_std
  return X, Y

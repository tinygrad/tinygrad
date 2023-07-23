import os, random, gzip, tarfile, pickle
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from extra.utils import download_file

def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse(os.path.dirname(__file__)+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(os.path.dirname(__file__)+"/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(os.path.dirname(__file__)+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(os.path.dirname(__file__)+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

def fetch_cifar(shuffle=False):
  def _load_disk_tensor(sz, bs, db_list, path, shuffle=False):
    idx=0
    X, Y = None, None
    for db in db_list:
      x = db[b'data']
      y = np.array(db[b'labels'])
      order = list(range(0, len(y)))
      if shuffle: random.shuffle(order)
      if X is None:
        X = Tensor.empty(sz, *x.shape[1:], device=f'disk:/tmp/{path}'+'_x', dtype=dtypes.uint8)
        Y = Tensor.empty(sz, *y.shape[1:], device=f'disk:/tmp/{path}'+'_y', dtype=dtypes.int64)
      X[idx:idx+bs].assign(x[order,:])
      Y[idx:idx+bs].assign(y[order])
      idx += bs
    return X, Y  
  fn = os.path.dirname(__file__)+"/cifar-10-python.tar.gz"
  download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', fn)
  tt = tarfile.open(fn, mode='r:gz')
  db = [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)]
  X_train, Y_train = _load_disk_tensor(50000, 10000, db, "cifar_train", shuffle=shuffle)
  db = [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")]
  X_test, Y_test = _load_disk_tensor(10000, 10000, db, "cifar_test", shuffle=shuffle)
  return X_train, Y_train, X_test, Y_test
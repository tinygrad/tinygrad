import gzip, tarfile
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch

def _fetch_mnist(file, offset): return Tensor(gzip.open(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file)).read()[offset:])
def mnist():
  return _fetch_mnist("train-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("train-labels-idx1-ubyte.gz", 8), \
         _fetch_mnist("t10k-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("t10k-labels-idx1-ubyte.gz", 8)

def cifar():
  tt = tarfile.open(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'), mode='r:gz')
  train = Tensor.cat(*[Tensor(tt.extractfile(f"cifar-10-batches-bin/data_batch_{i}.bin").read()).reshape(-1, 3073) for i in range(1,6)])
  test = Tensor(tt.extractfile("cifar-10-batches-bin/test_batch.bin").read()).reshape(-1, 3073)
  return train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]

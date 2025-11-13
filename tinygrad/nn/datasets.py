import ast
from tinygrad.tensor import Tensor
from tinygrad.nn.state import tar_extract

def mnist(device=None, fashion=False):
  base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" if fashion else "https://storage.googleapis.com/cvdf-datasets/mnist/"
  def _mnist(file): return Tensor.from_url(base_url+file, gunzip=True)
  return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("train-labels-idx1-ubyte.gz")[8:].to(device), \
         _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("t10k-labels-idx1-ubyte.gz")[8:].to(device)

def cifar(device=None):
  tt = tar_extract(Tensor.from_url('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', gunzip=True))
  train = Tensor.cat(*[tt[f"cifar-10-batches-bin/data_batch_{i}.bin"].reshape(-1, 3073).to(device) for i in range(1,6)])
  test = tt["cifar-10-batches-bin/test_batch.bin"].reshape(-1, 3073).to(device)
  return train[:, 1:].reshape(-1,3,32,32), train[:, 0], test[:, 1:].reshape(-1,3,32,32), test[:, 0]

def imagenet_labels():
  return ast.literal_eval(Tensor.from_url(
    "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
  ).tobytes().decode())

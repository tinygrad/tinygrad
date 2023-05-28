from tinygrad.tensor import Tensor
from datasets.imagenet import iterate, get_val_files

if __name__ == "__main__":
  #sz = len(get_val_files())
  sz = 32*100
  X,Y = None, None

  idx = 0
  for x,y in iterate(shuffle=False):
    print(x.shape, y.shape)
    assert x.shape[0] == y.shape[0]
    bs = x.shape[0]
    if X is None:
      # TODO: need uint8 support
      X = Tensor.empty(sz, *x.shape[1:], device="disk:/tmp/imagenet_x")
      Y = Tensor.empty(sz, *y.shape[1:], device="disk:/tmp/imagenet_y")
      print(X.shape, Y.shape)
    X[idx:idx+bs].assign(x)
    Y[idx:idx+bs].assign(y)
    idx += bs
    if idx >= sz: break

import numpy as np
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Timing

d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
N = 4096
FLOPS = N*N*N*2

# LazyBuffer should make three fields lists: self.st (all must have the same shape), self.realized, and self.device

def explicit_shard_W_axis_1(X, W):
  Xs = [X.to(d0), X.to(d1)]
  Ws = [W[:, :N//2].to(d0), W[:, N//2:].to(d1)]   # TODO: these shouldn't make copies on the original device
  # TODO: it shouldn't be faster with these realize
  for x in Xs+Ws: x.realize()
  Os = [Xs[0] @ Ws[0], Xs[1] @ Ws[1]]
  for x in Os: x.realize()
  return Tensor.cat(*[x.to(Device.DEFAULT) for x in Os], dim=1)   # TODO: someday we can remove this copy too

def matmul(X, W):
  return explicit_shard_W_axis_1(X, W)
  #return X@W

if __name__ == "__main__":
  with Timing("init devices: "):
    Device[d0], Device[d1]

  with Timing("create tensors: "):
    X = Tensor.kaiming_uniform(N, N).realize()
    W = Tensor.kaiming_uniform(N, N).realize()

  with Timing("warmup: "):
    O = matmul(X, W).numpy()

  GlobalCounters.reset()
  print("******** multiply start")
  with Timing("******** multiply done: ", lambda x: f"  {FLOPS/x:.2f} GFLOPS"):
    O = matmul(X, W).realize()
    Device[Device.DEFAULT].synchronize()

  with Timing("testing: "):
    val = X.numpy() @ W.numpy()
    np.testing.assert_allclose(val, O.numpy(), atol=1e-5)

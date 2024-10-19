from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, colored, trange, prod
from tinygrad.nn.datasets import mnist
import os
from extra.fsdp.utils import print_size

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
def reset_mem_high():
  for gpu in GPUS:
    Device[gpu].allocator.reset_mem_high()

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 3, bias=False),
      Tensor.relu,
      nn.Conv2d(32, 64, 3, bias=False),
      Tensor.relu,
      Tensor.max_pool2d,
      lambda x: x.dropout(0.25),
      lambda x: x.flatten(1),
      nn.Linear(9216, 128, bias=False),
      Tensor.relu,
      lambda x: x.dropout(0.5),
      nn.Linear(128, 10, bias=False),
      lambda x: x.log_softmax(1),
  ]
    
  def __call__(self, x):
    return x.sequential(self.layers)
  

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist("CLANG")


  model = Model()
  opt = nn.optim.SGD(nn.state.get_parameters(model))
  print_size("model", *nn.state.get_parameters(model))
  print_size("model with optimizer", *nn.state.get_parameters(opt))
  if SHARD > 1:
    print("SHARDING ON", GPUS)
    X_test.shard_(GPUS)
    Y_test.shard_(GPUS)
    seen = set()
    for k, p in nn.state.get_state_dict(model).items():
      if p in seen: continue
      seen.add(p)
      axis = 0
      if prod(p.shape) <= 1:
        axis = 1
      elif k == "layers.0.weight":
        axis = None
      elif k == "layers.2.weight":
        axis = 1
      
      print(f"{k}, {axis=}")
      p.shard_(GPUS, axis)
    for k, p in nn.state.get_state_dict(opt).items():
      if p in seen: continue
      seen.add(p)
      p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
      p.realize()
  else:
    print("NO SHARD")
    for p in nn.state.get_parameters(opt):
      p.realize()

  @TinyJit
  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      samples = Tensor.randint(getenv("BS", 4), high=X_train.shape[0])
      Xt, Yt = X_train[samples], Y_train[samples]
      Xt.to_(GPUS[0])
      Yt.to_(GPUS[0])
      if SHARD > 1:
        Xt.shard_(GPUS)
        Yt.shard_(GPUS)
      Xt.realize()
      Yt.realize()
      reset_mem_high()
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt)
      # loss = loss.sparse_categorical_crossentropy(Yt)
      loss = loss.sum(0).sum(0)
      loss.backward()
      opt.step()
      return loss

  test_acc = float('nan')
  for t in range(1):
    loss = train_step()
    print("ITER", t)
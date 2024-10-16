from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, colored, trange, prod
from tinygrad.nn.datasets import mnist
import os
from extra.fsdp.utils import print_size

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 3), Tensor.relu,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      Tensor.max_pool2d,
      lambda x: x.dropout(0.25),
      lambda x: x.flatten(1),
      nn.Linear(9216, 128),
      Tensor.relu,
      lambda x: x.dropout(0.5),
      nn.Linear(128, 10),
      lambda x: x.log_softmax(1),
  ]
    
  def __call__(self, x):
    return x.sequential(self.layers)
  

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()


  model = Model()
  opt = nn.optim.AdamW(nn.state.get_parameters(model))
  print_size("model", *nn.state.get_parameters(model))
  print_size("model with optimizer", *nn.state.get_parameters(opt))
  SHARD = int(os.environ.get("SHARD"))
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
  if SHARD > 1:
    print("SHARDING ON", GPUS)
    X_test.shard_(GPUS)
    Y_test.shard_(GPUS)
    for k, p in nn.state.get_state_dict(opt).items():
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
      if SHARD > 1:
        Xt.shard_(GPUS)
        Yt.shard_(GPUS)
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt).sparse_categorical_crossentropy(Yt).backward()
      opt.step()
      return loss


  @TinyJit
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 10))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    # if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))

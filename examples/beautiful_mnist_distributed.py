# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import os
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange, unwrap
from tinygrad.nn.datasets import mnist
from extra.distributed import DistributedDataParallel
from examples.mlperf.helpers import dedup_dict

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

if __name__ == "__main__":
  SEED = getenv('SEED', 1337)
  Tensor.manual_seed(SEED)
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  model = Model()
  params = nn.state.get_parameters(model)
  ibs = ['IB:0/192.168.200.30', 'IB:0/192.168.200.31']
  idx = int(unwrap(os.getenv('IDX')))
  dist = DistributedDataParallel(model, ibs[1-idx], bool(idx))
  opt = nn.optim.Adam(params)

  Tensor.manual_seed(SEED + idx + 1)

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    dist.sync()
    opt.step()
    return loss

  @TinyJit
  @Tensor.test()
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
    nn.state.safe_save(dedup_dict(nn.state.get_state_dict(model)), f"./ckpts/bmnst_{i}.safe")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))

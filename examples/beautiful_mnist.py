# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist

import itertools
from tinygrad.nn.optim import Optimizer
from tinygrad.dtype import dtypes
class FusedAdam(Optimizer):
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.pos_params = list(itertools.accumulate(self.params, lambda x,y: x+y.flatten().shape[0], initial=0))
    self.m = Tensor.zeros(self.pos_params[-1], dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous()
    self.v = Tensor.zeros(self.pos_params[-1], dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous()

  def schedule_step_with_grads(self, grads:list[Tensor]) -> list[Tensor]:
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    t = Tensor.cat(*[x.flatten() for x in self.params], dim=0)
    g = Tensor.cat(*[x.flatten() for x in grads], dim=0)
    self.m.assign(self.b1 * self.m + (1.0 - self.b1) * g)
    self.v.assign(self.b2 * self.v + (1.0 - self.b2) * (g * g))
    m_hat = self.m / (1.0 - self.b1_t)
    v_hat = self.v / (1.0 - self.b2_t)
    up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
    out = (t.detach() - self.lr * up).cast(t.dtype)
    # right where it belongs
    for i, tt in enumerate(self.params): tt.assign(out[self.pos_params[i]:self.pos_params[i+1]].reshape(tt.shape))
    return [self.b1_t, self.b2_t, self.m, self.v]

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
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  # TODO: these should not be needed
  X_train.realize()
  Y_train.realize()

  model = Model()
  if getenv("FUSE_ADAM"): opt = FusedAdam(nn.state.get_parameters(model))
  else: opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
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

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))

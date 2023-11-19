# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from extra.datasets import fetch_mnist
from tqdm import trange

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm2d(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm2d(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=True)

  model = Model()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  # TODO: there's a compiler error if you comment out TinyJit since randint isn't being realized and there's something weird with int
  @TinyJit
  def train_step(samples:Tensor) -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
      opt.step()
      return loss.realize()

  @TinyJit
  def get_test_acc() -> Tensor: return ((model(X_test).argmax(axis=1) == Y_test).mean()*100).realize()

  test_acc = float('nan')
  for i in (t:=trange(70)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    samples = Tensor.randint(512, high=X_train.shape[0])  # TODO: put this in the JIT when rand is fixed
    loss = train_step(samples)
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

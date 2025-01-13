# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad import Tensor, nn, Device
from typing import Callable
from extra.fsdp import fsdp, FSDPLinear
from extra.checkpoint import checkpoint
class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      FSDPLinear(784, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 2500), Tensor.relu,
      FSDPLinear(2500, 10)
  ]

  def __call__(self, x:Tensor) -> Tensor:
    x = x.flatten(1)
    for i, layer in enumerate(self.layers):
      if i == 0 or i % 2 == 1: #don't checkpoint the first layer, nor relus
        x = layer(x)
      else:
        x = checkpoint(x, fn=layer)

    return x

if __name__ == "__main__":
  Device.DEFAULT = "CUDA"
  GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))
  GPUS = ("CLANG", "CUDA")
  X_train, Y_train, X_test, Y_test = mnist()

  Device.DEFAULT = "CLANG" #initialize the parameters on CPU
  model = Model()
  opt = fsdp(nn.optim.Adam(nn.state.get_parameters(model)), GPUS)
  Device.DEFAULT = "CUDA"

  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
      Xt, Yt = X_train[samples].shard_(GPUS, axis=0), Y_train[samples].shard_(GPUS, axis=0)  # we shard the data on axis 0
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(Xt).sparse_categorical_crossentropy(Yt).backward()
      opt.step()
      return loss

  for i in (t:=trange(getenv("STEPS", 5))):
    loss = train_step()
    t.set_description(f"loss: {loss.item():6.2f}")



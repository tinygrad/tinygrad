# based on https://blog.keras.io/building-autoencoders-in-keras.html and beautiful_mnist.py
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from extra.datasets import fetch_mnist
from tqdm import trange
import numpy as np
from PIL import Image
from itertools import chain

#from examples/yolov8.py
class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])

class Autoencoder:
  def __init__(self):
    self.encoder: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 16, 3, padding=1),Tensor.relu,
      lambda x: x.max_pool2d(stride=2),
      nn.Conv2d(16, 8, 3, padding=1),Tensor.relu,
      lambda x: x.max_pool2d(stride=2),
      nn.Conv2d(8, 8, 3, padding=1),Tensor.relu,
      lambda x: x.max_pool2d(stride=2, dilation=0),
    ]

    self.decoder: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(8, 8, 3, padding=1),Tensor.relu,
      Upsample(2),
      nn.Conv2d(8, 8, 3, padding=1),Tensor.relu,
      Upsample(2),
      nn.Conv2d(8, 16, 3),Tensor.relu,
      Upsample(2),
      nn.Conv2d(16, 1, 3, padding=1),Tensor.sigmoid
    ]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.encoder).sequential(self.decoder)

if __name__ == "__main__":
  X_train, _, X_test, _ = fetch_mnist(tensors=True)
  X_train = X_train / 255.0
  X_test = X_test / 255.0

  model = Autoencoder()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  def train_step(samples:Tensor) -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(X_train[samples]).binary_crossentropy(X_train[samples]).backward()
      opt.step()
      return loss.realize()

  @TinyJit
  def get_test_loss() -> Tensor: return model(X_test).binary_crossentropy(X_test).realize()

  test_loss = float('nan')
  for i in (t:=trange(800)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    samples = Tensor.randint(128, high=X_train.shape[0])  # TODO: put this in the JIT when rand is fixed
    loss = train_step(samples)
    if i%10 == 9: test_loss = get_test_loss().item()
    t.set_description(f"loss: {loss.item():6.2f} test_loss: {test_loss:6.2f}")

  def write_image(path:str,image:Tensor):
    Image.fromarray(np.clip(image.numpy()*255,0,255).astype(np.uint8).reshape(28,28)).save(path)

  for i in range(4):
    write_image(f"examples/autoencoder_mnist_{i}_input.png",X_train[samples[i]].reshape(1,1,28,28))
    write_image(f"examples/autoencoder_mnist_{i}_output.png",model(X_train[samples[i]].reshape(1,1,28,28)))
import os
import numpy as np
import random
np.set_printoptions(suppress=True)
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from tqdm import trange

INNER = 256
class TinyNet:
  def __init__(self):
    self.l1 = Linear(240,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,2)
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    return self.l3(x)

@TinyJit
def train_step(x,y):
  Tensor.training = True
  out = net(x)
  loss = (out-y).square().mean()
  optim.zero_grad()
  loss.backward()
  optim.step()
  return loss.realize()

def get_minibatch(tx, ty, bs=32):
  xs, ys = [], []
  for _ in range(bs):
    sel = random.randint(0, tx.shape[0]-1)
    xs.append(tx[sel:sel+1])
    ys.append(ty[sel:sel+1])
  return Tensor(np.concatenate(xs, axis=0)), Tensor(np.concatenate(ys, axis=0))

def load_dset(fn):
  ty = Tensor.empty(os.path.getsize(f"/tmp/{fn}_y")//4, dtype=dtypes.float32, device=f"disk:/tmp/{fn}_y").cpu().reshape(-1, 2).log().realize()
  tx = Tensor.empty(os.path.getsize(f"/tmp/{fn}_x")//4, dtype=dtypes.float32, device=f"disk:/tmp/{fn}_x").cpu().reshape(ty.shape[0], -1).realize()
  print(tx.shape, ty.shape)
  return tx,ty

if __name__ == "__main__":
  ax, ay = load_dset("allopt")
  tx,vx = ax[:9000], ax[9000:]
  ty,vy = ay[:9000], ay[9000:]
  tx,ty = tx.numpy(), ty.numpy()
  vx,vy = vx.metal(), vy.metal()

  Tensor.no_grad = False
  Tensor.training = True
  net = TinyNet()
  optim = Adam(get_parameters(net))

  losses, vals = [], []
  for i in (t:=trange(2000)):
    x,y = get_minibatch(tx, ty, 512)
    loss = train_step(x, y)
    if i%10 == 0:
      val = (vy-net(vx)).square().mean()
      losses.append(loss.numpy().item())
      vals.append(val.numpy().item())
      print(losses[-1], vals[-1])

  import matplotlib.pyplot as plt
  plt.plot(losses[10:])
  plt.plot(vals[10:])
  plt.show()
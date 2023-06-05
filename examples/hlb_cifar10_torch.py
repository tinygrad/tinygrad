import numpy as np
import torch
import time
import platform
from torch import nn
from torch import optim

from datasets import fetch_cifar
from tinygrad.helpers import getenv

# allow TF32
torch.set_float32_matmul_precision('high')

OSX = platform.system() == "Darwin"
device = 'mps' if OSX else 'cuda'

num_classes = 10
class ConvGroup(nn.Module):
  def __init__(self, channels_in, channels_out, short, se=True):
    super().__init__()
    self.short, self.se = short, se and not short
    self.conv = nn.ModuleList([nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)])
    self.norm = nn.ModuleList([nn.BatchNorm2d(channels_out, track_running_stats=False, eps=1e-12, momentum=0.8) for _ in range(1 if short else 3)])
    if self.se: self.se1, self.se2 = nn.Linear(channels_out, channels_out//16), nn.Linear(channels_out//16, channels_out)

  def forward(self, x):
    x = nn.functional.max_pool2d(self.conv[0](x), 2)
    x = self.norm[0](x).relu()
    if self.short: return x
    residual = x
    mult = self.se2(self.se1(residual.mean((2,3))).relu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1) if self.se else 1.0
    x = self.norm[1](self.conv[1](x)).relu()
    x = self.norm[2](self.conv[2](x) * mult).relu()
    return x + residual

class GlobalMaxPool(nn.Module):
  def forward(self, x): return torch.amax(x, dim=(2,3))

class SpeedyResNet(nn.Module):
  def __init__(self):
    super().__init__()
    # TODO: add whitening
    self.net = nn.ModuleList([
      nn.Conv2d(3, 64, kernel_size=1),
      nn.BatchNorm2d(64, track_running_stats=False, eps=1e-12, momentum=0.8),
      nn.ReLU(),
      ConvGroup(64, 128, short=False),
      ConvGroup(128, 256, short=True),
      ConvGroup(256, 512, short=False),
      GlobalMaxPool(),
      nn.Linear(512, num_classes, bias=False)
    ])

  # note, pytorch just uses https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html instead of log_softmax
  def forward(self, x):
    for layer in self.net:
      x = layer(x)
    return x.log_softmax(-1)

def train_step_jitted(model, optimizer, X, Y):
  out = model(X)
  loss = (out * Y).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  correct = out.detach().argmax(axis=1) == Y.detach().argmin(axis=1)
  return loss, correct

def fetch_batch(X_train, Y_train, BS):
  # fetch a batch
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  Y = np.zeros((BS, num_classes), np.float32)
  Y[range(BS),Y_train[samp]] = -1.0*num_classes
  X = torch.tensor(X_train[samp])
  Y = torch.tensor(Y.reshape(BS, num_classes))
  return X.to(device), Y.to(device)

def train_cifar():
  BS = getenv("BS", 512)
  if getenv("FAKEDATA"):
    N = 2048
    X_train = np.random.default_rng().standard_normal(size=(N, 3, 32, 32), dtype=np.float32)
    Y_train = np.random.randint(0,10,size=(N), dtype=np.int32)
    X_test, Y_test = X_train, Y_train
  else:
    X_train,Y_train = fetch_cifar(train=True)
    X_test,Y_test = fetch_cifar(train=False)
  print(X_train.shape, Y_train.shape)
  Xt, Yt = fetch_batch(X_test, Y_test, BS=BS)

  model = SpeedyResNet().to(device)
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.85, nesterov=True)
  X, Y = fetch_batch(X_train, Y_train, BS=BS)
  for i in range(getenv("STEPS", 10)):
    #for param_group in optimizer.param_groups: print(param_group['lr'])
    if i%10 == 0:
      # use training batchnorm (and no_grad would change the kernels)
      out = model(Xt).detach()
      loss = (out * Yt).mean().cpu().numpy()
      outs = out.cpu().numpy().argmax(axis=1)
      correct = outs == Yt.detach().cpu().numpy().argmin(axis=1)
      print(f"eval {sum(correct)}/{len(correct)} {sum(correct)/len(correct)*100.0:.2f}%, {loss:7.2f} val_loss")
    st = time.monotonic()
    loss, correct = train_step_jitted(model, optimizer, X, Y)
    et = time.monotonic()
    X, Y = fetch_batch(X_train, Y_train, BS=BS)  # do this here
    loss_cpu = loss.detach().cpu().item()
    correct = correct.cpu().numpy()
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {sum(correct)/len(correct)*100.0:7.2f}% acc")

if __name__ == "__main__":
  train_cifar()

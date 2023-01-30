#!/usr/bin/env python3
# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
from datasets import fetch_cifar
from tinygrad import nn
from tinygrad.nn import optim
from extra.training import train
from extra.utils import get_parameters

class ConvGroup:
  def __init__(self, channels_in, channels_out, short):
    self.short = short
    self.conv = [nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)]
    self.norm = [nn.BatchNorm2D(channels_out) for _ in range(1 if short else 3)]

    if not self.short:
      self.se1 = nn.Linear(channels_out, channels_out//16)
      self.se2 = nn.Linear(channels_out//16, channels_out)

  def __call__(self, x):
    x = self.conv[0](x).max_pool2d(2)
    x = self.norm[0](x).gelu()
    if self.short: return x
    residual = x
    mult = self.se2(self.se1(residual.mean((2,3))).gelu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1)
    x = self.norm[1](self.conv[1](x)).gelu()
    x = self.norm[2](self.conv[2](x) * mult).gelu()
    return x + residual

class SpeedyResNet:
  def __init__(self):
    # TODO: add whitening
    self.net = [
      nn.Conv2d(3, 64, kernel_size=1),
      nn.BatchNorm2D(64),
      lambda x: x.gelu(),
      ConvGroup(64, 128, short=False),
      ConvGroup(128, 256, short=True),
      ConvGroup(256, 512, short=False),
      lambda x: x.max((2,3)),
      nn.Linear(512, 1000, bias=False)
    ]

  def __call__(self, x): return x.sequential(self.net)

def train_cifar():
  X,Y = fetch_cifar()
  model = SpeedyResNet()
  optimizer = optim.SGD(get_parameters(model))
  train(model, X, Y, optimizer, steps=X.shape[0]//512, BS=512)

if __name__ == "__main__":
  train_cifar()

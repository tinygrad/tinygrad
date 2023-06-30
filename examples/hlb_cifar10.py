#!/usr/bin/env python3
# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
# TODO: gelu is causing nans!
import time
import numpy as np
from datasets import fetch_cifar
from tinygrad import nn
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.ops import GlobalCounters

Tensor.manual_seed(getenv('SEED', 6)) # Deterministic
np.random.seed(getenv('SEED', 6))

num_classes = 10

# TODO: eval won't work with track_running_stats=False
class ConvGroup:
  def __init__(self, channels_in, channels_out, short, se=True):
    self.short, self.se = short, se and not short
    self.conv = [nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)]
    self.norm = [nn.BatchNorm2d(channels_out, track_running_stats=False, eps=1e-12, momentum=0.8) for _ in range(1 if short else 3)]
    if self.se: self.se1, self.se2 = nn.Linear(channels_out, channels_out//16), nn.Linear(channels_out//16, channels_out)

  def __call__(self, x):
    x = self.conv[0](x).max_pool2d(2)
    x = self.norm[0](x).relu()
    if self.short: return x
    residual = x
    mult = self.se2(self.se1(residual.mean((2,3))).relu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1) if self.se else 1.0
    x = self.norm[1](self.conv[1](x)).relu()
    x = self.norm[2](self.conv[2](x) * mult).relu()
    return x + residual

class SpeedyResNet:
  def __init__(self):
    # TODO: add whitening
    self.net = [
      nn.Conv2d(3, 64, kernel_size=1),
      nn.BatchNorm2d(64, track_running_stats=False, eps=1e-12, momentum=0.8),
      lambda x: x.relu(),
      ConvGroup(64, 128, short=False),
      ConvGroup(128, 256, short=True),
      ConvGroup(256, 512, short=False),
      lambda x: x.max((2,3)),
      nn.Linear(512, num_classes, bias=False)
    ]

  # note, pytorch just uses https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html instead of log_softmax
  def __call__(self, x): return x.sequential(self.net).log_softmax()

from tinygrad.jit import TinyJit
@TinyJit
def train_step_jitted(model, optimizer, optimizer_bias, X, Y):
  out = model(X)
  loss = out.mul(Y).mean()
  if not getenv("DISABLE_BACKWARD"):
    optimizer.zero_grad()
    optimizer_bias.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_bias.step()
  return loss.realize()

def fetch_batches(X_train, Y_train, BS, is_train=False):
  if not is_train:
    ind = np.arange(Y_train.shape[0])
    np.random.shuffle(ind)
    X_train, Y_train = X_train[ind, ...], Y_train[ind, ...]
  while True:
    for batch_start in range(0, Y_train.shape[0], BS):
      batch_end = min(batch_start+BS, Y_train.shape[0])
      X = Tensor(X_train[batch_end-BS:batch_end]) # batch_end-BS for padding
      Y = np.zeros((BS, num_classes), np.float32)
      Y[range(BS),Y_train[batch_end-BS:batch_end]] = -1.0*num_classes
      Y = Tensor(Y.reshape(BS, num_classes))
      yield X, Y
    if not is_train:
      break

def train_cifar(bs=256, eval_bs=250, steps=2000, lr=0.01, momentum=0.85, wd=0.01, lr_bias=0.1, momentum_bias=0.85, wd_bias=0.003):
  Tensor.training = True
  BS, EVAL_BS, STEPS = getenv("BS", bs), getenv('EVAL_BS', eval_bs), getenv("STEPS", steps)
  LR, MOMENTUM, WD = getenv("LR", lr), getenv('MOMENTUM', momentum), getenv("WD", wd)
  LR_BIAS, MOMENTUM_BIAS, WD_BIAS = getenv("LR_BIAS", lr_bias), getenv('MOMENTUM_BIAS', momentum_bias), getenv("WD_BIAS", wd_bias)

  if getenv("FAKEDATA"):
    N = 2048
    X_train = np.random.default_rng().standard_normal(size=(N, 3, 32, 32), dtype=np.float32)
    Y_train = np.random.randint(0,10,size=(N), dtype=np.int32)
    X_test, Y_test = X_train, Y_train
  else:
    X_train, Y_train = fetch_cifar(train=True)
    X_test, Y_test = fetch_cifar(train=False)
  model = SpeedyResNet()

  # init weights with torch
  # TODO: it doesn't learn with the tinygrad weights, likely since kaiming init
  if getenv("TORCHWEIGHTS"):
    from examples.hlb_cifar10_torch import SpeedyResNet as SpeedyResNetTorch
    torch_model = SpeedyResNetTorch()
    model_state_dict = optim.get_state_dict(model)
    torch_state_dict = torch_model.state_dict()
    for k,v in torch_state_dict.items():
      old_mean_std = model_state_dict[k].mean().numpy(), model_state_dict[k].std().numpy()
      model_state_dict[k].assign(Tensor(v.detach().numpy())).realize()
      new_mean_std = model_state_dict[k].mean().numpy(), model_state_dict[k].std().numpy()
      print(f"initted {k:40s} {str(model_state_dict[k].shape):20s} from torch mean:{old_mean_std[0]:8.5f} -> {new_mean_std[0]:8.5f} std:{old_mean_std[1]:8.5f} -> {new_mean_std[1]:8.5f}")
    exit(0)

  non_bias_params, bias_params = [], []
  for name, param in optim.get_state_dict(model).items():
    if 'bias' in name: bias_params.append(param)
    else: non_bias_params.append(param)
  optimizer = optim.SGD(non_bias_params, lr=LR, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
  optimizer_bias = optim.SGD(bias_params, lr=LR_BIAS, momentum=MOMENTUM_BIAS, nesterov=True, weight_decay=WD_BIAS)

  # 97 steps in 2 seconds = 20ms / step
  # step is 1163.42 GOPS = 56 TFLOPS!!!, 41% of max 136
  # 4 seconds for tfloat32 ~ 28 TFLOPS, 41% of max 68
  # 6.4 seconds for float32 ~ 17 TFLOPS, 50% of max 34.1
  # 4.7 seconds for float32 w/o channels last. 24 TFLOPS. we get 50ms then i'll be happy. only 64x off

  # https://www.anandtech.com/show/16727/nvidia-announces-geforce-rtx-3080-ti-3070-ti-upgraded-cards-coming-in-june
  # 136 TFLOPS is the theoretical max w float16 on 3080 Ti
  best_eval = -1
  i = 0
  for X, Y in fetch_batches(X_train, Y_train, BS=BS, is_train=True):
    if i > STEPS: break
    if i%20 == 0 and STEPS != 1:
      # use training batchnorm (and no_grad would change the kernels)
      corrects = []
      losses = []
      for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS):
        out = model(Xt)
        outs = out.numpy().argmax(axis=1)
        loss = (out * Yt).mean().numpy()
        losses.append(loss.tolist())
        correct = outs == Yt.numpy().argmin(axis=1)
        corrects.extend(correct.tolist())
      acc = sum(corrects)/len(corrects)*100.0
      if acc > best_eval:
        best_eval = acc
        print(f"eval {sum(corrects)}/{len(corrects)} {acc:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i}")
    if STEPS == 0: break
    GlobalCounters.reset()
    st = time.monotonic()
    loss = train_step_jitted(model, optimizer, optimizer_bias, X, Y)
    et = time.monotonic()
    loss_cpu = loss.numpy()
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    i += 1
  #train(model, X, Y, optimizer, steps=X.shape[0]//BS, BS=BS)
  #evaluate(model, X_test, Y_test)

if __name__ == "__main__":
  train_cifar()

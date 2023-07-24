#!/usr/bin/env python3
# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import time
import random
import numpy as np
from extra.datasets import fetch_cifar, cifar_mean, cifar_std
from tinygrad import nn
from tinygrad.state import get_parameters
from tinygrad.nn import optim
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.ops import GlobalCounters
from extra.lr_scheduler import OneCycleLR
from tinygrad.jit import TinyJit
from examples.train_resnet import ComposeTransforms

def set_seed(seed):
  Tensor.manual_seed(getenv('SEED', seed)) # Deterministic
  np.random.seed(getenv('SEED', seed))
  random.seed(getenv('SEED', seed))

num_classes = 10

# TODO remove dependency on torch, mainly unfold and eigh
def whitening(X):
  import torch
  def _cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

  def _patches(data, patch_size=(2,2)):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w)

  def _eigens(patches):
    n,c,h,w = patches.shape
    Σ = _cov(patches.reshape(n, c*h*w))
    Λ, V = torch.linalg.eigh(Σ)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)
  X = torch.tensor(transform(X).numpy())
  Λ, V = _eigens(_patches(X))
  W = (V/torch.sqrt(Λ+1e-2)[:,None,None,None]).numpy()

  return W

class ConvGroup:
  def __init__(self, channels_in, channels_out, short, se=True):
    self.short, self.se = short, se and not short
    self.conv = [nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)]
    self.norm = [nn.BatchNorm2d(channels_out, track_running_stats=False, eps=1e-12, momentum=0.5) for _ in range(1 if short else 3)]
    if self.se: self.se1, self.se2 = nn.Linear(channels_out, channels_out//16), nn.Linear(channels_out//16, channels_out)

  def __call__(self, x):
    x = self.conv[0](x).max_pool2d(2)
    x = self.norm[0](x).relu()
    if self.short: return x
    residual = x
    mult = self.se2((self.se1(residual.mean((2,3)))).relu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1) if self.se else 1.0
    x = self.norm[1](self.conv[1](x)).relu()
    x = self.norm[2](self.conv[2](x) * mult).relu()
    return x + residual

class SpeedyResNet:
  def __init__(self, W=None):
    if W:
      self.whitening = Tensor(W, requires_grad=False)
    self.net = [
      nn.Conv2d(12, 64, kernel_size=1),
      nn.BatchNorm2d(64, track_running_stats=False, eps=1e-12, momentum=0.5),
      lambda x: x.relu(),
      ConvGroup(64, 128, short=False),
      ConvGroup(128, 256, short=True),
      ConvGroup(256, 512, short=False),
      lambda x: x.max((2,3)),
      nn.Linear(512, num_classes, bias=False)
    ]
  # note, pytorch just uses https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html instead of log_softmax
  def __call__(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to do the rest conv layers
    forward = lambda x: x.conv2d(self.whitening).pad2d((1,0,0,1)).sequential(self.net) if W else lambda x: x.sequential(self.net)
    if not training and getenv('TTA', 1)==1: return ((forward(x)*0.5) + (forward(x[..., ::-1])*0.5)).log_softmax()
    return forward(x).log_softmax()

def Cutmix(X, Y, mask_size=3, p=0.5):
  if Tensor.rand(1) > 0.5:
    return X, Y
  # create a mask
  is_even = int(mask_size % 2 == 0)
  center_max = X.shape[-2]-mask_size//2-is_even
  center_min = mask_size//2-is_even
  center = Tensor.rand(X.shape[0])*(center_max-center_min)+center_min
  
  d_y = Tensor.arange(0, X.shape[-2]).reshape((1,1,X.shape[-2],1))
  d_x = Tensor.arange(0, X.shape[-1]).reshape((1,1,1,X.shape[-1]))
  d_y = d_y - center.reshape((-1,1,1,1))
  d_x = d_x - center.reshape((-1,1,1,1))
  d_y =(d_y >= -(mask_size / 2)) * (d_y <= mask_size / 2)
  d_x =(d_x >= -(mask_size / 2)) * (d_x <= mask_size / 2)
  mask = d_y * d_x
  # # Tensor.rand(X.shape) would trigger error
  # X_cutout = Tensor.where(mask, X_patch, X)

  # TODO shuffle instead of reverse inside tinygrad tensor, currently is not supported
  X_patch = X[::-1,...]
  X_cutmix = Tensor.where(mask, X_patch, X)
  Y_cutmix = Y[::-1]
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_cutmix + (1. - mix_portion) * Y
  
  return X_cutmix, Y_cutmix

transform = ComposeTransforms([
    lambda x: x.to(device=Device.DEFAULT).float(),
    lambda x: x / 255.0, # scale
    lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1), # normalize
    lambda x: x.reshape((-1,3,32,32)),
    lambda x: Tensor.where(Tensor.rand(x.shape[0],1,1,1) < 0.5, x[..., ::-1], x), # flip LR
])

transform_test = ComposeTransforms([
    lambda x: x.to(device=Device.DEFAULT).float(),
    lambda x: x / 255.0,
    lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1),
    lambda x: x.reshape((-1,3,32,32)),
])

def fetch_batches(X, Y, BS, seed, is_train=False):
  while True:
    set_seed(seed)
    order = list(range(0, X.shape[0], BS))
    random.shuffle(order)
    for i in order:
      # padding for matching buffer size during JIT
      batch_end = min(i+BS, Y.shape[0])
      if not is_train:
        x = transform_test(X[batch_end-BS:batch_end,:])
      x = transform(X[batch_end-BS:batch_end,:])
      # NOTE -10 was used instead of 1 as labels
      y = Tensor(-10*np.eye(10, dtype=np.float32)[Y[batch_end-BS:batch_end].numpy()])
      x, y = cutmix(x, y)
      yield x, y
    if not is_train: break
    seed += 1

def train_cifar(bs=512, eval_bs=500, steps=1000, 
                # training hyper-parameters (if including model sizes)
                div_factor=1e16, final_lr_ratio=0.004560827731448039, max_lr=0.01040497290691913, pct_start=0.22817715646040532, momentum=0.8468770654506089, wd=0.17921940728200592, label_smoothing=0.2, seed=32):
  set_seed(seed)
  Tensor.training = True

  BS, EVAL_BS, STEPS = getenv("BS", bs), getenv('EVAL_BS', eval_bs), getenv("STEPS", steps)
  # For SGD
  MOMENTUM, WD = getenv('MOMENTUM', momentum), getenv("WD", wd)
  # For LR Scheduler
  MAX_LR, PCT_START, DIV_FACTOR = getenv("MAX_LR", max_lr), getenv('PCT_START', pct_start), getenv('DIV_FACTOR', div_factor)
  FINAL_DIV_FACTOR = 1./(DIV_FACTOR*getenv('FINAL_LR_RATIO', final_lr_ratio)) 
  # Others
  LABEL_SMOOTHING = getenv('LABEL_SMOOTHING', label_smoothing)

  if getenv("FAKEDATA"):
    N = 2048
    X_train = np.random.default_rng().standard_normal(size=(N, 3, 32, 32), dtype=np.float32)
    Y_train = np.random.randint(0,10,size=(N), dtype=np.int32)
    X_test, Y_test = X_train, Y_train
  else:
    X_train, Y_train, X_test, Y_test = fetch_cifar()    # they are disk tensor now
 
  # precompute whitening patches
  W = whitening(X_train)

  model = SpeedyResNet(W)
  optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
  lr_scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, div_factor=DIV_FACTOR, final_div_factor=FINAL_DIV_FACTOR, total_steps=STEPS, pct_start=PCT_START)
  # JIT at every run
  @TinyJit
  def train_step_jitted(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss = (1 - LABEL_SMOOTHING) * out.mul(Y).mean() + (-1 * LABEL_SMOOTHING * out.mean())
    if not getenv("DISABLE_BACKWARD"):
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
    return loss.realize()

  @TinyJit
  def eval_step_jitted(model, X, Y):
    out = model(X, training=False)
    loss = out.mul(Y).mean()
    return out.realize(), loss.realize()

  # 97 steps in 2 seconds = 20ms / step  Tensor.training = True

  # step is 1163.42 GOPS = 56 TFLOPS!!!, 41% of max 136
  # 4 seconds for tfloat32 ~ 28 TFLOPS, 41% of max 68
  # 6.4 seconds for float32 ~ 17 TFLOPS, 50% of max 34.1
  # 4.7 seconds for float32 w/o channels last. 24 TFLOPS. we get 50ms then i'll be happy. only 64x off

  # https://www.anandtech.com/show/16727/nvidia-announces-geforce-rtx-3080-ti-3070-ti-upgraded-cards-coming-in-june
  # 136 TFLOPS is the theoretical max w float16 on 3080 Ti
  best_eval = -1
  i = 0
  batcher = fetch_batches(X_train, Y_train, BS=BS, seed=seed, is_train=True)
  while i <= STEPS:
    X, Y = next(batcher)
    if i%100 == 0 and i > 1:
      # batchnorm is frozen, no need for Tensor.training=False
      corrects = []
      losses = []
      for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, seed=seed):
        out, loss = eval_step_jitted(model, Xt, Yt)
        outs = out.numpy().argmax(axis=1)
        correct = outs == Yt.numpy().argmin(axis=1)
        losses.append(loss.numpy().tolist())
        corrects.extend(correct.tolist())
      acc = sum(corrects)/len(corrects)*100.0
      if acc > best_eval:
        best_eval = acc
        print(f"eval {sum(corrects)}/{len(corrects)} {acc:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i}")
    if STEPS == 0 or i==STEPS: break
    GlobalCounters.reset()
    st = time.monotonic()
    loss = train_step_jitted(model, optimizer, lr_scheduler, X, Y)
    et = time.monotonic()
    loss_cpu = loss.numpy()
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    i += 1

if __name__ == "__main__":
  train_cifar()

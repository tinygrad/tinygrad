#!/usr/bin/env python3
# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import time
import random
import numpy as np
from extra.datasets import fetch_cifar, cifar_mean, cifar_std
from tinygrad import nn
from tinygrad.state import get_state_dict
from tinygrad.nn import optim
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.ops import GlobalCounters
from extra.lr_scheduler import OneCycleLR
from tinygrad.jit import TinyJit

# TODO adjust dict for hyperparameters
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

batchsize = 512
bias_scaler = 56

hyp = {
    'opt': {
        'bias_lr':        1.64 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'non_bias_lr':    1.64 / 512,
        'bias_decay':     1.08 * 6.45e-4 * batchsize/bias_scaler,
        'non_bias_decay': 1.08 * 6.45e-4 * batchsize,
        'percent_start':  0.23,
        'div_factor':     1e16,
        'final_lr_ratio': 0.07,
        'scaling_factor': 1./9,
        'loss_scale_scaler': 1./128, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .5, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'conv_norm_pow': 2.6,
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    }
}

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
    Λ, V = torch.linalg.eigh(Σ, UPLO='L')
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

  X = torch.tensor(X.numpy())
  Λ, V = _eigens(_patches(X))
  W = Tensor((V/torch.sqrt(Λ+1e-2)[:,None,None,None]).numpy(), requires_grad=False)

  return W

class ConvGroup:
  def __init__(self, channels_in, channels_out, short, se=True):
    self.short, self.se = short, se and not short
    self.conv = [nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)]
    # eps needs to be 1e-5 to support fp16 but currently it will drop val acc on fp32
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
  def __init__(self, W):
    self.whitening = W
    self.net = [
      nn.Conv2d(12, 64, kernel_size=1),
      # eps needs to be 1e-5 to support fp16 but currently it will drop val acc on fp32
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
    forward = lambda x: x.conv2d(self.whitening).pad2d((1,0,0,1)).sequential(self.net)
    if not training and getenv('TTA', 0)==1: return forward(x)*0.5 + forward(x[..., ::-1])*0.5
    return forward(x)

def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
  y = (1 - label_smoothing)*y + label_smoothing / y.shape[1]
  if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
  if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
  return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

# TODO currently this only works for RGB in format of NxCxHxW and pads the HxW
# implemented in recursive fashion but figuring out how to switch indexing dim
# during the loop was a bit tricky
def pad_reflect(X, padding) -> Tensor:
  p = padding[3]
  s = X.shape[3]

  X_lr = X[...,:,1:1+p[0]].flip(3).pad(((0,0),(0,0),(0,0),(0,s+p[0]))) + X[...,:,-1-p[1]:-1].flip(3).pad(((0,0),(0,0),(0,0),(s+p[1],0)))
  X = X.pad(((0,0),(0,0),(0,0),p)) + X_lr

  p = padding[2]
  s = X.shape[2]
  X_lr = X[...,1:1+p[0],:].flip(2).pad(((0,0),(0,0),(0,s+p[0]),(0,0))) + X[...,-1-p[1]:-1,:].flip(2).pad(((0,0),(0,0),(s+p[1],0),(0,0)))
  X = X.pad(((0,0),(0,0),p,(0,0))) + X_lr

  return X

# return a mask in the format of BS x C x H x W where H x W are in bool
def make_square_mask(X, mask_size):
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

  return mask

def random_crop(X, crop_size=32):
  mask = make_square_mask(X, crop_size)
  mask = mask.repeat((1,3,1,1))
  X_cropped = Tensor(X.flatten().numpy()[mask.flatten().numpy().astype(bool)])

  return X_cropped.reshape((-1, 3, crop_size, crop_size))

def cutmix(X, Y, mask_size=5, p=0.5, mix=True):
  if Tensor.rand(1) > 0.5: return X, Y

  mask = make_square_mask(X, mask_size)

  if not mix: return Tensor.where(mask, Tensor.rand(*X.shape), X), Y

  # TODO shuffle instead of reverse, currently is not supported
  X_patch = X[::-1,...]
  Y_patch = Y[::-1]
  X_cutmix = Tensor.where(mask, X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

transform = [
  lambda x: x.to(device=Device.DEFAULT).float(),
  lambda x: x / 255.0, # scale
  lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1), # normalize
  lambda x: x.reshape((-1,3,32,32)),
  lambda x: pad_reflect(x, ((0,0),(0,0),(2,2),(2,2))),
  lambda x: random_crop(x),
  lambda x: Tensor.where(Tensor.rand(x.shape[0],1,1,1) < 0.5, x[..., ::-1], x), # flip LR
  # ideally cutmix can also be placed here but it also takes y
]

transform_test = [
  lambda x: x.to(device=Device.DEFAULT).float(),
  lambda x: x / 255.0,
  lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1),
  lambda x: x.reshape((-1,3,32,32)),
]

def fetch_batches(X, Y, BS, seed, is_train=False):
  while True:
    set_seed(seed)
    # here only shuffle by batches
    order = list(range(0, X.shape[0], BS))
    random.shuffle(order)
    for i in order:
      # padding to match buffer size during JIT
      batch_end = min(i+BS, Y.shape[0])
      x = X[batch_end-BS:batch_end,:]
      # Need fancy indexing support to remove numpy
      y = Tensor(np.eye(10, dtype=np.float32)[Y[batch_end-BS:batch_end].numpy()])
      if not is_train:
        x = x.sequential(transform_test)
      else:
        # ideally put cutmix inside transform
        x, y = cutmix(x.sequential(transform), y)
      yield x, y

    if not is_train: break
    seed += 1

def train_cifar(bs=512, eval_bs=500, steps=1000,
                momentum=0.8632474768028381, wd=0.07324837942480592,
                max_lr=0.0138319916999336,
                label_smoothing=0.24287006281063067,
                seed=32):
  set_seed(seed)
  Tensor.training = True

  BS, EVAL_BS, STEPS = getenv("BS", bs), getenv('EVAL_BS', eval_bs), getenv("STEPS", steps)
  # For SGD
  MOMENTUM, WD = getenv('MOMENTUM', momentum), getenv("WD", wd)
  # For LR Scheduler
  MAX_LR = getenv("MAX_LR", max_lr)
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
  W = whitening(X_train.sequential(transform_test))

  model = SpeedyResNet(W)

  # parse the training params into bias and non-bias
  params_dict = get_state_dict(model)
  params_non_bias = []
  params_bias = []
  for params in params_dict:
    if params_dict[params].requires_grad is not False:
      if 'bias' in params:
        params_bias.append(params_dict[params])
      else:
        params_non_bias.append(params_dict[params])

  opt_bias     = optim.SGD(params_bias,     lr=0.01, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=MOMENTUM, nesterov=True, weight_decay=WD)

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = hyp['opt']['div_factor']
  final_lr_ratio = hyp['opt']['final_lr_ratio']
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias     = OneCycleLR(opt_bias,     max_lr=MAX_LR ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)
  lr_sched_non_bias = OneCycleLR(opt_non_bias, max_lr=MAX_LR ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)

  @TinyJit
  def train_step_jitted(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss = cross_entropy(out, Y, label_smoothing=LABEL_SMOOTHING)
    if not getenv("DISABLE_BACKWARD"):
      # 0 for bias and 1 for non-bias
      optimizer[0].zero_grad()
      optimizer[1].zero_grad()
      loss.backward()
      optimizer[0].step()
      optimizer[1].step()
      lr_scheduler[0].step()
      lr_scheduler[1].step()
    return loss.realize()

  @TinyJit
  def eval_step_jitted(model, X, Y):
    out = model(X, training=False)
    loss = cross_entropy(out, Y, reduction='mean') 
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
        correct = out.numpy().argmax(axis=1) == Yt.numpy().argmax(axis=1)
        losses.append(loss.numpy().tolist())
        corrects.extend(correct.tolist())
      acc = sum(corrects)/len(corrects)*100.0
      if acc > best_eval:
        best_eval = acc
        print(f"eval {sum(corrects)}/{len(corrects)} {acc:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i}")
    if STEPS == 0 or i==STEPS: break
    GlobalCounters.reset()
    st = time.monotonic()
    loss = train_step_jitted(model, [opt_bias, opt_non_bias], [lr_sched_bias, lr_sched_non_bias], X, Y)
    et = time.monotonic()
    loss_cpu = loss.numpy()
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {opt_non_bias.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    i += 1

if __name__ == "__main__":
  train_cifar()

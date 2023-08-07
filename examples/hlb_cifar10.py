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
        'bias_lr':        1.64 * bias_scaler/512,
        'non_bias_lr':    1.64 / 512,
        'bias_decay':     1.08 * 6.45e-4 * batchsize/bias_scaler,
        'non_bias_decay': 1.08 * 6.45e-4 * batchsize,
        'momentum':       0.85,
        'percent_start':  0.25,
        'scaling_factor': 1./9,
        'loss_scale_scaler': 1./512, # regularizer inside the loss summing (range: ~1/512 - 16+). 
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .5, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'conv_norm_pow': 2.6,
        'cutmix_size': 3,
        'cutmix_steps': 588,        # original repo used epoch 6 which is roughly 6*98=588 STEPS
        'pad_amount': 2
    }
}

def set_seed(seed):
  Tensor.manual_seed(getenv('SEED', seed)) # Deterministic
  np.random.seed(getenv('SEED', seed))
  random.seed(getenv('SEED', seed))

# ========== Model ==========
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
    Λ, V = torch.linalg.eigh(Σ, UPLO='U')
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

  X = torch.tensor(X.numpy())
  Λ, V = _eigens(_patches(X))
  W = Tensor((V/torch.sqrt(Λ+1e-2)[:,None,None,None]).numpy(), requires_grad=False)

  return W

class ConvGroup:
  def __init__(self, channels_in, channels_out, short, se=True):
    self.short, self.se = short, se and not short
    self.conv = [nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=1, bias=False) for i in range(1 if short else 3)]
    self.norm = [nn.BatchNorm2d(channels_out, track_running_stats=False, eps=1e-12, momentum=0.5) for _ in range(1 if short else 3)]
    if self.se: self.se1, self.se2 = nn.Linear(channels_out, channels_out//16), nn.Linear(channels_out//16, channels_out)

  def __call__(self, x):
    x = self.conv[0](x).max_pool2d(2)
    x = self.norm[0](x).gelu()
    if self.short: return x
    residual = x
    mult = self.se2((self.se1(residual.mean((2,3)))).gelu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1) if self.se else 1.0
    x = self.norm[1](self.conv[1](x)).gelu()
    x = self.norm[2](self.conv[2](x) * mult).gelu()
    return x + residual

class SpeedyResNet:
  def __init__(self, W):
    self.whitening = W
    self.net = [
      nn.Conv2d(12, 32, kernel_size=1),
      lambda x: x.gelu(),
      ConvGroup(32, 64, short=False),
      ConvGroup(64, 256, short=True),
      ConvGroup(256, 512, short=False),
      lambda x: x.max((2,3)),
      nn.Linear(512, 10, bias=False),
      lambda x: x.mul(hyp['opt']['scaling_factor'])
    ]
  def __call__(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to do the rest conv layers
    forward = lambda x: x.conv2d(self.whitening).pad2d((1,0,0,1)).sequential(self.net)
    if not training: return forward(x)*0.5 + forward(x[..., ::-1])*0.5
    return forward(x)

# ========== Loss ==========
def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
  y = (1 - label_smoothing)*y + label_smoothing / y.shape[1]
  if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
  if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
  return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

# ========== Preprocessing ==========
# TODO currently this only works for RGB in format of NxCxHxW and pads the HxW
# implemented in recursive fashion but figuring out how to switch indexing dim
# during the loop was a bit tricky
def pad_reflect(X, size=2) -> Tensor:
  padding = ((0,0),(0,0),(size,size),(size,size))
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

transform = [
  lambda x: x.to(device=Device.DEFAULT).float(),
  lambda x: x / 255.0, # scale
  lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1), # normalize
  lambda x: x.reshape((-1,3,32,32)),
  lambda x: pad_reflect(x, size=hyp['net']['pad_amount']),
  lambda x: random_crop(x),
  lambda x: Tensor.where(Tensor.rand(x.shape[0],1,1,1) < 0.5, x[..., ::-1], x), # flip LR
]

transform_test = [
  lambda x: x.to(device=Device.DEFAULT).float(),
  lambda x: x / 255.0,
  lambda x: (x - Tensor(cifar_mean).repeat((1024,1)).T.reshape(1,-1))/ Tensor(cifar_std).repeat((1024,1)).T.reshape(1,-1),
  lambda x: x.reshape((-1,3,32,32)),
]

def cutmix(X, Y, mask_size=3, p=0.5):
  if Tensor.rand(1) > 0.5: return X, Y

  # fill the square with randomly selected images from the same batch
  mask = make_square_mask(X, mask_size)
  order = list(range(0, X.shape[0]))
  random.shuffle(order) 
  X_patch = Tensor(X.numpy()[order,...])
  Y_patch = Tensor(Y.numpy()[order])
  X_cutmix = Tensor.where(mask, X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

def fetch_batches(X, Y, BS, seed, is_train=False):
  while True:
    set_seed(seed)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    for i in range(0, X.shape[0], BS):
      # padding the last batch in order to match buffer size during JIT
      batch_end = min(i+BS, Y.shape[0])
      # TODO need indexing support for tinygrad Tensor
      x = Tensor(X.numpy()[order[batch_end-BS:batch_end],:])
      y = Tensor(np.eye(10, dtype=np.float32)[Y.numpy()[order[batch_end-BS:batch_end]]])
      x = x.sequential(transform) if is_train else x.sequential(transform_test)
      yield x, y

    if not is_train: break
    seed += 1

def train_cifar(bs=512, eval_bs=500, steps=1000, seed=32):
  set_seed(seed)
  Tensor.training = True

  BS, EVAL_BS, STEPS = getenv("BS", bs), getenv('EVAL_BS', eval_bs), getenv("STEPS", steps)

  if getenv("FAKEDATA"):
    N = 2048
    X_train = np.random.default_rng().standard_normal(size=(N, 3, 32, 32), dtype=np.float32)
    Y_train = np.random.randint(0,10,size=(N), dtype=np.int32)
    X_test, Y_test = X_train, Y_train
  else:
    X_train, Y_train, X_test, Y_test = fetch_cifar()

  # precompute whitening patches
  W = whitening(X_train.sequential(transform_test))

  model = SpeedyResNet(W)

  # parse the training params into bias and non-bias
  params_dict = get_state_dict(model)
  params_bias = []
  params_non_bias = []
  for params in params_dict:
    if params_dict[params].requires_grad is not False:
      if 'bias' in params:
        params_bias.append(params_dict[params])
      else:
        params_non_bias.append(params_dict[params])

  opt_bias     = optim.SGD(params_bias,     lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = 1e16
  final_lr_ratio = 0.07
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias     = OneCycleLR(opt_bias,     max_lr=hyp['opt']['bias_lr']     ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)
  lr_sched_non_bias = OneCycleLR(opt_non_bias, max_lr=hyp['opt']['non_bias_lr'] ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)

  loss_batchsize_scaler = 512/BS
  @TinyJit
  def train_step_jitted(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss = cross_entropy(out, Y, reduction='none' ,label_smoothing=0.2).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
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
    if i >= hyp['net']['cutmix_steps']: X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
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

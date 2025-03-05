#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM

import random, time, functools
import torch
import numpy as np
from torch import nn, optim
from tinygrad import getenv, Device, dtypes
from tinygrad.helpers import prod
from tinygrad.nn.datasets import cifar
import torch.nn.functional as F

from icecream import ic

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

BS, STEPS = getenv("BS", 512), getenv("STEPS", 1000)
EVAL_BS = getenv("EVAL_BS", BS)
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))]
assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"
assert EVAL_BS % len(GPUS) == 0, f"{EVAL_BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"

if getenv("TINY_BACKEND"): import tinygrad.frontend.torch
device = torch.device("tiny" if getenv("TINY_BACKEND") else "mps")

def sequential(ll, x):
  return functools.reduce(lambda x,f: f(x), ll, x)

class UnsyncedBatchNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=len(GPUS)):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    self.num_devices = num_devices

    if affine: self.weight, self.bias = torch.ones(sz, dtype=torch.float32), torch.zeros(sz, dtype=torch.float32)
    else: self.weight, self.bias = None, None

    self.running_mean = torch.zeros(num_devices, sz, dtype=torch.float32, requires_grad=False)
    self.running_var = torch.ones(num_devices, sz, dtype=torch.float32, requires_grad=False)
    self.num_batches_tracked = torch.zeros(1, dtype=torch.int, requires_grad=False)

  def forward(self, x:torch.Tensor):
    xr = x.reshape(self.num_devices, -1, *x.shape[1:]).to(torch.float32)
    batch_mean, batch_invstd = self.calc_stats(xr)
    ret = xr.batchnorm(
      self.weight.reshape(1, -1).expand((self.num_devices, -1)),
      self.bias.reshape(1, -1).expand((self.num_devices, -1)),
      batch_mean, batch_invstd, axis=(0, 2))
    return ret.reshape(x.shape).to(x.dtype)

  def calc_stats(self, x:torch.Tensor):
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(axis=(1,3,4))
      y = (x - batch_mean.detach().reshape(shape=[batch_mean.shape[0], 1, -1, 1, 1]))  # d(var)/d(mean) = 0
      batch_var = (y*y).mean(axis=(1,3,4))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().to(self.running_mean.dtype))
        batch_var_adjust = prod(y.shape[1:])/(prod(y.shape[1:])-y.shape[2])
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * batch_var_adjust * batch_var.detach().to(self.running_var.dtype))
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(self.running_var.shape[0], 1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
    return batch_mean, batch_invstd

class BatchNorm(nn.BatchNorm2d if getenv("SYNCBN") else UnsyncedBatchNorm):
  def __init__(self, num_features):
    super().__init__(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
    self.weight.requires_grad = False
    self.bias.requires_grad = True

class QuickGelu(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x): return x * nn.Sigmoid(x * 1.702)

class ConvGroup(nn.Module):
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.conv1 = nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
    self.max_pool2d = nn.MaxPool2d(2)
    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)
    self.quick_gelu = QuickGelu()

  def forward(self, x):
    x = self.conv1(x)
    x = self.max_pool2d(x)
    x = x.float()
    x = self.norm1(x)
    x = x.to(torch.float) # used to be float dtypes.default_float
    x = self.quick_gelu(x)
    residual = x
    x = self.conv2(x)
    x = x.float()
    x = self.norm2(x)
    x = x.to(torch.float) # used to be float dtypes.default_float
    x = self.quick_gelu(x)

    return x + residual

# replaces lambda x: x.max((2,3)), in SpeedyResNet.net
class Max(nn.Module):
  def __init__(self, dims):
    super().__init__()
    self.dims = dims
  def forward(self, x): return torch.amax(x, self.dims)

# replaces lambda x: x / 9., in SpeedyResNet.net
class Scale(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = scale
  def forward(self, x): return x * self.scale

class SpeedyResNet(nn.Module):
  def __init__(self, W):
    super().__init__()
    self.whitening = W
    self.net = nn.ModuleList([
      nn.Conv2d(12, 32, kernel_size=1, bias=False),
      nn.GELU(),
      ConvGroup(32, 64),
      ConvGroup(64, 256),
      ConvGroup(256, 512),
      Max((2,3)),
      nn.Linear(512, 10, bias=False),
      Scale(1/9.),
    ])

  def forward(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    # TODO: remove the pad but instead let the kernel optimize itself
    forward = lambda x: x.conv2d(self.whitening).pad((1,0,0,1)).sequential(self.net)
    return forward(x) if training else (forward(x) + forward(x[..., ::-1])) / 2.

# hyper-parameters were exactly the same as the original repo
bias_scaler = 58
hyp = {
  'seed' : 209,
  'opt': {
    'bias_lr':            1.76 * bias_scaler/512,
    'non_bias_lr':        1.76 / 512,
    'bias_decay':         1.08 * 6.45e-4 * BS/bias_scaler,
    'non_bias_decay':     1.08 * 6.45e-4 * BS,
    'final_lr_ratio':     0.025,
    'initial_div_factor': 1e6,
    'label_smoothing':    0.20,
    'momentum':           0.85,
    'percent_start':      0.23,
    'loss_scale_scaler':  1./128   # (range: ~1/512 - 16+, 1/128 w/ FP16)
  },
  'net': {
      'kernel_size': 2,             # kernel size for the whitening layer
      'cutmix_size': 3,
      'cutmix_steps': 499,
      'pad_amount': 2
  },
  'ema': {
      'steps': 399,
      'decay_base': .95,
      'decay_pow': 1.6,
      'every_n_steps': 5,
  },
}

def train_cifar():

  def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

  # ========== Model ==========
  def whitening(X, kernel_size=hyp['net']['kernel_size']):
    def _cov(X):
      return (X.T @ X) / (X.shape[0] - 1)

    def _patches(data, patch_size=(kernel_size,kernel_size)):
      h, w = patch_size
      c = data.shape[1]
      axis = (2, 3)
      return np.lib.stride_tricks.sliding_window_view(data, window_shape=(h,w), axis=axis).transpose((0,3,2,1,4,5)).reshape((-1,c,h,w))

    def _eigens(patches):
      n,c,h,w = patches.shape
      Σ = _cov(patches.reshape(n, c*h*w))
      Λ, V = np.linalg.eigh(Σ, UPLO='U')
      return np.flip(Λ, 0), np.flip(V.T.reshape(c*h*w, c, h, w), 0)

    # NOTE: np.linalg.eigh only supports float32 so the whitening layer weights need to be converted to float16 manually
    Λ, V = _eigens(_patches(X.float().cpu().numpy()))
    W = V/np.sqrt(Λ+1e-2)[:,None,None,None]

    return torch.tensor(W.astype(np.float32), requires_grad=False).to(torch.float) # used to be dtypes.default_float

  # ========== Loss ==========
  def cross_entropy(x:torch.Tensor, y:torch.Tensor, reduction:str='mean', label_smoothing:float=0.0) -> torch.Tensor:
    divisor = y.shape[1]
    assert isinstance(divisor, int), "only supported int divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='none': return ret
    if reduction=='sum': return ret.sum()
    if reduction=='mean': return ret.mean()
    raise NotImplementedError(reduction)

  # ========== Preprocessing ==========
  # NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
  def pad_reflect(X: torch.Tensor, size=2) -> torch.Tensor:
    X = torch.cat((X[...,:,1:size+1].flip(-1), X, X[...,:,-(size+1):-1].flip(-1)), dim=-1)
    X = torch.cat((X[...,1:size+1,:].flip(-2), X, X[...,-(size+1):-1,:].flip(-2)), dim=-2)
    return X

  # return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
  def make_square_mask(shape, mask_size) -> torch.Tensor:
    BS, _, H, W = shape
    low_x = torch.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
    low_y = torch.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
    idx_x = torch.arange(W, dtype=torch.int32).reshape((1,1,1,W))
    idx_y = torch.arange(H, dtype=torch.int32).reshape((1,1,H,1))
    return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

  def random_crop(X:torch.Tensor, crop_size=32):
    mask = make_square_mask(X.shape, crop_size)
    mask = mask.expand((-1,3,-1,-1))
    X_cropped = torch.Tensor(X.numpy()[mask.numpy()])
    return X_cropped.reshape((-1, 3, crop_size, crop_size))

  def cutmix(X:torch.Tensor, Y:torch.Tensor, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = torch.Tensor(X.numpy()[order], device=X.device, dtype=X.dtype)
    Y_patch = torch.Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
    X_cutmix = mask.where(X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in:torch.Tensor, Y_in:torch.Tensor, BS:int, is_train:bool):
    step, epoch = 0, 0
    while True:
      st = time.monotonic()
      X, Y = X_in, Y_in
      if is_train:
        # TODO: these are not jitted
        if getenv("RANDOM_CROP", 1):
          X = random_crop(X, crop_size=32)
        if getenv("RANDOM_FLIP", 1):
          X = (torch.rand(X.shape[0],1,1,1) < 0.5).where(X.flip(-1), X) # flip LR
        if getenv("CUTMIX", 1):
          if step >= hyp['net']['cutmix_steps']:
            X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
        order = list(range(0, X.shape[0]))
        random.shuffle(order)
        X, Y = X.numpy()[order], Y.numpy()[order]
      else:
        X, Y = X.numpy(), Y.numpy()
      et = time.monotonic()
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")
      for i in range(0, X.shape[0], BS):
        # pad the last batch  # TODO: not correct for test
        batch_end = min(i+BS, Y.shape[0])
        x = torch.tensor(X[batch_end-BS:batch_end], device=X_in.device, dtype=X_in.dtype)
        y = torch.tensor(Y[batch_end-BS:batch_end], device=Y_in.device, dtype=Y_in.dtype)
        step += 1
        yield x, y
      epoch += 1
      if not is_train: break

  transform = [
    lambda x: x.float() / 255.0,
    lambda x: x.reshape((-1,3,32,32)) - torch.tensor(cifar_mean, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
    lambda x: x / torch.tensor(cifar_std, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
  ]

  class modelEMA:
    def __init__(self, w, net):
      # self.model_ema = copy.deepcopy(net) # won't work for opencl due to unpickeable pyopencl._cl.Buffer
      self.net_ema = SpeedyResNet(w)
      for net_ema_param, net_param in zip(self.net_ema.state_dict().values(), net.state_dict().values()):
        net_ema_param.requires_grad = False
        net_ema_param.assign(net_param.numpy())

    # @TinyJit
    def update(self, net, decay):
      # TODO with Tensor.no_grad()
      Tensor.no_grad = True
      for net_ema_param, (param_name, net_param) in zip(self.net_ema.state_dict().values(), net.state_dict().items()):
        # batchnorm currently is not being tracked
        if not ("num_batches_tracked" in param_name) and not ("running" in param_name):
          net_ema_param.assign(net_ema_param.detach()*decay + net_param.detach()*(1.-decay)).realize()
      Tensor.no_grad = False

  set_seed(getenv('SEED', hyp['seed']))

  # download data
  X_train, Y_train, X_test, Y_test = cifar()
  X_train = torch.tensor(X_train.half().numpy(), device=device) # was X_train.float() but 'double' is not supported in Metal
  Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy(), device=device)
  X_test = torch.tensor(X_test.half().numpy(), device=device)
  Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy(), device=device)

  # one-hot encode labels
  Y_train, Y_test = F.one_hot(Y_train, 10), F.one_hot(Y_test, 10)
  # preprocess data
  X_train, X_test = sequential(transform, X_train), sequential(transform, X_test)

  # precompute whitening patches
  W = whitening(X_train) # on cpu

  # initialize model weights
  model = SpeedyResNet(W)

  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  # Convert data and labels to the default dtype
  X_train, Y_train = X_train.to(torch.float), Y_train.to(torch.float) # used to be default_float
  X_test, Y_test = X_test.to(torch.float), Y_test.to(torch.float)

  # # does not work on torch!
  # if len(GPUS) > 1:
  #   for k, x in model.state_dict().items():
  #     if not getenv('SYNCBN') and ('running_mean' in k or 'running_var' in k):
  #       x.shard_(GPUS, axis=0)
  #     else:
  #       x.to_(GPUS)
  model.to(device)

  # parse the training params into bias and non-bias
  params_dict = model.state_dict()
  params_bias = []
  params_non_bias = []
  for params in params_dict:
    ic(params, params_dict[params].requires_grad)
    if params_dict[params].requires_grad is not False:
      if 'bias' in params:
        params_bias.append(params_dict[params])
      else:
        params_non_bias.append(params_dict[params])

  ic(params_bias, params_non_bias)

  opt_bias = optim.SGD(params_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])


if __name__ == "__main__":
  train_cifar()

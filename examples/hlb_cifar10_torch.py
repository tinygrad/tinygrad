#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM

import random, time, functools
from typing import Optional
import torch
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from tinygrad import getenv, Device, dtypes, GlobalCounters
from tinygrad.helpers import prod, colored
from tinygrad.nn.datasets import cifar
import torch.nn.functional as F

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

BS, STEPS = getenv("BS", 512), getenv("STEPS", 1000)
EVAL_BS = getenv("EVAL_BS", BS)
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))]
assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"
assert EVAL_BS % len(GPUS) == 0, f"{EVAL_BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"

if getenv("TINY_BACKEND"): import tinygrad.frontend.torch
device = torch.device("tiny") if getenv("TINY_BACKEND") else torch.device("cpu")

def sequential(ll, x): return functools.reduce(lambda x,f: f(x), ll, x)

def quick_gelu(x): return x * F.sigmoid(x * 1.702)

class UnsyncedBatchNorm(nn.Module):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=len(GPUS)):
    super().__init__()
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    self.num_devices = num_devices

    if affine: self.weight, self.bias = nn.Parameter(torch.ones(sz, dtype=torch.float32)), nn.Parameter(torch.zeros(sz, dtype=torch.float32))
    else: self.weight, self.bias = None, None

    self.register_buffer('running_mean', torch.zeros(num_devices, sz, dtype=torch.float32, requires_grad=False))
    self.register_buffer('running_var', torch.ones(num_devices, sz, dtype=torch.float32, requires_grad=False))
    self.register_buffer('num_batches_tracked', torch.zeros(1, dtype=torch.int, requires_grad=False))

  def forward(self, x:torch.Tensor):
    xr = x.reshape(self.num_devices, -1, *x.shape[1:]).to(torch.float32)
    batch_mean, batch_invstd = self.calc_stats(xr)
    weight = self.weight.reshape(1, -1).expand((self.num_devices, -1))
    bias = self.bias.reshape(1, -1).expand((self.num_devices, -1))
    axis_ = (0,2)
    # compute batchnorm (in tinygrad: ret = xr.batchnorm(weight, bias, batch_mean, batch_invstd, axis=(0, 2)))
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(xr.shape))
    x_norm = xr - batch_mean.reshape(shape)
    x_norm = x_norm * weight.reshape(shape)
    ret = x_norm.mul(batch_invstd.reshape(shape) if len(batch_invstd.shape) == len(axis_) else batch_invstd)
    ret = (ret + bias.reshape(shape))
    return ret.reshape(x.shape).to(x.dtype)

  def calc_stats(self, x:torch.Tensor):
    if self.training:
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
  def __init__(self):
    super().__init__()
    self.sigmoid = nn.Sigmoid()
  def forward(self, x): return x * self.sigmoid(x * 1.702)

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
    x = x.to(torch.float)
    x = self.quick_gelu(x)
    residual = x
    x = self.conv2(x)
    x = x.float()
    x = self.norm2(x)
    x = x.to(torch.float)
    x = self.quick_gelu(x)
    return x + residual

class WhiteningConv(nn.Conv2d):
  def __init__(self, whitening):
    shape = whitening.shape
    super().__init__(shape[1], shape[0], shape[2:], bias=False)
    self.weight = nn.Parameter(whitening, requires_grad=False)

class SpeedyResNet(nn.Module):
  def __init__(self, W):
    super().__init__()
    self.whitening = W
    self.whitening_conv = WhiteningConv(self.whitening)
    self.conv2d = nn.Conv2d(12, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    self.conv_group_1 = ConvGroup(32, 64)
    self.conv_group_2 = ConvGroup(64, 256)
    self.conv_group_3 = ConvGroup(256, 512)
    self.linear = nn.Linear(512, 10, bias=False)

  def _forward(self, x):
    x = self.whitening_conv(x)
    x = F.pad(x, (1,0,0,1))
    x = self.conv2d(x)
    x = quick_gelu(x)
    x = self.conv_group_1(x)
    x = self.conv_group_2(x)
    x = self.conv_group_3(x)
    x = torch.amax(x, (2,3))
    x = self.linear(x)
    x = x / 9.
    return x

  def forward(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    # TODO: remove the pad but instead let the kernel optimize itself
    # torch does not support negative indexing so must use flip
    return self._forward(x) if training else (self._forward(x) + self._forward(torch.flip(x, (-1,)))) / 2.

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
    from tinygrad import Tensor
    Tensor.manual_seed(seed)
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

  # ========== Preprocessing ==========
  # NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
  def pad_reflect(X: torch.Tensor, size=2) -> torch.Tensor:
    X = torch.cat((X[...,:,1:size+1].flip(-1), X, X[...,:,-(size+1):-1].flip(-1)), dim=-1)
    X = torch.cat((X[...,1:size+1,:].flip(-2), X, X[...,-(size+1):-1,:].flip(-2)), dim=-2)
    return X

  # return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
  def make_square_mask(shape, mask_size, device) -> torch.Tensor:
    BS, _, H, W = shape
    low_x = torch.randint(low=0, high=W-mask_size, size=(BS,)).reshape(BS,1,1,1)
    low_y = torch.randint(low=0, high=H-mask_size, size=(BS,)).reshape(BS,1,1,1)
    idx_x = torch.arange(W, dtype=torch.int32).reshape((1,1,1,W))
    idx_y = torch.arange(H, dtype=torch.int32).reshape((1,1,H,1))
    mask = (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))
    ret = mask.to(device)
    return ret

  def random_crop(X:torch.Tensor, crop_size=32):
    mask = make_square_mask(X.shape, crop_size, X.device)
    mask = mask.expand((-1,3,-1,-1))
    X_cropped = X[mask] #torch.Tensor(X.numpy()[mask.numpy()])
    return X_cropped.reshape((-1, 3, crop_size, crop_size))

  def cutmix(X:torch.Tensor, Y:torch.Tensor, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size, X.device)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = X[order].to(device=X.device, dtype=X.dtype)
    Y_patch = Y[order].to(device=Y.device, dtype=Y.dtype)
    X_cutmix = torch.where(mask, X_patch, X)
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
          X = torch.where(torch.rand(X.shape[0],1,1,1, device=X.device) < 0.5, X.flip(-1), X) # flip LR
        if getenv("CUTMIX", 1):
          if step >= hyp['net']['cutmix_steps']:
            X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
        order = list(range(0, X.shape[0]))
        random.shuffle(order)
        X, Y = X[order], Y[order]
      et = time.monotonic()
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")
      for i in range(0, X.shape[0], BS):
        # pad the last batch  # TODO: not correct for test
        batch_end = min(i+BS, Y.shape[0])
        # Todo: confirm we don't need to clone X,Y here
        x, y = X[batch_end-BS:batch_end], Y[batch_end-BS:batch_end]
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

  # with open("cifar.safetensor", "w") as f:
  # torch.save(model.state_dict(), "cifar.safetensor")
  state_dict = torch.load("cifar.safetensor", weights_only=False)
  model.load_state_dict(state_dict)

  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  # Convert data and labels to the default dtype
  X_train, Y_train = X_train.to(torch.float), Y_train.to(torch.float) # used to be default_float
  X_test, Y_test = X_test.to(torch.float), Y_test.to(torch.float)

  model.to(device)
  model.train()

  # parse the training params into bias and non-bias
  params_bias = []
  params_non_bias = []
  for name, param in model.named_parameters():
    if param.requires_grad is not False:
      if 'bias' in name:
        params_bias.append(param)
      else:
        params_non_bias.append(param)

  opt_bias = optim.SGD(params_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = hyp['opt']['initial_div_factor']
  final_lr_ratio = hyp['opt']['final_lr_ratio']
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias     = OneCycleLR(opt_bias, max_lr=hyp['opt']['bias_lr'], pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS, anneal_strategy='linear', cycle_momentum=False)
  lr_sched_non_bias = OneCycleLR(opt_non_bias,  max_lr=hyp['opt']['non_bias_lr'], pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS, anneal_strategy='linear', cycle_momentum=False)

  train_loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=hyp['opt']['label_smoothing'])
  #@torch.compile
  def train_step(model, optimizers, lr_schedulers, X, Y):
    out = model(X)
    loss_batchsize_scaler = 512/BS
    loss = train_loss_fn(out, Y).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
      for optimizer in optimizers: optimizer.zero_grad()
      loss.backward()
      for optimizer, lr_scheduler in zip(optimizers, lr_schedulers):
        optimizer.step()
        lr_scheduler.step()
    return loss

  eval_loss_fn = nn.CrossEntropyLoss(reduction='mean')
  #@torch.compile
  def eval_step(model, X, Y):
    out = model(X, training=False)
    loss = eval_loss_fn(out, Y)
    correct = out.argmax(axis=1) == Y.argmax(axis=1)
    return correct, loss

  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
  i = 0
  eval_acc_pct = 0.0
  batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True)
  st = time.monotonic()
  while i <= STEPS:
    if i % getenv("EVAL_STEPS", STEPS) == 0 and i > 1 and not getenv("DISABLE_BACKWARD"):
      # Use Tensor.training = False here actually bricks batchnorm, even with track_running_stats=True
      corrects = []
      corrects_ema = []
      losses = []
      losses_ema = []
      for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
        with torch.no_grad():
          correct, loss = eval_step(model, Xt, Yt) # eval_step_jitted
          losses.append(loss.detach().cpu().numpy().tolist())
          corrects.extend(correct.detach().cpu().numpy().tolist())
          if model_ema:
            correct_ema, loss_ema = eval_step(model_ema.net_ema, Xt, Yt) # eval_step_ema_jitted
            losses_ema.append(loss_ema.detach().cpu().numpy().tolist())
            corrects_ema.extend(correct_ema.detach().cpu().numpy().tolist())

      # collect accuracy across ranks
      correct_sum, correct_len = sum(corrects), len(corrects)
      if model_ema: correct_sum_ema, correct_len_ema = sum(corrects_ema), len(corrects_ema)

      eval_acc_pct = correct_sum/correct_len*100.0
      if model_ema: acc_ema = correct_sum_ema/correct_len_ema*100.0
      print(f"eval     {correct_sum}/{correct_len} {eval_acc_pct:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i} (in {(time.monotonic()-st)*1e3:.2f} ms)")
      if model_ema: print(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {(sum(losses_ema)/len(losses_ema)):7.2f} val_loss STEP={i}")

    if STEPS == 0 or i == STEPS: break

    GlobalCounters.reset()
    X, Y = next(batcher)
    loss = train_step(model, [opt_bias, opt_non_bias], [lr_sched_bias, lr_sched_non_bias], X, Y) # train_step_jitted
    et = time.monotonic()

    # EMA for network weights
    if getenv("EMA") and i > hyp['ema']['steps'] and (i+1) % hyp['ema']['every_n_steps'] == 0:
      if model_ema is None:
        model_ema = modelEMA(W, model)
      model_ema.update(model, torch.tensor([projected_ema_decay_val*(i/STEPS)**hyp['ema']['decay_pow']]))
    cl = time.monotonic()
    device_str, loss_cpu, non_bias_lr = str(loss.device), loss.detach().cpu().numpy(), opt_non_bias.param_groups[0]['lr']
    #  53  221.74 ms run,    2.22 ms python,  219.52 ms CL,  803.39 loss, 0.000807 LR, 4.66 GB used,   3042.49 GFLOPS,    674.65 GOPS
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms {device_str}, {loss_cpu:7.2f} loss, {non_bias_lr:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS, {GlobalCounters.global_ops*1e-9:9.2f} GOPS")
    st = cl
    i += 1

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if eval_acc_pct >= target:
      print(colored(f"{eval_acc_pct=} >= {target}", "green"))
    else:
      raise ValueError(colored(f"{eval_acc_pct=} < {target}", "red"))


if __name__ == "__main__":
  train_cifar()

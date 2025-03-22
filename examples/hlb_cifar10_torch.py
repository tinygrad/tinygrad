#!/usr/bin/env python3

# re-implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py that matches examples/hlb_cifar10.py
import random, time
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tinygrad.device import Device
from tinygrad.helpers import getenv, colored

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

BS, STEPS = getenv("BS", 512), getenv("STEPS", 1000)
EVAL_BS = getenv("EVAL_BS", BS)

if getenv("TINYBACKEND", 0):
  import tinygrad.frontend.torch
  from tinygrad import GlobalCounters
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  device = torch.device("tiny")
else:
  GPUS = [f"cuda:{i}" for i in range(getenv("GPUS", 1))]
  device = torch.device(GPUS[0].split(':')[0] if len(GPUS) > 0 and torch.cuda.is_available() else "cpu")

assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"
assert EVAL_BS % len(GPUS) == 0, f"{EVAL_BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"

def quick_gelu(x:torch.tensor): return x * F.sigmoid(x * 1.702)

class UnsyncedBatchNorm(nn.Module):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=len(GPUS)):
    super().__init__()
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    self.num_devices = num_devices

    if affine:
      self.weight = nn.Parameter(torch.ones(sz))
      self.bias = nn.Parameter(torch.zeros(sz))
    else:
      self.register_parameter("weight", None)
      self.register_parameter("bias", None)

    self.register_buffer("running_mean", torch.zeros(num_devices, sz))
    self.register_buffer("running_var", torch.ones(num_devices, sz))
    self.register_buffer("num_batches_tracked", torch.zeros(1, dtype=torch.long))

  def forward(self, x:torch.tensor):
    xr = x.reshape(self.num_devices, -1, *x.shape[1:])
    batch_mean, batch_invstd = self.calc_stats(xr)
    weight = self.weight.reshape(1, -1).expand(self.num_devices, -1)
    bias = self.bias.reshape(1, -1).expand(self.num_devices, -1)

    # batchnorm
    axis_ = (0,2)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(xr.shape))
    x_norm = xr - batch_mean.reshape(shape)
    x_norm = x_norm * weight.reshape(shape)
    ret = x_norm.mul(batch_invstd.reshape(shape) if len(batch_invstd.shape) == len(axis_) else batch_invstd)
    ret = (ret + bias.reshape(shape))
    return ret.reshape(x.shape).to(x.dtype)

  def calc_stats(self, x:torch.tensor):
    if self.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(dim=(1, 3, 4))
      y = (x - batch_mean.detach().reshape(batch_mean.shape[0], 1, -1, 1, 1))
      batch_var = (y*y).mean(dim=(1, 3, 4))
      batch_invstd = (batch_var + self.eps).pow(-0.5)

      if self.track_running_stats:
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
        batch_var_adjust = np.prod(y.shape[1:]) / (np.prod(y.shape[1:]) - y.shape[2])
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_adjust * batch_var.detach()
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = (self.running_var.reshape(self.running_var.shape[0], 1, -1, 1, 1).expand(x.shape)+self.eps).rsqrt()
    return batch_mean, batch_invstd

class BatchNorm(UnsyncedBatchNorm if not getenv("SYNCBN") else nn.BatchNorm2d):
  def __init__(self, num_features):
    super().__init__(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
    self.weight.requires_grad = False
    self.bias.requires_grad = True

class ConvGroup(nn.Module):
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

  def forward(self, x):
    x = self.conv1(x)
    x = F.max_pool2d(x, 2)
    x = self.norm1(x.float())
    x = quick_gelu(x)
    residual = x
    x = self.conv2(x)
    x = self.norm2(x.float())
    x = quick_gelu(x)

    return x + residual

class SpeedyResNet(nn.Module):
  def __init__(self, W):
    super().__init__()
    self.whitening = nn.Parameter(W, requires_grad=False)

    self.conv1 = nn.Conv2d(12, 32, kernel_size=1, bias=False)
    self.conv_group1 = ConvGroup(32, 64)
    self.conv_group2 = ConvGroup(64, 256)
    self.conv_group3 = ConvGroup(256, 512)
    self.fc = nn.Linear(512, 10, bias=False)

  def forward(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    x = F.conv2d(x, self.whitening)
    x = F.pad(x, (1,0,0,1))
    x = self.conv1(x)
    x = quick_gelu(x)
    x = self.conv_group1(x)
    x = self.conv_group2(x)
    x = self.conv_group3(x)
    x = torch.amax(x, dim=(2,3))
    x = self.fc(x)
    x = x / 9.

    return x if training else (x + self.forward(x[..., :, ::-1], training=True)) / 2.

# Keep the original hyper-parameters
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
    'kernel_size': 2,       # kernel size for the whitening layer
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
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

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
    W = V/np.sqrt(np.maximum(0,Λ)+1e-2)[:,None,None,None]

    return torch.tensor(W.astype(np.float32), requires_grad=False)

  # ========== Loss ==========
  def cross_entropy(x:torch.tensor, y:torch.tensor, reduction:str='mean', label_smoothing:float=0.0) -> torch.tensor:
    divisor = y.shape[1]
    assert isinstance(divisor, int), "only supported int divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    ret = -F.log_softmax(x, dim=1) * y
    ret = ret.sum(dim=1)
    if reduction == 'none': return ret
    if reduction == 'sum': return ret.sum()
    if reduction == 'mean': return ret.mean()
    raise NotImplementedError(reduction)

  # ========== Preprocessing ==========
  def pad_reflect(X, size=2):
    return F.pad(X, (size, size, size, size), mode='reflect')

  # return a binary mask in the format of BS x C x H x W with a random square mask
  def make_square_mask(shape, mask_size, device) -> torch.tensor:
    BS, _, H, W = shape
    low_x = torch.randint(0, W-mask_size, (BS, 1, 1, 1), device=device)
    low_y = torch.randint(0, H-mask_size, (BS, 1, 1, 1), device=device)
    idx_x = torch.arange(W, dtype=torch.int32, device=device).reshape((1, 1, 1, W))
    idx_y = torch.arange(H, dtype=torch.int32, device=device).reshape((1, 1, H, 1))
    return ((idx_x >= low_x) & (idx_x < (low_x + mask_size)) &
        (idx_y >= low_y) & (idx_y < (low_y + mask_size)))

  def random_crop(X:torch.tensor, crop_size=32):
    mask = make_square_mask(X.shape, crop_size, X.device)
    mask = mask.expand((-1,3,-1,-1))
    X_cropped = X[mask] #torch.Tensor(X.numpy()[mask.numpy()])
    return X_cropped.reshape((-1, 3, crop_size, crop_size))

  def cutmix(X, Y, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = X[order].to(X.device)
    Y_patch = Y[order].to(Y.device)
    X_cutmix = torch.where(mask.expand(-1, X.shape[1], -1, -1), X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in, Y_in, BS, is_train):
    step, epoch = 0, 0
    device = X_in.device
    while True:
      st = time.monotonic()
      X, Y = X_in, Y_in
      if is_train:
        if getenv("RANDOM_CROP", 1):
          X = random_crop(X, crop_size=32)
        if getenv("RANDOM_FLIP", 1):
          X = torch.where((torch.rand(X.shape[0], 1, 1, 1, device=device) < 0.5), X.flip(-1), X)  # flip LR
        if getenv("CUTMIX", 1):
          if step >= hyp['net']['cutmix_steps']:
            X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
        order = torch.randperm(X.shape[0], device=device)
        X, Y = X[order], Y[order]
      et = time.monotonic()
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")
      for i in range(0, X.shape[0], BS):
        batch_end = min(i+BS, Y.shape[0])
        if batch_end - i < BS:  # pad the last batch if needed
          x = X[batch_end-BS:batch_end]
          y = Y[batch_end-BS:batch_end]
        else:
          x = X[i:batch_end]
          y = Y[i:batch_end]
        step += 1
        yield x, y
      epoch += 1
      if not is_train: break

  # transform function instead of list
  def apply_transforms(x):
    x = x.float() / 255.0
    x = x.reshape(-1,3,32,32) - torch.tensor(cifar_mean, device=x.device).reshape(1,3,1,1)
    x = x / torch.tensor(cifar_std, device=x.device).reshape(1, 3, 1, 1)
    return x

  class modelEMA:
    def __init__(self, w, net):
      self.net_ema = SpeedyResNet(w)
      self.net_ema.to(next(net.parameters()).device)
      with torch.no_grad():
        for ema_param, net_param in zip(self.net_ema.parameters(), net.parameters()):
          ema_param.copy_(net_param)
      for param in self.net_ema.parameters():
        param.requires_grad = False

    def update(self, net, decay):
      with torch.no_grad():
        for ema_param, net_param in zip(self.net_ema.parameters(), net.parameters()):
          ema_param.copy_(ema_param*decay + net_param*(1.-decay))

  set_seed(getenv('SEED', hyp['seed']))

  transform = transforms.Compose([transforms.ToTensor()])

  train_dataset = datasets.CIFAR10(root='extra/datasets/cifar-10-python.tar.gz', train=True, download=True, transform=transform)
  test_dataset = datasets.CIFAR10(root='extra/datasets/cifar-10-python.tar.gz', train=False, download=True, transform=transform)

  X_train = torch.stack([sample[0] for sample in train_dataset])
  Y_train = F.one_hot(torch.tensor([sample[1] for sample in train_dataset]), 10).float()
  X_test = torch.stack([sample[0] for sample in test_dataset])
  Y_test = F.one_hot(torch.tensor([sample[1] for sample in test_dataset]), 10).float()

  X_train, X_test = apply_transforms(X_train), apply_transforms(X_test)

  X_train, Y_train = X_train.to(device), Y_train.to(device)
  X_test, Y_test = X_test.to(device), Y_test.to(device)

  # precompute whitening patches
  W = whitening(X_train).to(device)

  # initialize model weights
  model = SpeedyResNet(W).to(device)

  # pad data
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  if len(GPUS) > 1 and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=range(len(GPUS)))

  # parse the training params into bias and non-bias
  params_bias = []
  params_non_bias = []
  for name, param in model.named_parameters():
    if param.requires_grad:
      if 'bias' in name:
        params_bias.append(param)
      else:
        params_non_bias.append(param)

  opt_bias     = SGD(params_bias,     lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
  opt_non_bias = SGD(params_non_bias, lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['non_bias_decay'])

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = hyp['opt']['initial_div_factor']
  final_lr_ratio = hyp['opt']['final_lr_ratio']
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias = OneCycleLR(
    opt_bias, max_lr=hyp['opt']['bias_lr'],
    pct_start=pct_start, div_factor=initial_div_factor,
    final_div_factor=1./(initial_div_factor*final_lr_ratio),
    total_steps=STEPS
  )
  lr_sched_non_bias = OneCycleLR(
    opt_non_bias, max_lr=hyp['opt']['non_bias_lr'],
    pct_start=pct_start, div_factor=initial_div_factor,
    final_div_factor=1./(initial_div_factor*final_lr_ratio),
    total_steps=STEPS
  )

  def train_step(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss_batchsize_scaler = 512/BS
    loss = cross_entropy(
      out, Y, reduction='none',
      label_smoothing=hyp['opt']['label_smoothing']
    ).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

    if not getenv("DISABLE_BACKWARD"):
      opt_bias.zero_grad()
      opt_non_bias.zero_grad()
      loss.backward()
      opt_bias.step()
      opt_non_bias.step()
      lr_sched_bias.step()
      lr_sched_non_bias.step()
    return loss

  def eval_step(model, X, Y):
    model.eval()
    with torch.no_grad():
      out = model(X, training=False)
      loss = cross_entropy(out, Y, reduction='mean')
      correct = (out.argmax(dim=1) == Y.argmax(dim=1))
    model.train()
    return correct, loss

  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
  i = 0
  eval_acc_pct = 0.0
  batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True)
  model.train()
  st = time.monotonic()
  while i <= STEPS:
    if i % getenv("EVAL_STEPS", STEPS) == 0 and i > 1 and not getenv("DISABLE_BACKWARD"):
      corrects = []
      corrects_ema = []
      losses = []
      losses_ema = []
      for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
        if len(GPUS) > 1 and torch.cuda.device_count() > 1:
          Xt = Xt.to(device)
          Yt = Yt.to(device)

        correct, loss = eval_step(model, Xt, Yt)
        losses.append(loss.item())
        corrects.extend(correct.cpu().numpy().tolist())
        if model_ema:
          model_ema.net_ema.eval()
          with torch.no_grad():
            out_ema = model_ema.net_ema(Xt, training=False)
            loss_ema = cross_entropy(out_ema, Yt, reduction='mean')
            correct_ema = (out_ema.argmax(dim=1) == Yt.argmax(dim=1))
          model_ema.net_ema.train()
          losses_ema.append(loss_ema.item())
          corrects_ema.extend(correct_ema.cpu().numpy().tolist())

      # collect accuracy across ranks
      correct_sum, correct_len = sum(corrects), len(corrects)
      if model_ema:
        correct_sum_ema, correct_len_ema = sum(corrects_ema), len(corrects_ema)

      eval_acc_pct = correct_sum/correct_len*100.0
      print(f"eval   {correct_sum}/{correct_len} {eval_acc_pct:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i} (in {(time.monotonic()-st)*1e3:.2f} ms)")

      if model_ema:
        acc_ema = correct_sum_ema/correct_len_ema*100.0
        print(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {(sum(losses_ema)/len(losses_ema)):7.2f} val_loss STEP={i}")

    if STEPS == 0 or i == STEPS: break

    X, Y = next(batcher)
    if len(GPUS) > 1 and torch.cuda.device_count() > 1:
      X = X.to(device)
      Y = Y.to(device)

    # Train step with timing
    torch.cuda.synchronize()
    start_time = time.monotonic()

    loss = train_step(model, None, None, X, Y)

    torch.cuda.synchronize()
    et = time.monotonic()
    loss_cpu = loss.item()

    # EMA for network weights
    if getenv("EMA") and i > hyp['ema']['steps'] and (i+1) % hyp['ema']['every_n_steps'] == 0:
      if model_ema is None:
        model_ema = modelEMA(W, model)
      decay = projected_ema_decay_val * (i/STEPS)**hyp['ema']['decay_pow']
      model_ema.update(model, decay)
    cl = time.monotonic()
    device_str = f"{device}" if len(GPUS) <= 1 else f"{device} * {len(GPUS)}"

    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms {device_str}, {loss_cpu:7.2f} loss, {opt_non_bias.param_groups[0]['lr']:.6f} LR, ", end="")
    if getenv("TINYBACKEND", 0):
      print(f"{GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS, {GlobalCounters.global_ops*1e-9:9.2f} GOPS")
    else:
      print(f"{(torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0):.2f} GB used")

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

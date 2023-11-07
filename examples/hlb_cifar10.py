#!/usr/bin/env python3
# setup for distributed
from extra import dist
from tinygrad.helpers import getenv, dtypes
if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import random, time
import numpy as np
from typing import Any, Dict, Optional, SupportsIndex, Type, Union
from extra.datasets import fetch_cifar, cifar_mean, cifar_std
from tinygrad import nn
from tinygrad.nn.state import get_state_dict
from tinygrad.nn import optim
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.shape.symbolic import Node
from extra.lr_scheduler import OneCycleLR
from tinygrad.jit import TinyJit
from extra.dist import collectives

BS, EVAL_BS, STEPS = getenv("BS", 512), getenv('EVAL_BS', 500), getenv("STEPS", 1000)

if getenv("HALF", 0):
  Tensor.default_type = dtypes.float16
  np_dtype: Type[Union[np.float16, np.float32]] = np.float16
else:
  Tensor.default_type = dtypes.float32
  np_dtype = np.float32

class BatchNorm(nn.BatchNorm2d):
  def __init__(self, num_features):
    super().__init__(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
    self.weight.requires_grad = False
    self.bias.requires_grad = True

class ConvGroup:
  def __init__(self, channels_in, channels_out):
    self.conv1 = nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

  def __call__(self, x):
    x = self.conv1(x)
    x = x.max_pool2d(2)
    x = x.float()
    x = self.norm1(x)
    x = x.cast(Tensor.default_type)
    x = x.gelu()
    residual = x
    x = self.conv2(x)
    x = x.float()
    x = self.norm2(x)
    x = x.cast(Tensor.default_type)
    x = x.gelu()

    return x + residual

class SpeedyResNet:
  def __init__(self, W):
    self.whitening = W
    self.net = [
      nn.Conv2d(12, 32, kernel_size=1, bias=False),
      lambda x: x.gelu(),
      ConvGroup(32, 64),
      ConvGroup(64, 256),
      ConvGroup(256, 512),
      lambda x: x.max((2,3)),
      nn.Linear(512, 10, bias=False),
      lambda x: x.mul(1./9)
    ]

  def __call__(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    # TODO: remove the pad but instead let the kernel optimizer itself
    forward = lambda x: x.conv2d(self.whitening).pad2d((1,0,0,1)).sequential(self.net)
    return forward(x) if training else forward(x)*0.5 + forward(x[..., ::-1])*0.5

def train_cifar():

  # hyper-parameters were exactly the same as the original repo
  bias_scaler = 58
  hyp: Dict[str, Any] = {
    'seed' : 209,
    'opt': {
      'bias_lr':            1.76 * bias_scaler/512,
      'non_bias_lr':        1.76 / 512,
      'bias_decay':         1.08 * 6.45e-4 * BS/bias_scaler,
      'non_bias_decay':     1.08 * 6.45e-4 * BS,
      'final_lr_ratio':     0.025,
      'initial_div_factor': 1e16,
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
    }
  }

  def set_seed(seed):
    Tensor.manual_seed(getenv('SEED', seed))
    random.seed(getenv('SEED', seed))

  # ========== Model ==========
  # NOTE: np.linalg.eigh only supports float32 so the whitening layer weights need to be converted to float16 manually
  def whitening(X, kernel_size=hyp['net']['kernel_size']):
    def _cov(X):
      X = X/np.sqrt(X.shape[0] - 1)
      return X.T @ X

    def _patches(data, patch_size=(kernel_size,kernel_size)):
      h, w = patch_size
      c = data.shape[1]
      axis: SupportsIndex = (2, 3) # type: ignore
      return np.lib.stride_tricks.sliding_window_view(data, window_shape=(h,w), axis=axis).transpose((0,3,2,1,4,5)).reshape((-1,c,h,w))

    def _eigens(patches):
      n,c,h,w = patches.shape
      Σ = _cov(patches.reshape(n, c*h*w))
      Λ, V = np.linalg.eigh(Σ, UPLO='U')
      return np.flip(Λ, 0), np.flip(V.T.reshape(c*h*w, c, h, w), 0)

    Λ, V = _eigens(_patches(X.numpy()))
    W = V/np.sqrt(Λ+1e-2)[:,None,None,None]

    return Tensor(W.astype(np_dtype), requires_grad=False)

  # ========== Loss ==========
  def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    divisor = y.shape[1]
    assert not isinstance(divisor, Node), "sint not supported as divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
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

  # return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
  def make_square_mask(shape, mask_size) -> Tensor:
    is_even = int(mask_size % 2 == 0)
    center_max = shape[-2]-mask_size//2-is_even
    center_min = mask_size//2-is_even
    center_x = (Tensor.rand(shape[0])*(center_max-center_min)+center_min).floor()
    center_y = (Tensor.rand(shape[0])*(center_max-center_min)+center_min).floor()
    d_x = Tensor.arange(0, shape[-1]).reshape((1,1,1,shape[-1])) - center_x.reshape((-1,1,1,1))
    d_y = Tensor.arange(0, shape[-2]).reshape((1,1,shape[-2],1)) - center_y.reshape((-1,1,1,1))
    d_x =(d_x >= -(mask_size // 2) + is_even) * (d_x <= mask_size // 2)
    d_y =(d_y >= -(mask_size // 2) + is_even) * (d_y <= mask_size // 2)
    mask = d_y * d_x
    return mask

  def random_crop(X:Tensor, crop_size=32):
    mask = make_square_mask(X.shape, crop_size)
    mask = mask.repeat((1,3,1,1))
    X_cropped = Tensor(X.flatten().numpy()[mask.flatten().numpy().astype(bool)])
    return X_cropped.reshape((-1, 3, crop_size, crop_size))

  def cutmix(X:Tensor, Y:Tensor, mask_size=3):
    # fill the square with randomly selected images from the same batch
    mask = make_square_mask(X.shape, mask_size)
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = Tensor(X.numpy()[order,...])
    Y_patch = Tensor(Y.numpy()[order])
    X_cutmix = Tensor.where(mask, X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in:Tensor, Y_in:Tensor, BS:int, is_train:bool):
    step, cnt = 0, 0
    while True:
      st = time.monotonic()
      X, Y = X_in, Y_in
      order = list(range(0, X.shape[0]))
      random.shuffle(order)
      if is_train:
        X = random_crop(X, crop_size=32)
        X = Tensor.where(Tensor.rand(X.shape[0],1,1,1) < 0.5, X[..., ::-1], X) # flip LR
        if step >= hyp['net']['cutmix_steps']: X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
      X, Y = X.numpy(), Y.numpy()
      et = time.monotonic()
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({cnt})")
      for i in range(0, X.shape[0], BS):
        # pad the last batch
        batch_end = min(i+BS, Y.shape[0])
        x = Tensor(X[order[batch_end-BS:batch_end],:])
        y = Tensor(Y[order[batch_end-BS:batch_end]])
        step += 1
        yield x, y
      cnt += 1
      if not is_train: break

  transform = [
    lambda x: x / 255.0,
    lambda x: (x.reshape((-1,3,32,32)) - Tensor(cifar_mean).reshape((1,3,1,1)))/Tensor(cifar_std).reshape((1,3,1,1))
  ]

  class modelEMA():
    def __init__(self, w, net):
      # self.model_ema = copy.deepcopy(net) # won't work for opencl due to unpickeable pyopencl._cl.Buffer
      self.net_ema = SpeedyResNet(w)
      for net_ema_param, net_param in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).values()):
        net_ema_param.requires_grad = False
        net_ema_param.assign(net_param.numpy())

    @TinyJit
    def update(self, net, decay):
      # TODO with Tensor.no_grad()
      Tensor.no_grad = True
      for net_ema_param, (param_name, net_param) in zip(get_state_dict(self.net_ema).values(), get_state_dict(net).items()):
        # batchnorm currently is not being tracked
        if not ("num_batches_tracked" in param_name) and not ("running" in param_name):
          net_ema_param.assign(net_ema_param.detach()*decay + net_param.detach()*(1.-decay)).realize()
      Tensor.no_grad = False

  set_seed(hyp['seed'])

  # this import needs to be done here because this is running in a subprocess
  from extra.dist import OOB
  assert OOB is not None or not getenv("DIST"), "OOB should be initialized"
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE", 1)

  X_train, Y_train, X_test, Y_test = fetch_cifar()
  # load data and label into GPU and convert to dtype accordingly
  X_train, X_test = X_train.to(device=Device.DEFAULT).float(), X_test.to(device=Device.DEFAULT).float()
  Y_train, Y_test = Y_train.to(device=Device.DEFAULT).float(), Y_test.to(device=Device.DEFAULT).float()
  # one-hot encode labels
  Y_train, Y_test = Tensor.eye(10)[Y_train], Tensor.eye(10)[Y_test]
  # preprocess data
  X_train, X_test = X_train.sequential(transform), X_test.sequential(transform)

  # precompute whitening patches
  W = whitening(X_train)

  # initialize model weights
  model = SpeedyResNet(W)

  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  # Convert data and labels to the default dtype
  X_train, Y_train, X_test, Y_test = X_train.cast(Tensor.default_type), Y_train.cast(Tensor.default_type), X_test.cast(Tensor.default_type), Y_test.cast(Tensor.default_type)

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
  initial_div_factor = hyp['opt']['initial_div_factor']
  final_lr_ratio = hyp['opt']['final_lr_ratio']
  pct_start = hyp['opt']['percent_start']
  lr_sched_bias     = OneCycleLR(opt_bias,     max_lr=hyp['opt']['bias_lr']     ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)
  lr_sched_non_bias = OneCycleLR(opt_non_bias, max_lr=hyp['opt']['non_bias_lr'] ,pct_start=pct_start, div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=STEPS)

  loss_batchsize_scaler = 512/BS
  @TinyJit
  def train_step_jitted(model, optimizer, lr_scheduler, X, Y):
    out = model(X)
    loss = cross_entropy(out, Y, reduction='none' ,label_smoothing=hyp['opt']['label_smoothing']).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
      optimizer[0].zero_grad()
      optimizer[1].zero_grad()
      loss.backward()

      if getenv("DIST"):
        # sync gradients across ranks
        bucket, offset = [], 0
        for _, v in params_dict.items():
          if v.grad is not None: bucket.append(v.grad.flatten())
        grads = collectives.allreduce(Tensor.cat(*bucket), cache_id="grads")
        for _, v in params_dict.items():
          if v.grad is not None:
            v.grad.assign(grads[offset:offset+v.grad.numel()].reshape(*v.grad.shape))
            offset += v.grad.numel()

      optimizer[0].step()
      optimizer[1].step()
      lr_scheduler[0].step()
      lr_scheduler[1].step()
    return loss.realize()

  def eval_step(model, X, Y):
    out = model(X, training=False)
    loss = cross_entropy(out, Y, reduction='mean')
    correct = out.argmax(axis=1) == Y.argmax(axis=1)
    return correct.realize(), loss.realize()
  eval_step_jitted     = TinyJit(eval_step)
  eval_step_ema_jitted = TinyJit(eval_step)

  # 97 steps in 2 seconds = 20ms / step
  # step is 1163.42 GOPS = 56 TFLOPS!!!, 41% of max 136
  # 4 seconds for tfloat32 ~ 28 TFLOPS, 41% of max 68
  # 6.4 seconds for float32 ~ 17 TFLOPS, 50% of max 34.1
  # 4.7 seconds for float32 w/o channels last. 24 TFLOPS. we get 50ms then i'll be happy. only 64x off

  # https://www.anandtech.com/show/16727/nvidia-announces-geforce-rtx-3080-ti-3070-ti-upgraded-cards-coming-in-june
  # 136 TFLOPS is the theoretical max w float16 on 3080 Ti

  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
  i = 0
  batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True)
  with Tensor.train():
    st = time.monotonic()
    while i <= STEPS:
      if i%getenv("EVAL_STEPS", STEPS) == 0 and i > 1:
        st_eval = time.monotonic()
        # Use Tensor.training = False here actually bricks batchnorm, even with track_running_stats=True
        corrects = []
        corrects_ema = []
        losses = []
        losses_ema = []
        for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
          # further split batch if distributed
          if getenv("DIST"):
            Xt, Yt = Xt.chunk(min(world_size, 5), 0)[min(rank, 4)], Yt.chunk(min(world_size, 5), 0)[min(rank, 4)]

          correct, loss = eval_step_jitted(model, Xt, Yt)
          losses.append(loss.numpy().tolist())
          corrects.extend(correct.numpy().tolist())
          if model_ema:
            correct_ema, loss_ema = eval_step_ema_jitted(model_ema.net_ema, Xt, Yt)
            losses_ema.append(loss_ema.numpy().tolist())
            corrects_ema.extend(correct_ema.numpy().tolist())

        # collect accuracy across ranks
        correct_sum, correct_len = sum(corrects), len(corrects)
        if model_ema: correct_sum_ema, correct_len_ema = sum(corrects_ema), len(corrects_ema)
        if getenv("DIST"):
          if rank == 0:
            for j in range(1, min(world_size, 5)):
              if model_ema:
                recv_sum, recv_len, recv_sum_ema, recv_len_ema = OOB.recv(j)
              else:
                recv_sum, recv_len = OOB.recv(j)
              correct_sum += recv_sum
              correct_len += recv_len
              if model_ema:
                correct_sum_ema += recv_sum_ema
                correct_len_ema += recv_len_ema
          elif rank < min(world_size, 5):
            if model_ema:
              OOB.send((correct_sum, correct_len, correct_sum_ema, correct_len_ema), 0)
            else:
              OOB.send((correct_sum, correct_len), 0)

        # only rank 0 prints
        if rank == 0:
          acc = correct_sum/correct_len*100.0
          if model_ema: acc_ema = correct_sum_ema/correct_len_ema*100.0
          print(f"eval     {correct_sum}/{correct_len} {acc:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i} (in {(time.monotonic()-st)*1e3:.2f} ms)")
          if model_ema: print(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {(sum(losses_ema)/len(losses_ema)):7.2f} val_loss STEP={i}")

      if STEPS == 0 or i==STEPS: break
      X, Y = next(batcher)
      if getenv("DIST"):
        X, Y = X.chunk(world_size, 0)[rank], Y.chunk(world_size, 0)[rank]
      GlobalCounters.reset()
      loss = train_step_jitted(model, [opt_bias, opt_non_bias], [lr_sched_bias, lr_sched_non_bias], X, Y)
      et = time.monotonic()
      loss_cpu = loss.numpy()
      # EMA for network weights
      if i > hyp['ema']['steps'] and (i+1) % hyp['ema']['every_n_steps'] == 0:
        if model_ema is None:
          model_ema = modelEMA(W, model)
        model_ema.update(model, Tensor([projected_ema_decay_val*(i/STEPS)**hyp['ema']['decay_pow']]))
      cl = time.monotonic()
      if not getenv("DIST"):
        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {opt_non_bias.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      else:
        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {opt_non_bias.lr.numpy()[0]:.6f} LR, {world_size*GlobalCounters.mem_used/1e9:.2f} GB used, {world_size*GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      st = cl
      i += 1

if __name__ == "__main__":
  if not getenv("DIST"):
    train_cifar()
  else: # distributed
    if getenv("HIP"):
      from tinygrad.runtime.ops_hip import HIP
      devices = [f"hip:{i}" for i in range(HIP.device_count)]
    else:
      from tinygrad.runtime.ops_gpu import CL
      devices = [f"gpu:{i}" for i in range(len(CL.devices))]
    world_size = len(devices)

    # ensure that the batch size is divisible by the number of devices
    assert BS % world_size == 0, f"batch size {BS} is not divisible by world size {world_size}"

    # ensure that the evaluation batch size is divisible by the number of devices
    assert EVAL_BS % min(world_size, 5) == 0, f"evaluation batch size {EVAL_BS} is not divisible by world size {min(world_size, 5)}"

    # init out-of-band communication
    dist.init_oob(world_size)

    # start the processes
    processes = []
    for rank, device in enumerate(devices):
      processes.append(dist.spawn(rank, device, fn=train_cifar, args=()))
    for p in processes: p.join()

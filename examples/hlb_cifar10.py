#!/usr/bin/env python3
# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import time
import numpy as np
from extra.datasets import fetch_cifar
from tinygrad import nn
from tinygrad.state import get_parameters, get_state_dict
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.ops import GlobalCounters
from extra.lr_scheduler import OneCycleLR
from tinygrad.jit import TinyJit
from copy import deepcopy


def set_seed(seed):
  Tensor.manual_seed(getenv('SEED', seed)) # Deterministic
  np.random.seed(getenv('SEED', seed))

num_classes = 10
HALF = getenv('HALF', 1) == 1
LOSS_SCALE = getenv('LOSS_SCALE', 1) 

cifar10_mean = Tensor(np.array([125.306918046875, 122.950394140625, 113.86538318359375], dtype=np.float16 if HALF else np.float32).reshape(1,3,1,1))
cifar10_std = Tensor(np.array([62.993219278136884, 62.08870764001421, 66.70489964063091], dtype=np.float16 if HALF else np.float32).reshape(1,3,1,1))

def change_dtype(layers, dtype):
  revert = False
  if not isinstance(layers, list): layers, revert = [layers], True
  for layer in layers:
    if hasattr(layer, '__dict__'):
      for attr, value in layer.__dict__.items():
        if isinstance(value, Tensor) and value.dtype != dtype:
          if value.grad is not None and value.grad.dtype != dtype: value.grad = value.grad.cast(dtype).realize()
          layer.__dict__[attr] = value.cast(dtype).realize()
        value = change_dtype(value, dtype)
  return layers[0] if revert else layers

def copy_weights(from_model, to_model):
  from_model_state = get_state_dict(from_model)
  to_model_state = get_state_dict(to_model)
  for k,v in from_model_state.items():
    # print(f"copying {k}, from {from_model_state[k].dtype} to {to_model_state[k].dtype}")
    to_model_state[k].assign(Tensor(v.cast(to_model_state[k].dtype).numpy()).realize())
    to_model_state[k].requires_grad = from_model_state[k].requires_grad
    if v.grad is not None:
      # print(f"copying {k}.grad, from {from_model_state[k].dtype} to {to_model_state[k].dtype}")
      if to_model_state[k].grad is None: to_model_state[k].grad = Tensor(v.grad.cast(to_model_state[k].dtype).numpy(), requires_grad=False).realize()
      to_model_state[k].grad.assign(Tensor(v.grad.cast(to_model_state[k].dtype).numpy(), requires_grad=False).realize())

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
    mult = self.se2((self.se1(residual.mean((2,3)))).relu()).sigmoid().reshape(x.shape[0], x.shape[1], 1, 1) if self.se else 1.0
    x = self.norm[1](self.conv[1](x)).relu()
    x = self.norm[2](self.conv[2](x) * mult).relu()
    return x + residual

class SpeedyResNet:
  def __init__(self):
    # TODO: add whitening
    BASE_DIM=getenv("BASE_DIM", 64)
    self.net = [
      nn.Conv2d(3, BASE_DIM, kernel_size=1),
      nn.BatchNorm2d(BASE_DIM, track_running_stats=False, eps=1e-12, momentum=0.8),
      lambda x: x.relu(),
      ConvGroup(BASE_DIM, BASE_DIM*2, short=False),
      ConvGroup(BASE_DIM*2, BASE_DIM*4, short=True),
      ConvGroup(BASE_DIM*4, BASE_DIM*8, short=False),
      lambda x: x.max((2,3)),
      nn.Linear(BASE_DIM*8, num_classes, bias=False)
    ]

  # note, pytorch just uses https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html instead of log_softmax
  def __call__(self, x, training=True): 
    if not training and getenv('TTA', 0)==1: return ((x.sequential(self.net) * 0.5) + (x[..., ::-1].sequential(self.net) * 0.5)).log_softmax()
    return x.sequential(self.net).log_softmax()

def fetch_batches(all_X, all_Y, BS, seed, is_train=False):
  def _shuffle(all_X, all_Y):
    if is_train:
      ind = np.arange(all_Y.shape[0])
      np.random.shuffle(ind)
      all_X, all_Y = all_X[ind, ...], all_Y[ind, ...]
    return all_X, all_Y
  while True:
    set_seed(seed)
    all_X, all_Y = _shuffle(all_X, all_Y)
    for batch_start in range(0, all_Y.shape[0], BS):
      batch_end = min(batch_start+BS, all_Y.shape[0])
      X = all_X[batch_end-BS:batch_end] # batch_end-BS for padding
      Y = np.zeros((BS, num_classes), np.float16 if HALF else np.float32)
      Y[range(BS),all_Y[batch_end-BS:batch_end]] = -1.0*num_classes
      Y = Y.reshape(BS, num_classes)
      yield X, Y
    if not is_train: break
    seed += 1


def train_cifar(bs=512, eval_bs=500, steps=1000, div_factor=1e16, final_lr_ratio=0.001, max_lr=0.01, pct_start=0.0546875, momentum=0.8, wd=0.15, label_smoothing=0., mixup_alpha=0.025, seed=6):
  set_seed(seed)
  Tensor.training = True

  BS, EVAL_BS, STEPS = getenv("BS", bs), getenv('EVAL_BS', eval_bs), getenv("STEPS", steps)
  MAX_LR, PCT_START, MOMENTUM, WD = getenv("MAX_LR", max_lr), getenv('PCT_START', pct_start), getenv('MOMENTUM', momentum), getenv("WD", wd)
  DIV_FACTOR, LABEL_SMOOTHING, MIXUP_ALPHA = getenv('DIV_FACTOR', div_factor), getenv('LABEL_SMOOTHING', label_smoothing), getenv('MIXUP_ALPHA', mixup_alpha)
  FINAL_DIV_FACTOR = 1./(DIV_FACTOR*getenv('FINAL_LR_RATIO', final_lr_ratio))
  if getenv("FAKEDATA"):
    N = 2048
    X_train = np.random.default_rng().standard_normal(size=(N, 3, 32, 32), dtype=np.float32)
    Y_train = np.random.randint(0,10,size=(N), dtype=np.int32)
    X_test, Y_test = X_train, Y_train
  else:
    X_train, Y_train = fetch_cifar(train=True)
    X_test, Y_test = fetch_cifar(train=False)

  model_fp32 = SpeedyResNet()
  model_fp16 = change_dtype(SpeedyResNet(), dtypes.float16 if HALF else dtypes.float32)

  optimizer = optim.SGD(get_parameters(model_fp32), lr=0.01, momentum=MOMENTUM, nesterov=True, weight_decay=WD)
  lr_scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, div_factor=DIV_FACTOR, final_div_factor=FINAL_DIV_FACTOR, 
                            total_steps=STEPS, pct_start=PCT_START)

  def train_step(model_fp16, model_fp32, optimizer, lr_scheduler, Xr, Xl, Yr, Yl, mixup_prob):
    Xr = (Xr - cifar10_mean) / cifar10_std
    Xl = (Xl - cifar10_mean) / cifar10_std
    X, Y = Xr*mixup_prob + Xl*(1-mixup_prob), Yr*mixup_prob + Yl*(1-mixup_prob)
    X = Tensor.where(Tensor.rand(X.shape[0],1,1,1, dtype=X.dtype) < 0.5, X[..., ::-1], X) # flip augmentation
    copy_weights(model_fp32, model_fp16)
    out = model_fp16(X)
    loss = (1 - LABEL_SMOOTHING) * out.mul(Y).mean() + (-1 * LABEL_SMOOTHING * out.mean())
    loss = loss/LOSS_SCALE
    if not getenv("DISABLE_BACKWARD"):
      optimizer.init_params(get_parameters(model_fp16))
      optimizer.zero_grad() # Zero grad needs to be called with fp16 params
      optimizer.init_params(get_parameters(model_fp32))
      loss.backward()
      copy_weights(model_fp16, model_fp32)
      optimizer.step()
      copy_weights(model_fp32, model_fp16) # TODO: is this needed?
      lr_scheduler.step()
    loss = loss * LOSS_SCALE
    return loss.realize()
  
  # JIT at every run
  @TinyJit 
  def train_step_jitted(model_fp16, model_fp32, optimizer, lr_scheduler, Xr, Xl, Yr, Yl, mixup_prob):
    return train_step(model_fp16, model_fp32, optimizer, lr_scheduler, Xr, Xl, Yr, Yl, mixup_prob)
  
  def eval_step(model_fp32, X, Y):
    out = model_fp32(X, training=False)
    loss = out.mul(Y).mean()
    return out.realize(), loss.realize()

  @TinyJit
  def eval_step_jitted(model_fp32, X, Y):
    return eval_step(model_fp32, X, Y)
  
  # 97 steps in 2 seconds = 20ms / step
  # step is 1163.42 GOPS = 56 TFLOPS!!!, 41% of max 136
  # 4 seconds for tfloat32 ~ 28 TFLOPS, 41% of max 68
  # 6.4 seconds for float32 ~ 17 TFLOPS, 50% of max 34.1
  # 4.7 seconds for float32 w/o channels last. 24 TFLOPS. we get 50ms then i'll be happy. only 64x off

  # https://www.anandtech.com/show/16727/nvidia-announces-geforce-rtx-3080-ti-3070-ti-upgraded-cards-coming-in-june
  # 136 TFLOPS is the theoretical max w float16 on 3080 Ti
  best_eval = -1
  i = 0
  left_batcher, right_batcher = fetch_batches(X_train, Y_train, BS=BS, seed=seed, is_train=True), fetch_batches(X_train, Y_train, BS=BS, seed=seed+1, is_train=True)
  while i <= STEPS:
    mixup_prob = Tensor(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA, (1, )).astype(np.float16 if HALF else np.float32)).contiguous() if MIXUP_ALPHA > 0 else Tensor.ones(Xr.shape[0], 1, 1, 1, dtype=dtypes.float16 if HALF else dtypes.float32).contiguous()
    if i%50 == 0 and i > 1:
      # batchnorm is frozen, no need for Tensor.training=False
      corrects = []
      losses = []
      eval_func = eval_step_jitted if getenv('JIT', 1) == 1 else eval_step
      for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, seed=seed):
        Xt, Yt = Tensor(Xt), Tensor(Yt) 
        out, loss = eval_func(model_fp32, Xt, Yt)
        outs = out.numpy().argmax(axis=1)
        correct = outs == Yt.numpy().argmin(axis=1)
        losses.append(loss.numpy().tolist())
        corrects.extend(correct.tolist())
      acc = sum(corrects)/len(corrects)*100.0
      if acc > best_eval:
        best_eval = acc
        print(f"eval {sum(corrects)}/{len(corrects)} {acc:.2f}%, {(sum(losses)/len(losses)):7.2f} val_loss STEP={i}")
    if STEPS == 0 or i==STEPS: break
    # TODO: JIT is broken with mixed precision training 
    train_func = train_step_jitted if getenv('JIT', 1) == 1 else train_step
    GlobalCounters.reset()
    st = time.monotonic()
    (Xr, Yr), (Xl, Yl) = next(right_batcher), next(left_batcher)
    Xr, Xl, Yr, Yl = Tensor(Xr), Tensor(Xl), Tensor(Yr), Tensor(Yl)
    lt = time.monotonic()
    loss = train_func(model_fp16, model_fp32, optimizer, lr_scheduler, Xr, Xl, Yr, Yl, mixup_prob)
    et = time.monotonic()
    loss_cpu = loss.numpy() 
    cl = time.monotonic()
    print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(lt-st)*1000.0:7.2f} ms copy-in, {(et-lt)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {loss.dtype} dtype, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    i += 1

if __name__ == "__main__":
  train_cifar()

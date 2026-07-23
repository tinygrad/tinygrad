#!/usr/bin/env python3
# tinygrad implementation of airbench94: https://github.com/KellerJordan/cifar10-airbench (https://arxiv.org/abs/2404.00498)
# trains CIFAR-10 to ~94% in 9.9 epochs / 476 steps, run with DEFAULT_FLOAT=HALF
import time
start_tm = time.perf_counter()
import math
import numpy as np
from tinygrad import Tensor, nn, dtypes, TinyJit, Variable
from tinygrad.helpers import getenv, colored, Context
from tinygrad.nn import optim
from extra.bench_log import BenchEvent, WallTimeEvent
import_tm = time.perf_counter()

BS           = getenv("BS", 1024)
EPOCHS       = getenv("EPOCHS", 9.9)
EVAL_BS      = getenv("EVAL_BS", 2000)
SEED         = getenv("SEED", 1337)
LOSS_SCALE   = getenv("LOSS_SCALE", 1/32)  # mathematically neutral, keeps the summed fp16 loss in range
LOG_INTERVAL = getenv("LOG_INTERVAL", 100)
TTA          = getenv("TTA", 2)  # test-time aug: 2=mirror+translate (6 views), 1=mirror (2), 0=none (1)

# hyperparameters from airbench94, lr/wd are given in decoupled "per 1024 examples" form and converted below
MOMENTUM, BIAS_SCALER, LABEL_SMOOTHING, WHITEN_BIAS_EPOCHS, EMA_EVERY, EMA_BASE = 0.85, 64.0, 0.2, 3, 5, 0.95
kilostep_scale = 1024 * (1 + 1 / (1 - MOMENTUM))
lr = 11.5 / kilostep_scale
wd = 0.0153 * BS / kilostep_scale
lr_bias = lr * BIAS_SCALER

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

act = Tensor.gelu if getenv("GELU") else Tensor.quick_gelu

def whitening(X:Tensor, kernel_size=2, eps=5e-4) -> Tensor:
  """frozen whitening conv weights: eigenvectors of the input patch covariance scaled by 1/sqrt(eigenvalue), and their negations"""
  def _patches(data:Tensor, patch_size=(kernel_size, kernel_size)):
    h, w = patch_size
    c = data.shape[1]
    return data._pool((h, w)).permute(1, 4, 5, 0, 3, 2).reshape(c*h*w, -1)
  patches = _patches(X.float())
  cov = ((patches @ patches.T) / patches.shape[1]).numpy()
  eigvals, eigvecs = np.linalg.eigh(cov, UPLO='U')
  eigvecs = np.flip(eigvecs.T.reshape(-1, X.shape[1], kernel_size, kernel_size), 0)
  eigvecs_scaled = eigvecs / np.sqrt(np.flip(eigvals, 0) + eps)[:, None, None, None]
  return Tensor(np.concatenate([eigvecs_scaled, -eigvecs_scaled]).astype(np.float32)).cast(dtypes.default_float).contiguous().is_param_(False)

def dirac_init(w:Tensor) -> Tensor:
  """airbench identity init: the first in_channels filters form an identity kernel, the rest keep the default init"""
  cout, cin, kh, kw = w.shape
  # NOTE: built from numpy so the weight is buffer-backed, a pure-const tensor is not a valid optimizer param
  eye = np.zeros((cin, cin, kh, kw), dtype=np.float32)
  eye[range(cin), range(cin), kh//2, kw//2] = 1.0
  eye_t = Tensor(eye).cast(w.dtype)
  return (eye_t if cout == cin else eye_t.cat(w[cin:], dim=0)).contiguous()

class BatchNorm(nn.BatchNorm):
  """airbench batchnorm: eps 1e-12, batch stats only, kept in fp32, scale frozen at 1, bias trainable"""
  def __init__(self, sz:int):
    super().__init__(sz, track_running_stats=False, eps=1e-12, momentum=0.4)
    self.weight = Tensor.ones(sz, dtype=dtypes.float32).is_param_(False)
    self.bias = Tensor.zeros(sz, dtype=dtypes.float32).contiguous()

class MatmulConv2d(nn.Conv2d):
  # im2col conv-as-GEMM: BEAM applies correct tensor cores to matmuls, unlike direct-conv backward (which miscompiles)
  def __call__(self, x:Tensor) -> Tensor:
    if not getenv("MATMUL_CONV", 0): return super().__call__(x)
    bs, cin, _, _ = x.shape; cout, _, ky, kx = self.weight.shape
    p = x.pad((1, 1, 1, 1))._pool((ky, kx)); oy, ox = p.shape[2:4]
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(bs*oy*ox, cin*ky*kx)
    return (p @ self.weight.reshape(cout, cin*ky*kx).T).reshape(bs, oy, ox, cout).permute(0, 3, 1, 2)

class ConvGroup:
  def __init__(self, channels_in:int, channels_out:int):
    self.conv1 = MatmulConv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = MatmulConv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
    self.conv1.weight, self.conv2.weight = dirac_init(self.conv1.weight), dirac_init(self.conv2.weight)
    self.norm1, self.norm2 = BatchNorm(channels_out), BatchNorm(channels_out)
  def __call__(self, x:Tensor) -> Tensor:
    # batchnorms are computed in fp32 islands, idiom from hlb_cifar10
    # NOTE: .contiguous() forces the conv into its own kernel; without it BEAM fuses conv+pool+bn and miscompiles
    c = (lambda t: t.contiguous()) if getenv("CONTIG", 1) else (lambda t: t)
    # NOTE: .contiguous_backward() materializes the grad at the act input: the bn bias grad becomes a cheap reduce
    # instead of a mega-fused kernel that recomputes the whole transposed conv (6ms -> ~1ms)
    cb = (lambda t: t.contiguous_backward()) if getenv("CBW", 1) else (lambda t: t)
    x = act(cb(self.norm1(c(self.conv1(x)).max_pool2d(2).float()).cast(dtypes.default_float)))
    return act(cb(self.norm2(c(self.conv2(x)).float()).cast(dtypes.default_float)))

class CifarNet:
  def __init__(self, whiten_weight:Tensor):
    self.whiten = nn.Conv2d(3, whiten_weight.shape[0], kernel_size=2, bias=True)
    self.whiten.weight = whiten_weight  # frozen
    self.whiten.bias = Tensor.zeros(whiten_weight.shape[0], dtype=dtypes.default_float).contiguous()  # trained for the first 3 epochs
    self.group1, self.group2, self.group3 = ConvGroup(whiten_weight.shape[0], 64), ConvGroup(64, 256), ConvGroup(256, 256)
    self.linear = nn.Linear(256, 10, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    # pad to 32x32 because the whitening conv creates 31x31 images that are awfully slow to compute with, hack from hlb_cifar10
    # NOTE: .contiguous() isolates each matmul/conv into its own kernel so BEAM's WMMA opt doesn't fuse with a downstream reduce and miscompile
    c = (lambda t: t.contiguous()) if getenv("CONTIG", 1) else (lambda t: t)
    cb = (lambda t: t.contiguous_backward()) if getenv("CBW", 1) else (lambda t: t)
    x = c(act(cb(self.whiten(x)))).pad((1, 0, 0, 1))
    x = x.sequential([self.group1, self.group2, self.group3])
    return self.linear(c(x.max((2, 3)))) / 9.

# NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
def pad_reflect(X:Tensor, size=2) -> Tensor:
  X = X[..., :, 1:size+1].flip(-1).cat(X, X[..., :, -(size+1):-1].flip(-1), dim=-1)
  X = X[..., 1:size+1, :].flip(-2).cat(X, X[..., -(size+1):-1, :].flip(-2), dim=-2)
  return X

def random_crop(X:Tensor, crop_size=32) -> Tensor:
  low_x = Tensor.randint(X.shape[0], low=0, high=X.shape[-1]-crop_size+1).reshape(-1, 1, 1, 1)
  low_y = Tensor.randint(X.shape[0], low=0, high=X.shape[-2]-crop_size+1).reshape(-1, 1, 1, 1)
  idx_x = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, 1, crop_size)
  idx_y = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, crop_size, 1)
  X = X.gather(-1, (low_x + idx_x).expand(X.shape[0], X.shape[1], X.shape[2], crop_size))
  return X.gather(-2, (low_y + idx_y).expand(X.shape[0], X.shape[1], crop_size, crop_size))

class TriangularLR:
  """airbench schedule: warmup 0.2x -> 1x over the first 23% of steps, then anneal 1x -> 0.07x.
  from freeze_step on, lr is exactly 0, which freezes the params (used for the whiten bias after 3 epochs)"""
  def __init__(self, opt:optim.Optimizer, base_lr:float, total_steps:int, freeze_step:int=0):
    self.opt, self.base_lr, self.warmup, self.total, self.freeze = opt, base_lr, int(0.23 * total_steps), total_steps, freeze_step
    self.counter = Tensor([0], dtype=dtypes.float32, device=opt.device)
    opt.lr.assign(self.get_lr())
  def get_lr(self) -> Tensor:
    mult = (self.counter < self.warmup).where(0.2 + 0.8*self.counter/self.warmup, 1.0 - 0.93*(self.counter - self.warmup)/(self.total - self.warmup))
    if self.freeze: mult = mult * (self.counter < self.freeze)
    return (self.base_lr * mult).cast(self.opt.lr.dtype)
  def schedule_step(self) -> list[Tensor]: return [self.counter.assign(self.counter + 1), self.opt.lr.assign(self.get_lr())]

def jit_now(j:TinyJit) -> TinyJit:
  j.cnt = 1  # skip the uncaptured warmup call, capture on the first call (the kernels are all in the warm cache anyway)
  return j

class Lookahead:
  """airbench lookahead/EMA: every 5 steps ema = decay*ema + (1-decay)*net, and the net is pulled back onto the ema"""
  def __init__(self, params:list[Tensor]):
    self.params, self.ema = params, [p.detach().clone().is_param_(False) for p in params]
  @jit_now
  @TinyJit
  def update(self, decay:Tensor):
    for p, e in zip(self.params, self.ema):
      e.assign((decay*e.float() + (1-decay)*p.detach().float()).cast(e.dtype))
      p.assign(e)
    Tensor.realize(*self.ema, *self.params)

if __name__ == "__main__":
  with WallTimeEvent(BenchEvent.FULL):
    Tensor.manual_seed(SEED)

    # *** data ***
    X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
    assert X_test.shape[0] % EVAL_BS == 0, f"{EVAL_BS=} must divide {X_test.shape[0]}"
    mean, std = Tensor(cifar_mean, dtype=dtypes.float32).reshape(1, 3, 1, 1), Tensor(cifar_std, dtype=dtypes.float32).reshape(1, 3, 1, 1)
    def normalize(x:Tensor) -> Tensor: return ((x.float()/255 - mean)/std).cast(dtypes.default_float)
    X_train, X_test = normalize(X_train), normalize(X_test)
    Y_train, Y_test = Y_train.cast(dtypes.int32), Y_test.cast(dtypes.int32)
    Tensor.realize(X_train, Y_train, X_test, Y_test)
    data_tm = time.perf_counter()

    # *** model and optimizer ***
    steps_per_epoch = X_train.shape[0] // BS  # 48, drop_last
    total_steps = math.ceil(steps_per_epoch * EPOCHS)
    model = CifarNet(whitening(X_train[:5000]))

    params = nn.state.get_state_dict(model)
    params_norm_bias = [v for k, v in params.items() if v.is_param and 'norm' in k]  # batchnorm biases (the scales are frozen)
    params_main = [v for k, v in params.items() if v.is_param and 'norm' not in k and k != 'whiten.bias']
    opt_main = optim.SGD(params_main, lr=lr, momentum=MOMENTUM, nesterov=True, weight_decay=wd/lr)
    opt_norm_bias = optim.SGD(params_norm_bias, lr=lr_bias, momentum=MOMENTUM, nesterov=True, weight_decay=wd/lr_bias)
    opt_whiten_bias = optim.SGD([model.whiten.bias], lr=lr, momentum=MOMENTUM, nesterov=True, weight_decay=wd/lr)
    opt = optim.OptimizerGroup(opt_main, opt_norm_bias, opt_whiten_bias)
    schedulers = [TriangularLR(opt_main, lr, total_steps), TriangularLR(opt_norm_bias, lr_bias, total_steps),
                  TriangularLR(opt_whiten_bias, lr, total_steps, freeze_step=math.ceil(steps_per_epoch * WHITEN_BIAS_EPOCHS))]
    Tensor.realize(*params.values(), *[t for o in opt.optimizers for t in (*o.b, o.lr)], *[s.counter for s in schedulers])
    ema = Lookahead(opt.params)
    # augmentation caches: one random 50% pre-flip + 2px reflect padding (train), 1px reflect padding (test TTA)
    X_train = pad_reflect((Tensor.rand(X_train.shape[0], 1, 1, 1) < 0.5).where(X_train.flip(-1), X_train), size=2).contiguous()
    X_test = pad_reflect(X_test, size=1).contiguous()
    Tensor.realize(X_train, X_test, *ema.ema)
    init_tm = time.perf_counter()

    # *** training ***
    @TinyJit
    @Context(TRAINING=1)
    def train_step(X:Tensor, Y:Tensor, off:Variable) -> Tensor:
      X, Y = X[off:off+BS], Y[off:off+BS]
      loss = model(X).sparse_categorical_crossentropy(Y, label_smoothing=LABEL_SMOOTHING, reduction='none').mul(LOSS_SCALE).sum().div(LOSS_SCALE)
      opt.zero_grad()
      loss.backward()
      return loss.realize(*opt.schedule_step(), *[t for s in schedulers for t in s.schedule_step()])

    @jit_now
    @TinyJit
    def epoch_aug(X:Tensor, Y:Tensor, flip:Tensor) -> tuple[Tensor, Tensor]:
      X = random_crop(X, crop_size=32)
      # alternating flip: every image is mirrored on odd epochs, the random pre-flip gives each image its own phase
      # NOTE: RANGEIFY=1 needs this contiguous or the X[perm] is very slow
      X = flip.where(X.flip(-1), X).contiguous()
      perm = Tensor.randperm(X.shape[0], device=X.device)
      return X[perm], Y[perm]

    vi = Variable("i", 0, (steps_per_epoch - 1) * BS)
    i = 0
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
      Xa, Ya = epoch_aug(X_train, Y_train, Tensor([epoch % 2 == 1]))
      for j in range(min(steps_per_epoch, total_steps - i)):
        loss = train_step(Xa, Ya, vi.bind(j * BS))
        i += 1
        if i % EMA_EVERY == 0: ema.update(Tensor([EMA_BASE**EMA_EVERY * (i/total_steps)**3], dtype=dtypes.float32))
        if LOG_INTERVAL and (i % LOG_INTERVAL == 0 or i == total_steps): print(f"step {i:4d}/{total_steps}, loss {loss.item():8.2f}")
    ema.update(Tensor([1.0], dtype=dtypes.float32))  # the final pullback is just a decay=1.0 update
    train_tm = time.perf_counter()

    # *** evaluation: airbench TTA (TTA=2: mirror+1px translations 6 views; TTA=1: mirror 2 views; TTA=0: 1 view) ***
    @jit_now
    @TinyJit
    @Context(TRAINING=1)  # batchnorm uses eval-batch stats (there are no running stats), like hlb_cifar10
    def eval_step(x:Tensor, y:Tensor, off:Variable) -> Tensor:
      x, y = x[off:off+EVAL_BS], y[off:off+EVAL_BS]
      def infer_mirror(z:Tensor) -> Tensor: return 0.5*model(z) + 0.5*model(z.flip(-1)) if TTA else model(z)
      if TTA >= 2: logits = 0.5*infer_mirror(x[:, :, 1:33, 1:33]) + 0.25*(infer_mirror(x[:, :, 0:32, 0:32]) + infer_mirror(x[:, :, 2:34, 2:34]))
      else: logits = infer_mirror(x[:, :, 1:33, 1:33])
      return (logits.argmax(axis=1) == y).cast(dtypes.int32).sum()

    vj = Variable("j", 0, X_test.shape[0] - EVAL_BS)
    correct = Tensor.zeros(1, dtype=dtypes.int32).contiguous().realize()
    for j in range(0, X_test.shape[0], EVAL_BS): correct.assign(correct + eval_step(X_test, Y_test, vj.bind(j))).realize()
    num_correct, num_test = correct.item(), X_test.shape[0]
    eval_acc_pct = 100.0 * num_correct / num_test
    eval_tm = time.perf_counter()

    print(f"eval {num_correct}/{num_test} {eval_acc_pct:.2f}%")
    print(f"ACC: {eval_acc_pct:.2f}")
    print(f"import {import_tm - start_tm:.2f}s, data {data_tm - import_tm:.2f}s, init {init_tm - data_tm:.2f}s, "
          f"train {train_tm - init_tm:.2f}s ({(train_tm - init_tm)/total_steps*1e3:.2f} ms/step), "
          f"eval {eval_tm - train_tm:.2f}s, total {time.perf_counter() - start_tm:.2f}s")

    if (target := getenv("TARGET_EVAL_ACC_PCT", 0.0)):
      if eval_acc_pct >= target: print(colored(f"{eval_acc_pct=} >= {target}", "green"))
      else: raise ValueError(colored(f"{eval_acc_pct=} < {target}", "red"))

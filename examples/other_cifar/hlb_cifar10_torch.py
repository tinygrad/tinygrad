import random, time
import numpy as np
from typing import Optional, Tuple

from tinygrad import dtypes, Tensor, Device, GlobalCounters
from tinygrad.helpers import Context, BEAM, WINO, getenv, colored, DEBUG
from tinygrad.nn.datasets import cifar
from extra.bench_log import BenchEvent, WallTimeEvent
import torch
from torch import nn, optim
from torch.nn import functional as F

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

BS, STEPS = getenv("BS", 512), getenv("STEPS", 1000)
EVAL_BS = getenv("EVAL_BS", BS)
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"
assert EVAL_BS % len(GPUS) == 0, f"{EVAL_BS=} is not a multiple of {len(GPUS)=}, uneven multi GPU is slow"


class BatchNorm(nn.BatchNorm2d):
  # use the same hyperparames as the ones used in the tinygrad example
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super().__init__(num_features, eps=eps, momentum=momentum)
    self.weight.data.fill_(1.0)
    self.bias.data.fill_(0.0)
    self.weight.requires_grad = False
    self.bias.requires_grad = True


class ConvGroup(nn.Module):
  def __init__(self, channels_in, channels_out):
    super().__init__()

    self.pool1 = nn.MaxPool2d(2)
    self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

    self.activ = nn.GELU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.norm1(x)
    x = self.activ(x)
    residual = x
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.activ(x)

    return x + residual


class SpeedyResNet(nn.Module):
  def __init__(self, W):
    super().__init__()

    self.whitening = W
    self.net = nn.Sequential(
      nn.Conv2d(12, 32, kernel_size=1, bias=False),
      nn.GELU(),
      ConvGroup(32, 64),
      ConvGroup(64, 256),
      ConvGroup(256, 512),
    )
    self.linear = nn.Linear(512, 10, bias=False)

  def forward(self, x):
    # TODO kept it to one line as in the original, perhaps split into two lines for clarity?
    x = F.pad(F.conv2d(x, self.whitening), ((1, 0, 0, 1)))
    x = self.net(x)
    x = torch.amax(x, dim=(2, 3))
    x = self.linear(x)
    x = x / 9.0
    return x


# hyper-parameters were exactly the same as the original repo
bias_scaler = 58
hyp = {
  "seed": 201,
  "opt": {
    "bias_lr": 1.76 * bias_scaler / 512,
    "non_bias_lr": 1.76 / 512,
    "bias_decay": 1.08 * 6.45e-4 * BS / bias_scaler,
    "non_bias_decay": 1.08 * 6.45e-4 * BS,
    "final_lr_ratio": 0.025,
    "initial_div_factor": 1e6,
    "label_smoothing": 0.20,
    "momentum": 0.85,
    "percent_start": 0.23,
    "loss_scale_scaler": 1.0 / 128,  # (range: ~1/512 - 16+, 1/128 w/ FP16)
  },
  "net": {
    "kernel_size": 2,  # kernel size for the whitening layer
    "cutmix_size": 3,
    "cutmix_steps": 499,
    "pad_amount": 2,
  },
  "ema": {
    "steps": 399,
    "decay_base": 0.95,
    "decay_pow": 1.6,
    "every_n_steps": 5,
  },
}


@Context(FUSE_ARANGE=getenv("FUSE_ARANGE", 1))
def train_cifar():
  def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

  # ========== Model ==========
  def whitening(X, kernel_size=hyp["net"]["kernel_size"]):
    def _cov(X):
      return (X.T @ X) / (X.shape[0] - 1)

    def _patches(data, patch_size=(kernel_size, kernel_size)):
      h, w = patch_size
      c = data.shape[1]
      axis = (2, 3)
      return np.lib.stride_tricks.sliding_window_view(data, window_shape=(h, w), axis=axis).transpose((0, 3, 2, 1, 4, 5)).reshape((-1, c, h, w))

    def _eigens(patches):
      n, c, h, w = patches.shape
      Σ = _cov(patches.reshape(n, c * h * w))
      Λ, V = np.linalg.eigh(Σ, UPLO="U")
      return np.flip(Λ, 0), np.flip(V.T.reshape(c * h * w, c, h, w), 0)

    # NOTE: np.linalg.eigh only supports float32 so the whitening layer weights need to be converted to float16 manually
    Λ, V = _eigens(_patches(X.float().cpu().numpy()))
    W = V / np.sqrt(Λ + 1e-2)[:, None, None, None]

    return torch.tensor((W.astype(np.float32)), dtype=torch.float32, requires_grad=False)

  # ========== Loss ==========
  # TODO do i keep the below code? test to make sure the outputs are the same
  # def cross_entropy(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", label_smoothing: float = 0.0) -> torch.Tensor:
  #   divisor = y.shape[1]
  #   assert isinstance(divisor, int), "only supported int divisor"
  #   y = (1 - label_smoothing) * y + label_smoothing / divisor
  #   ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
  #   if reduction == "none":
  #     return ret
  #   if reduction == "sum":
  #     return ret.sum()
  #   if reduction == "mean":
  #     return ret.mean()
  #   raise NotImplementedError(reduction)

  # ========== Preprocessing ==========
  # NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
  def pad_reflect(X, size=2) -> torch.Tensor:
    X = torch.cat([X[..., :, 1 : size + 1].flip(-1), X, X[..., :, -(size + 1) : -1].flip(-1)], dim=-1)
    X = torch.cat([X[..., 1 : size + 1, :].flip(-2), X, X[..., -(size + 1) : -1, :]], dim=-2)
    return X

  # return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
  def make_square_mask(shape, mask_size) -> torch.Tensor:
    BS, _, H, W = shape
    low_x = torch.randint(low=0, high=W - mask_size, size=(BS,)).reshape((BS, 1, 1, 1))
    low_y = torch.randint(low=0, high=H - mask_size, size=(BS,)).reshape((BS, 1, 1, 1))
    idx_x = torch.arange(W, dtype=torch.int32).reshape((1, 1, 1, W))
    idx_y = torch.arange(H, dtype=torch.int32).reshape((1, 1, H, 1))
    return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

  # Similar, but different enough.
  def make_random_crop_indices(shape, mask_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    BS, _, H, W = shape
    low_x = torch.randint(low=0, high=W - mask_size, size=(BS,)).reshape((BS, 1, 1, 1))
    low_y = torch.randint(low=0, high=H - mask_size, size=(BS,)).reshape((BS, 1, 1, 1))
    idx_x = torch.arange(mask_size, dtype=torch.int32).reshape((1, 1, 1, mask_size))
    idx_y = torch.arange(mask_size, dtype=torch.int32).reshape((1, 1, mask_size, 1))
    return low_x, low_y, idx_x, idx_y

  def random_crop(X: torch.Tensor, crop_size=32):
    Xs, Ys, Xi, Yi = make_random_crop_indices(X.shape, crop_size)
    return X.gather(-1, (Xs + Xi).expand(-1, 3, X.shape[2], -1)).gather(-2, ((Ys + Yi).expand(-1, 3, crop_size, crop_size)))

  def cutmix(X, Y, order, mask_size=3):
    mask = make_square_mask(X.shape, mask_size)
    X_patch, Y_patch = X[order], Y[order]
    X_cutmix = torch.where(mask, X_patch, X)
    mix_portion = float(mask_size**2) / (X.shape[-2] * X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1.0 - mix_portion) * Y
    return X_cutmix, Y_cutmix

  def augmentations(X: torch.Tensor, Y: torch.Tensor):
    perms = torch.randperm(X.shape[0], device=X.device)  # We reuse perms for cutmix, because they are expensivne to generate
    if getenv("RANDOM_CROP", 1):
      X = random_crop(X, crop_size=32)
    if getenv("RANDOM_FLIP", 1):
      X = torch.where(torch.rand(X.shape[0], 1, 1, 1) < 0.5, X.flip(-1), X)  # flip LR
    X, Y = X[perms], Y[perms]
    return X, Y, *cutmix(X, Y, perms, mask_size=hyp["net"]["cutmix_size"])

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in: torch.Tensor, Y_in: torch.Tensor, BS: int, is_train: bool):
    step, epoch = 0, 0
    while True:
      st = time.monotonic()
      X, Y = X_in, Y_in
      if is_train:
        X, Y, X_cm, Y_cm = augmentations(X, Y)
        if getenv("CUTMIX", 1) and step >= hyp["net"]["cutmix_steps"]:
          X, Y = X_cm, Y_cm
      et = time.monotonic()
      # i get a ~10x speedup by dataset shuffling with torch!
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et - st) * 1e3:.2f} ms ({epoch=})")

      full_batches = ((X.shape[0] // BS) * BS) - BS
      for i in range(0, full_batches, BS):
        step += 1
        yield X[i : i + BS], Y[i : i + BS]
      epoch += 1
      if not is_train:
        break

  def transform(x):
    x = x.float() / 255.0
    x = x.reshape((-1, 3, 32, 32)) - torch.tensor(cifar_mean, device=x.device, dtype=x.dtype).reshape((1, 3, 1, 1))
    x = x / torch.tensor(cifar_std, device=x.device, dtype=x.dtype).reshape((1, 3, 1, 1))
    return x

  class modelEMA:
    def __init__(self, w, net):
      # self.model_ema = copy.deepcopy(net) # won't work for opencl due to unpickeable pyopencl._cl.Buffer
      self.net_ema = SpeedyResNet(w)
      for net_ema_param, net_param in zip(self.net_ema.state_dict().values(), net.state_dict().values()):
        net_ema_param.requires_grad = False
        net_ema_param.assign(net_param.numpy())

    def update(self, net, decay):
      for net_ema_param, (param_name, net_param) in zip(self.net_ema.state_dict().values(), net.state_dict().items()):
        # batchnorm currently is not being tracked
        if not ("num_batches_tracked" in param_name) and not ("running" in param_name):
          net_ema_param.assign(net_ema_param.detach() * decay + net_param.detach() * (1.0 - decay)).realize()

  set_seed(getenv("SEED", hyp["seed"]))

  if getenv("TINY_BACKEND"):
    import tinygrad.frontend.torch  # noqa: F401

    device = torch.device("tiny")
  else:
    device = torch.device({"METAL": "mps", "NV": "cuda"}.get(Device.DEFAULT, "cpu"))
  if DEBUG >= 1:
    print(f"using torch backend {device}")
  X_train, Y_train, X_test, Y_test = cifar()
  X_train = torch.tensor(X_train.float().numpy(), device=device)
  Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy(), device=device)
  X_test = torch.tensor(X_test.float().numpy(), device=device)
  Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy(), device=device)
  # one-hot encode labels
  Y_train, Y_test = F.one_hot(Y_train, 10), F.one_hot(Y_test, 10)
  # preprocess data
  X_train, X_test = transform(X_train), transform(X_test)

  # precompute whitening patches
  W = whitening(X_train)

  # initialize model weights
  model = SpeedyResNet(W)

  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp["net"]["pad_amount"])

  # Convert data and labels to the default dtype
  X_train, Y_train = X_train.to(torch.float32), Y_train.to(torch.float32)
  X_test, Y_test = X_test.to(torch.float32), Y_test.to(torch.float32)

  # if len(GPUS) > 1:
  #   for k, x in model.state_dict().items():
  #     if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
  #       x.shard_(GPUS, axis=0)
  #     else:
  #       x.to_(GPUS)

  # parse the training params into bias and non-bias
  params_dict = model.named_parameters()
  params_bias = []
  params_non_bias = []
  for name, params in params_dict:
    if params.requires_grad:
      if "bias" in name:
        params_bias.append(params)
      else:
        params_non_bias.append(params)

  opt_bias = optim.SGD(params_bias, lr=0.01, momentum=hyp["opt"]["momentum"], nesterov=True, weight_decay=hyp["opt"]["bias_decay"])
  opt_non_bias = optim.SGD(params_non_bias, lr=0.01, momentum=hyp["opt"]["momentum"], nesterov=True, weight_decay=hyp["opt"]["non_bias_decay"])

  # NOTE taken from the hlb_CIFAR repository, might need to be tuned
  initial_div_factor = hyp["opt"]["initial_div_factor"]
  final_lr_ratio = hyp["opt"]["final_lr_ratio"]
  pct_start = hyp["opt"]["percent_start"]
  lr_sched_bias = optim.lr_scheduler.OneCycleLR(
    opt_bias,
    max_lr=hyp["opt"]["bias_lr"],
    pct_start=pct_start,
    div_factor=initial_div_factor,
    final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
    total_steps=STEPS,
  )
  lr_sched_non_bias = optim.lr_scheduler.OneCycleLR(
    opt_non_bias,
    max_lr=hyp["opt"]["non_bias_lr"],
    pct_start=pct_start,
    div_factor=initial_div_factor,
    final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
    total_steps=STEPS,
  )

  def train_step(model, optimizers, lr_schedulers, X, Y):
    out = model(X)
    loss_batchsize_scaler = 512 / BS
    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=hyp["opt"]["label_smoothing"])
    loss = loss_fn(out, Y).mul(hyp["opt"]["loss_scale_scaler"] * loss_batchsize_scaler).sum().div(hyp["opt"]["loss_scale_scaler"])

    if not getenv("DISABLE_BACKWARD"):
      # index 0 for bias and 1 for non-bias
      optimizers[0].zero_grad()
      optimizers[1].zero_grad()
      loss.backward()
      optimizers[0].step()
      optimizers[1].step()
      lr_schedulers[0].step()
      lr_schedulers[1].step()
    return loss

  def eval_step(model, X, Y):
    out = model(X, training=False)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(out, Y)
    correct = out.argmax(axis=1) == Y.argmax(axis=1)
    return correct.realize(), loss.realize()

  step_times = []
  model_ema: Optional[modelEMA] = None
  projected_ema_decay_val = hyp["ema"]["decay_base"] ** hyp["ema"]["every_n_steps"]
  i = 0
  eval_acc_pct = 0.0
  batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True)
  with Tensor.train():
    st = time.monotonic()
    while i <= STEPS:
      if i % getenv("EVAL_STEPS", STEPS) == 0 and i > 1 and not getenv("DISABLE_BACKWARD"):
        # Use Tensor.training = False here actually bricks batchnorm, even with track_running_stats=True
        corrects = []
        corrects_ema = []
        losses = []
        losses_ema = []
        for Xt, Yt in fetch_batches(X_test, Y_test, BS=EVAL_BS, is_train=False):
          # TODO sharding temporarily disabled
          # if len(GPUS) > 1:
          #   Xt.shard_(GPUS, axis=0)
          #   Yt.shard_(GPUS, axis=0)

          correct, loss = eval_step(model, Xt, Yt)
          losses.append(loss.numpy().tolist())
          corrects.extend(correct.numpy().tolist())
          if model_ema:
            correct_ema, loss_ema = eval_step(model_ema.net_ema, Xt, Yt)
            losses_ema.append(loss_ema.numpy().tolist())
            corrects_ema.extend(correct_ema.numpy().tolist())

        # collect accuracy across ranks
        correct_sum, correct_len = sum(corrects), len(corrects)
        if model_ema:
          correct_sum_ema, correct_len_ema = sum(corrects_ema), len(corrects_ema)

        eval_acc_pct = correct_sum / correct_len * 100.0
        if model_ema:
          acc_ema = correct_sum_ema / correct_len_ema * 100.0
        print(
          f"eval     {correct_sum}/{correct_len} {eval_acc_pct:.2f}%, {(sum(losses) / len(losses)):7.2f} val_loss STEP={i} (in {(time.monotonic() - st) * 1e3:.2f} ms)"
        )
        if model_ema:
          print(f"eval ema {correct_sum_ema}/{correct_len_ema} {acc_ema:.2f}%, {(sum(losses_ema) / len(losses_ema)):7.2f} val_loss STEP={i}")

      if STEPS == 0 or i == STEPS:
        break

      GlobalCounters.reset()

      with WallTimeEvent(BenchEvent.STEP):
        X, Y = next(batcher)
        # TODO sharding temporarily disabled
        # if len(GPUS) > 1:
        #   X.shard_(GPUS, axis=0)
        #   Y.shard_(GPUS, axis=0)

        with Context(BEAM=getenv("LATEBEAM", BEAM.value), WINO=getenv("LATEWINO", WINO.value)):
          loss = train_step(model, [opt_bias, opt_non_bias], [lr_sched_bias, lr_sched_non_bias], X, Y)
          et = time.monotonic()
          loss_cpu = loss.detach().numpy()
        # EMA for network weights
        if getenv("EMA") and i > hyp["ema"]["steps"] and (i + 1) % hyp["ema"]["every_n_steps"] == 0:
          if model_ema is None:
            model_ema = modelEMA(W, model)
          model_ema.update(model, Tensor([projected_ema_decay_val * (i / STEPS) ** hyp["ema"]["decay_pow"]]))

      cl = time.monotonic()
      step_times.append((cl - st) * 1000.0)
      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device}"
      #  53  221.74 ms run,    2.22 ms python,  219.52 ms CL,  803.39 loss, 0.000807 LR, 4.66 GB used,   3042.49 GFLOPS,    674.65 GOPS
      print(
        # TODO is the method to get the LR correct? check this!
        f"{i:3d} {(cl - st) * 1000.0:7.2f} ms run, {(et - st) * 1000.0:7.2f} ms python, {(cl - et) * 1000.0:7.2f} ms {device_str}, {loss_cpu:7.2f} loss, {lr_sched_non_bias.get_last_lr()[0]:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS, {GlobalCounters.global_ops * 1e-9:9.2f} GOPS"
      )
      st = cl
      i += 1

  if assert_time := getenv("ASSERT_MIN_STEP_TIME"):
    min_time = min(step_times)
    assert min_time < assert_time, f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if eval_acc_pct >= target:
      print(colored(f"{eval_acc_pct=} >= {target}", "green"))
    else:
      raise ValueError(colored(f"{eval_acc_pct=} < {target}", "red"))


if __name__ == "__main__":
  with WallTimeEvent(BenchEvent.FULL):
    train_cifar()

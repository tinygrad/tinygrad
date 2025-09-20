import time
import numpy as np

import torch
from torch import nn, optim

from tinygrad import dtypes, Device
from tinygrad.helpers import getenv, trange, colored, DEBUG
from tinygrad.nn.datasets import cifar


class ConvGroup(nn.Module):
  def __init__(self, c_in:int, c_out:int):
    super().__init__()
    self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
    # track_running_stats=False and small eps/momentum as used in the original hlb-CIFAR pipeline
    self.bn1 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=0.85, track_running_stats=False)
    self.bn2 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=0.85, track_running_stats=False)
    # freeze BN weight, keep bias trainable
    self.bn1.weight.requires_grad_(False)
    self.bn2.weight.requires_grad_(False)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = nn.functional.max_pool2d(x, 2)
    x = x.float()
    x = self.bn1(x)
    x = x.to(torch.get_default_dtype())  # cast back to keep model dtype (fp16 when HALF)
    x = nn.functional.gelu(x, approximate="tanh")
    residual = x
    x = self.conv2(x)
    x = x.float()
    x = self.bn2(x)
    x = x.to(torch.get_default_dtype())
    x = nn.functional.gelu(x, approximate="tanh")
    return x + residual


class SpeedyResNet(nn.Module):
  def __init__(self, whitening_weight:torch.Tensor):
    super().__init__()
    # whitening conv (out channels = 12 for kernel_size=2)
    self.whiten = nn.Conv2d(3, whitening_weight.shape[0], kernel_size=whitening_weight.shape[-1], padding=0, bias=False)
    with torch.no_grad():
      self.whiten.weight.copy_(whitening_weight)
      # keep whitening fixed
      self.whiten.weight.requires_grad_(False)
    self.conv1 = nn.Conv2d(12, 32, kernel_size=1, bias=False)
    self.group1 = ConvGroup(32, 64)
    self.group2 = ConvGroup(64, 256)
    self.group3 = ConvGroup(256, 512)
    self.linear = nn.Linear(512, 10, bias=False)

  def forward(self, x:torch.Tensor, training:bool=True) -> torch.Tensor:
    # Pad to restore 32x32 (whitening yields 31x31)
    def forward_once(x:torch.Tensor) -> torch.Tensor:
      x = self.whiten(x)
      x = nn.functional.pad(x, (1, 0, 0, 1))
      x = self.conv1(x)
      x = nn.functional.gelu(x, approximate="tanh")
      x = self.group1(x)
      x = self.group2(x)
      x = self.group3(x)
      x = nn.functional.max_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
      x = torch.flatten(x, 1)
      x = self.linear(x)
      return x / 9.0
    return forward_once(x) if training else (forward_once(x) + forward_once(torch.flip(x, dims=[-1]))) / 2.0


def whitening_weight_from_data(X:torch.Tensor, kernel_size:int=2) -> torch.Tensor:
  # X: [N, 3, 32, 32] in normalized domain, compute whitening patches on CPU via numpy
  X_np = X.float().cpu().numpy()
  h, w = kernel_size, kernel_size
  # patches: [num_patches, C, h, w]
  patches = np.lib.stride_tricks.sliding_window_view(X_np, window_shape=(h, w), axis=(2, 3))
  patches = patches.transpose((0, 2, 3, 1, 4, 5)).reshape((-1, 3, h, w))
  n, c = patches.shape[0], patches.shape[1]
  cov = (patches.reshape(n, c * h * w).T @ patches.reshape(n, c * h * w)) / (n - 1)
  evals, evecs = np.linalg.eigh(cov, UPLO='U')
  evals, evecs = evals[::-1], evecs[:, ::-1]
  # numerical guard: covariance can have tiny negative eigenvalues due to precision
  evals = np.clip(evals, 0.0, None)
  V = evecs.T.reshape(c * h * w, c, h, w)
  W = V / np.sqrt(evals + 1e-2)[:, None, None, None]
  W = W.astype(np.float32)[:12]  # first 12 filters
  return torch.tensor(W, dtype=torch.float32)


def prepare_data(device:torch.device):
  X_train, Y_train, X_test, Y_test = cifar()
  # compute stats on normalized domain (0..1)
  std, mean = (X_train.float() / 255.0).std_mean(axis=(0, 2, 3))
  # to torch
  X_train = torch.tensor(X_train.float().numpy(), device=device)
  Y_train = torch.tensor(Y_train.numpy(), device=device, dtype=torch.long)
  X_test  = torch.tensor(X_test.float().numpy(), device=device)
  Y_test  = torch.tensor(Y_test.numpy(), device=device, dtype=torch.long)
  # channel-wise normalize
  mean_t = torch.tensor(mean.numpy(), device=device, dtype=X_train.dtype).reshape(1, 3, 1, 1)
  std_t  = torch.tensor(std.numpy(),  device=device, dtype=X_train.dtype).reshape(1, 3, 1, 1)
  X_train = (X_train / 255.0 - mean_t) / std_t
  X_test  = (X_test / 255.0 - mean_t) / std_t
  # keep inputs at the training precision (fp16 when HALF)
  X_train = X_train.to(torch.get_default_dtype())
  X_test  = X_test.to(torch.get_default_dtype())
  return X_train, Y_train, X_test, Y_test


def train():
  # device selection follows examples/other_mnist/beautiful_mnist_torch.py
  if getenv("TINY_BACKEND"):
    import tinygrad.frontend.torch  # noqa: F401
    device = torch.device("tiny")
  else:
    device = torch.device({"METAL":"mps", "NV":"cuda"}.get(Device.DEFAULT, "cpu"))
  if DEBUG >= 1: print(f"using torch backend {device}")

  BS = getenv("BS", 512)
  STEPS = getenv("STEPS", 200)

  X_train, Y_train, X_test, Y_test = prepare_data(device)
  # compute whitening from a subset for speed
  W = whitening_weight_from_data(X_train[:10000].to("cpu"))  # compute on CPU
  W = W.to(device=device, dtype=torch.get_default_dtype())
  model = SpeedyResNet(W).to(device)

  params = [
    {
      "params": [p for n,p in model.named_parameters() if p.requires_grad and "bias" in n],
      "lr": 1.76 * (58/512),
      "weight_decay": 1.08*6.45e-4*(BS/58),
    },
    {
      "params": [p for n,p in model.named_parameters() if p.requires_grad and "bias" not in n],
      "lr": 1.76 / 512,
      "weight_decay": 1.08*6.45e-4*BS,
    },
  ]
  opt = optim.SGD(params, momentum=0.85, nesterov=True)
  sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], steps_per_epoch=STEPS, epochs=1,
                                        pct_start=0.23, final_div_factor=1/(1e6*0.025), div_factor=1e6)

  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.20)

  def fetch_batches(X:torch.Tensor, Y:torch.Tensor, BS:int):
    N = (X.shape[0] // BS) * BS
    while True:
      order = torch.randperm(X.shape[0], device=X.device)
      for s in range(0, N, BS):
        idx = order[s:s+BS]
        yield X[idx], Y[idx]

  total_time = 0.0
  model.train()
  batches = fetch_batches(X_train, Y_train, BS)
  for step in trange(STEPS):
    Xt, Yt = next(batches)
    st = time.perf_counter()
    out = model(Xt)
    loss = loss_fn(out, Yt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    et = time.perf_counter()
    total_time += et - st
  print(f"avg step time: {total_time/STEPS*1000:.2f} ms")

  # quick eval (configurable sample count)
  eval_n = min(getenv("EVAL_SAMPLES", 100), X_test.shape[0])
  if eval_n > 0:
    with torch.no_grad():
      model.eval()
      i = torch.randint(0, X_test.shape[0], (eval_n,), device=device)
      out = model(X_test[i], training=False)
      acc = (out.argmax(dim=1) == Y_test[i]).float().mean().item() * 100.0
    print(colored(f"eval acc on subset({eval_n}): {acc:.2f}%", "green"))
    target = getenv("TARGET_EVAL_ACC_PCT", 0.0)
    if target:
      # float tolerance to avoid off-by-1e-7 style failures
      eps = 1e-6
      if acc + eps >= target and round(acc, 6) != 100.0:
        print(colored(f"{acc=} >= {target}", "green"))
      else:
        raise ValueError(colored(f"{acc=} < {target}", "red"))
  else:
    print(colored("skipping eval (EVAL_SAMPLES=0)", "yellow"))


if __name__ == "__main__":
  # default to half on supported devices if requested
  if getenv("HALF", 0):
    dtypes.default_float = dtypes.half
    torch.set_default_dtype(torch.float16)
  train()

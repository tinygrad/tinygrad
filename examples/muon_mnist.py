#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Muon


class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5),
      Tensor.relu,
      nn.Conv2d(32, 32, 5),
      Tensor.relu,
      nn.BatchNorm(32),
      Tensor.max_pool2d,  # 14×14 → 7×7
      nn.Conv2d(32, 64, 3),
      Tensor.relu,
      nn.Conv2d(64, 64, 3),
      Tensor.relu,
      nn.BatchNorm(64),
      Tensor.max_pool2d,  # 7×7 → 3×3
      lambda x: x.flatten(1),  # 64·3·3 = 576
      nn.Linear(576, 10),  # logits
    ]

  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)


X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

model = Model()
try:
  Optim = Muon
  optim_kw = dict(lr=3e-3, momentum=0.96)
except ImportError:
  print(colored("Muon not found – using Adam instead", "yellow"))
  Optim = nn.optim.Adam
  optim_kw = dict(lr=1e-3)

opt = Optim(nn.state.get_parameters(model), **optim_kw)

BS = int(getenv("BS", 512))
STEPS = int(getenv("STEPS", 70))
TARGET = float(getenv("TARGET_EVAL_ACC_PCT", 0.0))


@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  opt.zero_grad()
  idx = Tensor.randint(BS, high=X_train.shape[0])
  loss = model(X_train[idx]).sparse_categorical_crossentropy(Y_train[idx])
  loss.backward()
  opt.step()
  return loss


@TinyJit
def eval_accuracy() -> Tensor:
  return (model(X_test).argmax(axis=1) == Y_test).mean() * 100


test_acc = float("nan")
for step in (pbar := trange(STEPS)):
  GlobalCounters.reset()
  loss = train_step()

  if step % 10 == 9:
    test_acc = eval_accuracy().item()

  pbar.set_description(f"loss {loss.item():6.2f} | test acc {test_acc:5.2f}%")

if TARGET:
  if test_acc >= TARGET and test_acc != 100.0:
    print(colored(f"✅ final acc {test_acc:.2f}% ≥ target {TARGET}", "green"))
  else:
    raise ValueError(colored(f"❌ final acc {test_acc:.2f}% < target {TARGET}", "red"))

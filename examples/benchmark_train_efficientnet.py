#!/usr/bin/env python3
import os
import time
from tqdm import trange
from extra.utils import get_parameters
from models.efficientnet import EfficientNet
import tinygrad.optim as optim
from tinygrad.tensor import Tensor

from test.test_gc import tensors_allocated

try:
  import pynvml
  pynvml.nvmlInit()
  handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  def get_memory_used():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used
except Exception:
  def get_memory_used():
    return 0

NUM = int(os.getenv("NUM", 2))
BS = int(os.getenv("BS", 8))
CNT = int(os.getenv("CNT", 10))
BACKWARD = int(os.getenv("BACKWARD", 0))

if __name__ == "__main__":
  print(f"NUM:{NUM} BS:{BS} CNT:{CNT}")
  model = EfficientNet(NUM, classes=1000, has_se=False)
  parameters = get_parameters(model)
  optimizer = optim.Adam(parameters, lr=0.001)

  Tensor.training = True
  for i in trange(CNT):
    x_train = Tensor.randn(BS, 3, 224, 224, requires_grad=False)
    y_train = Tensor.randn(BS, 1000, requires_grad=False)

    st = time.monotonic()
    out = model.forward(x_train)
    loss = out.logsoftmax().mul(y_train).mean()
    if BACKWARD:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    mt = time.monotonic()
    loss.realize()
    for p in parameters:
      p.realize()
    et = time.monotonic()

    loss = loss.cpu().data[0]
    print(f"{(et-st)*1000.0:7.2f} ms run, {(mt-st)*1000.0:7.2f} ms build, {loss:7.2f} loss, {tensors_allocated():4d} tensors, {get_memory_used()/1e9:.2f} GB used")






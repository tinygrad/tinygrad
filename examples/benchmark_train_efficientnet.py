#!/usr/bin/env python3
import os
import time
from tqdm import trange
from models.efficientnet import EfficientNet
import tinygrad.nn.optim as optim
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CL

import gc
def tensors_allocated():
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])

NUM = int(os.getenv("NUM", 2))
BS = int(os.getenv("BS", 8))
CNT = int(os.getenv("CNT", 10))
BACKWARD = int(os.getenv("BACKWARD", 0))
TRAINING = int(os.getenv("TRAINING", 1))
ADAM = int(os.getenv("ADAM", 0))

if __name__ == "__main__":
  print(f"NUM:{NUM} BS:{BS} CNT:{CNT}")
  model = EfficientNet(NUM, classes=1000, has_se=False, track_running_stats=False)
  parameters = optim.get_parameters(model)
  for p in parameters: p.realize()
  if ADAM: optimizer = optim.Adam(parameters, lr=0.001)
  else: optimizer = optim.SGD(parameters, lr=0.001)

  Tensor.training = TRAINING
  Tensor.no_grad = not BACKWARD
  for i in trange(CNT):
    cpy = time.monotonic()
    x_train = Tensor.randn(BS, 3, 224, 224, requires_grad=False).realize()
    y_train = Tensor.randn(BS, 1000, requires_grad=False).realize()

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
    mem_used = CL.mem_used
    loss = loss.detach().cpu().data[0]
    cl = time.monotonic()

    print(f"{(st-cpy)*1000.0:7.2f} ms cpy,  {(cl-st)*1000.0:7.2f} ms run, {(mt-st)*1000.0:7.2f} ms build, {(et-mt)*1000.0:7.2f} ms realize, {(cl-et)*1000.0:7.2f} ms CL, {loss:7.2f} loss, {tensors_allocated():4d} tensors, {mem_used/1e9:.2f} GB used")






#!/usr/bin/env python3
import gc
import time
from tqdm import trange
from models.efficientnet import EfficientNet
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.jit import CacheCollector

def tensors_allocated():
  return sum(isinstance(x, Tensor) for x in gc.get_objects())

NUM = getenv("NUM", 2)
BS = getenv("BS", 8)
CNT = getenv("CNT", 10)
BACKWARD = getenv("BACKWARD", 0)
TRAINING = getenv("TRAINING", 1)
ADAM = getenv("ADAM", 0)
CLCACHE = getenv("CLCACHE", 0)

if __name__ == "__main__":
  print(f"NUM:{NUM} BS:{BS} CNT:{CNT}")
  model = EfficientNet(NUM, classes=1000, has_se=False, track_running_stats=False)
  parameters = get_parameters(model)
  for p in parameters: p.realize()
  if ADAM: optimizer = optim.Adam(parameters, lr=0.001)
  else: optimizer = optim.SGD(parameters, lr=0.001)

  Tensor.training = TRAINING
  Tensor.no_grad = not BACKWARD
  for i in trange(CNT):
    GlobalCounters.reset()
    cpy = time.monotonic()
    x_train = Tensor.randn(BS, 3, 224, 224, requires_grad=False).realize()
    y_train = Tensor.randn(BS, 1000, requires_grad=False).realize()

    # TODO: replace with TinyJit
    if i < 3 or not CLCACHE:
      st = time.monotonic()
      out = model.forward(x_train)
      loss = out.log_softmax().mul(y_train).mean()
      if i == 2 and CLCACHE: CacheCollector.start()
      if BACKWARD:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      mt = time.monotonic()
      loss.realize()
      for p in parameters:
        p.realize()
      et = time.monotonic()
    else:
      st = mt = time.monotonic()
      for prg, args in cl_cache: prg(*args)
      et = time.monotonic()

    if i == 2 and CLCACHE:
      cl_cache = CacheCollector.finish()

    mem_used = GlobalCounters.mem_used
    loss_cpu = loss.detach().numpy()
    cl = time.monotonic()

    print(f"{(st-cpy)*1000.0:7.2f} ms cpy,  {(cl-st)*1000.0:7.2f} ms run, {(mt-st)*1000.0:7.2f} ms build, {(et-mt)*1000.0:7.2f} ms realize, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {tensors_allocated():4d} tensors, {mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")

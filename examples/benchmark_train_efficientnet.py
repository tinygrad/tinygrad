#!/usr/bin/env python3
import gc
import time
from tqdm import trange
from models.efficientnet import EfficientNet
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CL
from tinygrad.ops import GlobalCounters
from tinygrad.helpers import getenv

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
  parameters = optim.get_parameters(model)
  for p in parameters: p.realize()
  if ADAM: optimizer = optim.Adam(parameters, lr=0.001)
  else: optimizer = optim.SGD(parameters, lr=0.001)

  Tensor.training = TRAINING
  Tensor.no_grad = not BACKWARD
  for i in trange(CNT):
    GlobalCounters.time_sum = 0
    GlobalCounters.global_ops = 0
    cpy = time.monotonic()
    x_train = Tensor.randn(BS, 3, 224, 224, requires_grad=False).realize()
    y_train = Tensor.randn(BS, 1000, requires_grad=False).realize()

    if i < 3 or not CLCACHE:
      st = time.monotonic()
      out = model.forward(x_train)
      loss = out.logsoftmax().mul(y_train).mean()
      if ADAM: optimizer.t = 0    # TODO: fixing this requires optional constant folding
      if i == 2 and CLCACHE: CL.CACHE = []
      if BACKWARD:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      mt = time.monotonic()
      loss.realize()
      for p in parameters:
        p.realize()
      et = time.monotonic()
      ops = GlobalCounters.global_ops
    else:
      st = mt = time.monotonic()
      ops = 0
      for prg, args in cl_cache:
        prg.clprg(CL().cl_queue, *args)
        ops += prg.op_estimate
      et = time.monotonic()

    if i == 2 and CLCACHE:
      cl_cache = CL.CACHE
      CL.CACHE = None

    mem_used = CL.mem_used
    loss_cpu = loss.detach().cpu().data[0]
    cl = time.monotonic()

    print(f"{(st-cpy)*1000.0:7.2f} ms cpy,  {(cl-st)*1000.0:7.2f} ms run, {(mt-st)*1000.0:7.2f} ms build, {(et-mt)*1000.0:7.2f} ms realize, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {tensors_allocated():4d} tensors, {mem_used/1e9:.2f} GB used, {ops*1e-9/(cl-st):9.2f} GFLOPS")

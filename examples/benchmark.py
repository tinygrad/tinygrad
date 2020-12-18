#!/usr/bin/env python3
import numpy as np
from tinygrad.tensor import Tensor
import time

# Tensor has max size of 0x4000 for now
ba = Tensor(np.random.normal(size=(0x4000,)))
for dev in ["CPU", "GPU", "ANE"]:
  if dev == "GPU":
    baa = ba.gpu()
  elif dev == "ANE":
    baa = ba.ane()
  else:
    baa = ba
  for i in range(3):
    st = time.time()
    boaa = baa.relu()
    et = time.time()
    if i == 2:
      print("%s can do at least %.2f MEGAReLUs/sec" % (dev, (np.prod(boaa.shape)/1e6)/(et-st)))
    # decently reliable
    assert(np.all(boaa.cpu().data >= 0))

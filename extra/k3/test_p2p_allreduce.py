#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_p2p_allreduce.py: amd.partial_sum (p2p mailbox allreduce) of per gpu partials against numpy, repeated (epochs)
import time
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.llm.kernels.amd import partial_sum

devs = tuple(f"AMD:{i}" for i in range(4))
rng = np.random.default_rng(0)
for n in (3584, 7168, 21504):
  for it in range(4):
    parts = rng.standard_normal((4, n)).astype(np.float32)
    x = Tensor(parts).shard(devs, axis=0).realize()
    got = partial_sum(x)
    st = time.perf_counter()
    got.realize()
    Device[devs[0]].synchronize()
    dt = time.perf_counter() - st
    for g in range(4): np.testing.assert_allclose(Tensor(got.uop.mselect(g)).numpy(), parts.sum(0), rtol=1e-5, atol=1e-5)
  print(f"n {n}: ok ({dt * 1e6:.0f} us incl launch)")
# all_gather_p2p: a tensor split over the gpus on its last axis whole on every gpu, of a token and of a verify step's two
from tinygrad.llm.kernels.amd import all_gather_p2p
for T in (1, 2):
  full = rng.standard_normal((1, T, 7168)).astype(np.float32)
  got = all_gather_p2p(Tensor(full).shard(devs, axis=2).realize())
  assert got is not None
  got.realize()
  for g in range(4): np.testing.assert_equal(Tensor(got.uop.mselect(g)).numpy(), full)
  print(f"gather of {T} tokens: ok")

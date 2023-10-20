import random, os
import numpy as np
from extra.optimization.helpers import ast_str_to_lin, load_worlds
from tinygrad.features.search import bufs_from_lin, time_linearizer
from tinygrad.ops import Device

devices = [fn.replace("ops_", "").replace(".py", "").upper() for fn in os.listdir("./tinygrad/runtime") if fn.endswith(".py") and fn.startswith("ops_")]
ast_strs = load_worlds()
optimizers = [lambda lin: lin.hand_coded_optimizations()]

if __name__ == "__main__":
  for i in range(200):
    Device.DEFAULT = random.choice(devices)
    ast = random.choice(ast_strs)

    lin = ast_str_to_lin(ast)
    rawbufs = bufs_from_lin(lin)
    base_tm = time_linearizer(lin, rawbufs)
    ground_truth = rawbufs[0].toCPU()

    for opt in optimizers:
      opt(lin)
      tm = time_linearizer(lin, rawbufs)
      np.testing.assert_allclose(ground_truth, rawbufs[0].toCPU())

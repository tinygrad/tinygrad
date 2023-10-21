import random, os
import numpy as np
from extra.optimization.helpers import ast_str_to_lin, load_worlds
from tinygrad.features.search import bufs_from_lin, time_linearizer
from tinygrad.helpers import DEBUG, ansilen
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Compiled, Device
from tinygrad.shape.symbolic import sym_infer

devices = os.getenv("DEVICES").split(",") #type: ignore
ast_strs = load_worlds()
optimizers = [lambda lin: lin.hand_coded_optimizations()]

if __name__ == "__main__":
  for i in range(10): # TODO what's a good sample size here?
    Device.DEFAULT = random.choice(devices)
    assert isinstance(Device[Device.DEFAULT], Compiled)
    ast = random.choice(ast_strs)

    lin = ast_str_to_lin(ast)
    rawbufs = bufs_from_lin(lin)
    base_tm = time_linearizer(lin, rawbufs)
    ground_truth = rawbufs[0].toCPU()

    for j, opt in enumerate(optimizers):
      opt(lin)
      tm = time_linearizer(lin, rawbufs)
      gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/tm
      np.testing.assert_allclose(ground_truth, rawbufs[0].toCPU())
      if tm > base_tm and DEBUG >= 1: print(f"WARN - optimization {j} made things slower! {ast_strs.index(ast)}th ast, base {base_tm*1000:.2f} ms - this {tm*1000:.2f} ms")
      if DEBUG >= 1: print(f"opt {j} ast {ast_strs.index(ast)} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")

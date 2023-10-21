import random
import numpy as np
from os import getenv
from extra.optimization.helpers import ast_str_to_lin, load_worlds
from tinygrad.features.search import bufs_from_lin, time_linearizer
from tinygrad.helpers import DEBUG
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Compiled, Device
from tinygrad.shape.symbolic import sym_infer

ast_strs = load_worlds()
optimizers = [lambda lin: lin.hand_coded_optimizations()]

if __name__ == "__main__":
  for i in range(int(getenv("FUZZ", 100))):
    Device.DEFAULT = random.choice(getenv("DEVICES").split(",")) #type: ignore
    assert isinstance(Device[Device.DEFAULT], Compiled)
    ast = random.choice(ast_strs)

    lin = ast_str_to_lin(ast)
    rawbufs = bufs_from_lin(lin)
    base_tm = time_linearizer(lin, rawbufs)
    ground_truth = rawbufs[0].toCPU()

    for j, opt in enumerate(optimizers):
      opt(lin)
      tm = min([time_linearizer(lin, rawbufs) for _ in range(int(getenv("SAMPLE_SIZE", 10)))])
      gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/tm
      np.testing.assert_allclose(ground_truth, rawbufs[0].toCPU())
      if tm > base_tm and DEBUG >= 1: print(f"WARN - optimization {j} made things slower! {ast_strs.index(ast)}th ast, base {base_tm*1000:.2f} ms - this {tm*1000:.2f} ms")
      if DEBUG >= 1: print(f"opt {j} ast {ast_strs.index(ast)} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")

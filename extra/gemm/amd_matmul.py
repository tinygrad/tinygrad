# kernel8_batched_gmem.s from https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
import pathlib
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.opt.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem, ProgramSpec, get_program
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, Ops, UOp

N = 4096
run_count = 5

if __name__ == "__main__":
  ast = (Tensor.empty(N, N)@Tensor.empty(N, N)).schedule()[-1].ast
  prg = get_program(ast, Device.default.renderer)

  src = (pathlib.Path(__file__).parent / "kernel5_lds_optim.cpp").read_text()
  prgfast = replace(prg, name="kernel5_lds_optim", src=src, global_size=[N//128, N//128, 1], local_size=[128, 1, 1])
  runner = CompiledRunner(prgfast)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  c = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(BEAM=4):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  ei = ExecItem(runner, [a.uop.buffer, b.uop.buffer, c.uop.buffer])
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)

  GlobalCounters.reset()
  mse = (c-tc).square().mean().item()
  print(mse)

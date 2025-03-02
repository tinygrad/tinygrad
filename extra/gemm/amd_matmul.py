# kernel8_batched_gmem.s from https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
import pathlib
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, Device, Context
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

N = 4096
run_count = 5

#src = (pathlib.Path(__file__).parent / "fp32_sgemm_amd" / "src" / "kernel8_batched_gmem.s").read_text()
src = (pathlib.Path(__file__).parent / "kernel8_batched_gmem.s").read_text()

if __name__ == "__main__":
  rng = np.random.default_rng()
  a = Tensor(na:=rng.random((4096, 4096), dtype=np.float32)).realize()
  b = Tensor(nb:=rng.random((4096, 4096), dtype=np.float32)).realize()
  c = a @ b
  si = c.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.LOCAL, axis=1, arg=16),
  #        Opt(op=OptOps.LOCAL, axis=0, arg=8),
  #        Opt(op=OptOps.UPCAST, axis=2, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=1, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=0, arg=2)]
  #opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=0, arg=4),
  #        Opt(op=OptOps.LOCAL, axis=1, arg=8),
  #        Opt(op=OptOps.LOCAL, axis=0, arg=4)]
  #opts = [Opt(op=OptOps.LOCAL, axis=1, arg=16),
  #        Opt(op=OptOps.LOCAL, axis=0, arg=16)]
  #for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  prg = replace(prg, src=src, global_size=[N//128, N//128, 1], local_size=[128, 1, 1])
  print(prg.global_size, prg.local_size)
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  nc = c.numpy()
  np.testing.assert_allclose(na@nb, nc, rtol=1e-5)

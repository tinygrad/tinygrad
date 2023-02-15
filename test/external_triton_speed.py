import time
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_triton import TritonBuffer, TritonASTKernel, cuda
from extra.kernel_search import apply_intervention, Interventions, randomize_buffers
from extra.lib_test_ast import test_ast

def test_gemm():
  N = 768
  hb0 = TritonBuffer(shape=(N, N), force_create=True)
  hb1 = TritonBuffer(shape=(N, N), force_create=True)
  buf0 = TritonBuffer(shape=ShapeTracker(shape=(N, N, N), views=[View((N, N, N), (N, 0, 1), 0)]), hostbuf=hb0)
  buf1 = TritonBuffer(shape=ShapeTracker(shape=(N, N, N), views=[View((N, N, N), (0, 1, N), 0)]), hostbuf=hb1)
  op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
  op1 = LazyOp(ReduceOps.SUM, (op0,), (N, N, 1))
  ast = LazyOp(MovementOps.RESHAPE, (op1,), (N, N))
  randomize_buffers(ast)

  k = TritonASTKernel(ast)

  ii = []
  ii.append((Interventions.UPCAST, (0, 64)))
  ii.append((Interventions.UPCAST, (1, 64)))
  ii.append((Interventions.UPCAST, (2, 16)))
  #ii.append((Interventions.UPCAST, (1, 16)))
  #ii.append((Interventions.SHIFT, (1, 16, False)))
  #ii.append((Interventions.SHIFT, (1, 16, False)))
  for w in ii: apply_intervention(k, *w)

  runner = k.codegen()
  runner(*k.bufs)
  cuda.Context.synchronize()
  ops = k.info.flops

  n_repeat = 20
  start_event = [cuda.Event() for _ in range(n_repeat)]
  end_event = [cuda.Event() for _ in range(n_repeat)]
  for i in range(20):
    start_event[i].record()
    runner(*k.bufs)
    end_event[i].record()
  # there's like 10 us of launch overhead
  cuda.Context.synchronize()
  kt = min([e.time_since(s)*1e6 for s, e in zip(start_event, end_event)])
  print(f"{kt*1e-3:7.2f} us kernel, {ops/kt:5.2f} GFLOPS")

  real = hb0.toCPU() @ hb1.toCPU()
  test = k.ret.toCPU()
  #print(real, test)
  np.testing.assert_allclose(real, test, atol=1e-3)
  if not getenv("NOTEST"): test_ast(k)

if __name__ == "__main__":
  test_gemm()

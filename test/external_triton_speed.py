import time
from tinygrad.helpers import getenv
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_triton import TritonBuffer, TritonASTKernel, stream
from extra.kernel_search import apply_intervention, Interventions, randomize_buffers
from extra.lib_test_ast import test_ast

def test_gemm():
  N = 768
  buf0 = TritonBuffer(shape=ShapeTracker(shape=(1, 1, N, N, 1, 1, 1, N), views=[View((1, N, N, 1), (0, 1, N, 0), 0), View((1, 1, N, N, 1, 1, 1, N), (0, 0, 0, 1, 0, 0, 0, N), 0)]), hostbuf=TritonBuffer(shape=(N, N), force_create=True))
  buf1 = TritonBuffer(shape=ShapeTracker(shape=(1, 1, N, N, 1, 1, 1, N), views=[View((1, 1, N, N, 1, 1, 1, N), (0, 0, 1, 0, 0, 0, 0, N), 0)]), hostbuf=TritonBuffer(shape=(N, N), force_create=True))
  op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
  op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, N, N, 1, 1, 1, 1))
  ast = LazyOp(MovementOps.RESHAPE, (op1,), (N, N))
  randomize_buffers(ast)
  k = TritonASTKernel(ast)

  ii = []
  ii.append((Interventions.UPCAST, (0, 4)))
  #ii.append((Interventions.UPCAST, (1, 16)))
  #ii.append((Interventions.UPCAST, (1, 16)))
  #ii.append((Interventions.SHIFT, (1, 16, False)))
  #ii.append((Interventions.SHIFT, (1, 16, False)))
  for w in ii: apply_intervention(k, *w)

  runner = k.codegen()
  runner(*k.bufs)
  stream.synchronize()
  ops = k.info.flops

  t1 = time.monotonic_ns()
  runner(*k.bufs)
  stream.synchronize()
  t2 = time.monotonic_ns()

  print(f"{(t2-t1)*1e-3:7.2f} us  {ops/(t2-t1):5.2f} GFLOPS")

  if not getenv("NOTEST"): test_ast(k)

if __name__ == "__main__":
  test_gemm()

from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

N = 4096
if __name__ == "__main__":
  A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
  si = C.schedule()[-1]
  ast = si.ast
  k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
  opts = [Opt(op=OptOps.TC, axis=0, amt=0),
          Opt(op=OptOps.UPCAST, axis=1, amt=16),
          Opt(op=OptOps.UPCAST, axis=0, amt=2),
          Opt(op=OptOps.LOCAL, axis=0, amt=4),
          Opt(op=OptOps.UNROLL, axis=0, amt=4),
          Opt(op=OptOps.LOCAL, axis=1, amt=2),
  ]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  tflops = []
  for i in range(5):
    tm = ei.run(wait=True)
    tflops.append((2*N*N*N/tm)*1e-12)
  print(f"TFLOPS: {sum(tflops)/len(tflops):.2f}")

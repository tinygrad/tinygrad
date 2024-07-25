from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

# PTX=1 python3 extra/gemm/tinygrad_nv_matmul.py
# or
# PTX=1 IGNORE_BEAM_CACHE=1 BEAM=4 NV=1 HALF=1 DEBUG=2 python3 extra/gemm/simple_matmul.py

N = 4096
if __name__ == "__main__":
  A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
  si = C.schedule()[-1]
  ast = si.ast
  k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.TC, axis=0, amt=0),
  #        Opt(op=OptOps.UPCAST, axis=1, amt=16),
  #        Opt(op=OptOps.UPCAST, axis=0, amt=2),
  #        Opt(op=OptOps.LOCAL, axis=0, amt=4),
  #        Opt(op=OptOps.UNROLL, axis=0, amt=4),
  #        Opt(op=OptOps.LOCAL, axis=1, amt=2),
  #]
  opts_ptx = [
    Opt(op=OptOps.TC, axis=0, amt=0),
    Opt(op=OptOps.UPCAST, axis=0, amt=4),
    Opt(op=OptOps.UPCAST, axis=1, amt=8),
    Opt(op=OptOps.UNROLL, axis=0, amt=4),
    Opt(op=OptOps.LOCAL, axis=1, amt=4),
    Opt(op=OptOps.LOCAL, axis=0, amt=2),
  ]
  for opt in opts_ptx: k.apply_opt(opt)
  prg = k.to_program()
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  tflops = []
  for i in range(5):
    tm = ei.run(wait=True)
    tflops.append((2*N*N*N/tm)*1e-12)
  print(f"TFLOPS: {max(tflops):.2f}")

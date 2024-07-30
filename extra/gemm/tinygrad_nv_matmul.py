from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv, DEBUG
from tinygrad.engine.graph import print_globalcounters
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem
from dataclasses import replace

N = 4096
if __name__ == "__main__":
  if getenv("GEMV"):
    A, B = Tensor.empty(1, N, dtype=dtypes.float), Tensor.empty(14336, N, dtype=dtypes.float16).T
  else:
    A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
  si = C.schedule()[-1]
  ast = si.ast
  k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
  if getenv("GEMV"):
    opts = [
      #Opt(op=OptOps.GROUP, axis=0, amt=8),
      Opt(op=OptOps.UNROLL, axis=0, amt=8),
      Opt(op=OptOps.GROUP, axis=0, amt=32),
    ]
  else:
    opts = [
      Opt(op=OptOps.TC, axis=0, amt=0),
      Opt(op=OptOps.UPCAST, axis=0, amt=4),
      Opt(op=OptOps.UPCAST, axis=1, amt=8),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
      Opt(op=OptOps.LOCAL, axis=1, amt=2),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
    ]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  new_src = prg.src
  #new_src = new_src.replace("half8 val0 = *((half8*)(data2+(gidx0*4096)+alu0+alu1));", """
  #half8 val0;
  #((uint4*)&val0)[0] = *((uint4*)(data2+(gidx0*4096)+alu0+alu1));
  #""")
  #new_src = new_src.replace("""if ((!(bool)(lidx0))) {
  #  float acc1 = 0.0f;
  #  for (int ridx1 = 0; ridx1 < 32; ridx1++) {
  #    float val3 = temp1[ridx1];
  #    acc1 = (acc1+val3);
  #  }
  #  data0[gidx0] = acc1;
  #}""", "if ((!(bool)(lidx0))) { data0[gidx0] = temp1[0]; }")
  prg = replace(prg, src=new_src)
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  for i in range(5): ei.run(wait=True)
  if DEBUG < 2: print_globalcounters()

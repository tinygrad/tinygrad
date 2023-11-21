from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer
from test.external.fuzz_linearizer import run_linearizer
from tinygrad.codegen.kernel import Opt, OptOps

a = Tensor.rand(64, 61).sum(axis=0)
sched = [si for si in a.lazydata.schedule() if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

for i, st in enumerate(lin.sts):
  p = ((0, 3),) + ((0, 0),) * (len(st.shape)-1)
  lin.sts[i] = st.pad(p)

# lin.apply_opt(Opt(op=OptOps.LOCAL, axis=0, amt=32))

lin.hand_coded_optimizations()
lin.linearize()

run_linearizer(lin)
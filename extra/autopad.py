from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer
from test.external.fuzz_linearizer import run_linearizer
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.engine.schedule import create_schedule

N = 17**3

a = Tensor.rand(N, N)
b = Tensor.rand(N, N)
c = a @ b
sched = [si for si in create_schedule([c.lazydata]) if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

lin.apply_opt(Opt(op=OptOps.PADTO, axis=0, amt=32))
lin.apply_opt(Opt(op=OptOps.PADTO, axis=1, amt=32))
lin.hand_coded_optimizations()
lin.linearize()
print(f"{lin.applied_opts=}")

run_linearizer(lin)

###

a = Tensor.rand(61, 61).sum(axis=0)
sched = [si for si in create_schedule([a.lazydata]) if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

lin.apply_opt(Opt(op=OptOps.PADTO, axis=0, amt=32))
lin.hand_coded_optimizations()
lin.linearize()
print(f"{lin.applied_opts=}")

run_linearizer(lin)
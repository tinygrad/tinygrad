import numpy as np
from tinygrad.features.graph import print_tree
from tinygrad.helpers import panic
from tinygrad import Tensor
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.uops import UOps
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.realize import create_schedule
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.ops_metal import MetalCompiler

data = np.arange(16).reshape((4,4))
a = Tensor(data, dtype=dtypes.float).sum() + 4
ast = create_schedule([a.lazydata])[-1].ast #type:ignore
print_tree(ast)
lin = Linearizer(ast)
print("------------")
for st in lin.sts:
  print(st)
#panic()
lin.hand_coded_optimizations()
# self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
#panic(lin.full_unupcasted_shape)
#lin.apply_opt(Opt(OptOps.UNROLL, axis=panic(), amt=0))
lin.linearize()
for u in lin.uops: print(u)
code = MetalRenderer(lin.name, lin.uops)
print(code)

import numpy as np
from tinygrad import Tensor
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.uops import UOps
from tinygrad.dtype import dtypes
from tinygrad.realize import create_schedule
from tinygrad.renderer.cstyle import MetalRenderer

data = np.arange(10)
a = Tensor(data, dtype=dtypes.int).sum() + 4
ast = create_schedule([a.lazydata])[-1].ast
lin = Linearizer(ast)
#lin.hand_coded_optimizations()
lin.linearize()
for u in lin.uops:
    print(u)
code = MetalRenderer(lin.name, lin.uops)
print(code)

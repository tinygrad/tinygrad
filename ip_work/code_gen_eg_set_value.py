from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.codegen.uops import UOp, UOps, UOpGraph
import tinygrad.ops as ops
import tinygrad.dtype as dtype
from tinygrad.dtype import dtypes

# from https://mesozoic-egg.github.io/tinygrad-notes/backends.html

# uops = UOpGraph()
# _global = uops.add(UOps.DEFINE_GLOBAL, dtype.PtrDType(dtypes.int), (), (0, 'data0', True))
# _value = uops.add(UOps.CONST, dtypes.int, (), 199)
# _zero = uops.add(UOps.CONST, dtypes.int, (), 0)
# uops.add(UOps.STORE, dtypes.float, (_global, _zero, _value), None)
# output = uops_to_cstyle(MetalLanguage(), 'test', uops)
# print(output)

uops = []

_global = UOp(UOps.DEFINE_GLOBAL, dtype.PtrDType(dtypes.int), (), (0, 'data0', True))
_value = UOp(UOps.CONST, dtypes.int, (), 199)
_zero = UOp(UOps.CONST, dtypes.int, (), 0)

# uops.append(_global)
# uops.append(_value)
# uops.append(_zero)
_res = UOp(UOps.STORE, None, (_global, _zero, _value), None)
# uops.append(_res)

uops = [_global, _value, _zero, _res]

uops_graph = UOpGraph(uops)

renderer = MetalRenderer()

output = renderer.render('test', uops_graph)

# output = uops_to_cstyle(MetalLanguage(), 'test', UOpGraph(uops))
print(output)

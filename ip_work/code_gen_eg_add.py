
from tinygrad import dtype, ops
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer


uops_list = []

_global = UOp(UOps.DEFINE_GLOBAL, dtype.PtrDType(dtypes.int), (), (0, 'data0', True))
_value = UOp(UOps.CONST, dtypes.int, (), 199)
_value_2 = UOp(UOps.CONST, dtypes.int, (), 200)
_added = UOp(UOps.ALU, dtypes.int, (_value, _value_2), ops.BinaryOps.ADD)
_zero = UOp(UOps.CONST, dtypes.int, (), 0)

_store = UOp(UOps.STORE, None, (_global, _zero, _added), None)

# uops_list = [_global, _value, _zero, _store]
uops_list = [_global, _value, _value_2, _added, _zero, _store]



uops = UOpGraph(uops_list)

renderer = MetalRenderer()

output = renderer.render('test', uops)
print(output)
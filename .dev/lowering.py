from tinygrad.helpers import Target
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.uop.render import pyrender
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes

buf_a = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
buf_b = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 1)
buf_c = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 2)
buf_out = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 3)

tidx = UOp.special(256, 'lidx0')
load_a = buf_a.index(tidx).load()
load_b = buf_b.index(tidx).load()
load_c = buf_c.index(tidx).load()

result = load_a * load_b + load_c
store = buf_out.index(tidx).store(result)
sink = store.sink(arg=KernelInfo('fma'))

class FakeGPU(CStyleLanguage):
    has_local = True
    has_shared = True
    global_max = (0x8fffffff,) * 3
ren = FakeGPU(Target('GPU',''))

lowered = full_rewrite_to_sink(sink, ren)
print("lowered")
print(pyrender(lowered))


#print("sorted: ", list(lowered.toposort()))

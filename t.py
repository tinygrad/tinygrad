from tinygrad.device import Device
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.realize import create_schedule
from test.test_ops import helper_test_op

out = helper_test_op([(1,128), (128,128)], lambda x,y: (x@y).relu(), atol=1e-4)

ast = create_schedule([out.lazydata])[-1].ast
lin = Linearizer(ast)
lin.hand_coded_optimizations()
lin.linearize()

code = Device[Device.DEFAULT].compiler.render(lin.name, lin.uops)
#for i, c in enumerate(code.splitlines()): print(f"{i} {c}")
print(code)
#print(lin.name)
#print("\n".join(code.splitlines()[7:27]))
print(lin.applied_opts)

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.device import Device
from tinygrad.realize import create_schedule
from tinygrad import Tensor

N = 32
x = Tensor.empty(N, N)
w1 = Tensor.empty(N, N)
w2 = Tensor.empty(N, N)
out = (x@w1)@w2
ast = create_schedule([out.lazydata])[-1].ast
lin = Linearizer(ast)
lin.apply_tensor_cores()
lin.linearize()
for u in lin.uops: print(u)
code = Device[Device.DEFAULT].compiler.render("test", lin.uops)
print(code)

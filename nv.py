from os import getenv
from typing import List, cast
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import prod
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.runtime.lib import RawBuffer
from tinygrad.runtime.ops_cuda import renderer, CUDAProgram
from tinygrad.tensor import Tensor
import os

os.environ["TRITON"] = "0"
os.environ["CUDA"] = "1"

print(Device.DEFAULT)
si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
rout = Device[Device.DEFAULT].buffer(si.out.st.size(), si.out.dtype)
rin = [Device[Device.DEFAULT].buffer(x.st.size(), x.dtype) for x in si.inputs]
lin = Linearizer(si.ast)
lin.linearize()
code = renderer(lin.function_name, lin.uops)
prg = CUDAProgram(lin.function_name, code[0], code[1]["binary"])
tm = prg(lin.global_size, lin.local_size, rout, *rin, wait=True)
print(tm)

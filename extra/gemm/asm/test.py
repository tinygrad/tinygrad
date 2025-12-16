import pathlib
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import lower_schedule_item, ExecItem, CompiledRunner

N = 8192
scale = 10.0
dtype = dtypes.bfloat16

Tensor.manual_seed(0)

A = (Tensor.randn(N, N) / scale).cast(dtypes.bfloat16).contiguous()
B = (Tensor.randn(N, N) / scale).cast(dtypes.bfloat16).contiguous()
Bt = B.T.contiguous()
Tensor.realize(A, B, Bt)

# ** tinygrad gemm

C_tiny = Tensor.matmul(A, Bt, dtype=dtypes.float32).cast(dtype)
si = C_tiny.schedule()[-1]
eis:list[ExecItem] = [lower_schedule_item(si)]

C_asm = Tensor.empty_like(A)
C_asm.uop.buffer.allocate()

# ** raw assembly gemm

with open(pathlib.Path(__file__).parent/"lib", "rb") as f: lib = f.read()
prg = CompiledRunner(precompiled=lib, p=replace(eis[0].prg.p, src=lib, name="matmul", global_size=(128, 86, 1), local_size=(256, 1, 1)))
#Device[Device.DEFAULT].compiler.disassemble(lib)
# TODO: re assemble lib without the custom KernelArgs struct
#eis.append(ExecItem(prg, [C_asm.uop.buffer, A.uop.buffer, B.uop.buffer]))

for ei in eis: ei.run(wait=True)

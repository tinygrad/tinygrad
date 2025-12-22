import pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.renderer import ProgramSpec

# ** generate inputs on CPU

N = 8192
scale = 10.0

import torch
torch.manual_seed(0)
A = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
B = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
Bt = B.t().contiguous() # transpose B for the baseline gemm
C_torch = A@Bt

# ** copy buffers to AMD

# input creation and validation run on the copy engine for simpler tracing

def from_torch(t:torch.Tensor) -> Tensor:
  return Tensor.from_blob(t.data_ptr(), t.shape, dtype=dtypes.bfloat16, device="cpu").to(Device.DEFAULT).realize()

C_tiny = Tensor.matmul(from_torch(A), from_torch(Bt), dtype=dtypes.float32).cast(dtypes.bfloat16)
C_asm = Tensor.empty_like(C_tiny)
C_asm.uop.buffer.allocate()

# ** run gemms

sched = C_tiny.schedule()
assert len(sched) == 1
eis:list[ExecItem] = [sched[-1].lower()]
ast = eis[0].ast
src:str = (pathlib.Path(__file__).parent/"gemm.s").read_text()
prg = ProgramSpec("gemm", src, Device.DEFAULT, ast, global_size=[128, 86, 1], local_size=[256, 1, 1], globals=[0, 1, 2])
eis.append(ExecItem(ast, [C_asm.uop.buffer, from_torch(A).uop.buffer, from_torch(B).uop.buffer], prg=CompiledRunner(prg)))

for ei in eis: ei.run(wait=True)

# ** correctness

import ctypes, torch

def torch_bf16(t:Tensor) -> torch.tensor:
  asm_out = t.to("cpu").realize().uop.buffer._buf
  buf = (ctypes.c_uint16*C_asm.uop.size).from_address(asm_out.va_addr)
  return torch.frombuffer(buf, dtype=torch.bfloat16, count=C_asm.uop.size).reshape(C_asm.shape)

assert torch.allclose(torch_bf16(C_asm), C_torch, rtol=1e-2, atol=1e-3)
assert torch.allclose(torch_bf16(C_tiny), C_torch, rtol=1e-2, atol=1e-3)

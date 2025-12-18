import pathlib, ctypes
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import system, temp

# ** assemble

asm = pathlib.Path(__file__).parent.parent/"gemm.s"
system(f"clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=5 -c {str(asm)} -o {temp('test.o')}")
system(f"ld.lld -shared -o {temp('test.hsaco')} {temp('test.o')}")
with open(temp('test.hsaco'), 'rb') as f: lib:bytes = f.read()
name:str = "gemm"

dev = Device[Device.DEFAULT]

N = 8192
scale = 10.0

# ** generate random inputs with torch

import torch
dtype = torch.bfloat16
torch.manual_seed(0)
A = (torch.randn(N, N, dtype=torch.float32, device='cpu') / scale).to(dtype)
B = (torch.randn(N, N, dtype=torch.float32, device='cpu') / scale).to(dtype)
ref_out = A@B.t().contiguous()

A = A.contiguous()
B = B.contiguous()

# ** tinygrad

A_tiny = [Tensor.from_blob(A.data_ptr(), A.shape, dtype=dtypes.bfloat16, device="cpu").to(Device.DEFAULT).realize() for _ in range(2)]
B_tiny = [Tensor.from_blob(B.data_ptr(), B.shape, dtype=dtypes.bfloat16, device="cpu").to(Device.DEFAULT).realize() for _ in range(2)]
C_tiny = Tensor.matmul(A_tiny[0], B_tiny[0].T.contiguous(), dtype=dtypes.float32).cast(dtypes.bfloat16)
C_asm = Tensor.empty_like(C_tiny)

# ** run

prg = dev.runtime(name, lib)
et = prg(*[b._buf for b in [C_asm.uop.buffer.ensure_allocated(), A_tiny[1].uop.buffer, B_tiny[1].uop.buffer]], global_size=[128, 86, 1], local_size=[256, 1, 1], wait=True)
print(f"gemm finished in {et*1e3:9.2f} ms")

# ** correctness

def torch_bf16(t:Tensor):
  asm_out = t.to("cpu").realize().uop.buffer._buf
  buf = (ctypes.c_uint16*C_asm.uop.size).from_address(asm_out.va_addr)
  return torch.frombuffer(buf, dtype=torch.bfloat16, count=C_asm.uop.size).reshape(C_asm.shape)

assert torch.allclose(torch_bf16(C_asm), ref_out, rtol=1e-2, atol=1e-3)
assert torch.allclose(torch_bf16(C_tiny), ref_out, rtol=1e-2, atol=1e-3)

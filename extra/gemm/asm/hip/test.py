import pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import system, temp

# ** assemble

asm = pathlib.Path(__file__).parent.parent/"gemm.s"
system(f"clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=5 -c {str(asm)} -o {temp('test.o')}")
system(f"ld.lld -shared -o {temp('test.hsaco')} {temp('test.o')}")
with open(temp('test.hsaco'), 'rb') as f: lib:bytes = f.read()
name:str = "gemm"

Device.DEFAULT = "HIP"
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

# bitcast to uint16 since there's no bf16 on numpy
A, B = [t.view(torch.uint16).numpy() for t in [A, B]]

out = Tensor.empty(N, N, dtype=dtypes.uint16).uop.buffer.allocate()
bufs = [b._buf for b in [out, Tensor(A).realize().uop.buffer, Tensor(B).realize().uop.buffer]]

# ** run

prg = dev.runtime(name, lib)
et = prg(*bufs, global_size=[128, 86, 1], local_size=[256, 1, 1], wait=True)
print(f"gemm finished in {et*1e3:9.2f} ms")

# ** correctness

import torch
asm_out = torch.from_numpy(out.numpy()).view(torch.bfloat16).reshape(ref_out.shape)
print(asm_out)
assert torch.allclose(asm_out, ref_out, rtol=1e-2, atol=1e-3)

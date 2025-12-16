import pathlib
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import lower_schedule_item, ExecItem, CompiledRunner

N = 8192

# ** pre generate random inputs with torch

root = pathlib.Path("/tmp/gemm_inputs")
if not (root/"A").exists():
  import torch
  scale = 10.0
  dtype = torch.bfloat16
  torch.manual_seed(0)
  A = (torch.randn(N, N, dtype=torch.float32, device='cpu') / scale).to(dtype)
  B = (torch.randn(N, N, dtype=torch.float32, device='cpu') / scale).to(dtype)
  C = A@B.t().contiguous()
  A.contiguous().view(torch.uint16).cpu().numpy().tofile(root/"A")
  B.contiguous().view(torch.uint16).cpu().numpy().tofile(root/"B")
  C.contiguous().view(torch.uint16).cpu().numpy().tofile(root/"C")
  print("Saved tensors to", root)

dtype = dtypes.bfloat16
A = Tensor(np.fromfile(root/"A", dtype=np.uint16)).bitcast(dtype).reshape(N, N).contiguous().realize()
B = Tensor(np.fromfile(root/"B", dtype=np.uint16)).bitcast(dtype).reshape(N, N).contiguous().realize()
C = Tensor(np.fromfile(root/"C", dtype=np.uint16)).bitcast(dtype).reshape(N, N).contiguous().realize()

# ** tinygrad gemm

C_tiny = Tensor.matmul(A, B.T, dtype=dtypes.float32).cast(dtype)
si = C_tiny.schedule()[-1]
eis:list[ExecItem] = [lower_schedule_item(si)]

C_asm = Tensor.empty_like(C)
C_asm.uop.buffer.allocate()

# ** raw assembly gemm

with open(pathlib.Path(__file__).parent/"lib", "rb") as f: lib = f.read()
prg = CompiledRunner(precompiled=lib, p=replace(eis[0].prg.p, src=lib, name="matmul", global_size=(128, 86, 1), local_size=(256, 1, 1)))
#Device[Device.DEFAULT].compiler.disassemble(lib)
eis.append(ExecItem(prg, [C_asm.uop.buffer, A.uop.buffer, B.uop.buffer]))

for ei in eis: ei.run(wait=True)

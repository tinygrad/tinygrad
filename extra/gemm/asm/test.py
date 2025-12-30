# Run assembly on the AMD runtime and check correctness
# VIZ=2 to profile
import pathlib
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import track_rewrites, UOp
from tinygrad.helpers import TracingKey, getenv

fp = pathlib.Path(__file__).parent/"gemm.s"

N = getenv("N", 8192)
THREADS_PER_WG = 256
NUM_WG = N//THREADS_PER_WG * N//THREADS_PER_WG

assert N % THREADS_PER_WG == 0, "N must be divisible by THREADS_PER_WG"

# ** generate inputs on CPU

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

# baseline tinygrad
sched = C_tiny.schedule()
assert len(sched) == 1
eis:list[ExecItem] = [sched[-1].lower()]
ast = sched[-1].ast

# assembly gemm
@track_rewrites(name=lambda ret: TracingKey(ret.name, (ret.function_name,), ret))
def get_asm_prg() -> ProgramSpec:
  src = fp.read_text()
  lib = Device[Device.DEFAULT].compiler.compile(src)
  return ProgramSpec("gemm", src, Device.DEFAULT, ast, lib=lib, global_size=[NUM_WG, 1, 1], local_size=[THREADS_PER_WG, 1, 1],
                     globals=[0, 1, 2], vars=[UOp.variable("SZ", 256, 8192), UOp.variable("NUM_WG", 1, 1024)])
eis.append(ExecItem(ast, [C_asm.uop.buffer, from_torch(B).uop.buffer, from_torch(A).uop.buffer], fixedvars={"SZ":N, "NUM_WG":NUM_WG},
                    prg=CompiledRunner(get_asm_prg())))

with Context(DEBUG=2):
  for ei in eis:
    et = ei.run(wait=True)
    print(f"{(N*N*N*2 / et)*1e-12:.2f} REAL TFLOPS")

# ** correctness

import ctypes

def torch_bf16(t:Tensor) -> torch.tensor:
  asm_out = t.to("cpu").realize().uop.buffer._buf
  buf = (ctypes.c_uint16*C_asm.uop.size).from_address(asm_out.va_addr)
  return torch.frombuffer(buf, dtype=torch.bfloat16, count=C_asm.uop.size).reshape(C_asm.shape)

assert torch.allclose(torch_bf16(C_asm), C_torch, rtol=1e-2, atol=1e-3)
assert torch.allclose(torch_bf16(C_tiny), C_torch, rtol=1e-2, atol=1e-3)

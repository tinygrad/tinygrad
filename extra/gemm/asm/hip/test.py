import pathlib, ctypes
from tinygrad import Tensor, Device, dtypes
from extra.gemm.asm.hip.arg import KernelArgs
from tinygrad.helpers import system, temp, getenv

# ** assemble

if getenv("ASM", 1): # re assemble
  asm = pathlib.Path(__file__).parent.parent/"gemm.s"
  system(f"clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=5 -c {str(asm)} -o {temp('test.o')}")
  system(f"ld.lld -shared -o {temp('test.hsaco')} {temp('test.o')}")
  with open(temp('test.hsaco'), 'rb') as f: lib:bytes = f.read()
  name:str = "gemm"
else: # the literal hsaco dump in the aiter repo, keep for reference
  with open(pathlib.Path(__file__).parent.parent/"lib", "rb") as f: lib:bytes = f.read()
  name:str = "_ZN5aiter37bf16gemm_fp32bf16_tn_96x64_pf3_splitkE"

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

# ** construct launch args

def build_kernel_args(bufs):
  # bufs: [out, A, B]
  args = KernelArgs()
  args.ptr_D = bufs[0].value
  args.ptr_C = 0
  args.ptr_A = bufs[1].value
  args.ptr_B = bufs[2].value
  args.ptr_Bias = 0

  args.alpha = 1.0
  args.beta = 0.0

  ld_bytes = N * 2  # bf16/u16
  args.stride_D0 = 0
  args.stride_D1 = 0
  args.stride_C0 = ld_bytes
  args.stride_C1 = 0
  args.stride_A0 = ld_bytes
  args.stride_A1 = 0
  args.stride_B0 = ld_bytes
  args.stride_B1 = 0

  args.M = N
  args.N = N
  args.K = N
  args.splitk = 1
  args.is_out_b16 = 1
  args.add_bias = 0

  blob = ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args))
  return args, blob

def pack_kernel_args(args:KernelArgs):
  arg_size = ctypes.c_size_t(ctypes.sizeof(args))
  blob = (ctypes.c_ubyte * ctypes.sizeof(args)).from_buffer_copy(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))
  extra = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(blob), ctypes.c_void_p), 2,
                                ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p), 3)
  return extra, blob, arg_size  # keepalives: blob + arg_size

out = Tensor.empty(N, N, dtype=dtypes.uint16).uop.buffer.allocate()
bufs = [b._buf for b in [out, Tensor(A).realize().uop.buffer, Tensor(B).realize().uop.buffer]]
args, _ = build_kernel_args(bufs)
extra, _blob_keep, _sz_keep = pack_kernel_args(args)

# ** run

prg = dev.runtime(name, lib)
prg.vargs = extra
et = prg(global_size=[128, 86, 1], local_size=[256, 1, 1], wait=True)
print(f"gemm finished in {et*1e3:9.2f} ms")

# ** correctness

import torch
asm_out = torch.from_numpy(out.numpy()).view(torch.bfloat16).reshape(ref_out.shape)
print(asm_out)
assert torch.allclose(asm_out, ref_out, rtol=1e-2, atol=1e-3)

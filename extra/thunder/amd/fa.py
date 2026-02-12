import math, pathlib, functools, time, struct

from tinygrad import Device, Tensor
from tinygrad.dtype import DTypeLike, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context, DEBUG
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

import numpy as np

def _sharded_empty(shape:Tensor, ref:Tensor, axis:int|None, dtype:DTypeLike|None=None) -> Tensor:
  dtype = dtype or ref.dtype
  if not isinstance(ref.device, tuple): return Tensor.empty(*shape, dtype=dtype, device=ref.device)
  shape = tuple(s // len(ref.device) if i == ref.uop.axis else s for i, s in enumerate(shape))
  axis = ref.uop.axis if axis is None else axis
  return Tensor(Tensor.empty(*shape, dtype=dtype, device=ref.device).uop.multi(axis), dtype=dtype, device=ref.device)

def _sharded_empty_like(ref:Tensor, axis:int|None=None) -> Tensor:
  return _sharded_empty(ref.shape, ref, axis)

def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False):
  assert attn_mask is None, "attn_mask not supported"
  assert is_causal, "only causal attention supported"

  B, N, H, D = xq.shape
  H_KV = xk.shape[2]
  assert D == 128, "only D=128 supported"

  num_devices = len(xq.device) if isinstance(xq.device, tuple) else 1
  B_local = B // num_devices
  if DEBUG >= 2: print(f"Flash Attention {B=} {B_local=} {N=} {H=} {H_KV=} {D=}")

  single_device = xq.device[0] if isinstance(xq.device, tuple) else xq.device
  arch = Device[single_device].renderer.arch

  attn = _sharded_empty_like(xq, axis=0)
  l_vec = _sharded_empty((B, H, 1, N), xq, axis=0, dtype=dtypes.float32)

  attn, l_vec = Tensor.custom_kernel(attn, l_vec, xq, xk, xv, fxn=functools.partial(custom_fa_forward, device=single_device, arch=arch))[:2]

  return attn

@functools.cache
def custom_fa_forward(o:UOp, l_vec:UOp, q:UOp, k:UOp, v:UOp, device:str, arch:str):
  B, N, H, _ = q.shape
  H_KV = k.shape[2]

  code = (pathlib.Path(__file__).parent / "fa_fwd_causal.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}"]

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 8
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H, (math.ceil((N // Q_BLOCK_SIZE) / NUM_WARPS)), B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x = UOp.special(gsz[0], "gidx0")
  blockIdx_y = UOp.special(gsz[1], "gidx1")
  blockIdx_z = UOp.special(gsz[2], "gidx2")

  sink = UOp.sink(o.base, l_vec.base, q.base, k.base, v.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_forward"))
  lib = HIPCCCompiler(arch, compile_args).compile(code)

  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

if __name__ == "__main__":
  B, N, H, H_KV, D = 16, 8192, 32, 8, 128
  q = Tensor.randn(B, N, H, D, device="AMD", dtype="bfloat16").contiguous()
  k = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  v = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  Tensor.realize(q, k, v)

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 8
  NUM_THREADS = 64 * NUM_WARPS

  fa_jitted = TinyJit(flash_attention)

  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  for _ in range(5):
    st = time.perf_counter()
    out = fa_jitted(q, k, v, is_causal=True)
    Device["AMD"].synchronize()
    et = time.perf_counter() - st
    print(f"{attn_flops/(et*1e12):2f} TFLOPS")

  with Context(DEBUG=2):
    ref = q.transpose(1,2).scaled_dot_product_attention(k.transpose(1,2), v.transpose(1,2), is_causal=True, enable_gqa=True).transpose(1,2)

  ref_np, out_np = ref.float().numpy(), out.float().numpy()
  np.testing.assert_allclose(ref_np, out_np, atol=2e-2, rtol=1e-2)

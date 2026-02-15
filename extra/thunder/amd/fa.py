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

  xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

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

  def grad(dou:UOp, _) -> tuple[None, None, UOp, UOp, UOp]:
    do = Tensor(dou, device=dou.device)
    dq_in = _sharded_empty((B, H, N, D), xq, axis=0)
    dk = _sharded_empty_like(xk, axis=0)
    dv = _sharded_empty_like(xv, axis=0)

    # delta_vec = (do * attn).sum(-1, dtype=dtypes.float32).transpose(1, 2).unsqueeze(-2).detach()
    delta_vec = _sharded_empty((B, H, 1, N), xq, axis=0, dtype=dtypes.float32)
    delta_vec, dq_in = Tensor.custom_kernel(delta_vec, dq_in, attn, do, fxn=functools.partial(custom_fa_backward_pre, device=single_device, arch=arch))[:2]

    dq_in, dk, dv = Tensor.custom_kernel(dq_in, dk, dv, do, xq, xk, xv, l_vec, delta_vec, fxn=functools.partial(custom_fa_backward, device=single_device, arch=arch))[:3]

    # unshuffle dq
    dq = _sharded_empty_like(xq, axis=0)
    dq = Tensor.custom_kernel(dq, dq_in, fxn=functools.partial(custom_fa_backward_post, device=single_device, arch=arch))[0]

    return None, None, dq.uop, dk.uop, dv.uop

  attn, l_vec = Tensor.custom_kernel(attn, l_vec, xq, xk, xv, fxn=functools.partial(custom_fa_forward, device=single_device, arch=arch), grad_fxn=grad)[:2]

  return attn.transpose(1, 2)

@functools.cache
def custom_fa_forward(o:UOp, l_vec:UOp, q:UOp, k:UOp, v:UOp, device:str, arch:str):
  B, N, H, _ = q.shape
  H_KV = k.shape[2]

  code = (pathlib.Path(__file__).parent / "fa_fwd_causal.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}"]

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 8
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H, (math.ceil((N // Q_BLOCK_SIZE) / NUM_WARPS)), B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

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

@functools.cache
def custom_fa_backward_pre(delta_vec:UOp, dq:UOp, o:UOp, do:UOp, device:str, arch:str):
  B, N, H, _ = o.shape

  code = (pathlib.Path(__file__).parent / "fa_bwd_pre.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  sink = UOp.sink(delta_vec.base, dq.base, o.base, do.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_pre"))

  lib = HIPCCCompiler(arch, compile_args).compile(code)

  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward(dq:UOp, dk:UOp, dv:UOp, do:UOp, q:UOp, k:UOp, v:UOp, l_vec:UOp, delta_vec:UOp, device:str, arch:str):
  B, N, H, _ = q.shape
  H_KV = k.shape[2]

  code = (pathlib.Path(__file__).parent / "fa_bwd_causal.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}"]

  BLOCK_SIZE_KV = 256
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H_KV, N // BLOCK_SIZE_KV, B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  sink = UOp.sink(dq.base, dk.base, dv.base, do.base, q.base, k.base, v.base, l_vec.base, delta_vec.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward"))

  lib = HIPCCCompiler(arch, compile_args).compile(code)

  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward_post(dq_out:UOp, dq_in:UOp, device:str, arch:str):
  B, N, H, _ = dq_out.shape

  code = (pathlib.Path(__file__).parent / "fa_bwd_post.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  sink = UOp.sink(dq_out.base, dq_in.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_post"))

  lib = HIPCCCompiler(arch, compile_args).compile(code)

  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

if __name__ == "__main__":
  B, N, H, H_KV, D = 1, 8192, 32, 8, 128
  # q = Tensor.randn(B, N, H, D, device="AMD", dtype="bfloat16").contiguous()
  # k = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  # v = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  # Tensor.realize(q, k, v)
  #
  # Q_BLOCK_SIZE = 32
  # NUM_WARPS = 8
  # NUM_THREADS = 64 * NUM_WARPS
  #
  # fa_jitted = TinyJit(flash_attention)
  #
  attn_flops = 2 * B * H * N * N * D + \
               4 * B * H * N * N + \
               2 * B * H * N * N * D
  # for _ in range(5):
  #   st = time.perf_counter()
  #   out = fa_jitted(q, k, v, is_causal=True)
  #   Device["AMD"].synchronize()
  #   et = time.perf_counter() - st
  #   print(f"{attn_flops/(et*1e12):2f} TFLOPS")
  #
  # ref = q.transpose(1,2).scaled_dot_product_attention(k.transpose(1,2), v.transpose(1,2), is_causal=True, enable_gqa=True).transpose(1,2)
  #
  # ref_np, out_np = ref.float().numpy(), out.float().numpy()
  # np.testing.assert_allclose(ref_np, out_np, atol=2e-2, rtol=1e-2)

  # backward pass
  q2 = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
  k2 = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
  v2 = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
  Tensor.realize(q2, k2, v2)
  do = Tensor.ones(B, N, H, D, dtype=dtypes.bfloat16).contiguous()

  bwd_flops = attn_flops * 5 // 2  # backward is ~2.5x forward flops

  @TinyJit
  def fa_bwd(q, k, v, do):
    out = flash_attention(q, k, v, is_causal=True)
    out.backward(do)
    Tensor.realize(out, q.grad, k.grad, v.grad)
    return q.grad, k.grad, v.grad

  for _ in range(5):
    st = time.perf_counter()
    q2.grad, k2.grad, v2.grad = fa_bwd(q2, k2, v2, do)
    Device["AMD"].synchronize()
    et = time.perf_counter() - st
    print(f"bwd: {bwd_flops/(et*1e12):2f} TFLOPS")

  # reference backward
  q_ref = q2.detach().clone().requires_grad_(True)
  k_ref = k2.detach().clone().requires_grad_(True)
  v_ref = v2.detach().clone().requires_grad_(True)
  Tensor.realize(q_ref, k_ref, v_ref)

  q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
  ref = q_ref_.scaled_dot_product_attention(k_ref_, v_ref_, is_causal=True, enable_gqa=True).float().transpose(1, 2)
  ref.sum().backward()
  Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

  np.testing.assert_allclose(q2.grad.numpy(), q_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
  np.testing.assert_allclose(v2.grad.numpy(), v_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
  np.testing.assert_allclose(k2.grad.numpy(), k_ref.grad.numpy(), atol=2e-2, rtol=2e-2)

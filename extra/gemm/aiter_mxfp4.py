import functools, pathlib, struct
from dataclasses import replace
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
def aiter_mxfp4_kernargs(M:int, N:int, K:int) -> bytes:
  args = bytearray(384)
  struct.pack_into("<f", args, 64, 1.0)
  struct.pack_into("<f", args, 80, 0.0)
  for offset, value in ((96, N), (112, 1), (128, N), (144, 1),
                        (160, K), (176, 1), (192, K), (208, 1),
                        (224, M), (240, N), (256, K),
                        (304, K // 32), (320, 1), (336, K // 32), (352, 1), (368, 0)):
    struct.pack_into("<I", args, offset, value)
  return bytes(args)
def _aiter_mxfp4_program_info(sink:UOp, M:int, N:int, K:int, tile_m:int, tile_n:int) -> ProgramInfo:
  info = ProgramInfo.from_sink(sink)
  return replace(info, global_size=(ceildiv(N, tile_n), ceildiv(M, tile_m), 1), local_size=(256, 1, 1),
                 outs=(info.globals[0],), ins=info.globals[1:],
                 aux=("packed", aiter_mxfp4_kernargs(M, N, K), (0, 32, 48, 272, 288)))
@functools.cache
def _custom_aiter_mxfp4(C:UOp, A:UOp, B:UOp, scale_a:UOp, scale_b:UOp, tile_m:int, tile_n:int) -> UOp:
  M, half_k = A.shape
  N, half_k_b = B.shape
  K = half_k * 2
  assert half_k == half_k_b and C.shape == (M, N)
  threads = UOp.special(256, "lidx0")
  groups_x, groups_y = UOp.special(ceildiv(N, tile_n), "gidx0"), UOp.special(ceildiv(M, tile_m), "gidx1")
  sink = UOp.sink(C.base, A.base, B.base, scale_a.base, scale_b.base, threads, groups_x, groups_y,
                  arg=KernelInfo(f"aiter_mxfp4_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K)))
  root = pathlib.Path(__file__).parent/"amd"/"aiter_f4"
  stem = f"f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_m}x{tile_n}"
  info = replace(_aiter_mxfp4_program_info(sink, M, N, K, tile_m, tile_n), target=Device[C.device].renderer.target)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=f"AITER gfx950 opaque code object: {stem}"),
                               UOp(Ops.BINARY, arg=(root/f"{stem}.co").read_bytes())),
             arg=info)
def shuffle_mxfp4_weight(x:Tensor) -> Tensor:
  assert x.ndim == 2 and x.dtype == dtypes.uint8
  N, half_k = x.shape
  assert N % 16 == 0 and half_k % 32 == 0
  return x.reshape(N // 16, 16, half_k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().reshape(N, half_k)
def shuffle_mxfp4_scales(x:Tensor) -> Tensor:
  assert x.ndim == 2 and x.dtype == dtypes.uint8
  rows, scale_k = x.shape
  assert rows % 32 == 0 and scale_k % 8 == 0
  return x.reshape(rows // 32, 2, 16, scale_k // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous().reshape(rows, scale_k)
def aiter_mxfp4_gemm(a:Tensor, b:Tensor, scale_a:Tensor, scale_b:Tensor, tile_m:int=256, tile_n:int=256) -> Tensor:
  assert (tile_m, tile_n) in ((256, 256), (192, 256), (128, 512))
  assert a.ndim == b.ndim == 2 and a.dtype == b.dtype == dtypes.uint8
  M, half_k = a.shape
  N, half_k_b = b.shape
  K = half_k * 2
  assert half_k == half_k_b and M % tile_m == N % tile_n == 0 and K % 256 == 0
  assert scale_a.shape == (M, K // 32) and scale_b.shape == (N, K // 32)
  assert a.device == b.device == scale_a.device == scale_b.device and isinstance(a.device, str)
  assert Device[a.device].renderer.target.arch == "gfx950"
  out = Tensor.invalids(M, N, dtype=dtypes.bfloat16, device=a.device)
  fxn = functools.partial(_custom_aiter_mxfp4, tile_m=tile_m, tile_n=tile_n)
  return Tensor.custom_kernel(out, a, b, scale_a, scale_b, fxn=fxn)[0]

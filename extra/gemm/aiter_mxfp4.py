import ctypes, functools, pathlib
from dataclasses import replace
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
class _AiterMXFP4Args(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("ptr_D", ctypes.c_void_p), ("_p0", ctypes.c_uint32 * 2),
    ("ptr_C", ctypes.c_void_p), ("_p1", ctypes.c_uint32 * 2),
    ("ptr_A", ctypes.c_void_p), ("_p2", ctypes.c_uint32 * 2),
    ("ptr_B", ctypes.c_void_p), ("_p3", ctypes.c_uint32 * 2),
    ("alpha", ctypes.c_float), ("_p4", ctypes.c_uint32 * 3),
    ("beta", ctypes.c_float), ("_p5", ctypes.c_uint32 * 3),
    ("stride_D0", ctypes.c_uint32), ("_p6", ctypes.c_uint32 * 3),
    ("stride_D1", ctypes.c_uint32), ("_p7", ctypes.c_uint32 * 3),
    ("stride_C0", ctypes.c_uint32), ("_p8", ctypes.c_uint32 * 3),
    ("stride_C1", ctypes.c_uint32), ("_p9", ctypes.c_uint32 * 3),
    ("stride_A0", ctypes.c_uint32), ("_p10", ctypes.c_uint32 * 3),
    ("stride_A1", ctypes.c_uint32), ("_p11", ctypes.c_uint32 * 3),
    ("stride_B0", ctypes.c_uint32), ("_p12", ctypes.c_uint32 * 3),
    ("stride_B1", ctypes.c_uint32), ("_p13", ctypes.c_uint32 * 3),
    ("M", ctypes.c_uint32), ("_p14", ctypes.c_uint32 * 3),
    ("N", ctypes.c_uint32), ("_p15", ctypes.c_uint32 * 3),
    ("K", ctypes.c_uint32), ("_p16", ctypes.c_uint32 * 3),
    ("ptr_ScaleA", ctypes.c_void_p), ("_p17", ctypes.c_uint32 * 2),
    ("ptr_ScaleB", ctypes.c_void_p), ("_p18", ctypes.c_uint32 * 2),
    ("stride_ScaleA0", ctypes.c_uint32), ("_p19", ctypes.c_uint32 * 3),
    ("stride_ScaleA1", ctypes.c_uint32), ("_p20", ctypes.c_uint32 * 3),
    ("stride_ScaleB0", ctypes.c_uint32), ("_p21", ctypes.c_uint32 * 3),
    ("stride_ScaleB1", ctypes.c_uint32), ("_p22", ctypes.c_uint32 * 3),
    ("log2_k_split", ctypes.c_int32), ("_p23", ctypes.c_uint32 * 3),
  ]
assert ctypes.sizeof(_AiterMXFP4Args) == 384
_AITER_MXFP4_BUFFER_OFFSETS = tuple(getattr(_AiterMXFP4Args, name).offset for name in ("ptr_D", "ptr_A", "ptr_B", "ptr_ScaleA", "ptr_ScaleB"))
def aiter_mxfp4_kernargs(M:int, N:int, K:int) -> bytes:
  return bytes(_AiterMXFP4Args(alpha=1.0, beta=0.0, stride_D0=N, stride_D1=1, stride_C0=N, stride_C1=1,
                              stride_A0=K, stride_A1=1, stride_B0=K, stride_B1=1, M=M, N=N, K=K,
                              stride_ScaleA0=K//32, stride_ScaleA1=1, stride_ScaleB0=K//32, stride_ScaleB1=1, log2_k_split=0))
def _aiter_mxfp4_program_info(sink:UOp, M:int, N:int, K:int, tile_m:int, tile_n:int) -> ProgramInfo:
  info = ProgramInfo.from_sink(sink)
  return replace(info, global_size=(ceildiv(N, tile_n), ceildiv(M, tile_m), 1), local_size=(256, 1, 1),
                 outs=(info.globals[0],), ins=info.globals[1:],
                 aux=("packed", aiter_mxfp4_kernargs(M, N, K), _AITER_MXFP4_BUFFER_OFFSETS))
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

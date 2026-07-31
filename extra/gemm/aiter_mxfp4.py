import functools, pathlib
from dataclasses import replace
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
def _normalize_mfma_scale_encoding(lib:bytes, tile:tuple[int, int]) -> bytes:
  exceptions = {(128, 512):((16, 0x2000), (80, 0x4000), (82, 0x4000), (88, 0x2000), (306, 0)),
                (192, 256):((63, 0x4000), (65, 0x4000), (67, 0x4000), (68, 0x4000), (69, 0x4000), (71, 0x4000), (72, 0x4000),
                            (75, 0x4000), (76, 0x4000), (77, 0x4000), (79, 0x4000), (80, 0x4000), (85, 0x2000), (90, 0x2000), (211, 0)),
                (256, 256):((58, 0x4000), (72, 0x4000), (76, 0x4000), (80, 0x4000), (82, 0x4000), (84, 0x4000), (88, 0x4000),
                            (90, 0x2000), (93, 0x2000), (123, 0), (124, 0), (189, 0), (190, 0))}
  masks = dict(exceptions[tile])
  out, hits = bytearray(lib), 0
  text = next(x for x in elf_loader(lib)[1] if x.name == ".text").header
  for off in range(text.sh_offset, text.sh_offset+text.sh_size-15, 4):
    w0, w2 = int.from_bytes(out[off:off+4], "little"), int.from_bytes(out[off+8:off+12], "little")
    if w0 & 0xffff0000 == 0xd3ac0000 and w2 & 0xffff0000 == 0xd3ad0000:
      out[off:off+4], hits = (w0 | masks.get(hits, 0x6000)).to_bytes(4, "little"), hits+1
  assert hits == {(128, 512):512, (192, 256):384, (256, 256):512}[tile]
  return bytes(out)
def _aiter_mxfp4_program_info(sink:UOp, M:int, N:int, tile_m:int, tile_n:int) -> ProgramInfo:
  info = ProgramInfo.from_sink(sink)
  return replace(info, global_size=(ceildiv(N, tile_n), ceildiv(M, tile_m), 1), local_size=(256, 1, 1),
                 outs=(info.globals[0],), ins=info.globals[1:])
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
  src = (root/"aiter_mxfp4.s").read_text()
  for name, value in (("TILE", tile_m*1000+tile_n), ("M", M), ("N", N), ("K", K), ("SCALE_K", K//32)):
    src = src.replace(f".set AITER_MXFP4_{name}, 0", f".set AITER_MXFP4_{name}, {value}")
  info = replace(_aiter_mxfp4_program_info(sink, M, N, tile_m, tile_n), target=Device[C.device].renderer.target)
  lib = _normalize_mfma_scale_encoding(Device[C.device].compiler.compile_cached(src), (tile_m, tile_n))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)),
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

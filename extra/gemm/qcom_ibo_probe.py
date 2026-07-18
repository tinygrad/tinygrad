#!/usr/bin/env python3
"""Compile minimal image-buffer kernels and show the generated A630 ISA."""
import numpy as np
import struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import disasm, get_envelope


KERNELS = {
  "half4": r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_only image1d_buffer_t src, __global half4 *dst) {
  int i = get_global_id(0); dst[i] = read_imageh(src, i);
}""",
  "float4": r"""__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_only image1d_buffer_t src, __global float4 *dst) {
  int i = get_global_id(0); dst[i] = read_imagef(src, i);
}""",
  "uint4": r"""__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_only image1d_buffer_t src, __global uint4 *dst) {
  int i = get_global_id(0); dst[i] = read_imageui(src, i);
}""",
  "read_write_half4": r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_write image1d_buffer_t src, __global half4 *dst) {
  int i = get_global_id(0); dst[i] = read_imageh(src, i);
}""",
  "read_write_2d": r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_write image2d_t src, __global half4 *dst) {
  int i = get_global_id(0); dst[i] = read_imageh(src, (int2)(i, 0));
}""",
}


def binfos(lib:bytes, name:str="probe") -> list[tuple[int, int]]:
  u32 = lambda off: struct.unpack_from("<I", lib, off)[0]
  image_desc_off = u32(0x110)
  samp_count = u32(image_desc_off + 0xdc)
  off = (image_desc_off + 0x158 + len(name) + 3) & -4
  off += 8 * samp_count
  ret = []
  while off + 32 <= len(lib):
    vals = struct.unpack_from("<8I", lib, off)
    if vals[0] == 0: break
    ret.append((vals[3] * 4, vals[7]))
    off += vals[0]
  return ret


def main() -> None:
  dev = Device["QCOM"]
  for name, src in KERNELS.items():
    try:
      lib, image_off, image_size, _ = get_envelope(dev, src)
      prg = dev.runtime("probe", bytes(lib), buf_dtypes=[])
      print(f"=== {name}: image={image_size} tex={prg.tex_cnt} ibo={prg.ibo_cnt} samp={prg.samp_cnt} binfos={binfos(bytes(lib))} ===")
      print(disasm(bytes(lib[image_off:image_off+image_size])))
    except Exception as exc:
      print(f"=== {name}: ERROR {type(exc).__name__}: {exc} ===")

  count = 4096
  values = np.random.default_rng(123).standard_normal((count, 4)).astype(np.float16)
  src_buf = Buffer("QCOM", values.size, dtypes.half).allocate()
  dst_buf = Buffer("QCOM", values.size, dtypes.half).allocate()
  src_buf.copyin(memoryview(values).cast("B"))
  src = KERNELS["half4"]
  lib = dev.compiler.compile(src)
  specs = [((0, dtypes.half, (1, count, 4)),), ((1, dtypes.half, None),)]
  prg = dev.runtime("probe", lib, buf_dtypes=specs)
  times = [prg(src_buf._buf, dst_buf._buf, global_size=(count//128, 1, 1),
               local_size=(128, 1, 1), wait=True)*1e3 for _ in range(20)]
  got = np.empty_like(values)
  dst_buf.copyout(memoryview(got).cast("B"))
  print(f"=== half4 runtime: best_ms={min(times):.6f} exact={np.array_equal(got, values)} "
        f"max_abs={float(np.max(np.abs(got.astype(np.float32)-values.astype(np.float32))))} ===")

  count = 147456
  values = np.random.default_rng(456).standard_normal((count, 4)).astype(np.float16)
  src_buf = Buffer("QCOM", values.size, dtypes.half).allocate()
  dst_buf = Buffer("QCOM", values.size, dtypes.half).allocate()
  src_buf.copyin(memoryview(values).cast("B"))
  lib = dev.compiler.compile(KERNELS["read_write_half4"])
  specs = [((0, dtypes.half, (1, count, 4)),), ((1, dtypes.half, None),)]
  prg = dev.runtime("probe", lib, buf_dtypes=specs)
  times = [prg(src_buf._buf, dst_buf._buf, global_size=(count//128, 1, 1),
               local_size=(128, 1, 1), wait=True)*1e3 for _ in range(20)]
  got = np.empty_like(values)
  dst_buf.copyout(memoryview(got).cast("B"))
  print(f"=== read_write_half4 runtime: best_ms={min(times):.6f} exact={np.array_equal(got, values)} "
        f"max_abs={float(np.max(np.abs(got.astype(np.float32)-values.astype(np.float32))))} ===")
  print("expected_head", values[:4].tolist(), "got_head", got[:4].tolist())
  bad = np.flatnonzero(np.any(got != values, axis=1))
  print("first_bad", int(bad[0]) if bad.size else None, "bad_vectors", int(bad.size))


if __name__ == "__main__": main()

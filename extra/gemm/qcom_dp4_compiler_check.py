#!/usr/bin/env python3
"""Check Qualcomm's compiler-generated packed uint8 dot-product instruction."""
import struct, time

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import disasm


def main() -> None:
  source = """__kernel void dp4(__global int *O,__global uint *A,__global uint *B) {
  int i=get_global_id(0); uchar4 a=as_uchar4(A[i]),b=as_uchar4(B[i]);
  O[i]=(int)a.x*(int)b.x+(int)a.y*(int)b.y+(int)a.z*(int)b.z+(int)a.w*(int)b.w;
}"""
  dev = Device["QCOM"]
  lib = dev.compiler.compile(source)
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  print("\n".join(x for x in disasm(lib[image_off:image_off+image_size]).splitlines()
                  if "dp4" in x or "mad" in x or "mul" in x or "stg" in x))
  rng = np.random.default_rng(0)
  n = 131072
  a8, b8 = rng.integers(0, 16, (n, 4), dtype=np.uint8), rng.integers(0, 16, (n, 4), dtype=np.uint8)
  a, b = a8.view(np.uint32).reshape(-1), b8.view(np.uint32).reshape(-1)
  ab, bb, ob = Buffer("QCOM", n, dtypes.uint).allocate(), Buffer("QCOM", n, dtypes.uint).allocate(), Buffer("QCOM", n, dtypes.int).allocate()
  ab.copyin(memoryview(a).cast("B"))
  bb.copyin(memoryview(b).cast("B"))
  prg = dev.runtime("dp4", lib, buf_dtypes=[((0, dtypes.int, None),), ((1, dtypes.uint, None),), ((2, dtypes.uint, None),)])
  times = [prg(ob._buf, ab._buf, bb._buf, global_size=(n//128, 1, 1), local_size=(128, 1, 1), wait=True) for _ in range(20)]
  out = np.empty(n, np.int32)
  ob.copyout(memoryview(out).cast("B"))
  expected = (a8.astype(np.int32)*b8.astype(np.int32)).sum(axis=1)
  print(f"min_us={min(times)*1e6:.3f} max_abs={int(np.max(np.abs(out-expected)))} first={out[:8].tolist()}")


if __name__ == "__main__":
  start = time.perf_counter()
  main()
  print(f"wall_ms={(time.perf_counter()-start)*1e3:.1f}")

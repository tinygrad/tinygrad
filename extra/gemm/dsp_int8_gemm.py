#!/usr/bin/env python3
"""Standalone correctness/throughput harness for the Hexagon HVX int8 GEMM."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


KERNEL = r"""
typedef int int32x32 __attribute__((aligned(128),vector_size(128)));
typedef unsigned char uchar4 __attribute__((aligned(4),vector_size(4)));
typedef signed char char128 __attribute__((aligned(128),vector_size(128)));
typedef unsigned char uchar128 __attribute__((aligned(128),vector_size(128)));
typedef unsigned char uchar256 __attribute__((aligned(256),vector_size(256)));
union V256 { uchar256 vec256; struct { uchar128 lo128, hi128; }; };

__attribute__((noinline)) void gemm(unsigned char * restrict __attribute__((align_value(128))) out,
                                    unsigned char * restrict __attribute__((align_value(128))) weight,
                                    signed char * restrict __attribute__((align_value(128))) activation) {
  for (int n = 0; n < 512; n++) {
    int noff = n << 9;
    for (int mb = 0; mb < 4; mb++) {
      int moff = mb << 7;
      int32x32 acc0 = __builtin_HEXAGON_V6_vd0_128B();
      int32x32 acc1 = __builtin_HEXAGON_V6_vd0_128B();
      int32x32 acc2 = __builtin_HEXAGON_V6_vd0_128B();
      int32x32 acc3 = __builtin_HEXAGON_V6_vd0_128B();
      for (int k4 = 0; k4 < 128; k4++) {
        uchar4 w4 = *((uchar4 *)(weight + noff + (k4 << 2)));
        int aoff = moff + (k4 << 11);
        char128 x0 = *((char128 *)(activation + aoff));
        char128 x1 = *((char128 *)(activation + aoff + 512));
        char128 x2 = *((char128 *)(activation + aoff + 1024));
        char128 x3 = *((char128 *)(activation + aoff + 1536));
        union V256 s01, s23, slo, shi;
        s01.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x1, x0);
        s23.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x3, x2);
        slo.vec256 = __builtin_HEXAGON_V6_vdealvdd_128B(s23.lo128, s01.lo128, 2);
        shi.vec256 = __builtin_HEXAGON_V6_vdealvdd_128B(s23.hi128, s01.hi128, 2);
        uchar128 w = __builtin_HEXAGON_V6_lvsplatw_128B(*((unsigned int *)&w4));
        acc0 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc0, w, slo.lo128);
        acc1 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc1, w, shi.lo128);
        acc2 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc2, w, slo.hi128);
        acc3 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc3, w, shi.hi128);
      }
      acc0 /= 1000; acc1 /= 1000; acc2 /= 1000; acc3 /= 1000;
      uchar128 packed = __builtin_HEXAGON_V6_vpackhub_sat_128B(
        __builtin_HEXAGON_V6_vpackwh_sat_128B(acc3, acc2),
        __builtin_HEXAGON_V6_vpackwh_sat_128B(acc1, acc0));
      packed = __builtin_HEXAGON_V6_vshuffb_128B(packed);
      packed = __builtin_HEXAGON_V6_vshuffb_128B(packed);
      *((uchar128 *)(out + noff + moff)) = packed;
    }
  }
}

struct dcvs_v2_req { int type; int pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency;
  _Bool set_dcvs_params; short pad2; char target_corner; char min_corner; char max_corner; int pad3[3]; };
typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;
int HAP_power_set(void *, void *);
void *HAP_mmap(void *, int, int, int, int, long);
int HAP_munmap(void *, int);
unsigned long long HAP_perf_get_time_us(void);

int entry(unsigned long long handle, unsigned int sc, remote_arg *pra) {
  struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100,
    .set_dcvs_params=1, .target_corner=6};
  HAP_power_set((void *)handle, (void *)&req);
  if ((sc >> 24) != 2) return 0;
  int *sizes = (int *)pra[0].buf.pv, *offs = (int *)pra[1].buf.pv;
  void *out = HAP_mmap(0, sizes[0], 3, 0, pra[3].dma.fd, 0) + offs[0];
  void *weight = HAP_mmap(0, sizes[1], 3, 0, pra[4].dma.fd, 0) + offs[1];
  void *activation = HAP_mmap(0, sizes[2], 3, 0, pra[5].dma.fd, 0) + offs[2];
  unsigned long long start = HAP_perf_get_time_us();
  gemm(out, weight, activation);
  *(unsigned long long *)pra[2].buf.pv = HAP_perf_get_time_us() - start;
  HAP_munmap(out-offs[0], sizes[0]); HAP_munmap(weight-offs[1], sizes[1]); HAP_munmap(activation-offs[2], sizes[2]);
  return 0;
}
"""


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--raw", action="store_true", help="store raw int32 accumulators without requantization")
  args = parser.parse_args()
  dev = Device["DSP"]
  source = KERNEL
  if args.raw:
    source = source.replace(
      "unsigned char * restrict __attribute__((align_value(128))) out,\n                                    unsigned char * restrict",
      "int * restrict __attribute__((align_value(128))) out,\n                                    unsigned char * restrict", 1)
    old = """      acc0 /= 1000; acc1 /= 1000; acc2 /= 1000; acc3 /= 1000;
      uchar128 packed = __builtin_HEXAGON_V6_vpackhub_sat_128B(
        __builtin_HEXAGON_V6_vpackwh_sat_128B(acc3, acc2),
        __builtin_HEXAGON_V6_vpackwh_sat_128B(acc1, acc0));
      packed = __builtin_HEXAGON_V6_vshuffb_128B(packed);
      packed = __builtin_HEXAGON_V6_vshuffb_128B(packed);
      *((uchar128 *)(out + noff + moff)) = packed;"""
    new = """      int base = noff + moff;
      *((int32x32 *)(out + base + 0)) = acc0;
      *((int32x32 *)(out + base + 32)) = acc1;
      *((int32x32 *)(out + base + 64)) = acc2;
      *((int32x32 *)(out + base + 96)) = acc3;"""
    if old not in source: raise RuntimeError("raw-kernel source pattern not found")
    source = source.replace(old, new)
  lib = dev.compiler.compile(source)
  prg = dev.runtime("entry", lib)
  rng = np.random.default_rng(0)
  # Kernel contract is weight[N,K] and activation[K,M], both contiguous.
  weight_np = rng.integers(0, 16, (512, 512), dtype=np.uint8)
  activation_np = rng.integers(-8, 8, (512, 512), dtype=np.int8)
  out_dtype = dtypes.int if args.raw else dtypes.uint8
  bufs = [Buffer("DSP", 512*512, dt, preallocate=True) for dt in (out_dtype, dtypes.uint8, dtypes.int8)]
  bufs[1].copyin(memoryview(weight_np).cast("B"))
  bufs[2].copyin(memoryview(activation_np).cast("B"))
  for _ in range(2): prg(*(x._buf for x in bufs), wait=True)
  times = [prg(*(x._buf for x in bufs), wait=True) for _ in range(args.iters)]
  best = min(times)
  print(f"{2*512**3/best/1e9:.1f} GOPS ({best*1e3:.3f} ms)")
  if args.check:
    raw = bytearray(bufs[0].nbytes)
    bufs[0].copyout(memoryview(raw))
    expected_dot = weight_np.astype(np.int32) @ activation_np.astype(np.int32)
    if args.raw:
      got = np.frombuffer(raw, dtype=np.int32).reshape(512, 4, 4, 32).transpose(0, 1, 3, 2).reshape(512, 512)
      expected = expected_dot
      delta = np.abs(got.astype(np.int64)-expected.astype(np.int64))
    else:
      got = np.frombuffer(raw, dtype=np.uint8).reshape(512, 512)
      expected = (expected_dot // 1000).clip(0, 255).astype(np.uint8)
      delta = np.abs(got.astype(np.int16)-expected.astype(np.int16))
    print(f"check={np.array_equal(got, expected)} max_abs={delta.max()} mismatches={np.count_nonzero(delta)}")
    if not np.array_equal(got, expected): raise SystemExit(1)


if __name__ == "__main__": main()

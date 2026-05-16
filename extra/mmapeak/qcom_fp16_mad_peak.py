#!/usr/bin/env python3
"""FP16/FP32 MAD peak repro for comparing DEV=CL and DEV=QCOM.

Example:
  DEV=CL python3 extra/mmapeak/qcom_fp16_mad_peak.py
  DEV=QCOM python3 extra/mmapeak/qcom_fp16_mad_peak.py --dtype fp32
"""
from __future__ import annotations

import argparse

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


MAD_OPS_PER_LOOP = 16
VEC = 16


def kernel_name(dtype:str) -> str:
  return f"{dtype}_mad_peak"


def make_kernel(loops:int, dtype:str="fp16") -> str:
  assert dtype in {"fp16", "fp32"}
  scalar = "half" if dtype == "fp16" else "float"
  vec_type = f"{scalar}{VEC}"
  prefix = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" if dtype == "fp16" else ""
  cast = "(half)" if dtype == "fp16" else ""
  suffix = "f"
  mad_block = "\n".join([
    "    x = mad(y, x, y);",
    "    y = mad(x, y, x);",
  ] * (MAD_OPS_PER_LOOP // 2))

  x_init = ",\n    ".join(f"bx + {cast}{(i + 1) * 0.001:.3f}{suffix}" for i in range(VEC))
  y_init = ",\n    ".join(f"by + {cast}{(i + 17) * 0.001:.3f}{suffix}" for i in range(VEC))
  sum_terms = " + ".join([f"x.s{'0123456789abcdef'[i]}" for i in range(VEC)] +
                         [f"y.s{'0123456789abcdef'[i]}" for i in range(VEC)])
  return f"""{prefix}__kernel void {kernel_name(dtype)}(__global {scalar} *out) {{
  int lid = get_local_id(0);
  int gid = get_group_id(0);
  {scalar} bx = {cast}1.0f + {cast}(lid & 15) * {cast}0.001f;
  {scalar} by = {cast}1.0f + {cast}(gid & 15) * {cast}0.001f;
  {vec_type} x = ({vec_type})(
    {x_init});
  {vec_type} y = ({vec_type})(
    {y_init});

  for (int i = 0; i < {loops}; i++) {{
{mad_block}
  }}

  out[get_global_id(0)] = {sum_terms};
}}"""


def run(args:argparse.Namespace) -> None:
  dev = Device[Device.DEFAULT]
  renderer = type(dev.renderer).__name__
  if renderer == "IR3Renderer":
    raise SystemExit("This repro uses OpenCL source. Use DEV=QCOM or DEV=CL, not DEV=QCOM:IR3.")

  dtype = args.dtype
  dt = dtypes.half if dtype == "fp16" else dtypes.float
  src = make_kernel(args.loops, dtype)
  if args.print_source: print(src)
  lib = dev.compiler.compile_cached(src)
  if args.disasm: dev.compiler.disassemble(lib)

  # Runtime aux mirrors OpenCLRenderer.aux: one __global output pointer at kernel arg 0.
  global_size = (args.groups, 1, 1)
  local_size = (args.local, 1, 1)
  workitems = args.groups * args.local
  flops = workitems * args.loops * MAD_OPS_PER_LOOP * VEC * 2

  prg = dev.runtime(kernel_name(dtype), lib, (((0, dt.ptr()),),))
  out = Buffer(dev.device, workitems, dt, preallocate=True)

  for _ in range(args.warmup):
    prg(out._buf, global_size=global_size, local_size=local_size, wait=True)

  times = [prg(out._buf, global_size=global_size, local_size=local_size, wait=True) for _ in range(args.iters)]
  best = min(t for t in times if t is not None)
  out_bits = out.copyout(memoryview(bytearray(out.nbytes))).cast("H" if dtype == "fp16" else "I")[0]
  out_fmt = "04x" if dtype == "fp16" else "08x"

  print(f"device={dev.device} renderer={renderer} arch={dev.arch}")
  print(f"dtype={dtype} groups={args.groups} local={args.local} workitems={workitems} loops={args.loops} flops={flops}")
  print(f"best={best*1e6:.2f} us  {dtype}_mad_peak={flops / best * 1e-9:.2f} GFLOPS  out0=0x{out_bits:{out_fmt}}")
  if args.show_times:
    print("times_us=" + ",".join(f"{t*1e6:.2f}" for t in times if t is not None))


def main() -> None:
  parser = argparse.ArgumentParser(description="FP16/FP32 MAD peak repro for DEV=CL vs DEV=QCOM")
  parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16", help="MAD datatype")
  parser.add_argument("--groups", type=int, default=2048, help="number of workgroups")
  parser.add_argument("--local", type=int, default=256, help="workitems per workgroup")
  parser.add_argument("--loops", type=int, default=8, help="inner loop count; default matches clpeak vec16")
  parser.add_argument("--warmup", type=int, default=2, help="warmup launches")
  parser.add_argument("--iters", type=int, default=10, help="timed launches")
  parser.add_argument("--show-times", action="store_true", help="print every timed launch")
  parser.add_argument("--print-source", action="store_true", help="print generated OpenCL source")
  parser.add_argument("--disasm", action="store_true", help="call the tinygrad compiler disassembler after compile")
  run(parser.parse_args())


if __name__ == "__main__":
  main()

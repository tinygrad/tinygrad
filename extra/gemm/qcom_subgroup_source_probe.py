#!/usr/bin/env python3
"""Compile OpenCL subgroup builtins and print the emitted A630 instructions."""
from tinygrad import Device
from extra.gemm.ir3asm import get_envelope, disasm


TEMPLATES = {
  "broadcast": "uint y=sub_group_broadcast(x,0);",
  "shuffle": "uint y=sub_group_shuffle(x,0);",
  "intel_shuffle": "uint y=intel_sub_group_shuffle(x,0);",
}


def main() -> None:
  dev = Device["QCOM"]
  for name, body in TEMPLATES.items():
    src = f"""#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(__global uint *out) {{
  uint x=get_local_id(0)+1; {body} out[get_local_id(0)]=y;
}}"""
    try:
      lib, io, sz, _ = get_envelope(dev, src)
      print(f"=== {name} bytes={sz} ===")
      print(disasm(bytes(lib[io:io+sz])))
    except Exception as exc:
      print(f"=== {name} ERROR {type(exc).__name__}: {exc} ===")


if __name__ == "__main__": main()

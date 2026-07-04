#!/usr/bin/env python3
"""Capture small RDNA3 SQTT traces for emulator timing comparison.

Run on an RDNA3 AMD machine from the tinygrad repo:
  DEBUG=0 DEV=AMD python extra/sqtt/capture_rdna3_sqtt.py
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import decode
from tinygrad.runtime.autogen.amd.rdna3.ins import *

def _assemble(instructions: list) -> bytes:
  return b"".join(inst.to_bytes() for inst in instructions)

KERNELS = {
  "salu_valu_branch": [
    s_mov_b32(s[0], 2),
    v_mov_b32_e32(v[0], 1),
    s_sub_u32(s[0], s[0], 1),
    v_add_nc_u32_e32(v[1], v[0], v[0]),
    s_cmp_lg_u32(s[0], 0),
    s_cbranch_scc1(simm16=-4),
    s_endpgm(),
  ],
  "two_wave_pipe": [
    s_mov_b32(s[0], 0),
    v_mov_b32_e32(v[0], 1),
    s_endpgm(),
  ],
}

def _asm_source(name: str, code: bytes) -> str:
  byte_str = ", ".join(f"0x{b:02x}" for b in code)
  return f""".text
.globl {name}
.p2align 8
.type {name},@function
{name}:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
  .amdhsa_wavefront_size32 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 8
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 64
...
.end_amdgpu_metadata
"""

def _packet_to_dict(packet) -> dict:
  out = {"type": type(packet).__name__, "time": packet._time}
  for name in ("wave", "simd", "wgp", "cu", "se", "delta", "id7"):
    if hasattr(packet, name): out[name] = getattr(packet, name)
  if hasattr(packet, "op"): out["op"] = getattr(packet.op, "name", str(packet.op))
  return out

def _capture_kernel(name: str, instructions: list, local_size: tuple[int, int, int]) -> dict:
  from tinygrad.device import Compiled, Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  code = _assemble(instructions)
  dev = Device["AMD"]
  compiler = HIPCompiler(dev.arch)
  Compiled.profile_events.clear()
  lib = compiler.compile(_asm_source(name, code))
  prg = AMDProgram(dev, name, lib)
  prg(global_size=(1, 1, 1), local_size=local_size, wait=True)

  program_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"]
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
  return {
    "name": name,
    "local_size": local_size,
    "instructions": [repr(x) for x in instructions],
    "kernel_bytes_hex": code.hex(),
    "elf_md5": hashlib.md5(lib).hexdigest(),
    "program_events": [{"name": e.name, "base": e.base, "tag": e.tag, "lib_size": len(e.lib or b"")} for e in program_events],
    "sqtt_events": [{
      "se": e.se,
      "itrace": e.itrace,
      "blob_size": len(e.blob),
      "blob_b64": base64.b64encode(e.blob).decode(),
      "packets": [_packet_to_dict(p) for p in decode(e.blob) if type(p).__name__ != "NOP"],
    } for e in sqtt_events],
  }

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--kernel", choices=tuple(KERNELS) + ("all",), default="all")
  parser.add_argument("--simd", type=int, default=0)
  args = parser.parse_args()

  selected = KERNELS.items() if args.kernel == "all" else [(args.kernel, KERNELS[args.kernel])]
  with Context(PROFILE=1, SQTT=1, SQTT_LIMIT_SE=1, SQTT_ITRACE_SE_MASK=1, SQTT_SIMD_SEL=args.simd):
    from tinygrad.device import Device
    dev = Device["AMD"]
    result = {
      "env": {k: os.environ.get(k) for k in ("DEV", "DEBUG", "PROFILE", "SQTT", "SQTT_LIMIT_SE", "SQTT_ITRACE_SE_MASK", "SQTT_SIMD_SEL")},
      "device": {"arch": dev.arch, "target": dev.target, "device": dev.device},
      "kernels": [],
    }
    if dev.target[0] != 11:
      result["warning"] = f"expected RDNA3 gfx11xx hardware, got arch={dev.arch} target={dev.target}"
    for name, instructions in selected:
      local = (64, 1, 1) if name == "two_wave_pipe" else (32, 1, 1)
      result["kernels"].append(_capture_kernel(name, instructions, local))
  print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
  main()

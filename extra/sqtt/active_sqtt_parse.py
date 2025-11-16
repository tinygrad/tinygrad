import os
os.environ["PYTHONPATH"] = "."
os.environ["SQTT"] = "1"
if "DEV" not in os.environ: os.environ["DEV"] = "AMD"
os.environ["PROFILE"] = "1"
os.environ["AMD_LLVM"] = "0"

from dataclasses import replace
import atexit, contextlib
from tinygrad import Tensor
from tinygrad.helpers import system, getenv
from tinygrad.runtime.ops_amd import AMDProgram
from extra.sqtt.roc import decode, WaveExec, ProfileSQTTEvent
from tinygrad.device import Device, ProfileDeviceEvent

from extra.sqtt.attempt_sqtt_parse import parse_sqtt_print_packets

def set_power(x): system(f"sudo /opt/rocm/bin/amd-smi set -l {x}")
@atexit.register
def reset_power(): set_power("auto")
set_power("stable_std")

dev = Device["AMD"]

@contextlib.contextmanager
def save_sqtt():
  # clear the old traces
  dev.profile_events.clear()
  sqtt:dict[str, list[WaveExec]] = {}
  yield sqtt
  events = dev.profile_events+[ProfileDeviceEvent("AMD", props=dev.device_props())]

  #rctx = decode(events)
  #assert len(rctx.inst_execs) > 0, "empty sqtt output"
  #sqtt.update(rctx.inst_execs)

  for e in events:
    if isinstance(e, ProfileSQTTEvent):
      print(replace(e, blob=b''))
      if e.se == 0:
        parse_sqtt_print_packets(e.blob)

template = """.text
.globl matmul
.p2align 8
.type matmul,@function
matmul:
  INSTRUCTION
  s_endpgm

.rodata
.p2align 6
.amdhsa_kernel matmul
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: matmul
    .symbol: matmul.kd
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 32
    .max_flat_workgroup_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .args:
      - .address_space:  global
        .name:           a
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
...
.end_amdgpu_metadata
"""

def run_asm(src):
  NUM_WORKGROUPS = 1
  WAVE_SIZE = 32
  NUM_WAVES = 1
  t = Tensor.empty(0x1000).realize()
  buf = t.uop.buffer.ensure_allocated()
  lib = dev.compiler.compile(template.replace("INSTRUCTION", '\n'.join(src)))
  dev.compiler.disassemble(lib)
  fxn = AMDProgram(dev, "matmul", lib)
  fxn(buf._buf, global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True)

if __name__ == "__main__":
  with save_sqtt() as sqtt:
    #(Tensor.empty(16,16) @ Tensor.empty(16,16)).elu().realize()
    Tensor.empty(1).elu().realize()
  exit(0)

  with save_sqtt() as sqtt:
    # what's in v0?
    run_asm([
      "v_mov_b32_e32 v0, 0",
      "v_mov_b32_e32 v1, 0",
      "s_clause 0x1",
      "s_load_b64 s[0:1], s[0:1], null",
      "s_waitcnt lgkmcnt(0)",
    ]+[
      "global_load_b32 v1, v0, s[0:1]",
    ]*10+[
      "global_load_b32 v10, v1, s[0:1]",
      "s_waitcnt vmcnt(0)",

      #"v_rcp_f32 v1, v0"
      #"v_add_f32_e32 v1 v0 v0",
      #"v_add_f32_e32 v5 v4 v4",
      #"v_add_f32_e32 v7 v6 v6",
      #"v_add_f32_e32 v1 v0 v0",
      #"v_add_f32_e32 v2 v1 v1",
      #"s_nop 1"
    ]*5+[
      "v_add_f32_e32 v3 v2 v2",
    ]*5+[
      "v_mul_f32_e32 v3 v2 v2",
    ]*7)

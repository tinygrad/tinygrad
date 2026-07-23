#!/usr/bin/env python3
"""Replace selected driving_vision 1x1 convolutions with vector FP16-acc kernels."""
import struct
from dataclasses import replace

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_openpilot_ir3 import branch as BR, mad_f32 as MAD_F32, nop as NOP, plain_name

TARGET = "r_32_192_4_4_64_4"
INVERSE_TARGET = "r_32_64_4_4_192_4"
FIRST_CONV_TARGET = "r_64_32_16_4_4_6_3_3_4"
FULL_Y_TARGETS = {TARGET, "r_32_64_4_4_64_4"}
FULL_Z_TARGETS = {"r_8_384_4_4_128_4"}
GAP_Y_TARGETS = {"r_512_16_4_4_16_4", "r_512_48_4_4_16_4", "r_128_32_4_4_32_4", "r_128_96_4_4_32_4"}
INVERSE_W_TARGETS = {"r_512_16_4_4_48_4", "r_128_32_4_4_96_4"}
OTHER_INVERSE_TARGETS: set[str] = set()
FP32_TARGETS = FULL_Y_TARGETS | FULL_Z_TARGETS | GAP_Y_TARGETS | INVERSE_W_TARGETS | OTHER_INVERSE_TARGETS | {INVERSE_TARGET, FIRST_CONV_TARGET}

def pack_fp32_mads(lib:bytes, component:str="y") -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) < 116: raise RuntimeError(f"expected FP32 matmul loop through instruction 115, got {len(instrs)}")
  out = instrs[:44]
  for k_component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(tuple(f"r{reg}.{component}" for reg in range(13, 17)), ("r7", "r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{k_component}", weight, acc, rpt=3,
                         sy=len(out) == 44, r=True))
  out += instrs[108:114]
  branch_index = len(out)
  out.append(BR(20-branch_index))
  out += instrs[115:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed FP32 image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_gap_y_fp32_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) < 237: raise RuntimeError(f"expected at least 237 gap-Y instructions, got {len(instrs)}")
  out = instrs[:66]
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r14.y", "r15.y", "r16.y"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[114:120]
  branch_index = len(out)
  out.append(BR(26-branch_index))
  out += instrs[121:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed gap-Y image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_inverse_w_fp32_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) < 174: raise RuntimeError(f"expected at least 174 inverse-W instructions, got {len(instrs)}")
  out = instrs[:59]
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r13.w", "r14.w", "r15.w"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[107:112]
  branch_index = len(out)
  out.append(BR(19-branch_index))
  out += instrs[113:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed inverse-W image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_other_inverse_fp32_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 179: raise RuntimeError(f"expected 179 other-inverse instructions, got {len(instrs)}")
  # The first output vector is split by loop-control registers. Keep it scalar;
  # the remaining three vectors are contiguous from r14.y through r17.x.
  out = instrs[:60]
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r14.y", "r15.y", "r16.y"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[108:114]
  branch_index = len(out)
  out.append(BR(20-branch_index))
  out += instrs[115:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed other-inverse image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_first_conv_fp32_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 262: raise RuntimeError(f"expected 262 first-conv instructions, got {len(instrs)}")
  out = instrs[:60]
  first = True
  for component, weight in zip("xyzw", ("r5", "r2", "r3", "r4")):
    out.append(MAD_F32("r11.w", f"r7.{component}", f"{weight}.x", "r11.w", sy=first, r=True))
    first = False
    out.append(MAD_F32("r12.y", f"r7.{component}", f"{weight}.y", "r12.y", rpt=2, r=True))
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r13.x", "r14.x", "r15.x"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[124:130]
  branch_index = len(out)
  out.append(BR(31-branch_index))
  out += instrs[131:]
  # Compacting the innermost loop also relocates the two enclosing-loop branches.
  # Their targets remain in the untouched prologue, so rebuild their relative offsets.
  out[92] = BR(26-92)
  out[99] = BR(24-99)
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed first-conv image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_inverse_fp32_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 175: raise RuntimeError(f"expected 175 inverse instructions, got {len(instrs)}")
  # The first output vector straddles loop-control r13.x, so retain its scalar
  # instructions. The remaining r14/r15/r16 accumulator vectors are contiguous.
  out = instrs[:56]
  for component, weight in zip("xyzw", ("r5.x", "r2.x", "r3.x", "r4.x")):
    for acc, activation in zip(("r14.x", "r15.x", "r16.x"), ("r6", "r1", "r0")):
      out.append(MAD_F32(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[104:109]
  branch_index = len(out)
  out.append(BR(16-branch_index))
  out += instrs[110:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed inverse image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def patch_fp32_rpt(jit, names:set[str]|None=None) -> int:
  """Apply the verified FP32-accumulate repeat packing to a captured vision JIT."""
  outer = jit.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  new_batch, replaced = [], 0
  for call in batch:
    name = plain_name(call.src[0].arg.name) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM else ""
    if name in FP32_TARGETS and (names is None or name in names):
      program = call.src[0]
      patch = (pack_fp32_mads if name in FULL_Y_TARGETS else
               (lambda lib:pack_fp32_mads(lib, "z")) if name in FULL_Z_TARGETS else
               pack_gap_y_fp32_mads if name in GAP_Y_TARGETS else
               pack_inverse_w_fp32_mads if name in INVERSE_W_TARGETS else
               pack_other_inverse_fp32_mads if name in OTHER_INVERSE_TARGETS else
               pack_first_conv_fp32_mads if name == FIRST_CONV_TARGET else pack_inverse_fp32_mads)
      program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=patch(program.src[3].arg)),))
      if name == INVERSE_TARGET:
        program = program.replace(arg=replace(program.arg, global_size=(8, 2, 1), local_size=(8, 16, 1)))
      call, replaced = call.replace(src=(program, *call.src[1:])), replaced+1
    new_batch.append(call)
  if replaced:
    jit.captured._linear = jit.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
    jit.captured.__dict__.pop("linear", None)
  return replaced

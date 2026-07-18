#!/usr/bin/env python3
"""Replace selected driving_vision 1x1 convolutions with vector FP16-acc kernels."""
import argparse, pickle, struct
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import BR, ISAM_F16, MAD_F16, MAD_F32, NOP
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"
INVERSE_TARGET = "r_32_64_4_4_192_4"
FIRST_CONV_TARGET = "r_64_32_16_4_4_6_3_3_4"
FULL_Y_TARGETS = {TARGET, "r_32_64_4_4_64_4"}
FULL_Z_TARGETS = {"r_8_384_4_4_128_4"}
GAP_Y_TARGETS = {"r_512_16_4_4_16_4", "r_512_48_4_4_16_4", "r_128_32_4_4_32_4", "r_128_96_4_4_32_4"}
INVERSE_W_TARGETS = {"r_512_16_4_4_48_4", "r_128_32_4_4_96_4"}
OTHER_INVERSE_TARGETS: set[str] = set()
FP32_TARGETS = FULL_Y_TARGETS | FULL_Z_TARGETS | GAP_Y_TARGETS | INVERSE_W_TARGETS | OTHER_INVERSE_TARGETS | {INVERSE_TARGET, FIRST_CONV_TARGET}

SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__kernel void r_32_192_4_4_64_4(write_only image2d_t O, read_only image2d_t A,
                                read_only image2d_t W, read_only image2d_t B) {
  int n=get_global_id(0), m=get_global_id(1), abase=m*260;
  half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
  for (int k=0;k<64;k++) {
    half4 a0=read_imageh(A,smp,(int2)(abase+k,0));
    half4 a1=read_imageh(A,smp,(int2)(abase+k+65,0));
    half4 a2=read_imageh(A,smp,(int2)(abase+k+130,0));
    half4 a3=read_imageh(A,smp,(int2)(abase+k+195,0));
    int x=k*4;
    half4 w0=read_imageh(W,smp,(int2)(x,n));
    half4 w1=read_imageh(W,smp,(int2)(x+1,n));
    half4 w2=read_imageh(W,smp,(int2)(x+2,n));
    half4 w3=read_imageh(W,smp,(int2)(x+3,n));
    r0+=(half4)(a0.x)*w0; r0+=(half4)(a0.y)*w1; r0+=(half4)(a0.z)*w2; r0+=(half4)(a0.w)*w3;
    r1+=(half4)(a1.x)*w0; r1+=(half4)(a1.y)*w1; r1+=(half4)(a1.z)*w2; r1+=(half4)(a1.w)*w3;
    r2+=(half4)(a2.x)*w0; r2+=(half4)(a2.y)*w1; r2+=(half4)(a2.z)*w2; r2+=(half4)(a2.w)*w3;
    r3+=(half4)(a3.x)*w0; r3+=(half4)(a3.y)*w1; r3+=(half4)(a3.z)*w2; r3+=(half4)(a3.w)*w3;
  }
  float4 b=read_imagef(B,smp,(int2)(n,0));
  write_imagef(O,(int2)(n,m),gelu(convert_float4(r0)+b));
  write_imagef(O,(int2)(n+192,m),gelu(convert_float4(r1)+b));
  write_imagef(O,(int2)(n+384,m),gelu(convert_float4(r2)+b));
  write_imagef(O,(int2)(n+576,m),gelu(convert_float4(r3)+b));
}"""


def pack_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 222: raise RuntimeError(f"expected 222 instructions, got {len(instrs)}")
  out = instrs[:36]
  activations = ("hr3", "hr2", "hr1", "hr0")
  accumulators = ("hr8.x", "hr7.x", "hr6.x", "hr5.x")
  load_chunks = (instrs[52:55], instrs[68:71], instrs[87:90], ())
  for component, load_after in zip("xyzw", load_chunks):
    for index, (acc, activation) in enumerate(zip(accumulators, activations)):
      out.append(MAD_F16(acc, f"{activation}.{component}", "hr4.x", acc, rpt=3, sy=index == 0, r=True))
    out += list(load_after)
  out += instrs[102:108]
  branch_index = len(out)
  out.append(BR(21-branch_index))
  out += instrs[109:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed image has {len(out)} instructions")
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def pack_preloaded_mads(lib:bytes) -> bytes:
  """Keep all four weight vectors resident and synchronize the sampler once per K."""
  lib = pack_mads(lib)
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  instrs = [lib[image_offset+i:image_offset+i+8] for i in range(0, image_size, 8)]
  if len(instrs) != 222: raise RuntimeError(f"expected 222 instructions, got {len(instrs)}")
  out = instrs[:23] + instrs[23:33]
  for coord_setup, coord, weight in ((instrs[33:35], "r0.x", "hr9.x"), (instrs[40:42], "r0.z", "hr10.x"),
                                     (instrs[47:49], "r1.x", "hr11.x"), (instrs[54:56], "r1.z", "hr12.x")):
    out += coord_setup + [ISAM_F16(weight, coord, 1)]
  activations, accumulators = ("hr3", "hr2", "hr1", "hr0"), ("hr8.x", "hr7.x", "hr6.x", "hr5.x")
  first = True
  for component, weight in zip("xyzw", ("hr9.x", "hr10.x", "hr11.x", "hr12.x")):
    for acc, activation in zip(accumulators, activations):
      out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, sy=first, r=True))
      first = False
  out += instrs[61:68] + instrs[68:]
  if len(out) != len(instrs): raise RuntimeError(f"preloaded image has {len(out)} instructions")
  patched = bytearray(lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:])
  reg_offset = struct.unpack_from("<I", patched, 0x34)[0]
  old_hregs = struct.unpack_from("<I", patched, reg_offset+0x18)[0]
  struct.pack_into("<I", patched, reg_offset+0x18, (old_hregs & 0x80000000) | 13)
  return bytes(patched)


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


def pack_native_f16_forward_mads(lib:bytes) -> bytes:
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  image = lib[image_offset:image_offset+image_size]
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) == 232:
    # Current QCOM compiler scalarizes the 4x4 outer product into 64 MADs even
    # though both the four output lanes and accumulator lanes are contiguous.
    # Preserve its sampler schedule and replace only that scalar MAD block.
    # Hoist all independent texture coordinates, then issue the eight samples
    # contiguously. The original compiler inserted rpt5 after every coordinate.
    address_indices = (20, 23, 26, 29, 32, 35, 38, 41)
    load_indices = (22, 25, 28, 31, 34, 37, 40, 43)
    out = instrs[:20] + [instrs[i] for i in address_indices] + [instrs[i] for i in load_indices]
    for component, weight in zip("xyzw", ("hr5.x", "hr2.x", "hr3.x", "hr4.x")):
      for acc, activation in (("hr8.x", "hr7"), ("hr9.x", "hr6"), ("hr10.x", "hr1"), ("hr11.x", "hr0")):
        out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True,
                           sy=len(out) == 36))
    out += instrs[108:114]
    out.append(BR(20-len(out)))
    out += instrs[115:]
    out += [NOP()] * (len(instrs)-len(out))
    if len(out) != len(instrs): raise RuntimeError(f"packed native-FP16 image has {len(out)} instructions")
    return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]
  if len(instrs) != 241: raise RuntimeError(f"expected 232/241 native-FP16 forward instructions, got {len(instrs)}")
  already_packed = instrs[53] == MAD_F16("hr0.x", "hr7.y", "hr8.x", "hr0.x", rpt=3, r=True)
  out = instrs[:52]
  for component, weight in zip("xyzw", ("hr11.x", "hr8.x", "hr9.x", "hr10.x")):
    for acc, activation in (("hr0.x", "hr7"), ("hr1.x", "hr4"), ("hr2.x", "hr5"), ("hr3.x", "hr6")):
      out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True))
  out += instrs[68:74] if already_packed else instrs[116:122]
  branch_index = len(out)
  out.append(BR(20-branch_index))
  out += instrs[75:] if already_packed else instrs[123:]
  out += [NOP()] * (len(instrs)-len(out))
  if len(out) != len(instrs): raise RuntimeError(f"packed native-FP16 image has {len(out)} instructions")
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


def pack_first_conv_f16_mads(lib:bytes) -> bytes:
  """Vectorize the native-half first convolution's four 4x4 dot products."""
  image_offset, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  instrs = [lib[image_offset+i:image_offset+i+8] for i in range(0, image_size, 8)]
  if len(instrs) < 200: raise RuntimeError(f"expected native-FP16 first-conv shader, got {len(instrs)} instructions")
  out = instrs[:60]
  first = True
  for activation, acc in (("hr7", "hr8.x"), ("hr6", "hr9.x"), ("hr1", "hr10.x"), ("hr0", "hr11.x")):
    for component, weight in zip("xyzw", ("hr5.x", "hr2.x", "hr3.x", "hr4.x")):
      out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True, sy=first))
      first = False
  out += instrs[124:130]
  out.append(BR(31-len(out)))
  out += instrs[131:]
  # The compact inner loop relocates both enclosing-loop branches too.
  out[88] = BR(26-88)
  out[95] = BR(24-95)
  if len(out) > len(instrs): raise RuntimeError(f"packed native-FP16 first conv grew from {len(instrs)} to {len(out)}")
  out += [NOP()] * (len(instrs)-len(out))
  return lib[:image_offset] + b"".join(out) + lib[image_offset+image_size:]


def patch_native_f16_firstconv(jit) -> int:
  outer = jit.captured.linear.src[0]
  batch, replaced = list(outer.src[0].src[0].src), 0
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != FIRST_CONV_TARGET: continue
    program = call.src[0]
    program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=pack_first_conv_f16_mads(program.src[3].arg)),))
    batch[index], replaced = call.replace(src=(program, *call.src[1:])), replaced+1
  if replaced:
    jit.captured._linear = jit.captured.linear.substitute({outer:create_graph_call(batch)}, walk=True)
    jit.captured.__dict__.pop("linear", None)
  return replaced


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


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  ap.add_argument("--fp32-rpt", action="store_true")
  ap.add_argument("--fp32-rpt-names", help="comma-separated subset for --fp32-rpt")
  ap.add_argument("--native-f16-rpt", action="store_true")
  ap.add_argument("--preload-f16", action="store_true")
  ap.add_argument("--native-f16-firstconv-rpt", action="store_true")
  args = ap.parse_args()
  with open(args.input, "rb") as f: jit = pickle.load(f)
  if args.native_f16_firstconv_rpt:
    replaced = patch_native_f16_firstconv(jit)
    if not replaced: raise RuntimeError(f"no {FIRST_CONV_TARGET} calls found")
    with open(args.output, "wb") as f: pickle.dump(jit, f)
    print(f"patched {replaced} call(s)")
    return
  if args.fp32_rpt:
    replaced = patch_fp32_rpt(jit, set(args.fp32_rpt_names.split(",")) if args.fp32_rpt_names else None)
    if not replaced: raise RuntimeError(f"no {TARGET} calls found")
    with open(args.output, "wb") as f: pickle.dump(jit, f)
    print(f"patched {replaced} call(s)")
    return
  hand_lib = None if args.fp32_rpt else (pack_preloaded_mads if args.preload_f16 else pack_mads)(Device["QCOM"].compiler.compile_cached(SOURCE))
  outer = jit.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  new_batch, replaced = [], 0
  for call in batch:
    name = plain_name(call.src[0].arg.name) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM else ""
    if name == TARGET or (args.fp32_rpt and name in FP32_TARGETS):
      program = call.src[0]
      if args.native_f16_rpt:
        program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=pack_native_f16_forward_mads(program.src[3].arg)),))
      elif args.fp32_rpt:
        patch = (pack_fp32_mads if name in FULL_Y_TARGETS else
                 (lambda lib:pack_fp32_mads(lib, "z")) if name in FULL_Z_TARGETS else
                 pack_gap_y_fp32_mads if name in GAP_Y_TARGETS else
                 pack_inverse_w_fp32_mads if name in INVERSE_W_TARGETS else
                 pack_other_inverse_fp32_mads if name in OTHER_INVERSE_TARGETS else
                 pack_first_conv_fp32_mads if name == FIRST_CONV_TARGET else pack_inverse_fp32_mads)
        program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=patch(program.src[3].arg)),))
        if name == INVERSE_TARGET:
          program = program.replace(arg=replace(program.arg, global_size=(8, 2, 1), local_size=(8, 16, 1)))
      else:
        program = program.replace(src=program.src[:2] +
          (program.src[2].replace(arg=SOURCE), program.src[3].replace(arg=hand_lib)))
      call, replaced = call.replace(src=(program, *call.src[1:])), replaced+1
    new_batch.append(call)
  if not replaced: raise RuntimeError(f"no {TARGET} calls found")
  jit.captured._linear = jit.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
  jit.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  print(f"patched {replaced} call(s)")


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""High-intensity 8x8 FP16-accumulate tile for OpenPilot vision projections."""
import argparse, os, pickle, struct
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import ADD_S, ADD_S_REG, BR, ISAM_F16, MAD_F16, MOV_F32, MOV_H_IMM, NOP, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"
SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) { return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v; }
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void r_32_192_4_4_64_4(write_only image2d_t O,read_only image2d_t A,read_only image2d_t W,read_only image2d_t B) {
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31,row0=get_group_id(1)*32+tm*8,n0=get_group_id(0)*64+tid*2,n1=n0+1;
  half8 c0=(half8)(0),c1=(half8)(0),c2=(half8)(0),c3=(half8)(0);
  half8 c4=(half8)(0),c5=(half8)(0),c6=(half8)(0),c7=(half8)(0);
  for(int k=0;k<64;k++) {
    int x=k*4;
    half8 w0=(half8)(read_imageh(W,smp,(int2)(x,n0)),read_imageh(W,smp,(int2)(x,n1)));
    half8 w1=(half8)(read_imageh(W,smp,(int2)(x+1,n0)),read_imageh(W,smp,(int2)(x+1,n1)));
    half8 w2=(half8)(read_imageh(W,smp,(int2)(x+2,n0)),read_imageh(W,smp,(int2)(x+2,n1)));
    half8 w3=(half8)(read_imageh(W,smp,(int2)(x+3,n0)),read_imageh(W,smp,(int2)(x+3,n1)));
    int r=row0,base=(r&31)*260+(r>>5)*65;
    half4 a0=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a1=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a2=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a3=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a4=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a5=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a6=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a7=read_imageh(A,smp,(int2)(base+k,0));
    c0+=(half8)(a0.x)*w0+(half8)(a0.y)*w1+(half8)(a0.z)*w2+(half8)(a0.w)*w3;
    c1+=(half8)(a1.x)*w0+(half8)(a1.y)*w1+(half8)(a1.z)*w2+(half8)(a1.w)*w3;
    c2+=(half8)(a2.x)*w0+(half8)(a2.y)*w1+(half8)(a2.z)*w2+(half8)(a2.w)*w3;
    c3+=(half8)(a3.x)*w0+(half8)(a3.y)*w1+(half8)(a3.z)*w2+(half8)(a3.w)*w3;
    c4+=(half8)(a4.x)*w0+(half8)(a4.y)*w1+(half8)(a4.z)*w2+(half8)(a4.w)*w3;
    c5+=(half8)(a5.x)*w0+(half8)(a5.y)*w1+(half8)(a5.z)*w2+(half8)(a5.w)*w3;
    c6+=(half8)(a6.x)*w0+(half8)(a6.y)*w1+(half8)(a6.z)*w2+(half8)(a6.w)*w3;
    c7+=(half8)(a7.x)*w0+(half8)(a7.y)*w1+(half8)(a7.z)*w2+(half8)(a7.w)*w3;
  }
  float4 b0=read_imagef(B,smp,(int2)(n0,0)),b1=read_imagef(B,smp,(int2)(n1,0));
  int r=row0;
#define STORE(v) { int m=r&31,p=r>>5; write_imagef(O,(int2)(n0+p*192,m),gelu(convert_float4(v.lo)+b0)); \
 write_imagef(O,(int2)(n1+p*192,m),gelu(convert_float4(v.hi)+b1)); r++; }
  STORE(c0); STORE(c1); STORE(c2); STORE(c3); STORE(c4); STORE(c5); STORE(c6); STORE(c7);
}"""

SOURCE_4X2 = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) { return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v; }
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void r_32_192_4_4_64_4(write_only image2d_t O,read_only image2d_t A,read_only image2d_t W,read_only image2d_t B) {
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31,row0=get_group_id(1)*16+tm*4,n0=get_group_id(0)*64+tid*2,n1=n0+1;
  half8 c0=(half8)(0),c1=(half8)(0),c2=(half8)(0),c3=(half8)(0);
  for(int k=0;k<64;k++) {
    int x=k*4;
    half8 w0=(half8)(read_imageh(W,smp,(int2)(x,n0)),read_imageh(W,smp,(int2)(x,n1)));
    half8 w1=(half8)(read_imageh(W,smp,(int2)(x+1,n0)),read_imageh(W,smp,(int2)(x+1,n1)));
    half8 w2=(half8)(read_imageh(W,smp,(int2)(x+2,n0)),read_imageh(W,smp,(int2)(x+2,n1)));
    half8 w3=(half8)(read_imageh(W,smp,(int2)(x+3,n0)),read_imageh(W,smp,(int2)(x+3,n1)));
    int r=row0,base=(r&31)*260+(r>>5)*65;
    half4 a0=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a1=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a2=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a3=read_imageh(A,smp,(int2)(base+k,0));
    c0+=(half8)(a0.x)*w0+(half8)(a0.y)*w1+(half8)(a0.z)*w2+(half8)(a0.w)*w3;
    c1+=(half8)(a1.x)*w0+(half8)(a1.y)*w1+(half8)(a1.z)*w2+(half8)(a1.w)*w3;
    c2+=(half8)(a2.x)*w0+(half8)(a2.y)*w1+(half8)(a2.z)*w2+(half8)(a2.w)*w3;
    c3+=(half8)(a3.x)*w0+(half8)(a3.y)*w1+(half8)(a3.z)*w2+(half8)(a3.w)*w3;
  }
  float4 b0=read_imagef(B,smp,(int2)(n0,0)),b1=read_imagef(B,smp,(int2)(n1,0));
  int r=row0;
#define STORE(v) { int m=r&31,p=r>>5; write_imagef(O,(int2)(n0+p*192,m),gelu(convert_float4(v.lo)+b0)); \
 write_imagef(O,(int2)(n1+p*192,m),gelu(convert_float4(v.hi)+b1)); r++; }
  STORE(c0); STORE(c1); STORE(c2); STORE(c3);
}"""


def pack_tile(lib:bytes) -> bytes:
  image_off,image_size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  reg_off=struct.unpack_from("<I",lib,0x34)[0]
  ins=[lib[i:i+8] for i in range(image_off,image_off+image_size,8)]
  if len(ins)<500: raise RuntimeError(f"unexpected tile8 shader length {len(ins)}")
  # Pair each pair of output-channel texture rows in consecutive half registers.
  for index,dst,coord in ((171,"hr3.x","r3.x"),(174,"hr4.x","r4.z"),
                          (178,"hr5.x","r3.z"),(181,"hr6.x","r5.x"),
                          (185,"hr7.x","r4.x"),(188,"hr8.x","r5.z"),
                          (191,"hr32.x","r6.x"),(194,"hr33.x","r6.z")):
    ins[index]=ISAM_F16(dst,coord,tex=1,sy=index==194)
  out=ins[:198]
  for activation,acc in (("hr15",30),("hr14",28),("hr13",26),("hr12",24),
                         ("hr11",22),("hr10",20),("hr2",18),("hr0",16)):
    for component,(weight_lo,weight_hi) in zip("xyzw",((3,4),(5,6),(7,8),(32,33))):
      for offset,weight in enumerate((weight_lo,weight_hi)):
        out.append(MAD_F16(f"hr{acc+offset}.x",f"{activation}.{component}",f"hr{weight}.x",f"hr{acc+offset}.x",
                           rpt=3,r=True,sy=len(out)==198))
  out.append(ins[366])
  out.append(BR(143-len(out)))
  out+=ins[368:]
  if len(out)>len(ins): raise RuntimeError(f"packed shader grew from {len(ins)} to {len(out)}")
  out += [NOP()]*(len(ins)-len(out))
  fregs,hregs=struct.unpack_from("<II",lib,reg_off+0x14)
  hregs=(hregs&0x80000000)|max(hregs&0x7fffffff,34)
  return inject(lib,image_off,image_size,reg_off,b"".join(out),fregs,hregs)


def pack_4x2_tile(lib: bytes, wg256: bool = False) -> bytes:
  """Collapse the compiler's scalarized 4-row x 2-output FP16 dot products."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 450: raise RuntimeError(f"unexpected 4x2 shader length {len(ins)}")

  # The donor loads four A vectors into hr11,hr10,hr5,hr4 and four pairs of
  # W vectors into the low/high output banks below. Accumulators are already
  # initialized in hr12..hr19; update them directly instead of materializing
  # scalar partial sums and adding those afterwards.
  loop_body, control, epilogue, loop_target = ((104, 228, 235, 65) if wg256 else (108, 232, 239, 69))
  out = list(ins[:loop_body])
  weight_pairs = ((6, 0), (9, 3), (8, 2), (7, 1))
  first = True
  rows = ((11, (18, 19)), (10, (16, 17)), (5, (14, 15)), (4, (12, 13)))
  # Component-major order leaves eight instructions between updates of the
  # same accumulator, hiding the dependent half-MAD latency.
  for component, (weight_lo, weight_hi) in zip("xyzw", weight_pairs):
    for column, weight in enumerate((weight_lo, weight_hi)):
      for activation, accs in rows:
        acc = accs[column]
        out.append(MAD_F16(f"hr{acc}.x", f"hr{activation}.{component}", f"hr{weight}.x", f"hr{acc}.x",
                           rpt=3, r=True, sy=first))
        first = False

  # Retain the compiler's K/coordinate updates and predicate setup, then
  # relocate the loop backedge to the unchanged load block at instruction 69.
  out += ins[control:epilogue-1]
  out.append(BR(loop_target-len(out)))
  out += ins[epilogue:]
  if len(out) > len(ins): raise RuntimeError(f"packed shader grew from {len(ins)} to {len(out)}")
  out += [NOP()] * (len(ins)-len(out))
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  hregs = (hregs & 0x80000000) | 20
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def pack_4x2_wide_tile(lib: bytes) -> bytes:
  """Use one rpt7 half-MAD for both adjacent output vectors."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 450: raise RuntimeError(f"unexpected 4x2 shader length {len(ins)}")

  # Pack each low/high output weight pair into adjacent high half registers so
  # a single rpt7 instruction covers all eight output lanes. High destinations
  # avoid the merged-register aliases of the still-live coordinate registers.
  # hr12..19 remain the compiler donor's initialized accumulator bank.
  for index, dst, coord in ((72, "hr20.x", "r0.x"), (73, "hr21.x", "r0.z"),
                            (77, "hr22.x", "r1.x"), (80, "hr23.x", "r1.z"),
                            (84, "hr24.x", "r2.x"), (87, "hr25.x", "r2.z"),
                            (90, "hr26.x", "r3.x"), (93, "hr27.x", "r3.z")):
    ins[index] = ISAM_F16(dst, coord, tex=1)

  out = list(ins[:108])
  first = True
  for component, weight in zip("xyzw", (20, 22, 24, 26)):
    for activation, acc in ((11, 18), (10, 16), (5, 14), (4, 12)):
      out.append(MAD_F16(f"hr{acc}.x", f"hr{activation}.{component}", f"hr{weight}.x", f"hr{acc}.x",
                         rpt=7, r=True, sy=first))
      first = False
  out += ins[232:238]
  out.append(BR(69-len(out)))
  out += ins[239:]
  if len(out) > len(ins): raise RuntimeError(f"packed shader grew from {len(ins)} to {len(out)}")
  out += [NOP()] * (len(ins)-len(out))
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  hregs = (hregs & 0x80000000) | 28
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def pack_split_tile(lib: bytes) -> bytes:
  """Use the verified split-A register footprint while retaining the fused epilogue."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 500: raise RuntimeError(f"unexpected tile8 shader length {len(ins)}")

  # The compiler donor already calculated all padded vision-image row bases in
  # r9.w..r11.z and keeps K4/weight-x in r8.y/r8.x. Keep that coordinate work,
  # but use the 28-half-register layout of the validated general 8x8 kernel:
  # weights hr0..7, four reusable A vectors hr8..11, accumulators hr12..27.
  out = ins[:127]
  for acc in range(12, 28): out.append(MOV_H_IMM(f"hr{acc}.x", 0, rpt=3))
  out.append(NOP())
  if len(out) != 144: raise RuntimeError(f"split prologue ended at {len(out)}")

  weight_coords = tuple((f"r{20+i//2}.{'xz'[i&1]}", f"r{20+i//2}.{'yw'[i&1]}") for i in range(8))
  for component in range(4):
    for col in range(2):
      xreg, yreg = weight_coords[component*2+col]
      out.append(MOV_F32(xreg, "r8.x") if component == 3 else ADD_S(xreg, "r8.x", component-3))
      out.append(MOV_F32(yreg, "r8.w" if col == 0 else "r8.z"))
  out.append(NOP(rpt=5))
  for weight, (xreg, _) in enumerate(weight_coords): out.append(ISAM_F16(f"hr{weight}.x", xreg, 1))

  activation_bases = ("r11.z", "r11.y", "r11.x", "r10.w", "r10.z", "r10.y", "r10.x", "r9.w")
  def load_rows(first_row: int) -> None:
    for slot, row in enumerate(range(first_row, first_row+4)):
      out.extend((ADD_S_REG("r25.x", "r8.y", activation_bases[row]), NOP(rpt=5),
                  ISAM_F16(f"hr{8+slot}.x", "r25.x", 0)))

  def mad_rows(first_row: int) -> None:
    first = True
    for slot, row in enumerate(range(first_row, first_row+4)):
      for component, (weight0, weight1) in zip("xyzw", ((0, 1), (2, 3), (4, 5), (6, 7))):
        for col, weight in enumerate((weight0, weight1)):
          acc = 12 + row*2 + col
          out.append(MAD_F16(f"hr{acc}.x", f"hr{8+slot}.{component}", f"hr{weight}.x", f"hr{acc}.x",
                             rpt=3, r=True, sy=first))
          first = False

  load_rows(0)
  mad_rows(0)
  load_rows(4)
  mad_rows(4)
  out += ins[195:198] + [ins[366]]
  out.append(BR(143-len(out)))

  # Retain the exact compiler-generated bias/GELU/image-store epilogue. Its
  # first stage converts the old accumulator bank; redirect those sources to
  # the compact bank without disturbing any later full-register scheduling.
  lane_map = {}
  for row in range(8):
    for col in range(2):
      old_vec, new_vec = 30-row*2+col, 12+row*2+col
      for lane in range(4): lane_map[old_vec*4+lane] = new_vec*4+lane
  epilogue = list(ins[368:])
  for index in range(4, min(31, len(epilogue))):
    lo, hi = struct.unpack("<II", epilogue[index])
    if lo in lane_map and (hi & 0x00F04000) == 0x00004000:
      epilogue[index] = struct.pack("<II", lane_map[lo], hi)
  out += epilogue
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  hregs = (hregs & 0x80000000) | 28
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def pack_split_safe_tile(lib: bytes) -> bytes:
  """Split the eight A rows without changing the donor's sampler register assignment."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 500: raise RuntimeError(f"unexpected tile8 shader length {len(ins)}")
  out = list(ins[:144])

  # Preserve the compiler's proven sampler destinations. Only hr32/hr33 are
  # remapped, to the dead activation slots hr10/hr11, reducing hregs 34 -> 32.
  weight_regs = (6, 1, 9, 5, 8, 4, 7, 3)
  weight_coords = ("r3.x", "r4.z", "r3.z", "r5.x", "r4.x", "r5.z", "r6.x", "r6.z")
  weight_loads = ((168, 171), (172, 174), (175, 178), (179, 181),
                  (182, 185), (186, 188), (189, 191), (192, 194))
  def load_weights() -> None:
    for index, (((start, load), coord), weight) in enumerate(zip(zip(weight_loads, weight_coords), weight_regs)):
      out.extend(ins[start:load])
      out.append(ISAM_F16(f"hr{weight}.x", coord, 1))

  activation_bases = ("r11.z", "r11.y", "r11.x", "r10.w", "r10.z", "r10.y", "r10.x", "r9.w")
  activation_regs = (15, 14, 13, 12)
  def load_rows(first_row: int) -> None:
    for slot, row in enumerate(range(first_row, first_row+4)):
      out.extend((ADD_S_REG("r25.x", "r8.y", activation_bases[row]), NOP(rpt=5),
                  ISAM_F16(f"hr{activation_regs[slot]}.x", "r25.x", 0)))

  def mad_rows(first_row: int) -> None:
    first = True
    for slot, row in enumerate(range(first_row, first_row+4)):
      for component, weights in zip("xyzw", ((6, 1), (9, 5), (8, 4), (7, 3))):
        for col, weight in enumerate(weights):
          acc = 30-row*2+col
          out.append(MAD_F16(f"hr{acc}.x", f"hr{activation_regs[slot]}.{component}", f"hr{weight}.x", f"hr{acc}.x",
                             rpt=3, r=True, sy=first))
          first = False

  load_rows(0)
  load_weights()
  mad_rows(0)
  load_rows(4)
  mad_rows(4)
  out += ins[195:198] + [ins[366]]
  out.append(BR(143-len(out)))
  out += ins[368:]
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  hregs = (hregs & 0x80000000) | 32
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def patch_model(model) -> int:
  outer=model.captured.linear.src[0]; batch=list(outer.src[0].src[0].src)
  source = SOURCE_4X2 if os.getenv("TILE4X2") else SOURCE
  if os.getenv("TILE4X2_WG256"):
    source = source.replace("reqd_work_group_size(128,1,1)", "reqd_work_group_size(256,1,1)") \
                   .replace("get_group_id(1)*16+tm*4", "get_group_id(1)*32+tm*4")
  if os.getenv("TILE_LINEAR"):
    source = "\n".join(line for line in source.splitlines() if not line.startswith("inline float4 gelu"))
    source = source.replace("gelu(convert", "(convert")
  raw_lib = Device["QCOM"].compiler.compile_cached(source)
  lib=((raw_lib if os.getenv("TILE4X2_RAW") else pack_4x2_wide_tile(raw_lib) if os.getenv("TILE4X2_WIDE") else
        pack_4x2_tile(raw_lib, bool(os.getenv("TILE4X2_WG256")))) if os.getenv("TILE4X2") else
       pack_split_safe_tile(raw_lib) if os.getenv("TILE8_SAFE") else pack_split_tile(raw_lib)); patched=0
  for index,call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name)!=TARGET: continue
    program=call.src[0]
    program=program.replace(arg=replace(program.arg,global_size=((3,4 if os.getenv("TILE4X2_WG256") else 8,1)
                                                                  if os.getenv("TILE4X2") else (3,4,1)),
      local_size=((256,1,1) if os.getenv("TILE4X2_WG256") else (128,1,1))),
      src=program.src[:2]+(program.src[2].replace(arg=source),program.src[3].replace(arg=lib)))
    batch[index]=call.replace(src=(program,*call.src[1:])); patched+=1
  if patched:
    model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(batch)},walk=True)
    model.captured.__dict__.pop("linear",None)
  return patched


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_model(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__=="__main__":main()

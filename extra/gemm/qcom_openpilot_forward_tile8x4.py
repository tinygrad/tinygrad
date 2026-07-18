#!/usr/bin/env python3
"""High-intensity 8x4 FP16-accumulate tile for OpenPilot vision projections."""
import argparse, os, pickle, struct
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import BR, ISAM_F16, MAD_F16, MOV_H_IMM, NOP, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"
SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) { return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v; }
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void r_32_192_4_4_64_4(write_only image2d_t O,read_only image2d_t A,read_only image2d_t W,read_only image2d_t B) {
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31,row0=get_group_id(1)*32+tm*8,n=get_group_id(0)*32+tid;
  half4 c0=(half4)(0),c1=(half4)(0),c2=(half4)(0),c3=(half4)(0);
  half4 c4=(half4)(0),c5=(half4)(0),c6=(half4)(0),c7=(half4)(0);
  for(int k=0;k<64;k++) {
    int x=k*4;
    half4 w0=read_imageh(W,smp,(int2)(x,n));
    half4 w1=read_imageh(W,smp,(int2)(x+1,n));
    half4 w2=read_imageh(W,smp,(int2)(x+2,n));
    half4 w3=read_imageh(W,smp,(int2)(x+3,n));
    int r=row0,base=(r&31)*260+(r>>5)*65;
    half4 a0=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a1=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a2=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a3=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a4=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a5=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a6=read_imageh(A,smp,(int2)(base+k,0)); r++; base=(r&31)*260+(r>>5)*65;
    half4 a7=read_imageh(A,smp,(int2)(base+k,0));
    c0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
    c1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
    c2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
    c3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
    c4+=(half4)(a4.x)*w0+(half4)(a4.y)*w1+(half4)(a4.z)*w2+(half4)(a4.w)*w3;
    c5+=(half4)(a5.x)*w0+(half4)(a5.y)*w1+(half4)(a5.z)*w2+(half4)(a5.w)*w3;
    c6+=(half4)(a6.x)*w0+(half4)(a6.y)*w1+(half4)(a6.z)*w2+(half4)(a6.w)*w3;
    c7+=(half4)(a7.x)*w0+(half4)(a7.y)*w1+(half4)(a7.z)*w2+(half4)(a7.w)*w3;
  }
  float4 b=read_imagef(B,smp,(int2)(n,0));
  int r=row0;
#define STORE(v) { int m=r&31,p=r>>5; write_imagef(O,(int2)(n+p*192,m),gelu(convert_float4(v)+b)); r++; }
  STORE(c0); STORE(c1); STORE(c2); STORE(c3); STORE(c4); STORE(c5); STORE(c6); STORE(c7);
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


def pack_compact_tile(lib:bytes) -> bytes:
  """Replace the compiler's split partial sums with eight direct half4 accumulators."""
  image_off,image_size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  reg_off=struct.unpack_from("<I",lib,0x34)[0]
  ins=[lib[i:i+8] for i in range(image_off,image_off+image_size,8)]
  if len(ins) < 500: raise RuntimeError(f"unexpected tile8x4 shader length {len(ins)}")
  out=ins[:91] + [MOV_H_IMM(f"hr{acc}.x",0,rpt=3) for acc in range(10,18)] + ins[99:123]
  out += ins[123:126] + ins[126:129] + ins[177:179] + [ISAM_F16("hr18.x","r4.x",1)] \
       + ins[211:213] + [ISAM_F16("hr19.x","r4.z",1)]
  first=True
  for weight,component in (("hr8.x","x"),("hr9.x","y"),("hr18.x","z"),("hr19.x","w")):
    for activation,acc in zip(("hr7","hr6","hr5","hr4","hr3","hr2","hr1","hr0"),range(17,9,-1)):
      out.append(MAD_F16(f"hr{acc}.x",f"{activation}.{component}",weight,f"hr{acc}.x",rpt=3,r=True,sy=first))
      first=False
  out += ins[256:261]
  out.append(BR(99-len(out)))
  out += ins[262:]
  if len(out)>len(ins): raise RuntimeError(f"compact shader grew from {len(ins)} to {len(out)}")
  out += [NOP()]*(len(ins)-len(out))
  fregs,hregs=struct.unpack_from("<II",lib,reg_off+0x14)
  hregs=(hregs&0x80000000)|20
  return inject(lib,image_off,image_size,reg_off,b"".join(out),fregs,hregs)


def patch_model(model) -> int:
  outer=model.captured.linear.src[0]; batch=list(outer.src[0].src[0].src)
  lib=pack_compact_tile(Device["QCOM"].compiler.compile_cached(SOURCE)); patched=0
  seen=0
  for index,call in enumerate(batch):
    if patched >= int(os.getenv("MAX_PATCH", "1000000")): break
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name)!=TARGET: continue
    if seen != int(os.getenv("TARGET_INDEX", str(seen))): seen+=1; continue
    seen+=1
    program=call.src[0]
    program=program.replace(arg=replace(program.arg,global_size=(6,4,1),local_size=(128,1,1)),
      src=program.src[:2]+(program.src[2].replace(arg=SOURCE),program.src[3].replace(arg=lib)))
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

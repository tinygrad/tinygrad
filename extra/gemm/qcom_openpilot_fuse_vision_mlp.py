#!/usr/bin/env python3
"""Fuse the six driving_vision MLP pairs through work-group local memory."""
import argparse, pickle, re, struct
from dataclasses import replace

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import BR, ISAM_F16, MAD_F16, NOP, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name

FORWARD, INVERSE = "r_32_192_4_4_64_4", "r_32_64_4_4_192_4"

SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void fused_vision_mlp(write_only image2d_t O,read_only image2d_t X,read_only image2d_t A,
                               read_only image2d_t W1,read_only image2d_t B1,read_only image2d_t W2,
                               read_only image2d_t B2,read_only image2d_t S) {
  int lid=get_local_id(0),g=get_group_id(1);
  __local half4 hidden[768];
  {
    half4 z0[3]={(half4)(0),(half4)(0),(half4)(0)};
    half4 z1[3]={(half4)(0),(half4)(0),(half4)(0)};
    half4 z2[3]={(half4)(0),(half4)(0),(half4)(0)};
    half4 z3[3]={(half4)(0),(half4)(0),(half4)(0)};
    for(int k=0;k<64;k++) {
      int ax=g*260+k,wx=k*4;
      half4 a0=read_imageh(A,smp,(int2)(ax,0)),a1=read_imageh(A,smp,(int2)(ax+65,0));
      half4 a2=read_imageh(A,smp,(int2)(ax+130,0)),a3=read_imageh(A,smp,(int2)(ax+195,0));
      #pragma unroll 3
      for(int j=0;j<3;j++) {
        int n=lid+j*64;
        half4 w0=read_imageh(W1,smp,(int2)(wx,n)),w1=read_imageh(W1,smp,(int2)(wx+1,n));
        half4 w2=read_imageh(W1,smp,(int2)(wx+2,n)),w3=read_imageh(W1,smp,(int2)(wx+3,n));
        z0[j]+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
        z1[j]+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
        z2[j]+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
        z3[j]+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
      }
    }
    #pragma unroll 3
    for(int j=0;j<3;j++) {
      int n=lid+j*64; float4 b=read_imagef(B1,smp,(int2)(n,0));
      hidden[n]=convert_half4(gelu(convert_float4(z0[j])+b));
      hidden[192+n]=convert_half4(gelu(convert_float4(z1[j])+b));
      hidden[384+n]=convert_half4(gelu(convert_float4(z2[j])+b));
      hidden[576+n]=convert_half4(gelu(convert_float4(z3[j])+b));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    half4 z0=(half4)(0),z1=(half4)(0),z2=(half4)(0),z3=(half4)(0);
    for(int k=0;k<192;k++) {
      int wx=k*4;
      half4 a0=hidden[k],a1=hidden[192+k],a2=hidden[384+k],a3=hidden[576+k];
      half4 w0=read_imageh(W2,smp,(int2)(wx,lid)),w1=read_imageh(W2,smp,(int2)(wx+1,lid));
      half4 w2=read_imageh(W2,smp,(int2)(wx+2,lid)),w3=read_imageh(W2,smp,(int2)(wx+3,lid));
      z0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
      z1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
      z2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
      z3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
    }
    int x=lid+g*256; float4 b=read_imagef(B2,smp,(int2)(lid,0)),s=read_imagef(S,smp,(int2)(lid,0));
    write_imagef(O,(int2)(x,0),read_imagef(X,smp,(int2)(x,0))+(convert_float4(z0)+b)*s);
    write_imagef(O,(int2)(x+64,0),read_imagef(X,smp,(int2)(x+64,0))+(convert_float4(z1)+b)*s);
    write_imagef(O,(int2)(x+128,0),read_imagef(X,smp,(int2)(x+128,0))+(convert_float4(z2)+b)*s);
    write_imagef(O,(int2)(x+192,0),read_imagef(X,smp,(int2)(x+192,0))+(convert_float4(z3)+b)*s);
  }
}"""

SOURCE_TILED = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void fused_vision_mlp(write_only image2d_t O,read_only image2d_t X,read_only image2d_t A,
                               read_only image2d_t W1,read_only image2d_t B1,read_only image2d_t W2,
                               read_only image2d_t B2,read_only image2d_t S) {
  int lid=get_local_id(0),g=get_group_id(1);
  __local half4 hidden[256];
  half4 o0=(half4)(0),o1=(half4)(0),o2=(half4)(0),o3=(half4)(0);
  #pragma unroll 3
  for(int j=0;j<3;j++) {
    int n=lid+j*64;
    half4 z0=(half4)(0),z1=(half4)(0),z2=(half4)(0),z3=(half4)(0);
    for(int k=0;k<64;k++) {
      int ax=g*260+k,wx=k*4;
      half4 a0=read_imageh(A,smp,(int2)(ax,0)),a1=read_imageh(A,smp,(int2)(ax+65,0));
      half4 a2=read_imageh(A,smp,(int2)(ax+130,0)),a3=read_imageh(A,smp,(int2)(ax+195,0));
      half4 w0=read_imageh(W1,smp,(int2)(wx,n)),w1=read_imageh(W1,smp,(int2)(wx+1,n));
      half4 w2=read_imageh(W1,smp,(int2)(wx+2,n)),w3=read_imageh(W1,smp,(int2)(wx+3,n));
      z0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
      z1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
      z2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
      z3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
    }
    float4 b=read_imagef(B1,smp,(int2)(n,0));
    hidden[lid]=convert_half4(gelu(convert_float4(z0)+b));
    hidden[64+lid]=convert_half4(gelu(convert_float4(z1)+b));
    hidden[128+lid]=convert_half4(gelu(convert_float4(z2)+b));
    hidden[192+lid]=convert_half4(gelu(convert_float4(z3)+b));
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k=0;k<64;k++) {
      int wx=(j*64+k)*4;
      half4 a0=hidden[k],a1=hidden[64+k],a2=hidden[128+k],a3=hidden[192+k];
      half4 w0=read_imageh(W2,smp,(int2)(wx,lid)),w1=read_imageh(W2,smp,(int2)(wx+1,lid));
      half4 w2=read_imageh(W2,smp,(int2)(wx+2,lid)),w3=read_imageh(W2,smp,(int2)(wx+3,lid));
      o0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
      o1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
      o2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
      o3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int x=lid+g*256; float4 b=read_imagef(B2,smp,(int2)(lid,0)),s=read_imagef(S,smp,(int2)(lid,0));
  write_imagef(O,(int2)(x,0),read_imagef(X,smp,(int2)(x,0))+(convert_float4(o0)+b)*s);
  write_imagef(O,(int2)(x+64,0),read_imagef(X,smp,(int2)(x+64,0))+(convert_float4(o1)+b)*s);
  write_imagef(O,(int2)(x+128,0),read_imagef(X,smp,(int2)(x+128,0))+(convert_float4(o2)+b)*s);
  write_imagef(O,(int2)(x+192,0),read_imagef(X,smp,(int2)(x+192,0))+(convert_float4(o3)+b)*s);
}"""

SOURCE_PARALLEL = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__attribute__((reqd_work_group_size(192,1,1)))
__kernel void fused_vision_mlp(write_only image2d_t O,read_only image2d_t X,read_only image2d_t A,
                               read_only image2d_t W1,read_only image2d_t B1,read_only image2d_t W2,
                               read_only image2d_t B2,read_only image2d_t S) {
  int lid=get_local_id(0),g=get_group_id(1),n=lid;
  __local half4 hidden[768];
  __local half4 partial[768];
  half4 z0=(half4)(0),z1=(half4)(0),z2=(half4)(0),z3=(half4)(0);
  for(int k=0;k<64;k++) {
    int ax=g*260+k,wx=k*4;
    half4 a0=read_imageh(A,smp,(int2)(ax,0)),a1=read_imageh(A,smp,(int2)(ax+65,0));
    half4 a2=read_imageh(A,smp,(int2)(ax+130,0)),a3=read_imageh(A,smp,(int2)(ax+195,0));
    half4 w0=read_imageh(W1,smp,(int2)(wx,n)),w1=read_imageh(W1,smp,(int2)(wx+1,n));
    half4 w2=read_imageh(W1,smp,(int2)(wx+2,n)),w3=read_imageh(W1,smp,(int2)(wx+3,n));
    z0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
    z1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
    z2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
    z3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
  }
  float4 b1=read_imagef(B1,smp,(int2)(n,0));
  hidden[n]=convert_half4(gelu(convert_float4(z0)+b1));
  hidden[192+n]=convert_half4(gelu(convert_float4(z1)+b1));
  hidden[384+n]=convert_half4(gelu(convert_float4(z2)+b1));
  hidden[576+n]=convert_half4(gelu(convert_float4(z3)+b1));
  barrier(CLK_LOCAL_MEM_FENCE);
  int part=lid>>6; n=lid&63;
  half4 o0=(half4)(0),o1=(half4)(0),o2=(half4)(0),o3=(half4)(0);
  for(int k=part*64;k<(part+1)*64;k++) {
    int wx=k*4;
    half4 a0=hidden[k],a1=hidden[192+k],a2=hidden[384+k],a3=hidden[576+k];
    half4 w0=read_imageh(W2,smp,(int2)(wx,n)),w1=read_imageh(W2,smp,(int2)(wx+1,n));
    half4 w2=read_imageh(W2,smp,(int2)(wx+2,n)),w3=read_imageh(W2,smp,(int2)(wx+3,n));
    o0+=(half4)(a0.x)*w0+(half4)(a0.y)*w1+(half4)(a0.z)*w2+(half4)(a0.w)*w3;
    o1+=(half4)(a1.x)*w0+(half4)(a1.y)*w1+(half4)(a1.z)*w2+(half4)(a1.w)*w3;
    o2+=(half4)(a2.x)*w0+(half4)(a2.y)*w1+(half4)(a2.z)*w2+(half4)(a2.w)*w3;
    o3+=(half4)(a3.x)*w0+(half4)(a3.y)*w1+(half4)(a3.z)*w2+(half4)(a3.w)*w3;
  }
  int po=part*256+n;
  partial[po]=o0; partial[po+64]=o1; partial[po+128]=o2; partial[po+192]=o3;
  barrier(CLK_LOCAL_MEM_FENCE);
  if(part==0) {
    o0=partial[n]+partial[256+n]+partial[512+n];
    o1=partial[64+n]+partial[320+n]+partial[576+n];
    o2=partial[128+n]+partial[384+n]+partial[640+n];
    o3=partial[192+n]+partial[448+n]+partial[704+n];
    int x=n+g*256; float4 b=read_imagef(B2,smp,(int2)(n,0)),s=read_imagef(S,smp,(int2)(n,0));
    write_imagef(O,(int2)(x,0),read_imagef(X,smp,(int2)(x,0))+(convert_float4(o0)+b)*s);
    write_imagef(O,(int2)(x+64,0),read_imagef(X,smp,(int2)(x+64,0))+(convert_float4(o1)+b)*s);
    write_imagef(O,(int2)(x+128,0),read_imagef(X,smp,(int2)(x+128,0))+(convert_float4(o2)+b)*s);
    write_imagef(O,(int2)(x+192,0),read_imagef(X,smp,(int2)(x+192,0))+(convert_float4(o3)+b)*s);
  }
}"""


def pack_tiled_inner(lib:bytes) -> bytes:
  """Pack the compiler's scalar 4x4 half outer product into 16 repeated MADs."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 500: raise RuntimeError(f"unexpected fused shader length {len(ins)}")
  # 37:49 loads the four spatial activation vectors. 49:57 loads W0/W1; the
  # compiler's W2/W3 coordinates are at 79:82 and 98:101. Give all four weights
  # stable destinations, then accumulate directly into z0..z3.
  out = list(ins[:57])
  out += ins[79:82] + [ISAM_F16("hr6.x", "r0.x", 2)]
  out += ins[98:101] + [ISAM_F16("hr7.x", "r0.x", 2)]
  loop_start = 37
  first = True
  for activation, acc in zip(("hr3", "hr2", "hr1", "hr0"), ("hr13.y", "hr12.y", "hr11.y", "hr10.y")):
    for component, weight in zip("xyzw", ("hr4.x", "hr5.x", "hr6.x", "hr7.x")):
      out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True, sy=first))
      first = False
  out += ins[124:130]
  out.append(BR(loop_start-len(out)))
  # Compacting the first inner loop relocates the GELU, local-memory inverse,
  # and outer-j tail. Preserve branches wholly inside that tail, and rebuild
  # every branch whose target remains in the untouched prologue.
  tail_start = len(out)
  shift = tail_start-131
  for old_index, instruction in enumerate(ins[131:], 131):
    lo, hi = struct.unpack("<iI", instruction)
    new_index = old_index+shift
    if hi in (0x00800000, 0x00900000) and (old_target:=old_index+lo) < 131:
      instruction = BR(old_target-new_index, inv=hi == 0x00900000)
    out.append(instruction)
  if len(out) > len(ins): raise RuntimeError(f"packed fused shader grew from {len(ins)} to {len(out)}")
  out += [NOP()] * (len(ins)-len(out))
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def pack_tiled_inverse(lib:bytes) -> bytes:
  """Pack the local-hidden x W2 4x4 outer product in the inverse phase."""
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off, image_off+image_size, 8)]
  if len(ins) < 529: raise RuntimeError(f"unexpected fused shader length {len(ins)}")
  out = list(ins[:396])
  out += ins[396:407] + [ISAM_F16("hr12.x", "r3.y", 4)]
  out += ins[408:410] + [ISAM_F16("hr13.x", "r3.w", 4)]
  out += ins[432:434] + [ISAM_F16("hr14.x", "r4.y", 4)]
  out += ins[450:452] + [ISAM_F16("hr15.x", "r4.w", 4)]
  first = True
  for activation, acc in (("hr2", "hr9.y"), ("hr3", "hr8.y"), ("hr4", "hr7.y"), ("hr5", "hr6.y")):
    for component, weight in zip("xyzw", ("hr12.x", "hr13.x", "hr14.x", "hr15.x")):
      out.append(MAD_F16(acc, f"{activation}.{component}", weight, acc, rpt=3, r=True, sy=first))
      first = False
  out += ins[470:476]
  out.append(BR(400-len(out)))
  tail_start = len(out)
  shift = tail_start-477
  for old_index, instruction in enumerate(ins[477:], 477):
    lo, hi = struct.unpack("<iI", instruction)
    new_index = old_index+shift
    if hi in (0x00800000, 0x00900000) and (old_target:=old_index+lo) < 477:
      instruction = BR(old_target-new_index, inv=hi == 0x00900000)
    out.append(instruction)
  if len(out) > len(ins): raise RuntimeError(f"packed fused shader grew from {len(ins)} to {len(out)}")
  out += [NOP()] * (len(ins)-len(out))
  fregs, hregs = struct.unpack_from("<II", lib, reg_off+0x14)
  return inject(lib, image_off, image_size, reg_off, b"".join(out), fregs, hregs)


def fp32_acc_source(source:str) -> str:
  """Keep the hidden local tile in half, but accumulate both projections in float."""
  source = source.replace("read_imageh(", "read_imagef(")
  source = source.replace("half4 z", "float4 z").replace("half4 o", "float4 o")
  source = source.replace("half4 a", "float4 a").replace("half4 w", "float4 w")
  source = source.replace("(half4)(a", "(float4)(a")
  source = source.replace("(half4)(0)", "(float4)(0)")
  source = re.sub(r"=hidden\[([^]]+)\]", r"=convert_float4(hidden[\1])", source)
  return source


def patch_model(model, packed:bool=True, barriers:bool=True, inverse_packed:bool=False, parallel:bool=False,
                fp32_acc:bool=False) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  # Keep the local-memory barriers until the packed shader has been checked for
  # both timing and changed-input correctness on device.
  source = SOURCE_PARALLEL if parallel else SOURCE_TILED
  if fp32_acc: source = fp32_acc_source(source)
  if not barriers: source=source.replace("barrier(CLK_LOCAL_MEM_FENCE);", "")
  lib=Device["QCOM"].compiler.compile(source)
  if inverse_packed and not fp32_acc: lib=pack_tiled_inverse(lib)
  if packed and not fp32_acc: lib=pack_tiled_inner(lib)
  specs=((dtypes.half,(1,8192,4)),(dtypes.half,(1,8192,4)),(dtypes.half,(1,8320,4)),
         (dtypes.half,(192,320,4)),(dtypes.half,(1,192,4)),(dtypes.half,(64,768,4)),
         (dtypes.half,(1,64,4)),(dtypes.half,(1,64,4)))
  aux=(tuple(((i,dtype,shape),) for i,(dtype,shape) in enumerate(specs)),)
  replacements, skip, patched = {}, set(), 0
  for index in range(len(batch)-1):
    forward,inverse=batch[index:index+2]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in (forward,inverse)): continue
    if (plain_name(forward.src[0].arg.name),plain_name(inverse.src[0].arg.name)) != (FORWARD,INVERSE): continue
    info=replace(forward.src[0].arg,name="fused_vision_mlp",global_size=(1,32,1),local_size=((192 if parallel else 64),1,1),
                 globals=tuple(range(8)),outs=(0,),ins=tuple(range(1,8)),aux=aux)
    program=forward.src[0].replace(arg=info,src=forward.src[0].src[:2]+
      (forward.src[0].src[2].replace(arg=source),forward.src[0].src[3].replace(arg=lib)))
    replacements[index]=program.call(inverse.src[1],inverse.src[2],forward.src[2],forward.src[3],forward.src[4],
                                     inverse.src[4],inverse.src[5],inverse.src[6])
    skip.add(index+1)
    patched+=1
  new_batch=[replacements.get(i,call) for i,call in enumerate(batch) if i not in skip]
  if patched:
    model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
    model.captured.__dict__.pop("linear",None)
  return patched


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  ap.add_argument("--no-pack", action="store_true")
  ap.add_argument("--no-barrier", action="store_true")
  ap.add_argument("--inverse-pack", action="store_true")
  ap.add_argument("--parallel", action="store_true")
  ap.add_argument("--fp32-acc", action="store_true")
  args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_model(model, not args.no_pack, not args.no_barrier, args.inverse_pack, args.parallel, args.fp32_acc))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__=="__main__": main()

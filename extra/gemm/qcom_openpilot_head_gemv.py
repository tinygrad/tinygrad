#!/usr/bin/env python3
"""Transpose head-GEMV weight microtiles for vector FP32 accumulation."""
import argparse, itertools, os, pickle, struct
from dataclasses import replace

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name
from extra.gemm.ir3asm import BR, ISAM_F32, MAD_F32, MOV_F32, NOP, inject

TARGET = "r_128_16_4_32_4_batch2"

SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(16,1,1)))
__kernel void head_gemv_2048_512(write_only image2d_t O, read_only image2d_t A,
                                 read_only image2d_t W, read_only image2d_t B) {
  int out4=get_group_id(0), lid=get_local_id(0);
  float4 z=(float4)(0.0f);
  for (int r=0;r<32;r++) {
    float4 a=read_imagef(A,smp,(int2)(lid*32+r,0));
    int x=lid*128+r*4;
    float4 w0=read_imagef(W,smp,(int2)(x+0,out4));
    float4 w1=read_imagef(W,smp,(int2)(x+1,out4));
    float4 w2=read_imagef(W,smp,(int2)(x+2,out4));
    float4 w3=read_imagef(W,smp,(int2)(x+3,out4));
    z+=(float4)(a.x)*w0; z+=(float4)(a.y)*w1;
    z+=(float4)(a.z)*w2; z+=(float4)(a.w)*w3;
  }
  __local float4 partial[16];
  partial[lid]=z;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid==0) {
    z=(float4)(0.0f);
    for (int i=0;i<16;i++) z+=partial[i];
    z+=read_imagef(B,smp,(int2)(out4,0));
    z=select((float4)(0.0f),convert_float4(convert_half4(z)),isgreater(z,(float4)(0.0f)));
    write_imagef(O,(int2)(out4,0),z);
  }
}"""


def packed_weight(weight:UOp) -> UOp:
  original = np.asarray(weight.buffer.numpy()).reshape(128, 2112, 4)
  packed = np.array(original, copy=True)
  for out4 in range(128):
    for lid in range(16):
      for r in range(32):
        x = lid*128+r*4
        packed[out4, x:x+4] = original[out4, x:x+4].T
  return UOp.from_buffer(Buffer("QCOM", packed.size, weight.dtype, initial_value=packed.tobytes()))


def pack_lib(lib:bytes) -> bytes:
  image_off, image_size = struct.unpack_from("<I",lib,0xc0)[0], struct.unpack_from("<I",lib,0x100)[0]
  reg_off = struct.unpack_from("<I",lib,0x34)[0]
  ins = [lib[i:i+8] for i in range(image_off,image_off+image_size,8)]
  if len(ins) != 127: raise RuntimeError(f"expected 127 head GEMV instructions, got {len(ins)}")
  init_lo,init_hi = struct.unpack("<II",ins[14])
  init = struct.pack("<II",init_lo,(init_hi&~0xff)|24|0x300)
  out = list(ins[:14]) + [init,NOP(),NOP(),NOP()]
  loop_start = len(out)
  out += ins[18:21]
  for coord_ins,reg,coord in zip((21,28,35,42),(7,8,9,10),("r2.y","r2.w","r3.y","r3.w")):
    out += [ins[coord_ins],NOP(rpt=5),ISAM_F32(f"r{reg}.x",coord,1,0)]
  for component,weight_reg in zip("xyzw",range(7,11)):
    if os.getenv("HEAD_SCALAR"):
      for lane in "xyzw":
        out.append(MAD_F32(f"r6.{lane}",f"r0.{component}",f"r{weight_reg}.{lane}",f"r6.{lane}",
                           sy=component=="x" and lane=="x"))
    else:
      out.append(MAD_F32("r6.x",f"r0.{component}",f"r{weight_reg}.x","r6.x",rpt=3,r=True,sy=component=="x"))
  out += ins[49:55]
  out.append(BR(loop_start-len(out)))
  out += [ins[56],MOV_F32("r0.x","r6.x",rpt=3,r=True),NOP(),NOP(),NOP()] + ins[61:]
  out += [NOP()]*(len(ins)-len(out))
  if len(out) != len(ins): raise RuntimeError(f"packed head GEMV overflow: {len(out)}")
  fregs,hregs = struct.unpack_from("<II",lib,reg_off+0x14)
  return inject(lib,image_off,image_size,reg_off,b"".join(out),max(fregs&0x7fffffff,11)|(fregs&0x80000000),hregs)


def aux(*specs): return (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)


def patch_model(model) -> int:
  existing_slots = [x.arg.slot for x in model.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1) + 1)
  outer = model.captured.linear.src[0]
  batch, new_batch, patched = outer.src[0].src[0].src, [], 0
  lib = pack_lib(Device["QCOM"].compiler.compile_cached(SOURCE))
  for call in batch:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET:
      new_batch.append(call)
      continue
    template = call.src[0]
    info = replace(template.arg, name="head_gemv_2048_512", global_size=(128,1,1), local_size=(16,1,1),
                   globals=(0,1,2,3), outs=(0,), ins=(1,2,3),
                   aux=aux((dtypes.half,(1,136,4)), (dtypes.half,(1,520,4)),
                           (dtypes.half,(128,2112,4)), (dtypes.half,(1,128,4))))
    program = template.replace(arg=info, src=template.src[:2]+(template.src[2].replace(arg=SOURCE), template.src[3].replace(arg=lib)))
    new_batch += [program.call(call.src[1], call.src[3], packed_weight(call.src[4]), call.src[5]),
                  program.call(call.src[2], call.src[6], packed_weight(call.src[7]), call.src[8])]
    patched += 1
  if patched:
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); args=ap.parse_args()
  with open(args.input,"rb") as f: model=pickle.load(f)
  print("patched",patch_model(model))
  with open(args.output,"wb") as f: pickle.dump(model,f)


if __name__ == "__main__": main()

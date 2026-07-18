"""Vectorize the driving-vision uint8 input normalization kernel on QCOM."""
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_openpilot_ir3 import plain_name

TARGET = "E_8192_3_4_2_4"
SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void E_8192_3_4_2_4(write_only image2d_t O,__global uchar *A,__global uchar *B,
                             __global half *MEAN,__global half *STD) {
  int c=get_global_id(0),i=get_global_id(1),off=(c<<17)+(i<<2),mc=c<<2;
  uchar4 a0=vload4(0,A+off),a1=vload4(0,A+off+32768);
  uchar4 a2=vload4(0,A+off+65536),a3=vload4(0,A+off+98304);
  uchar4 b0=vload4(0,B+off),b1=vload4(0,B+off+32768);
  uchar4 b2=vload4(0,B+off+65536),b3=vload4(0,B+off+98304);
  half4 ma=vload4(0,MEAN+mc),mb=vload4(0,MEAN+mc+12);
  half4 ia=(half4)(1)/vload4(0,STD+mc),ib=(half4)(1)/vload4(0,STD+mc+12);
  int x=c+(i&7)*24,y=i>>3;
  write_imagef(O,(int2)(x,y),convert_float4(((half4)(a0.x,a1.x,a2.x,a3.x)-ma)*ia));
  write_imagef(O,(int2)(x+3,y),convert_float4(((half4)(b0.x,b1.x,b2.x,b3.x)-mb)*ib));
  write_imagef(O,(int2)(x+6,y),convert_float4(((half4)(a0.y,a1.y,a2.y,a3.y)-ma)*ia));
  write_imagef(O,(int2)(x+9,y),convert_float4(((half4)(b0.y,b1.y,b2.y,b3.y)-mb)*ib));
  write_imagef(O,(int2)(x+12,y),convert_float4(((half4)(a0.z,a1.z,a2.z,a3.z)-ma)*ia));
  write_imagef(O,(int2)(x+15,y),convert_float4(((half4)(b0.z,b1.z,b2.z,b3.z)-mb)*ib));
  write_imagef(O,(int2)(x+18,y),convert_float4(((half4)(a0.w,a1.w,a2.w,a3.w)-ma)*ia));
  write_imagef(O,(int2)(x+21,y),convert_float4(((half4)(b0.w,b1.w,b2.w,b3.w)-mb)*ib));
}"""


def patch_input_pack(jit) -> int:
  outer = jit.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  lib, new_batch, replaced = None, [], 0
  for call in batch:
    name = plain_name(call.src[0].arg.name) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM else ""
    if name == TARGET:
      if lib is None: lib = Device["QCOM"].compiler.compile_cached(SOURCE)
      program = call.src[0]
      program = program.replace(arg=replace(program.arg, global_size=(1, 64, 1), local_size=(3, 128, 1)),
                                src=program.src[:2] +
                                (program.src[2].replace(arg=SOURCE), program.src[3].replace(arg=lib)))
      call, replaced = call.replace(src=(program, *call.src[1:])), replaced+1
    new_batch.append(call)
  if replaced:
    jit.captured._linear = jit.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
    jit.captured.__dict__.pop("linear", None)
  return replaced

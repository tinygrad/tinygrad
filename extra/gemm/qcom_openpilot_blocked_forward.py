#!/usr/bin/env python3
"""Experimental 64-term FP16 partial / FP32 total OpenPilot projection."""
import argparse, pickle
from dataclasses import replace

from tinygrad import Device
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_32_192_4_4_64_4"

SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__kernel void r_32_192_4_4_64_4(write_only image2d_t O, read_only image2d_t A,
                                read_only image2d_t W, read_only image2d_t B) {
  int n=get_global_id(0), m=get_global_id(1), abase=m*260;
  float4 t0=(float4)(0),t1=(float4)(0),t2=(float4)(0),t3=(float4)(0);
  for (int kb=0;kb<64;kb+=16) {
    half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
    for (int k=kb;k<kb+16;k++) {
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
    t0+=convert_float4(r0); t1+=convert_float4(r1); t2+=convert_float4(r2); t3+=convert_float4(r3);
  }
  float4 b=read_imagef(B,smp,(int2)(n,0));
  write_imagef(O,(int2)(n,m),gelu(t0+b));
  write_imagef(O,(int2)(n+192,m),gelu(t1+b));
  write_imagef(O,(int2)(n+384,m),gelu(t2+b));
  write_imagef(O,(int2)(n+576,m),gelu(t3+b));
}"""


def patch_model(model, block4:int=16) -> int:
  if 64 % block4: raise ValueError("block4 must divide 64")
  source = SOURCE.replace("kb<64;kb+=16", f"kb<64;kb+={block4}").replace("k<kb+16", f"k<kb+{block4}")
  outer = model.captured.linear.src[0]
  batch, patched = list(outer.src[0].src[0].src), 0
  lib = Device["QCOM"].compiler.compile_cached(source)
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET: continue
    program = call.src[0]
    program = program.replace(arg=replace(program.arg, global_size=(24, 1, 1), local_size=(8, 32, 1)),
                              src=program.src[:2]+(program.src[2].replace(arg=source), program.src[3].replace(arg=lib)))
    batch[index] = call.replace(src=(program, *call.src[1:]))
    patched += 1
  if patched:
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("input"); ap.add_argument("output")
  ap.add_argument("--block4", type=int, default=16)
  args = ap.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  print("patched", patch_model(model, args.block4))
  with open(args.output, "wb") as f: pickle.dump(model, f)


if __name__ == "__main__": main()

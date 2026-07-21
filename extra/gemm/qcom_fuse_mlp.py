#!/usr/bin/env python3
"""Fuse OpenPilot transformer MLP projection pairs through local memory."""
import argparse, itertools, pickle
from dataclasses import replace

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def aux(*specs): return (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)


SOURCE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void fused_mlp(write_only image2d_t O,read_only image2d_t A,read_only image2d_t W1,
                        __global float *MUL,__global float *BIAS,read_only image2d_t W2,
                        read_only image2d_t X,read_only image2d_t S) {
  int lid=get_local_id(0),row=get_group_id(1);
  __local half4 hidden[384];
  for(int tile=0;tile<3;tile++) {
    int n4=lid+tile*128;
    float4 z=(float4)(0.0f);
    for(int k4=0;k4<96;k4++) {
      float4 a=convert_float4(read_imageh(A,smp,(int2)(k4,row)));
      float4 w0=convert_float4(read_imageh(W1,smp,(int2)(n4,k4*4+0)));
      float4 w1=convert_float4(read_imageh(W1,smp,(int2)(n4,k4*4+1)));
      float4 w2=convert_float4(read_imageh(W1,smp,(int2)(n4,k4*4+2)));
      float4 w3=convert_float4(read_imageh(W1,smp,(int2)(n4,k4*4+3)));
      z+=a.x*w0+a.y*w1+a.z*w2+a.w*w3;
    }
    z=select((float4)(0.0f),z,isgreater(z,(float4)(0.0f)));
    hidden[n4]=convert_half4((float4)(*MUL)*z*z+(float4)(*BIAS));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(lid<96) {
    float4 z=(float4)(0.0f);
    for(int k4=0;k4<384;k4++) {
      float4 a=convert_float4(hidden[k4]);
      float4 w0=convert_float4(read_imageh(W2,smp,(int2)(lid,k4*4+0)));
      float4 w1=convert_float4(read_imageh(W2,smp,(int2)(lid,k4*4+1)));
      float4 w2=convert_float4(read_imageh(W2,smp,(int2)(lid,k4*4+2)));
      float4 w3=convert_float4(read_imageh(W2,smp,(int2)(lid,k4*4+3)));
      z+=a.x*w0+a.y*w1+a.z*w2+a.w*w3;
    }
    int t=row*96+lid;
    write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(lid,0))+z);
  }
}"""


def build_program(template:UOp, lib:bytes):
  specs = ((dtypes.float, (1, 12288, 4)), (dtypes.half, (128, 96, 4)),
           (dtypes.half, (384, 384, 4)), (dtypes.float, (1,)), (dtypes.float, (1,)),
           (dtypes.half, (1536, 96, 4)), (dtypes.float, (1, 12288, 4)),
           (dtypes.float, (1, 96, 4)))
  info = replace(template.arg, name="fused_mlp", global_size=(1, 128, 1), local_size=(128, 1, 1),
                 globals=tuple(range(8)), outs=(0,), ins=(1,2,3,4,5,6,7), aux=aux(*specs))
  return template.replace(arg=info, src=template.src[:2]+(template.src[2].replace(arg=SOURCE), template.src[3].replace(arg=lib)))


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); args=ap.parse_args()
  with open(args.input,"rb") as f: model=pickle.load(f)
  existing=[x.arg.slot for x in model.captured.linear.toposort() if x.op is Ops.BUFFER and hasattr(x.arg,"slot") and x.arg.slot>=0]
  UOp.unique_num=itertools.count(max(existing,default=-1)+1)
  batch=model.captured.linear.src[0].src[0].src[0].src
  lib=Device["QCOM"].compiler.compile(SOURCE)
  repl, count={},0
  for i in range(len(batch)-3):
    calls=batch[i:i+4]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in calls): continue
    names=tuple(plain_name(x.src[0].arg.name) for x in calls)
    if names != ("gemm_h","epi3_fp32","gemm_h","epi_fp32"): continue
    if tuple(calls[0].src[0].arg.global_size)!=(12,8,1) or tuple(calls[2].src[0].arg.global_size)!=(3,8,1): continue
    g1,e1,g2,e2=calls; p=build_program(g1.src[0],lib)
    repl[i]=(p.call(e2.src[1],g1.src[1],g1.src[2],e1.src[2],e1.src[3],g2.src[2],e2.src[2],e2.src[3]),)
    repl[i+1]=repl[i+2]=repl[i+3]=()
    count+=1
  if count!=17: raise ValueError(f"expected 17 MLPs, found {count}")
  outer=model.captured.linear.src[0]
  new_batch=[new for i,old in enumerate(batch) for new in repl.get(i,(old,))]
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  with open(args.output,"wb") as f: pickle.dump(model,f)
  print(f"wrote {args.output}: fused {count} MLPs, calls {len(batch)} -> {len(new_batch)}")


if __name__=="__main__": main()

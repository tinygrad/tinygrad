#!/usr/bin/env python3
"""Fuse OpenPilot's QK, softmax, and AV calls with online-softmax attention."""
import argparse, pickle
from dataclasses import replace

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name

QK, SM, AV = "r_12_32_32_4_4_8_4", "softmax512", "r_32_96_4_4_32_4"


def aux(*specs): return (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)


def build_program(template:UOp, source:str, lib:bytes):
  specs = ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 13824, 4)),
           (dtypes.float, (1, 13824, 4)), (dtypes.float, (1, 12672, 4)))
  info = replace(template.arg, name="attention_online", global_size=(12, 8, 1), local_size=(8, 16, 1),
                 globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), aux=aux(*specs))
  return template.replace(arg=info, src=template.src[:2]+(template.src[2].replace(arg=source), template.src[3].replace(arg=lib)))


def make_source() -> str:
  qdecls = "\n".join(f"  float4 q{r};" for r in range(8))
  qloads = "\n".join(f"    q{r}=read_imagef(Q,smp,(int2)(h*9+{r}+qg*432+ql*108,0));" for r in range(8))
  score = []
  for r in range(8):
    score += [f"    int kb{r}=keyb*36+h*1152+{r*4};",
              f"    float4 k{r}0=read_imagef(K,smp,(int2)(kb{r},0));",
              f"    float4 k{r}1=read_imagef(K,smp,(int2)(kb{r}+1,0));",
              f"    float4 k{r}2=read_imagef(K,smp,(int2)(kb{r}+2,0));",
              f"    float4 k{r}3=read_imagef(K,smp,(int2)(kb{r}+3,0));",
              f"    s+=q{r}.xxxx*k{r}0+q{r}.yyyy*k{r}1+q{r}.zzzz*k{r}2+q{r}.wwww*k{r}3;"]
  updates = ["    float4 aa,bb;"]
  for lane, comp in enumerate("xyzw"):
    updates += [f"    float nm{lane}=fmax(mx,s.{comp});",
                f"    aa.{comp}=exp2((mx-nm{lane})*1.4426950408889634f);",
                f"    bb.{comp}=exp2((s.{comp}-nm{lane})*1.4426950408889634f); mx=nm{lane};"]
  return f"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(8,16,1)))
__kernel void attention_online(write_only image2d_t O,read_only image2d_t Q,read_only image2d_t K,read_only image2d_t V) {{
  int x=get_global_id(0),query=get_global_id(1),ox=get_local_id(0),ly=get_local_id(1);
  int h=x>>3,qg=query>>2,ql=query&3;
  __local float4 la[16],lb[16];
{qdecls}
  if(ox==0) {{
{qloads}
  }}
  float mx=-INFINITY,den=0.0f; float4 acc=(float4)(0.0f);
  for(int keyb=0;keyb<32;keyb++) {{
    if(ox==0) {{
      float4 s=(float4)(0.0f);
{chr(10).join(score)}
      s*=0.1767766922712326f;
{chr(10).join(updates)}
      la[ly]=aa; lb[ly]=bb;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    int vb=x*132+keyb*4;
    float4 v0=read_imagef(V,smp,(int2)(vb,0)),v1=read_imagef(V,smp,(int2)(vb+1,0));
    float4 v2=read_imagef(V,smp,(int2)(vb+2,0)),v3=read_imagef(V,smp,(int2)(vb+3,0));
    float4 aa=la[ly],bb=lb[ly];
    acc=acc*(float4)(aa.x)+v0*(float4)(bb.x); den=den*aa.x+bb.x;
    acc=acc*(float4)(aa.y)+v1*(float4)(bb.y); den=den*aa.y+bb.y;
    acc=acc*(float4)(aa.z)+v2*(float4)(bb.z); den=den*aa.z+bb.z;
    acc=acc*(float4)(aa.w)+v3*(float4)(bb.w); den=den*aa.w+bb.w;
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  write_imagef(O,(int2)(x+qg*384+ql*96,0),acc/(float4)(den));
}}"""


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("input"); ap.add_argument("output")
  args = ap.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  source = make_source(); lib = Device["QCOM"].compiler.compile(source)
  replacements, count = {}, 0
  for i in range(len(batch)-2):
    calls = batch[i:i+3]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in calls): continue
    if tuple(plain_name(x.src[0].arg.name) for x in calls) != (QK, SM, AV): continue
    qk, _sm, av = calls
    program = build_program(qk.src[0], source, lib)
    replacements[i] = (program.call(av.src[1], qk.src[2], qk.src[3], av.src[3]),)
    replacements[i+1] = replacements[i+2] = ()
    count += 1
  if count != 18: raise ValueError(f"expected 18 attention triples, found {count}")
  outer = model.captured.linear.src[0]
  new_batch = [new for i, old in enumerate(batch) for new in replacements.get(i, (old,))]
  model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
  model.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(model, f)
  print(f"wrote {args.output}: fused {count} attention triples, calls {len(batch)} -> {len(new_batch)}")


if __name__ == "__main__": main()

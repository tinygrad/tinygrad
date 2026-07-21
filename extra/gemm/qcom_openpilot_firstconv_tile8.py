#!/usr/bin/env python3
"""Replace driving_vision's first convolution with a wider spatial tile."""
import argparse, os, pickle
from dataclasses import replace

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET = "r_64_32_16_4_4_6_3_3_4"


def make_source(spatial:int, output_blocks:int, split:bool=False) -> str:
  fp32 = bool(int(os.getenv("FP32_TILE", "0")))
  vec, read, scalar = ("float4", "read_imagef", "float4") if fp32 else ("half4", "read_imageh", "half4")
  local_x = 16//output_blocks
  local_y = 128//local_x
  lines = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable", """
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
""", f"__attribute__((reqd_work_group_size({local_x},{local_y},1)))", """
__kernel void firstconv_tile8(write_only image2d_t O,read_only image2d_t A,
                              read_only image2d_t W,read_only image2d_t B) {
  int ob=get_global_id(0), xb=get_global_id(1), y=get_global_id(2);
"""]
  lines += [f"  {vec} z{s}_{n}=({vec})(0);" for s in range(spatial) for n in range(output_blocks)]
  lines.append("  for(int ic=0;ic<6;ic++) for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) {")
  lines.append(f"    int ax=xb*{spatial*12}+kx*6+ic, ay=y*2+ky-1;")
  lines += [f"    {vec} a{s}={read}(A,smp,(int2)(ax+{12*s-6},ay));" for s in range(spatial)]
  for n in range(output_blocks):
    lines.append(f"    int wp{n}=ic*12+kx*4+ky*72+(ob*{output_blocks}+{n})*216;")
    lines += [f"    {vec} w{n}{k}={read}(W,smp,(int2)(wp{n}+{k},0));" for k in (0, 1, 2, 3)]
  for s in range(spatial):
    for n in range(output_blocks):
      lines.append(f"    z{s}_{n}+=({scalar})(a{s}.x)*w{n}0+({scalar})(a{s}.y)*w{n}1+"
                   f"({scalar})(a{s}.z)*w{n}2+({scalar})(a{s}.w)*w{n}3;")
  lines.append("  }")
  if not split:
    for n in range(output_blocks):
      lines.append(f"  float4 b{n}=read_imagef(B,smp,(int2)(ob*{output_blocks}+{n},0));")
  for s in range(spatial):
    for n in range(output_blocks):
      raw = f"z{s}_{n}" if fp32 else f"convert_float4(z{s}_{n})"
      value = raw if split else f"gelu({raw}+b{n})"
      lines.append(f"  write_imagef(O,(int2)(ob*{output_blocks}+{n}+xb*{spatial*16}+{s*16},y),{value});")
  lines.append("}")
  return "\n".join(lines)


EPILOGUE = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void firstconv_gelu(write_only image2d_t O,read_only image2d_t B,read_only image2d_t T) {
  int x=get_global_id(0),y=get_global_id(1);
  write_imagef(O,(int2)(x,y),gelu(convert_float4(read_imageh(T,smp,(int2)(x,y)))+read_imagef(B,smp,(int2)(x&15,0))));
}"""


def patch_model(model, spatial:int, output_blocks:int, split:bool=False) -> int:
  if spatial*output_blocks not in (4, 8) or 16%output_blocks: raise ValueError("tile must contain four or eight vectors")
  outer, source = model.captured.linear.src[0], make_source(spatial, output_blocks, split)
  batch, lib, patched = list(outer.src[0].src[0].src), Device["QCOM"].compiler.compile(source), 0
  replacements:dict[int, tuple[UOp, ...]] = {}
  epi_lib = Device["QCOM"].compiler.compile(EPILOGUE) if split else None
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET: continue
    old = call.src[0]
    local_x, local_y = 16//output_blocks, 128//(16//output_blocks)
    global_y = (128//spatial)//local_y
    info = replace(old.arg, name=f"firstconv_tile{spatial}x{output_blocks*4}", global_size=(1, global_y, 64),
                   local_size=(local_x, local_y, 1))
    program = old.replace(arg=info, src=old.src[:2]+(old.src[2].replace(arg=source), old.src[3].replace(arg=lib)))
    if split:
      temporary = UOp.new_buffer("QCOM", call.src[1].buffer.size, dtypes.half, num=-3_000_000)
      temporary.buffer.ensure_allocated()
      compute = call.replace(src=(program, temporary, *call.src[2:]))
      epi_aux = ((((0, dtypes.half, (64, 2048, 4)),), ((1, dtypes.half, (1, 16, 4)),),
                  ((2, dtypes.half, (64, 2048, 4)),)),)
      epi_info = replace(old.arg, name="firstconv_gelu", global_size=(16, 64, 1), local_size=(128, 1, 1),
                         globals=(0, 1, 2), outs=(0,), ins=(1, 2), aux=epi_aux)
      epi_program = old.replace(arg=epi_info, src=old.src[:2]+(old.src[2].replace(arg=EPILOGUE), old.src[3].replace(arg=epi_lib)))
      replacements[index] = (compute, epi_program.call(call.src[1], call.src[4], temporary))
    else: replacements[index] = (call.replace(src=(program, *call.src[1:])),)
    patched += 1
  if patched:
    new_batch = [new for index, call in enumerate(batch) for new in replacements.get(index, (call,))]
    model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return patched


def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--spatial", type=int, default=8)
  parser.add_argument("--output-blocks", type=int, default=1)
  parser.add_argument("--split", action="store_true")
  args=parser.parse_args()
  with open(args.input, "rb") as f: model=pickle.load(f)
  print("patched", patch_model(model, args.spatial, args.output_blocks, args.split))
  with open(args.output, "wb") as f: pickle.dump(model, f)


if __name__ == "__main__": main()

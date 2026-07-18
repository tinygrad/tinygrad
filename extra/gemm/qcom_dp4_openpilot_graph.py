#!/usr/bin/env python3
"""Replace selected cached OpenPilot FP16 GEMMs with dynamically-scaled A630 DP4 kernels."""
import argparse, itertools, pickle
from dataclasses import replace

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def aux(*specs):
  return (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)


def build_program(template:UOp, name:str, source:str, lib:bytes, global_size, local_size, specs, outs, ins):
  info = replace(template.arg, name=name, global_size=global_size, local_size=local_size,
                 globals=tuple(range(len(specs))), outs=outs, ins=ins, aux=aux(*specs))
  return template.replace(arg=info, src=template.src[:2]+(template.src[2].replace(arg=source), template.src[3].replace(arg=lib)))


def pack_unsigned_weights(matrix:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Per-output-channel symmetric int8 quantization, biased to uint8 for A630 signed*unsigned DP4."""
  k, n = matrix.shape
  scale = np.max(np.abs(matrix), axis=0).astype(np.float32) / 127.0
  scale[scale == 0] = 1.0
  signed = np.clip(np.rint(matrix/scale), -127, 127).astype(np.int16)
  unsigned = (signed+128).astype(np.uint8).reshape(k//16, 4, 4, n//4, 4)
  words = np.zeros((k//16, 4, n//4, 4), dtype=np.uint32)
  for lane in range(4): words |= unsigned[:, :, lane].astype(np.uint32) << (8*lane)
  return words.reshape(k//4, n//4, 4), scale


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  ap.add_argument("--indices", required=True, help="comma-separated indices in the cached gemm_h call sequence")
  args = ap.parse_args()
  selected = {int(x) for x in args.indices.split(",") if x}
  with open(args.input, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  existing_slots = [x.arg.slot for x in model.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1)+1)
  dev = Device["QCOM"]

  pack_sources, pack_libs, dp4_libs = {}, {}, {}
  epi_sources, epi_libs = {}, {}
  replacements = {}
  candidates = [(i, call) for i, call in enumerate(batch) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                plain_name(call.src[0].arg.name) == "gemm_h" and int(call.src[0].arg.global_size[0]) in (3, 12)]
  for occurrence, (index, call) in enumerate(candidates):
    if occurrence not in selected: continue
    gsx = int(call.src[0].arg.global_size[0])
    m, k, n = (128, 384, 1536) if gsx == 12 else (128, 1536, 384)
    epi_call = batch[index+1]
    expected_epi = "epi3_fp32" if gsx == 12 else "epi_fp32"
    if epi_call.op is not Ops.CALL or plain_name(epi_call.src[0].arg.name) != expected_epi:
      raise ValueError(f"cached GEMM {occurrence} is followed by {plain_name(epi_call.src[0].arg.name)}, expected {expected_epi}")

    if k not in pack_libs:
      pack_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void qpack(__global uint *O,__global float *S,__global int *SUM,read_only image2d_t A) {{
  int lid=get_local_id(0),row=get_group_id(0); __local float vmax[128]; __local int vsum[128];
  float mx=0.0f; for(int k4=lid;k4<{k//4};k4+=128) {{
    float4 v=fabs(convert_float4(read_imageh(A,smp,(int2)(k4,row))));
    mx=fmax(mx,fmax(fmax(v.x,v.y),fmax(v.z,v.w))); }}
  vmax[lid]=mx; barrier(CLK_LOCAL_MEM_FENCE);
  for(int d=64;d;d>>=1) {{ if(lid<d) vmax[lid]=fmax(vmax[lid],vmax[lid+d]); barrier(CLK_LOCAL_MEM_FENCE); }}
  float sc=vmax[0]==0.0f?1.0f:vmax[0]/127.0f; int sm=0;
  for(int k4=lid;k4<{k//4};k4+=128) {{
    float4 v=convert_float4(read_imageh(A,smp,(int2)(k4,row)))/(float4)(sc);
    char4 z=convert_char4_sat_rte(v); O[row*{k//4}+k4]=as_uint(z);
    sm+=(int)z.x+(int)z.y+(int)z.z+(int)z.w; }}
  vsum[lid]=sm; barrier(CLK_LOCAL_MEM_FENCE);
  for(int d=64;d;d>>=1) {{ if(lid<d) vsum[lid]+=vsum[lid+d]; barrier(CLK_LOCAL_MEM_FENCE); }}
  if(lid==0) {{ S[row]=sc; SUM[row]=vsum[0]; }}
}}"""
      pack_sources[k] = pack_source
      pack_libs[k] = dev.compiler.compile(pack_source)
    pack = build_program(call.src[0], "qpack", pack_sources[k], pack_libs[k], (m, 1, 1), (128, 1, 1),
                         ((dtypes.uint, (m*k//4,)), (dtypes.float, (m,)), (dtypes.int, (m,)),
                          (dtypes.half, (m, k//4, 4))), (0, 1, 2), (3,))

    if (m, n, k) not in dp4_libs:
      q.M, q.N, q.K, q.K4 = m, n, k, k//4
      env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_u32(1, 128))
      shader, hregs, fregs, _ = q.build_4x4_dp4_shader(dev, 128, k, mixed=True, coord_delay=4)
      dp4_libs[(m, n, k)] = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
    dp4 = build_program(call.src[0], "gemm_h", "packed signed-u8 DP4 GEMM", dp4_libs[(m, n, k)],
                        (n//128, m//16, 1), (128, 1, 1),
                        ((dtypes.uint, (m, k//16, 4)), (dtypes.uint, (k//4, n//4, 4)),
                         (dtypes.int, (m*n,))), (2,), (0, 1))

    if gsx not in epi_libs:
      if gsx == 12:
        epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi3_dp4(__global half *O,__global float *S,__global float *B,__global int *C,
                        __global float *AS,__global int *SUM,__global float *WS) {
  int t=get_global_id(0),row=t/384,col=t-row*384,y=row>>2,r=row&3,o=(y*1536+r*384+col)*4;
  int4 d=vload4(0,C+row*1536+col*4)-(int4)(128*SUM[row]);
  float4 z=convert_float4(d)*(float4)(AS[row])*vload4(0,WS+col*4);
  z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
}"""
      else:
        epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi_dp4(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global int *C,
                       __global float *AS,__global int *SUM,__global float *WS) {
  int t=get_global_id(0),row=t/96,col=t-row*96;
  int4 d=vload4(0,C+row*384+col*4)-(int4)(128*SUM[row]);
  float4 v=convert_float4(d)*(float4)(AS[row])*vload4(0,WS+col*4);
  write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(col,0))+v);
}"""
      epi_sources[gsx], epi_libs[gsx] = epi_source, dev.compiler.compile(epi_source)

    matrix = np.asarray(call.src[2].buffer.numpy(), dtype=np.float32).reshape(k, n)
    packed_weight_np, weight_scale_np = pack_unsigned_weights(matrix)
    packed_weight = UOp.new_buffer("QCOM", packed_weight_np.size, dtypes.uint)
    packed_weight.buffer.ensure_allocated(); packed_weight.buffer.copyin(memoryview(packed_weight_np).cast("B"))
    weight_scale = UOp.new_buffer("QCOM", n, dtypes.float)
    weight_scale.buffer.ensure_allocated(); weight_scale.buffer.copyin(memoryview(weight_scale_np).cast("B"))
    packed_activation = UOp.new_buffer("QCOM", m*k//4, dtypes.uint); packed_activation.buffer.ensure_allocated()
    activation_scale = UOp.new_buffer("QCOM", m, dtypes.float); activation_scale.buffer.ensure_allocated()
    activation_sum = UOp.new_buffer("QCOM", m, dtypes.int); activation_sum.buffer.ensure_allocated()
    scratch = UOp.new_buffer("QCOM", m*n, dtypes.int); scratch.buffer.ensure_allocated()

    pack_call = pack.call(packed_activation, activation_scale, activation_sum, call.src[1])
    dp4_call = dp4.call(packed_activation, packed_weight, scratch)
    if gsx == 12:
      epi = build_program(epi_call.src[0], "epi3_dp4", epi_sources[gsx], epi_libs[gsx], (384, 1, 1), (128, 1, 1),
                          ((dtypes.half, (128*1536,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                           (dtypes.int, (m*n,)), (dtypes.float, (m,)), (dtypes.int, (m,)), (dtypes.float, (n,))),
                          (0,), (1, 2, 3, 4, 5, 6))
      epi_new = epi.call(epi_call.src[1], epi_call.src[2], epi_call.src[3], scratch,
                         activation_scale, activation_sum, weight_scale)
    else:
      epi = build_program(epi_call.src[0], "epi_dp4", epi_sources[gsx], epi_libs[gsx], (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 96, 4)),
                           (dtypes.int, (m*n,)), (dtypes.float, (m,)), (dtypes.int, (m,)), (dtypes.float, (n,))),
                          (0,), (1, 2, 3, 4, 5, 6))
      epi_new = epi.call(epi_call.src[1], epi_call.src[2], epi_call.src[3], scratch,
                         activation_scale, activation_sum, weight_scale)
    replacements[index] = (pack_call, dp4_call)
    replacements[index+1] = (epi_new,)
    print(f"occurrence={occurrence} geometry={gsx} shape={m}x{n}x{k}")

  outer = model.captured.linear.src[0]
  new_batch = [new for i, old in enumerate(batch) for new in replacements.get(i, (old,))]
  model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
  model.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(model, f)
  print(f"wrote {args.output} with {len(replacements)//2} DP4 GEMMs")


if __name__ == "__main__": main()

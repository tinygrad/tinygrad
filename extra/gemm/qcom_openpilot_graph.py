#!/usr/bin/env python3
"""Replace the dominant openpilot projection with a hand FP16 GEMM experiment."""
import argparse, itertools, pickle, struct
from dataclasses import replace

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.engine.jit import _prepare_jit_inputs, create_graph_call
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm.ir3asm import BR, CMPS_S_EQ, MOV_F32, MOV_S32, NOP, QUAD_BRCST, SHL_B, get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import patch_auto_rpt_f32, patch_openpilot_target2_rpt3, patch_openpilot_target46_f16_rpt3, patch_openpilot_target46_rpt3, patch_openpilot_target5_rpt3, patch_openpilot_target7_rpt3, patch_openpilot_target9_rpt3, plain_name

TARGET = "r_32_96_4_4_384_4"
TARGET2 = "r_48_128_4_4_192_4"
TARGET3 = "r_32_384_4_4_96_4"
TARGET4 = "r_128_192_4_4_48_4"
TARGET5 = "r_48_128_4_4_96_4"
TARGET6 = "r_128_96_4_4_48_4"
TARGET7 = "r_36_32_8_4_4_96_4"
TARGET8 = "r_144_16_4_576_2_4"
TARGET9 = "r_32_96_4_4_96_4n1"
TARGET10 = "r_24_32_16_4_4_6_7_7_4"
TARGET11 = "r_16_8_96_4_4_7_7"
TARGET12 = "r_32_16_48_4_4_7_7"
M, N, K = 128, 384, 1536

EPILOGUE = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global half *C) {
  int t=get_global_id(0),row=t/96,col=t-row*96;
  float4 v=convert_float4(vload4(0,C+row*1024+col*4));
  write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(col,0))+v);
}"""

EPILOGUE_FP32 = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi_fp32(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global float *C) {
  int t=get_global_id(0),row=t/96,col=t-row*96;
  float4 v=vload4(0,C+row*1024+col*4);
  write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(col,0))+v);
}"""

EPILOGUE2 = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi2(write_only image2d_t O,read_only image2d_t X,__global half *C) {
  int t=get_global_id(0),row=t/128,col=t-row*128,y=row>>2,r=row&3,x=col+r*128;
  float4 v=convert_float4(vload4(0,C+row*1024+col*4));
  write_imagef(O,(int2)(x,y),read_imagef(X,smp,(int2)(x,y))+v);
}"""

EPILOGUE2_FP32 = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi2_fp32(write_only image2d_t O,read_only image2d_t X,__global float *C) {
  int t=get_global_id(0),row=t/128,col=t-row*128,y=row>>2,r=row&3,x=col+r*128;
  float4 v=vload4(0,C+row*1024+col*4);
  write_imagef(O,(int2)(x,y),read_imagef(X,smp,(int2)(x,y))+v);
}"""


def aux(*specs):
  return (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)


def enable_fp16(source:str) -> str:
  pragma = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
  return source if source.startswith(pragma) else pragma+source


def build_program(template:UOp, name:str, source:str, lib:bytes, global_size, local_size, specs, outs, ins):
  info = replace(template.arg, name=name, global_size=global_size, local_size=local_size, globals=tuple(range(len(specs))),
                 outs=outs, ins=ins, aux=aux(*specs))
  return template.replace(arg=info, src=template.src[:2]+(template.src[2].replace(arg=source), template.src[3].replace(arg=lib)))


def make_ternary_sparse(dev, template:UOp, transposed:np.ndarray, m:int, n:int, k:int, stride:int=1024):
  """Build a CSR GEMM for global-scale ternary weights (-scale, 0, +scale)."""
  matrix = np.asarray(transposed, dtype=np.float32).reshape(k, n//4, 4)
  scale = float(np.max(np.abs(matrix))) * float(getenv("OPENPILOT_TERNARY_SCALE", 1.0))
  ternary = np.where(np.abs(matrix) >= scale * 0.5, np.copysign(scale, matrix), 0).astype(np.float16)
  offsets, indices, values = [0], [], []
  for col4 in range(n//4):
    for kk in range(k):
      if np.any(ternary[kk, col4] != 0):
        indices.append(kk)
        values.append(ternary[kk, col4])
    offsets.append(len(indices))
  indices_np = np.asarray(indices or [0], dtype=np.uint16)
  values_np = np.asarray(values or [[0, 0, 0, 0]], dtype=np.float16).reshape(-1, 4)
  offsets_np = np.asarray(offsets, dtype=np.int32)

  def upload(arr, dtype):
    ret = UOp.new_buffer("QCOM", arr.size, dtype)
    ret.buffer.ensure_allocated()
    ret.buffer.copyin(memoryview(arr).cast("B"))
    return ret
  offbuf, idxbuf, valbuf = upload(offsets_np, dtypes.int), upload(indices_np, dtypes.ushort), upload(values_np, dtypes.half)
  temporary = UOp.new_buffer("QCOM", m*stride, dtypes.half)
  temporary.buffer.ensure_allocated()
  if not values: temporary.buffer.copyin(memoryview(np.zeros(m*stride, dtype=np.float16)).cast("B"))
  source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_sparse(read_only image2d_t A,__global int *OFF,__global ushort *IDX,
                          __global half *W,__global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,col4=get_group_id(0)*32+(lid&31),row=get_group_id(1)*16+tm*4;
  if(col4>={n//4}) return;
  half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
  for(int i=OFF[col4];i<OFF[col4+1];i++) {{
    int kk=IDX[i],k4=kk>>2,kc=kk&3; half4 w=vload4(i,W);
    half4 a0=read_imageh(A,smp,(int2)(k4,min(row,{m-1})));
    half4 a1=read_imageh(A,smp,(int2)(k4,min(row+1,{m-1})));
    half4 a2=read_imageh(A,smp,(int2)(k4,min(row+2,{m-1})));
    half4 a3=read_imageh(A,smp,(int2)(k4,min(row+3,{m-1})));
    half x0=kc==0?a0.x:kc==1?a0.y:kc==2?a0.z:a0.w;
    half x1=kc==0?a1.x:kc==1?a1.y:kc==2?a1.z:a1.w;
    half x2=kc==0?a2.x:kc==1?a2.y:kc==2?a2.z:a2.w;
    half x3=kc==0?a3.x:kc==1?a3.y:kc==2?a3.z:a3.w;
    r0+=x0*w;r1+=x1*w;r2+=x2*w;r3+=x3*w;
  }}
  if(row<{m}) vstore4(r0,0,C+row*{stride}+col4*4);
  if(row+1<{m}) vstore4(r1,0,C+(row+1)*{stride}+col4*4);
  if(row+2<{m}) vstore4(r2,0,C+(row+2)*{stride}+col4*4);
  if(row+3<{m}) vstore4(r3,0,C+(row+3)*{stride}+col4*4);
}}"""
  lib = dev.compiler.compile_cached(source)
  program = build_program(template, "gemm_sparse", source, lib, ((n+127)//128, (m+15)//16, 1), (128, 1, 1),
                          ((dtypes.half, (m, k//4, 4)), (dtypes.int, (len(offsets_np),)),
                           (dtypes.ushort, (len(indices_np),)), (dtypes.half, (len(values_np)*4,)),
                           (dtypes.half, (m*stride,))), (4,), (0, 1, 2, 3))
  print(f"ternary {m}x{n}x{k}: {len(values)} nonzero half4 vectors ({len(values)/(k*n/4):.3%})")
  return (None if not values else program), offbuf, idxbuf, valbuf, temporary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()
  with open(args.input, "rb") as f: jit = pickle.load(f)

  # Pickle restores buffer UOps but not UOp.unique_num. Without advancing it,
  # the first newly allocated buffer reuses slot zero and can alias an existing
  # model buffer in the global UOp -> Buffer map. This is silent corruption.
  existing_slots = [x.arg.slot for x in jit.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1) + 1)

  outer = jit.captured.linear.src[0]
  graph = outer.src[0]
  batch = graph.src[0].src
  materialize_weights = bool(getenv("OPENPILOT_MATERIALIZE_DYNAMIC_WEIGHTS"))
  if materialize_weights:
    materialize_inputs = {name:Tensor.zeros(*view.shape, dtype=dtype, device=device).contiguous().realize()
                          for name, (view, _vars, dtype, device) in
                          zip(jit.captured.expected_names, jit.captured.expected_input_info)}
    materialize_uops, materialize_vars = _prepare_jit_inputs((), materialize_inputs)[:2]
    batch_indices = {call:i for i, call in enumerate(batch)}

  def materialize_before(call:UOp) -> None:
    if not materialize_weights: return
    run_linear(UOp(Ops.LINEAR, src=(create_graph_call(batch[:batch_indices[call]]),)), materialize_vars,
               input_uops=materialize_uops, jit=True, wait=True)
  targets = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
             plain_name(call.src[0].arg.name) == TARGET]
  targets2 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET2]
  targets3 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET3]
  targets4 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET4]
  targets5 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET5]
  targets6 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET6]
  targets7 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET7]
  targets8 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET8]
  targets9 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == TARGET9]
  cached_gemm1_mixed_k = getenv("OPENPILOT_PATCH_GEMM1_MIXEDK", 0)
  cached_gemm1_split_k = getenv("OPENPILOT_PATCH_GEMM1_SPLITK", 0)
  if cached_gemm1_mixed_k and cached_gemm1_split_k: raise ValueError("choose cached GEMM1 mixed-K or split-K, not both")
  cached_gemm1_mixed = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                        plain_name(call.src[0].arg.name) == "gemm_h" and tuple(call.src[0].arg.global_size) == (3, 8, 1)] \
                       if cached_gemm1_mixed_k else []
  cached_gemm1_split = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                        plain_name(call.src[0].arg.name) == "gemm_h" and tuple(call.src[0].arg.global_size) == (3, 8, 1)] \
                       if cached_gemm1_split_k else []
  quant_gemm_bits = getenv("OPENPILOT_QUANT_GEMM_BITS", 0)
  quant_gemm_block = getenv("OPENPILOT_QUANT_GEMM_BLOCK", 0)
  quant_gemm_activations = bool(getenv("OPENPILOT_QUANT_GEMM_ACTIVATIONS"))
  quant_gemm_affine = bool(getenv("OPENPILOT_QUANT_GEMM_AFFINE"))
  quant_gemm_geoms = set(getenv("OPENPILOT_QUANT_GEMM_GEOMS", "3,12").split(",")) if quant_gemm_bits else set()
  all_cached_gemms = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                      plain_name(call.src[0].arg.name) == "gemm_h"]
  cached_fp16_target3_all = [call for call in all_cached_gemms if tuple(call.src[0].arg.global_size) == (12, 8, 1)]
  cached_fp16_target3_indices = {int(x) for x in str(getenv("OPENPILOT_CACHED_FP16_TARGET3_INDICES", "")).split(",") if x}
  cached_fp16_target3 = [call for i, call in enumerate(cached_fp16_target3_all)
                         if getenv("OPENPILOT_CACHED_FP16_TARGET3") and
                         (not cached_fp16_target3_indices or i in cached_fp16_target3_indices)]
  cached_fp16_target1_all = [call for call in all_cached_gemms
                             if tuple(call.src[0].arg.global_size) == (3, 8, 1) and
                             call.src[1].buffer.size == 128*1536]
  cached_fp16_target1_indices = {int(x) for x in str(getenv("OPENPILOT_CACHED_FP16_TARGET1_INDICES", "")).split(",") if x}
  cached_fp16_target1 = [call for i, call in enumerate(cached_fp16_target1_all)
                         if getenv("OPENPILOT_CACHED_FP16_TARGET1") and
                         (not cached_fp16_target1_indices or i in cached_fp16_target1_indices)]
  cached_fp32_refresh_geoms = {int(x) for x in str(getenv("OPENPILOT_CACHED_FP32_REFRESH", "")).split(",") if x}
  cached_fp32_refresh = [call for call in all_cached_gemms
                         if int(call.src[0].arg.global_size[0]) in cached_fp32_refresh_geoms]
  quant_indices = ({int(x) for x in getenv("OPENPILOT_QUANT_GEMM_INDICES", "").split(",") if x} or
                   set(range(len(all_cached_gemms)))) if quant_gemm_bits else set()
  quant_gemms = [call for i, call in enumerate(all_cached_gemms)
                 if i in quant_indices and str(call.src[0].arg.global_size[0]) in quant_gemm_geoms]
  targets10 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
               plain_name(call.src[0].arg.name) in (TARGET10, "firstconv_fast")]
  targets11 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
               plain_name(call.src[0].arg.name) == TARGET11]
  targets12 = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
               plain_name(call.src[0].arg.name) == TARGET12]
  if not getenv("OPENPILOT_REDUCE8"): targets8 = []
  if not (getenv("OPENPILOT_FIRSTCONV") or getenv("OPENPILOT_FIRSTCONV_LOCAL") or getenv("OPENPILOT_FIRSTCONV_LOCALW") or
          getenv("OPENPILOT_FIRSTCONV_HALF") or getenv("OPENPILOT_FIRSTCONV_HALF_ACC")): targets10 = []
  if not getenv("OPENPILOT_DW_PAIR_Y"): targets11 = []
  if not getenv("OPENPILOT_DW32_PAIR_Y"): targets12 = []
  # Restrict experimental rewrites to exact model occurrences.  Repeated programs
  # share the same UOp shape, so family-only selection is too coarse for numerical
  # sensitivity sweeps.
  def select_target_indices(calls:list[UOp], family:int) -> list[UOp]:
    indices = str(getenv(f"OPENPILOT_TARGET{family}_INDICES", ""))
    return calls if not indices else [calls[int(i)] for i in indices.split(",")]
  targets = select_target_indices(targets, 1)
  targets2 = select_target_indices(targets2, 2)
  targets3 = select_target_indices(targets3, 3)
  targets4 = select_target_indices(targets4, 4)
  targets5 = select_target_indices(targets5, 5)
  targets6 = select_target_indices(targets6, 6)
  targets7 = select_target_indices(targets7, 7)
  targets9 = select_target_indices(targets9, 9)
  lowrank2_rank = getenv("OPENPILOT_LOWRANK2", 0)
  targets2_lowrank = targets2 if lowrank2_rank else []
  targets_fp32 = targets if getenv("OPENPILOT_FP32_HAND_GEMM") else []
  targets3_fp32 = targets3 if getenv("OPENPILOT_FP32_TARGET3") else []
  split_k = getenv("OPENPILOT_SPLITK", 0)
  targets_split = targets if split_k else []
  mixed_k = getenv("OPENPILOT_MIXEDK", 0)
  targets_mixed = targets if mixed_k else []
  localb_tile = getenv("OPENPILOT_LOCALB_TILE", 0)
  targets_localb = targets if localb_tile else []
  targets_globalb = targets if getenv("OPENPILOT_GLOBALB") else []
  targets_quadmap = targets if getenv("OPENPILOT_QUADMAP") else []
  wide_fp32 = set(getenv("OPENPILOT_FP32_WIDE_TARGETS", "2,5").split(",")) if getenv("OPENPILOT_FP32_WIDE") else set()
  targets2_fp32 = targets2 if "2" in wide_fp32 else []
  targets5_fp32 = targets5 if "5" in wide_fp32 else []
  targets5_patch = targets5 if getenv("OPENPILOT_PATCH_TARGET5_RPT3") else []
  targets2_patch = targets2 if getenv("OPENPILOT_PATCH_TARGET2_RPT3") else []
  targets7_patch = targets7 if getenv("OPENPILOT_PATCH_TARGET7_RPT3") else []
  targets7_local = targets7 if str(getenv("OPENPILOT_TARGET7_LOCAL", "")) else []
  targets7_local_cache = targets7 if getenv("OPENPILOT_TARGET7_LOCAL_CACHE") else []
  targets7_quad = targets7 if getenv("OPENPILOT_TARGET7_QUAD") else []
  target7_coord_delay = str(getenv("OPENPILOT_TARGET7_COORD_DELAY", ""))
  targets7_coord = targets7 if target7_coord_delay else []
  targets7_half_only = targets7 if getenv("OPENPILOT_HALF_TARGET7_ONLY") else []
  target7_half_weights = {int(x) for x in getenv("OPENPILOT_HALF_TARGET7_WEIGHTS", "").split(",") if x}
  half_weight_only_families = set(getenv("OPENPILOT_HALF_WEIGHT_ONLY", "").split(","))-{""}
  half_weight_only_calls = {family:calls for family, calls in {"4":targets4, "6":targets6}.items()
                            if family in half_weight_only_families}
  extra_half_weight_indices = {
    "r_512_48_4_4_24_4":4, "r_32_16_48_4_4_7_7":3, "r_512_96_4_4_24_4":4,
    "r_24_512_4_4_96_4":4, "r_48_16_8_4_4_24_4_3_3":3, "r_96_8_4_4_4_48_4_3_3":3,
    "r_48_16_8_4_4_24_3_3_4":3, "r_96_8_4_4_4_48_3_3_4":3,
    "r_144_4_2_4_4_96_4_3_3":3, "r_8_576_4_4_144_4":4, "r_144_16_4_576_2_4":5,
    "r_24_512_4_4_48_4":4, "r_24_512_4_4_48_4n1":4, "r_16_8_96_4_4_7_7":3,
    "r_32_96_4_4_96_4":5, "r_96_32_4_4_384_4":5, "r_54_8_8_4_4_144_4":3,
    "r_8_144_4_4_144_4":5, "r_8_144_4_4_576_4":5, "r_8_144_4_4_144_4n1":5,
    "r_48_128_4_4_96_4":4, "r_36_32_8_4_4_96_4":3,
  }
  selected_extra_half = set(getenv("OPENPILOT_HALF_EXTRA_WEIGHTS", "").split(","))-{""}
  all_extra_half_weight_calls = [(call, extra_half_weight_indices[name]) for call in batch
                                 if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                                 (name:=plain_name(call.src[0].arg.name)) in selected_extra_half]
  extra_half_indices = {int(x) for x in str(getenv("OPENPILOT_HALF_EXTRA_WEIGHT_INDICES", "")).split(",") if x}
  extra_half_weight_calls = [item for i, item in enumerate(all_extra_half_weight_calls)
                             if not extra_half_indices or i in extra_half_indices]
  half_patch_weights = set(getenv("OPENPILOT_HALF_PATCH_WEIGHTS", "").split(","))-{""}
  half_patch_weight_indices = {family:{int(x) for x in str(getenv(f"OPENPILOT_HALF_PATCH_WEIGHT_INDICES_{family}", "")).split(",") if x}
                               for family in half_patch_weights}
  def should_half_patch_weight(family:str, index:int) -> bool:
    return family in half_patch_weights and (not half_patch_weight_indices[family] or index in half_patch_weight_indices[family])
  targets9_patch = targets9 if getenv("OPENPILOT_PATCH_TARGET9_RPT3") else []
  patch46 = str(getenv("OPENPILOT_PATCH_TARGET46_RPT3", ""))
  targets46_patch = ((targets4 if patch46 in ("1", "4") else []) +
                     (targets6 if patch46 in ("1", "6") else []))
  targets7_native = targets7 if getenv("OPENPILOT_FP16_TARGET7_NATIVE") else []
  targets3_native = targets3 if getenv("OPENPILOT_FP16_TARGET3_NATIVE") else []
  # Keep the compiler-generated kernel's launch geometry and epilogue, but let selected
  # matrix inputs and accumulators use native half arithmetic.  Capturing these lists
  # before the hand-GEMM filtering below keeps this experiment independent of that path.
  selected_native = set(getenv("OPENPILOT_FP16_NATIVE_TARGETS", "").split(","))-{""}
  native_families = {name:calls for name, calls in {
    "2": targets2, "4": targets4, "5": targets5, "6": targets6, "9": targets9,
  }.items() if name in selected_native}
  selected_auto_rpt = set(getenv("OPENPILOT_AUTO_RPT_TARGETS", "").split(","))-{""}
  auto_rpt_families = {name:calls for name, calls in {
    "2": targets2, "4": targets4, "5": targets5, "6": targets6, "7": targets7, "9": targets9,
  }.items() if name in selected_auto_rpt}
  targets3_split = targets3 if getenv("OPENPILOT_FP16_TARGET3_SPLIT") else []
  targets9_fp32 = targets9 if getenv("OPENPILOT_FP32_TARGET9") else []
  fp32_small_targets = set(str(getenv("OPENPILOT_FP32_SMALL_TARGETS", "4,6")).split(","))
  targets4_fp32 = targets4 if getenv("OPENPILOT_FP32_SMALL") and "4" in fp32_small_targets else []
  targets6_fp32 = targets6 if getenv("OPENPILOT_FP32_SMALL") and "6" in fp32_small_targets else []
  targets4_half_output = targets4 if getenv("OPENPILOT_HALF_TARGET4_OUTPUT") else []
  # FP16 accumulation materially changes the model output, so keep the hand GEMMs experimental
  # and allow family-by-family sensitivity sweeps.
  selected_fp16 = set(getenv("OPENPILOT_FP16_TARGETS", "").split(","))-{""}
  if selected_fp16:
    targets = targets if "1" in selected_fp16 else []
    targets2 = targets2 if "2" in selected_fp16 else []
    targets3 = targets3 if "3" in selected_fp16 else []
    targets4 = targets4 if "4" in selected_fp16 else []
    targets5 = targets5 if "5" in selected_fp16 else []
    targets6 = targets6 if "6" in selected_fp16 else []
    targets7 = targets7 if "7" in selected_fp16 else []
    targets9 = targets9 if "9" in selected_fp16 else []
  elif not getenv("OPENPILOT_FP16_HAND_GEMM"):
    targets = targets2 = targets3 = targets4 = targets5 = targets6 = targets7 = targets9 = []
  if targets_fp32: targets = []
  if targets3_fp32: targets3 = []
  if targets_split: targets = targets_fp32 = []
  if targets_mixed: targets = targets_fp32 = targets_split = []
  if targets_localb: targets = targets_fp32 = targets_split = targets_mixed = []
  if targets_globalb: targets = targets_fp32 = targets_split = targets_mixed = targets_localb = []
  if targets_quadmap: targets = targets_fp32 = targets_split = targets_mixed = targets_localb = targets_globalb = []
  if targets2_fp32: targets2 = []
  if targets2_lowrank: targets2 = []
  if targets5_fp32: targets5 = []
  if targets5_patch: targets5 = []
  if targets9_fp32: targets9 = []
  if targets4_fp32: targets4 = []
  if targets6_fp32: targets6 = []
  if targets7_native: targets7 = []
  if targets3_native: targets3 = []
  if targets3_split: targets3 = []

  q.M, q.N, q.K, q.K4 = M, N, K, K//4
  dev, partial_hand = Device["QCOM"], bool(getenv("OPENPILOT_PARTIAL_HAND"))
  float_hand_activations = bool(getenv("OPENPILOT_FP16_FLOAT_ACTIVATIONS"))
  hand_activation_dtype = dtypes.float if float_hand_activations else dtypes.half
  hand_ncols = 2 if partial_hand else 1
  # The partial-accumulator shader is larger than the two-column donor. A four-column donor has
  # identical buffer metadata and enough executable space for the injected shader.
  envelope, image_off, image_size, reg_off = get_envelope(dev, q.make_donor_src(4 if partial_hand else max(hand_ncols, 3), 128))
  if partial_hand:
    shader, _ = q.build_4xn_shader(dev, 128, ncols=2, direct=False, k_unroll=4, first_sync_only=True, coord_delay=-1)
    hand_fregs, hand_hregs, hand_global = 8, 48, (2, 8, 1)
  else:
    shader, _ = q.build_4xn_shader(dev, 128, ncols=1, direct=True, compact_acc=True, alu_order="row_col_kk",
                                   k_unroll=4, first_sync_only=False, coord_delay=4)
    hand_fregs, hand_hregs, hand_global = 10, 24, (3, 8, 1)
  hand_lib = inject(envelope, image_off, image_size, reg_off, shader, fregs=hand_fregs, hregs=hand_hregs)
  q.M, q.N, q.K, q.K4 = M, 512, K, K//4
  wide1_env, wide1_io, wide1_sz, wide1_ro = get_envelope(dev, q.make_donor_src(4, 128))
  wide1_libs = {}
  for wide1_ncols in (2, 4):
    wide1_kwargs = dict(ncols=wide1_ncols, direct=True, compact_acc=True, first_sync_only=True,
                        k_unroll=4, coord_delay=-1)
    if wide1_ncols == 4:
      wide1_kwargs.update(stable_bx=True, stable_ay=True, inc_coords=True, persistent_coords=True, b_first=True)
    wide1_shader, _ = q.build_4xn_shader(dev, 128, **wide1_kwargs)
    wide1_libs[wide1_ncols] = inject(wide1_env, wide1_io, wide1_sz, wide1_ro, wide1_shader,
                                     fregs=10, hregs=28)
  q8.K, q8.K4 = K, K//4
  wide1_8env, wide1_8io, wide1_8sz, wide1_8ro = get_envelope(dev, q8.make_donor_src8(4, 128))
  wide1_8shader, _, _, _ = q8.build_8x8_split_a_unroll_shader(
    dev, 128, k_unroll=8, b_coord_delay=0, fast_coords=True,
    prefetch_next_b=True, add256_store_mode="tight")
  wide1_libs[8] = inject(wide1_8env, wide1_8io, wide1_8sz, wide1_8ro, wide1_8shader, fregs=8, hregs=28)
  q.M, q.N, q.K, q.K4 = M, 1024, K, K//4
  fp32_envelope, fp32_image_off, fp32_image_size, fp32_reg_off = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  fp32_shader, fp32_hregs, fp32_fregs, _ = q.build_4x4_fp32_compact_preload_shader(
    dev, 128, coord_delay=getenv("OPENPILOT_FP32_DELAY", 4), sampler_per_texture=True,
    batch_coords=bool(getenv("OPENPILOT_FP32_BATCH_COORDS")), quad_map=bool(getenv("OPENPILOT_FP32_QUAD_HAND")),
    quad_b=bool(getenv("OPENPILOT_FP32_QUAD_B")), quad_b_load_all=bool(getenv("OPENPILOT_FP32_QUAD_B_LOAD_ALL")),
    first_coord_wait_only=bool(getenv("OPENPILOT_FP32_FIRST_WAIT_ONLY")))
  fp32_lib = inject(fp32_envelope, fp32_image_off, fp32_image_size, fp32_reg_off, fp32_shader,
                    fregs=fp32_fregs, hregs=fp32_hregs)
  epi_fp32_lib = dev.compiler.compile(EPILOGUE_FP32)
  q.M, q.N, q.K, q.K4 = 128, 2048, 384, 96
  fp32_envelope3, fp32_image_off3, fp32_image_size3, fp32_reg_off3 = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  fp32_shader3, fp32_hregs3, fp32_fregs3, _ = q.build_4x4_fp32_compact_preload_shader(
    dev, 128, coord_delay=getenv("OPENPILOT_FP32_DELAY", 4), sampler_per_texture=True,
    batch_coords=bool(getenv("OPENPILOT_FP32_BATCH_COORDS")),
    quad_map=bool(getenv("OPENPILOT_FP32_QUAD_HAND")), quad_b=bool(getenv("OPENPILOT_FP32_QUAD_B")),
    quad_b_load_all=bool(getenv("OPENPILOT_FP32_QUAD_B_LOAD_ALL")),
    first_coord_wait_only=bool(getenv("OPENPILOT_FP32_FIRST_WAIT_ONLY")))
  fp32_lib3 = inject(fp32_envelope3, fp32_image_off3, fp32_image_size3, fp32_reg_off3, fp32_shader3,
                     fregs=fp32_fregs3, hregs=fp32_hregs3)
  def build_fp32_wide(k):
    q.M, q.N, q.K, q.K4 = 192, 1024, k, k//4
    env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
    shader, hregs, fregs, _ = q.build_4x8_fp32_low_shader(
      dev, 128, coord_delay=getenv("OPENPILOT_FP32_WIDE_DELAY", -1), sampler_per_texture=True,
      alu_order="kk_col_row", preload_b=True, batch_coords=bool(getenv("OPENPILOT_FP32_WIDE_BATCH", 1)), interleaved_a=True)
    return inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  fp32_wide_libs = {k:build_fp32_wide(k) for k in (384, 768)}
  def build_fp32_wide4(k):
    q.M, q.N, q.K, q.K4 = 192, 1024, k, k//4
    env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
    shader, hregs, fregs, _ = q.build_4x4_fp32_compact_preload_shader(dev, 128, coord_delay=4, sampler_per_texture=True)
    return inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  fp32_wide4_libs = {k:build_fp32_wide4(k) for k in (384, 768)}
  epi2_fp32_lib = dev.compiler.compile(EPILOGUE2_FP32)
  q.M, q.N, q.K, q.K4 = 128, 1024, 384, 96
  fp32_envelope9, fp32_image_off9, fp32_image_size9, fp32_reg_off9 = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  fp32_shader9, fp32_hregs9, fp32_fregs9, _ = q.build_4x4_fp32_compact_preload_shader(
    dev, 128, coord_delay=4, sampler_per_texture=True)
  fp32_lib9 = inject(fp32_envelope9, fp32_image_off9, fp32_image_size9, fp32_reg_off9, fp32_shader9,
                     fregs=fp32_fregs9, hregs=fp32_hregs9)
  q.M, q.N, q.K, q.K4 = 512, 1024, 192, 48
  fp32_envelope_small, fp32_image_off_small, fp32_image_size_small, fp32_reg_off_small = \
    get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  fp32_shader_small, fp32_hregs_small, fp32_fregs_small, _ = q.build_4x4_fp32_compact_preload_shader(
    dev, 128, coord_delay=4, sampler_per_texture=True)
  fp32_lib_small = inject(fp32_envelope_small, fp32_image_off_small, fp32_image_size_small, fp32_reg_off_small,
                          fp32_shader_small, fregs=fp32_fregs_small, hregs=fp32_hregs_small)
  q.M, q.N, q.K, q.K4 = 512, 1024, 192, 48
  fp32_target4_env, fp32_target4_io, fp32_target4_sz, fp32_target4_ro = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  fp32_target4_shader, fp32_target4_hregs, fp32_target4_fregs, _ = q.build_4x8_fp32_low_shader(
    dev, 128, coord_delay=-1, sampler_per_texture=True, alu_order="kk_col_row",
    preload_b=True, batch_coords=True, interleaved_a=False)
  fp32_target4_lib = inject(fp32_target4_env, fp32_target4_io, fp32_target4_sz, fp32_target4_ro,
                            fp32_target4_shader, fregs=fp32_target4_fregs, hregs=fp32_target4_hregs)
  epi_lib = dev.compiler.compile(EPILOGUE)
  q.M, q.N, q.K, q.K4 = 192, 1024, 768, 192
  envelope2, image_off2, image_size2, reg_off2 = get_envelope(dev, q.make_donor_src(4, 128))
  shader2, _ = q.build_4xn_shader(dev, 128, ncols=1, direct=True, compact_acc=True,
                                  first_sync_only=False, k_unroll=8, coord_delay=4, alu_order="row_col_kk")
  hand_lib2 = inject(envelope2, image_off2, image_size2, reg_off2, shader2, fregs=10, hregs=24)
  epi_lib2 = dev.compiler.compile(EPILOGUE2)
  pack_target2_b_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void pack_target2_b(__global uint *O,read_only image2d_t X) {
  int t=get_group_id(0)*128+get_local_id(0),col=t&127,pos=t>>7;
  float4 p0=read_imagef(X,smp,(int2)(pos,col)),p1=read_imagef(X,smp,(int2)(pos+192,col));
  float4 p2=read_imagef(X,smp,(int2)(pos+384,col)),p3=read_imagef(X,smp,(int2)(pos+576,col));
  int o=(pos*512+col)*2;
  vstore2(as_uint2(convert_half4((float4)(p0.x,p1.x,p2.x,p3.x))),0,O+o);
  vstore2(as_uint2(convert_half4((float4)(p0.y,p1.y,p2.y,p3.y))),0,O+o+256);
  vstore2(as_uint2(convert_half4((float4)(p0.z,p1.z,p2.z,p3.z))),0,O+o+512);
  vstore2(as_uint2(convert_half4((float4)(p0.w,p1.w,p2.w,p3.w))),0,O+o+768);
}"""
  pack_target2_b_lib = dev.compiler.compile(pack_target2_b_source)
  pack_target5_b_source = pack_target2_b_source.replace("pos+192", "pos+96").replace(
    "pos+384", "pos+192").replace("pos+576", "pos+288")
  pack_target5_b_lib = dev.compiler.compile(pack_target5_b_source)
  q.M, q.N, q.K, q.K4 = 128, 1536, 384, 96
  envelope3, image_off3, image_size3, reg_off3 = get_envelope(dev, q.make_donor_src(4, 128))
  shader3, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True, stable_bx=True, stable_ay=True,
                                  inc_coords=True, persistent_coords=True, first_sync_only=True, k_unroll=4,
                                  b_first=True, coord_delay=-1, store_row_shift=11, alu_order="kk_col_row",
                                  separate_b_coords=True, reuse_separate_b_y=True,
                                  acc_hr=20, save_output_coords=True)
  hand_lib3 = inject(envelope3, image_off3, image_size3, reg_off3, shader3, fregs=11, hregs=36)
  q.M, q.N, q.K, q.K4 = 128, 1024, 384, 96
  envelope3s, image_off3s, image_size3s, reg_off3s = get_envelope(dev, q.make_donor_src(4, 128))
  shader3s, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True,
                                   stable_bx=True, stable_ay=True, inc_coords=True, persistent_coords=True,
                                   first_sync_only=True, k_unroll=4, b_first=True, coord_delay=-1, alu_order="row_col_kk")
  hand_lib3_split = inject(envelope3s, image_off3s, image_size3s, reg_off3s, shader3s, fregs=10, hregs=28)
  def build_8x8_lib(k:int, thread_store_gx:int=0, store_row_shift:int=10):
    q8.K, q8.K4 = k, k//4
    env, io, sz, ro = get_envelope(dev, q8.make_donor_src8(4, 128))
    shader, hregs, fregs, _ = q8.build_8x8_split_a_unroll_shader(dev, 128, k_unroll=8, b_coord_delay=0,
                                                                 fast_coords=True, prefetch_next_b=True,
                                                                 add256_store_mode="tight",
                                                                 safe_coords=bool(getenv("OPENPILOT_8X8_SAFE_COORDS")),
                                                                 separate_coords=bool(getenv("OPENPILOT_8X8_SEPARATE_COORDS")),
                                                                 thread_store_gx=thread_store_gx, store_row_shift=store_row_shift)
    return inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  need_hand8 = bool(targets2 or targets3 or targets4 or targets5 or targets6 or targets7 or cached_fp16_target3)
  hand_lib8 = {k:build_8x8_lib(k) for k in (192, 384, 768)} if need_hand8 else {}
  hand_lib8_target3 = build_8x8_lib(384, store_row_shift=11) if targets3 or cached_fp16_target3 else None
  hand_lib8_thread = ({(192, gx):build_8x8_lib(192, gx) for gx in (1, 2)}
                      if targets4 and getenv("OPENPILOT_8X8_THREAD_TARGET4") else {})
  hand_lib4 = {}
  thread_target4 = bool(getenv("OPENPILOT_FP16_THREAD_TARGET4"))
  for ncols in ((2, 4) if targets4 else ()):
    q.M, q.N, q.K, q.K4 = 512, 1024, 192, 48
    # The persistent-coordinate 4-column shader is slightly larger than the
    # compiler's 4-column executable.  A 5-column donor has the same argument
    # ABI and provides enough code space; the injected launch still uses the
    # requested ncols geometry below.
    env4, io4, sz4, ro4 = get_envelope(dev, q.make_donor_src(8, 128))
    kwargs = dict(ncols=ncols, direct=True, compact_acc=True, first_sync_only=True, k_unroll=4,
                  coord_delay=-1, alu_order="row_col_kk")
    if ncols == 4:
      if thread_target4:
        # Reusing the compact coordinate registers while texture reads are in flight corrupts
        # arbitrary B blocks. Keep each B sample's coordinate pair distinct and store the tile
        # thread-major; epi40 below converts that private layout directly to the model layout.
        kwargs.update(first_sync_only=False, coord_delay=4, separate_b_coords=True, thread_store=True)
      else: kwargs.update(stable_bx=True, stable_ay=True, inc_coords=True, persistent_coords=True, b_first=True)
    shader4, _ = q.build_4xn_shader(dev, 128, **kwargs)
    hregs4 = (48 + 16*ncols + 3)//4
    hand_lib4[ncols] = inject(env4, io4, sz4, ro4, shader4, fregs=11 if ncols == 4 and thread_target4 else 10 if ncols == 4 else 8, hregs=hregs4)
  hand_lib5 = hand_lib6 = hand_lib7 = hand_lib9 = None
  if targets5:
    q.M, q.N, q.K, q.K4 = 192, 1024, 384, 96
    env5, io5, sz5, ro5 = get_envelope(dev, q.make_donor_src(4, 128))
    shader5, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True, stable_bx=True, stable_ay=True,
                                    inc_coords=True, persistent_coords=True, first_sync_only=True, k_unroll=4,
                                    b_first=True, coord_delay=-1, alu_order="row_col_kk")
    hand_lib5 = inject(env5, io5, sz5, ro5, shader5, fregs=10, hregs=28)
  if targets6:
    q.M, q.N, q.K, q.K4 = 512, 1024, 192, 48
    env6, io6, sz6, ro6 = get_envelope(dev, q.make_donor_src(3, 128))
    shader6, _ = q.build_4xn_shader(dev, 128, ncols=3, direct=True, compact_acc=True, first_sync_only=True,
                                    k_unroll=4, coord_delay=-1, alu_order="row_col_kk")
    hand_lib6 = inject(env6, io6, sz6, ro6, shader6, fregs=8, hregs=24)
  if targets7:
    q.M, q.N, q.K, q.K4 = 128, 1024, 384, 96
    env7, io7, sz7, ro7 = get_envelope(dev, q.make_donor_src(4, 128))
    shader7, _ = q.build_4xn_shader(dev, 128, ncols=1, direct=True, compact_acc=True, first_sync_only=True,
                                    k_unroll=4, coord_delay=-1, alu_order="row_col_kk")
    hand_lib7 = inject(env7, io7, sz7, ro7, shader7, fregs=8, hregs=16)
  if targets9:
    q.M, q.N, q.K, q.K4 = 128, 1024, 384, 96
    env9, io9, sz9, ro9 = get_envelope(dev, q.make_donor_src(3, 128))
    shader9, _ = q.build_4xn_shader(dev, 128, ncols=3, direct=True, compact_acc=True, first_sync_only=True,
                                    k_unroll=4, coord_delay=-1, alu_order="row_col_kk")
    hand_lib9 = inject(env9, io9, sz9, ro9, shader9, fregs=8, hregs=24)

  replacements:dict[UOp, tuple[UOp, ...]] = {}
  indexed_replacements:dict[int, tuple[UOp, ...]] = {}
  if targets7_coord:
    delay = int(target7_coord_delay)
    if delay not in range(6): raise ValueError("OPENPILOT_TARGET7_COORD_DELAY must be 0..5")
    coord_libs = {}
    for call in targets7_coord:
      program, old_lib = call.src[0], call.src[0].src[3].arg
      if old_lib not in coord_libs:
        image_off, image_size = struct.unpack_from("<I", old_lib, 0xc0)[0], struct.unpack_from("<I", old_lib, 0x100)[0]
        lib = bytearray(old_lib)
        for index in (23,26,29,32,35,38,41,44):
          if lib[image_off+index*8:image_off+(index+1)*8] != NOP(rpt=5):
            raise ValueError(f"target7 coordinate wait {index} does not match the compiler donor")
          lib[image_off+index*8:image_off+(index+1)*8] = NOP(rpt=delay)
        coord_libs[old_lib] = bytes(lib)
      refreshed = program.replace(src=program.src[:3]+(program.src[3].replace(arg=coord_libs[old_lib]),))
      replacements[call] = (call.replace(src=(refreshed, *call.src[1:])),)
  if targets7_quad:
    quad_libs = {}
    for call in targets7_quad:
      program, old_lib = call.src[0], call.src[0].src[3].arg
      if old_lib not in quad_libs:
        image_off, image_size = struct.unpack_from("<I", old_lib, 0xc0)[0], struct.unpack_from("<I", old_lib, 0x100)[0]
        reg_off = struct.unpack_from("<I", old_lib, 0x34)[0]
        old = [old_lib[image_off+i:image_off+i+8] for i in range(0, image_size, 8)]
        if len(old) < 133: raise ValueError("target7 quad donor is unexpectedly short")
        out = [MOV_F32("r17.z", "r0.x")] + old[:22]
        loop_start = len(out)
        out += [CMPS_S_EQ("r17.z", 0), NOP(rpt=5), BR(13)] + old[22:34]
        out += [SHL_B("r17.z", "r17.z", 0, jp=True, ss=True, nop=3), MOV_S32("r17.w", 0),
                MOV_F32("r17.w", "r17.w", sy=True), NOP(rpt=5)]
        for reg in ("r7.x", "r4.x", "r1.x", "r0.x"):
          out.append(QUAD_BRCST(reg, reg, "r17.w", typ=3, wrmask=15))
        out += old[34:68]
        out.append(BR(loop_start-len(out)))
        out += old[69:]
        while len(out) > len(old) and out[-1] == NOP(): out.pop()
        if len(out) > len(old): raise ValueError(f"target7 quad shader exceeds donor: {len(out)} > {len(old)}")
        fregs, hregs = struct.unpack_from("<II", old_lib, reg_off+0x14)
        quad_libs[old_lib] = inject(old_lib, image_off, image_size, reg_off, b"".join(out), fregs=fregs, hregs=hregs)
      refreshed = program.replace(src=program.src[:3]+(program.src[3].replace(arg=quad_libs[old_lib]),))
      replacements[call] = (call.replace(src=(refreshed, *call.src[1:])),)
  if targets7_local_cache:
    target7_cache_source = """__attribute__((reqd_work_group_size(4,8,4)))
__kernel void r_36_32_8_4_4_96_4(write_only image2d_t O,read_only image2d_t A,read_only image2d_t B) {
  const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
  int lx=get_local_id(0),ly=get_local_id(1),lz=get_local_id(2),lid=lx+4*(ly+8*lz);
  int ix=get_global_id(0),iy=get_global_id(1),iz=get_global_id(2);
  __local float4 lm[768];
  float4 z0=(float4)(0),z1=(float4)(0),z2=(float4)(0),z3=(float4)(0);
  for(int kb=0;kb<96;kb+=8) {
    for(int p=lid;p<768;p+=128) {
      if(p<256) {
        int r=p&3,q=(p>>2)&7,y=p>>5;
        int k=(get_group_id(1)*8+y)*384+kb+q;
        lm[p]=read_imagef(A,smp,(int2)(k+r*96,0));
      } else {
        int t=p-256,j=t&3,q=(t>>2)&7,x=(t>>5)&3,z=t>>7;
        lm[p]=read_imagef(B,smp,(int2)((get_group_id(0)*4+x)*384+(kb+q)*4+j,get_group_id(2)*4+z));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int q=0;q<8;q++) {
      int bp=256+(((lz*4+lx)*8+q)*4);
      float4 b0=lm[bp],b1=lm[bp+1],b2=lm[bp+2],b3=lm[bp+3];
      int ap=(ly*8+q)*4;
      float4 a=lm[ap]; z0+=a.x*b0+a.y*b1+a.z*b2+a.w*b3;
      a=lm[ap+1]; z1+=a.x*b0+a.y*b1+a.z*b2+a.w*b3;
      a=lm[ap+2]; z2+=a.x*b0+a.y*b1+a.z*b2+a.w*b3;
      a=lm[ap+3]; z3+=a.x*b0+a.y*b1+a.z*b2+a.w*b3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int x=ix+(iy<<3);
  write_imagef(O,(int2)(x,iz),z0);
  write_imagef(O,(int2)(x+8,iz),z1);
  write_imagef(O,(int2)(x+16,iz),z2);
  write_imagef(O,(int2)(x+24,iz),z3);
}"""
    target7_cache_lib = dev.compiler.compile(target7_cache_source)
    for call in targets7_local_cache:
      program = call.src[0]
      info = replace(program.arg, global_size=(2,4,9), local_size=(4,8,4))
      refreshed = program.replace(arg=info, src=program.src[:2]+(
        program.src[2].replace(arg=target7_cache_source), program.src[3].replace(arg=target7_cache_lib)))
      replacements[call] = (call.replace(src=(refreshed, *call.src[1:])),)
  if target7_local := str(getenv("OPENPILOT_TARGET7_LOCAL", "")):
    new_local = tuple(int(x) for x in target7_local.split(","))
    if len(new_local) != 3 or np.prod(new_local) != 128: raise ValueError("OPENPILOT_TARGET7_LOCAL must contain three integers with product 128")
    for call in targets7_local:
      program, old_local = call.src[0], call.src[0].arg.local_size
      new_global = tuple(program.arg.global_size[i]*old_local[i]/new_local[i] for i in range(3))
      refreshed = program.replace(arg=replace(program.arg, global_size=new_global, local_size=new_local))
      replacements[call] = (call.replace(src=(refreshed, *call.src[1:])),)

  selected_half_intermediates = set(getenv("OPENPILOT_HALF_INTERMEDIATE_PRODUCERS", "").split(","))-{""}
  half_intermediate_indices = {int(x) for x in str(getenv("OPENPILOT_HALF_INTERMEDIATE_PRODUCER_INDICES", "")).split(",") if x}
  half_intermediate_producers = [(batch_index, call) for batch_index, call in enumerate(batch)
                                 if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                                 plain_name(call.src[0].arg.name) in selected_half_intermediates]
  for producer_index, (producer_batch_index, producer) in enumerate(half_intermediate_producers):
    if half_intermediate_indices and producer_index not in half_intermediate_indices: continue
    if len(producer.src[0].arg.outs) != 1: continue
    output_index = producer.src[0].arg.outs[0]
    output = producer.src[output_index+1]
    # Captured memory planning reuses both call UOps and output buffers across layers.
    # Scope consumers to this write's lifetime and key replacements by batch position,
    # otherwise selecting one occurrence silently rewrites every structurally equal call.
    next_write = next((i for i in range(producer_batch_index+1, len(batch)) if batch[i].op is Ops.CALL and
                       batch[i].src[0].op is Ops.PROGRAM and output in
                       tuple(batch[i].src[j+1] for j in batch[i].src[0].arg.outs)), len(batch))
    consumers = [(i, call) for i, call in enumerate(batch[producer_batch_index+1:next_write], producer_batch_index+1)
                 if call.op is Ops.CALL and output in call.src[1:]]
    if not consumers: continue
    half_output = UOp.new_buffer("QCOM", output.buffer.size, dtypes.half)
    half_output.buffer.ensure_allocated()
    producer_groups = tuple(tuple((i, dtypes.half if i == output_index else dtype, shape) for i, dtype, shape in group)
                            for group in producer.src[0].arg.aux[0])
    producer_program = producer.src[0].replace(arg=replace(producer.src[0].arg, aux=(producer_groups,)))
    indexed_replacements[producer_batch_index] = (producer.replace(src=(producer_program,) + tuple(
      half_output if i == output_index else x for i, x in enumerate(producer.src[1:]))),)
    for consumer_batch_index, consumer in consumers:
      param_indices = {i for i, x in enumerate(consumer.src[1:]) if x is output}
      consumer_groups = tuple(tuple((i, dtypes.half if i in param_indices else dtype, shape) for i, dtype, shape in group)
                              for group in consumer.src[0].arg.aux[0])
      consumer_program = consumer.src[0].replace(arg=replace(consumer.src[0].arg, aux=(consumer_groups,)))
      indexed_replacements[consumer_batch_index] = (consumer.replace(src=(consumer_program,) + tuple(
        half_output if x is output else x for x in consumer.src[1:])),)

  if getenv("OPENPILOT_STATIC_PACK_TARGET2_A"):
    produced = {call.src[i+1] for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM
                for i in call.src[0].arg.outs}
    for pack_call in (call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                      plain_name(call.src[0].arg.name) == "pack_banked"):
      output, source = pack_call.src[1], pack_call.src[2]
      if source in produced: continue
      source_np = np.asarray(source.buffer.numpy(), dtype=np.float32).reshape(48, 768, 4)
      packed_np = source_np.reshape(48, 192, 4, 4).transpose(0, 3, 1, 2).reshape(192, 192, 4).astype(np.float16)
      packed = UOp.new_buffer("QCOM", packed_np.size, dtypes.half)
      packed.buffer.ensure_allocated()
      packed.buffer.copyin(memoryview(packed_np).cast("B"))
      consumers = [call for call in batch if call is not pack_call and call.op is Ops.CALL and output in call.src[1:]]
      if len(consumers) != 1: raise ValueError(f"pack_banked output has {len(consumers)} consumers")
      consumer = consumers[0]
      replacements[pack_call] = ()
      replacements[consumer] = (consumer.replace(src=(consumer.src[0],) + tuple(packed if x is output else x for x in consumer.src[1:])),)

  pack_banked_cache:dict[tuple[int, int], tuple[str, bytes]] = {}
  def pack_banked_activation(template:UOp, source:UOp, m:int, k:int) -> tuple[UOp, UOp]:
    """Convert four contiguous x-axis row banks into row-major [M,K/4,half4]."""
    source_code, lib = pack_banked_cache.get((m, k), ("", b""))
    if not lib:
      source_code = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void pack_banked(__global uint *O, read_only image2d_t X) {{
  int t=get_global_id(0),row=t/{k//4},k4=t-row*{k//4},y=row>>2,r=row&3;
  float4 p=read_imagef(X,smp,(int2)(r*{k//4}+k4,y));
  vstore2(as_uint2(convert_half4(p)),0,O+t*2);
}}"""
      lib = dev.compiler.compile(source_code)
      pack_banked_cache[(m, k)] = (source_code, lib)
    packed = UOp.new_buffer("QCOM", m*k, dtypes.half)
    packed.buffer.ensure_allocated()
    program = build_program(template, "pack_banked", source_code, lib, (m*k//(4*128), 1, 1), (128, 1, 1),
                            ((dtypes.half, None), (source.dtype, (m//4, k, 4))), (0,), (1,))
    return program.call(packed, source), packed

  def replace_half_weight(call:UOp, program:UOp, src_index:int) -> tuple[UOp, UOp]:
    quantized = np.asarray(call.src[src_index].buffer.numpy(), dtype=np.float16)
    weight = UOp.new_buffer("QCOM", quantized.size, dtypes.half)
    weight.buffer.ensure_allocated()
    weight.buffer.copyin(memoryview(quantized).cast("B"))
    groups = tuple(tuple((i, dtypes.half if i == src_index-1 else dtype, shape) for i, dtype, shape in group)
                   for group in program.arg.aux[0])
    return program.replace(arg=replace(program.arg, aux=(groups,))), weight

  if targets4_half_output:
    pack_consumers = {call.src[2]:call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                      plain_name(call.src[0].arg.name) == "pack_target2_b"}
    for call in targets4_half_output:
      if call.src[1] not in pack_consumers: continue
      pack_call = pack_consumers[call.src[1]]
      half_output = UOp.new_buffer("QCOM", call.src[1].buffer.size, dtypes.half)
      half_output.buffer.ensure_allocated()
      target_groups = tuple(tuple((i, dtypes.half if i == 0 else dtype, shape) for i, dtype, shape in group)
                            for group in call.src[0].arg.aux[0])
      target_program = call.src[0].replace(arg=replace(call.src[0].arg, aux=(target_groups,)))
      pack_groups = tuple(tuple((i, dtypes.half if i == 1 else dtype, shape) for i, dtype, shape in group)
                          for group in pack_call.src[0].arg.aux[0])
      pack_program = pack_call.src[0].replace(arg=replace(pack_call.src[0].arg, aux=(pack_groups,)))
      replacements[call] = (call.replace(src=(target_program, half_output, *call.src[2:])),)
      replacements[pack_call] = (pack_call.replace(src=(pack_program, pack_call.src[1], half_output)),)

  for calls in half_weight_only_calls.values():
    for call in calls:
      program, weight = replace_half_weight(call, call.src[0], 4)
      replacements[call] = (call.replace(src=(program, *call.src[1:4], weight, *call.src[5:])),)

  for call, src_index in extra_half_weight_calls:
    program, weight = replace_half_weight(call, call.src[0], src_index)
    replacements[call] = (call.replace(src=(program, *call.src[1:src_index], weight, *call.src[src_index+1:])),)

  for call in cached_fp32_refresh:
    lib = fp32_lib3 if tuple(call.src[0].arg.global_size) == (12, 8, 1) else fp32_lib
    program = call.src[0]
    refreshed = program.replace(src=program.src[:3] + (program.src[3].replace(arg=lib),))
    replacements[call] = (call.replace(src=(refreshed,) + call.src[1:]),)

  if cached_fp16_target3:
    # The exact cached graph already contains transposed half weights. Replace
    # its FP32-accumulating GEMM + epilogue pair with the checked 4x16 FP16 hand
    # kernel, leaving every other program on the default THREAD64 dispatch.
    cached3_epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void cached_epi3(__global half *O,__global float *S,__global float *B,__global half *C) {
  int t=get_global_id(0),row=t/384,col=t-row*384,y=row>>2,r=row&3;
  int o=((y*1536+r*384+col)*4);
  float4 z=convert_float4(vload4(0,C+row*2048+col*4));
  z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
}"""
    cached3_epi_lib = dev.compiler.compile(cached3_epi_source)
    batch_pos = {call:i for i, call in enumerate(batch)}
    for call in cached_fp16_target3:
      idx = batch_pos[call]
      if idx+1 >= len(batch): raise ValueError("cached target3 GEMM has no epilogue")
      old_epi = batch[idx+1]
      if old_epi.op is not Ops.CALL or plain_name(old_epi.src[0].arg.name) != "epi3_fp32":
        raise ValueError(f"cached target3 GEMM is followed by {plain_name(old_epi.src[0].arg.name)}")
      temporary = UOp.new_buffer("QCOM", 128*2048, dtypes.half)
      temporary.buffer.ensure_allocated()
      use_8x8 = bool(getenv("OPENPILOT_8X8_TARGET3")) or bool(getenv("OPENPILOT_8X8_ALL"))
      hand = build_program(call.src[0], "gemm_h", "checked FP16 hand GEMM", hand_lib8_target3 if use_8x8 else hand_lib3,
                           (6, 4, 1) if use_8x8 else (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (128, 96, 4)), (dtypes.half, (384, 384, 4)),
                            (dtypes.half, (128*2048,))), (2,), (0, 1))
      epi = build_program(old_epi.src[0], "cached_epi3", cached3_epi_source, cached3_epi_lib,
                          (384, 1, 1), (128, 1, 1),
                          ((dtypes.half, (128*1536,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                           (dtypes.half, (128*2048,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[1], call.src[2], temporary),)
      replacements[old_epi] = (epi.call(old_epi.src[1], old_epi.src[2], old_epi.src[3], temporary),)

  if cached_fp16_target1:
    # The cached target-1 path already has row-major half activations and weights.
    # Pad its 384 output columns to the 512-column 4x16 kernel envelope, then keep
    # the original residual epilogue while reading the temporary at half precision.
    cached1_epi_lib = dev.compiler.compile(EPILOGUE)
    batch_pos = {call:i for i, call in enumerate(batch)}
    for call in cached_fp16_target1:
      idx = batch_pos[call]
      if idx+1 >= len(batch): raise ValueError("cached target1 GEMM has no epilogue")
      old_epi = batch[idx+1]
      if old_epi.op is not Ops.CALL or plain_name(old_epi.src[0].arg.name) != "epi_fp32":
        raise ValueError(f"cached target1 GEMM is followed by {plain_name(old_epi.src[0].arg.name)}")
      source_weight = np.asarray(call.src[2].buffer.numpy(), dtype=np.float16).reshape(1536, 96, 4)
      padded_weight = np.zeros((1536, 128, 4), dtype=np.float16)
      padded_weight[:, :96] = source_weight
      weight = UOp.new_buffer("QCOM", padded_weight.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(padded_weight).cast("B"))
      temporary = UOp.new_buffer("QCOM", 128*1024, dtypes.half)
      temporary.buffer.ensure_allocated()
      hand = build_program(call.src[0], "gemm_h", "checked 4x16 FP16 target1 GEMM", wide1_libs[4],
                           (1, 8, 1), (128, 1, 1),
                           ((dtypes.half, (128, 384, 4)), (dtypes.half, (1536, 128, 4)),
                            (dtypes.half, (128*1024,))), (2,), (0, 1))
      epi = build_program(old_epi.src[0], "epi", EPILOGUE, cached1_epi_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.half, (128*1024,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[1], weight, temporary),)
      replacements[old_epi] = (epi.call(old_epi.src[1], old_epi.src[2], old_epi.src[3], temporary),)

  cached_epi3_rows = getenv("OPENPILOT_CACHED_EPI3_ROWS", 4 if getenv("OPENPILOT_CACHED_EPI3_ROWS4") else 0)
  if cached_epi3_rows:
    if cached_epi3_rows not in (1, 2, 4, 8, 16): raise ValueError("OPENPILOT_CACHED_EPI3_ROWS must divide 16")
    cached_epi3_rows4_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi3_fp32(__global half *O,__global float *S,__global float *B,__global float *C) {{
  int t=get_global_id(0),y=t/384,col=t-y*384;
  for(int r=0;r<{cached_epi3_rows};r++) {{
    int row=y*{cached_epi3_rows}+r,oy=row>>2,rr=row&3,o=(oy*1536+rr*384+col)*4;
    float4 z=vload4(0,C+row*2048+col*4); z=select((float4)(0),z,isgreater(z,(float4)(0)));
    vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
  }}
}}"""
    cached_epi3_rows4_lib = dev.compiler.compile(cached_epi3_rows4_source)
    for call in batch:
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != "epi3_fp32": continue
      program = call.src[0]
      info = replace(program.arg, global_size=(384//cached_epi3_rows, 1, 1))
      refreshed = program.replace(arg=info, src=program.src[:2] +
                                  (program.src[2].replace(arg=cached_epi3_rows4_source),
                                   program.src[3].replace(arg=cached_epi3_rows4_lib)))
      replacements[call] = (call.replace(src=(refreshed,) + call.src[1:]),)

  cached_epi1_rows = getenv("OPENPILOT_CACHED_EPI1_ROWS", 0)
  if cached_epi1_rows:
    if cached_epi1_rows not in (1, 2, 4, 8, 16): raise ValueError("OPENPILOT_CACHED_EPI1_ROWS must divide 16")
    cached_epi1_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi_fp32(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global float *C) {{
  int t=get_global_id(0),rb=t/96,col=t-rb*96; float4 scale=read_imagef(S,smp,(int2)(col,0));
  for(int rr=0;rr<{cached_epi1_rows};rr++) {{
    int row=rb*{cached_epi1_rows}+rr,pos=row*96+col; float4 v=vload4(0,C+row*1024+col*4);
    write_imagef(O,(int2)(pos,0),read_imagef(X,smp,(int2)(pos,0))*scale+v);
  }}
}}"""
    cached_epi1_lib = dev.compiler.compile(cached_epi1_source)
    for call in batch:
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != "epi_fp32": continue
      program = call.src[0]
      info = replace(program.arg, global_size=(96//cached_epi1_rows, 1, 1))
      refreshed = program.replace(arg=info, src=program.src[:2] +
                                  (program.src[2].replace(arg=cached_epi1_source), program.src[3].replace(arg=cached_epi1_lib)))
      replacements[call] = (call.replace(src=(refreshed,) + call.src[1:]),)

  auto_rpt_cache = {}
  for calls in auto_rpt_families.values():
    for call in calls:
      program, lib = call.src[0], bytearray(call.src[0].src[3].arg)
      image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
      image = bytes(lib[image_off:image_off+image_size])
      patched = auto_rpt_cache.setdefault(image, patch_auto_rpt_f32(image))
      lib[image_off:image_off+image_size] = patched
      patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
      replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  for target5_index, call in enumerate(targets5_patch):
    program = call.src[0]
    lib = bytearray(program.src[3].arg)
    image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
    lib[image_off:image_off+image_size] = patch_openpilot_target5_rpt3(lib[image_off:image_off+image_size])
    patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
    if should_half_patch_weight("5", target5_index):
      patched_program, weight = replace_half_weight(call, patched_program, 3)
      replacements[call] = (call.replace(src=(patched_program, call.src[1], call.src[2], weight, *call.src[4:])),)
    else: replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  for target2_index, call in enumerate(targets2_patch):
    program = call.src[0]
    lib = bytearray(program.src[3].arg)
    image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
    lib[image_off:image_off+image_size] = patch_openpilot_target2_rpt3(lib[image_off:image_off+image_size])
    patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
    if should_half_patch_weight("2", target2_index):
      patched_program, weight = replace_half_weight(call, patched_program, 3)
      replacements[call] = (call.replace(src=(patched_program, call.src[1], call.src[2], weight, *call.src[4:])),)
    else: replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  for target7_index, call in enumerate(targets7_patch):
    program = call.src[0]
    lib = bytearray(program.src[3].arg)
    image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
    lib[image_off:image_off+image_size] = patch_openpilot_target7_rpt3(lib[image_off:image_off+image_size])
    patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
    if target7_index in target7_half_weights:
      quantized = np.asarray(call.src[3].buffer.numpy(), dtype=np.float16)
      weight = UOp.new_buffer("QCOM", quantized.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(quantized).cast("B"))
      info = replace(patched_program.arg, aux=aux((dtypes.float, (36, 1024, 4)), (dtypes.half, (1, 12288, 4)),
                                                   (dtypes.half, (36, 3072, 4))))
      patched_program = patched_program.replace(arg=info)
      replacements[call] = (call.replace(src=(patched_program, call.src[1], call.src[2], weight)),)
    else: replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  for call in targets7_half_only:
    program, weight = replace_half_weight(call, call.src[0], 3)
    replacements[call] = (call.replace(src=(program, call.src[1], call.src[2], weight)),)

  for target9_index, call in enumerate(targets9_patch):
    program = call.src[0]
    lib = bytearray(program.src[3].arg)
    image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
    lib[image_off:image_off+image_size] = patch_openpilot_target9_rpt3(lib[image_off:image_off+image_size])
    patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
    if should_half_patch_weight("9", target9_index):
      patched_program, weight = replace_half_weight(call, patched_program, 5)
      replacements[call] = (call.replace(src=(patched_program, *call.src[1:5], weight, *call.src[6:])),)
    else: replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  family_indices = {"4":0, "6":0}
  for call in targets46_patch:
    program = call.src[0]
    lib = bytearray(program.src[3].arg)
    image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
    lib[image_off:image_off+image_size] = patch_openpilot_target46_rpt3(lib[image_off:image_off+image_size])
    patched_program = program.replace(src=program.src[:3] + (program.src[3].replace(arg=bytes(lib)),))
    family = "4" if plain_name(program.arg.name) == TARGET4 else "6"
    family_index = family_indices[family]
    family_indices[family] += 1
    if should_half_patch_weight(family, family_index):
      patched_program, weight = replace_half_weight(call, patched_program, 4)
      replacements[call] = (call.replace(src=(patched_program, *call.src[1:4], weight, *call.src[5:])),)
    else: replacements[call] = (call.replace(src=(patched_program,) + call.src[1:]),)

  if cached_gemm1_mixed:
    if 384 % cached_gemm1_mixed_k: raise ValueError("OPENPILOT_PATCH_GEMM1_MIXEDK must divide 384")
    cached_mixed_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_h(read_only image2d_t A,read_only image2d_t B,__global float *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid;
  float4 t0=(float4)(0),t1=(float4)(0),t2=(float4)(0),t3=(float4)(0);
  for(int kb=0;kb<384;kb+={cached_gemm1_mixed_k}) {{
    half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
    for(int k4=kb;k4<kb+{cached_gemm1_mixed_k};k4++) {{
      half4 a0=read_imageh(A,smp,(int2)(k4,row)),a1=read_imageh(A,smp,(int2)(k4,row+1));
      half4 a2=read_imageh(A,smp,(int2)(k4,row+2)),a3=read_imageh(A,smp,(int2)(k4,row+3));
      half4 b0=read_imageh(B,smp,(int2)(col4,k4*4)),b1=read_imageh(B,smp,(int2)(col4,k4*4+1));
      half4 b2=read_imageh(B,smp,(int2)(col4,k4*4+2)),b3=read_imageh(B,smp,(int2)(col4,k4*4+3));
      r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
      r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
      r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
      r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
    }}
    t0+=convert_float4(r0); t1+=convert_float4(r1); t2+=convert_float4(r2); t3+=convert_float4(r3);
  }}
  vstore4(t0,0,C+row*1024+col4*4); vstore4(t1,0,C+(row+1)*1024+col4*4);
  vstore4(t2,0,C+(row+2)*1024+col4*4); vstore4(t3,0,C+(row+3)*1024+col4*4);
}}"""
    cached_mixed_lib = dev.compiler.compile(cached_mixed_source)
    for call in cached_gemm1_mixed:
      program = call.src[0]
      mixed_program = program.replace(src=program.src[:2] + (program.src[2].replace(arg=cached_mixed_source),
                                                             program.src[3].replace(arg=cached_mixed_lib)))
      replacements[call] = (call.replace(src=(mixed_program,) + call.src[1:]),)

  if cached_gemm1_split:
    if 384 % cached_gemm1_split_k: raise ValueError("OPENPILOT_PATCH_GEMM1_SPLITK must divide 384")
    chunk4 = 384 // cached_gemm1_split_k
    old_dims = q.M, q.N, q.K, q.K4
    q.M, q.N, q.K, q.K4 = 128, 1024, 1536, 384
    # The unrolled split shader is larger than the three-column donor image.
    split_env, split_io, split_sz, split_ro = get_envelope(dev, q.make_donor_src(4, 128))
    split_libs = []
    for split in range(cached_gemm1_split_k):
      split_shader, _ = q.build_4xn_shader(dev, 128, ncols=1, direct=True, compact_acc=True,
        alu_order="row_col_kk", k_unroll=4 if chunk4 % 4 == 0 else 2 if chunk4 % 2 == 0 else 1,
        first_sync_only=False, coord_delay=4, k_start=split*chunk4, k_count=chunk4, store_row_base=split*128)
      if len(split_shader) > split_sz: raise RuntimeError(f"cached split shader {len(split_shader)} exceeds donor image {split_sz}")
      split_libs.append(inject(split_env, split_io, split_sz, split_ro, split_shader, fregs=10, hregs=24))
    q.M, q.N, q.K, q.K4 = old_dims
    reduce_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_split_reduce(__global float *C,__global half *S) {{
  int t=get_global_id(0),row=t/96,col=t-row*96; float4 v=(float4)(0);
  for(int s=0;s<{cached_gemm1_split_k};s++) v+=convert_float4(vload4(0,S+(s*128+row)*1024+col*4));
  vstore4(v,0,C+row*1024+col*4);
}}"""
    reduce_lib = dev.compiler.compile(reduce_source)
    scratch = UOp.new_buffer("QCOM", cached_gemm1_split_k*128*1024, dtypes.half)
    scratch.buffer.ensure_allocated()
    template = cached_gemm1_split[0].src[0]
    # The injected donor ELF exports gemm_h; ProgramInfo must retain that symbol name.
    hands = [build_program(template, "gemm_h", "hand_split", lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (128, 384, 4)), (dtypes.half, (1536, 96, 4)),
                            (dtypes.half, (cached_gemm1_split_k*128*1024,))), (2,), (0, 1)) for lib in split_libs]
    reducer = build_program(template, "gemm_split_reduce", reduce_source, reduce_lib, (96, 1, 1), (128, 1, 1),
                            ((dtypes.float, (128*1024,)),
                             (dtypes.half, (cached_gemm1_split_k*128*1024,))), (0,), (1,))
    for call in cached_gemm1_split:
      replacements[call] = tuple(hand.call(call.src[1], call.src[2], scratch) for hand in hands) + \
                           (reducer.call(call.src[3], scratch),)

  if quant_gemms:
    if not 2 <= quant_gemm_bits <= 16: raise ValueError("OPENPILOT_QUANT_GEMM_BITS must be in 2..16")
    qmax = (1 << (quant_gemm_bits-1)) - 1
    qdq_cache = {}
    for quant_idx, call in enumerate(quant_gemms):
      gsx = int(call.src[0].arg.global_size[0])
      shape = (384, 1536) if gsx == 12 else (1536, 384)
      matrix = np.asarray(call.src[2].buffer.numpy(), dtype=np.float32).reshape(shape)
      block = quant_gemm_block or shape[0]
      if shape[0] % block: raise ValueError(f"quant block {block} must divide K={shape[0]}")
      blocked = matrix.reshape(shape[0]//block, block, shape[1])
      scale = np.max(np.abs(blocked), axis=1, keepdims=True) / qmax
      scale[scale == 0] = 1
      dequantized = (np.clip(np.rint(blocked/scale), -qmax, qmax)*scale).reshape(shape).astype(np.float16)
      # Pickled BUFFER slots do not advance UOp.unique_num. Use an explicit
      # disjoint namespace so a new constant cannot alias an existing weight.
      weight = UOp.new_buffer("QCOM", dequantized.size, dtypes.half, num=-(1_000_000+quant_idx))
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(dequantized).cast("B"))
      quant_call = call.replace(src=call.src[:2] + (weight,) + call.src[3:])
      if quant_gemm_activations:
        k = shape[0]
        if block % 4: raise ValueError("activation QDQ quant block must be divisible by 4")
        block4 = block//4
        unsigned_activation = gsx == 3 and not quant_gemm_affine
        aqmax, aqmin = (255, 0) if unsigned_activation else (127, -127)
        if quant_gemm_affine:
          quant_body = f"""float lo=INFINITY,hi=-INFINITY;
  for(int i=0;i<{block4};i++) {{ float4 v=read_imagef(A,smp,(int2)(x+i,row));
    lo=fmin(lo,fmin(fmin(v.x,v.y),fmin(v.z,v.w))); hi=fmax(hi,fmax(fmax(v.x,v.y),fmax(v.z,v.w))); }}
  float s=hi==lo?1.0f:(hi-lo)/255.0f,z=clamp(rint(-lo/s),0.0f,255.0f);
  for(int i=0;i<{block4};i++) {{ float4 v=read_imagef(A,smp,(int2)(x+i,row));
    float4 q=clamp(rint(v/(float4)(s))+(float4)(z),(float4)(0.0f),(float4)(255.0f));
    write_imageh(O,(int2)(x+i,row),convert_half4((q-(float4)(z))*(float4)(s))); }}"""
        else:
          quant_body = f"""float m=0.0f;
  for(int i=0;i<{block4};i++) {{ float4 v=fabs(read_imagef(A,smp,(int2)(x+i,row)));
    m=fmax(m,fmax(fmax(v.x,v.y),fmax(v.z,v.w))); }}
  float s=m==0.0f?1.0f:m/{float(aqmax):.1f}f;
  for(int i=0;i<{block4};i++) {{ float4 v=read_imagef(A,smp,(int2)(x+i,row));
    v=clamp(rint(v/(float4)(s)),(float4)({float(aqmin):.1f}f),(float4)({float(aqmax):.1f}f))*(float4)(s);
    write_imageh(O,(int2)(x+i,row),convert_half4(v)); }}"""
        qdq_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void qdq_block(write_only image2d_t O,read_only image2d_t A) {{
  int t=get_global_id(0),b=t%{k//block},row=t/{k//block},x=b*{block4}; {quant_body}
}}"""
        qdq_lib = qdq_cache.setdefault((k, block, unsigned_activation, quant_gemm_affine), dev.compiler.compile(qdq_source))
        image_shape = (128, k//4, 4)
        qdq = build_program(call.src[0], "qdq_block", qdq_source, qdq_lib, (k//block, 1, 1), (128, 1, 1),
                            ((dtypes.half, image_shape), (dtypes.half, image_shape)), (0,), (1,))
        activation = UOp.new_buffer("QCOM", 128*k, dtypes.half, num=-(2_000_000+quant_idx))
        activation.buffer.ensure_allocated()
        quant_call = quant_call.replace(src=quant_call.src[:1] + (activation,) + quant_call.src[2:])
        replacements[call] = (qdq.call(activation, call.src[1]), quant_call)
      else: replacements[call] = (quant_call,)

  for call in targets7_native:
    program = call.src[0]
    source = enable_fp16(program.src[2].arg.replace("float buf0[16]", "half buf0[16]"))
    source = source.replace("float4 val", "half4 val").replace("read_imagef(", "read_imageh(")
    native = program.replace(src=program.src[:2] + (program.src[2].replace(arg=source),
                                                    program.src[3].replace(arg=dev.compiler.compile(source))))
    replacements[call] = (call.replace(src=(native,) + call.src[1:]),)

  for call in targets3_native:
    program = call.src[0]
    source = enable_fp16(program.src[2].arg.replace("float buf0[16]", "half buf0[16]"))
    source = source.replace("float4 val", "half4 val").replace("read_imagef(", "read_imageh(")
    native = program.replace(src=program.src[:2] + (program.src[2].replace(arg=source),
                                                    program.src[3].replace(arg=dev.compiler.compile(source))))
    replacements[call] = (call.replace(src=(native,) + call.src[1:]),)

  for calls in native_families.values():
    for call in calls:
      program = call.src[0]
      source = enable_fp16(program.src[2].arg.replace("float buf0[16]", "half buf0[16]"))
      # val0..val7 are the two matrix operands.  Later val temporaries belong to the
      # fused residual/epilogue and must remain float to avoid extra model drift.
      for i in range(8):
        source = source.replace(f"float4 val{i} = read_imagef(", f"half4 val{i} = read_imageh(")
      native_lib = dev.compiler.compile(source)
      if getenv("OPENPILOT_FP16_NATIVE_RPT3"):
        patched_lib = bytearray(native_lib)
        image_off, image_size = struct.unpack_from("<I", patched_lib, 0xc0)[0], struct.unpack_from("<I", patched_lib, 0x100)[0]
        patched_lib[image_off:image_off+image_size] = patch_openpilot_target46_f16_rpt3(
          patched_lib[image_off:image_off+image_size])
        native_lib = bytes(patched_lib)
      native = program.replace(src=program.src[:2] + (program.src[2].replace(arg=source),
                                                      program.src[3].replace(arg=native_lib)))
      replacements[call] = (call.replace(src=(native,) + call.src[1:]),)

  if targets3_split:
    epi3_split_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(8,32,1)))
__kernel void epi3_split(write_only image2d_t O,__global float *S,__global float *B,
                         __global half *C0,__global half *C1) {
  int col=get_global_id(0),y=get_global_id(1),row=y*4,lc=col<256?col:col-256;
  __global half *C=col<256?C0:C1;
  for(int block=0;block<4;block++) {
    int n=lc*4+block;
    float4 z=(float4)(C[(row+0)*1024+n],C[(row+1)*1024+n],C[(row+2)*1024+n],C[(row+3)*1024+n]);
    z=select((float4)(0),z,isgreater(z,(float4)(0)));
    write_imagef(O,(int2)(col+block*384,y),(float4)(*S)*z*z+(float4)(*B));
  }
}"""
    epi3_split_lib = dev.compiler.compile(epi3_split_source)
    for call in targets3_split:
      original = np.array(call.src[4].buffer.numpy(), copy=False).reshape(384, 384, 4).transpose(1, 0, 2)
      weights, temporaries, hand_calls = [], [], []
      for chunk, (start, width, gx) in enumerate(((0, 256, 2), (256, 128, 1))):
        weight = UOp.new_buffer("QCOM", 384*width*4, dtypes.half)
        weight.buffer.ensure_allocated()
        weight.buffer.copyin(memoryview(original[:, start:start+width].copy()).cast("B"))
        temporary = UOp.new_buffer("QCOM", 128*1024, dtypes.half)
        temporary.buffer.ensure_allocated()
        hand = build_program(call.src[0], "gemm_h", "hand3_split", hand_lib3_split,
                             (gx, 8, 1), (128, 1, 1),
                             ((dtypes.half, (128, 96, 4)), (dtypes.half, (384, width, 4)),
                              (dtypes.half, (128*1024,))), (2,), (0, 1))
        weights.append(weight); temporaries.append(temporary)
        hand_calls.append(hand.call(call.src[3], weight, temporary))
      epi = build_program(call.src[0], "epi3_split", epi3_split_source, epi3_split_lib,
                          (48, 1, 1), (8, 32, 1),
                          ((dtypes.half, (32, 1536, 4)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                           (dtypes.half, (128*1024,)), (dtypes.half, (128*1024,))), (0,), (1, 2, 3, 4))
      replacements[call] = tuple(hand_calls)+(epi.call(call.src[1], call.src[2], call.src[5], *temporaries),)

  if getenv("OPENPILOT_DW_PAIR_Y") or getenv("OPENPILOT_DW32_PAIR_Y"):
    dw_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(4,4,8)))
__kernel void dw7_pair_y(write_only image2d_t O,read_only image2d_t X,read_only image2d_t W) {
  const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
  int c=get_global_id(0),gx=get_global_id(1),py=get_global_id(2),base=c+gx*384,y0=py*2;
  float4 a0=(float4)(0),a1=(float4)(0),a2=(float4)(0),a3=(float4)(0);
  float4 b0=(float4)(0),b1=(float4)(0),b2=(float4)(0),b3=(float4)(0);
  for(int ky=0;ky<7;ky++) {
    int wy=c*56+ky*7,iy=y0+ky-3;
    float4 w0=read_imagef(W,smp,(int2)(wy,0)),w1=read_imagef(W,smp,(int2)(wy+1,0));
    float4 w2=read_imagef(W,smp,(int2)(wy+2,0)),w3=read_imagef(W,smp,(int2)(wy+3,0));
    float4 w4=read_imagef(W,smp,(int2)(wy+4,0)),w5=read_imagef(W,smp,(int2)(wy+5,0));
    float4 w6=read_imagef(W,smp,(int2)(wy+6,0));
    float4 x0=read_imagef(X,smp,(int2)(base-288,iy)),x1=read_imagef(X,smp,(int2)(base-192,iy));
    float4 x2=read_imagef(X,smp,(int2)(base-96,iy)),x3=read_imagef(X,smp,(int2)(base,iy));
    float4 x4=read_imagef(X,smp,(int2)(base+96,iy)),x5=read_imagef(X,smp,(int2)(base+192,iy));
    float4 x6=read_imagef(X,smp,(int2)(base+288,iy)),x7=read_imagef(X,smp,(int2)(base+384,iy));
    float4 x8=read_imagef(X,smp,(int2)(base+480,iy)),x9=read_imagef(X,smp,(int2)(base+576,iy));
    a0+=x0*w0+x1*w1+x2*w2+x3*w3+x4*w4+x5*w5+x6*w6;
    a1+=x1*w0+x2*w1+x3*w2+x4*w3+x5*w4+x6*w5+x7*w6;
    a2+=x2*w0+x3*w1+x4*w2+x5*w3+x6*w4+x7*w5+x8*w6;
    a3+=x3*w0+x4*w1+x5*w2+x6*w3+x7*w4+x8*w5+x9*w6;
    x0=read_imagef(X,smp,(int2)(base-288,iy+1)); x1=read_imagef(X,smp,(int2)(base-192,iy+1));
    x2=read_imagef(X,smp,(int2)(base-96,iy+1)); x3=read_imagef(X,smp,(int2)(base,iy+1));
    x4=read_imagef(X,smp,(int2)(base+96,iy+1)); x5=read_imagef(X,smp,(int2)(base+192,iy+1));
    x6=read_imagef(X,smp,(int2)(base+288,iy+1)); x7=read_imagef(X,smp,(int2)(base+384,iy+1));
    x8=read_imagef(X,smp,(int2)(base+480,iy+1)); x9=read_imagef(X,smp,(int2)(base+576,iy+1));
    b0+=x0*w0+x1*w1+x2*w2+x3*w3+x4*w4+x5*w5+x6*w6;
    b1+=x1*w0+x2*w1+x3*w2+x4*w3+x5*w4+x6*w5+x7*w6;
    b2+=x2*w0+x3*w1+x4*w2+x5*w3+x6*w4+x7*w5+x8*w6;
    b3+=x3*w0+x4*w1+x5*w2+x6*w3+x7*w4+x8*w5+x9*w6;
  }
  write_imagef(O,(int2)(base,y0),a0); write_imagef(O,(int2)(base+96,y0),a1);
  write_imagef(O,(int2)(base+192,y0),a2); write_imagef(O,(int2)(base+288,y0),a3);
  write_imagef(O,(int2)(base,y0+1),b0); write_imagef(O,(int2)(base+96,y0+1),b1);
  write_imagef(O,(int2)(base+192,y0+1),b2); write_imagef(O,(int2)(base+288,y0+1),b3);
}"""
    dw_lib = dev.compiler.compile(dw_source)
    for call in targets11:
      dw = build_program(call.src[0], "dw7_pair_y", dw_source, dw_lib, (24, 2, 1), (4, 4, 8),
                         ((dtypes.float, (16, 3072, 4)), (dtypes.float, (16, 3072, 4)),
                          (dtypes.float, (1, 5376, 4))), (0,), (1, 2))
      replacements[call] = (dw.call(call.src[1], call.src[2], call.src[3]),)
  if getenv("OPENPILOT_DW32_PAIR_Y"):
    # Same 7x7 depthwise contraction as dw7_pair_y, with 48 channel vectors,
    # 16 x-groups, and 32 output rows. Pair adjacent rows to reuse all weights.
    dw32_source = dw_source.replace("dw7_pair_y", "dw7_pair_y32").replace("base=c+gx*384", "base=c+gx*192")
    offsets = {"base-288":"base-144", "base-192":"base-96", "base-96":"base-48",
               "base+96":"base+48", "base+192":"base+96", "base+288":"base+144",
               "base+384":"base+192", "base+480":"base+240", "base+576":"base+288"}
    for old in offsets: dw32_source = dw32_source.replace(old, f"@{old}@")
    for old, new in offsets.items(): dw32_source = dw32_source.replace(f"@{old}@", new)
    dw32_lib = dev.compiler.compile(dw32_source)
    for call in targets12:
      dw32 = build_program(call.src[0], "dw7_pair_y32", dw32_source, dw32_lib, (12, 4, 2), (4, 4, 8),
                           ((dtypes.float, (32, 3072, 4)), (dtypes.float, (32, 3072, 4)),
                            (dtypes.float, (1, 2688, 4))), (0,), (1, 2))
      replacements[call] = (dw32.call(call.src[1], call.src[2], call.src[3]),)
  for call in targets:
    template = call.src[0]
    transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
    epi = build_program(template, "epi", EPILOGUE, epi_lib, (96, 1, 1), (128, 1, 1),
                        ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                         (dtypes.float, (1, 96, 4)), (dtypes.half, (M*1024,))), (0,), (1, 2, 3))
    if getenv("OPENPILOT_TERNARY"):
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, transposed, M, N, K)
      replacements[call] = ((epi.call(call.src[1], call.src[2], call.src[3], temporary),) if sparse is None else
                            (sparse.call(call.src[4], offbuf, idxbuf, valbuf, temporary),
                             epi.call(call.src[1], call.src[2], call.src[3], temporary)))
    else:
      wide1_ncols = getenv("OPENPILOT_FP16_WIDE1", 0)
      use_wide1 = wide1_ncols in (2, 4, 8)
      if use_wide1:
        padded = np.zeros((K, 128, 4), dtype=np.float16)
        padded[:, :N//4] = transposed
        transposed = padded
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", M*1024, dtypes.half)
      temporary.buffer.ensure_allocated()
      wide1_global = (2, 4, 1) if wide1_ncols == 8 else (4//wide1_ncols, 8, 1)
      hand = build_program(template, "gemm_h", "hand", wide1_libs[wide1_ncols] if use_wide1 else hand_lib,
                           wide1_global if use_wide1 else hand_global, (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, 128 if use_wide1 else N//4, 4)),
                            (dtypes.half, (M*1024,))), (2,), (0, 1))
      replacements[call] = (hand.call(call.src[4], weight, temporary),
                            epi.call(call.src[1], call.src[2], call.src[3], temporary))

  for call in targets_fp32:
    template = call.src[0]
    float_weight = bool(getenv("OPENPILOT_FP32_HAND_FLOAT_WEIGHT"))
    weight_dtype = dtypes.float if float_weight else dtypes.half
    transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy() \
      .astype(np.float32 if float_weight else np.float16)
    weight = UOp.new_buffer("QCOM", transposed.size, weight_dtype)
    weight.buffer.ensure_allocated()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", M*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    hand = build_program(template, "gemm_h", "hand_fp32", fp32_lib, (3, 8, 1), (128, 1, 1),
                         ((dtypes.half, (M, K//4, 4)), (weight_dtype, (K, N//4, 4)),
                          (dtypes.float, (M*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                        ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                         (dtypes.float, (1, 96, 4)), (dtypes.float, (M*1024,))), (0,), (1, 2, 3))
    replacements[call] = (hand.call(call.src[4], weight, temporary),
                          epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets_split:
    assert (K//4) % split_k == 0
    chunk4 = (K//4)//split_k
    hand_split_libs = []
    if getenv("OPENPILOT_HAND_SPLITK"):
      q.M, q.N, q.K, q.K4 = M, 1024, K, K//4
      split_env, split_io, split_sz, split_ro = get_envelope(dev, q.make_donor_src(3, 128))
      split_unroll = 4 if chunk4 % 4 == 0 else 2 if chunk4 % 2 == 0 else 1
      for split in range(split_k):
        split_shader, _ = q.build_4xn_shader(dev, 128, ncols=1, direct=True, compact_acc=True,
          alu_order="row_col_kk", k_unroll=split_unroll, first_sync_only=False, coord_delay=4,
          k_start=split*chunk4, k_count=chunk4, store_row_base=split*M)
        hand_split_libs.append(inject(split_env, split_io, split_sz, split_ro, split_shader, fregs=10, hregs=24))
    split_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_split(read_only image2d_t A,read_only image2d_t B,__global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31,split=get_group_id(2);
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid,k0=split*{chunk4};
  half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
  for(int k4=k0;k4<k0+{chunk4};k4++) {{
    half4 a0=read_imageh(A,smp,(int2)(k4,row)),a1=read_imageh(A,smp,(int2)(k4,row+1));
    half4 a2=read_imageh(A,smp,(int2)(k4,row+2)),a3=read_imageh(A,smp,(int2)(k4,row+3));
    half4 b0=read_imageh(B,smp,(int2)(col4,k4*4)),b1=read_imageh(B,smp,(int2)(col4,k4*4+1));
    half4 b2=read_imageh(B,smp,(int2)(col4,k4*4+2)),b3=read_imageh(B,smp,(int2)(col4,k4*4+3));
    r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
    r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
    r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
    r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
  }}
  int base=split*{M}*1024; vstore4(r0,0,C+base+row*1024+col4*4);
  vstore4(r1,0,C+base+(row+1)*1024+col4*4); vstore4(r2,0,C+base+(row+2)*1024+col4*4);
  vstore4(r3,0,C+base+(row+3)*1024+col4*4);
}}"""
    split_epi_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi_split(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global half *C) {{
  int t=get_global_id(0),row=t/96,col=t-row*96; float4 v=(float4)(0);
  for(int s=0;s<{split_k};s++) v+=convert_float4(vload4(0,C+(s*{M}+row)*1024+col*4));
  write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(col,0))+v);
}}"""
    split_lib, split_epi_lib = dev.compiler.compile(split_source), dev.compiler.compile(split_epi_source)
    for call in targets_split:
      template = call.src[0]
      transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", split_k*M*1024, dtypes.half)
      temporary.buffer.ensure_allocated()
      hand = build_program(template, "gemm_split", split_source, split_lib, (3, 8, split_k), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, N//4, 4)),
                            (dtypes.half, (split_k*M*1024,))), (2,), (0, 1))
      epi = build_program(template, "epi_split", split_epi_source, split_epi_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.half, (split_k*M*1024,))), (0,), (1, 2, 3))
      if hand_split_libs:
        hand_calls = tuple(build_program(template, "gemm_h", "hand_split", lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, N//4, 4)),
                            (dtypes.half, (split_k*M*1024,))), (2,), (0, 1)).call(call.src[4], weight, temporary)
                           for lib in hand_split_libs)
        replacements[call] = hand_calls+(epi.call(call.src[1], call.src[2], call.src[3], temporary),)
      else:
        replacements[call] = (hand.call(call.src[4], weight, temporary),
                              epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets_mixed:
    assert (K//4) % mixed_k == 0
    mixed_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_mixed(read_only image2d_t A,read_only image2d_t B,__global float *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid;
  float4 t0=(float4)(0),t1=(float4)(0),t2=(float4)(0),t3=(float4)(0);
  for(int kb=0;kb<{K//4};kb+={mixed_k}) {{
    half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
    for(int k4=kb;k4<kb+{mixed_k};k4++) {{
      half4 a0=read_imageh(A,smp,(int2)(k4,row)),a1=read_imageh(A,smp,(int2)(k4,row+1));
      half4 a2=read_imageh(A,smp,(int2)(k4,row+2)),a3=read_imageh(A,smp,(int2)(k4,row+3));
      half4 b0=read_imageh(B,smp,(int2)(col4,k4*4)),b1=read_imageh(B,smp,(int2)(col4,k4*4+1));
      half4 b2=read_imageh(B,smp,(int2)(col4,k4*4+2)),b3=read_imageh(B,smp,(int2)(col4,k4*4+3));
      r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
      r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
      r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
      r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
    }}
    t0+=convert_float4(r0); t1+=convert_float4(r1); t2+=convert_float4(r2); t3+=convert_float4(r3);
  }}
  vstore4(t0,0,C+row*1024+col4*4); vstore4(t1,0,C+(row+1)*1024+col4*4);
  vstore4(t2,0,C+(row+2)*1024+col4*4); vstore4(t3,0,C+(row+3)*1024+col4*4);
}}"""
    mixed_lib = dev.compiler.compile(mixed_source)
    for call in targets_mixed:
      template = call.src[0]
      transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", M*1024, dtypes.float)
      temporary.buffer.ensure_allocated()
      hand = build_program(template, "gemm_mixed", mixed_source, mixed_lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, N//4, 4)),
                            (dtypes.float, (M*1024,))), (2,), (0, 1))
      epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.float, (M*1024,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[4], weight, temporary),
                            epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets_localb:
    assert (K//4) % localb_tile == 0 and localb_tile <= 16
    localb_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_localb(read_only image2d_t A,read_only image2d_t B,__global float *C) {{
  __local float4 lb[{localb_tile}][4][32];
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid;
  float4 r0=(float4)(0),r1=(float4)(0),r2=(float4)(0),r3=(float4)(0);
  for(int kb=0;kb<{K//4};kb+={localb_tile}) {{
    if(tm==0) for(int q=0;q<{localb_tile};q++) for(int k=0;k<4;k++)
      lb[q][k][tid]=convert_float4(read_imageh(B,smp,(int2)(col4,(kb+q)*4+k)));
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int q=0;q<{localb_tile};q++) {{
      int k4=kb+q;
      float4 a0=convert_float4(read_imageh(A,smp,(int2)(k4,row)));
      float4 a1=convert_float4(read_imageh(A,smp,(int2)(k4,row+1)));
      float4 a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+2)));
      float4 a3=convert_float4(read_imageh(A,smp,(int2)(k4,row+3)));
      float4 b0=lb[q][0][tid],b1=lb[q][1][tid],b2=lb[q][2][tid],b3=lb[q][3][tid];
      r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
      r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
      r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
      r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  vstore4(r0,0,C+row*1024+col4*4); vstore4(r1,0,C+(row+1)*1024+col4*4);
  vstore4(r2,0,C+(row+2)*1024+col4*4); vstore4(r3,0,C+(row+3)*1024+col4*4);
}}"""
    localb_lib = dev.compiler.compile(localb_source)
    for call in targets_localb:
      template = call.src[0]
      transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", M*1024, dtypes.float)
      temporary.buffer.ensure_allocated()
      hand = build_program(template, "gemm_localb", localb_source, localb_lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, N//4, 4)),
                            (dtypes.float, (M*1024,))), (2,), (0, 1))
      epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.float, (M*1024,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[4], weight, temporary),
                            epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets_globalb:
    globalb_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_globalb(read_only image2d_t A,__global half *B,__global float *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid;
  float4 r0=(float4)(0),r1=(float4)(0),r2=(float4)(0),r3=(float4)(0);
  for(int k4=0;k4<{K//4};k4++) {{
    float4 a0=convert_float4(read_imageh(A,smp,(int2)(k4,row)));
    float4 a1=convert_float4(read_imageh(A,smp,(int2)(k4,row+1)));
    float4 a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+2)));
    float4 a3=convert_float4(read_imageh(A,smp,(int2)(k4,row+3)));
    int p=(k4*4)*{N}+col4*4;
    float4 b0=convert_float4(vload4(0,B+p)),b1=convert_float4(vload4(0,B+p+{N}));
    float4 b2=convert_float4(vload4(0,B+p+{2*N})),b3=convert_float4(vload4(0,B+p+{3*N}));
    r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
    r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
    r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
    r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
  }}
  vstore4(r0,0,C+row*1024+col4*4); vstore4(r1,0,C+(row+1)*1024+col4*4);
  vstore4(r2,0,C+(row+2)*1024+col4*4); vstore4(r3,0,C+(row+3)*1024+col4*4);
}}"""
    globalb_lib = dev.compiler.compile(globalb_source)
    for call in targets_globalb:
      template = call.src[0]
      transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", M*1024, dtypes.float)
      temporary.buffer.ensure_allocated()
      hand = build_program(template, "gemm_globalb", globalb_source, globalb_lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K*N,)),
                            (dtypes.float, (M*1024,))), (2,), (0, 1))
      epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.float, (M*1024,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[4], weight, temporary),
                            epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets_quadmap:
    quadmap_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_quadmap(read_only image2d_t A,read_only image2d_t B,__global float *C) {{
  int lid=get_local_id(0),row=get_group_id(1)*16+(lid&3)*4;
  int col4=get_group_id(0)*32+(lid>>2);
  float4 r0=(float4)(0),r1=(float4)(0),r2=(float4)(0),r3=(float4)(0);
  for(int k4=0;k4<{K//4};k4++) {{
    float4 a0=read_imagef(A,smp,(int2)(k4,row));
    float4 a1=read_imagef(A,smp,(int2)(k4,row+1));
    float4 a2=read_imagef(A,smp,(int2)(k4,row+2));
    float4 a3=read_imagef(A,smp,(int2)(k4,row+3));
    float4 b0=read_imagef(B,smp,(int2)(col4,k4*4));
    float4 b1=read_imagef(B,smp,(int2)(col4,k4*4+1));
    float4 b2=read_imagef(B,smp,(int2)(col4,k4*4+2));
    float4 b3=read_imagef(B,smp,(int2)(col4,k4*4+3));
    r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
    r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
    r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
    r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
  }}
  vstore4(r0,0,C+row*1024+col4*4); vstore4(r1,0,C+(row+1)*1024+col4*4);
  vstore4(r2,0,C+(row+2)*1024+col4*4); vstore4(r3,0,C+(row+3)*1024+col4*4);
}}"""
    quadmap_lib = dev.compiler.compile(quadmap_source)
    for call in targets_quadmap:
      template = call.src[0]
      transposed = np.array(call.src[5].buffer.numpy(), copy=True).reshape(96, 1536, 4).transpose(1, 0, 2).copy()
      weight = UOp.new_buffer("QCOM", transposed.size, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      temporary = UOp.new_buffer("QCOM", M*1024, dtypes.float)
      temporary.buffer.ensure_allocated()
      hand = build_program(template, "gemm_quadmap", quadmap_source, quadmap_lib, (3, 8, 1), (128, 1, 1),
                           ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, N//4, 4)),
                            (dtypes.float, (M*1024,))), (2,), (0, 1))
      epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                          ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                           (dtypes.float, (1, 96, 4)), (dtypes.float, (M*1024,))), (0,), (1, 2, 3))
      replacements[call] = (hand.call(call.src[4], weight, temporary),
                            epi.call(call.src[1], call.src[2], call.src[3], temporary))

  if targets2_lowrank:
    assert lowrank2_rank % 4 == 0 and lowrank2_rank <= 512
    lr1_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void lowrank1(read_only image2d_t A,__global half *L,__global float *T) {{
  int t=get_global_id(0),row=t/{lowrank2_rank//4},col4=t-row*{lowrank2_rank//4}; if(row>=192) return;
  float4 z=(float4)(0);
  for(int k4=0;k4<192;k4++) {{
    float4 a=convert_float4(read_imageh(A,smp,(int2)(k4,row))); int p=k4*4;
    z+=a.x*convert_float4(vload4(0,L+(p+0)*{lowrank2_rank}+col4*4));
    z+=a.y*convert_float4(vload4(0,L+(p+1)*{lowrank2_rank}+col4*4));
    z+=a.z*convert_float4(vload4(0,L+(p+2)*{lowrank2_rank}+col4*4));
    z+=a.w*convert_float4(vload4(0,L+(p+3)*{lowrank2_rank}+col4*4));
  }}
  vstore4(z,0,T+row*{lowrank2_rank}+col4*4);
}}"""
    lr2_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void lowrank2(write_only image2d_t O,read_only image2d_t X,__global float *T,__global half *R) {{
  int t=get_global_id(0),row=t/128,col=t-row*128,y=row>>2,r=row&3,x=col+r*128; float4 z=(float4)(0);
  for(int k=0;k<{lowrank2_rank};k++) z+=T[row*{lowrank2_rank}+k]*convert_float4(vload4(0,R+k*512+col*4));
  write_imagef(O,(int2)(x,y),read_imagef(X,smp,(int2)(x,y))+z);
}}"""
    lr1_lib, lr2_lib = dev.compiler.compile(lr1_source), dev.compiler.compile(lr2_source)
    for call in targets2_lowrank:
      template = call.src[0]
      matrix = np.array(call.src[3].buffer.numpy(), copy=True).reshape(128, 768, 4).transpose(1, 0, 2).reshape(768, 512)
      u, s, vh = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
      left, right = (u[:, :lowrank2_rank]*s[:lowrank2_rank]).astype(np.float16), vh[:lowrank2_rank].astype(np.float16)
      def upload_lr(arr):
        ret = UOp.new_buffer("QCOM", arr.size, dtypes.half)
        ret.buffer.ensure_allocated()
        ret.buffer.copyin(memoryview(arr).cast("B"))
        return ret
      lbuf, rbuf = upload_lr(left), upload_lr(right)
      temporary = UOp.new_buffer("QCOM", 192*lowrank2_rank, dtypes.float)
      temporary.buffer.ensure_allocated()
      lr1 = build_program(template, "lowrank1", lr1_source, lr1_lib, ((192*lowrank2_rank//4+127)//128, 1, 1), (128, 1, 1),
                          ((dtypes.half, (192, 192, 4)), (dtypes.half, (768, lowrank2_rank)),
                           (dtypes.float, (192, lowrank2_rank))), (2,), (0, 1))
      lr2 = build_program(template, "lowrank2", lr2_source, lr2_lib, (192, 1, 1), (128, 1, 1),
                          ((dtypes.float, (48, 512, 4)), (dtypes.float, (48, 512, 4)),
                           (dtypes.float, (192, lowrank2_rank)), (dtypes.half, (lowrank2_rank, 512))), (0,), (1, 2, 3))
      replacements[call] = (lr1.call(call.src[4], lbuf, temporary), lr2.call(call.src[1], call.src[2], temporary, rbuf))

  for call in targets2:
    template = call.src[0]
    weight = UOp.new_buffer("QCOM", 768*128*4, dtypes.half)
    weight.buffer.ensure_allocated()
    pack_weight = build_program(template, "pack_target2_b", pack_target2_b_source, pack_target2_b_lib,
                                (192, 1, 1), (128, 1, 1),
                                ((dtypes.half, None), (dtypes.float, (128, 768, 4))), (0,), (1,))
    temporary = UOp.new_buffer("QCOM", 192*1024, dtypes.half)
    temporary.buffer.ensure_allocated()
    use_8x8 = bool(getenv("OPENPILOT_8X8_ALL"))
    pack_call, activation = pack_banked_activation(template, call.src[4], 192, 768) if float_hand_activations else (None, call.src[4])
    hand = build_program(template, "gemm_h", "hand", hand_lib8[768] if use_8x8 else hand_lib2,
                         (2, 6, 1) if use_8x8 else (4, 12, 1), (128, 1, 1),
                         ((dtypes.half, (192, 192, 4)), (dtypes.half, (768, 128, 4)),
                          (dtypes.half, (192*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi2", EPILOGUE2, epi_lib2, (192, 1, 1), (128, 1, 1),
                        ((dtypes.float, (48, 512, 4)), (dtypes.float, (48, 512, 4)),
                         (dtypes.half, (192*1024,))), (0,), (1, 2))
    if getenv("OPENPILOT_TERNARY"):
      original = np.array(call.src[3].buffer.numpy(), copy=True).reshape(128, 4, 192, 4)
      transposed = original.transpose(2, 3, 0, 1).reshape(768, 128, 4).astype(np.float16).copy()
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, transposed, 192, 512, 768)
      replacements[call] = ((epi.call(call.src[1], call.src[2], temporary),) if sparse is None else
                            (sparse.call(call.src[4], offbuf, idxbuf, valbuf, temporary), epi.call(call.src[1], call.src[2], temporary)))
    else: replacements[call] = (pack_weight.call(weight, call.src[3]), *(() if pack_call is None else (pack_call,)),
                                hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], temporary))

  for call in targets2_fp32:
    materialize_before(call)
    template = call.src[0]
    float_inputs = bool(getenv("OPENPILOT_FP32_WIDE_FLOAT_INPUTS"))
    input_dtype = dtypes.float if float_inputs else dtypes.half
    # The compiler kernel stores K as four 192-wide component blocks, while the
    # hand kernel walks scalar K contiguously. Interleave those blocks first.
    weight = UOp.new_buffer("QCOM", 768*128*4, input_dtype)
    weight.buffer.ensure_allocated()
    temporary = UOp.new_buffer("QCOM", 192*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    use_4x4 = bool(getenv("OPENPILOT_FP32_WIDE_4X4"))
    if use_4x4 and float_inputs: raise ValueError("dynamic target2 4x4 packing currently produces half inputs")
    pack_call, activation = pack_banked_activation(template, call.src[4], 192, 768) if use_4x4 else (None, call.src[4])
    pack_weight = build_program(template, "pack_target2_b", pack_target2_b_source, pack_target2_b_lib,
                                (192, 1, 1), (128, 1, 1),
                                ((dtypes.half, None), (dtypes.float, (128, 768, 4))), (0,), (1,)) if use_4x4 else None
    activation_shape = (192, 192, 4) if use_4x4 else (48, 768, 4)
    hand = build_program(template, "gemm_h" if use_4x4 else "gemm_f", "hand_fp32_wide",
                         fp32_wide4_libs[768] if use_4x4 else fp32_wide_libs[768],
                         (4, 12, 1) if use_4x4 else (2, 12, 1), (128, 1, 1),
                         ((input_dtype, activation_shape), (input_dtype, (768, 128, 4)),
                          (dtypes.float, (192*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi2_fp32", EPILOGUE2_FP32, epi2_fp32_lib, (192, 1, 1), (128, 1, 1),
                        ((dtypes.float, (48, 512, 4)), (dtypes.float, (48, 512, 4)),
                         (dtypes.float, (192*1024,))), (0,), (1, 2))
    replacements[call] = (*(() if pack_weight is None else (pack_weight.call(weight, call.src[3]),)),
                          *(() if pack_call is None else (pack_call,)),
                          hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], temporary))

  for call in targets3:
    template, temporary = call.src[0], UOp.new_buffer("QCOM", 128*2048, dtypes.half)
    temporary.buffer.ensure_allocated()
    weight = UOp.new_buffer("QCOM", call.src[4].buffer.size, dtypes.half)
    weight.buffer.ensure_allocated()
    original = np.array(call.src[4].buffer.numpy(), copy=False).reshape(384, 384, 4)
    weight.buffer.copyin(memoryview(original.transpose(1, 0, 2).copy()).cast("B"))
    use_8x8 = bool(getenv("OPENPILOT_8X8_TARGET3")) or bool(getenv("OPENPILOT_8X8_ALL"))
    hand = build_program(template, "gemm_h", "hand", hand_lib8_target3 if use_8x8 else hand_lib3,
                         (6, 4, 1) if use_8x8 else (3, 8, 1), (128, 1, 1),
                         ((dtypes.half, (128, 96, 4)), (dtypes.half, (384, 384, 4)),
                          (dtypes.half, (128*2048,))), (2,), (0, 1))
    epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi3(__global half *O,__global float *S,__global float *B,__global half *C) {
  int t=get_global_id(0),col=t%384,q=t/384,block=q&3,y=q>>2;
  int n=col*4+block,row=y*4,o=((y*1536+block*384+col)*4);
  float4 z=(float4)(C[(row+0)*2048+n],C[(row+1)*2048+n],C[(row+2)*2048+n],C[(row+3)*2048+n]);
  z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
}"""
    epi_lib3 = dev.compiler.compile(epi_source)
    epi = build_program(template, "epi3", epi_source, epi_lib3, (384, 1, 1), (128, 1, 1),
                        ((dtypes.half, (128*1536,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                         (dtypes.half, (128*2048,))), (0,), (1, 2, 3))
    if getenv("OPENPILOT_TERNARY"):
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, original.transpose(1, 0, 2), 128, 1536, 384, 2048)
      replacements[call] = ((epi.call(call.src[1], call.src[2], call.src[5], temporary),) if sparse is None else
                            (sparse.call(call.src[3], offbuf, idxbuf, valbuf, temporary),
                             epi.call(call.src[1], call.src[2], call.src[5], temporary)))
    else: replacements[call] = (hand.call(call.src[3], weight, temporary), epi.call(call.src[1], call.src[2], call.src[5], temporary))

  for call in targets3_fp32:
    template, temporary = call.src[0], UOp.new_buffer("QCOM", 128*2048, dtypes.float)
    temporary.buffer.ensure_allocated()
    float_weight = bool(getenv("OPENPILOT_FP32_TARGET3_FLOAT_WEIGHT"))
    weight_dtype = dtypes.float if float_weight else dtypes.half
    original = np.array(call.src[4].buffer.numpy(), copy=False).reshape(384, 384, 4)
    transposed = original.transpose(1, 0, 2).copy().astype(np.float32 if float_weight else np.float16)
    weight = UOp.new_buffer("QCOM", transposed.size, weight_dtype)
    weight.buffer.ensure_allocated()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    hand = build_program(template, "gemm_h", "hand_fp32", fp32_lib3, (12, 8, 1), (128, 1, 1),
                         ((dtypes.half, (128, 96, 4)), (weight_dtype, (384, 384, 4)),
                          (dtypes.float, (128*2048,))), (2,), (0, 1))
    epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi3_fp32(__global half *O,__global float *S,__global float *B,__global float *C) {
  int t=get_global_id(0),row=t/384,col=t-row*384,y=row>>2,r=row&3;
  int o=((y*1536+r*384+col)*4);
  float4 z=vload4(0,C+row*2048+col*4); z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
}"""
    epi = build_program(template, "epi3_fp32", epi_source, dev.compiler.compile(epi_source), (384, 1, 1), (128, 1, 1),
                        ((dtypes.half, (128*1536,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                         (dtypes.float, (128*2048,))), (0,), (1, 2, 3))
    replacements[call] = (hand.call(call.src[3], weight, temporary), epi.call(call.src[1], call.src[2], call.src[5], temporary))

  for call in targets4:
    template, temporary, calls = call.src[0], UOp.new_buffer("QCOM", 512*1024, dtypes.half), []
    temporary.buffer.ensure_allocated()
    use_8x8 = bool(getenv("OPENPILOT_8X8_ALL"))
    pack_call, activation = pack_banked_activation(template, call.src[3], 512, 192) \
                            if use_8x8 or float_hand_activations else (None, call.src[3])
    if pack_call is not None: calls.append(pack_call)
    original = np.array(call.src[4].buffer.numpy(), copy=False).reshape(192, 192, 4)
    # The compiler image is [output_index, K, output_component]. A conventional
    # GEMM needs K rows with output_index-major, four-component columns.
    logical_weight = original.transpose(1, 0, 2).reshape(192, 192, 4).astype(np.float16)
    for chunk, (base, colgroups, ncols) in enumerate(((0, 128, 4), (128, 64, 2))):
      shader_colgroups = 128 if thread_target4 else colgroups
      weight = UOp.new_buffer("QCOM", shader_colgroups*192*4, dtypes.half)
      weight.buffer.ensure_allocated()
      transposed = np.zeros((192, shader_colgroups, 4), dtype=np.float16)
      transposed[:, :colgroups] = logical_weight[:, base:base+colgroups]
      weight.buffer.copyin(memoryview(transposed).cast("B"))
      thread_8x8 = use_8x8 and bool(getenv("OPENPILOT_8X8_THREAD_TARGET4"))
      hand_ncols = 4 if thread_target4 else ncols
      hand8_lib = hand_lib8_thread[(192, colgroups//64)] if thread_8x8 else hand_lib8[192] if use_8x8 else None
      hand = build_program(template, "gemm_h", "hand", hand8_lib if use_8x8 else hand_lib4[hand_ncols],
                           ((colgroups//64), 16, 1) if use_8x8 else (1, 32, 1), (128, 1, 1),
                           ((dtypes.half, (512, 48, 4)), (dtypes.half, (192, shader_colgroups, 4)),
                            (dtypes.half, (512*1024,))), (2,), (0, 1))
      output_decl = "float" if float_hand_activations else "half"
      output_value = "(float4)(*S)*z*z+(float4)(*B)" if float_hand_activations else \
                     "convert_half4((float4)(*S)*z*z+(float4)(*B))"
      thread_load = """int gy=row>>4,rr=row&3,tm=(row>>2)&3,tid=col&31,cg=col>>5;
  int thread=(gy*2)*128+tm*32+tid;
  float4 z=convert_float4(vload4(0,C+thread*64+(rr*4+cg)*4));""" if thread_target4 else \
                    f"""int gy=row>>5,rr=row&7,tm=(row>>3)&3,gx=col>>6,x=col&63,tid=x&31,cg=x>>5;
  int thread=(gy*{colgroups//64}+gx)*128+tm*32+tid;
  float4 z=convert_float4(vload4(0,C+thread*64+(rr*2+cg)*4));""" if thread_8x8 else \
                    "float4 z=convert_float4(vload4(0,C+row*1024+col*4));"
      epi_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi4{chunk}(__global {output_decl} *O,__global float *S,__global float *B,__global half *C) {{
  int t=get_global_id(0),row=t/{colgroups},col=t-row*{colgroups},y=row>>2,r=row&3;
  int o=((y*768+r*192+{base}+col)*4);
  {thread_load} z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4({output_value},0,O+o);
}}"""
      epi_lib4 = dev.compiler.compile(epi_source)
      epi = build_program(template, f"epi4{chunk}", epi_source, epi_lib4, (512 if chunk == 0 else 256, 1, 1), (128, 1, 1),
                          ((dtypes.float if float_hand_activations else dtypes.half, (512*768,)),
                           (dtypes.float, (1,)), (dtypes.float, (1,)),
                           (dtypes.half, (512*1024,))), (0,), (1, 2, 3))
      if getenv("OPENPILOT_TERNARY"):
        sparse, offbuf, idxbuf, valbuf, sparse_tmp = make_ternary_sparse(dev, template, transposed, 512, colgroups*4, 192)
        calls += ([epi.call(call.src[1], call.src[2], call.src[5], sparse_tmp)] if sparse is None else
                  [sparse.call(call.src[3], offbuf, idxbuf, valbuf, sparse_tmp),
                   epi.call(call.src[1], call.src[2], call.src[5], sparse_tmp)])
      else: calls += [hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], call.src[5], temporary)]
    replacements[call] = tuple(calls)

  for call in targets4_fp32:
    materialize_before(call)
    template = call.src[0]
    float_inputs = bool(getenv("OPENPILOT_FP32_SMALL_FLOAT_INPUTS"))
    input_dtype = dtypes.float if float_inputs else dtypes.half
    original = np.array(call.src[4].buffer.numpy(), copy=False).reshape(192, 192, 4)
    # The compiler image is [output_index, K, output_component]. Preserve the
    # output-index/component order expected by the fused output image.
    transposed = original.transpose(1, 0, 2).reshape(192, 768).astype(
      np.float32 if float_inputs else np.float16).reshape(192, 192, 4).copy()
    weight = UOp.new_buffer("QCOM", transposed.size, input_dtype)
    weight.buffer.ensure_allocated()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", 512*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    hand = build_program(template, "gemm_f", "hand_fp32", fp32_target4_lib if float_inputs else fp32_lib_small,
                         (3, 32, 1) if float_inputs else (6, 32, 1), (128, 1, 1),
                         ((input_dtype, (512, 48, 4)), (input_dtype, (192, 192, 4)),
                          (dtypes.float, (512*1024,))), (2,), (0, 1))
    output_decl, store_expr = ("float", "(float4)(*S)*z*z+(float4)(*B)") if float_inputs else \
                              ("half", "convert_half4((float4)(*S)*z*z+(float4)(*B))")
    epi_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi4_fp32(__global {output_decl} *O,__global float *S,__global float *B,__global float *C) {{
  int t=get_global_id(0),y=t/192,col=t-y*192;
  for (int r=0;r<4;r++) {{
    int row=y*4+r,o=(y*768+r*192+col)*4;
    float4 z=(float4)(C[row*1024+col],C[row*1024+col+192],C[row*1024+col+384],C[row*1024+col+576]);
    z=select((float4)(0),z,isgreater(z,(float4)(0)));
    vstore4({store_expr},0,O+o);
  }}
}}"""
    epi = build_program(template, "epi4_fp32", epi_source, dev.compiler.compile(epi_source), (192, 1, 1), (128, 1, 1),
                        ((dtypes.float if float_inputs else dtypes.half, (512*768,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                         (dtypes.float, (512*1024,))), (0,), (1, 2, 3))
    replacements[call] = (hand.call(call.src[3], weight, temporary),
                          epi.call(call.src[1], call.src[2], call.src[5], temporary))

  for call in targets5:
    template = call.src[0]
    weight = UOp.new_buffer("QCOM", call.src[3].buffer.size, dtypes.half)
    weight.buffer.ensure_allocated()
    original = np.array(call.src[3].buffer.numpy(), copy=True).reshape(128, 4, 96, 4)
    transposed = original.transpose(2, 3, 0, 1).reshape(384, 128, 4).astype(np.float16).copy()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", 192*1024, dtypes.half)
    temporary.buffer.ensure_allocated()
    use_8x8 = bool(getenv("OPENPILOT_8X8_ALL"))
    pack_call, activation = pack_banked_activation(template, call.src[4], 192, 384) if float_hand_activations else (None, call.src[4])
    hand = build_program(template, "gemm_h", "hand", hand_lib8[384] if use_8x8 else hand_lib5,
                         (2, 6, 1) if use_8x8 else (1, 12, 1), (128, 1, 1),
                         ((dtypes.half, (192, 96, 4)), (dtypes.half, (384, 128, 4)),
                          (dtypes.half, (192*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi2", EPILOGUE2, epi_lib2, (192, 1, 1), (128, 1, 1),
                        ((dtypes.float, (48, 512, 4)), (dtypes.float, (48, 512, 4)),
                         (dtypes.half, (192*1024,))), (0,), (1, 2))
    if getenv("OPENPILOT_TERNARY"):
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, transposed, 192, 512, 384)
      replacements[call] = ((epi.call(call.src[1], call.src[2], temporary),) if sparse is None else
                            (sparse.call(call.src[4], offbuf, idxbuf, valbuf, temporary), epi.call(call.src[1], call.src[2], temporary)))
    else: replacements[call] = (*(() if pack_call is None else (pack_call,)),
                                hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], temporary))

  for call in targets5_fp32:
    materialize_before(call)
    template = call.src[0]
    float_inputs = bool(getenv("OPENPILOT_FP32_WIDE_FLOAT_INPUTS"))
    input_dtype = dtypes.float if float_inputs else dtypes.half
    weight = UOp.new_buffer("QCOM", 384*128*4, input_dtype)
    weight.buffer.ensure_allocated()
    temporary = UOp.new_buffer("QCOM", 192*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    use_4x4 = bool(getenv("OPENPILOT_FP32_WIDE_4X4"))
    if use_4x4 and float_inputs: raise ValueError("dynamic target5 4x4 packing currently produces half inputs")
    pack_call, activation = pack_banked_activation(template, call.src[4], 192, 384) if use_4x4 else (None, call.src[4])
    pack_weight = build_program(template, "pack_target5_b", pack_target5_b_source, pack_target5_b_lib,
                                (96, 1, 1), (128, 1, 1),
                                ((dtypes.half, None), (dtypes.float, (128, 384, 4))), (0,), (1,)) if use_4x4 else None
    activation_shape = (192, 96, 4) if use_4x4 else (48, 384, 4)
    hand = build_program(template, "gemm_h" if use_4x4 else "gemm_f", "hand_fp32_wide",
                         fp32_wide4_libs[384] if use_4x4 else fp32_wide_libs[384],
                         (4, 12, 1) if use_4x4 else (2, 12, 1), (128, 1, 1),
                         ((input_dtype, activation_shape), (input_dtype, (384, 128, 4)),
                          (dtypes.float, (192*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi2_fp32", EPILOGUE2_FP32, epi2_fp32_lib, (192, 1, 1), (128, 1, 1),
                        ((dtypes.float, (48, 512, 4)), (dtypes.float, (48, 512, 4)),
                         (dtypes.float, (192*1024,))), (0,), (1, 2))
    replacements[call] = (*(() if pack_weight is None else (pack_weight.call(weight, call.src[3]),)),
                          *(() if pack_call is None else (pack_call,)),
                          hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], temporary))

  for call in targets6:
    template = call.src[0]
    use_8x8 = bool(getenv("OPENPILOT_8X8_ALL"))
    weight = UOp.new_buffer("QCOM", 192*(128 if use_8x8 else 96)*4, dtypes.half)
    weight.buffer.ensure_allocated()
    original = np.array(call.src[4].buffer.numpy(), copy=True).reshape(96, 192, 4)
    transposed = (original.transpose(1, 0, 2) if use_8x8 else
                  original.transpose(1, 2, 0).reshape(192, 96, 4)).astype(np.float16).copy()
    if use_8x8:
      padded = np.zeros((192, 128, 4), dtype=np.float16)
      padded[:, :96] = transposed
      weight.buffer.copyin(memoryview(padded).cast("B"))
    else: weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", 512*1024, dtypes.half)
    temporary.buffer.ensure_allocated()
    pack_call, activation = pack_banked_activation(template, call.src[3], 512, 192) \
                            if use_8x8 or float_hand_activations else (None, call.src[3])
    hand = build_program(template, "gemm_h", "hand", hand_lib8[192] if use_8x8 else hand_lib6,
                         (2, 16, 1) if use_8x8 else (1, 32, 1), (128, 1, 1),
                         ((dtypes.half, (512, 48, 4)), (dtypes.half, (192, 128 if use_8x8 else 96, 4)),
                          (dtypes.half, (512*1024,))), (2,), (0, 1))
    output_decl = "float" if float_hand_activations else "half"
    output_value = "(float4)(*S)*z*z+(float4)(*B)" if float_hand_activations else \
                   "convert_half4((float4)(*S)*z*z+(float4)(*B))"
    epi_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi6(__global {output_decl} *O,__global float *S,__global float *B,__global half *C) {{
  int t=get_global_id(0),row=t/96,col=t-row*96,y=row>>2,r=row&3,o=((y*384+r*96+col)*4);
  float4 z=convert_float4(vload4(0,C+row*1024+col*4)); z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4({output_value},0,O+o);
}}"""
    epi_lib6 = dev.compiler.compile(epi_source)
    epi = build_program(template, "epi6", epi_source, epi_lib6, (384, 1, 1), (128, 1, 1),
                        ((dtypes.float if float_hand_activations else dtypes.half, (512*384,)),
                         (dtypes.float, (1,)), (dtypes.float, (1,)),
                         (dtypes.half, (512*1024,))), (0,), (1, 2, 3))
    if getenv("OPENPILOT_TERNARY"):
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, transposed, 512, 384, 192)
      replacements[call] = ((epi.call(call.src[1], call.src[2], call.src[5], temporary),) if sparse is None else
                            (sparse.call(call.src[3], offbuf, idxbuf, valbuf, temporary),
                             epi.call(call.src[1], call.src[2], call.src[5], temporary)))
    else: replacements[call] = (*(() if pack_call is None else (pack_call,)),
                                hand.call(activation, weight, temporary), epi.call(call.src[1], call.src[2], call.src[5], temporary))

  for call in targets6_fp32:
    template = call.src[0]
    pack_call, activation = pack_banked_activation(template, call.src[3], 512, 192) \
                            if getenv("OPENPILOT_FP32_SMALL_FLOAT_INPUTS") else (None, call.src[3])
    weight = UOp.new_buffer("QCOM", call.src[4].buffer.size, dtypes.half)
    weight.buffer.ensure_allocated()
    transposed = np.array(call.src[4].buffer.numpy(), copy=True).reshape(96, 192, 4).transpose(1, 0, 2).astype(np.float16).copy()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", 512*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    hand = build_program(template, "gemm_h", "hand_fp32", fp32_lib_small,
                         (3, 32, 1), (128, 1, 1),
                         ((dtypes.half, (512, 48, 4)), (dtypes.half, (192, 96, 4)),
                          (dtypes.float, (512*1024,))), (2,), (0, 1))
    epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi6_fp32(__global half *O,__global float *S,__global float *B,__global float *C) {
  int t=get_global_id(0),row=t/96,col=t-row*96,y=row>>2,r=row&3,o=((y*384+r*96+col)*4);
  float4 z=vload4(0,C+row*1024+col*4); z=select((float4)(0),z,isgreater(z,(float4)(0)));
  vstore4(convert_half4((float4)(*S)*z*z+(float4)(*B)),0,O+o);
}"""
    epi = build_program(template, "epi6_fp32", epi_source, dev.compiler.compile(epi_source), (384, 1, 1), (128, 1, 1),
                        ((dtypes.half, (512*384,)), (dtypes.float, (1,)), (dtypes.float, (1,)),
                         (dtypes.float, (512*1024,))), (0,), (1, 2, 3))
    replacements[call] = (*(() if pack_call is None else (pack_call,)), hand.call(activation, weight, temporary),
                          epi.call(call.src[1], call.src[2], call.src[5], temporary))

  for call in targets7:
    template, temporary, calls = call.src[0], UOp.new_buffer("QCOM", 128*1024, dtypes.half), []
    temporary.buffer.ensure_allocated()
    original = np.array(call.src[3].buffer.numpy(), copy=False).reshape(36, 8, 96, 4, 4)
    all_weights = original.transpose(2, 3, 0, 1, 4).reshape(384, 288, 4).astype(np.float16)
    for chunk, (base, colgroups, ncols) in enumerate(((0, 128, 4), (128, 128, 4), (256, 32, 1))):
      weight = UOp.new_buffer("QCOM", colgroups*384*4, dtypes.half)
      weight.buffer.ensure_allocated()
      weight.buffer.copyin(memoryview(all_weights[:, base:base+colgroups].copy()).cast("B"))
      use_8x8 = bool(getenv("OPENPILOT_8X8_ALL")) and ncols == 4
      hand = build_program(template, "gemm_h", "hand", hand_lib8[384] if use_8x8 else hand_lib3 if ncols == 4 else hand_lib7,
                           (2, 4, 1) if use_8x8 else (1, 8, 1), (128, 1, 1),
                           ((hand_activation_dtype, (128, 96, 4)), (dtypes.half, (384, colgroups, 4)),
                            (dtypes.half, (128*1024,))), (2,), (0, 1))
      epi_source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi7{chunk}(__global float *O,__global half *C) {{
  int t=get_global_id(0),row=t/{colgroups},lc=t-row*{colgroups},col={base}+lc;
  int x=(row>>2)*32+(row&3)*8+(col&7),y=col>>3,o=(y*1024+x)*4;
  vstore4(convert_float4(vload4(0,C+row*1024+lc*4)),0,O+o);
}}"""
      epi_lib7 = dev.compiler.compile(epi_source)
      epi = build_program(template, f"epi7{chunk}", epi_source, epi_lib7, (colgroups, 1, 1), (128, 1, 1),
                          ((dtypes.float, (128*1152,)), (dtypes.half, (128*1024,))), (0,), (1,))
      if getenv("OPENPILOT_TERNARY"):
        sparse, offbuf, idxbuf, valbuf, sparse_tmp = make_ternary_sparse(
          dev, template, all_weights[:, base:base+colgroups], 128, colgroups*4, 384)
        calls += ([epi.call(call.src[1], sparse_tmp)] if sparse is None else
                  [sparse.call(call.src[2], offbuf, idxbuf, valbuf, sparse_tmp), epi.call(call.src[1], sparse_tmp)])
      else: calls += [hand.call(call.src[2], weight, temporary), epi.call(call.src[1], temporary)]
    replacements[call] = tuple(calls)

  for call in targets8:
    template = call.src[0]
    sum_a, sum_r = UOp.new_buffer("QCOM", 2*576*4, dtypes.float), UOp.new_buffer("QCOM", 2*144*4, dtypes.float)
    sum_a.buffer.ensure_allocated()
    sum_r.buffer.ensure_allocated()
    sum_a_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void suma(__global float *O,read_only image2d_t A) {
  int t=get_global_id(0); float4 v=(float4)(0);
  for(int l=0;l<16;l++) v+=read_imagef(A,smp,(int2)(t,l)); vstore4(v,0,O+t*4);
}"""
    sum_r_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void sumr(__global float *O,read_only image2d_t R) {
  int t=get_global_id(0); if(t<288) { int g=t%144,r=t/144; float4 v=(float4)(0);
  for(int l=0;l<16;l++) v+=read_imagef(R,smp,(int2)(g+l*288+r*144,0)); vstore4(v,0,O+t*4); }
}"""
    main8_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(16,1,1)))
__kernel void reduce8(write_only image2d_t O,read_only image2d_t V,read_only image2d_t B,__global float *A,__global float *R) {
  __local float4 partial[16]; int g=get_group_id(0),l=get_local_id(0); float4 acc=(float4)(0);
  for(int r=0;r<2;r++) for(int k=l;k<576;k+=16) { float4 a=vload4(0,A+(r*576+k)*4); int x=k*4;
    float4 b0=read_imagef(B,smp,(int2)(x,g)),b1=read_imagef(B,smp,(int2)(x+1,g));
    float4 b2=read_imagef(B,smp,(int2)(x+2,g)),b3=read_imagef(B,smp,(int2)(x+3,g));
    acc+=a.x*b0+a.y*b1+a.z*b2+a.w*b3; }
  partial[l]=acc; barrier(CLK_LOCAL_MEM_FENCE); if(l==0) { for(int i=1;i<16;i++) acc+=partial[i];
  float4 v=read_imagef(V,smp,(int2)(g,0)); acc+=v*(vload4(0,R+g*4)+vload4(0,R+(144+g)*4));
  write_imagef(O,(int2)(g,0),acc*(float4)(0.03125f)); }
}"""
    suma = build_program(template, "suma", sum_a_source, dev.compiler.compile(sum_a_source), (9, 1, 1), (128, 1, 1),
                         ((dtypes.float, (2*576*4,)), (dtypes.float, (16, 1152, 4))), (0,), (1,))
    sumr = build_program(template, "sumr", sum_r_source, dev.compiler.compile(sum_r_source), (3, 1, 1), (128, 1, 1),
                         ((dtypes.float, (2*144*4,)), (dtypes.float, (1, 4608, 4))), (0,), (1,))
    main8 = build_program(template, "reduce8", main8_source, dev.compiler.compile(main8_source), (144, 1, 1), (16, 1, 1),
                          ((dtypes.float, (1, 144, 4)), (dtypes.float, (1, 144, 4)), (dtypes.half, (144, 2304, 4)),
                           (dtypes.float, (2*576*4,)), (dtypes.float, (2*144*4,))), (0,), (1, 2, 3, 4))
    replacements[call] = (suma.call(sum_a, call.src[4]), sumr.call(sum_r, call.src[2]),
                          main8.call(call.src[1], call.src[3], call.src[5], sum_a, sum_r))

  for call in targets9:
    template = call.src[0]
    weight = UOp.new_buffer("QCOM", call.src[5].buffer.size, dtypes.half)
    weight.buffer.ensure_allocated()
    original = np.array(call.src[5].buffer.numpy(), copy=False).reshape(96, 384, 4)
    transposed = original.transpose(1, 2, 0).reshape(384, 96, 4).astype(np.float16).copy()
    weight.buffer.copyin(memoryview(transposed).cast("B"))
    temporary = UOp.new_buffer("QCOM", 128*1024, dtypes.half)
    temporary.buffer.ensure_allocated()
    hand = build_program(template, "gemm_h", "hand", hand_lib9, (1, 8, 1), (128, 1, 1),
                         ((hand_activation_dtype, (128, 96, 4)), (dtypes.half, (384, 96, 4)),
                          (dtypes.half, (128*1024,))), (2,), (0, 1))
    epi_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi9(write_only image2d_t O,read_only image2d_t X,read_only image2d_t S,__global half *C) {
  int t=get_global_id(0),row=t/96,col=t-row*96;
  float4 v=convert_float4(vload4(0,C+row*1024+col*4));
  write_imagef(O,(int2)(t,0),read_imagef(X,smp,(int2)(t,0))*read_imagef(S,smp,(int2)(col,0))+v);
}"""
    epi9 = build_program(template, "epi9", epi_source, dev.compiler.compile(epi_source), (96, 1, 1), (128, 1, 1),
                         ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                          (dtypes.float, (1, 96, 4)), (dtypes.half, (128*1024,))), (0,), (1, 2, 3))
    if getenv("OPENPILOT_TERNARY"):
      sparse, offbuf, idxbuf, valbuf, temporary = make_ternary_sparse(dev, template, original.transpose(1, 0, 2), 128, 384, 384)
      replacements[call] = ((epi9.call(call.src[1], call.src[2], call.src[3], temporary),) if sparse is None else
                            (sparse.call(call.src[4], offbuf, idxbuf, valbuf, temporary),
                             epi9.call(call.src[1], call.src[2], call.src[3], temporary)))
    else: replacements[call] = (hand.call(call.src[4], weight, temporary), epi9.call(call.src[1], call.src[2], call.src[3], temporary))

  for call in targets9_fp32:
    template = call.src[0]
    weight = UOp.new_buffer("QCOM", call.src[5].buffer.size, dtypes.half)
    weight.buffer.ensure_allocated()
    original = np.array(call.src[5].buffer.numpy(), copy=False).reshape(96, 384, 4)
    weight.buffer.copyin(memoryview(original.transpose(1, 0, 2).astype(np.float16).copy()).cast("B"))
    temporary = UOp.new_buffer("QCOM", 128*1024, dtypes.float)
    temporary.buffer.ensure_allocated()
    hand = build_program(template, "gemm_f", "hand_fp32", fp32_lib9, (3, 8, 1), (128, 1, 1),
                         ((dtypes.float, (128, 96, 4)), (dtypes.half, (384, 96, 4)),
                          (dtypes.float, (128*1024,))), (2,), (0, 1))
    epi = build_program(template, "epi_fp32", EPILOGUE_FP32, epi_fp32_lib, (96, 1, 1), (128, 1, 1),
                        ((dtypes.float, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                         (dtypes.float, (1, 96, 4)), (dtypes.float, (128*1024,))), (0,), (1, 2, 3))
    replacements[call] = (hand.call(call.src[4], weight, temporary),
                          epi.call(call.src[1], call.src[2], call.src[3], temporary))

  # The generated first convolution reconstructs flattened border coordinates with
  # div/mod operations. The physical image layout is six float4 channel groups per
  # input x coordinate, so clamped image reads can express the same four taps directly.
  firstconv_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void firstconv_fast(write_only image2d_t O, read_only image2d_t A,
                             read_only image2d_t W, read_only image2d_t B) {
  const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
  int ix=get_global_id(0), iy=get_global_id(1), oc=get_global_id(2);
  float4 z0=(float4)(0), z1=(float4)(0), z2=(float4)(0), z3=(float4)(0);
  for (int ic=0; ic<6; ic++) for (int ky=0; ky<7; ky++) {
    int ay=iy*4+ky-2;
    for (int kx=0; kx<7; kx++) {
      int ax=ix*16+kx, wb=ky*168+ic*28+kx*4;
      float4 w0=read_imagef(W,smp,(int2)(wb,oc));
      float4 w1=read_imagef(W,smp,(int2)(wb+1,oc));
      float4 w2=read_imagef(W,smp,(int2)(wb+2,oc));
      float4 w3=read_imagef(W,smp,(int2)(wb+3,oc));
      float4 a0=read_imagef(A,smp,(int2)((ax-2)*6+ic,ay));
      float4 a1=read_imagef(A,smp,(int2)((ax+2)*6+ic,ay));
      float4 a2=read_imagef(A,smp,(int2)((ax+6)*6+ic,ay));
      float4 a3=read_imagef(A,smp,(int2)((ax+10)*6+ic,ay));
      z0+=a0.x*w0+a0.y*w1+a0.z*w2+a0.w*w3;
      z1+=a1.x*w0+a1.y*w1+a1.z*w2+a1.w*w3;
      z2+=a2.x*w0+a2.y*w1+a2.z*w2+a2.w*w3;
      z3+=a3.x*w0+a3.y*w1+a3.z*w2+a3.w*w3;
    }
  }
  float4 b=read_imagef(B,smp,(int2)(oc,0)); int x=ix+(iy<<4);
  write_imagef(O,(int2)(x,oc),(float4)(z0.x+b.x,z1.x+b.x,z2.x+b.x,z3.x+b.x));
  write_imagef(O,(int2)(x+512,oc),(float4)(z0.y+b.y,z1.y+b.y,z2.y+b.y,z3.y+b.y));
  write_imagef(O,(int2)(x+1024,oc),(float4)(z0.z+b.z,z1.z+b.z,z2.z+b.z,z3.z+b.z));
  write_imagef(O,(int2)(x+1536,oc),(float4)(z0.w+b.w,z1.w+b.w,z2.w+b.w,z3.w+b.w));
}"""
  if getenv("OPENPILOT_FIRSTCONV_LOCAL"):
    firstconv_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void firstconv_fast(write_only image2d_t O, read_only image2d_t A,
                             read_only image2d_t W, read_only image2d_t B) {
  const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
  __local float4 la[8][7][7][4];
  int lx=get_local_id(0), lc=get_local_id(2), lid=lx+8*lc;
  int ix=get_global_id(0), iy=get_global_id(1), oc=get_global_id(2);
  float4 z0=(float4)(0), z1=(float4)(0), z2=(float4)(0), z3=(float4)(0);
  for (int ic=0; ic<6; ic++) {
    for (int p=lid; p<1568; p+=192) {
      int pos=p&3,q=p>>2,kx=q%7; q/=7; int ky=q%7,lxi=q/7;
      int ax=(get_group_id(0)*8+lxi)*16+kx, ay=iy*4+ky-2;
      la[lxi][ky][kx][pos]=read_imagef(A,smp,(int2)((ax-2+pos*4)*6+ic,ay));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ky=0; ky<7; ky++) for (int kx=0; kx<7; kx++) {
      int wb=ky*168+ic*28+kx*4;
      float4 w0=read_imagef(W,smp,(int2)(wb,oc));
      float4 w1=read_imagef(W,smp,(int2)(wb+1,oc));
      float4 w2=read_imagef(W,smp,(int2)(wb+2,oc));
      float4 w3=read_imagef(W,smp,(int2)(wb+3,oc));
      float4 a0=la[lx][ky][kx][0],a1=la[lx][ky][kx][1];
      float4 a2=la[lx][ky][kx][2],a3=la[lx][ky][kx][3];
      z0+=a0.x*w0+a0.y*w1+a0.z*w2+a0.w*w3;
      z1+=a1.x*w0+a1.y*w1+a1.z*w2+a1.w*w3;
      z2+=a2.x*w0+a2.y*w1+a2.z*w2+a2.w*w3;
      z3+=a3.x*w0+a3.y*w1+a3.z*w2+a3.w*w3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float4 b=read_imagef(B,smp,(int2)(oc,0)); int x=ix+(iy<<4);
  write_imagef(O,(int2)(x,oc),(float4)(z0.x+b.x,z1.x+b.x,z2.x+b.x,z3.x+b.x));
  write_imagef(O,(int2)(x+512,oc),(float4)(z0.y+b.y,z1.y+b.y,z2.y+b.y,z3.y+b.y));
  write_imagef(O,(int2)(x+1024,oc),(float4)(z0.z+b.z,z1.z+b.z,z2.z+b.z,z3.z+b.z));
  write_imagef(O,(int2)(x+1536,oc),(float4)(z0.w+b.w,z1.w+b.w,z2.w+b.w,z3.w+b.w));
}"""
  if getenv("OPENPILOT_FIRSTCONV_LOCALW"):
    firstconv_source = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void firstconv_fast(write_only image2d_t O, read_only image2d_t A,
                             read_only image2d_t W, read_only image2d_t B) {
  const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
  __local half4 lw[24][4][7][4];
  int lx=get_local_id(0), lc=get_local_id(2), lid=lx+8*lc;
  int ix=get_global_id(0), iy=get_global_id(1), oc=get_global_id(2);
  float4 z0=(float4)(0), z1=(float4)(0), z2=(float4)(0), z3=(float4)(0);
  for (int ic=0; ic<6; ic++) for (int yb=0; yb<7; yb+=4) {
    for (int p=lid; p<2688; p+=192) {
      int w=p&3,q=p>>2,kx=q%7; q/=7; int ky=q&3,c=q>>2;
      int wb=(yb+ky)*168+ic*28+kx*4+w;
      lw[c][ky][kx][w]=read_imageh(W,smp,(int2)(wb,c));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ko=0; ko<4 && yb+ko<7; ko++) for (int kx=0; kx<7; kx++) {
      int ky=yb+ko,ax=ix*16+kx,ay=iy*4+ky-2;
      float4 w0=convert_float4(lw[lc][ko][kx][0]);
      float4 w1=convert_float4(lw[lc][ko][kx][1]);
      float4 w2=convert_float4(lw[lc][ko][kx][2]);
      float4 w3=convert_float4(lw[lc][ko][kx][3]);
      float4 a0=read_imagef(A,smp,(int2)((ax-2)*6+ic,ay));
      float4 a1=read_imagef(A,smp,(int2)((ax+2)*6+ic,ay));
      float4 a2=read_imagef(A,smp,(int2)((ax+6)*6+ic,ay));
      float4 a3=read_imagef(A,smp,(int2)((ax+10)*6+ic,ay));
      z0+=a0.x*w0+a0.y*w1+a0.z*w2+a0.w*w3;
      z1+=a1.x*w0+a1.y*w1+a1.z*w2+a1.w*w3;
      z2+=a2.x*w0+a2.y*w1+a2.z*w2+a2.w*w3;
      z3+=a3.x*w0+a3.y*w1+a3.z*w2+a3.w*w3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float4 b=read_imagef(B,smp,(int2)(oc,0)); int x=ix+(iy<<4);
  write_imagef(O,(int2)(x,oc),(float4)(z0.x+b.x,z1.x+b.x,z2.x+b.x,z3.x+b.x));
  write_imagef(O,(int2)(x+512,oc),(float4)(z0.y+b.y,z1.y+b.y,z2.y+b.y,z3.y+b.y));
  write_imagef(O,(int2)(x+1024,oc),(float4)(z0.z+b.z,z1.z+b.z,z2.z+b.z,z3.z+b.z));
  write_imagef(O,(int2)(x+1536,oc),(float4)(z0.w+b.w,z1.w+b.w,z2.w+b.w,z3.w+b.w));
}"""
  if getenv("OPENPILOT_FIRSTCONV_HALF"):
    firstconv_source = "\n".join(line.replace("read_imagef(W", "convert_float4(read_imageh(W").replace("));", ")));" )
                                     if "read_imagef(W" in line else line for line in firstconv_source.splitlines())
  if getenv("OPENPILOT_FIRSTCONV_HALF_ACC"):
    firstconv_source = firstconv_source.replace("float4", "half4").replace("read_imagef(", "read_imageh(")
    firstconv_source = firstconv_source.replace("write_imagef(O", "write_imageh(O")
  firstconv_lib = dev.compiler.compile(firstconv_source)
  for call in targets10:
    template = call.src[0]
    program = template.replace(arg=replace(template.arg, name="firstconv_fast"),
                               src=template.src[:2]+(template.src[2].replace(arg=firstconv_source),
                                                     template.src[3].replace(arg=firstconv_lib)))
    replacements[call] = (program.call(*call.src[1:]),)

  softmax_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void softmax512(write_only image2d_t O, read_only image2d_t X) {
  __local float4 lm[32]; int l=get_local_id(0),g=get_group_id(0);
  float4 v0=read_imagef(X,smp,(int2)(l,g)),v1=read_imagef(X,smp,(int2)(l+32,g));
  float4 v2=read_imagef(X,smp,(int2)(l+64,g)),v3=read_imagef(X,smp,(int2)(l+96,g));
  lm[l]=(float4)(max(max(v0.x,v0.y),max(v0.z,v0.w)),max(max(v1.x,v1.y),max(v1.z,v1.w)),
                       max(max(v2.x,v2.y),max(v2.z,v2.w)),max(max(v3.x,v3.y),max(v3.z,v3.w)));
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int d=16;d;d>>=1) { if(l<d) lm[l]=max(lm[l],lm[l+d]); barrier(CLK_LOCAL_MEM_FENCE); }
  float4 m=lm[0];
  float4 e0=exp2((v0-(float4)(m.x))*(float4)(1.4426950408889634f));
  float4 e1=exp2((v1-(float4)(m.y))*(float4)(1.4426950408889634f));
  float4 e2=exp2((v2-(float4)(m.z))*(float4)(1.4426950408889634f));
  float4 e3=exp2((v3-(float4)(m.w))*(float4)(1.4426950408889634f));
  float4 one=(float4)(1);
  lm[l]=(float4)(dot(e0,one),dot(e1,one),dot(e2,one),dot(e3,one)); barrier(CLK_LOCAL_MEM_FENCE);
  for(int d=16;d;d>>=1) { if(l<d) lm[l]+=lm[l+d]; barrier(CLK_LOCAL_MEM_FENCE); }
  float4 inv=(float4)(1)/lm[0]; int x=l+(g>>5)*48,y=g&31;
  write_imagef(O,(int2)(x,y),e0*(float4)(inv.x)); write_imagef(O,(int2)(x+576,y),e1*(float4)(inv.y));
  write_imagef(O,(int2)(x+1152,y),e2*(float4)(inv.z)); write_imagef(O,(int2)(x+1728,y),e3*(float4)(inv.w));
}"""
  softmax_lib = dev.compiler.compile(softmax_source)
  softmax_replacements, softmax_remaps, softmax_skip = {}, {}, set()
  for i in range(len(batch)-2) if (getenv("OPENPILOT_FUSIONS") or getenv("OPENPILOT_SOFTMAX_FUSION")) else ():
    max_call, sum_call, epi_call = batch[i:i+3]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in (max_call, sum_call, epi_call)): continue
    if tuple(plain_name(x.src[0].arg.name) for x in (max_call, sum_call, epi_call)) != \
       ("r_384_4_32_4", "r_384_4_32_4n1", "E_32_12_32_4_4"): continue
    softmax = build_program(epi_call.src[0], "softmax512", softmax_source, softmax_lib, (384, 1, 1), (32, 1, 1),
                            ((dtypes.float, (32, 2304, 4)), (dtypes.float, (384, 128, 4))), (0,), (1,))
    safe_output = UOp.new_buffer("QCOM", epi_call.src[1].buffer.size, epi_call.src[1].dtype)
    safe_output.buffer.ensure_allocated()
    softmax_replacements[i] = (softmax.call(safe_output, max_call.src[2]),)
    softmax_remaps[i] = (epi_call.src[1], safe_output)
    softmax_skip.update((i+1, i+2))

  layernorm_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(16,1,1)))
__kernel void layernorm192(write_only image2d_t O, read_only image2d_t XR, read_only image2d_t XE, read_only image2d_t S) {
  __local float4 tmp[16]; int l=get_local_id(0), g=get_group_id(0); float4 sum=(float4)(0);
  for(int r=0;r<12;r++) sum+=read_imagef(XR,smp,(int2)(g+r*128,l));
  tmp[l]=sum; barrier(CLK_LOCAL_MEM_FENCE);
  if(l==0) { float4 z=(float4)(0); for(int i=0;i<16;i++) z+=tmp[i]; tmp[0]=z*(float4)(0.005208333333333333f); }
  barrier(CLK_LOCAL_MEM_FENCE); float4 mean=tmp[0], var=(float4)(0);
  for(int r=0;r<12;r++) { float4 d=read_imagef(XR,smp,(int2)(g+r*128,l))-mean; var+=d*d; }
  tmp[l]=var; barrier(CLK_LOCAL_MEM_FENCE);
  if(l==0) { float4 z=(float4)(0); for(int i=0;i<16;i++) z+=tmp[i];
    tmp[0]=rsqrt(z*(float4)(0.005208333333333333f)+(float4)(0.000001f)); }
  barrier(CLK_LOCAL_MEM_FENCE); float4 iv=tmp[0];
  for(int qoff=0;qoff<3;qoff++) {
    int q=l*3+qoff;
    float4 v0=(read_imagef(XE,smp,(int2)(g+0*128,q))-mean)*iv;
    float4 v1=(read_imagef(XE,smp,(int2)(g+1*128,q))-mean)*iv;
    float4 v2=(read_imagef(XE,smp,(int2)(g+2*128,q))-mean)*iv;
    float4 v3=(read_imagef(XE,smp,(int2)(g+3*128,q))-mean)*iv;
    float4 s=read_imagef(S,smp,(int2)(q,0));
    write_imagef(O,(int2)(q,g),(float4)(v0.x*s.x,v1.x*s.y,v2.x*s.z,v3.x*s.w));
    write_imagef(O,(int2)(q+48,g),(float4)(v0.y*s.x,v1.y*s.y,v2.y*s.z,v3.y*s.w));
    write_imagef(O,(int2)(q+96,g),(float4)(v0.z*s.x,v1.z*s.y,v2.z*s.z,v3.z*s.w));
    write_imagef(O,(int2)(q+144,g),(float4)(v0.w*s.x,v1.w*s.y,v2.w*s.z,v3.w*s.w));
  }
}"""
  layernorm_lib = dev.compiler.compile(layernorm_source)
  for i in range(len(batch)-2) if (getenv("OPENPILOT_FUSIONS") or getenv("OPENPILOT_LN192_FUSION")) else ():
    mean_call, var_call, epi_call = batch[i:i+3]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in (mean_call, var_call, epi_call)): continue
    if tuple(plain_name(x.src[0].arg.name) for x in (mean_call, var_call, epi_call)) != \
       ("r_128_16_4_12", "r_128_16_4_12n1", "E_128_48_4_4"): continue
    layernorm = build_program(epi_call.src[0], "layernorm192", layernorm_source, layernorm_lib, (128, 1, 1), (16, 1, 1),
                              ((dtypes.float, (128, 192, 4)), (dtypes.float, (16, 1536, 4)),
                               (dtypes.float, (48, 512, 4)), (dtypes.float, (1, 48, 4))), (0,), (1, 2, 3))
    indexed_replacements[i] = (layernorm.call(epi_call.src[1], mean_call.src[2], mean_call.src[2], epi_call.src[5]),)
    indexed_replacements[i+1] = indexed_replacements[i+2] = ()

  cached_ln192_threads = getenv("OPENPILOT_CACHED_LN192_THREADS", 0)
  if cached_ln192_threads:
    if cached_ln192_threads not in (16, 32, 64, 128):
      raise ValueError("OPENPILOT_CACHED_LN192_THREADS must be 16, 32, 64, or 128")
    cached_ln192_source = f"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({cached_ln192_threads},1,1)))
__kernel void layernorm192(write_only image2d_t O, read_only image2d_t XR, read_only image2d_t XE, read_only image2d_t S) {{
  __local float4 tmp[{cached_ln192_threads}]; int l=get_local_id(0), g=get_group_id(0); float4 sum=(float4)(0);
  for(int c=l;c<192;c+={cached_ln192_threads}) sum+=read_imagef(XR,smp,(int2)(g+(c>>4)*128,c&15));
  tmp[l]=sum; barrier(CLK_LOCAL_MEM_FENCE);
  for(int d={cached_ln192_threads//2};d;d>>=1) {{ if(l<d) tmp[l]+=tmp[l+d]; barrier(CLK_LOCAL_MEM_FENCE); }}
  float4 mean=tmp[0]*(float4)(0.005208333333333333f), var=(float4)(0);
  for(int c=l;c<192;c+={cached_ln192_threads}) {{ float4 d=read_imagef(XR,smp,(int2)(g+(c>>4)*128,c&15))-mean; var+=d*d; }}
  tmp[l]=var; barrier(CLK_LOCAL_MEM_FENCE);
  for(int d={cached_ln192_threads//2};d;d>>=1) {{ if(l<d) tmp[l]+=tmp[l+d]; barrier(CLK_LOCAL_MEM_FENCE); }}
  float4 iv=rsqrt(tmp[0]*(float4)(0.005208333333333333f)+(float4)(0.000001f));
  for(int q=l;q<48;q+={cached_ln192_threads}) {{
    float4 v0=(read_imagef(XE,smp,(int2)(g+0*128,q))-mean)*iv;
    float4 v1=(read_imagef(XE,smp,(int2)(g+1*128,q))-mean)*iv;
    float4 v2=(read_imagef(XE,smp,(int2)(g+2*128,q))-mean)*iv;
    float4 v3=(read_imagef(XE,smp,(int2)(g+3*128,q))-mean)*iv;
    float4 s=read_imagef(S,smp,(int2)(q,0));
    write_imagef(O,(int2)(q,g),(float4)(v0.x*s.x,v1.x*s.y,v2.x*s.z,v3.x*s.w));
    write_imagef(O,(int2)(q+48,g),(float4)(v0.y*s.x,v1.y*s.y,v2.y*s.z,v3.y*s.w));
    write_imagef(O,(int2)(q+96,g),(float4)(v0.z*s.x,v1.z*s.y,v2.z*s.z,v3.z*s.w));
    write_imagef(O,(int2)(q+144,g),(float4)(v0.w*s.x,v1.w*s.y,v2.w*s.z,v3.w*s.w));
  }}
}}"""
    cached_ln192_lib = dev.compiler.compile(cached_ln192_source)
    for call in batch:
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != "layernorm192": continue
      program = call.src[0]
      info = replace(program.arg, local_size=(cached_ln192_threads, 1, 1))
      refreshed = program.replace(arg=info, src=program.src[:2] +
                                  (program.src[2].replace(arg=cached_ln192_source), program.src[3].replace(arg=cached_ln192_lib)))
      replacements[call] = (call.replace(src=(refreshed,) + call.src[1:]),)

  layernorm384_source = """const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void layernorm384(write_only image2d_t O, read_only image2d_t X, read_only image2d_t S) {
  __local float4 lm[128], lv[128]; int l=get_local_id(0), g=get_group_id(0), base=g*384+l;
  float4 v0=(float4)(0),v1=(float4)(0),v2=(float4)(0),v3=(float4)(0);
  if (l<96) { v0=read_imagef(X,smp,(int2)(base,0)); v1=read_imagef(X,smp,(int2)(base+96,0));
              v2=read_imagef(X,smp,(int2)(base+192,0)); v3=read_imagef(X,smp,(int2)(base+288,0)); }
  float4 one=(float4)(1);
  lm[l]=l<96 ? (float4)(dot(v0,one),dot(v1,one),dot(v2,one),dot(v3,one)) : (float4)(0);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int d=64; d; d>>=1) { if (l<d) lm[l]+=lm[l+d]; barrier(CLK_LOCAL_MEM_FENCE); }
  float4 mean=lm[0]*(float4)(0.0026041666666666665f);
  float4 d0=v0-(float4)(mean.x),d1=v1-(float4)(mean.y),d2=v2-(float4)(mean.z),d3=v3-(float4)(mean.w);
  lv[l]=l<96 ? (float4)(dot(d0,d0),dot(d1,d1),dot(d2,d2),dot(d3,d3)) : (float4)(0);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int d=64; d; d>>=1) { if (l<d) lv[l]+=lv[l+d]; barrier(CLK_LOCAL_MEM_FENCE); }
  if (l<96) {
    float4 iv=rsqrt(lv[0]*(float4)(0.0026041666666666665f)+(float4)(0.000001f));
    float4 s=read_imagef(S,smp,(int2)(l,0));
    write_imagef(O,(int2)(base,0),d0*(float4)(iv.x)*s);
    write_imagef(O,(int2)(base+96,0),d1*(float4)(iv.y)*s);
    write_imagef(O,(int2)(base+192,0),d2*(float4)(iv.z)*s);
    write_imagef(O,(int2)(base+288,0),d3*(float4)(iv.w)*s);
  }
}"""
  layernorm384_lib = dev.compiler.compile(layernorm384_source)
  for i in range(len(batch)-2) if (getenv("OPENPILOT_FUSIONS") or getenv("OPENPILOT_LN384_FUSION")) else ():
    mean_call, var_call, epi_call = batch[i:i+3]
    if not all(x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM for x in (mean_call, var_call, epi_call)): continue
    if tuple(plain_name(x.src[0].arg.name) for x in (mean_call, var_call, epi_call)) != \
       ("r_32_16_4_6_4", "r_32_16_4_6_4n1", "E_32_96_4_4n1"): continue
    layernorm = build_program(epi_call.src[0], "layernorm384", layernorm384_source, layernorm384_lib, (32, 1, 1), (128, 1, 1),
                              ((dtypes.half, (1, 12288, 4)), (dtypes.float, (1, 12288, 4)),
                               (dtypes.float, (1, 96, 4))), (0,), (1, 2))
    indexed_replacements[i] = (layernorm.call(epi_call.src[1], mean_call.src[2], epi_call.src[5]),)
    indexed_replacements[i+1] = indexed_replacements[i+2] = ()

  # Fusing the reduction at max_call moves the epilogue write earlier. The cached memory plan can alias
  # epi_call's output with the softmax input, which is unsafe for a whole-reduction kernel. Keep the fused
  # result in a dedicated buffer and remap its consumers until the planned output buffer is next overwritten.
  flat_batch, active_remaps = [], {}
  for call_index, call in enumerate(batch):
    emitted_calls = () if call_index in softmax_skip else softmax_replacements.get(
      call_index, indexed_replacements.get(call_index, replacements.get(call, (call,))))
    for emitted in emitted_calls:
      if emitted.op is Ops.CALL and emitted.src[0].op is Ops.PROGRAM:
        outs = set(emitted.src[0].arg.outs)
        call_args = tuple(x if i in outs else active_remaps.get(x, x) for i, x in enumerate(emitted.src[1:]))
        emitted = emitted.replace(src=(emitted.src[0],)+call_args)
        for i in outs: active_remaps.pop(emitted.src[i+1], None)
      flat_batch.append(emitted)
    if call_index in softmax_remaps:
      old, new = softmax_remaps[call_index]
      active_remaps[old] = new
  flat_batch = tuple(flat_batch)
  new_outer = create_graph_call(list(flat_batch))
  jit.captured._linear = jit.captured._linear.substitute({outer: new_outer}, walk=True)
  jit.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  # Some target lists are deliberately cleared after their replacement mode is captured, so
  # recounting those lists here can misleadingly report zero even when programs were patched.
  print(f"replaced {len(replacements)} calls")


if __name__ == "__main__": main()

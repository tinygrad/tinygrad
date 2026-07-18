#!/usr/bin/env python3
"""Replace the large 2048x192x64 vision projection with a checked FP16 GEMM."""
import argparse
import pickle

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name
from extra.gemm.qcom_openpilot_graph import build_program

TARGET = "r_512_48_4_4_16_4"
M, N, K, PAD_N, STRIDE = 2048, 192, 64, 256, 1024

PACK = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void pack_large(__global half *O,read_only image2d_t A) {
  int t=get_global_id(0),row=t/16,k4=t-row*16,idx1=row>>2,block=row&3;
  int x=(idx1&15)*68+k4+block*17,y=idx1>>4;
  vstore4(read_imageh(A,smp,(int2)(x,y)),0,O+t*4);
}"""

EPI = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
inline float4 gelu(float4 v) {
  return ((float4)(1)/(1+exp2((v+(float4)(0.044708251953125f)*v*v*v)*(float4)(-2.3021129851685216f))))*v;
}
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void epi_large(write_only image2d_t O,read_only image2d_t B,__global half *C) {
  int t=get_global_id(0),row=t/48,col=t-row*48,idx1=row>>2,block=row&3;
  float4 z=convert_float4(vload4(0,C+row*1024+col*4));
  write_imagef(O,(int2)(col+block*48,idx1),gelu(z+read_imagef(B,smp,(int2)(col,0))));
}"""


def copyin(buffer, array: np.ndarray) -> None:
  raw = memoryview(np.ascontiguousarray(array)).cast("B")
  if hasattr(buffer, "copyin"):
    buffer.copyin(raw)
  else:
    buffer.copy_from(Buffer("PYTHON", buffer.size, buffer.dtype, opaque=raw))


def patch_model(model) -> int:
  outer = model.captured.linear.src[0]
  batch, replacements = list(outer.src[0].src[0].src), {}
  dev = Device["QCOM"]
  pack_lib, epi_lib = dev.compiler.compile_cached(PACK), dev.compiler.compile_cached(EPI)
  q8.M, q8.N, q8.K, q8.K4 = M, STRIDE, K, K//4
  envelope, image_offset, image_size, register_offset = get_envelope(dev, q8.make_donor_src8(4, 128))
  shader, hregs, _fregs, _ = q8.build_8x8_split_a_unroll_shader(
    dev, 128, k_unroll=8, b_coord_delay=0, fast_coords=True,
    prefetch_next_b=True, add256_store_mode="tight", high_a=True, split_low_pairs=True)
  gemm_lib = inject(envelope, image_offset, image_size, register_offset, shader, fregs=10, hregs=hregs)
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != TARGET:
      continue
    template = call.src[0]
    activation = UOp.new_buffer("QCOM", M*K, dtypes.half)
    activation.buffer.ensure_allocated()
    weight = UOp.new_buffer("QCOM", K*PAD_N, dtypes.half)
    weight.buffer.ensure_allocated()
    original = call.src[3].buffer.numpy().view(np.float16).reshape(48, 72, 4)
    packed_weight = np.zeros((K, PAD_N//4, 4), dtype=np.float16)
    packed_weight[:, :48] = original[:, :K].transpose(1, 0, 2)
    copyin(weight.buffer, packed_weight)
    temporary = UOp.new_buffer("QCOM", M*STRIDE, dtypes.half)
    temporary.buffer.ensure_allocated()
    pack = build_program(template, "pack_large", PACK, pack_lib, (M*(K//4)//128, 1, 1), (128, 1, 1),
                         ((dtypes.half, (M*K,)), (dtypes.half, (32, 1088, 4))), (0,), (1,))
    gemm = build_program(template, "gemm_h", "checked FP16 8x8 GEMM", gemm_lib,
                         (PAD_N//256, M//32, 1), (128, 1, 1),
                         ((dtypes.half, (M, K//4, 4)), (dtypes.half, (K, PAD_N//4, 4)),
                          (dtypes.half, (M*STRIDE,))), (2,), (0, 1))
    epi = build_program(template, "epi_large", EPI, epi_lib, (M*(N//4)//128, 1, 1), (128, 1, 1),
                        ((dtypes.half, (512, 192, 4)), (dtypes.half, (1, 48, 4)),
                         (dtypes.half, (M*STRIDE,))), (0,), (1, 2))
    replacements[index] = (pack.call(activation, call.src[2]), gemm.call(activation, weight, temporary),
                           epi.call(call.src[1], call.src[4], temporary))
  if replacements:
    new_batch = [item for index, call in enumerate(batch) for item in replacements.get(index, (call,))]
    model.captured._linear = model.captured.linear.substitute({outer: create_graph_call(new_batch)}, walk=True)
    model.captured.__dict__.pop("linear", None)
  return len(replacements)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()
  with open(args.input, "rb") as f:
    model = pickle.load(f)
  print("patched", patch_model(model))
  with open(args.output, "wb") as f:
    pickle.dump(model, f)


if __name__ == "__main__":
  main()

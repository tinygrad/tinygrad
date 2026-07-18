#!/usr/bin/env python3
"""Call-boundary oracle for the openpilot target-3 hand GEMM replacement."""
import os, pickle

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name


EPILOGUE3 = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(8,32,1)))
__kernel void epi3(write_only image2d_t O,__global float *S,__global float *B,__global half *C) {
  int col=get_global_id(0),y=get_global_id(1),row=y*4;
  for(int block=0;block<4;block++) {
    int n=col*4+block;
    float4 z=(float4)(C[(row+0)*2048+n],C[(row+1)*2048+n],C[(row+2)*2048+n],C[(row+3)*2048+n]);
    z=select((float4)(0),z,isgreater(z,(float4)(0)));
    write_imagef(O,(int2)(col+block*384,y),(float4)(*S)*z*z+(float4)(*B));
  }
}"""


def upload(arr: np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", arr.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(arr)).cast("B"))
  return ret


def read(buf: Buffer, shape: tuple[int, ...], dtype=np.float16) -> np.ndarray:
  ret = np.empty(np.prod(shape), dtype=dtype)
  buf.copyout(memoryview(ret).cast("B"))
  return ret.reshape(shape)


def report(name: str, got: np.ndarray, expected: np.ndarray) -> None:
  delta = np.abs(got.astype(np.float32)-expected.astype(np.float32))
  at = np.unravel_index(int(delta.argmax()), delta.shape)
  print(name, "max_abs", float(delta[at]), "mean_abs", float(delta.mean()), "at", at,
        "got", float(got[at]), "expected", float(expected[at]))


def main() -> None:
  dev = Device["QCOM"]
  with open("/data/openpilot_vision.pkl", "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  call = next(x for x in batch if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
              plain_name(x.src[0].arg.name) == "r_32_384_4_4_96_4")
  original = call.src[0]
  print("original", original.arg.global_size, original.arg.local_size, original.arg.aux[0])

  rng = np.random.default_rng(3)
  split = bool(int(os.getenv("SPLIT", "0")))
  activation = (rng.standard_normal((1, 12288, 4))*0.05).astype(np.float16)
  weight_original = np.array(call.src[4].buffer.numpy(), copy=True).reshape(384, 384, 4)
  weight_hand = weight_original.transpose(1, 0, 2).copy()
  if split: weight_hand = weight_hand[:, :256].copy()
  scale, bias = np.array([0.7], np.float32), np.array([-0.03], np.float32)
  output_shape = (32, 1536, 4)

  out_original = upload(np.zeros(output_shape, np.float16), dtypes.half)
  out_hand = upload(np.zeros(output_shape, np.float16), dtypes.half)
  sb, bb = upload(scale, dtypes.float), upload(bias, dtypes.float)
  ab, wob, whb = upload(activation, dtypes.half), upload(weight_original, dtypes.half), upload(weight_hand, dtypes.half)
  stride = 1024 if split else 2048
  temporary = upload(np.zeros(128*stride, np.float16), dtypes.half)

  original_prg = dev.runtime(plain_name(original.arg.name), original.src[3].arg, buf_dtypes=original.arg.aux[0])
  original_prg(out_original._buf, sb._buf, ab._buf, wob._buf, bb._buf,
               global_size=original.arg.global_size, local_size=original.arg.local_size, wait=True)

  q.M, q.N, q.K, q.K4 = 128, stride, 384, 96
  if int(os.getenv("COMPILER", "0")):
    hand_lib, _, _, _ = get_envelope(dev, q.make_direct_donor_src(4, 128))
  else:
    envelope, image_off, image_size, reg_off = get_envelope(dev, q.make_donor_src(4, 128))
    fast = not split or bool(int(os.getenv("FAST", "1")))
    shader, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True, donor_store=False,
                                   stable_bx=fast, stable_ay=fast,
                                   inc_coords=fast, persistent_coords=fast, first_sync_only=split and fast, k_unroll=4,
                                   b_first=fast, coord_delay=-1 if fast else 4,
                                   post_constant=bool(int(os.getenv("POST", "0"))), store_row_shift=11)
    hand_lib = inject(envelope, image_off, image_size, reg_off, shader, fregs=10, hregs=int(os.getenv("HREGS", "28")))
  hand = dev.runtime("gemm_h", hand_lib, buf_dtypes=[((0, dtypes.half, (128, 96, 4)),),
                     ((0, dtypes.half, (384, weight_hand.shape[1], 4)),), ((0, dtypes.half, None),)])
  gx = 2 if split else 3
  hand(ab._buf, whb._buf, temporary._buf, global_size=(gx, 8, 1), local_size=(128, 1, 1), wait=True)
  if split:
    hand_times = [hand(ab._buf, whb._buf, temporary._buf, global_size=(gx, 8, 1),
                       local_size=(128, 1, 1), wait=True)*1e3 for _ in range(20)]
    a = activation.reshape(32, 4, 96, 4).reshape(128, 384).astype(np.float32)
    expected_tmp = (a @ weight_hand.reshape(384, 1024).astype(np.float32)).astype(np.float16)
    got_tmp = read(temporary, (128, 1024))
    print("timing_ms", "gemm", min(hand_times))
    report("temporary_split", got_tmp, expected_tmp)
    return
  epi_lib = dev.compiler.compile(EPILOGUE3)
  epi = dev.runtime("epi3", epi_lib, buf_dtypes=[((0, dtypes.half, output_shape),), ((0, dtypes.float, (1,)),),
                    ((0, dtypes.float, (1,)),), ((0, dtypes.half, None),)])
  epi(out_hand._buf, sb._buf, bb._buf, temporary._buf,
      global_size=(48, 1, 1), local_size=(8, 32, 1), wait=True)
  hand_times = [hand(ab._buf, whb._buf, temporary._buf, global_size=(3, 8, 1),
                     local_size=(128, 1, 1), wait=True)*1e3 for _ in range(20)]
  epi_times = [epi(out_hand._buf, sb._buf, bb._buf, temporary._buf, global_size=(48, 1, 1),
                   local_size=(8, 32, 1), wait=True)*1e3 for _ in range(20)]
  print("timing_ms", "gemm", min(hand_times), "epi", min(epi_times), "total", min(hand_times)+min(epi_times))

  a = activation.reshape(32, 4, 96, 4).reshape(128, 384).astype(np.float32)
  b = weight_hand.reshape(384, 1536).astype(np.float32)
  expected_tmp = (a @ b).astype(np.float16)
  got_tmp = read(temporary, (128, stride))
  report("temporary_dense", got_tmp[:, :1536], expected_tmp)
  # Also diagnose whether each 512-column hand tile is correct independently.
  for tile in range(3): report(f"temporary_tile{tile}", got_tmp[:, tile*512:(tile+1)*512],
                              expected_tmp[:, tile*512:(tile+1)*512])

  z = np.maximum(expected_tmp.astype(np.float32), 0)
  expected_hand_layout = (scale[0]*z*z+bias[0]).astype(np.float16)
  expected_output = np.empty(output_shape, np.float16)
  for y in range(32):
    for block in range(4):
      expected_output[y, block*384:(block+1)*384] = expected_hand_layout[y*4:(y+1)*4, block::4].T
  got_original, got_hand = read(out_original, output_shape), read(out_hand, output_shape)
  report("original_cpu", got_original, expected_output)
  report("hand_cpu", got_hand, expected_output)
  report("original_hand", got_original, got_hand)


if __name__ == "__main__": main()

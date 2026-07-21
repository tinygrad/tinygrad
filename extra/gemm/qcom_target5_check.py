#!/usr/bin/env python3
"""Compare the original openpilot target-5 kernel and its hand replacement."""
import pickle, struct

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import patch_openpilot_target5_rpt3, plain_name
from extra.gemm.qcom_openpilot_graph import EPILOGUE2_FP32


def upload(arr: np.ndarray, dtype):
  ret = Buffer("QCOM", arr.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(arr)).cast("B"))
  return ret


def read(buf: Buffer, shape: tuple[int, ...]) -> np.ndarray:
  ret = np.empty(np.prod(shape), dtype=np.float32)
  buf.copyout(memoryview(ret).cast("B"))
  return ret.reshape(shape)


def main() -> None:
  rng = np.random.default_rng(0)
  residual = (rng.standard_normal((48, 512, 4))*0.1).astype(np.float32)
  weight = (rng.standard_normal((128, 384, 4))*0.1).astype(np.float16)
  activation = (rng.standard_normal((48, 384, 4))*0.1).astype(np.float16)
  out0, out1 = upload(np.zeros_like(residual), dtypes.float), upload(np.zeros_like(residual), dtypes.float)
  out2 = upload(np.zeros_like(residual), dtypes.float)
  rb, wb, ab = upload(residual, dtypes.float), upload(weight, dtypes.half), upload(activation, dtypes.half)
  dev = Device["QCOM"]

  with open("/data/openpilot_vision.pkl", "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  call = next(x for x in batch if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
              plain_name(x.src[0].arg.name) == "r_48_128_4_4_96_4")
  original = call.src[0]
  # Use the model's structured weight values; random weights can conceal layout-sensitive hazards.
  weight = np.array(call.src[3].buffer.numpy(), copy=True).reshape(128, 384, 4)
  wb.copyin(memoryview(weight).cast("B"))
  try:
    activation = np.load("/tmp/openpilot_target5_activation.npy").reshape(48, 384, 4)
    ab.copyin(memoryview(activation).cast("B"))
    residual = np.load("/tmp/openpilot_target5_residual.npy").reshape(48, 512, 4)
    rb.copyin(memoryview(residual).cast("B"))
  except FileNotFoundError: pass
  orig_prg = dev.runtime(plain_name(original.arg.name), original.src[3].arg, buf_dtypes=original.arg.aux[0])
  orig_prg(out0._buf, rb._buf, wb._buf, ab._buf, global_size=original.arg.global_size,
           local_size=original.arg.local_size, wait=True)
  patched_lib = bytearray(original.src[3].arg)
  image_off, image_size = struct.unpack_from("<I", patched_lib, 0xc0)[0], struct.unpack_from("<I", patched_lib, 0x100)[0]
  patched_lib[image_off:image_off+image_size] = patch_openpilot_target5_rpt3(patched_lib[image_off:image_off+image_size])
  patched_prg = dev.runtime(plain_name(original.arg.name), bytes(patched_lib), buf_dtypes=original.arg.aux[0])
  patched_prg(out2._buf, rb._buf, wb._buf, ab._buf, global_size=original.arg.global_size,
              local_size=original.arg.local_size, wait=True)

  q.M, q.N, q.K, q.K4 = 192, 1024, 384, 96
  env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_fp32(2, 128))
  shader, hregs, fregs, _ = q.build_4x8_fp32_low_shader(
    dev, 128, coord_delay=-1, sampler_per_texture=True, alu_order="kk_col_row",
    preload_b=True, batch_coords=True, interleaved_a=True)
  hand_lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  logical_w = weight.reshape(128, 4, 96, 4).transpose(2, 3, 0, 1).reshape(384, 128, 4).copy()
  tw, tmp = upload(logical_w, dtypes.half), upload(np.zeros(192*1024, np.float32), dtypes.float)
  hand = dev.runtime("gemm_f", hand_lib, buf_dtypes=[((0, dtypes.half, (48, 384, 4)),),
                     ((0, dtypes.half, (384, 128, 4)),), ((0, dtypes.float, None),)])
  hand(ab._buf, tw._buf, tmp._buf, global_size=(2, 12, 1), local_size=(128, 1, 1), wait=True)
  epi_lib = dev.compiler.compile(EPILOGUE2_FP32)
  epi = dev.runtime("epi2_fp32", epi_lib, buf_dtypes=[((0, dtypes.float, (48, 512, 4)),),
                    ((0, dtypes.float, (48, 512, 4)),), ((0, dtypes.float, None),)])
  epi(out1._buf, rb._buf, tmp._buf, global_size=(192, 1, 1), local_size=(128, 1, 1), wait=True)

  got0, got1, got2 = read(out0, residual.shape), read(out1, residual.shape), read(out2, residual.shape)
  a = activation.transpose(0, 2, 1).reshape(192, 384).astype(np.float32)
  w = logical_w.reshape(384, 512).astype(np.float32)
  y = a @ w
  expected = np.empty_like(residual)
  for row in range(192): expected[row//4, (row%4)*128:(row%4+1)*128] = y[row].reshape(128, 4)
  expected += residual
  for name, got in (("original", got0), ("hand", got1), ("patched", got2)):
    delta = np.abs(got-expected)
    print(name, "max_abs", float(delta.max()), "mean_abs", float(delta.mean()),
          "allclose", bool(np.allclose(got, expected, rtol=1e-4, atol=1e-4)))
  delta = np.abs(got0-got1)
  print("original_vs_hand", float(delta.max()), float(delta.mean()))
  print("original_vs_patched", float(np.abs(got0-got2).max()), float(np.abs(got0-got2).mean()))
  print("original_vs_residual", float(np.abs(got0-residual).max()), float(np.abs(got0-residual).mean()),
        "samples", got0[0, 0].tolist(), residual[0, 0].tolist(), expected[0, 0].tolist())
  try:
    saved = np.load("/tmp/openpilot_target5_baseline_output.npy").reshape(got0.shape)
    print("direct_vs_prefix_baseline", float(np.abs(got0-saved).max()), float(np.abs(got0-saved).mean()))
  except FileNotFoundError: pass


if __name__ == "__main__": main()

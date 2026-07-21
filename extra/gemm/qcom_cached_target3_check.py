#!/usr/bin/env python3
"""Compare the cached exact target-3 GEMM against the THREAD128 FP16 hand kernel."""
import pickle
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def upload(x:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def read(buf:Buffer, count:int, dtype) -> np.ndarray:
  ret = np.empty(count, dtype=dtype)
  buf.copyout(memoryview(ret).cast("B"))
  return ret


with open("/data/openpilot_p3_rpt245679.pkl", "rb") as f: model = pickle.load(f)
batch = model.captured.linear.src[0].src[0].src[0].src
call = next(x for x in batch if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
            plain_name(x.src[0].arg.name) == "gemm_h" and tuple(x.src[0].arg.global_size) == (12, 8, 1))
dev, rng = Device["QCOM"], np.random.default_rng(7)
a = upload((rng.standard_normal(128*384)*0.05).astype(np.float16), dtypes.half)
a_np = read(a, 128*384, np.float16).reshape(128, 384)
w_np = np.array(call.src[2].buffer.numpy(), copy=True).reshape(384, 384, 4).reshape(384, 1536)
exact_out = upload(np.zeros(128*2048, np.float32), dtypes.float)
hand_out = upload(np.zeros(128*2048, np.float16), dtypes.half)

exact = dev.runtime("gemm_h", call.src[0].src[3].arg, buf_dtypes=call.src[0].arg.aux[0])
exact(a._buf, call.src[2].buffer._buf, exact_out._buf,
      global_size=call.src[0].arg.global_size, local_size=call.src[0].arg.local_size, wait=True)

q.M, q.N, q.K, q.K4 = 128, 1536, 384, 96
env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(4, 128))
shader, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True,
  stable_bx=True, stable_ay=True, inc_coords=True, persistent_coords=True,
  first_sync_only=True, k_unroll=4, b_first=True, coord_delay=-1, stable_settle_delay=0,
  store_row_shift=11, image_store=True, high_inputs=True)
lib = inject(env, io, sz, ro, shader, fregs=10, hregs=48)
hand = dev.runtime("gemm_h", lib, buf_dtypes=[((0, dtypes.half, (128, 512, 4)),),
  ((0, dtypes.half, (128, 96, 4)),), ((1, dtypes.half, (384, 384, 4)),)])
hand(hand_out._buf, a._buf, call.src[2].buffer._buf,
     global_size=(3, 8, 1), local_size=(128, 1, 1), wait=True)

expected = read(exact_out, 128*2048, np.float32).reshape(128, 2048)[:, :1536]
got = read(hand_out, 128*2048, np.float16).reshape(128, 2048)[:, :1536].astype(np.float32)
delta = np.abs(got-expected)
cpu0 = a_np[0].astype(np.float32) @ w_np.astype(np.float32)
for name, value in (("exact", expected[0]), ("hand", got[0])):
  d_cpu = np.abs(value-cpu0)
  print(name+"_cpu0", "max_abs", float(d_cpu.max()), "mean_abs", float(d_cpu.mean()))
at = np.unravel_index(int(np.argmax(delta)), delta.shape)
print("exact_hand", "max_abs", float(delta[at]), "mean_abs", float(delta.mean()), "at", at,
      "got", float(got[at]), "expected", float(expected[at]))
for tile in range(3):
  d = np.abs(got[:, tile*512:(tile+1)*512]-expected[:, tile*512:(tile+1)*512])
  print("tile", tile, "max_abs", float(d.max()), "mean_abs", float(d.mean()))
print("timing_ms", min(hand(hand_out._buf, a._buf, call.src[2].buffer._buf,
  global_size=(3,8,1), local_size=(128,1,1), wait=True) for _ in range(20))*1e3)

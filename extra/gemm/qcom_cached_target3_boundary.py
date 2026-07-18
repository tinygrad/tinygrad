#!/usr/bin/env python3
"""Compare the exact cached target-3 GEMM with its graph replacement."""
import argparse, pickle

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def upload(x:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def read(buf:Buffer, count:int, dtype) -> np.ndarray:
  ret = np.empty(count, dtype=dtype)
  buf.copyout(memoryview(ret).cast("B"))
  return ret


def batch(model): return model.captured.linear.src[0].src[0].src[0].src


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("reference")
  parser.add_argument("candidate")
  args = parser.parse_args()
  with open(args.reference, "rb") as f: reference = pickle.load(f)
  with open(args.candidate, "rb") as f: candidate = pickle.load(f)
  ref_call = next(c for c in batch(reference) if c.op is Ops.CALL and c.src[0].op is Ops.PROGRAM and
                  plain_name(c.src[0].arg.name) == "gemm_h" and tuple(c.src[0].arg.global_size) == (12, 8, 1))
  cbatch = batch(candidate)
  cand_call = next(cbatch[i] for i in range(len(cbatch)-1) if cbatch[i].op is Ops.CALL and
                   cbatch[i].src[0].op is Ops.PROGRAM and plain_name(cbatch[i+1].src[0].arg.name) == "cached_epi3")
  rng = np.random.default_rng(7)
  a_np = (rng.standard_normal((128, 384))*0.05).astype(np.float16)
  a = upload(a_np, dtypes.half)
  ref_out = upload(np.zeros(128*2048, np.float32), dtypes.float)
  cand_out = upload(np.zeros(128*2048, np.float16), dtypes.half)
  dev = Device["QCOM"]
  ref_runtime = dev.runtime("ref", ref_call.src[0].src[3].arg, buf_dtypes=ref_call.src[0].arg.aux[0])
  cand_runtime = dev.runtime("cand", cand_call.src[0].src[3].arg, buf_dtypes=cand_call.src[0].arg.aux[0])
  ref_runtime(a._buf, ref_call.src[2].buffer._buf, ref_out._buf,
              global_size=ref_call.src[0].arg.global_size, local_size=ref_call.src[0].arg.local_size, wait=True)
  cand_runtime(a._buf, cand_call.src[2].buffer._buf, cand_out._buf,
               global_size=cand_call.src[0].arg.global_size, local_size=cand_call.src[0].arg.local_size, wait=True)
  ref = read(ref_out, 128*2048, np.float32).reshape(128, 2048)[:, :1536]
  got = read(cand_out, 128*2048, np.float16).reshape(128, 2048)[:, :1536].astype(np.float32)
  weight = np.asarray(ref_call.src[2].buffer.numpy()).reshape(384, 1536)
  cpu = a_np.astype(np.float32) @ weight.astype(np.float32)
  delta = np.abs(got-ref)
  at = np.unravel_index(int(delta.argmax()), delta.shape)
  print(f"max_abs={float(delta[at]):.9g} mean_abs={float(delta.mean()):.9g} at={at} "
        f"got={float(got[at]):.9g} reference={float(ref[at]):.9g}")
  print(f"weight_max={float(np.max(np.abs(weight))):.9g} cpu_max={float(np.max(np.abs(cpu))):.9g} "
        f"reference_max={float(np.max(np.abs(ref))):.9g} candidate_max={float(np.max(np.abs(got))):.9g}")


if __name__ == "__main__": main()

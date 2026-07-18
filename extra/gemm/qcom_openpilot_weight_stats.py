#!/usr/bin/env python3
"""Inspect dominant cached OpenPilot GEMM weight distributions."""
import argparse, pickle

import numpy as np

from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("--svd", default="", help="comma-separated GEMM indices for singular-value tail analysis")
  parser.add_argument("--compare", help="second pickle whose cached GEMM weights should be compared")
  args = parser.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  rows = []
  for call in batch:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or plain_name(call.src[0].arg.name) != "gemm_h": continue
    weight = np.asarray(call.src[2].buffer.numpy(), dtype=np.float32).reshape(-1)
    absw = np.abs(weight)
    scale = float(absw.max())
    rows.append((tuple(call.src[0].arg.global_size), scale, float(np.mean(weight == 0)),
                 *(float(np.mean(absw <= scale*x)) for x in (1/1024, 1/512, 1/256, 1/128, 1/64))))
  for i, row in enumerate(rows):
    print(f"{i:2d} gs={row[0]} max={row[1]:.7g} zero={row[2]:.5f} "
          f"near=[{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f},{row[7]:.4f}]")
  if not rows:
    produced, seen = set(), set()
    for call in batch:
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
      for arg_index, buf in enumerate(call.src[1:]):
        if arg_index in call.src[0].arg.outs or buf.op is not Ops.BUFFER or buf in produced or buf in seen or buf.buffer.size < 4096: continue
        seen.add(buf)
        values = np.asarray(buf.buffer.numpy())
        if not np.issubdtype(values.dtype, np.floating): continue
        absw = np.abs(values.astype(np.float32)); scale = float(absw.max())
        print(f"{plain_name(call.src[0].arg.name)} arg={arg_index} size={values.size} max={scale:.7g} "
              f"zero={float(np.mean(values == 0)):.6f} near={float(np.mean(absw <= scale/1024)):.6f}")
      for out in call.src[0].arg.outs: produced.add(call.src[out+1])
  if args.svd:
    selected = {int(x) for x in args.svd.split(",")}
    gemms = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
             plain_name(call.src[0].arg.name) == "gemm_h"]
    for i in sorted(selected):
      call = gemms[i]
      weight = np.asarray(call.src[2].buffer.numpy(), dtype=np.float32)
      matrix = weight.reshape((384, 1536) if tuple(call.src[0].arg.global_size) == (12, 8, 1) else (1536, 384))
      singular = np.linalg.svd(matrix, compute_uv=False)
      energy = np.cumsum(singular[::-1]**2)[::-1]
      total = energy[0]
      ranks = (32, 64, 96, 128, 192, 256, 320)
      tails = [float(np.sqrt(energy[r]/total)) if r < len(singular) else 0.0 for r in ranks]
      print(f"svd {i:2d} shape={matrix.shape} rel_frob_tail=" + ",".join(f"r{r}:{e:.5f}" for r, e in zip(ranks, tails)))
  if args.compare:
    with open(args.compare, "rb") as f: other = pickle.load(f)
    other_batch = other.captured.linear.src[0].src[0].src[0].src
    lhs = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and plain_name(call.src[0].arg.name) == "gemm_h"]
    rhs = [call for call in other_batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and plain_name(call.src[0].arg.name) == "gemm_h"]
    for i, (a, b) in enumerate(zip(lhs, rhs)):
      av, bv = np.asarray(a.src[2].buffer.numpy()), np.asarray(b.src[2].buffer.numpy())
      print(f"compare {i:2d} max={float(np.max(np.abs(av.astype(np.float32)-bv.astype(np.float32)))):.9g} equal={np.array_equal(av,bv)}")


if __name__ == "__main__": main()

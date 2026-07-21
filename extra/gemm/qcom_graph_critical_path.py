#!/usr/bin/env python3
"""Estimate a captured QCOM graph's data-dependency critical path from a call profile."""
import argparse, hashlib, pickle, re, sys

from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name

LINE = re.compile(r"^\s*[\d.]+ ms.*?total=\s*[\d.]+ ms (\S+?)(?: global=|$)")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("profile")
  ap.add_argument("--min-duration", type=float, default=.1)
  args = ap.parse_args()
  durations = {}
  profile = sys.stdin if args.profile == "-" else open(args.profile)
  for line in profile:
    if not (m := LINE.match(line)): continue
    key = m.group(1)
    durations[key] = float(line.split("ms", 1)[0])
  with open(args.model, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  finish, writer, records = {}, {}, []
  for index, call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    program = call.src[0]
    name = plain_name(program.arg.name)
    digest = hashlib.sha1(program.src[3].arg).hexdigest()[:8]
    key = f"{name}#{digest}"
    duration = durations.get(key, durations.get(name, 0.0))
    cid = len(records)
    deps = [(finish.get(writer.get(arg), 0.0), writer.get(arg)) for arg in call.src[1:]]
    start, pred = max(deps, default=(0.0, None), key=lambda x:x[0])
    finish[cid] = start+duration
    for out in program.arg.outs: writer[call.src[out+1]] = cid
    records.append((cid, index, key, duration, start, start+duration, pred))
  end = max(records, key=lambda x:x[5])
  chain, cur = [], end[0]
  by_call = {x[0]:x for x in records}
  while cur is not None:
    rec = by_call[cur]
    chain.append(rec)
    cur = rec[6]
  chain.reverse()
  print(f"profiled_total_ms={sum(x[3] for x in records):.3f} critical_path_ms={end[5]:.3f} "
        f"profiled_calls={sum(x[3] > 0 for x in records)}/{len(records)}")
  print(f"critical path (profiled calls >={args.min_duration} ms):")
  for _, index, key, duration, start, stop, _ in chain:
    if duration >= args.min_duration: print(f"{index:4d} {start:8.3f}->{stop:8.3f} {duration:7.3f} {key}")


if __name__ == "__main__": main()

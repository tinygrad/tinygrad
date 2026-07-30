#!/usr/bin/env python3
"""Remove byte-identical duplicate linear chains in the driving-vision head."""
import argparse, hashlib, pickle

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_openpilot_ir3 import plain_name

TARGETS = {"r_128_16_4_32_4", "r_256_4_128_4", "r_128_16_4_16_4"}


def dedupe_identical_calls(model, all_calls:bool=True) -> list[tuple[int, str]]:
  """Alias calls with identical programs, inputs, and byte-identical constants."""
  outer = model.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  produced:dict[UOp, UOp] = {}
  static_hash:dict[UOp, str] = {}
  seen:dict[tuple, tuple[UOp, ...]] = {}
  new_batch, removed = [], []

  def representative(buf:UOp) -> UOp:
    while buf in produced and produced[buf] is not buf: buf = produced[buf]
    return buf

  def content_hash(buf:UOp) -> str:
    if buf not in static_hash:
      static_hash[buf] = hashlib.sha256(memoryview(buf.buffer.numpy()).cast("B")).hexdigest()
    return static_hash[buf]

  for index, original in enumerate(batch):
    call = original.replace(src=tuple(representative(x) if x in produced else x for x in original.src))
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or (not all_calls and plain_name(call.src[0].arg.name) not in TARGETS):
      new_batch.append(call)
      if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM:
        for out_index in call.src[0].arg.outs: produced[original.src[out_index+1]] = call.src[out_index+1]
      continue
    program = call.src[0]
    output_indices = set(program.arg.outs)
    signature_args = []
    for arg_index, (before, after) in enumerate(zip(original.src[1:], call.src[1:])):
      if arg_index in output_indices: continue
      if before.op is Ops.PARAM:
        signature_args.append(("param", before.arg))
      elif before in produced:
        signature_args.append(("dynamic", representative(before)))
      else:
        signature_args.append((str(after.dtype), after.buffer.size, content_hash(after)))
    signature = (plain_name(program.arg.name), program.src[3].arg, tuple(signature_args))
    outputs = tuple(original.src[i+1] for i in program.arg.outs)
    if signature in seen:
      canonical_outputs = seen[signature]
      for output, canonical in zip(outputs, canonical_outputs): produced[output] = representative(canonical)
      removed.append((index, plain_name(program.arg.name)))
    else:
      new_batch.append(call)
      canonical_outputs = tuple(call.src[i+1] for i in program.arg.outs)
      seen[signature] = canonical_outputs
      for output, canonical in zip(outputs, canonical_outputs): produced[output] = canonical

  # Apply aliases to consumers which occur after the duplicate chains.
  new_batch = [call.replace(src=tuple(representative(x) if x in produced else x for x in call.src)) for call in new_batch]
  model.captured._linear = model.captured.linear.substitute({outer:create_graph_call(new_batch)}, walk=True)
  model.captured.__dict__.pop("linear", None)
  return removed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--all", action="store_true", help="deduplicate every program family, not only the head linears")
  args = parser.parse_args()
  with open(args.input, "rb") as f: model = pickle.load(f)
  removed = dedupe_identical_calls(model, args.all)
  with open(args.output, "wb") as f: pickle.dump(model, f)
  print(f"removed {len(removed)} duplicate head calls: {removed}")


if __name__ == "__main__": main()

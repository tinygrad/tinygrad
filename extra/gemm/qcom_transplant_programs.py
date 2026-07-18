#!/usr/bin/env python3
"""Transplant selected compiled QCOM programs between equivalent cached model graphs."""
import argparse, itertools, pickle
from dataclasses import replace

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def graph_batch(model): return model.captured.linear.src[0].src[0].src[0].src


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("base")
  parser.add_argument("donor")
  parser.add_argument("output")
  parser.add_argument("--map", action="append", required=True, metavar="BASE=DONOR")
  parser.add_argument("--adapt-dtypes", action="store_true",
                      help="bind donor image/buffer views using the base call's argument dtypes")
  args = parser.parse_args()
  with open(args.base, "rb") as f: model = pickle.load(f)
  with open(args.donor, "rb") as f: donor = pickle.load(f)

  existing_slots = [x.arg.slot for x in model.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1) + 1)
  batch, donor_batch = graph_batch(model), graph_batch(donor)
  replacements: dict[int, UOp] = {}
  for mapping in args.map:
    base_name, donor_name = mapping.split("=", 1)
    base_calls = [(i, x) for i, x in enumerate(batch) if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
                  plain_name(x.src[0].arg.name) == base_name]
    donor_calls = [x for x in donor_batch if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
                   plain_name(x.src[0].arg.name) == donor_name]
    if len(base_calls) != len(donor_calls):
      raise ValueError(f"{base_name} has {len(base_calls)} calls but {donor_name} has {len(donor_calls)}")
    for occurrence, ((index, base_call), donor_call) in enumerate(zip(base_calls, donor_calls)):
      bp, dp = base_call.src[0], donor_call.src[0]
      if len(base_call.src) != len(donor_call.src) or bp.arg.globals != dp.arg.globals or bp.arg.outs != dp.arg.outs:
        raise ValueError(f"incompatible call contract for {mapping} occurrence {occurrence}")
      base_sizes = tuple(x.buffer.size for x in base_call.src[1:])
      donor_sizes = tuple(x.buffer.size for x in donor_call.src[1:])
      base_dtypes = tuple(x.dtype for x in base_call.src[1:])
      donor_dtypes = tuple(x.dtype for x in donor_call.src[1:])
      if base_sizes != donor_sizes or (base_dtypes != donor_dtypes and not args.adapt_dtypes):
        raise ValueError(f"incompatible buffers for {mapping} occurrence {occurrence}: "
                         f"{base_sizes}/{base_dtypes} != {donor_sizes}/{donor_dtypes}")
      if args.adapt_dtypes:
        def adapt_aux(x):
          if isinstance(x, tuple) and len(x) == 3 and isinstance(x[0], int): return (x[0], base_dtypes[x[0]], x[2])
          return tuple(adapt_aux(y) for y in x) if isinstance(x, tuple) else x
        dp = dp.replace(arg=replace(dp.arg, aux=adapt_aux(dp.arg.aux)))
      replacements[index] = dp.call(*base_call.src[1:])
    print(f"{base_name} <- {donor_name}: {len(base_calls)} calls")

  outer = model.captured.linear.src[0]
  new_outer = create_graph_call([replacements.get(i, call) for i, call in enumerate(batch)])
  model.captured._linear = model.captured.linear.substitute({outer: new_outer}, walk=True)
  model.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(model, f)
  print(f"wrote {args.output} with {len(replacements)} transplanted calls")


if __name__ == "__main__": main()

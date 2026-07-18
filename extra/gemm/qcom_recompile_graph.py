#!/usr/bin/env python3
"""Recompile every source-backed PROGRAM in a captured QCOM graph.

This preserves the scheduled graph, static buffers, and mixed-precision
boundaries while allowing an apples-to-apples comparison of QCOM compilers.
"""
import argparse, itertools, pickle

from tinygrad import Device, Context
from tinygrad.uop.ops import Ops, UOp


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--compiler-device", default="QCOM:IR3")
  args = parser.parse_args()

  with open(args.input, "rb") as f: jit = pickle.load(f)
  existing_slots = [x.arg.slot for x in jit.captured.linear.toposort()
                    if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(existing_slots, default=-1) + 1)

  # The suffix selects a renderer/compiler target, not a physical device ID.
  # Open the base device while that target is active.
  with Context(DEV=args.compiler_device): compiler = Device[args.compiler_device.split(":", 1)[0]].compiler
  programs = [x for x in jit.captured.linear.toposort() if x.op is Ops.PROGRAM]
  replacements, binaries = {}, {}
  for number, program in enumerate(programs, 1):
    source = program.src[2].arg
    if source not in binaries:
      print(f"compiling source {len(binaries)+1}: {program.arg.name} ({len(source)} bytes)", flush=True)
      binaries[source] = compiler.compile_cached(source)
      print(f"compiled {len(binaries)} unique sources ({number}/{len(programs)} programs)", flush=True)
    replacements[program] = program.replace(src=program.src[:3] + (program.src[3].replace(arg=binaries[source]),))
  jit.captured._linear = jit.captured.linear.substitute(replacements, walk=True)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  print(f"wrote {args.output}: {len(programs)} programs, {len(binaries)} unique sources")


if __name__ == "__main__": main()

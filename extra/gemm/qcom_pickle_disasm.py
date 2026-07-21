#!/usr/bin/env python3
"""Disassemble one representative program family from a compiled-model pickle."""
import argparse, hashlib, pickle, struct

from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import disasm
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("family")
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--stop", type=int)
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--source", action="store_true")
  parser.add_argument("--meta", action="store_true")
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--calls", help="list graph calls in a start:stop index range")
  args = parser.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  if args.calls:
    start, stop = (int(x) for x in args.calls.split(":"))
    for i, call in enumerate(batch[start:stop], start):
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
      def desc(x):
        return f"slot={getattr(x.arg, 'slot', '?')} dtype={x.dtype} size={x.buffer.size}" if x.op is Ops.BUFFER else str(x.op)
      print(i, plain_name(call.src[0].arg.name), "outs", call.src[0].arg.outs,
            "args", [desc(x) for x in call.src[1:]])
    return
  programs = [call.src[0] for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
              plain_name(call.src[0].arg.name) == args.family]
  if args.list:
    for i, item in enumerate(programs): print(i, item.arg.global_size, hashlib.sha1(item.src[3].arg).hexdigest()[:8])
    return
  program = programs[args.index]
  if args.meta:
    from tinygrad.engine.realize import get_runtime
    runtime = get_runtime("QCOM", program)
    print("global", program.arg.global_size, "local", program.arg.local_size, "globals", program.arg.globals)
    print("aux", program.arg.aux)
    print("buf_offs", runtime.buf_offs, "tex", runtime.tex_cnt, "ibo", runtime.ibo_cnt, "nir", runtime.NIR)
    return
  if args.source:
    print(program.src[2].arg)
    return
  lib = program.src[3].arg
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  lines = [line for line in disasm(lib[image_off:image_off+image_size]).splitlines() if not line.rstrip().endswith(":")]
  stop = len(lines) if args.stop is None else args.stop
  for i, line in enumerate(lines[args.start:stop], args.start): print(f"{i:3d}: {line}")


if __name__ == "__main__": main()

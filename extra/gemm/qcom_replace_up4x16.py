#!/usr/bin/env python3
"""Replace cached openpilot FFN-up GEMMs with the random-checked 4x16 kernel."""
import argparse, itertools, pickle
from dataclasses import replace

from tinygrad import Device, dtypes
from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def build_program(template:UOp, lib:bytes) -> UOp:
  specs = ((dtypes.float, (128, 512, 4)), (dtypes.half, (128, 96, 4)), (dtypes.half, (384, 384, 4)))
  aux = (tuple(((i, dtype, shape),) for i, (dtype, shape) in enumerate(specs)),)
  # The QCOM ELF argument parser locates descriptors using the embedded symbol name length.
  # Keep the donor's `gemm_h` name or its image arguments are parsed as empty.
  info = replace(template.arg, name="gemm_h", global_size=(3, 8, 1), local_size=(128, 1, 1),
                 globals=(0, 1, 2), outs=(0,), ins=(1, 2), aux=aux)
  return template.replace(arg=info, src=template.src[:2] +
                          (template.src[2].replace(arg="random-checked separate-bank 4x16 FP16 GEMM"),
                           template.src[3].replace(arg=lib)))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args = parser.parse_args()
  with open(args.input, "rb") as f: jit = pickle.load(f)
  slots = [x.arg.slot for x in jit.captured.linear.toposort()
           if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(slots, default=-1) + 1)

  outer = jit.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  targets = [call for call in batch if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
             plain_name(call.src[0].arg.name) == "gemm_h" and tuple(call.src[0].arg.global_size) == (12, 8, 1)]
  if len(targets) != 18: raise RuntimeError(f"expected 18 FFN-up GEMMs, found {len(targets)}")

  q.M, q.N, q.K, q.K4 = 128, 1536, 384, 96
  dev = Device["QCOM"]
  envelope, image_off, image_size, reg_off = get_envelope(dev, q.make_direct_image_donor_src(4, 128))
  shader, _ = q.build_4xn_shader(dev, 128, ncols=4, direct=True, compact_acc=True, image_store=True,
    stable_bx=True, stable_ay=True, inc_coords=True, persistent_coords=True, first_sync_only=True,
    k_unroll=4, b_first=True, coord_delay=-1, separate_b_coords=True)
  lib = inject(envelope, image_off, image_size, reg_off, shader, fregs=10, hregs=28, mergedregs=False)
  program = build_program(targets[0].src[0], lib)
  replacements = {call: program.call(call.src[3], call.src[1], call.src[2]) for call in targets}
  new_batch = [replacements.get(call, call) for call in batch]
  new_outer = create_graph_call(new_batch)
  jit.captured._linear = jit.captured.linear.substitute({outer:new_outer}, walk=True)
  jit.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  print(f"replaced {len(targets)} FFN-up GEMMs")


if __name__ == "__main__": main()

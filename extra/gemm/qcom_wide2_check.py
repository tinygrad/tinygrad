#!/usr/bin/env python3
"""Check the first float-input target-2 hand GEMM against its CPU matrix product."""
import argparse, pickle, re

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs, create_graph_call
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp


def plain(name: str) -> str: return re.sub(r"\x1b\[[0-9;]*m", "", name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("corpus")
  ap.add_argument("--reference")
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--case", type=int, default=9)
  args = ap.parse_args()
  with open(args.model, "rb") as f: model = pickle.load(f)
  corpus = np.load(args.corpus)
  inputs = {name: Tensor(corpus[f"case{args.case}:input:{name}"].astype(np.dtype(dtype.fmt), copy=False), device=device).realize()
            for name, (_view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info)}
  input_uops, var_vals = _prepare_jit_inputs((), inputs)[:2]
  batch = model.captured.linear.src[0].src[0].src[0].src
  indices = [i for i, call in enumerate(batch) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
             ((plain(call.src[0].arg.name) == "gemm_f" and tuple(call.src[0].arg.global_size) == (2, 12, 1)) or
              (plain(call.src[0].arg.name) == "gemm_h" and tuple(call.src[0].arg.global_size) in ((1, 12, 1), (4, 12, 1))))]
  if not args.all: indices = indices[:1]
  candidate_output = None
  for index in indices:
    run_linear(UOp(Ops.LINEAR, src=(create_graph_call(batch[:index]),)), var_vals,
               input_uops=input_uops, jit=True, wait=True)
    hand, epi = batch[index:index+2]
    pack_b_source = np.array(batch[index-2].src[2].buffer.numpy(), copy=True).reshape(128, 768, 4)
    activation_raw = np.array(hand.src[1].buffer.numpy(), copy=True)
    activation = (activation_raw.reshape(48, 768, 4).transpose(0, 2, 1).reshape(192, 768)
                  if plain(hand.src[0].arg.name) == "gemm_f" else activation_raw.reshape(192, 768))
    weight = np.array(hand.src[2].buffer.numpy(), copy=True).reshape(768, 512)
    run_linear(UOp(Ops.LINEAR, src=(hand,)), var_vals, input_uops=input_uops, jit=True, wait=True)
    temporary = np.array(hand.src[3].buffer.numpy(), copy=True).reshape(192, 1024)[:, :512]
    run_linear(UOp(Ops.LINEAR, src=(epi,)), var_vals, input_uops=input_uops, jit=True, wait=True)
    candidate_output = np.array(epi.src[1].buffer.numpy(), copy=True).reshape(48, 4, 128, 4).reshape(192, 512)
    expected = activation @ weight
    delta = np.abs(expected-temporary)
    print("index", index, "activation", activation.shape, "weight", weight.shape,
          "temp_max_abs", float(delta.max()), "temp_mean_abs", float(delta.mean()),
          "temp_worst", np.unravel_index(int(delta.argmax()), delta.shape),
          "activation_max", float(np.abs(activation).max()), "weight_max", float(np.abs(weight).max()),
          "expected_max", float(np.abs(expected).max()), "temporary_max", float(np.abs(temporary).max()),
          "output_max", float(np.abs(np.array(epi.src[1].buffer.numpy(), copy=False)).max()))
  if args.reference:
    with open(args.reference, "rb") as f: reference = pickle.load(f)
    ref_batch = reference.captured.linear.src[0].src[0].src[0].src
    ref_index = next(i for i, call in enumerate(ref_batch) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and
                     plain(call.src[0].arg.name) == "r_48_128_4_4_192_4")
    run_linear(UOp(Ops.LINEAR, src=(create_graph_call(ref_batch[:ref_index+1]),)), var_vals,
               input_uops=input_uops, jit=True, wait=True)
    call = ref_batch[ref_index]
    ref_activation = np.array(call.src[4].buffer.numpy(), copy=True).reshape(48, 768, 4).transpose(0, 2, 1).reshape(192, 768)
    original = np.array(call.src[3].buffer.numpy(), copy=True).reshape(128, 4, 192, 4)
    ref_weight = original.transpose(2, 3, 0, 1).reshape(768, 512)
    ref_output = np.array(call.src[1].buffer.numpy(), copy=True).reshape(48, 4, 128, 4).reshape(192, 512)
    ref_residual = np.array(call.src[2].buffer.numpy(), copy=True).reshape(48, 4, 128, 4).reshape(192, 512)
    ref_expected = ref_activation @ ref_weight + ref_residual
    ref_delta = np.abs(ref_expected-ref_output)
    assert candidate_output is not None
    print("reference_index", ref_index, "reference_cpu", float(ref_delta.max()), float(ref_delta.mean()),
          "candidate_vs_reference", float(np.abs(candidate_output-ref_output).max()),
          "weight_copy", float(np.abs(weight-ref_weight).max()), "activation_copy", float(np.abs(activation-ref_activation).max()),
          "reference_matmul_max", float(np.abs(ref_expected-ref_residual).max()),
          "candidate_effect_max", float(np.abs(candidate_output-ref_residual).max()))
    packed = weight.reshape(768, 128, 4)
    expected_packed = pack_b_source.reshape(128, 4, 192, 4).transpose(2, 3, 0, 1).reshape(768, 128, 4)
    print("pack_expected", float(np.abs(packed-expected_packed).max()),
          "packed_head", packed[:8, 0, 0].tolist(), "expected_head", expected_packed[:8, 0, 0].tolist())
    for k in range(4):
      value = packed[k, 0, 0]
      nearest = np.unravel_index(int(np.abs(pack_b_source-value).argmin()), pack_b_source.shape)
      print("packed_probe", k, float(value), nearest, float(pack_b_source[nearest]))
    for k in range(8):
      errs = np.abs(pack_b_source-packed[k, :, :][None, :, :].transpose(1, 0, 2)).mean(axis=(0, 2))
      xs = np.argsort(errs)[:3]
      print("packed_xmatch", k, [(int(x), float(errs[x])) for x in xs])


if __name__ == "__main__": main()

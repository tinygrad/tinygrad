#!/usr/bin/env python3
"""Rapidly measure ONNX output sensitivity to FP16-rounded initializers."""
import argparse, copy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("corpus")
  ap.add_argument("--case", type=int, default=9)
  ap.add_argument("--chunks", type=int, default=8)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--stop", type=int)
  ap.add_argument("--list", action="store_true")
  args = ap.parse_args()

  model = onnx.load(args.model)
  consumers: dict[str, list[str]] = {}
  for node in model.graph.node:
    for name in node.input: consumers.setdefault(name, []).append(node.op_type)
  initializers = [(init, numpy_helper.to_array(init)) for init in model.graph.initializer]
  selected = [(init, arr) for init, arr in initializers if arr.dtype == np.float32 and arr.ndim >= 2 and
              any(op in {"Conv", "Gemm", "MatMul"} for op in consumers.get(init.name, []))]
  if args.list:
    for i, (init, arr) in enumerate(selected): print(i, init.name, arr.shape, consumers.get(init.name))

  # Listing an initializer as a graph input lets one ORT session override it at run time.
  known_inputs = {x.name for x in model.graph.input}
  for init, _ in selected:
    if init.name not in known_inputs:
      model.graph.input.append(copy.deepcopy(onnx.helper.make_tensor_value_info(init.name, init.data_type, init.dims)))
  session_options = ort.SessionOptions()
  session_options.log_severity_level = 3
  session = ort.InferenceSession(model.SerializeToString(), session_options, providers=["CPUExecutionProvider"])
  corpus = np.load(args.corpus)
  feeds = {spec.name: corpus[f"case{args.case}:input:{spec.name}"] for spec in session.get_inputs()
           if f"case{args.case}:input:{spec.name}" in corpus}
  expected = corpus[f"case{args.case}:output"].astype(np.float32)

  def check(indices: list[int]) -> tuple[float, float]:
    overrides = {selected[i][0].name: selected[i][1].astype(np.float16).astype(np.float32) for i in indices}
    got = session.run(None, feeds | overrides)[0].astype(np.float32)
    delta = np.abs(expected.reshape(got.shape)-got)
    return float(delta.max()), float(delta.mean())

  scan = list(range(args.start, len(selected) if args.stop is None else args.stop))
  print(f"selected={len(selected)} scan={scan[0]}..{scan[-1]} baseline={check([])} scan_error={check(scan)}")
  for chunk in np.array_split(np.asarray(scan), args.chunks):
    ids = [int(x) for x in chunk]
    print(f"range={ids[0]}..{ids[-1]} count={len(ids)} error={check(ids)}")


if __name__ == "__main__": main()

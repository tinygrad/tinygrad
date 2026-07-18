#!/usr/bin/env python3
"""Compare baseline and replacement at the first target-5 boundary on a real model input."""
import pickle

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def load(path):
  with open(path, "rb") as f: return pickle.load(f)


def inputs_for(model):
  corpus = np.load("/tmp/openpilot_onnx_scale8_5seed.npz")
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    inputs[name] = Tensor(corpus[f"case0:input:{name}"].astype(np.dtype(dtype.fmt), copy=False), device=device).realize()
  return _prepare_jit_inputs((), inputs)[:2]


def batch(model): return model.captured.linear.src[0].src[0].src[0].src


def main() -> None:
  base = load("/data/openpilot_target1_fp32hand.pkl")
  cand = load("/data/openpilot_target1_wide5_fp32_v4.pkl")
  input_uops, var_vals = inputs_for(base)
  bb = batch(base)
  bi = next(i for i, x in enumerate(bb) if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
            plain_name(x.src[0].arg.name) == "r_48_128_4_4_96_4")
  run_linear(UOp(Ops.LINEAR, src=tuple(bb[:bi+1])), var_vals, input_uops=input_uops, jit=True, wait=True)
  bo = np.array(bb[bi].src[1].buffer.numpy(), copy=True)
  ba = np.array(bb[bi].src[4].buffer.numpy(), copy=True)
  br = np.array(bb[bi].src[2].buffer.numpy(), copy=True)

  cb = batch(cand)
  ci = next(i for i, x in enumerate(cb) if x.op is Ops.CALL and x.src[0].op is Ops.PROGRAM and
            plain_name(x.src[0].arg.name) == "gemm_f" and x.src[0].arg.aux[0][0][0][2] == (48, 384, 4))
  assert plain_name(cb[ci+1].src[0].arg.name) == "epi2_fp32"
  run_linear(UOp(Ops.LINEAR, src=tuple(cb[:ci+2])), var_vals, input_uops=input_uops, jit=True, wait=True)
  co = np.array(cb[ci+1].src[1].buffer.numpy(), copy=True)
  ca = np.array(cb[ci].src[1].buffer.numpy(), copy=True)
  cr = np.array(cb[ci+1].src[2].buffer.numpy(), copy=True)
  delta = np.abs(bo-co)
  print("indices", bi, ci, "max_abs", float(delta.max()), "mean_abs", float(delta.mean()),
        "allclose", bool(np.allclose(bo, co, rtol=1e-5, atol=1e-5)))
  print("samples", bo[:8].tolist(), co[:8].tolist())
  print("activation", float(np.abs(ba-ca).max()), "residual", float(np.abs(br-cr).max()))
  print("activation_stats", float(ba.min()), float(ba.max()), float(np.mean(np.abs(ba))),
        "residual_stats", float(br.min()), float(br.max()))
  for name, u in (("activation", cb[ci].src[1]), ("temporary", cb[ci].src[3]),
                  ("output", cb[ci+1].src[1]), ("residual", cb[ci+1].src[2])):
    b = u.buffer._buf
    print(name, hex(b.va_addr), b.size)
  tmp = np.array(cb[ci].src[3].buffer.numpy(), copy=True).reshape(192, 1024)[:, :512]
  w = np.array(cb[ci].src[2].buffer.numpy(), copy=True).reshape(384, 512).astype(np.float32)
  a = ca.reshape(48, 384, 4).transpose(0, 2, 1).reshape(192, 384).astype(np.float32)
  td = np.abs(tmp-a@w)
  print("temporary_cpu", float(td.max()), float(td.mean()), np.unravel_index(td.argmax(), td.shape))
  for name, u in (("base_output", bb[bi].src[1]), ("base_residual", bb[bi].src[2]),
                  ("base_weight", bb[bi].src[3]), ("base_activation", bb[bi].src[4])):
    b = u.buffer._buf
    print(name, hex(b.va_addr), b.size)
  np.save("/tmp/openpilot_target5_activation.npy", ba)
  np.save("/tmp/openpilot_target5_residual.npy", br)
  np.save("/tmp/openpilot_target5_baseline_output.npy", bo)


if __name__ == "__main__": main()

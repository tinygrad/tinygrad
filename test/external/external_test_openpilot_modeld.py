#!/usr/bin/env python3
# End-to-end test of openpilot's modeld compile entrypoint.
# Unlike examples/openpilot/compile3.py, this downloads the openpilot model and runs
# openpilot/selfdrive/modeld/compile_modeld.py (same script openpilot's SCons build runs),
# which builds the warp + policy JITs and asserts jit/pickle-roundtrip correctness internally.
# Additionally this checks exact kernel counts per jit and numerics vs onnxruntime (SELFTEST in compile3).
import os, sys, runpy, hashlib, argparse
from pathlib import Path

import numpy as np

TINYGRAD_ROOT = Path(__file__).resolve().parents[2]

# driving_supercombo.onnx from openpilot master (git-lfs, URL is the sha256)
MODEL_SHA256 = "659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
MODEL_URL = f"https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/{MODEL_SHA256}"

# args mirrored from openpilot/selfdrive/modeld/SConscript
# (MEDMODEL_INPUT_SIZE, CAMERA_CONFIGS, MODEL_RUN_FREQ // MODEL_CONTEXT_FREQ)
COMPILE_ARGS = ["--model-size", "512x256",
                "--camera-resolutions", "1928x1208", "1344x760",
                "--frame-skip", "4"]

# exact kernel counts per jit, keyed by the renderer/backend. override any of them with EXPECTED_KERNELS_{TAG}
EXPECTED_KERNELS = {"CUDARenderer": {"run_policy": 166, "(1928, 1208)": 7, "(1344, 760)": 7},
                    "CPULLVMRenderer": {"run_policy": 168, "(1928, 1208)": 7, "(1344, 760)": 7}}
EXPECTED_KERNELS_TAGS = {"run_policy": "RUN_POLICY", (1928, 1208): "WARP_1928_1208", (1344, 760): "WARP_1344_760"}
# fp16 has no stable reference: onnxruntime 1.27 and 1.29 already differ from each other, so this is a coarse
# sanity gate against structural breakage, not a tight numerics check. measured diff/threshold margin is ~0.6
ATOL, RTOL = 2.5, 0.2

def download_model() -> Path:
  from tinygrad import fetch
  path = fetch(MODEL_URL, name="driving_supercombo.onnx")
  h = hashlib.sha256(path.read_bytes()).hexdigest()
  assert h == MODEL_SHA256, f"model hash mismatch: {h} != {MODEL_SHA256}"
  print(f"downloaded model {path} ({path.stat().st_size/1e6:.2f} MB), sha256 verified")
  return path

def count_kernels(jit) -> int:
  from tinygrad.uop.ops import Ops
  return sum(1 for u in jit.captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
             if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM)

def test_kernel_counts(out):
  from tinygrad import Device, getenv
  counts = {k: count_kernels(out[k]) for k in EXPECTED_KERNELS_TAGS}
  renderer = type(Device[Device.DEFAULT].renderer).__name__
  print(f"kernel counts on {Device.DEFAULT} ({renderer}): {counts}")
  expected = EXPECTED_KERNELS.get(renderer, {})
  for key, tag in EXPECTED_KERNELS_TAGS.items():
    want = getenv(f"EXPECTED_KERNELS_{tag}", expected.get(key, -1))
    if want != -1: assert counts[key] == want, f"different kernels in {key}! {counts[key]=}, {want=}"

def test_vs_onnx(out, model_runner, onnx_file, atol, rtol):
  import onnx, onnxruntime as ort
  rng = np.random.default_rng(42)
  input_shapes = {k: tuple(s if isinstance(s, int) else 1 for s in shp) for k, shp in out['metadata']['input_shapes'].items()}
  def rand_input(k, shp):  # roughly in-distribution keeps activation magnitudes (and fp16 noise) sane
    if k in ('img', 'big_img'): return rng.integers(0, 256, shp).astype(np.float32)          # warped camera frames are uint8
    if k == 'traffic_convention': return np.eye(1, 2, -1, dtype=np.float32).reshape(shp)  # one-hot
    return (0.1 * rng.standard_normal(shp)).astype(np.float32)
  inputs = {k: rand_input(k, shp) for k, shp in input_shapes.items()}

  from tinygrad import Tensor
  from tinygrad.dtype import _to_np_dtype as to_np_dtype
  dtypes = {name: spec.dtype for name, spec in model_runner.graph_inputs.items()}
  tinygrad_out = next(iter(model_runner({k: Tensor(inputs[k].astype(to_np_dtype(dtypes[k]))) for k in sorted(inputs)}).values()))

  onnx_model = onnx.load(onnx_file)
  session = ort.InferenceSession(onnx_file)
  ort_dtypes = {x.name: np.dtype(x.type.replace('tensor(', '').replace(')', '')) for x in session.get_inputs()}
  ort_out = session.run([onnx_model.graph.output[0].name], {k: inputs[k].astype(ort_dtypes[k]) for k in inputs})

  tg_np = tinygrad_out.cast('float32').numpy()
  diff = np.abs(ort_out[0].reshape(tg_np.shape) - tg_np)
  print(f"max diff vs onnxruntime: {diff.max():.6f} (mean {diff.mean():.6f})")
  flat = np.argsort(diff.flatten())[::-1][:4]
  print(f"worst diffs (idx, ort, tinygrad): {[(int(i), float(ort_out[0].flat[i]), float(tg_np.flat[i])) for i in flat]}")
  margin = diff / (atol + rtol * np.abs(ort_out[0].reshape(diff.shape)))
  i = int(np.argmax(margin))
  print(f"worst diff/threshold: {margin.max():.2f} (must be < 1), at {i}: ort={float(ort_out[0].flat[i])}, tinygrad={float(tg_np.flat[i])}")
  np.testing.assert_allclose(ort_out[0].reshape(tg_np.shape), tg_np, atol=atol, rtol=rtol)
  print("test vs onnx passed")

def main():
  from tinygrad import getenv
  p = argparse.ArgumentParser()
  p.add_argument("--openpilot-root", type=Path,
                 default=Path(os.getenv("OPENPILOT_ROOT", TINYGRAD_ROOT.parent)),
                 help="repo root containing the openpilot/ package (default: sibling of tinygrad_repo)")
  p.add_argument("--output", type=Path, default=Path("/tmp/driving_tinygrad_test.pkl"))
  args = p.parse_args()

  compile_script = args.openpilot_root / "openpilot/selfdrive/modeld/compile_modeld.py"
  assert compile_script.is_file(), f"{compile_script} not found, set --openpilot-root/OPENPILOT_ROOT"

  model = download_model()

  # run the production entrypoint in-process so the compiled JITs can be introspected below
  sys.path.insert(0, str(args.openpilot_root))
  argv, sys.argv = sys.argv, [str(compile_script), "--onnx", str(model), "--output", str(args.output), *COMPILE_ARGS]
  try: ns = runpy.run_path(str(compile_script), run_name="__main__")
  finally: sys.argv = argv
  out, model_runner = ns['out'], ns['model_runner']

  test_kernel_counts(out)
  if getenv("SELFTEST", 1): test_vs_onnx(out, model_runner, model, ATOL, RTOL)

  assert args.output.is_file() and args.output.stat().st_size > 0, f"missing output {args.output}"
  from openpilot.selfdrive.modeld.helpers import load_oob
  with open(args.output, "rb") as f: out_loaded = load_oob(f)
  assert {'metadata', 'run_policy', (1928, 1208), (1344, 760)} <= set(out_loaded), f"unexpected pickle keys: {set(out_loaded)}"
  print(f"PASS: compiled pickle at {args.output} ({args.output.stat().st_size/1e6:.2f} MB), "
        f"model output {out['metadata']['output_shapes']}")

if __name__ == "__main__":
  main()

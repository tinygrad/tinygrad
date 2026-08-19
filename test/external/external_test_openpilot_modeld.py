#!/usr/bin/env python3
# End-to-end test of openpilot's real modeld compile entrypoint.
# Unlike examples/openpilot/compile3.py, this downloads the openpilot model and runs
# openpilot/selfdrive/modeld/compile_modeld.py (same script openpilot's SCons build runs),
# which builds the warp + policy JITs and asserts jit/pickle-roundtrip correctness internally.
import os, sys, hashlib, argparse, subprocess
from pathlib import Path

TINYGRAD_ROOT = Path(__file__).resolve().parents[2]

# driving_supercombo.onnx from openpilot master (git-lfs, URL is the sha256)
MODEL_SHA256 = "659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
MODEL_URL = f"https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/{MODEL_SHA256}"

# args mirrored from openpilot/selfdrive/modeld/SConscript
# (MEDMODEL_INPUT_SIZE, CAMERA_CONFIGS, MODEL_RUN_FREQ // MODEL_CONTEXT_FREQ)
COMPILE_ARGS = ["--model-size", "512x256",
                "--camera-resolutions", "1928x1208", "1344x760",
                "--frame-skip", "4"]

def download_model() -> Path:
  from tinygrad import fetch
  path = fetch(MODEL_URL, name="driving_supercombo.onnx")
  h = hashlib.sha256(path.read_bytes()).hexdigest()
  assert h == MODEL_SHA256, f"model hash mismatch: {h} != {MODEL_SHA256}"
  print(f"downloaded model {path} ({path.stat().st_size/1e6:.2f} MB), sha256 verified")
  return path

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--openpilot-root", type=Path,
                 default=Path(os.getenv("OPENPILOT_ROOT", TINYGRAD_ROOT.parent)),
                 help="repo root containing the openpilot/ package (default: sibling of tinygrad_repo)")
  p.add_argument("--output", type=Path, default=Path("/tmp/driving_tinygrad_test.pkl"))
  args = p.parse_args()

  compile_script = args.openpilot_root / "openpilot/selfdrive/modeld/compile_modeld.py"
  assert compile_script.is_file(), f"{compile_script} not found, set --openpilot-root/OPENPILOT_ROOT"

  model = download_model()

  env = os.environ.copy()
  env.setdefault("DEV", "CPU")
  env["PYTHONPATH"] = f"{args.openpilot_root}:{TINYGRAD_ROOT}:{env.get('PYTHONPATH', '')}"
  cmd = [sys.executable, str(compile_script), "--onnx", str(model), "--output", str(args.output), *COMPILE_ARGS]
  print(f"running: DEV={env['DEV']} {' '.join(cmd)}")
  ret = subprocess.run(cmd, cwd=args.openpilot_root, env=env)
  assert ret.returncode == 0, f"compile_modeld.py failed with exit code {ret.returncode}"

  assert args.output.is_file() and args.output.stat().st_size > 0, f"missing output {args.output}"
  sys.path.insert(0, str(args.openpilot_root))
  from openpilot.selfdrive.modeld.helpers import load_oob
  with open(args.output, "rb") as f: out = load_oob(f)
  assert {'metadata', 'run_policy', (1928, 1208), (1344, 760)} <= set(out), f"unexpected pickle keys: {set(out)}"
  print(f"PASS: compiled pickle at {args.output} ({args.output.stat().st_size/1e6:.2f} MB), "
        f"model output {out['metadata']['output_shapes']}")

if __name__ == "__main__":
  main()

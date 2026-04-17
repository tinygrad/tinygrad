import os, subprocess, tempfile, textwrap, unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_DIR = (
  REPO_ROOT / "examples" / "mlperf" / "training_submission_v6.0" / "tinycorp" / "benchmarks" /
  "llama2_70b_lora" / "implementations" / "tinybox_8xMI350X"
)


def _make_fake_python3(bin_dir:Path, capture_path:Path) -> Path:
  fake_python = bin_dir / "python3"
  fake_python.write_text(textwrap.dedent(f"""\
    #!/bin/sh
    {{
      printf 'argv=%s\\n' "$*"
      printf 'pwd=%s\\n' "$PWD"
      printf 'MODEL=%s\\n' "$MODEL"
      printf 'FAKEDATA=%s\\n' "${{FAKEDATA-}}"
      printf 'MODEL_PATH=%s\\n' "${{MODEL_PATH-}}"
      printf 'LLAMA_LAYERS=%s\\n' "${{LLAMA_LAYERS-}}"
      printf 'PYTHONPATH=%s\\n' "${{PYTHONPATH-}}"
    }} > "{capture_path}"
  """))
  fake_python.chmod(0o755)
  return fake_python


class TestMLPerfLlamaSubmissionWrapper(unittest.TestCase):
  def test_wrapper_shell_scripts_parse(self):
    for script_name in ("dev_run.sh", "run_and_time.sh"):
      subprocess.run(["bash", "-n", str(WRAPPER_DIR / script_name)], check=True)

  def test_dev_run_sets_expected_env_for_fake_data_init(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "FAKEDATA": "1",
        "MODEL_PATH": "",
        "TOKENIZER_PATH": "",
        "LLAMA_LAYERS": "2",
        "BENCHMARK": "1",
      }
      subprocess.run(["bash", str(script)], check=True, env=env)

      captured = capture_path.read_text().splitlines()
      self.assertIn(f"argv={REPO_ROOT / 'examples' / 'mlperf' / 'model_train.py'}", captured)
      self.assertIn(f"pwd={REPO_ROOT}", captured)
      self.assertIn("MODEL=llama2_70b_lora", captured)
      self.assertIn("FAKEDATA=1", captured)
      self.assertIn("MODEL_PATH=", captured)
      self.assertIn("LLAMA_LAYERS=2", captured)
      self.assertIn(f"PYTHONPATH={REPO_ROOT}", captured)

  def test_dev_run_requires_model_path_for_real_runs(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "FAKEDATA": "",
        "MODEL_PATH": "",
        "TOKENIZER_PATH": "",
      }
      proc = subprocess.run(["bash", str(script)], check=False, env=env, capture_output=True, text=True)

      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("MODEL_PATH must point to converted Llama2 70B weights", proc.stderr)
      self.assertFalse(capture_path.exists())


if __name__ == "__main__":
  unittest.main()

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

def _make_real_run_assets(tmpdir_path:Path) -> tuple[Path, Path]:
  dataset_dir = tmpdir_path / "dataset"
  dataset_dir.mkdir()
  (dataset_dir / "train.jsonl").write_text('{"input_ids":[1,2,3,4],"labels":[1,2,3,4]}\n')
  (dataset_dir / "validation.jsonl").write_text('{"input_ids":[1,2,3,4],"labels":[-1,2,3,-1]}\n')

  model_dir = tmpdir_path / "weights"
  model_dir.mkdir()
  (model_dir / "model.safetensors").write_text("")
  (model_dir / "tokenizer.model").write_text("")
  return dataset_dir, model_dir / "model.safetensors"


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

  def test_dev_run_requires_existing_model_path_for_real_runs(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      dataset_dir, _ = _make_real_run_assets(tmpdir_path)
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "DATASET_PATH": str(dataset_dir),
        "FAKEDATA": "",
        "MODEL_PATH": str(tmpdir_path / "missing-model.safetensors"),
      }
      proc = subprocess.run(["bash", str(script)], check=False, env=env, capture_output=True, text=True)

      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("does not exist", proc.stderr)
      self.assertFalse(capture_path.exists())

  def test_dev_run_requires_tokenizer_for_real_runs(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      dataset_dir = tmpdir_path / "dataset"
      dataset_dir.mkdir()
      (dataset_dir / "train.jsonl").write_text('{"input_ids":[1,2,3,4],"labels":[1,2,3,4]}\n')
      (dataset_dir / "validation.jsonl").write_text('{"input_ids":[1,2,3,4],"labels":[-1,2,3,-1]}\n')
      model_path = tmpdir_path / "model.safetensors"
      model_path.write_text("")
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "DATASET_PATH": str(dataset_dir),
        "FAKEDATA": "",
        "MODEL_PATH": str(model_path),
      }
      proc = subprocess.run(["bash", str(script)], check=False, env=env, capture_output=True, text=True)

      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("tokenizer.model not found alongside MODEL_PATH", proc.stderr)
      self.assertFalse(capture_path.exists())

  def test_dev_run_requires_dataset_splits_for_real_runs(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      dataset_dir = tmpdir_path / "dataset"
      dataset_dir.mkdir()
      (dataset_dir / "train.jsonl").write_text('{"input_ids":[1,2,3,4],"labels":[1,2,3,4]}\n')
      model_dir = tmpdir_path / "weights"
      model_dir.mkdir()
      (model_dir / "model.safetensors").write_text("")
      (model_dir / "tokenizer.model").write_text("")
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "DATASET_PATH": str(dataset_dir),
        "FAKEDATA": "",
        "MODEL_PATH": str(model_dir / "model.safetensors"),
        "RUNMLPERF": "1",
      }
      proc = subprocess.run(["bash", str(script)], check=False, env=env, capture_output=True, text=True)

      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("missing validation split files", proc.stderr)
      self.assertFalse(capture_path.exists())

  def test_dev_run_accepts_real_run_with_preflighted_assets(self):
    script = WRAPPER_DIR / "dev_run.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      capture_path = tmpdir_path / "capture.txt"
      _make_fake_python3(tmpdir_path, capture_path)
      dataset_dir, model_path = _make_real_run_assets(tmpdir_path)
      env = os.environ | {
        "PATH": f"{tmpdir_path}:{os.environ['PATH']}",
        "DATASET_PATH": str(dataset_dir),
        "FAKEDATA": "",
        "MODEL_PATH": str(model_path),
      }
      subprocess.run(["bash", str(script)], check=True, env=env)

      captured = capture_path.read_text().splitlines()
      self.assertIn(f"argv={REPO_ROOT / 'examples' / 'mlperf' / 'model_train.py'}", captured)
      self.assertIn(f"MODEL_PATH={model_path}", captured)

  def test_run_and_time_invokes_init_then_real_run(self):
    script = WRAPPER_DIR / "run_and_time.sh"
    with tempfile.TemporaryDirectory(prefix="llama-wrapper-") as tmpdir:
      tmpdir_path = Path(tmpdir)
      fake_wrapper_dir = tmpdir_path / "wrapper"
      fake_wrapper_dir.mkdir()
      run_copy = fake_wrapper_dir / "run_and_time.sh"
      run_copy.write_text(script.read_text())
      run_copy.chmod(0o755)

      capture_path = tmpdir_path / "phases.txt"
      fake_dev_run = fake_wrapper_dir / "dev_run.sh"
      fake_dev_run.write_text("\n".join([
        "#!/bin/sh",
        "{",
        "  printf '%s\\n' \\",
        (
          '    "phase MODEL_PATH=${MODEL_PATH-} FAKEDATA=${FAKEDATA-} '
          'INITMLPERF=${INITMLPERF-} RUNMLPERF=${RUNMLPERF-} '
          'LLAMA_LAYERS=${LLAMA_LAYERS-} DATASET_PATH=${DATASET_PATH-}" \\'
        ),
        '    "controls TRAIN=${TRAIN-} CKPT=${CKPT-} FP8=${FP8-}" \\',
        (
          '    "pins DEBUG=${DEBUG-} HK_FLASH_ATTENTION=${HK_FLASH_ATTENTION-} '
          'ASM_GEMM=${ASM_GEMM-} OFFLOAD_OPTIM=${OFFLOAD_OPTIM-} '
          'JITBEAM=${JITBEAM-} BEAM_UOPS_MAX=${BEAM_UOPS_MAX-} '
          'BEAM_UPCAST_MAX=${BEAM_UPCAST_MAX-} BEAM_LOCAL_MAX=${BEAM_LOCAL_MAX-} '
          'BEAM_MIN_PROGRESS=${BEAM_MIN_PROGRESS-} BEAM_PADTO=${BEAM_PADTO-}" \\'
        ),
        (
          '    "common LOGMLPERF=${LOGMLPERF-} '
          'SUBMISSION_PLATFORM=${SUBMISSION_PLATFORM-} DATA_SEED=${DATA_SEED-}"'
        ),
        f'}} >> "{capture_path}"',
        "",
      ]))
      fake_dev_run.chmod(0o755)

      logfile = tmpdir_path / "wrapper.log"
      env = os.environ | {
        "ASM_GEMM": "0",
        "BEAM_LOCAL_MAX": "1",
        "BEAM_MIN_PROGRESS": "1",
        "BEAM_PADTO": "0",
        "BEAM_UOPS_MAX": "1",
        "BEAM_UPCAST_MAX": "1",
        "DATASET_PATH": "/tmp/govreport",
        "DATA_SEED": "999",
        "DEBUG": "9",
        "FP8": "1",
        "HK_FLASH_ATTENTION": "0",
        "JITBEAM": "1",
        "CKPT": "1",
        "LOGMLPERF": "0",
        "MODEL_PATH": "/tmp/llama2",
        "OFFLOAD_OPTIM": "0",
        "LOGFILE": str(logfile),
        "SEED": "7",
        "SUBMISSION_PLATFORM": "wrong",
        "TRAIN": "0",
      }
      subprocess.run(["bash", str(run_copy)], check=True, env=env)

      phases = capture_path.read_text().splitlines()
      self.assertEqual(len(phases), 8)
      self.assertEqual(
        phases[0],
        "phase MODEL_PATH= FAKEDATA=1 INITMLPERF=1 RUNMLPERF= LLAMA_LAYERS=2 DATASET_PATH=",
      )
      self.assertEqual(
        phases[1],
        "controls TRAIN= CKPT= FP8=",
      )
      self.assertEqual(
        phases[2],
        "pins DEBUG=0 HK_FLASH_ATTENTION=1 ASM_GEMM=1 OFFLOAD_OPTIM=1 JITBEAM=3 "
        "BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 "
        "BEAM_MIN_PROGRESS=5 BEAM_PADTO=1",
      )
      self.assertEqual(
        phases[3],
        "common LOGMLPERF=1 SUBMISSION_PLATFORM=tinybox_8xMI350X DATA_SEED=7",
      )
      self.assertEqual(
        phases[4],
        "phase MODEL_PATH=/tmp/llama2 FAKEDATA= INITMLPERF= RUNMLPERF=1 LLAMA_LAYERS= DATASET_PATH=/tmp/govreport",
      )
      self.assertEqual(
        phases[5],
        "controls TRAIN= CKPT= FP8=",
      )
      self.assertEqual(
        phases[6],
        "pins DEBUG=0 HK_FLASH_ATTENTION=1 ASM_GEMM=1 OFFLOAD_OPTIM=1 JITBEAM=3 "
        "BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 "
        "BEAM_MIN_PROGRESS=5 BEAM_PADTO=1",
      )
      self.assertEqual(
        phases[7],
        "common LOGMLPERF=1 SUBMISSION_PLATFORM=tinybox_8xMI350X DATA_SEED=7",
      )
      self.assertTrue(logfile.exists())


if __name__ == "__main__":
  unittest.main()

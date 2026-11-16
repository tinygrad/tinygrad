import os, pathlib, subprocess, tempfile, typing, unittest
from unittest.mock import patch
import tinygrad.llm.cli as cli

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "extra/llm_service"

def mock_environment(**values:str) -> typing.Callable[..., typing.Any]:
  return lambda key, default=None: values.get(key, default)


class TestMandatoryVision(unittest.TestCase):
  def test_backend_exemptions_and_nonempty_file(self):
    """Only production models with a supported tower require it; file checks do not validate GGUF contents (launchers hash it)."""
    with tempfile.TemporaryDirectory() as td:
      missing = pathlib.Path(td) / "missing-mmproj.gguf"
      for architecture in ("qwen4exp", "glm5-next"):
        if missing.exists(): missing.unlink()
        for backend, required in (("HIP", True), ("AMD", True), ("PCI+AMD", True), ("", True),
                                  ("NULL", False), ("NULL:HIP:gfx1201", False), ("CPU", False)):
          with self.subTest(architecture=architecture, backend=backend), \
               patch.object(cli, "getenv", side_effect=mock_environment(DEV=backend, VISION_MMPROJ=str(missing))):
            self.assertEqual(cli._production_vision_projector_missing(architecture), required)
        with self.subTest(architecture=architecture, picture="valid-path"), \
             patch.object(cli, "getenv", side_effect=mock_environment(DEV="HIP", VISION_MMPROJ=str(missing))):
          missing.write_bytes(b"GGUF")
          self.assertFalse(cli._production_vision_projector_missing(architecture))

  def test_models_without_a_supported_tower_do_not_require_one(self):
    for architecture in ("glm-dsa", "llama", "unknown"):
      with self.subTest(architecture=architecture), \
           patch.object(cli, "getenv", side_effect=mock_environment(DEV="PCI+AMD", VISION_MMPROJ="")):
        self.assertFalse(cli._production_vision_projector_missing(architecture))

  def test_qwen_and_c5_require_a_projector_before_start(self):
    for script in ("qwen_6x7900xtx.sh", "glm_flash_6x7900xtx.sh"):
      with self.subTest(script=script):
        result = subprocess.run([SCRIPTS / script, "start"], cwd=ROOT, text=True, capture_output=True,
                                env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
                                     "VISION_MMPROJ": "/nonexistent/mmproj-F16.gguf"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("VISION_MMPROJ is mandatory", result.stderr)

  def test_c9_is_text_only_and_ignores_a_supplied_projector(self):
    base_env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    preview = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "--print-command"], cwd=ROOT, text=True,
                             capture_output=True, env=base_env, check=True)
    self.assertIn('VISION_MMPROJ= XDG_CACHE_HOME=', preview.stdout)
    self.assertNotIn("mmproj-F16.gguf", preview.stdout)
    # Even a caller exporting a picture setting cannot accidentally load the incompatible GLM Flash tower.
    poisoned = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "--print-command"], cwd=ROOT, text=True,
                              capture_output=True, env={**base_env, "VISION_MMPROJ": "/nonexistent/mmproj-F16.gguf"}, check=True)
    self.assertEqual(poisoned.stdout, preview.stdout)
    failed = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "start"], cwd=ROOT, text=True, capture_output=True, env=base_env)
    self.assertEqual(failed.returncode, 2)
    self.assertIn("PENDING", failed.stderr)
    self.assertIn("210 W is not safe", failed.stderr)


if __name__ == "__main__": unittest.main()

import os, sys, pathlib, tempfile, unittest, subprocess, typing
from unittest.mock import patch
import tinygrad.llm.cli as cli

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "extra/llm_service"

def mock_environment(**values:str) -> typing.Callable[..., typing.Any]:
  return lambda key, default=None: values.get(key, default)

class TestMandatoryVision(unittest.TestCase):
  def test_backend_exemptions_and_nonempty_file(self):
    """The CLI's helper tests only real test backends and file presence, not GGUF validity (launchers hash the real projector)."""
    with tempfile.TemporaryDirectory() as td:
      missing = pathlib.Path(td) / "missing-mmproj.gguf"
      for backend, required in (("HIP", True), ("AMD", True), ("PCI+AMD", True), ("", True),
                                ("NULL", False), ("NULL:HIP:gfx1201", False), ("CPU", False)):
        with self.subTest(backend=backend), patch.object(cli, "getenv", side_effect=mock_environment(DEV=backend, VISION_MMPROJ=str(missing))):
          self.assertEqual(cli._production_vision_projector_missing(), required)
      mmproj = pathlib.Path(td) / "mmproj.gguf"
      mmproj.write_bytes(b"GGUF")
      with patch.object(cli, "getenv", side_effect=mock_environment(DEV="HIP", VISION_MMPROJ=str(mmproj))):
        self.assertFalse(cli._production_vision_projector_missing())
      with patch.object(cli, "getenv", side_effect=mock_environment(VISION_MMPROJ=str(mmproj))):
        self.assertFalse(cli._production_vision_projector_missing())

  def test_missing_projector_stops_before_model_load(self):
    with patch.object(cli, "getenv", side_effect=mock_environment(DEV="HIP", VISION_MMPROJ="/nonexistent/mmproj-F16.gguf")), \
         patch.object(cli, "fetch", side_effect=AssertionError("model loading must not be attempted")), \
         patch.object(sys, "argv", ["tinygrad.llm", "--serve", "18999"]), \
         self.assertRaises(SystemExit) as stopped:
      cli.main()
    self.assertEqual(stopped.exception.code, 2)

  def test_qwen_and_c5_require_a_projector_before_start(self):
    for script in ("qwen_6x7900xtx.sh", "glm_flash_6x7900xtx.sh"):
      with self.subTest(script=script):
        result = subprocess.run([SCRIPTS / script, "start"], cwd=ROOT, text=True, capture_output=True,
                                env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "VISION_MMPROJ": "/nonexistent/mmproj-F16.gguf"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("VISION_MMPROJ is mandatory", result.stderr)

  def test_c9_never_silently_starts_without_picture_support(self):
    with tempfile.TemporaryDirectory() as td:
      projector = pathlib.Path(td) / "mmproj.gguf"
      projector.write_bytes(b"GGUF")  # Presence test only; this is not an approved model/projector.
      base_env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "VISION_MMPROJ": str(projector), "GLM53_POWER_LIMIT": "120"}
      failed = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "start"], cwd=ROOT, text=True, capture_output=True, env=base_env)
      self.assertEqual(failed.returncode, 2)
      self.assertIn("picture input path", failed.stderr)
      preview = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "--print-command"], cwd=ROOT, text=True,
                               capture_output=True, env=base_env, check=True)
      self.assertIn("VISION_MMPROJ=" + str(projector), preview.stdout)
      self.assertIn("PENDING: preview only", preview.stderr)
    without = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "--print-command"], cwd=ROOT, text=True, capture_output=True,
                             env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]})
    self.assertEqual(without.returncode, 2)
    self.assertIn("no text-only command", without.stderr)

if __name__ == "__main__": unittest.main()

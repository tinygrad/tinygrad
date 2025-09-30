from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pytest

if sys.platform.startswith("win"):
  pytest.skip("HEVC demo subprocess test skipped on Windows CI", allow_module_level=True)


def test_decode_hevc_demo_stub_mode(tmp_path):
  demo = Path(__file__).resolve().parents[2] / "examples" / "decode_hevc_demo.py"
  cmd = [sys.executable, str(demo), "--use-stub", "--timestamps"]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  stdout = result.stdout
  assert "Running HEVC demo in stub mode" in stdout
  assert "Frames decoded" in stdout
  assert "Total frames" in stdout

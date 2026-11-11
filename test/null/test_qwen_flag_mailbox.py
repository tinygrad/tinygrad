import os
import subprocess
import sys
import unittest


def check_flag(value: int | None) -> None:
  code = """
from tinygrad.llm.kernels.amd_qwen import FLAG_AR_MAXN, flag_ar_shape
expected = 1 << 18 if FLAG_AR_MAXN != (1 << 20) else 1 << 20
assert FLAG_AR_MAXN == expected
assert 2*6*FLAG_AR_MAXN*4 == (48 if FLAG_AR_MAXN == 1 << 20 else 12) << 20
assert flag_ar_shape(FLAG_AR_MAXN, 64) is not None
assert flag_ar_shape(FLAG_AR_MAXN + 1, 64) is None
"""
  env = os.environ.copy()
  env["DEV"] = "NULL"
  if value is None:
    env.pop("FLAG_AR_MAXN", None)
  else:
    env["FLAG_AR_MAXN"] = str(value)
  subprocess.run([sys.executable, "-c", code], env=env, check=True)


class TestQwenFlagMailbox(unittest.TestCase):
  def test_default_sized_for_decode(self):
    check_flag(None)

  def test_legacy_size_override(self):
    check_flag(1 << 20)


if __name__ == "__main__":
  unittest.main()

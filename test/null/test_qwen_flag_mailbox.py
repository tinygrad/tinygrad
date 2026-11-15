import os
import subprocess
import sys
import unittest


def check_flag() -> None:
  code = """
from tinygrad.llm.kernels.amd_qwen import FLAG_AR_MAXN, FLAG_AR_POOLN, flag_ar_shape
assert FLAG_AR_MAXN == 1 << 18
assert FLAG_AR_POOLN == 65536
assert 2*6*FLAG_AR_MAXN*4 == 12 << 20
assert flag_ar_shape(FLAG_AR_POOLN, 64) is not None
assert flag_ar_shape(FLAG_AR_POOLN + 1, 64) is None
"""
  env = os.environ.copy()
  env["DEV"] = "NULL"
  subprocess.run([sys.executable, "-c", code], env=env, check=True)


class TestQwenFlagMailbox(unittest.TestCase):
  def test_production_mailbox_boundaries(self):
    check_flag()


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
import subprocess, sys
from tinygrad.helpers import getenv

LOOPS = getenv("LOOPS", 10)

for i in range(LOOPS):
  print(f"=== Running hive_reset.py ({i+1}/{LOOPS}) ===")
  subprocess.run([sys.executable, "extra/amdpci/hive_reset.py"], check=True)
  print("=== hive_reset complete ===")

  print(f"=== Running test_tiny.py ({i+1}/{LOOPS}) ===")
  ret = subprocess.run([sys.executable, "test/test_tiny.py", "TestTiny.test_plus"])
  print(f"=== test_tiny.py exited with code {ret.returncode} ===")

import os, subprocess, sys
from pathlib import Path
from tinygrad.helpers import temp

EXAMPLES_DIR = Path(__file__).parent
PROFILE_PATH = Path(temp("profile.pkl", append_user=True))

EXAMPLES = [
  "test/backend/test_custom_kernel.py TestCustomKernel.test_empty",
  "test/test_tiny.py TestTiny.test_plus",
  "test/test_tiny.py TestTiny.test_gemm",
  "extra/sqtt/examples/discover_ops.py"
]

if __name__ == "__main__":
  arch = subprocess.check_output(["python", "-c", "from tinygrad import Device; print(Device['AMD'].arch)"], text=True,
                                 env={**os.environ, "DEBUG":"0"}).rstrip()
  (EXAMPLES_DIR/arch).mkdir(exist_ok=True)
  for test in EXAMPLES:
    for i in range(2):
      subprocess.run([sys.executable, *test.split()], cwd=EXAMPLES_DIR.parent.parent.parent,
                     env={**os.environ, "AMD":"1", "AM_RESET":"1", "VIZ":"-2", "PYTHONPATH":"."})
      PROFILE_PATH.rename(dest:=EXAMPLES_DIR/arch/f"profile_{test.split('.')[-1].replace('test_', '')}_run_{i}.pkl")
      print(f"saved SQTT trace to {dest}")

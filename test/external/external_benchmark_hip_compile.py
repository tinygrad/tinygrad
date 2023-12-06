import random, os
from tinygrad.helpers import Timing
from tinygrad.runtime.ops_hip import HIPDevice
from tinygrad.runtime.ops_gpu import CLDevice

# OMP_NUM_THREADS=1 strace -tt -f -e trace=file python3 test/external/external_benchmark_hip_compile.py
# AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1 python3 test/external/external_benchmark_hip_compile.py

# issue is in https://github.com/ROCm-Developer-Tools/clr/

if __name__ == "__main__":
  hipdev = HIPDevice()
  cldev = CLDevice()

  # warmup
  name = "none"+str(random.randint(0, 1000000))
  cldev.compiler._compile(f"void {name}() {{}}")
  print("compile cl warmed up")
  hipdev.compiler._compile(f"void {name}() {{}}")
  print("compile hip warmed up")

  print("**** benchmark ****")
  name = "none"+str(random.randint(0, 1000000))
  # this uses AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, then it links the lib on the next step
  with Timing("compile cl:  "): cldev.compiler._compile(f"void {name}() {{}}")
  # this uses AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, much slower
  with Timing("compile hip: "): hipdev.compiler._compile(f"void {name}() {{}}")
  os._exit(0)




import os
import sys
import pytest
import subprocess

PROGRAM = """
{imports}

mesa.glsl_type_singleton_init_or_ref()

ctx = llvm.LLVMContextCreate()
assert ctx
llvm.LLVMContextDispose(ctx)

mesa.glsl_type_singleton_decref()
"""

@pytest.mark.skipif(
  sys.platform != "darwin",
  reason="requires macOS"
)
@pytest.mark.parametrize("imports", [
    "from tinygrad.runtime.autogen import mesa, llvm",
    "from tinygrad.runtime.autogen import llvm, mesa",
])
def test_mesa_llvm_import_order(imports):
  env = os.environ.copy()
  env["DEV"] = "CPU:LVP"

  proc = subprocess.run(
    [sys.executable, "-c", PROGRAM.format(imports=imports)],
    env=env,
    text=True,
    capture_output=True,
  )

  assert proc.returncode == 0, proc.stderr


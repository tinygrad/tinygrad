# Full C2 start on NULL: model load, picture tower, sequence warm-up and run-again; stop before serving.
# Open the AMD custom-kernel gate for NULL so production-shaped graphs are traced without executing on a GPU.
# mypy: ignore-errors
import os, runpy, socket, sys
from tinygrad import Tensor

_realize = Tensor.realize


def realize(*args, **kwargs):
  return None if not args else _realize(*args, **kwargs)


Tensor.realize = realize  # Empty expert-table realize calls need this NULL-only guard.
import tinygrad.llm.kernels.amd as amdk

_supported = amdk.amd_custom_kernels_supported


def supported(device):
  first = device[0] if isinstance(device, tuple) else device
  return True if first is not None and str(first).split(":")[0] == "NULL" else _supported(device)


amdk.amd_custom_kernels_supported = supported
_gfx = amdk._gfx_version


def gfx(device):
  first = device[0] if isinstance(device, tuple) else device
  return (11, 0, 0) if first is not None and str(first).split(":")[0] == "NULL" else _gfx(device)


amdk._gfx_version = gfx
for module in (
  "tinygrad.llm.qwen4exp",
  "tinygrad.llm.kernels.amd_qwen",
  "tinygrad.llm.kernels.amd_gguf",
  "tinygrad.llm.qwen_vision",
  "tinygrad.llm.model",
  "tinygrad.llm.engine",
  "tinygrad.llm.cli",
  "tinygrad.llm.serve",
):
  try:
    __import__(module, fromlist=["x"])
  except Exception as exc:
    print("C2 NULLWARM import", module, exc, flush=True)


def wrap(original):
  def wrapped(device):
    first = device[0] if isinstance(device, tuple) else device
    return True if first is not None and str(first).split(":")[0] == "NULL" else original(device)

  return wrapped


for _, module in list(sys.modules.items()):
  if _.startswith("tinygrad") and callable(original := getattr(module, "amd_custom_kernels_supported", None)) and module is not amdk:
    module.amd_custom_kernels_supported = wrap(original)


def hg_for(original):
  def hg(x):
    result = original(x)
    if result is None and isinstance(x.device, tuple) and str(x.device[0]).split(":")[0] == "NULL":
      return x.to(x.device[0]).shard(x.device)
    return result

  return hg


for _, module in list(sys.modules.items()):
  if _.startswith("tinygrad") and callable(original := getattr(module, "host_allgather", None)):
    module.host_allgather = hg_for(original)

import tinygrad.llm.qwen4exp as qwen

_wait_host = qwen.Transformer._wait_host


def wait_host(self, get, what):
  value = get()
  return 0 if isinstance(value, int) and value < 0 else _wait_host(self, get, what)


qwen.Transformer._wait_host = wait_host
import tinygrad.llm.serve as serve


def done(self, *args, **kwargs):
  print("NULLWARM: warm-up DONE, the server would serve now", flush=True)
  os._exit(0)


serve.LLMServer.serve_forever = done
def _free_port() -> int:
  # (a fixed port collided with another agent's warm-up binding it at the same moment: Errno 98, Nov 16 16:3x)
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


sys.argv = ["tinygrad.llm", "--model", os.environ["MODEL"], "--shard", "6", "--max_context", "262144", "--serve", str(_free_port())]
runpy.run_module("tinygrad.llm", run_name="__main__")

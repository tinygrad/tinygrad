# New C9 production-shaped start line on NULL: model load and full sequence warm-up, never real GPU execution.
# mypy: ignore-errors
import os, runpy, sys
from tinygrad import Device, Tensor

_realize = Tensor.realize


def realize(*args, **kwargs):
  return None if not args else _realize(*args, **kwargs)


Tensor.realize = realize
import tinygrad.llm.kernels.amd as amdk

_supported = amdk.amd_custom_kernels_supported


def supported(device):
  first = device[0] if isinstance(device, tuple) else device
  return True if first is not None and str(first).split(":")[0] == "NULL" else _supported(device)


amdk.amd_custom_kernels_supported = supported
_gfx = amdk._gfx_version


def gfx(device):
  first = device[0] if isinstance(device, tuple) else device
  return (12, 0, 1) if first is not None and str(first).split(":")[0] == "NULL" else _gfx(device)


amdk._gfx_version = gfx
# RDNA4 sparse-index initialization reads the device target directly.
for i in range(14):
  Device["NULL" if i == 0 else f"NULL:{i}"].target = (12, 0, 1)
for module in (
  "tinygrad.llm.glm_dsa",
  "tinygrad.llm.glm_dsa_gguf",
  "tinygrad.llm.kernels.amd_glm_dsa",
  "tinygrad.llm.glm5next",
  "tinygrad.llm.kernels.amd_glm",
  "tinygrad.llm.model",
  "tinygrad.llm.cli",
  "tinygrad.llm.engine",
  "tinygrad.llm.serve",
):
  try:
    __import__(module, fromlist=["x"])
  except Exception as exc:
    print("C9 NULLWARM import", module, exc, flush=True)


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

import tinygrad.llm.glm_dsa as glm

_wait_host = glm.Transformer._wait_host


def wait_host(self, get, what):
  value = get()
  if isinstance(value, int) and value < 0:
    # NULL writes no sampled tokens; sequence steps slice one zero token list per warmed sequence.
    if what.startswith("a step of") and what.endswith("sequences"):
      return [0] * len(self._seq_host_out)
    return 0
  return _wait_host(self, get, what)


glm.Transformer._wait_host = wait_host

import tinygrad.llm.serve as serve


def done(self, *args, **kwargs):
  print("C9 NULLWARM: warm-up DONE, the server would serve now", flush=True)
  os._exit(0)


serve.LLMServer.serve_forever = done
sys.argv = ["tinygrad.llm", "--model", os.environ["MODEL"], "--shard", "14", "--max_context", "262144", "--serve", "18998"]
runpy.run_module("tinygrad.llm", run_name="__main__")

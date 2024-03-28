"""
Vulkan Backend via Kompute

```
# install dependencies
sudo apt install glslang
pip3 install git+https://github.com/iceychris/kompute.git@expose-set-raw-data

# run tests
DEBUG=4 KOMPUTE=1 GPU=1 PYTHONPATH=. pytest -s -v test/test_ops.py::TestOps::test_add_dummy
DEBUG=4 KOMPUTE=1 GPU=1 PYTHONPATH=. pytest -s -v test/test_ops.py::TestOps
```
"""

from typing import NamedTuple, Tuple
import numpy as np
import functools
from tinygrad.helpers import flat_mv
from tinygrad.device import Compiled, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.kompute import KomputeLanguage
from tinygrad.device import Allocator
import os
import kp


def compile_source(source): 
    open("tmp_kp_shader.comp", "w").write(source) 
    os.system("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv") 
    res = open("tmp_kp_shader.comp.spv", "rb").read()
    return res


mgr = kp.Manager()
CI = os.getenv("CI", "") != ""

class KomputeProgram:
  def __init__(self, name: str, prg: str, lib: str):
    self.name, self.prg, self.lib = name, prg, lib
  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    bufs = args
    params = [b.tensor for b in bufs]
    algo = mgr.algorithm(
        params,
        self.lib,
        local_size,
        [],
        []
    )
    (mgr.sequence()
     .record(kp.OpTensorSyncDevice(params))
     .record(kp.OpAlgoDispatch(algo))
     .record(kp.OpTensorSyncLocal(params))
     .eval())


class K(NamedTuple):
  array: np.ndarray
  tensor: mgr.tensor  # type: ignore


class KomputeAllocator(Allocator):
  def _alloc(self, size:int) -> K:
    array = np.empty(size, dtype=np.uint8)
    tensor = mgr.tensor(np.frombuffer(array, dtype=np.float32))
    return K(array, tensor)
  def as_buffer(self, src:K) -> memoryview:
    return flat_mv(memoryview(src.tensor.data()))
  def copyin(self, dest:K, src:memoryview):
    dest.tensor.set_raw_data(src)
    return dest
  def copyout(self, dest:memoryview, src:K):
    dest[:] = flat_mv(memoryview(src.tensor.data()))
    return dest


class KomputeCompiler(Compiler):
  linearizer_opts = LinearizerOptions("KOMPUTE", supports_float4=False, has_local=False, has_shared=False)
  def __init__(self, device:str):
    super().__init__(f"kompute_{device}")
  def render(self, name:str, uops) -> str: return uops_to_cstyle(KomputeLanguage(), name, uops)
  def compile(self, src:str) -> bytes: return compile_source(src)
  def compile_cached(self, src:str) -> bytes: return compile_source(src)


class KomputeDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, KomputeAllocator(), KomputeCompiler(device), functools.partial(KomputeProgram, self))

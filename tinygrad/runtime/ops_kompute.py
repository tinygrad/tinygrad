from typing import NamedTuple, Tuple
import numpy as np
import functools
import subprocess
import hashlib

import kp

from tinygrad.helpers import flat_mv
from tinygrad.device import Compiled, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.kompute import KomputeLanguage
from tinygrad.device import Allocator


def compile_source(source: str) -> bytes:
  fname = f"/tmp/kompute-{hashlib.md5(source.encode('utf-8')).hexdigest()}.comp.spv"
  proc = subprocess.run(["glslangValidator", "--stdin", "-S", "comp", "-V", "-o", fname], input=source.encode("utf-8"), capture_output=True)
  if proc.returncode != 0:
    errors = proc.stdout.decode("utf-8") + "\n" + proc.stderr.decode("utf-8")
    raise ValueError(f"Failed to compile shader with glslangValidator: {errors}")
  return open(fname, "rb").read()


mgr = kp.Manager()

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
    size = (size + 3) & ~3
    array = np.empty(size, dtype=np.uint8)
    tensor = mgr.tensor_t(np.frombuffer(array, dtype=np.float32))
    return K(array, tensor)
  def as_buffer(self, src:K) -> memoryview:
    return flat_mv(memoryview(src.tensor.data()))
  def copyin(self, dest:K, src:memoryview):
    dest.tensor.set_raw_data(src)
    return dest
  def copyout(self, dest:memoryview, src:K):
    dest[:] = flat_mv(memoryview(src.tensor.data()))[:len(dest)]
    return dest


class KomputeCompiler(Compiler):
  linearizer_opts = LinearizerOptions("KOMPUTE", supports_float4=False, has_local=False, has_shared=False)
  def __init__(self, device:str):
    super().__init__(cachekey=f"kompute_v1_{device}")
  def render(self, name:str, uops) -> str: return uops_to_cstyle(KomputeLanguage(), name, uops)
  def compile(self, src:str) -> bytes: return compile_source(src)


class KomputeDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, KomputeAllocator(), KomputeCompiler(device), functools.partial(KomputeProgram, self))

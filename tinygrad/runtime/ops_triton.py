from __future__ import annotations
import hashlib
import pycuda.driver as cuda # type: ignore
from triton.compiler import compile as triton_compile
from typing import Any
from tinygrad.ops import  Compiled
from tinygrad.runtime.ops_cuda import RawCUDABuffer
from tinygrad.codegen.linearizer import LinearizerOptions
from tinygrad.renderer.triton import uops_to_triton


class TritonProgram:

  def __init__(self, name:str, prg:str, binary:bool=False):
    signature = ','.join(["*fp32" for _ in range(prg.splitlines()[1].count("data"))])

    prg = "import triton\nimport triton.language as tl\ntl.core.TRITON_MAX_TENSOR_NUMEL = float('inf')\n" + prg
    
    hash = hashlib.md5(prg.encode('utf-8')).hexdigest()
    fn = f"/tmp/{hash}.py"
    with open(fn, "w") as f: f.write(prg)
    codeObject = compile(prg, fn, "exec")
    exec(codeObject, globals())
    self.program = triton_compile(globals()[name], signature=signature, device_type="cuda", debug=True).asm["ptx"]
    self.program = cuda.module_from_buffer(self.program.encode('utf-8')).get_function(self.program.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, global_size, local_size, *args, wait=False) -> Any:
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.program(*[x._buf for x in args], block = tuple(local_size), grid = tuple(global_size))
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3


TritonBuffer = Compiled(RawCUDABuffer, LinearizerOptions(supports_float4=False, supports_float4_alu=False), uops_to_triton, TritonProgram)
from __future__ import annotations
import hashlib
from weakref import WeakValueDictionary
from torch import float32
import numpy as np
import pycuda.autoprimaryctx # type: ignore # noqa: F401
import pycuda.driver as cuda # type: ignore

import triton # type: ignore # noqa: F401
import triton.language as tl  # type: ignore # noqa: F401

from typing import Union, Tuple, Optional, Dict
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, LazyOp, Op, GlobalCounters
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.helpers import prod, DEBUG
from tinygrad.runtime.ops_gpu import CLBuffer

class TritonProgram:

  def __init__(self, name:str, prg:str):
    hash = hashlib.md5(prg.encode('utf-8')).hexdigest()
    fn = f"/tmp/{hash}.py"
    with open(fn, "w") as f: f.write(prg)
    codeObject = compile(prg, fn, "exec")
    exec(codeObject, globals())
    self.program = globals()["fxn"]
    

  def __call__(self, global_size, local_size, *args, wait=False) -> Any:
    self.program(*[x._buf for x in args])

class TritonDeviceAllocation(CLBuffer):
  def __init__(self, size):
    super().__init__(size)
    self.dtype = float32

  def data_ptr(self): return int(self._cl)

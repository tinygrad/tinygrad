from typing import List, Tuple, Dict, Union
import numpy as np
import unittest
from dataclasses import replace
from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, Kernel
from tinygrad.codegen.lowerer import get_grouped_dims
from tinygrad.ops import UOp, UOps
from tinygrad.device import Device, Buffer
from tinygrad.renderer import TensorCore
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
# from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule, lower_schedule, CompiledRunner
from tinygrad.helpers import DEBUG, prod, Context, getenv, CI, flatten, dedup
from tinygrad.dtype import DType, dtypes

def helper_tc_allclose(n:int, m:int, k:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_opt:int=0):
  a, b = Tensor.rand(m, k, dtype=dtype_in), Tensor.rand(k, n, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, acc_dtype=dtype_out)
  out = r.numpy()
  np_c = np_a @ np_b
  np.testing.assert_allclose(np_c, out, rtol=0.1)

#helper_tc_allclose(8, 16, 16, dtypes.half, dtypes.float32)
#helper_tc_allclose(8, 16, 32, dtypes.f8e4m3, dtypes.float32)
with Context(BEAM=1):
  helper_tc_allclose(4096, 4096, 2048, dtypes.f8e4m3, dtypes.float32)
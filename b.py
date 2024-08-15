from typing import List, Tuple, Dict, Union
import numpy as np
import unittest
from dataclasses import replace
from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, Kernel
from tinygrad.codegen.lowerer import get_grouped_dims
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Device, Buffer
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, ConstBuffer, LazyOp, MetaOps, TernaryOps, ReduceOps, UnaryOps
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
  sched = create_schedule([r.lazydata])
  realized_ast = sched[-1].ast
  run_schedule(sched)
  out = r.numpy()
  k = Kernel(realized_ast)
  k.apply_tensor_cores(1, axis=axis, tc_opt=tc_opt)
  k.linearize()
  assert len([uop for uop in k.uops if uop.op is UOps.WMMA]) > 0, "tensor core not triggered"
  assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  np_c = np_a @ np_b
  if dtype_out == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = 1e-2, 3e-3
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  np.testing.assert_allclose(np_c, out, atol=tc_atol, rtol=tc_rtol)

#helper_tc_allclose(16, 16, 32, dtypes.half, dtypes.float32)
helper_tc_allclose(16, 16, 32, dtypes.f8e4m3, dtypes.float32)
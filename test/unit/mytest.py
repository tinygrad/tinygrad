# import tinygrad
# from tinygrad.tensor import Tensor, dtypes
# import itertools

# print(Tensor.ones(1, dtype=dtypes.short).bitcast(dtypes.half).numpy())

# ten = Tensor.ones(1).bitcast(dtypes.int32)
# # ten = Tensor.ones(1).cast(dtypes.int32)
# print(ten.numpy()) 

# types = dtypes.fields().values()
# for d1, d2 in itertools.product(types, types):
#     if d1.itemsize != d2.itemsize: continue
#     print(Tensor.ones(1, dtype=d1).bitcast(d2))

import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.ops import UOp
from tinygrad.codegen.transcendental import payne_hanek_reduction, cody_waite_reduction, frexp, rintk, pow2if
import time
from typing import Callable, Optional
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.ops import UOp, Ops, sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner
from tinygrad.dtype import ConstType, DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, unwrap
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.rewriter import full_graph_rewrite
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer, PythonCompiler, PythonAllocator

#Tensor.sin(Tensor([12 * math.pi + 0.1])).numpy() 
print(payne_hanek_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1))[0].simplify())
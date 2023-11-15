from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, getenv
from tinygrad.realize import run_schedule
from tinygrad.ops import Device
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.features.search import time_linearizer
import numpy as np

N = getenv("N", 2048)
BS = getenv("BS", 1)
ITER = getenv("ITER", 10)

a = Tensor.rand(BS, N, N, dtype=dtypes.half).realize()
b = Tensor.rand(BS, N, N, dtype=dtypes.half).realize()

c = (a.reshape((BS, N, 1, N)) * b.permute([0, 2, 1]).reshape((BS, 1, N, N))).cast(dtypes.float).sum(axis=-1).cast(dtypes.half)
s = c.lazydata.schedule()
c.realize()
si = s[-1]
if getenv("MANUAL", 0):
  lin = Linearizer(si.ast)
  lin.apply_tensor_cores(1, hand_coded=False)
  lin.apply_opt(Opt(OptOps.UPCAST, 0, 4))
  lin.apply_opt(Opt(OptOps.UPCAST, 1, 4))
  lin.apply_opt(Opt(OptOps.LASTLOCAL, 1, 4))
  rawbufs = [c.lazydata.base.realized]+[buf.base.realized for buf in si.inputs]
  time_linearizer(lin, rawbufs, allow_test_size=False, cnt=1, disable_cache=True, clear_l2=False)
else:
  for _ in range(ITER):
    Device[si.out.device].exec_ast(si.ast, output=si.out, inputs=si.inputs, var_vals=si.var_vals, **si.out._device_extra_args())

if getenv("TEST", 1):
  import torch
  np_a, np_b = a.numpy(), b.numpy()
  ta, tb = torch.Tensor(np_a), torch.Tensor(np_b)
  np.testing.assert_allclose((ta @ tb).numpy(), c.numpy(), atol=1e-2, rtol=1e-2)

from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, getenv
from tinygrad.realize import run_schedule
from tinygrad.ops import Device
import torch
import numpy as np

N = getenv("N", 224)
CIN, COUT = getenv("CIN", 256), getenv("COUT", 256)
K = getenv("K", 3)
BS = getenv("BS", 1)
ITER = getenv("ITER", 1)

x = Tensor.rand(BS, CIN, N, N, dtype=dtypes.half).realize()
k = Tensor.rand(CIN, COUT, K, K, dtype=dtypes.half).realize()

c = x.conv2d(k, padding=1)
s = c.lazydata.schedule()
si = s[-1]
for _ in range(ITER):
  Device[si.out.device].exec_ast(si.ast, output=si.out, inputs=si.inputs, var_vals=si.var_vals, **si.out._device_extra_args())

if getenv("TEST", 1):
  torch_x, torch_k = torch.Tensor(x.numpy()), torch.Tensor(k.numpy())
  torch_c = torch.nn.functional.conv2d(torch_x, torch_k, padding=1)

  np.testing.assert_allclose(torch_c.numpy(), c.numpy(), atol=1e-2, rtol=1e-2)

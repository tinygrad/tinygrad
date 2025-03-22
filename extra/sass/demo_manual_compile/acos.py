from tinygrad import Tensor
import torch, time, unittest
import numpy as np

low = -1
high = 1
np.random.seed(0)
shps = [(45, 65)]
np_data = [np.random.uniform(low=low, high=high, size=size).astype(np.float32) for size in shps]
ts = [torch.tensor(data, requires_grad=False) for data in np_data]
tst = [Tensor(x.detach().cpu().numpy(), requires_grad=False) for x in ts]

tst[0].acos().realize()

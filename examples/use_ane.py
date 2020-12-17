#!/usr/bin/env python3
import numpy as np
from tinygrad.tensor import Tensor
import time

a = Tensor([-2,-1,0,1,2]).ane()
print(a.cpu())
b = a.relu()
print(b.cpu())
assert(np.all(b.cpu().data >= 0))

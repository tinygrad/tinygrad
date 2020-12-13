#!/usr/bin/env python3
import numpy as np
from tinygrad.tensor import Tensor

a = Tensor([-2,-1,0,1,2]).ane_()
print(a.cpu())
b = a.relu()
print(b.cpu())


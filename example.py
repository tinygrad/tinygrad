# pip install cppyy

import os
os.environ["DEBUG"] = "6"
os.environ["OPT"] = "0"
os.environ["NOOPT"] = "1"
# os.environ["VIZ"] = "1"
import numpy as np
from tinygrad import Tensor, dtypes

x_cpu = Tensor(np.random.normal(0, 1, (1000)), dtype=dtypes.float32, device="cpu")
y_cpu = Tensor(np.random.normal(0, 1, (1000)), dtype=dtypes.float32, device="cpu")

# x_cpu = Tensor([-0.003850722685456276, 0.7894648909568787], dtype=dtypes.float32, device="cpu")
# y_cpu = Tensor([0.05602724477648735, 0.981860339641571], dtype=dtypes.float32, device="cpu")

z_ref = (x_cpu / y_cpu).tolist()

x = x_cpu.to("tt")
y = y_cpu.to("tt")
z = x / y

print(z_ref[0:10])
print(z.tolist()[0:10])
print(z_ref[0:10])

# print("Source")
# print(x_cpu.tolist()[0:10])
# print(y_cpu.tolist()[0:10])

print("Finished")

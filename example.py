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
z_cpu = Tensor(np.random.normal(0, 1, (1000)), dtype=dtypes.float32, device="cpu")
x, y, z = x_cpu.to("tt"), y_cpu.to("tt"), z_cpu.to("tt")

out_ref = (x_cpu + y_cpu + z_cpu).tolist()
out = (x + y + z).tolist()

print(out_ref[0:4])
print(out[0:4])

# print("Source")
# print(x_cpu.tolist()[0:10])
# print(y_cpu.tolist()[0:10])

print("Finished")

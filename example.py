# pip install cppyy

import os
os.environ["DEBUG"] = "4"
os.environ["OPT"] = "0"

from tinygrad import Tensor, dtypes

x_cpu = Tensor(list(range(64 * 1024)), dtype=dtypes.float32)
z_ref = x_cpu.exp().tolist()

os.environ["DEBUG"] = "6"

x = x_cpu.to("tt")
z = -x

print(z.tolist()[0:10])
print(z_ref[0:10])

print("Finished")

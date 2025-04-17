# pip install cppyy

import os
os.environ["DEBUG"] = "6"
os.environ["OPT"] = "0"
os.environ["NOOPT"] = "1"
# os.environ["VIZ"] = "1"

from tinygrad import Tensor, dtypes

x_cpu = Tensor(list(range(1024)), dtype=dtypes.float32, device="cpu")
y_cpu = Tensor(list(range(1024)), dtype=dtypes.float32, device="cpu")
z_ref = (x_cpu + y_cpu).tolist()

x = x_cpu.to("tt")
y = y_cpu.to("tt")
z = x + y # any unary op uses the exp kernel

print(z_ref[0:10])
print(z.tolist()[0:10])
print(z_ref[0:10])

print("Finished")

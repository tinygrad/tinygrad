import os
from tinygrad import Tensor, dtypes

os.environ["DEBUG"] = "4"
os.environ["OPT"] = "0"

x_cpu = Tensor(list(range(64 * 1024)), dtype=dtypes.float32)
x_ref = (-x_cpu).tolist()

os.environ["DEBUG"] = "6"

x = x_cpu.to("tt")
z = -x

# Currently returns the wrong results because the exp kernel works on bfloat16
print(z.tolist()[0:10])
print(x_ref[0:10])

print("Finished")

from tinygrad import Tensor, TinyJit
import torch
import time
import numpy as np


width = 50
dims = 4
kernel_size = 5
dilation = (1,2)
stride = (2,1)
padding = 0
ceil_mode = False

a = Tensor.randint(width ** dims, high=width ** dims).reshape(*([width]*dims))

@TinyJit
def tiny_max(a, return_indices):
    out = a.max_pool2d(kernel_size=kernel_size,
                        dilation=dilation,
                        stride=stride,
                        padding=padding,
                        ceil_mode=ceil_mode,
                        return_indices=return_indices)
    if return_indices:
        return out[0].numpy(), out[1].numpy()
    else:
        return out.numpy()

def torch_max(a, return_indices):
    return torch.nn.functional.max_pool2d(a, kernel_size=kernel_size,
                                          dilation=dilation,
                                          stride=stride,
                                          padding=padding,
                                          ceil_mode=ceil_mode,
                                          return_indices=return_indices)

def avg_std_test(times, function, arg1, arg2):
    out = []
    for _ in range(times):
        time.sleep(0.1)
        start = time.perf_counter()
        function(arg1, arg2)
        end = time.perf_counter()
        out.append(end-start)
    out = np.array(out)
    return out.mean(), out.std()

times = 50

print("tiny - return_indices = False")
for _ in range(10): tiny_max(a, False)
avg, std = avg_std_test(times, tiny_max, a, False)
print(f"    {avg=}, {std=}")

print("tiny - return_indices = True")
for _ in range(10): tiny_max(a, True)
avg, std = avg_std_test(times, tiny_max, a, True)
print(f"    {avg=}, {std=}")

t = torch.tensor(a.numpy())

print("torch - return_indices = False")
for _ in range(10): torch_max(t, False)
avg, std = avg_std_test(times, torch_max, t, False)
print(f"    {avg=}, {std=}")

print("torch - return_indices = True")
for _ in range(10): torch_max(t, True)
avg, std = avg_std_test(times, torch_max, t, True)
print(f"    {avg=}, {std=}")



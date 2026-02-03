from tinygrad.tensor import Tensor
import numpy as np, ctypes

lol = []

# arr = np.ones(1000000, dtype=np.uint8)
# print(f"numpy: {(arr + 1)[:10]}")
# ptr = arr.ctypes.data
# orig = Tensor.from_blob(ptr, arr.shape, dtype='uint8', device='QCOM').contiguous().realize()

while True:
    arr = np.ones(100000, dtype=np.uint8)

    ptr = arr.ctypes.data
    print(hex(ptr), arr.ctypes.data)
    c_pointer = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    # print(f"c_pointer: {c_pointer[:10]}")
    
    orig = Tensor.from_blob(ptr, arr.shape, dtype='uint8', device='QCOM').realize()
    # orig = Tensor(arr, dtype='uint8', device='QCOM').realize()
    tensor = orig + 1
    assert (x:=tensor.numpy()[0]) == 2, f"tensor.numpy()[0]={tensor.numpy()[0]} | {x}"

    print(f"from_blob: {tensor.numpy()[:10]}")
    # print(f"numpy: {(arr + 1)[:10]}") 
    # ptr = arr.ctypes.data
    # print(hex(ptr), arr.ctypes.data)

    # lol.append((tensor, arr, arr.ctypes.data))
    # print(f"c_pinter : {c_pointer[:10]}")
    # del tensor
# print(f"\tnumpy: {(arr + 1)[:10]}")

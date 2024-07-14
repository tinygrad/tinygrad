from tinygrad import Tensor
import tinygrad as tg
import numpy as np


inta = np.int32(0b11111111111111111111111111111111)
intafirsthalf = np.int16(0b1111111111111111)
intasecondhalf = np.int16(0b1111111111111111)

print(type(intafirsthalf))

shape = (2000, 1000)
a = Tensor([inta] * np.prod(shape), dtype=tg.dtypes.int32).reshape(*shape)

# try bitcast
b = a.bitcast(tg.dtypes.int16)
assert tuple(b.shape) == (shape[0], shape[1] * 2)

# Create bexpected with np.int16 type explicitly
bexpected = np.array([[intasecondhalf, intafirsthalf] * shape[1]] * shape[0], dtype=np.int16)

print(b.numpy())
print(bexpected)
np.testing.assert_array_equal(b.numpy(), bexpected)

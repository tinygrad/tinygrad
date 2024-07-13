from tinygrad import Tensor
import tinygrad as tg

inta = 0b00000000000000000000000000000001 # 1.0
intafirsthalf = 0b0000000000000000
intasecondhalf = 0b0000000000000001
a = Tensor([inta] * 100, dtype=tg.dtypes.int32).reshape(10, 10)

# try bitcast
b = a.bitcast(tg.dtypes.int16)
assert tuple(b.shape) == (10, 20)

# bnumpy should be for each row: [1, 0, 1, 0, 1...]

bexpected = [[intasecondhalf, intafirsthalf] * 10] * 10

assert (b.numpy() == bexpected).all()

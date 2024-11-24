from icecream import ic, install
install()

from tinygrad import Tensor, dtypes

t = Tensor([2, 0.5]*9)
t = Tensor([1, 2, 3, 4, 5])
ic(t.numpy())

ic(t.cumprod().numpy())

# x = (Tensor([0, 1], dtype=dtypes.int16)).cumprod(0).dtype
# ic(x)
# (Tensor([0, 1], dtype=dtypes.bool)).cumprod(0).dtype == dtypes.bool
# (Tensor([0, 1], dtype=dtypes.int8)).cumprod(0).dtype == dtypes.int8
# (Tensor([0, 1], dtype=dtypes.int16)).cumprod(0).dtype == dtypes.int32
# (Tensor([0, 1], dtype=dtypes.int32)).cumprod(0).dtype == dtypes.int32
# (Tensor([0, 1], dtype=dtypes.int64)).cumprod(0).dtype == dtypes.int64
# (Tensor([0, 1], dtype=dtypes.uint8)).cumprod(0).dtype == dtypes.uint32
# (Tensor([0, 1], dtype=dtypes.uint16)).cumprod(0).dtype == dtypes.uint32
# (Tensor([0, 1], dtype=dtypes.uint32)).cumprod(0).dtype == dtypes.uint32
# (Tensor([0, 1], dtype=dtypes.uint64)).cumprod(0).dtype == dtypes.uint64
# (Tensor([0, 1], dtype=dtypes.float16)).cumprod(0).dtype == dtypes.float16
# #(Tensor([0, 1], dtype=dtypes.bfloat16)).cumprod(0).dtype == dtypes.bfloat16
# (Tensor([0, 1], dtype=dtypes.float32)).cumprod(0).dtype == dtypes.float32
# (Tensor([0, 1], dtype=dtypes.float64)).cumprod(0).dtype == dtypes.float64

from tinygrad import Tensor, dtypes
dtypes.default_float = dtypes.float16

if __name__ == "__main__":
  # softmax in bert layers
  BS = 96//6
  t = Tensor.empty(BS, 16, 512, 512)
  t.softmax(-1, dtype="half").realize()

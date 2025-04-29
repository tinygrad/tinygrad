from tinygrad import Tensor, dtypes, GlobalCounters

if __name__ == "__main__":
  t = Tensor.empty(81920, 4096, dtype=dtypes.half)
  GlobalCounters.reset()
  t.softmax(-1, dtype="half").realize()
  GlobalCounters.reset()
  t.softmax(-1, dtype="half", _single_kernel=True).realize()

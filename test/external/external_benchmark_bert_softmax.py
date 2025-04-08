from tinygrad import Tensor, dtypes, Context, GlobalCounters
dtypes.default_float = dtypes.float16
from test.test_softmax_fusion import single_kernel_softmax

if __name__ == "__main__":
  # softmax in bert layers
  BS = 96//6
  t = Tensor.empty(BS, 16, 512, 512)
  t.softmax(-1, dtype="half").realize()

  # test single kernel softmax
  GlobalCounters.reset()
  with Context(DONT_GROUP_REDUCES=1):
    single_kernel_softmax(t, -1, "half").realize()


from tinygrad import Tensor, dtypes

Tensor.empty(1).bitcast(dtypes.int).bitcast(dtypes.float).realize()

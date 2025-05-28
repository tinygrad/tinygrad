from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

if __name__ == "__main__":
  BS = getenv("BS", 32)
  NDATA = getenv("NDATA", 4096)
  HASHFN = getenv("HASHFN", "shake_128")

  t = Tensor.randn(BS, NDATA, dtype=dtypes.uint8).realize()
  t.keccak(HASHFN).realize()

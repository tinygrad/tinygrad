import weakref
import numpy as np
from tinygrad.tensor import Tensor, Device
Device.DEFAULT = "METAL"

if __name__ == "__main__":
  t = Tensor.zeros(3).realize()
  wt = weakref.ref(t.lazydata.realized)
  n = t.numpy()
  t += 1
  n2 = t.numpy()
  print(wt)
  del t
  print(wt)
  print(n, n.base, n.base.base)
  print(n2, n2.base, n2.base.base)
  assert wt() is not None
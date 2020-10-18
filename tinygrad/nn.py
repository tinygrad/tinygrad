import numpy as np

def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class SGD:
  def __init__(self, tensors, lr):
    self.tensors = tensors
    self.lr = lr

  def step(self):
    for t in self.tensors:
      t.data -= self.lr * t.grad

